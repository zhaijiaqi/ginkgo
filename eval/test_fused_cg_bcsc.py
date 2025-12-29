#!/usr/bin/env python3
"""
测试脚本：使用 bcsc_spmv_mixed_prequant 和 fused_cg_step 两个 kernel 进行 CG 求解
测试仅使用这两个 kernel 时解决一个 CG 问题的总时间

使用方法：
  方式1（推荐）：使用包装脚本（会自动激活环境）
    ./run_test_fused_cg_bcsc.sh --matrix_name Muu

  方式2：手动激活环境后运行
    source ~/.rlcg_env.sh
    python3 test_fused_cg_bcsc.py --matrix_name Muu
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import time
import argparse
import yaml
from typing import Optional

# 导入必要的模块
from env.cg_env import CGEnvironment
from kernels.bcsc_spmv_kernels import bcsc_spmv_mixed_prequant
from kernels.fused_cg_kernel import fused_cg_step

# 使用明确的导入路径避免与 tilelang.utils 冲突
import importlib.util
utils_spec = importlib.util.spec_from_file_location("utils", os.path.join(project_root, "utils", "__init__.py"))
utils = importlib.util.module_from_spec(utils_spec)
sys.modules["utils"] = utils
utils_spec.loader.exec_module(utils)
from utils import create_env_config


def solve_cg_with_fused_kernels(
    bcsc,
    b: torch.Tensor,
    max_iter: int = 1000,
    tol: float = 1e-10,
    actions: Optional[torch.Tensor] = None,
    device: str = "cuda",
    block_size: int = 256,
    verbose: bool = True,
):
    """
    使用 bcsc_spmv_mixed_prequant 和 fused_cg_step 两个 kernel 求解 CG 问题
    
    Args:
        bcsc: BCSC 格式的矩阵
        b: 右端项向量 (torch.Tensor)
        max_iter: 最大迭代次数
        tol: 收敛容差
        actions: 精度选择动作 (n_bc,) int32，如果为 None 则使用全 fp64 (action=0)
        device: 设备
        block_size: fused_cg_step 的 block size
        verbose: 是否打印详细信息
    
    Returns:
        (x, residual_norms, converged, total_time, iteration_count, spmv_time, fused_cg_time)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("此测试需要 CUDA")
    
    dev = torch.device(device)
    N = int(b.shape[0])
    
    # 确保 bcsc 在 CUDA 上
    if bcsc.A_fp64.device.type != "cuda":
        bcsc = bcsc.to("cuda")
    
    # 确保 b 在 CUDA 上
    if not torch.is_tensor(b):
        b = torch.as_tensor(b, dtype=torch.float64, device=dev)
    else:
        b = b.to(device=dev, dtype=torch.float64)
    
    # 初始化 CG 变量
    x = torch.zeros((N,), device=dev, dtype=torch.float64)
    r = b.clone()
    p = r.clone()
    
    # 计算初始残差
    b_norm = float(torch.linalg.vector_norm(b, ord=2).item())
    initial_residual_norm = float(torch.linalg.vector_norm(r, ord=2).item())
    
    # 如果未提供 actions，使用全 fp64 (action=0)
    if actions is None:
        actions = torch.zeros((bcsc.n_bc,), dtype=torch.int32, device=dev)
    else:
        actions = actions.to(device=dev, dtype=torch.int32)
        if int(actions.numel()) != int(bcsc.n_bc):
            raise ValueError(f"actions 长度 {int(actions.numel())} 与 bcsc.n_bc {int(bcsc.n_bc)} 不匹配")
    
    # 记录残差历史
    residual_norms = [initial_residual_norm]
    
    time_spmv = 0.0
    time_fused_cg = 0.0
    
    converged = False
    iteration = 0
    
    if verbose:
        print(f"开始 CG 求解 (N={N}, max_iter={max_iter}, tol={tol})")
        print(f"初始残差: {initial_residual_norm:.6e}, b_norm: {b_norm:.6e}")
    
    for iteration in range(max_iter):
        # 1. 使用 bcsc_spmv_mixed_prequant 计算 Ap
        time_spmv_start = time.time()
        Ap = bcsc_spmv_mixed_prequant(
            bcsc,
            actions,
            p,
            device=device,
            use_tensorcore_bf16=False,  # 可以根据需要启用
            return_torch=True,
        )
        time_spmv_end = time.time()
        time_spmv += time_spmv_end - time_spmv_start
        
        # 确保 Ap 长度正确（BCSC 可能输出更长的向量）
        if int(Ap.shape[0]) > N:
            Ap = Ap[:N]
        
        # 2. 使用 fused_cg_step 完成一次 CG 迭代
        time_fused_cg_start = time.time()
        stats = fused_cg_step(r, Ap, p, x, device=device, block_size=block_size)
        time_fused_cg_end = time.time()
        time_fused_cg += time_fused_cg_end - time_fused_cg_start
        
        
        # 提取统计信息
        r_dot_r_old = float(stats[0].item())  # ||r||² (旧)
        r_dot_r_new = float(stats[1].item())  # ||r_new||² (新)
        aj = float(stats[2].item())  # alpha (aj)
        
        # 检查发散
        if abs(aj) < 1e-300:
            if verbose:
                print(f"⚠️  警告: 迭代 {iteration+1} 发散 (aj={aj:.3e})")
            break
        
        # 计算当前残差范数
        residual_norm = float(np.sqrt(r_dot_r_new))
        residual_norms.append(residual_norm)
        
        # 检查收敛
        relative_residual = residual_norm / b_norm
        converged = relative_residual < tol
        
        if verbose and (iteration % 10 == 0 or converged):
            print(f"迭代 {iteration+1:4d}: 残差={residual_norm:.6e}, 相对残差={relative_residual:.6e}, aj={aj:.6e}")
        
        if converged:
            if verbose:
                print(f"✓ 收敛于迭代 {iteration+1}")
            break
    
    # 结束计时
    total_time = time_spmv + time_fused_cg
    
    if not converged and iteration == max_iter - 1:
        if verbose:
            print(f"⚠️  达到最大迭代次数 {max_iter}，未收敛")
    
    return x, residual_norms, converged, total_time, iteration + 1, time_spmv, time_fused_cg


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='测试使用 bcsc_spmv_mixed_prequant 和 fused_cg_step 两个 kernel 的 CG 求解性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用指定矩阵测试
  python test_fused_cg_bcsc.py --matrix_name Muu

  # 使用随机矩阵
  python test_fused_cg_bcsc.py --matrix_size 1024

  # 使用全 fp64 精度
  python test_fused_cg_bcsc.py --matrix_name Muu --precision fp64

  # 使用全 fp32 精度
  python test_fused_cg_bcsc.py --matrix_name Muu --precision fp32

  # 使用全 bf16 精度
  python test_fused_cg_bcsc.py --matrix_name Muu --precision bf16
        """
    )
    
    parser.add_argument(
        '--matrix_name',
        type=str,
        default=None,
        help='矩阵名称（从matrix_set.csv中选择，如Muu）。如果为None，则使用随机矩阵'
    )
    
    parser.add_argument(
        '--matrix_size',
        type=int,
        default=None,
        help='矩阵大小（当matrix_name为None时使用，用于生成随机矩阵）'
    )
    
    parser.add_argument(
        '--config_path',
        type=str,
        default='config/default.yaml',
        help='配置文件路径（默认: config/default.yaml）'
    )
    
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp64', 'fp32', 'bf16'],
        default='fp64',
        help='精度选择：fp64 (action=0), fp32 (action=1), bf16 (action=2)'
    )
    
    parser.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help='最大迭代次数（默认: 1000）'
    )
    
    parser.add_argument(
        '--tol',
        type=float,
        default=1e-10,
        help='收敛容差（默认: 1e-10）'
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=256,
        help='fused_cg_step 的 block size（默认: 256）'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=3,
        help='预热次数（默认: 3）'
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=10,
        help='测试运行次数（默认: 10）'
    )
    
    args = parser.parse_args()
    
    # 处理 matrix_name 参数
    matrix_name = None
    if args.matrix_name:
        if args.matrix_name.lower() == 'none':
            matrix_name = None
        else:
            matrix_name = args.matrix_name
    
    # 处理 matrix_size 参数
    matrix_size = args.matrix_size
    
    # 如果指定了 matrix_name，忽略 matrix_size
    if matrix_name is not None and matrix_size is not None:
        print("⚠️  警告: 指定了 matrix_name 时，matrix_size 将被忽略")
        matrix_size = None
    
    # 精度映射
    precision_map = {'fp64': 0, 'fp32': 1, 'bf16': 2}
    action_value = precision_map[args.precision]
    
    print("=" * 80)
    print("🚀 测试 fused CG + BCSC SpMV kernels")
    print("=" * 80)
    print(f"矩阵: {matrix_name if matrix_name else f'随机 {matrix_size}x{matrix_size}'}")
    print(f"精度: {args.precision} (action={action_value})")
    print(f"最大迭代次数: {args.max_iter}")
    print(f"收敛容差: {args.tol}")
    print(f"Block size: {args.block_size}")
    print("=" * 80)
    
    # 加载配置
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建环境配置
    env_config = create_env_config(config, matrix_name=matrix_name, matrix_size=matrix_size)
    env_config['use_bcsc_prequant'] = True
    env_config['spmv_impl'] = 'bcsc_prequant'
    env_config['bcsc_device'] = 'cuda'
    env_config['verbose'] = False  # 环境初始化时不打印
    
    # 创建环境并加载矩阵
    print("\n📊 加载矩阵并构建 BCSC 格式...")
    env = CGEnvironment(env_config)
    
    # 确保 BCSC 已构建
    if env.bcsc is None:
        raise RuntimeError("BCSC 矩阵未构建，请检查配置")
    
    bcsc = env.bcsc
    b = env.b  # 使用环境中的 b
    
    print(f"矩阵大小: {env.matrix_size}x{env.matrix_size}")
    print(f"BCSC: n_br={bcsc.n_br}, n_bc={bcsc.n_bc}, nnzb={bcsc.nnzb}")
    print(f"Tile size: {env.spmv_sim.tilesize}")
    
    # 创建 actions（全使用指定精度）
    actions = torch.full((bcsc.n_bc,), action_value, dtype=torch.int32, device='cuda')
    
    # 预热
    print(f"\n🔥 预热 {args.warmup} 次...")
    for _ in range(args.warmup):
        _ = solve_cg_with_fused_kernels(
            bcsc, b, max_iter=min(10, args.max_iter), tol=args.tol,
            actions=actions, device='cuda', block_size=args.block_size,
            verbose=False
        )
    
    # 正式测试
    print(f"\n⏱️  运行 {args.runs} 次测试...")
    times = []
    spmv_times = []
    fused_cg_times = []
    converged_count = 0
    
    for run in range(args.runs):
        x, residual_norms, converged, total_time, iterations, spmv_time, fused_cg_time = solve_cg_with_fused_kernels(
            bcsc, b, max_iter=args.max_iter, tol=args.tol,
            actions=actions, device='cuda', block_size=args.block_size,
            verbose=(run == 0)  # 只打印第一次的详细信息
        )
        
        times.append(total_time)
        spmv_times.append(spmv_time)
        fused_cg_times.append(fused_cg_time)
        if converged:
            converged_count += 1
        
        if run == 0:
            print(f"\n第一次运行结果:")
            print(f"  迭代次数: {iterations}")
            print(f"  是否收敛: {'是' if converged else '否'}")
            print(f"  最终残差: {residual_norms[-1]:.6e}")
            print(f"  总时间: {total_time:.4f} 秒")
            print(f"  SpMV 时间: {spmv_time:.4f} 秒 ({spmv_time/total_time*100:.1f}%)")
            print(f"  Fused CG 时间: {fused_cg_time:.4f} 秒 ({fused_cg_time/total_time*100:.1f}%)")
    
    # 统计结果
    times_array = np.array(times)
    spmv_times_array = np.array(spmv_times)
    fused_cg_times_array = np.array(fused_cg_times)
    
    # 总时间统计
    mean_time = float(np.mean(times_array))
    std_time = float(np.std(times_array))
    median_time = float(np.median(times_array))
    min_time = float(np.min(times_array))
    max_time = float(np.max(times_array))
    
    # SpMV 时间统计
    mean_spmv_time = float(np.mean(spmv_times_array))
    std_spmv_time = float(np.std(spmv_times_array))
    median_spmv_time = float(np.median(spmv_times_array))
    min_spmv_time = float(np.min(spmv_times_array))
    max_spmv_time = float(np.max(spmv_times_array))
    
    # Fused CG 时间统计
    mean_fused_cg_time = float(np.mean(fused_cg_times_array))
    std_fused_cg_time = float(np.std(fused_cg_times_array))
    median_fused_cg_time = float(np.median(fused_cg_times_array))
    min_fused_cg_time = float(np.min(fused_cg_times_array))
    max_fused_cg_time = float(np.max(fused_cg_times_array))
    
    print("\n" + "=" * 80)
    print("📈 性能统计")
    print("=" * 80)
    print(f"运行次数: {args.runs}")
    print(f"收敛次数: {converged_count}/{args.runs}")
    
    print(f"\n总时间统计 (毫秒):")
    print(f"  平均: {mean_time*1000:.2f} ± {std_time*1000:.2f}")
    print(f"  中位数: {median_time*1000:.2f}")
    print(f"  最小: {min_time*1000:.2f}")
    print(f"  最大: {max_time*1000:.2f}")
    
    print(f"\nSpMV 时间统计 (毫秒):")
    print(f"  平均: {mean_spmv_time*1000:.2f} ± {std_spmv_time*1000:.2f}")
    print(f"  中位数: {median_spmv_time*1000:.2f}")
    print(f"  最小: {min_spmv_time*1000:.2f}")
    print(f"  最大: {max_spmv_time*1000:.2f}")
    
    print(f"\nFused CG 算子时间统计 (毫秒):")
    print(f"  平均: {mean_fused_cg_time*1000:.2f} ± {std_fused_cg_time*1000:.2f}")
    print(f"  中位数: {median_fused_cg_time*1000:.2f}")
    print(f"  最小: {min_fused_cg_time*1000:.2f}")
    print(f"  最大: {max_fused_cg_time*1000:.2f}")
    print("=" * 80)
    
    return {
        'matrix_name': matrix_name,
        'matrix_size': env.matrix_size,
        'precision': args.precision,
        'runs': args.runs,
        'converged_count': converged_count,
        'total_time': {
            'mean': mean_time,
            'std': std_time,
            'median': median_time,
            'min': min_time,
            'max': max_time,
        },
        'spmv_time': {
            'mean': mean_spmv_time,
            'std': std_spmv_time,
            'median': median_spmv_time,
            'min': min_spmv_time,
            'max': max_spmv_time,
            'percentage': mean_spmv_time/mean_time*100,
        },
        'fused_cg_time': {
            'mean': mean_fused_cg_time,
            'std': std_fused_cg_time,
            'median': median_fused_cg_time,
            'min': min_fused_cg_time,
            'max': max_fused_cg_time,
            'percentage': mean_fused_cg_time/mean_time*100,
        },
    }


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
