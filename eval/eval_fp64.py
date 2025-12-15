#!/usr/bin/env python3
"""
全fp64精度 CG 求解性能评估脚本
评估全fp64精度的CG求解性能
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import numpy as np
import torch
import json
import argparse
import random
import time
from typing import Dict, Optional, Tuple

# 导入项目模块
from env.cg_env import CGEnvironment

# 显式导入本地utils模块
import importlib.util
utils_spec = importlib.util.spec_from_file_location("utils", os.path.join(project_root, "utils", "__init__.py"))
utils = importlib.util.module_from_spec(utils_spec)
sys.modules["utils"] = utils
utils_spec.loader.exec_module(utils)

from utils import create_env_config, FullFp64Agent, configure_matplotlib_chinese


def run_fp64_evaluation(matrix_name: Optional[str] = None,
                      matrix_size: Optional[int] = None,
                      config_path: str = 'config/default.yaml',
                      random_seed: int = 42):
    """
    运行全fp64精度评估

    Args:
        matrix_name: 矩阵名称（如果为None，则使用随机矩阵）
        matrix_size: 矩阵大小（当matrix_name为None时使用，用于生成随机矩阵）
        config_path: 配置文件路径
        random_seed: 随机种子
    """
    print("=" * 80)
    print("🔥 全fp64精度 CG 求解性能评估")
    print("=" * 80)
    if matrix_name:
        print(f"矩阵名称: {matrix_name}")
    else:
        print(f"矩阵类型: 随机生成")
        if matrix_size:
            print(f"矩阵大小: {matrix_size}x{matrix_size}")
    print(f"随机种子: {random_seed}")
    print("=" * 80)

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 创建环境配置
    env_config = create_env_config(config, matrix_name=matrix_name, matrix_size=matrix_size)
    env_config['random_seed'] = random_seed

    # 创建环境
    print("\n📊 创建评估环境...")
    env = CGEnvironment(env_config)
    
    print(f"env.x: {env.x[:10]}")
    print(f"env.b: {env.b[:10]}")

    # 计算 tile 数量
    tilesize = env.spmv_sim.tilesize
    num_tiles = (env.matrix_size + tilesize - 1) // tilesize

    print(f"环境配置:")
    print(f"  - 矩阵大小: {env.matrix_size}x{env.matrix_size}")
    print(f"  - Tile 数量: {num_tiles}")
    print(f"  - Tile 大小: {tilesize}")
    print(f"  - 最大迭代次数: {env.max_iter}")
    print(f"  - 停止容差: {env.stop_tol}")

    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 全fp64精度评估
    print("\n" + "=" * 80)
    print("🔥 运行全fp64精度评估")
    print("=" * 80)

    fp64_agent = FullFp64Agent(num_tiles)
    fp64_agent.eval_mode()

    start_time = time.time()
    # 运行全fp64评估
    obs = env.reset(seed=random_seed)
    done = False
    step_count = 0

    while not done:
        actions = fp64_agent.act(obs)
        next_obs, reward, done, info = env.step(actions)
        obs = next_obs
        step_count += 1
        if step_count % 10 == 0:
            print(f"  步骤 {step_count}: 残差 = {info['residual_norm']:.6e}, 成本 = {info['iteration_cost']:.6f}")

    end_time = time.time()
    runtime = end_time - start_time

    # 获取结果
    episode_info = env.get_episode_info()
    result = {
        'iterations': episode_info['iterations'],
        'compute_cost': episode_info['total_cost'],
        'final_residual': episode_info['final_residual'],
        'converged': bool(episode_info['converged']),
        'avg_tile_cost': episode_info['avg_tile_cost'],
        'initial_residual': episode_info.get('initial_residual', None)
    }

    print(f"\n全fp64精度结果:")
    print(f"  - 收敛迭代次数: {result['iterations']}")
    print(f"  - 总计算成本: {result['compute_cost']:.6f}")
    print(f"  - 最终残差: {result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {runtime:.2f} 秒")

    if result.get('initial_residual'):
        print(f"  - 初始残差: {result['initial_residual']:.6e}")

    # 保存评估结果
    eval_result = {
        'evaluation_type': 'full_fp64',
        'matrix_name': matrix_name,
        'matrix_size': env.matrix_size,
        'specified_matrix_size': matrix_size,
        'random_seed': random_seed,
        'num_tiles': num_tiles,
        'tilesize': tilesize,
        'full_fp64': {
            'iterations': result['iterations'],
            'compute_cost': result['compute_cost'],
            'final_residual': result['final_residual'],
            'converged': result['converged'],
            'avg_tile_cost': result['avg_tile_cost'],
            'runtime_seconds': runtime
        }
    }
    
    
    # 保存结果到JSON文件
    result_dir = './log/fp64/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_filename = f"{matrix_name}_fp64_evaluation_result_{int(time.time())}.json"
    result_path = os.path.join(result_dir, result_filename)

    with open(result_path, 'w') as f:
        json.dump(eval_result, f, indent=2)

    print(f"\n📄 详细评估结果已保存至: {result_path}")

    # 总结
    print("\n" + "=" * 80)
    print("📊 评估总结:")
    if result['converged']:
        print("✅ CG 求解收敛成功")
    else:
        print("❌ CG 求解未在最大迭代次数内收敛")
    print("=" * 80)

    return eval_result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='全fp64精度CG求解性能评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用指定矩阵评估fp64性能
  python eval/eval_fp64.py --matrix_name Muu

  # 使用自定义随机种子
  python eval/eval_fp64.py --matrix_name Muu --seed 123

  # 使用随机矩阵（使用配置文件中的matrix_size）
  python eval/eval_fp64.py --matrix_name None

  # 使用指定大小的随机矩阵
  python eval/eval_fp64.py --matrix_size 512

  # 使用指定大小的随机矩阵并指定随机种子
  python eval/eval_fp64.py --matrix_size 1024 --seed 42
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
        help='矩阵大小（当matrix_name为None时使用，用于生成随机矩阵，如512、1024等）'
    )

    parser.add_argument(
        '--config_path',
        type=str,
        default='config/default.yaml',
        help='配置文件路径（默认: config/default.yaml）'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，用于确保两次运行使用相同的b（默认: 42）'
    )

    args = parser.parse_args()

    # 处理matrix_name参数
    matrix_name = args.matrix_name

    # 处理matrix_size参数
    matrix_size = args.matrix_size

    # 如果指定了真实的矩阵名称，忽略matrix_size（因为真实矩阵的大小由矩阵文件决定）
    # 但如果matrix_name是字符串"None"，则使用matrix_size生成随机矩阵
    if matrix_name is not None and matrix_name != "None" and matrix_size is not None:
        print("⚠️  警告: 指定了matrix_name时，matrix_size将被忽略（矩阵大小由矩阵文件决定）")
        matrix_size = None

    # 运行评估
    try:
        run_fp64_evaluation(
            matrix_name=matrix_name,
            matrix_size=matrix_size,
            config_path=args.config_path,
            random_seed=args.seed
        )
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
