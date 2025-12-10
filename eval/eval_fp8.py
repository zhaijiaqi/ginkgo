#!/usr/bin/env python3
"""
全fp8精度 CG 求解性能评估脚本
评估全fp8精度相对于全精度baseline的性能表现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
import json
import argparse
import random
import time
from typing import Dict, Optional

# 导入项目模块
from env.cg_env import CGEnvironment
from utils import create_env_config, DoublePrecisionAgent, FullFp8Agent, configure_matplotlib_chinese


def run_fp8_evaluation(matrix_name: Optional[str] = None,
                      matrix_size: Optional[int] = None,
                      config_path: str = 'config/default.yaml',
                      random_seed: int = 42):
    """
    运行全fp8精度评估：比较全fp8与全精度baseline的性能

    Args:
        matrix_name: 矩阵名称（如果为None，则使用随机矩阵）
        matrix_size: 矩阵大小（当matrix_name为None时使用，用于生成随机矩阵）
        config_path: 配置文件路径
        random_seed: 随机种子，用于确保两次运行使用相同的b
    """
    print("=" * 80)
    print("🔥 全fp8精度 CG 求解性能评估")
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

    # 计算 tile 数量
    tilesize = env.spmv_sim.tilesize
    num_tiles = (env.matrix_size + tilesize - 1) // tilesize

    print(f"环境配置:")
    print(f"  - 矩阵大小: {env.matrix_size}x{env.matrix_size}")
    print(f"  - Tile 数量: {num_tiles}")
    print(f"  - Tile 大小: {tilesize}")
    print(f"  - 最大迭代次数: {env.max_iter}")
    print(f"  - 停止容差: {env.stop_tol}")

    # 设置随机种子，确保两次运行使用相同的b
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # 1. 全fp8精度评估
    print("\n" + "=" * 80)
    print("🔥 步骤 1: 运行全fp8精度评估")
    print("=" * 80)

    fp8_agent = FullFp8Agent(num_tiles)
    fp8_agent.eval_mode()

    fp8_start_time = time.time()
    # 运行全fp8评估
    obs = env.reset(seed=random_seed)
    done = False
    step_count = 0

    while not done:
        actions = fp8_agent.act(obs)
        next_obs, reward, done, info = env.step(actions)
        obs = next_obs
        step_count += 1
        if step_count % 10 == 0:
            print(f"  步骤 {step_count}: 残差 = {info['residual_norm']:.6e}, 成本 = {info['iteration_cost']:.6f}")

    fp8_end_time = time.time()
    fp8_time = fp8_end_time - fp8_start_time

    # 获取全fp8结果
    fp8_episode_info = env.get_episode_info()
    fp8_result = {
        'iterations': fp8_episode_info['iterations'],
        'compute_cost': fp8_episode_info['total_cost'],
        'final_residual': fp8_episode_info['final_residual'],
        'converged': bool(fp8_episode_info['converged']),
        'avg_tile_cost': fp8_episode_info['avg_tile_cost'],
        'initial_residual': fp8_episode_info.get('initial_residual', None)
    }

    print(f"\n全fp8精度结果:")
    print(f"  - 收敛迭代次数: {fp8_result['iterations']}")
    print(f"  - 总计算成本: {fp8_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {fp8_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if fp8_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {fp8_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {fp8_time:.2f} 秒")

    # 保存b的值，以便第二次运行使用相同的b
    saved_b = env.b.copy() if hasattr(env, 'b') and env.b is not None else None
    if saved_b is not None:
        print(f"  - 保存b向量用于第二次运行 (长度: {len(saved_b)})")

    # 2. 全精度 baseline 评估
    print("\n" + "=" * 80)
    print("📊 步骤 2: 运行全精度 baseline 评估")
    print("=" * 80)

    # 重新创建环境（确保使用相同的配置）
    env_dp = CGEnvironment(env_config)

    dp_agent = DoublePrecisionAgent(num_tiles)
    dp_agent.eval_mode()

    dp_start_time = time.time()
    # 运行全精度评估
    obs = env_dp.reset(seed=random_seed)

    # 如果有保存的b，使用相同的b
    if saved_b is not None:
        env_dp.b = saved_b.copy()
        env_dp.b_norm = np.linalg.norm(env_dp.b)
        # 重新计算初始残差
        env_dp.x = np.zeros(env_dp.matrix_size)
        env_dp.r = env_dp._compute_exact_residual(env_dp.x, env_dp.A_diagonal, env_dp.b)
        env_dp.p = env_dp.r.copy()
        # 重新记录初始残差
        initial_residual_norm = env_dp.math_sim.vector_norm(env_dp.r)
        env_dp.residual_tracker.reset()
        env_dp.residual_tracker.record_residual(initial_residual_norm)
        # 重新获取状态
        obs = env_dp.get_state_features(env_dp.p, env_dp.current_iteration)

    done = False
    step_count = 0

    while not done:
        actions = dp_agent.act(obs)
        next_obs, reward, done, info = env_dp.step(actions)
        obs = next_obs
        step_count += 1
        if step_count % 10 == 0:
            print(f"  步骤 {step_count}: 残差 = {info['residual_norm']:.6e}, 成本 = {info['iteration_cost']:.6f}")

    dp_end_time = time.time()
    dp_time = dp_end_time - dp_start_time

    # 获取全精度结果
    dp_episode_info = env_dp.get_episode_info()
    dp_result = {
        'iterations': dp_episode_info['iterations'],
        'compute_cost': dp_episode_info['total_cost'],
        'final_residual': dp_episode_info['final_residual'],
        'converged': bool(dp_episode_info['converged']),
        'avg_tile_cost': dp_episode_info['avg_tile_cost'],
        'initial_residual': dp_episode_info.get('initial_residual', None)
    }

    print(f"\n全精度 baseline 结果:")
    print(f"  - 收敛迭代次数: {dp_result['iterations']}")
    print(f"  - 总计算成本: {dp_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {dp_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if dp_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {dp_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {dp_time:.2f} 秒")

    # 3. 性能对比
    print("\n" + "=" * 80)
    print("📈 性能对比分析")
    print("=" * 80)

    dp_cost = dp_result['compute_cost']
    fp8_cost = fp8_result['compute_cost']
    dp_iterations = dp_result['iterations']
    fp8_iterations = fp8_result['iterations']

    cost_change = fp8_cost - dp_cost  # fp8相对于全精度的成本变化
    cost_change_percent = (cost_change / dp_cost * 100) if dp_cost > 0 else 0
    iteration_change = fp8_iterations - dp_iterations  # fp8相对于全精度的迭代变化
    iteration_change_percent = (iteration_change / dp_iterations * 100) if dp_iterations > 0 else 0

    print(f"\n计算成本对比:")
    print(f"  全fp8精度:       {fp8_cost:.6f}")
    print(f"  全精度 baseline: {dp_cost:.6f}")
    print(f"  成本变化:        {cost_change:+.6f} ({cost_change_percent:+.2f}%)")

    print(f"\n迭代次数对比:")
    print(f"  全fp8精度:       {fp8_iterations}")
    print(f"  全精度 baseline: {dp_iterations}")
    print(f"  迭代变化:        {iteration_change:+d} ({iteration_change_percent:+.2f}%)")

    print(f"\n收敛性对比:")
    print(f"  全fp8精度收敛:   {'是' if fp8_result['converged'] else '否'}")
    print(f"  全精度收敛:      {'是' if dp_result['converged'] else '否'}")

    if dp_result.get('initial_residual') and fp8_result.get('initial_residual'):
        print(f"  初始残差 (全fp8):  {fp8_result['initial_residual']:.6e}")
        print(f"  初始残差 (全精度): {dp_result['initial_residual']:.6e}")

    print(f"\n最终残差对比:")
    print(f"  全fp8最终残差:   {fp8_result['final_residual']:.6e}")
    print(f"  全精度最终残差:  {dp_result['final_residual']:.6e}")
    residual_ratio = fp8_result['final_residual'] / dp_result['final_residual'] if dp_result['final_residual'] > 0 else None
    if residual_ratio is not None:
        print(f"  残差比率:        {residual_ratio:.4f}")

    # 保存评估结果
    eval_result = {
        'evaluation_type': 'full_fp8_vs_double_precision',
        'matrix_name': matrix_name,
        'matrix_size': env.matrix_size,
        'specified_matrix_size': matrix_size,
        'random_seed': random_seed,
        'num_tiles': num_tiles,
        'tilesize': tilesize,
        'double_precision': {
            'iterations': dp_result['iterations'],
            'compute_cost': dp_result['compute_cost'],
            'final_residual': dp_result['final_residual'],
            'converged': dp_result['converged'],
            'avg_tile_cost': dp_result['avg_tile_cost'],
            'runtime_seconds': dp_time
        },
        'full_fp8': {
            'iterations': fp8_result['iterations'],
            'compute_cost': fp8_result['compute_cost'],
            'final_residual': fp8_result['final_residual'],
            'converged': fp8_result['converged'],
            'avg_tile_cost': fp8_result['avg_tile_cost'],
            'runtime_seconds': fp8_time
        },
        'comparison': {
            'cost_change': cost_change,
            'cost_change_percent': cost_change_percent,
            'iteration_change': iteration_change,
            'iteration_change_percent': iteration_change_percent,
            'residual_ratio': residual_ratio
        }
    }

    # 保存结果到JSON文件
    result_filename = f"fp8_evaluation_result_{int(time.time())}.json"
    result_path = os.path.join('.', result_filename)

    with open(result_path, 'w') as f:
        json.dump(eval_result, f, indent=2)

    print(f"\n📄 详细评估结果已保存至: {result_path}")

    # 总结
    print("\n" + "=" * 80)
    print("📊 评估总结:")
    if cost_change < 0:
        print(f"  全fp8相对于全精度成本降低: {abs(cost_change_percent):.2f}%")
        print("✅ 全fp8精度实现了成本节省！")
    else:
        print(f"  全fp8相对于全精度成本增加: {abs(cost_change_percent):.2f}%")
        print("⚠️  全fp8精度成本高于全精度，可能影响数值稳定性。")
    print("=" * 80)

    return eval_result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='全fp8精度CG求解性能评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用指定矩阵评估fp8性能
  python eval/eval_fp8.py --matrix_name Muu

  # 使用自定义随机种子
  python eval/eval_fp8.py --matrix_name Muu --seed 123

  # 使用随机矩阵（使用配置文件中的matrix_size）
  python eval/eval_fp8.py --matrix_name None

  # 使用指定大小的随机矩阵
  python eval/eval_fp8.py --matrix_size 512

  # 使用指定大小的随机矩阵并指定随机种子
  python eval/eval_fp8.py --matrix_size 1024 --seed 42
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

    # 如果指定了matrix_name，忽略matrix_size（因为真实矩阵的大小由矩阵文件决定）
    if matrix_name is not None and matrix_size is not None:
        print("⚠️  警告: 指定了matrix_name时，matrix_size将被忽略（矩阵大小由矩阵文件决定）")
        matrix_size = None

    # 运行评估
    try:
        run_fp8_evaluation(
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
