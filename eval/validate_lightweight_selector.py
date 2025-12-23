#!/usr/bin/env python3
"""
验证轻量级精度选择器的性能
比较轻量级函数和模型指导的精度选择对CG收敛和计算速度的影响
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
import argparse
import random
import time
import json
from typing import Dict, Optional, List
from collections import defaultdict

# 导入项目模块
from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import PPOAgentFactory
from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent
from utils import create_env_config, load_model_weights, configure_matplotlib_chinese
from eval.lightweight_precision_selector import (
    LightweightPrecisionSelector, 
    AdaptiveLightweightPrecisionSelector
)
from eval.evaluator import run_episode_with_agent

# 设置matplotlib支持中文显示
configure_matplotlib_chinese()


class LightweightSelectorAgent:
    """轻量级选择器代理包装器"""
    
    def __init__(self, selector, env: CGEnvironment):
        self.selector = selector
        self.env = env
        self.tilesize = env.spmv_sim.tilesize
    
    def act(self, obs):
        """根据当前 p 向量和迭代次数选择精度"""
        iteration = self.env.current_iteration
        p_vector = self.env.p
        
        if isinstance(self.selector, AdaptiveLightweightPrecisionSelector):
            residual_norm = self.env.math_sim.vector_norm(self.env.r)
            actions = self.selector.select_precisions(
                p_vector, iteration, self.tilesize, residual_norm
            )
        else:
            actions = self.selector.select_precisions(
                p_vector, iteration, self.tilesize
            )
        
        return actions
    
    def observe(self, obs, reward, done, done2):
        """观察转换（轻量级选择器不需要学习）"""
        pass
    
    def eval_mode(self):
        """设置为评估模式"""
        pass


def run_episode_with_selector(env: CGEnvironment, selector_agent, 
                              seed: int = 42, 
                              fixed_b: Optional[np.ndarray] = None) -> Dict:
    """
    使用轻量级选择器运行一个完整的 CG episode
    
    Args:
        env: CG 环境
        selector_agent: 轻量级选择器代理
        seed: 随机种子
        fixed_b: 固定的右端项向量
        
    Returns:
        包含收敛信息和性能指标的字典
    """
    random.seed(seed)
    np.random.seed(seed)
    
    obs = env.reset(seed=seed)
    
    if fixed_b is not None:
        env.b = fixed_b.copy()
        env.b_norm = np.linalg.norm(env.b)
        env.x = np.zeros(env.matrix_size)
        env.r = env._compute_exact_residual(env.x, env.A_diagonal, env.b)
        env.p = env.r.copy()
        initial_residual_norm = env.math_sim.vector_norm(env.r)
        env.residual_tracker.reset()
        env.residual_tracker.record_residual(initial_residual_norm)
        obs = env.get_state_features(env.p)
    
    done = False
    step_count = 0
    
    # 记录精度选择历史
    precision_history = []
    
    while not done:
        # 选择动作
        actions = selector_agent.act(obs)
        precision_history.append(actions.copy())
        
        # 执行一步
        next_obs, reward, done, info = env.step(actions)
        
        selector_agent.observe(obs, reward, done, done)
        obs = next_obs
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"  步骤 {step_count}: 残差 = {info['residual_norm']:.6e}, 成本 = {info['iteration_cost']:.6f}")
    
    episode_info = env.get_episode_info()
    
    result = {
        'iterations': episode_info['iterations'],
        'compute_cost': episode_info['total_cost'],
        'final_residual': episode_info['final_residual'],
        'converged': bool(episode_info['converged']),
        'avg_tile_cost': episode_info['avg_tile_cost'],
        'initial_residual': episode_info.get('initial_residual', None),
        'precision_history': precision_history
    }
    
    return result


def analyze_precision_distribution(precision_history: List[List[int]]) -> Dict:
    """分析精度选择分布"""
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    counts = defaultdict(int)
    total_selections = 0
    
    for iteration_actions in precision_history:
        for action in iteration_actions:
            counts[action] += 1
            total_selections += 1
    
    distribution = {}
    for code in range(6):
        count = counts[code]
        percentage = (count / total_selections * 100) if total_selections > 0 else 0
        distribution[precision_names[code]] = {
            'count': count,
            'percentage': percentage
        }
    
    return distribution


def compare_strategies(model_path: str, 
                      matrix_name: Optional[str] = None,
                      matrix_size: Optional[int] = None,
                      config_path: str = 'config/default.yaml',
                      random_seed: int = 42,
                      output_path: Optional[str] = None):
    """
    比较模型指导策略和轻量级选择器策略
    
    Args:
        model_path: 模型权重文件路径
        matrix_name: 矩阵名称
        matrix_size: 矩阵大小
        config_path: 配置文件路径
        random_seed: 随机种子
        output_path: 输出文件路径
    """
    print("=" * 80)
    print("🔬 轻量级精度选择器验证")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    if matrix_name:
        print(f"矩阵名称: {matrix_name}")
    else:
        print(f"矩阵类型: 随机生成")
        if matrix_size:
            print(f"矩阵大小: {matrix_size}x{matrix_size}")
    print(f"随机种子: {random_seed}")
    print("=" * 80)
    
    # 处理配置文件路径
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if config_path.startswith('../'):
            config_path = os.path.join(project_root, config_path[3:])
        elif config_path.startswith('./'):
            config_path = os.path.join(project_root, config_path[2:])
        else:
            config_path = os.path.join(project_root, config_path)
    
    print(f"配置文件路径: {config_path}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建环境配置
    env_config = create_env_config(config, matrix_name=matrix_name, matrix_size=matrix_size)
    env_config['random_seed'] = random_seed
    
    # 创建环境
    print("\n📊 创建环境...")
    env = CGEnvironment(env_config)
    
    tilesize = env.spmv_sim.tilesize
    num_tiles = (env.matrix_size + tilesize - 1) // tilesize
    
    print(f"环境配置:")
    print(f"  - 矩阵大小: {env.matrix_size}x{env.matrix_size}")
    print(f"  - Tile 数量: {num_tiles}")
    print(f"  - Tile 大小: {tilesize}")
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 保存初始 b 向量（用于确保所有策略使用相同的问题）
    env_test = CGEnvironment(env_config)
    env_test.reset(seed=random_seed)
    saved_b = env_test.b.copy()
    
    results = {}
    
    # 1. 模型指导策略
    print("\n" + "=" * 80)
    print("🤖 策略 1: 模型指导的混合精度")
    print("=" * 80)
    
    env_model = CGEnvironment(env_config)
    cg_agent = load_model_weights(model_path, config, env_model)
    pfrl_agent = PfrlCompatibleCGPPOAgent(cg_agent)
    pfrl_agent.eval_mode()
    
    model_start_time = time.time()
    model_result = run_episode_with_agent(env_model, pfrl_agent, 
                                         seed=random_seed, fixed_b=saved_b)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    
    model_precision_dist = analyze_precision_distribution(model_result['precision_history'])
    
    print(f"\n模型指导策略结果:")
    print(f"  - 收敛迭代次数: {model_result['iterations']}")
    print(f"  - 总计算成本: {model_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {model_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if model_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {model_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {model_time:.2f} 秒")
    print(f"\n精度选择分布:")
    for prec_name, prec_info in sorted(model_precision_dist.items(), 
                                       key=lambda x: x[1]['count'], reverse=True):
        print(f"  - {prec_name}: {prec_info['count']} 次 ({prec_info['percentage']:.2f}%)")
    
    results['model'] = {
        'iterations': model_result['iterations'],
        'compute_cost': model_result['compute_cost'],
        'final_residual': model_result['final_residual'],
        'converged': model_result['converged'],
        'avg_tile_cost': model_result['avg_tile_cost'],
        'runtime': model_time,
        'precision_distribution': model_precision_dist
    }
    
    # 2. 基础轻量级选择器
    print("\n" + "=" * 80)
    print("⚡ 策略 2: 基础轻量级选择器")
    print("=" * 80)
    
    env_lw = CGEnvironment(env_config)
    lw_selector = LightweightPrecisionSelector(
        l2_norm_threshold=0.1,
        max_abs_threshold=0.05,
        early_iter_threshold=70,
        use_iteration_factor=False
    )
    lw_agent = LightweightSelectorAgent(lw_selector, env_lw)
    
    lw_start_time = time.time()
    lw_result = run_episode_with_selector(env_lw, lw_agent, 
                                        seed=random_seed, fixed_b=saved_b)
    lw_end_time = time.time()
    lw_time = lw_end_time - lw_start_time
    
    lw_precision_dist = analyze_precision_distribution(lw_result['precision_history'])
    
    print(f"\n基础轻量级选择器结果:")
    print(f"  - 收敛迭代次数: {lw_result['iterations']}")
    print(f"  - 总计算成本: {lw_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {lw_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if lw_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {lw_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {lw_time:.2f} 秒")
    print(f"\n精度选择分布:")
    for prec_name, prec_info in sorted(lw_precision_dist.items(), 
                                       key=lambda x: x[1]['count'], reverse=True):
        print(f"  - {prec_name}: {prec_info['count']} 次 ({prec_info['percentage']:.2f}%)")
    
    results['lightweight'] = {
        'iterations': lw_result['iterations'],
        'compute_cost': lw_result['compute_cost'],
        'final_residual': lw_result['final_residual'],
        'converged': lw_result['converged'],
        'avg_tile_cost': lw_result['avg_tile_cost'],
        'runtime': lw_time,
        'precision_distribution': lw_precision_dist
    }
    
    # 3. 自适应轻量级选择器
    print("\n" + "=" * 80)
    print("🎯 策略 3: 自适应轻量级选择器")
    print("=" * 80)
    
    env_adaptive = CGEnvironment(env_config)
    adaptive_selector = AdaptiveLightweightPrecisionSelector(
        l2_norm_threshold=0.1,
        max_abs_threshold=0.05,
        early_iter_threshold=70,
        use_iteration_factor=True,
        residual_adaptation=True
    )
    adaptive_agent = LightweightSelectorAgent(adaptive_selector, env_adaptive)
    
    adaptive_start_time = time.time()
    adaptive_result = run_episode_with_selector(env_adaptive, adaptive_agent, 
                                               seed=random_seed, fixed_b=saved_b)
    adaptive_end_time = time.time()
    adaptive_time = adaptive_end_time - adaptive_start_time
    
    adaptive_precision_dist = analyze_precision_distribution(adaptive_result['precision_history'])
    
    print(f"\n自适应轻量级选择器结果:")
    print(f"  - 收敛迭代次数: {adaptive_result['iterations']}")
    print(f"  - 总计算成本: {adaptive_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {adaptive_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if adaptive_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {adaptive_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {adaptive_time:.2f} 秒")
    print(f"\n精度选择分布:")
    for prec_name, prec_info in sorted(adaptive_precision_dist.items(), 
                                      key=lambda x: x[1]['count'], reverse=True):
        print(f"  - {prec_name}: {prec_info['count']} 次 ({prec_info['percentage']:.2f}%)")
    
    results['adaptive'] = {
        'iterations': adaptive_result['iterations'],
        'compute_cost': adaptive_result['compute_cost'],
        'final_residual': adaptive_result['final_residual'],
        'converged': adaptive_result['converged'],
        'avg_tile_cost': adaptive_result['avg_tile_cost'],
        'runtime': adaptive_time,
        'precision_distribution': adaptive_precision_dist
    }

    # 4. 全精度策略（基准）
    print("\n" + "=" * 80)
    print("🎯 策略 4: 全精度基准 (fp64)")
    print("=" * 80)

    env_fp64 = CGEnvironment(env_config)

    class FullPrecisionAgent:
        """全精度代理，总是选择fp64"""
        def __init__(self, env):
            self.env = env

        def act(self, obs):
            # 总是返回0（fp64）
            tilesize = self.env.spmv_sim.tilesize
            matrix_size = self.env.matrix_size
            num_tiles = (matrix_size + tilesize - 1) // tilesize
            return [0] * num_tiles

        def observe(self, obs, reward, done, done2):
            pass

        def eval_mode(self):
            pass

    fp64_agent = FullPrecisionAgent(env_fp64)

    fp64_start_time = time.time()
    fp64_result = run_episode_with_selector(env_fp64, fp64_agent,
                                           seed=random_seed, fixed_b=saved_b)
    fp64_end_time = time.time()
    fp64_time = fp64_end_time - fp64_start_time

    fp64_precision_dist = {'fp64': {'count': fp64_result['iterations'] * num_tiles, 'percentage': 100.0}}

    print(f"\n全精度基准结果:")
    print(f"  - 收敛迭代次数: {fp64_result['iterations']}")
    print(f"  - 总计算成本: {fp64_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {fp64_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if fp64_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {fp64_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {fp64_time:.2f} 秒")
    print(f"\n精度选择分布:")
    print("  - fp64: 100.00% (全精度基准)")

    results['full_precision'] = {
        'iterations': fp64_result['iterations'],
        'compute_cost': fp64_result['compute_cost'],
        'final_residual': fp64_result['final_residual'],
        'converged': fp64_result['converged'],
        'avg_tile_cost': fp64_result['avg_tile_cost'],
        'runtime': fp64_time,
        'precision_distribution': fp64_precision_dist
    }
    
    # 4. 性能对比分析
    print("\n" + "=" * 80)
    print("📈 性能对比分析")
    print("=" * 80)
    
    model_cost = results['model']['compute_cost']
    lw_cost = results['lightweight']['compute_cost']
    adaptive_cost = results['adaptive']['compute_cost']
    fp64_cost = results['full_precision']['compute_cost']

    model_iter = results['model']['iterations']
    lw_iter = results['lightweight']['iterations']
    adaptive_iter = results['adaptive']['iterations']
    fp64_iter = results['full_precision']['iterations']
    
    print(f"\n计算成本对比 (相对于全精度fp64):")
    print(f"  全精度基准:   {fp64_cost:.6f} (基准)")
    print(f"  模型指导:     {model_cost:.6f} ({((model_cost - fp64_cost) / fp64_cost * 100):+.2f}%)")
    print(f"  基础轻量级:   {lw_cost:.6f} ({((lw_cost - fp64_cost) / fp64_cost * 100):+.2f}%)")
    print(f"  自适应轻量级: {adaptive_cost:.6f} ({((adaptive_cost - fp64_cost) / fp64_cost * 100):+.2f}%)")

    print(f"\n迭代次数对比:")
    print(f"  全精度基准:   {fp64_iter}")
    print(f"  模型指导:     {model_iter} ({model_iter - fp64_iter:+d})")
    print(f"  基础轻量级:   {lw_iter} ({lw_iter - fp64_iter:+d})")
    print(f"  自适应轻量级: {adaptive_iter} ({adaptive_iter - fp64_iter:+d})")

    print(f"\n最终残差对比:")
    print(f"  全精度基准:   {results['full_precision']['final_residual']:.6e}")
    print(f"  模型指导:     {results['model']['final_residual']:.6e}")
    print(f"  基础轻量级:   {results['lightweight']['final_residual']:.6e}")
    print(f"  自适应轻量级: {results['adaptive']['final_residual']:.6e}")

    print(f"\n收敛性对比:")
    print(f"  全精度基准:   {'✓' if results['full_precision']['converged'] else '✗'}")
    print(f"  模型指导:     {'✓' if results['model']['converged'] else '✗'}")
    print(f"  基础轻量级:   {'✓' if results['lightweight']['converged'] else '✗'}")
    print(f"  自适应轻量级: {'✓' if results['adaptive']['converged'] else '✗'}")
    
    # 保存结果
    if output_path is None:
        result_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
        output_path = os.path.join(result_dir, f"lightweight_validation_{matrix_name if matrix_name else matrix_size}.json")
    
    save_data = {
        'model_path': model_path,
        'matrix_name': matrix_name,
        'matrix_size': env.matrix_size,
        'random_seed': random_seed,
        'num_tiles': num_tiles,
        'tilesize': tilesize,
        'results': results
    }
    
    from utils import convert_to_serializable
    save_data = convert_to_serializable(save_data)
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n📄 验证结果已保存至: {output_path}")
    
    return save_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='验证轻量级精度选择器的性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用指定矩阵验证
  python validate_lightweight_selector.py --model_path ./log/best_model --matrix_name Muu

  # 使用自定义随机种子
  python validate_lightweight_selector.py --model_path ./log/best_model --matrix_name Muu --seed 123

  # 使用指定大小的随机矩阵
  python validate_lightweight_selector.py --model_path ./log/best_model --matrix_size 1024
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型权重文件路径'
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
        help='配置文件路径（默认: config/default.yaml，相对于项目根目录）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（默认: 自动生成）'
    )
    
    args = parser.parse_args()
    
    # 处理matrix_name参数
    matrix_name = None
    if args.matrix_name:
        if args.matrix_name.lower() == 'none':
            matrix_name = None
        else:
            matrix_name = args.matrix_name
    
    # 处理matrix_size参数
    matrix_size = args.matrix_size
    
    if matrix_name is not None and matrix_size is not None:
        print("⚠️  警告: 指定了matrix_name时，matrix_size将被忽略")
        matrix_size = None
    
    # 运行验证
    try:
        compare_strategies(
            model_path=args.model_path,
            matrix_name=matrix_name,
            matrix_size=matrix_size,
            config_path=args.config_path,
            random_seed=args.seed,
            output_path=args.output
        )
    except Exception as e:
        print(f"\n❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

