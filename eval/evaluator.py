#!/usr/bin/env python3
"""
评估脚本：比较全精度 baseline 和模型指导的混合精度 CG 求解性能
用户指定模型权重文件和矩阵，随机生成b，x初始值设为全0
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from typing import Dict, Optional, List

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入项目模块
from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import PPOAgentFactory, CGPPOAgent
from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent


class DoublePrecisionAgent:
    """总是选择双精度 (fp64) 的简单代理"""

    def __init__(self, num_tiles: int):
        self.num_tiles = num_tiles

    def act(self, obs):
        """总是返回 fp64 动作 (0)"""
        return [0] * self.num_tiles  # 0 = fp64

    def observe(self, obs, reward, done, reset):
        """什么都不做"""
        pass

    def eval_mode(self):
        """评估模式"""
        return self


def create_env_config(config: Dict, matrix_name: Optional[str] = None, 
                      matrix_size: Optional[int] = None) -> Dict:
    """
    从训练配置创建环境配置

    Args:
        config: 训练配置字典
        matrix_name: 矩阵名称（如果指定，会覆盖配置中的值）
        matrix_size: 矩阵大小（如果指定且matrix_name为None，会覆盖配置中的值）

    Returns:
        环境配置字典
    """
    # 确定matrix_name
    final_matrix_name = matrix_name if matrix_name is not None else config.get('cg', {}).get('matrix_name', 'None')
    
    if matrix_size is not None and final_matrix_name == 'None':
        final_matrix_size = matrix_size
    if matrix_size is not None:
        final_matrix_size = matrix_size
    else:
        final_matrix_size = config.get('cg', {}).get('matrix_size', 1024)
    
    env_config = {
        'max_iter': config.get('cg', {}).get('max_iter', 100),
        'stop_tol': config.get('cg', {}).get('stop_tol', 1e-10),
        'matrix_size': final_matrix_size,
        'matrix_name': final_matrix_name,
        'matrix_data_dir': config.get('cg', {}).get('matrix_data_dir', '~/data/matrix'),
        'matrix_set_csv': config.get('cg', {}).get('matrix_set_csv', 'matrix_set.csv'),
        'tilesize': config.get('spmv', {}).get('tilesize', 32),
        'precision_cost_table': config.get('spmv', {}).get('precision_cost_table', {
            'fp64': 1.0, 'fp32': 0.7, 'tf32': 0.55,
            'fp16': 0.35, 'bf16': 0.33, 'fp8': 0.15
        }),
        'reward': config.get('reward'),
        'normalize_state': config.get('env', {}).get('normalize_state', True),
        'random_seed': config.get('random_seed', 42)  # 使用固定种子确保可重复性
    }
    return env_config


def run_episode_with_agent(env: CGEnvironment, agent, seed: int = 42, 
                           fixed_b: Optional[np.ndarray] = None) -> Dict:
    """
    使用给定的代理运行一个完整的 CG episode

    Args:
        env: CG 环境
        agent: 代理（可以是 DoublePrecisionAgent 或其他代理）
        seed: 随机种子，用于确保b的生成一致
        fixed_b: 如果提供，使用固定的b向量（用于确保两次运行使用相同的b）

    Returns:
        包含收敛 iterations、compute_cost 和精度选择历史的字典
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 重置环境
    obs = env.reset(seed=seed)
    
    # 如果提供了固定的b，使用它覆盖环境中的b
    if fixed_b is not None:
        env.b = fixed_b.copy()
        env.b_norm = np.linalg.norm(env.b)
        # 重新计算初始残差
        env.x = np.zeros(env.matrix_size)
        env.r = env._compute_exact_residual(env.x, env.A_diagonal, env.b)
        env.p = env.r.copy()
        # 重新记录初始残差
        initial_residual_norm = env.math_sim.vector_norm(env.r)
        env.residual_tracker.reset()
        env.residual_tracker.record_residual(initial_residual_norm)
        # 重新获取状态
        obs = env.get_state_features(env.p, env.current_iteration)
    done = False
    total_reward = 0.0
    step_count = 0
    
    # 记录每次迭代的精度选择
    precision_history = []  # List[List[int]]: 每次迭代的tile精度选择

    while not done:
        # 选择动作
        actions = agent.act(obs)
        
        # 记录当前迭代的精度选择
        precision_history.append(actions.copy())

        # 执行一步
        next_obs, reward, done, info = env.step(actions)
        total_reward += reward

        # 观察转换
        agent.observe(obs, reward, done, done)

        obs = next_obs
        step_count += 1

        if step_count % 10 == 0:
            print(f"  步骤 {step_count}: 残差 = {info['residual_norm']:.6e}, 成本 = {info['iteration_cost']:.6f}")

    # 获取 episode 统计信息
    episode_info = env.get_episode_info()

    result = {
        'iterations': episode_info['iterations'],
        'compute_cost': episode_info['total_cost'],
        'final_residual': episode_info['final_residual'],
        'converged': bool(episode_info['converged']),  # 转换为布尔值
        'avg_tile_cost': episode_info['avg_tile_cost'],
        'initial_residual': episode_info.get('initial_residual', None),
        'precision_history': precision_history  # 添加精度选择历史
    }

    return result


def load_model_weights(model_path: str, config: Dict, env: CGEnvironment):
    """
    加载模型权重文件并创建代理

    Args:
        model_path: 模型权重文件路径（可以是.pt文件或目录，会自动处理）
        config: 训练配置
        env: CG环境（用于获取状态和动作空间维度）

    Returns:
        CGPPOAgent 实例
    """
    # 检查模型文件是否存在
    original_path = model_path
    if not os.path.exists(model_path):
        # 尝试添加.pt后缀
        if os.path.exists(f"{model_path}.pt"):
            model_path = f"{model_path}.pt"
        else:
            raise FileNotFoundError(f"找不到模型权重文件: {original_path} 或 {original_path}.pt")

    print(f"加载模型权重文件: {model_path}")
    
    # 创建代理
    state_dim = env.get_state_dim()
    action_size = env.get_action_space_size()
    cg_agent = PPOAgentFactory(config).create_agent(state_dim, action_size)

    # 加载模型权重（CGPPOAgent的load方法会处理路径）
    try:
        cg_agent.load(model_path)
    except Exception as e:
        # 如果加载失败，尝试直接加载权重文件（兼容旧格式）
        print(f"标准加载方式失败: {e}")
        print("尝试直接加载权重文件...")
        if os.path.exists(model_path):
            weights = torch.load(model_path, map_location='cpu', weights_only=False)
            # 检查是否是直接的state_dict格式（包含child_modules键）
            if isinstance(weights, dict):
                # 检查是否包含模型权重键（如child_modules.0.0.0.weight）
                has_model_keys = any('child_modules' in str(k) or 'weight' in str(k) for k in weights.keys())
                if has_model_keys:
                    # 直接加载到模型
                    model = cg_agent.tile_agents[0].model
                    model.load_state_dict(weights)
                    print("直接加载权重成功")
                else:
                    raise ValueError(f"无法识别的权重文件格式: {model_path}")
            else:
                raise ValueError(f"权重文件不是字典格式: {model_path}")
        else:
            raise
    
    # 设置为评估模式
    cg_agent.eval_mode()
    for tile_agent in cg_agent.tile_agents:
        if hasattr(tile_agent, 'eval'):
            tile_agent.eval()
        if hasattr(tile_agent, 'training'):
            tile_agent.training = False

    print("模型权重加载完成")
    return cg_agent


def plot_precision_heatmap(
    precision_history: List[List[int]],
    num_tiles: int,
    save_path: str,
    title: str = "Precision Selection Heatmap"
):
    """
    Plot precision selection heatmap (English only in the figure)

    Args:
        precision_history: List[List[int]], where precision_history[i] contains the precision selection for all tiles at iteration i
        num_tiles: number of tiles
        save_path: path to save the heatmap image
        title: plot title (defaults to "Precision Selection Heatmap")
    """
    if not precision_history:
        print("Warning: No precision selection data. Skipping heatmap plot.")
        return

    # Precision mapping: 0=fp64, 1=fp32, 2=tf32, 3=fp16, 4=bf16, 5=fp8
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    # precision_colors = ['#2e4e77', '#336e99', '#47a49d', '#6fbd8f', '#9acda2', '#ebf6e6']
    precision_colors = ['#e66d50', '#f3a361', '#e7c66b', '#297270', '#299d8f', '#8ab07c']

    num_iterations = len(precision_history)
    # Ensure consistent tile count (take the largest per iteration if variable)
    max_tiles = max(len(actions) for actions in precision_history) if precision_history else num_tiles

    # Build heatmap matrix (rows=iterations, columns=tile indices)
    precision_matrix = np.zeros((num_iterations, max_tiles), dtype=int)
    
    # Count precision usage for calculating proportions
    precision_counts = {i: 0 for i in range(6)}
    total_selections = 0

    for iter_idx, actions in enumerate(precision_history):
        for tile_idx, action in enumerate(actions):
            if tile_idx < max_tiles:
                precision_matrix[iter_idx, tile_idx] = action
                if 0 <= action < 6:
                    precision_counts[action] += 1
                    total_selections += 1

    # Calculate proportions
    precision_proportions = {}
    if total_selections > 0:
        for prec_idx in range(6):
            count = precision_counts[prec_idx]
            proportion = count / total_selections * 100
            precision_proportions[prec_idx] = proportion
    else:
        precision_proportions = {i: 0.0 for i in range(6)}

    # Figure size (scales with tiles/iterations)
    # Calculate figure size to accommodate all tiles
    # Use a reasonable base size and scale based on tile count
    # For many tiles, we need a wider figure but keep reasonable aspect ratio
    base_width = 12
    base_height = 8
    # Scale width based on tile count (each tile needs some space)
    # Use smaller per-tile width when there are many tiles to fit everything
    if max_tiles > 100:
        # For many tiles, use smaller per-tile width to fit all
        fig_width = max(base_width, max_tiles * 0.05)  # ~0.05 inches per tile
    else:
        fig_width = max(base_width, max_tiles * 0.1)  # ~0.1 inches per tile
    
    fig_height = max(base_height, num_iterations * 0.3)  # ~0.3 inches per iteration
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Custom colormap
    cmap = mcolors.ListedColormap(precision_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Draw the heatmap
    im = ax.imshow(precision_matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    # Axis labels and title in English
    ax.set_xlabel('Tile Index', fontsize=24)
    ax.set_ylabel('Iteration', fontsize=24)
    ax.set_title(title, fontsize=32, fontweight='bold')

    # Y-axis: iteration count (starts at 1, not 0)
    ax.set_yticks(range(num_iterations))
    ax.set_yticklabels(range(1, num_iterations + 1))
    ax.tick_params(axis='y', labelsize=18)  # Increase y-axis tick font size

    # X-axis: show all tiles, but adjust label display for readability
    # Always set all tick positions to ensure full range is displayed
    ax.set_xlim(-0.5, max_tiles - 0.5)  # Ensure full range is visible
    
    # For readability, we can show fewer labels but still display all tiles
    if max_tiles <= 100:
        # Show all tile indices when tile count is reasonable
        ax.set_xticks(range(max_tiles))
        ax.set_xticklabels(range(max_tiles))
    else:
        # For many tiles, show labels at intervals but still display all tiles
        # The heatmap will show all tiles, labels are just for reference
        step = max(1, max_tiles // 50)  # Show ~50 labels max
        tick_positions = list(range(0, max_tiles, step))
        # Always include the last tile index
        if tick_positions[-1] != max_tiles - 1:
            tick_positions.append(max_tiles - 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions)
    
    ax.tick_params(axis='x', labelsize=18)  # Increase x-axis tick font size

    # Colorbar with proportions displayed
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5])
    cbar.set_label('Precision Type', fontsize=24)
    # Format labels with precision name and proportion
    tick_labels = []
    for prec_idx in range(6):
        prec_name = precision_names[prec_idx]
        prop = precision_proportions[prec_idx]
        tick_labels.append(f'{prec_name}\n({prop:.1f}%)')
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=18)  # Increase colorbar tick font size

    # Optional: no grid
    ax.grid(False)

    # Layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision selection heatmap saved to: {save_path}")

    # Close to free memory
    plt.close(fig)


def evaluate_model(model_path: str, matrix_name: Optional[str] = None, 
                   matrix_size: Optional[int] = None,
                   config_path: str = 'config/default.yaml', 
                   random_seed: int = 42):
    """
    评估模型性能：比较全精度 baseline 和模型指导的混合精度求解

    Args:
        model_path: 模型权重文件路径
        matrix_name: 矩阵名称（如果为None，则使用随机矩阵）
        matrix_size: 矩阵大小（当matrix_name为None时使用，用于生成随机矩阵）
        config_path: 配置文件路径
        random_seed: 随机种子，用于确保两次运行使用相同的b
    """
    print("=" * 80)
    print("🎯 CG 混合精度模型评估")
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

    # 1. 全精度 baseline 评估
    print("\n" + "=" * 80)
    print("📊 步骤 1: 运行全精度 baseline 评估")
    print("=" * 80)
    
    dp_agent = DoublePrecisionAgent(num_tiles)
    dp_agent.eval_mode()
    
    dp_start_time = time.time()
    dp_result = run_episode_with_agent(env, dp_agent, seed=random_seed)
    dp_end_time = time.time()
    dp_time = dp_end_time - dp_start_time

    print(f"\n全精度 baseline 结果:")
    print(f"  - 收敛迭代次数: {dp_result['iterations']}")
    print(f"  - 总计算成本: {dp_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {dp_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if dp_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {dp_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {dp_time:.2f} 秒")

    # 保存b的值，以便第二次运行使用相同的b
    saved_b = env.b.copy() if hasattr(env, 'b') and env.b is not None else None
    if saved_b is not None:
        print(f"  - 保存b向量用于第二次运行 (长度: {len(saved_b)})")

    # 2. 模型指导的混合精度评估
    print("\n" + "=" * 80)
    print("🤖 步骤 2: 运行模型指导的混合精度评估")
    print("=" * 80)

    # 重新创建环境（确保使用相同的配置和seed）
    env_model = CGEnvironment(env_config)
    
    # 加载模型
    print("\n加载模型权重...")
    cg_agent = load_model_weights(model_path, config, env_model)
    
    # 创建评估代理
    pfrl_agent = PfrlCompatibleCGPPOAgent(cg_agent)
    pfrl_agent.eval_mode()

    # 运行模型指导的 episode（使用相同的b）
    model_start_time = time.time()
    model_result = run_episode_with_agent(env_model, pfrl_agent, seed=random_seed, fixed_b=saved_b)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time

    print(f"\n模型指导的混合精度结果:")
    print(f"  - 收敛迭代次数: {model_result['iterations']}")
    print(f"  - 总计算成本: {model_result['compute_cost']:.6f}")
    print(f"  - 最终残差: {model_result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if model_result['converged'] else '否'}")
    print(f"  - 平均 tile 成本: {model_result['avg_tile_cost']:.6f}")
    print(f"  - 运行时间: {model_time:.2f} 秒")

    # 3. 性能对比
    print("\n" + "=" * 80)
    print("📈 性能对比分析")
    print("=" * 80)

    dp_cost = dp_result['compute_cost']
    model_cost = model_result['compute_cost']
    dp_iterations = dp_result['iterations']
    model_iterations = model_result['iterations']

    cost_improvement = (dp_cost - model_cost) / dp_cost * 100 if dp_cost > 0 else 0
    iteration_change = model_iterations - dp_iterations
    iteration_change_percent = (iteration_change / dp_iterations * 100) if dp_iterations > 0 else 0

    print(f"\n计算成本对比:")
    print(f"  全精度 baseline:     {dp_cost:.6f}")
    print(f"  模型指导混合精度:   {model_cost:.6f}")
    print(f"  成本减少:           {dp_cost - model_cost:.6f}")
    print(f"  性能提升:           {cost_improvement:.2f}%")

    print(f"\n迭代次数对比:")
    print(f"  全精度 baseline:     {dp_iterations}")
    print(f"  模型指导混合精度:   {model_iterations}")
    print(f"  迭代次数变化:       {iteration_change:+d} ({iteration_change_percent:+.2f}%)")

    print(f"\n收敛性对比:")
    print(f"  全精度收敛:         {'是' if dp_result['converged'] else '否'}")
    print(f"  模型指导收敛:       {'是' if model_result['converged'] else '否'}")
    
    if dp_result.get('initial_residual') and model_result.get('initial_residual'):
        print(f"  初始残差 (全精度):  {dp_result['initial_residual']:.6e}")
        print(f"  初始残差 (模型):    {model_result['initial_residual']:.6e}")

    print(f"\n最终残差对比:")
    print(f"  全精度最终残差:     {dp_result['final_residual']:.6e}")
    print(f"  模型指导最终残差:   {model_result['final_residual']:.6e}")
    print(f"  残差比率:           {model_result['final_residual'] / dp_result['final_residual']:.4f}")

    # 4. 绘制精度选择热力图
    plot_path = None
    if model_result.get('precision_history'):
        print("\n" + "=" * 80)
        print("📊 步骤 3: 绘制精度选择热力图")
        print("=" * 80)
        
        # 确定保存路径
        result_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
        plot_filename = f"precision_heatmap_{int(time.time())}.png"
        plot_path = os.path.join(result_dir, plot_filename)
        
        # 绘制热力图
        plot_precision_heatmap(
            precision_history=model_result['precision_history'],
            num_tiles=num_tiles,
            save_path=plot_path,
            title=f"Precision Selection Heatmap\n(Matrix: {matrix_name if matrix_name else f'Random {env.matrix_size}x{env.matrix_size}'})"
        )

    # 保存评估结果
    eval_result = {
        'model_path': model_path,
        'matrix_name': matrix_name,
        'matrix_size': env.matrix_size,
        'specified_matrix_size': matrix_size,  # 用户指定的matrix_size（如果指定了）
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
        'model_guided': {
            'iterations': model_result['iterations'],
            'compute_cost': model_result['compute_cost'],
            'final_residual': model_result['final_residual'],
            'converged': model_result['converged'],
            'avg_tile_cost': model_result['avg_tile_cost'],
            'runtime_seconds': model_time
        },
        'comparison': {
            'cost_reduction': dp_cost - model_cost,
            'cost_improvement_percent': cost_improvement,
            'iteration_change': iteration_change,
            'iteration_change_percent': iteration_change_percent,
            'residual_ratio': model_result['final_residual'] / dp_result['final_residual'] if dp_result['final_residual'] > 0 else None
        }
    }
    
    # 添加精度热力图路径（如果已生成）
    if plot_path:
        eval_result['precision_heatmap_path'] = plot_path

    # 保存结果到JSON文件
    result_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    result_filename = f"evaluation_result_{int(time.time())}.json"
    result_path = os.path.join(result_dir, result_filename)
    
    with open(result_path, 'w') as f:
        json.dump(eval_result, f, indent=2)
    
    print(f"\n📄 详细评估结果已保存至: {result_path}")

    # 总结
    print("\n" + "=" * 80)
    if cost_improvement > 0:
        print("🎉 评估完成！模型成功实现了性能提升！")
    else:
        print("⚠️  评估完成！模型成本高于全精度 baseline，可能需要进一步优化。")
    print("=" * 80)

    return eval_result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='评估CG混合精度模型性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用指定矩阵评估模型
  python eval/evaluator.py --model_path ./log/best_model --matrix_name Muu

  # 使用自定义随机种子
  python eval/evaluator.py --model_path ./log/best_model --matrix_name Muu --seed 123

  # 使用随机矩阵（使用配置文件中的matrix_size）
  python eval/evaluator.py --model_path ./log/best_model --matrix_name None

  # 使用指定大小的随机矩阵
  python eval/evaluator.py --model_path ./log/best_model --matrix_size 512

  # 使用指定大小的随机矩阵并指定随机种子
  python eval/evaluator.py --model_path ./log/best_model --matrix_size 1024 --seed 42
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型权重文件路径（不包含.pt后缀，会自动尝试添加）'
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
    matrix_name = None
    if args.matrix_name:
        if args.matrix_name.lower() == 'none':
            matrix_name = None
        else:
            matrix_name = args.matrix_name

    # 处理matrix_size参数
    matrix_size = args.matrix_size
    
    # 如果指定了matrix_name，忽略matrix_size（因为真实矩阵的大小由矩阵文件决定）
    if matrix_name is not None and matrix_size is not None:
        print("⚠️  警告: 指定了matrix_name时，matrix_size将被忽略（矩阵大小由矩阵文件决定）")
        matrix_size = None

    # 运行评估
    try:
        evaluate_model(
            model_path=args.model_path,
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

