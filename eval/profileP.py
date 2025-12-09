#!/usr/bin/env python3
"""
分析脚本：统计模型指导的 episode 中，选择相同精度的 sub_p 的特征
运行一次模型指导的 episode，记录每个 tile 的精度选择和对应的 sub_p，
然后统计选择相同精度的 sub_p 的规律
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
from typing import Dict, Optional, List, Tuple
from collections import defaultdict

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入项目模块
from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import PPOAgentFactory
from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent


def create_env_config(config: Dict, matrix_name: Optional[str] = None, 
                      matrix_size: Optional[int] = None) -> Dict:
    """
    从训练配置创建环境配置
    """
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
        'random_seed': config.get('random_seed', 42)
    }
    return env_config


def load_model_weights(model_path: str, config: Dict, env: CGEnvironment):
    """
    加载模型权重文件并创建代理
    """
    original_path = model_path
    if not os.path.exists(model_path):
        if os.path.exists(f"{model_path}.pt"):
            model_path = f"{model_path}.pt"
        else:
            raise FileNotFoundError(f"找不到模型权重文件: {original_path} 或 {original_path}.pt")

    print(f"加载模型权重文件: {model_path}")
    
    state_dim = env.get_state_dim()
    action_size = env.get_action_space_size()
    cg_agent = PPOAgentFactory(config).create_agent(state_dim, action_size)

    try:
        cg_agent.load(model_path)
    except Exception as e:
        print(f"标准加载方式失败: {e}")
        print("尝试直接加载权重文件...")
        if os.path.exists(model_path):
            weights = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(weights, dict):
                has_model_keys = any('child_modules' in str(k) or 'weight' in str(k) for k in weights.keys())
                if has_model_keys:
                    model = cg_agent.tile_agents[0].model
                    model.load_state_dict(weights)
                    print("直接加载权重成功")
                else:
                    raise ValueError(f"无法识别的权重文件格式: {model_path}")
            else:
                raise ValueError(f"权重文件不是字典格式: {model_path}")
        else:
            raise
    
    cg_agent.eval_mode()
    for tile_agent in cg_agent.tile_agents:
        if hasattr(tile_agent, 'eval'):
            tile_agent.eval()
        if hasattr(tile_agent, 'training'):
            tile_agent.training = False

    print("模型权重加载完成")
    return cg_agent


def extract_sub_p_features(sub_p: np.ndarray) -> Dict[str, float]:
    """
    提取 sub_p 的特征统计信息
    
    Args:
        sub_p: 子向量（可以是 numpy 数组或列表）
        
    Returns:
        特征字典
    """
    # 确保 sub_p 是 numpy 数组
    if not isinstance(sub_p, np.ndarray):
        sub_p = np.array(sub_p)
    
    # 确保是浮点数类型
    sub_p = sub_p.astype(np.float64)
    
    features = {
        'mean': float(np.mean(sub_p)),
        'std': float(np.std(sub_p)),
        'min': float(np.min(sub_p)),
        'max': float(np.max(sub_p)),
        'l1_norm': float(np.linalg.norm(sub_p, ord=1)),
        'l2_norm': float(np.linalg.norm(sub_p, ord=2)),
        'max_abs': float(np.max(np.abs(sub_p))),
        'mean_abs': float(np.mean(np.abs(sub_p))),
        'range': float(np.max(sub_p) - np.min(sub_p)),
        'size': len(sub_p)
    }
    
    # 计算非零元素比例
    non_zero_count = np.count_nonzero(sub_p)
    features['non_zero_ratio'] = float(non_zero_count / len(sub_p)) if len(sub_p) > 0 else 0.0
    
    # 计算正负元素比例
    positive_count = np.count_nonzero(sub_p > 0)
    negative_count = np.count_nonzero(sub_p < 0)
    features['positive_ratio'] = float(positive_count / len(sub_p)) if len(sub_p) > 0 else 0.0
    features['negative_ratio'] = float(negative_count / len(sub_p)) if len(sub_p) > 0 else 0.0
    
    return features


def run_episode_with_profiling(env: CGEnvironment, agent, seed: int = 42, 
                                fixed_b: Optional[np.ndarray] = None) -> Dict:
    """
    运行一个完整的 CG episode，并记录每个 tile 的精度选择和 sub_p 特征
    
    Returns:
        包含精度选择历史和 sub_p 特征的字典
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
        obs = env.get_state_features(env.p, env.current_iteration)
    
    done = False
    step_count = 0
    
    # 记录数据：按精度分组存储 sub_p 特征
    # precision_data[precision_code] = List[Dict] 每个Dict包含一次选择的sub_p特征和迭代信息
    precision_data = defaultdict(list)
    
    # 记录每次迭代的详细信息
    iteration_records = []
    
    tilesize = env.spmv_sim.tilesize
    num_tiles = (env.matrix_size + tilesize - 1) // tilesize
    
    while not done:
        # 记录当前迭代的 p 向量和 tile 信息
        iteration_info = {
            'iteration': env.current_iteration,
            'tiles': []
        }
        
        # 为每个 tile 提取 sub_p
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tilesize
            end_idx = min(start_idx + tilesize, env.matrix_size)
            sub_p = env.p[start_idx:end_idx].copy()
            
            # 确保 sub_p 是 numpy 数组
            if not isinstance(sub_p, np.ndarray):
                sub_p = np.array(sub_p)
            
            # 提取特征
            sub_p_features = extract_sub_p_features(sub_p)
            sub_p_features['tile_idx'] = tile_idx
            sub_p_features['iteration'] = env.current_iteration
            
            iteration_info['tiles'].append({
                'tile_idx': tile_idx,
                'sub_p': sub_p.tolist(),  # 保存原始数据用于后续分析
                'features': sub_p_features
            })
        
        # 选择动作
        actions = agent.act(obs)
        
        # 记录每个 tile 的精度选择
        for tile_idx, action in enumerate(actions):
            tile_info = iteration_info['tiles'][tile_idx]
            tile_info['precision'] = action
            tile_info['precision_name'] = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8'][action]
            
            # 添加到按精度分组的数据中
            precision_data[action].append({
                'iteration': env.current_iteration,
                'tile_idx': tile_idx,
                'features': tile_info['features'],
                'sub_p': tile_info['sub_p']
            })
        
        iteration_records.append(iteration_info)
        
        # 执行一步
        next_obs, reward, done, info = env.step(actions)
        
        agent.observe(obs, reward, done, done)
        obs = next_obs
        step_count += 1
        
        if step_count % 10 == 0:
            print(f"  步骤 {step_count}: 残差 = {info['residual_norm']:.6e}")
    
    episode_info = env.get_episode_info()
    
    result = {
        'iterations': episode_info['iterations'],
        'compute_cost': episode_info['total_cost'],
        'final_residual': episode_info['final_residual'],
        'converged': bool(episode_info['converged']),
        'precision_data': dict(precision_data),  # 转换为普通字典
        'iteration_records': iteration_records
    }
    
    return result


def compute_precision_statistics(precision_data: Dict[int, List[Dict]]) -> Dict[int, Dict]:
    """
    计算每种精度的统计特征
    
    Args:
        precision_data: 按精度分组的数据
        
    Returns:
        每种精度的统计特征字典
    """
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    statistics = {}
    
    for precision_code in range(6):
        if precision_code not in precision_data or len(precision_data[precision_code]) == 0:
            statistics[precision_code] = {
                'precision_name': precision_names[precision_code],
                'count': 0,
                'message': '未选择此精度'
            }
            continue
        
        data_list = precision_data[precision_code]
        count = len(data_list)
        
        # 提取所有特征值
        features_list = [item['features'] for item in data_list]
        
        # 计算每个特征的统计量
        feature_stats = {}
        feature_keys = ['mean', 'std', 'min', 'max', 'l1_norm', 'l2_norm', 
                       'max_abs', 'mean_abs', 'range', 'non_zero_ratio',
                       'positive_ratio', 'negative_ratio']
        
        for key in feature_keys:
            values = [f[key] for f in features_list]
            feature_stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # 计算迭代分布
        iterations = [item['iteration'] for item in data_list]
        iteration_stats = {
            'mean': float(np.mean(iterations)),
            'std': float(np.std(iterations)),
            'min': int(np.min(iterations)),
            'max': int(np.max(iterations)),
            'median': float(np.median(iterations))
        }
        
        statistics[precision_code] = {
            'precision_name': precision_names[precision_code],
            'count': count,
            'feature_statistics': feature_stats,
            'iteration_statistics': iteration_stats
        }
    
    return statistics


def print_statistics_report(statistics: Dict[int, Dict], total_selections: int):
    """
    打印统计报告
    """
    print("\n" + "=" * 80)
    print("📊 精度选择统计报告")
    print("=" * 80)
    
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    
    for precision_code in range(6):
        stat = statistics[precision_code]
        print(f"\n【{stat['precision_name']} (精度代码: {precision_code})】")
        print(f"  选择次数: {stat['count']}")
        
        if stat['count'] == 0:
            print(f"  {stat['message']}")
            continue
        
        count = stat['count']
        percentage = (count / total_selections * 100) if total_selections > 0 else 0
        print(f"  占比: {percentage:.2f}%")
        
        # 迭代统计
        iter_stat = stat['iteration_statistics']
        print(f"\n  迭代分布:")
        print(f"    平均迭代次数: {iter_stat['mean']:.2f}")
        print(f"    迭代次数范围: [{iter_stat['min']}, {iter_stat['max']}]")
        print(f"    迭代次数中位数: {iter_stat['median']:.2f}")
        
        # 特征统计
        feat_stat = stat['feature_statistics']
        print(f"\n  Sub-p 特征统计:")
        print(f"    L2范数: 均值={feat_stat['l2_norm']['mean']:.6e}, "
              f"标准差={feat_stat['l2_norm']['std']:.6e}, "
              f"范围=[{feat_stat['l2_norm']['min']:.6e}, {feat_stat['l2_norm']['max']:.6e}]")
        print(f"    L1范数: 均值={feat_stat['l1_norm']['mean']:.6e}, "
              f"标准差={feat_stat['l1_norm']['std']:.6e}")
        print(f"    最大值绝对值: 均值={feat_stat['max_abs']['mean']:.6e}, "
              f"范围=[{feat_stat['max_abs']['min']:.6e}, {feat_stat['max_abs']['max']:.6e}]")
        print(f"    均值绝对值: 均值={feat_stat['mean_abs']['mean']:.6e}")
        print(f"    标准差: 均值={feat_stat['std']['mean']:.6e}")
        print(f"    范围: 均值={feat_stat['range']['mean']:.6e}")
        print(f"    非零比例: 均值={feat_stat['non_zero_ratio']['mean']:.4f}, "
              f"范围=[{feat_stat['non_zero_ratio']['min']:.4f}, {feat_stat['non_zero_ratio']['max']:.4f}]")
        print(f"    正数比例: 均值={feat_stat['positive_ratio']['mean']:.4f}")
        print(f"    负数比例: 均值={feat_stat['negative_ratio']['mean']:.4f}")


def aggregate_precision_statistics(precision_stats_list: List[Tuple[int, Dict]]) -> Dict:
    """
    Aggregate statistics from multiple precision types (weighted by count).
    
    Args:
        precision_stats_list: List of (code, stat) tuples
        
    Returns:
        Aggregated statistics dictionary
    """
    if not precision_stats_list:
        return None
    
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    
    # Calculate total count
    total_count = sum(stat['count'] for _, stat in precision_stats_list)
    if total_count == 0:
        return None
    
    # Aggregate feature statistics (weighted average by count)
    feature_keys = ['mean', 'std', 'min', 'max', 'l1_norm', 'l2_norm', 
                   'max_abs', 'mean_abs', 'range', 'non_zero_ratio',
                   'positive_ratio', 'negative_ratio']
    
    aggregated_feat_stats = {}
    for feat_key in feature_keys:
        # Weighted mean of means
        weighted_mean = sum(stat['feature_statistics'][feat_key]['mean'] * stat['count'] 
                           for _, stat in precision_stats_list) / total_count
        
        # Weighted mean of stds
        weighted_std = sum(stat['feature_statistics'][feat_key]['std'] * stat['count'] 
                          for _, stat in precision_stats_list) / total_count
        
        # Min and max across all precisions
        all_mins = [stat['feature_statistics'][feat_key]['min'] 
                   for _, stat in precision_stats_list]
        all_maxs = [stat['feature_statistics'][feat_key]['max'] 
                   for _, stat in precision_stats_list]
        
        aggregated_feat_stats[feat_key] = {
            'mean': float(weighted_mean),
            'std': float(weighted_std),
            'min': float(min(all_mins)),
            'max': float(max(all_maxs)),
            'median': float(np.median([stat['feature_statistics'][feat_key]['median'] 
                                      for _, stat in precision_stats_list]))
        }
    
    # Aggregate iteration statistics (weighted average)
    weighted_iter_mean = sum(stat['iteration_statistics']['mean'] * stat['count'] 
                            for _, stat in precision_stats_list) / total_count
    weighted_iter_std = sum(stat['iteration_statistics']['std'] * stat['count'] 
                           for _, stat in precision_stats_list) / total_count
    
    all_iter_mins = [stat['iteration_statistics']['min'] 
                    for _, stat in precision_stats_list]
    all_iter_maxs = [stat['iteration_statistics']['max'] 
                    for _, stat in precision_stats_list]
    
    aggregated_iter_stats = {
        'mean': float(weighted_iter_mean),
        'std': float(weighted_iter_std),
        'min': int(min(all_iter_mins)),
        'max': int(max(all_iter_maxs)),
        'median': float(np.median([stat['iteration_statistics']['median'] 
                                  for _, stat in precision_stats_list]))
    }
    
    # Create list of precision names
    precision_name_list = [precision_names[code] for code, _ in precision_stats_list]
    
    return {
        'precision_names': precision_name_list,
        'count': total_count,
        'feature_statistics': aggregated_feat_stats,
        'iteration_statistics': aggregated_iter_stats
    }


def analyze_precision_patterns(statistics: Dict[int, Dict]) -> Dict[str, str]:
    """
    Analyze precision selection patterns and generate summary in English.
    """
    patterns = []
    
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    
    # Find the most frequently used precision
    counts = [(code, stat['count']) for code, stat in statistics.items() if stat['count'] > 0]
    if counts:
        counts.sort(key=lambda x: x[1], reverse=True)
        most_used = counts[0]
        patterns.append(f"The most frequently used precision is {precision_names[most_used[0]]}, selected {most_used[1]} times.")
    
    # Analyze feature differences among precisions
    used_precisions = [(code, stat) for code, stat in statistics.items() if stat['count'] > 0]
    
    if len(used_precisions) >= 2:
        # Collect all high precision types (fp64, fp32, tf32)
        high_precision_list = [(code, stat) for code, stat in used_precisions if code < 3]
        # Collect all low precision types (fp16, bf16, fp8)
        low_precision_list = [(code, stat) for code, stat in used_precisions if code >= 3]
        
        # Aggregate statistics for high and low precision groups
        high_precision_agg = aggregate_precision_statistics(high_precision_list) if high_precision_list else None
        low_precision_agg = aggregate_precision_statistics(low_precision_list) if low_precision_list else None
        
        if high_precision_agg and low_precision_agg:
            hp_feat = high_precision_agg['feature_statistics']
            lp_feat = low_precision_agg['feature_statistics']
            
            hp_names_str = ', '.join(high_precision_agg['precision_names'])
            lp_names_str = ', '.join(low_precision_agg['precision_names'])
            
            patterns.append(f"\nPrecision selection pattern analysis:")
            patterns.append(f"Comparing high precision group ({hp_names_str}) vs low precision group ({lp_names_str}):")
            patterns.append(f"  - High precision group: {high_precision_agg['count']} total selections")
            patterns.append(f"  - Low precision group: {low_precision_agg['count']} total selections")
            
            # Define all features to analyze with their display names and thresholds
            feature_configs = [
                ('mean', 'Mean value', 1e-10),
                ('std', 'Standard deviation', 1e-10),
                ('min', 'Minimum value', 1e-10),
                ('max', 'Maximum value', 1e-10),
                ('l1_norm', 'L1 norm', 1e-10),
                ('l2_norm', 'L2 norm', 1e-10),
                ('max_abs', 'Maximum absolute value', 1e-10),
                ('mean_abs', 'Mean absolute value', 1e-10),
                ('range', 'Range (max - min)', 1e-10),
                ('non_zero_ratio', 'Non-zero ratio', 1e-6),
                ('positive_ratio', 'Positive ratio', 1e-6),
                ('negative_ratio', 'Negative ratio', 1e-6),
            ]
            
            # Analyze each feature
            for feat_key, feat_display, threshold in feature_configs:
                hp_mean = hp_feat[feat_key]['mean']
                lp_mean = lp_feat[feat_key]['mean']
                
                patterns.append(f"\n  {feat_display}:")
                patterns.append(f"    - High precision group: {hp_mean:.6e} (std: {hp_feat[feat_key]['std']:.6e})")
                patterns.append(f"    - Low precision group: {lp_mean:.6e} (std: {lp_feat[feat_key]['std']:.6e})")
                
                # Calculate ratio and pattern
                if abs(hp_mean) > threshold and abs(lp_mean) > threshold:
                    ratio = hp_mean / lp_mean if lp_mean != 0 else float('inf')
                    if abs(ratio) < 1000 and abs(ratio) > 0.001:  # Reasonable ratio range
                        patterns.append(f"    - Ratio: {ratio:.2f}x")
                        
                        if ratio > 1.5:
                            patterns.append(f"    → Pattern: High precision group selects sub_p with larger {feat_display.lower()}.")
                        elif ratio < 0.67:
                            patterns.append(f"    → Pattern: Low precision group selects sub_p with larger {feat_display.lower()}.")
                        else:
                            patterns.append(f"    → Pattern: No significant difference in {feat_display.lower()}.")
                    else:
                        patterns.append(f"    → Pattern: Ratio is extreme ({ratio:.2e}x), values may be near zero or very different.")
                else:
                    patterns.append(f"    → Pattern: Values are too small (< {threshold:.0e}) to compare meaningfully.")
            
            # Compare iteration distribution
            hp_iter = high_precision_agg['iteration_statistics']['mean']
            lp_iter = low_precision_agg['iteration_statistics']['mean']
            patterns.append(f"\nIteration distribution analysis:")
            patterns.append(f"  - High precision group is selected at mean iteration {hp_iter:.1f}")
            patterns.append(f"  - Low precision group is selected at mean iteration {lp_iter:.1f}")
            
            if hp_iter < lp_iter:
                patterns.append(f"  → Pattern: High precision group is used more in early iterations.")
            elif hp_iter > lp_iter:
                patterns.append(f"  → Pattern: Low precision group is used more in early iterations.")
            else:
                patterns.append(f"  → Pattern: No significant difference in iteration distribution.")
    
    return {'patterns': '\n'.join(patterns)}


def plot_precision_feature_analysis(statistics: Dict[int, Dict], save_path: str, 
                                    matrix_name: Optional[str] = None,
                                    matrix_size: Optional[int] = None):
    """
    Plot precision selection and feature relationship analysis.
    
    Args:
        statistics: Statistics dictionary from compute_precision_statistics
        save_path: Path to save the figure
        matrix_name: Matrix name for title
        matrix_size: Matrix size for title
    """
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    
    # Collect high and low precision groups
    used_precisions = [(code, stat) for code, stat in statistics.items() if stat['count'] > 0]
    high_precision_list = [(code, stat) for code, stat in used_precisions if code < 3]
    low_precision_list = [(code, stat) for code, stat in used_precisions if code >= 3]
    
    # Aggregate statistics
    high_precision_agg = aggregate_precision_statistics(high_precision_list) if high_precision_list else None
    low_precision_agg = aggregate_precision_statistics(low_precision_list) if low_precision_list else None
    
    if not (high_precision_agg and low_precision_agg):
        print("Warning: Cannot plot - need both high and low precision groups.")
        return
    
    hp_feat = high_precision_agg['feature_statistics']
    lp_feat = low_precision_agg['feature_statistics']
    
    # Define features to plot
    feature_configs = [
        ('l2_norm', 'L2 Norm', 'log'),
        ('l1_norm', 'L1 Norm', 'log'),
        ('max_abs', 'Max Absolute Value', 'log'),
        ('mean_abs', 'Mean Absolute Value', 'log'),
        ('std', 'Standard Deviation', 'log'),
        ('range', 'Range (max - min)', 'log'),
        ('non_zero_ratio', 'Non-zero Ratio', 'linear'),
        ('positive_ratio', 'Positive Ratio', 'linear'),
        ('negative_ratio', 'Negative Ratio', 'linear'),
    ]
    
    # Create figure with subplots
    num_features = len(feature_configs)
    cols = 3
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    hp_names_str = ', '.join(high_precision_agg['precision_names'])
    lp_names_str = ', '.join(low_precision_agg['precision_names'])
    
    for idx, (feat_key, feat_display, scale_type) in enumerate(feature_configs):
        ax = axes[idx]
        
        hp_mean = hp_feat[feat_key]['mean']
        lp_mean = lp_feat[feat_key]['mean']
        hp_std = hp_feat[feat_key]['std']
        lp_std = lp_feat[feat_key]['std']
        
        # Create bar plot
        x_pos = np.arange(2)
        bars = ax.bar(x_pos, [hp_mean, lp_mean], 
                     yerr=[hp_std, lp_std],
                     capsize=5, width=0.6,
                     color=['#e66d50', '#8ab07c'], alpha=0.8)
        
        # Set labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['High Precision\n(' + hp_names_str + ')', 
                            'Low Precision\n(' + lp_names_str + ')'],
                           fontsize=12)
        ax.set_ylabel(feat_display, fontsize=14, fontweight='bold')
        ax.set_title(feat_display, fontsize=16, fontweight='bold')
        
        # Set scale
        if scale_type == 'log':
            ax.set_yscale('log')
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, [hp_mean, lp_mean], [hp_std, lp_std])):
            height = bar.get_height()
            if scale_type == 'log':
                label = f'{mean_val:.2e}\n±{std_val:.2e}'
            else:
                label = f'{mean_val:.4f}\n±{std_val:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label, ha='center', va='bottom', fontsize=10)
        
        # Add ratio annotation
        if abs(lp_mean) > 1e-10:
            ratio = hp_mean / lp_mean if lp_mean != 0 else float('inf')
            if abs(ratio) < 1000 and abs(ratio) > 0.001:
                ax.text(0.5, 0.95, f'Ratio: {ratio:.2f}x',
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(num_features, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    matrix_info = f"Matrix: {matrix_name}" if matrix_name else f"Random {matrix_size}x{matrix_size}" if matrix_size else "Unknown"
    fig.suptitle(f'Precision Selection vs Sub-p Features\n{matrix_info}', 
                fontsize=20, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-feature analysis plot saved to: {save_path}")
    plt.close(fig)
    
    # Create iteration distribution plot
    fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
    
    hp_iter_mean = high_precision_agg['iteration_statistics']['mean']
    lp_iter_mean = low_precision_agg['iteration_statistics']['mean']
    hp_iter_std = high_precision_agg['iteration_statistics']['std']
    lp_iter_std = low_precision_agg['iteration_statistics']['std']
    
    x_pos = np.arange(2)
    bars = ax_iter.bar(x_pos, [hp_iter_mean, lp_iter_mean],
                      yerr=[hp_iter_std, lp_iter_std],
                      capsize=10, width=0.6,
                      color=['#e66d50', '#8ab07c'], alpha=0.8)
    
    ax_iter.set_xticks(x_pos)
    ax_iter.set_xticklabels(['High Precision\n(' + hp_names_str + ')', 
                            'Low Precision\n(' + lp_names_str + ')'],
                           fontsize=14)
    ax_iter.set_ylabel('Mean Iteration Number', fontsize=16, fontweight='bold')
    ax_iter.set_title(f'Iteration Distribution Analysis\n{matrix_info}', 
                     fontsize=18, fontweight='bold')
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, [hp_iter_mean, lp_iter_mean], 
                                      [hp_iter_std, lp_iter_std]):
        height = bar.get_height()
        ax_iter.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean_val:.1f}\n±{std_val:.1f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax_iter.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    iter_plot_path = save_path.replace('.png', '_iteration.png')
    plt.savefig(iter_plot_path, dpi=300, bbox_inches='tight')
    print(f"Iteration distribution plot saved to: {iter_plot_path}")
    plt.close(fig_iter)


def profile_model(model_path: str, matrix_name: Optional[str] = None, 
                  matrix_size: Optional[int] = None,
                  config_path: str = 'config/default.yaml', 
                  random_seed: int = 42,
                  output_path: Optional[str] = None):
    """
    主分析函数：运行一次模型指导的 episode 并分析精度选择规律
    """
    print("=" * 80)
    print("🔍 Sub-p 精度选择特征分析")
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
    
    # 处理配置文件路径：如果是相对路径，转换为相对于项目根目录的路径
    if not os.path.isabs(config_path):
        # 获取项目根目录（脚本在 eval 目录下，需要向上两级）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # 从 eval 目录到项目根目录
        # 如果路径以 ../ 开头，去掉它；否则直接拼接
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
    
    # 加载模型
    print("\n🤖 加载模型权重...")
    cg_agent = load_model_weights(model_path, config, env)
    
    # 创建评估代理
    pfrl_agent = PfrlCompatibleCGPPOAgent(cg_agent)
    pfrl_agent.eval_mode()
    
    # 运行 episode 并记录数据
    print("\n🚀 运行模型指导的 episode...")
    result = run_episode_with_profiling(env, pfrl_agent, seed=random_seed)
    
    print(f"\nEpisode 完成:")
    print(f"  - 迭代次数: {result['iterations']}")
    print(f"  - 总计算成本: {result['compute_cost']:.6f}")
    print(f"  - 最终残差: {result['final_residual']:.6e}")
    print(f"  - 是否收敛: {'是' if result['converged'] else '否'}")
    
    # 计算统计信息
    print("\n📈 计算统计信息...")
    statistics = compute_precision_statistics(result['precision_data'])
    
    # 计算总选择次数
    total_selections = sum(stat['count'] for stat in statistics.values())
    
    # 打印报告
    print_statistics_report(statistics, total_selections)
    
    # 分析规律
    print("\n" + "=" * 80)
    print("🔬 精度选择规律分析")
    print("=" * 80)
    pattern_analysis = analyze_precision_patterns(statistics)
    print(pattern_analysis['patterns'])
    
    # 保存结果
    if output_path is None:
        result_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
        output_path = os.path.join(result_dir, f"profile_result_{int(time.time())}.json")
    
    # 准备保存的数据（移除原始 sub_p 数据以减小文件大小）
    save_data = {
        'model_path': model_path,
        'matrix_name': matrix_name,
        'matrix_size': env.matrix_size,
        'random_seed': random_seed,
        'num_tiles': num_tiles,
        'tilesize': tilesize,
        'episode_info': {
            'iterations': result['iterations'],
            'compute_cost': result['compute_cost'],
            'final_residual': result['final_residual'],
            'converged': result['converged']
        },
        'statistics': statistics,
        'pattern_analysis': pattern_analysis,
        'total_selections': total_selections
    }
    
    # 转换 numpy 类型为 Python 原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    save_data = convert_numpy_types(save_data)
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n📄 详细分析结果已保存至: {output_path}")
    
    # 绘制精度选择和特征关系图
    print("\n" + "=" * 80)
    print("📊 绘制精度选择特征分析图")
    print("=" * 80)
    plot_path = output_path.replace('.json', '_features.png')
    plot_precision_feature_analysis(
        statistics=statistics,
        save_path=plot_path,
        matrix_name=matrix_name,
        matrix_size=env.matrix_size
    )
    
    # 将绘图路径添加到保存的数据中
    save_data['feature_plot_path'] = plot_path
    save_data['iteration_plot_path'] = plot_path.replace('.png', '_iteration.png')
    
    return save_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分析模型指导的 episode 中 sub_p 的精度选择特征',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用指定矩阵分析模型
  python profileP.py --model_path ./log/best_model --matrix_name Muu

  # 使用自定义随机种子
  python profileP.py --model_path ./log/best_model --matrix_name Muu --seed 123

  # 使用指定大小的随机矩阵
  python profileP.py --model_path ./log/best_model --matrix_size 1024
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
    
    # 运行分析
    try:
        profile_model(
            model_path=args.model_path,
            matrix_name=matrix_name,
            matrix_size=matrix_size,
            config_path=args.config_path,
            random_seed=args.seed,
            output_path=args.output
        )
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

