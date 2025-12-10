"""
Training Logger - 训练过程日志记录和统计
"""

import os
import logging
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

    # 导入序列化工具
try:
    from .data_utils import convert_to_serializable
except ImportError:
    # 如果无法导入，定义本地版本（兼容numpy 1.x和2.x）
    def convert_to_serializable(obj: Any) -> Any:
        """将numpy数据类型转换为JSON可序列化的Python原生类型"""
        # 处理numpy标量类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                              np.intc, np.intp, np.uint8, np.uint16,
                              np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj


class TrainingLogger:
    """
    训练日志记录器，支持 CG 环境详细统计
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志
        self.logger = logging.getLogger('cg_training')
        self.logger.setLevel(logging.INFO)

        log_file = os.path.join(log_dir, 'training.log')
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # 训练统计 - 基础数据
        self.episode_stats = []
        self.step_rewards = []

        # CG 特定统计数据
        self.step_stats = []  # 每步的详细统计（从info中获取）
        self.residual_histories = []  # 每个episode的残差历史
        self.tile_action_histories = []  # 每个episode的tile动作历史
        self.performance_stats_history = []  # 性能统计历史

    def log_episode(self, episode: int, stats: Dict, detailed_info: Dict = None):
        """记录 episode 统计"""
        stats['episode'] = episode
        self.episode_stats.append(stats)

        # 记录 CG 特定的详细统计
        if detailed_info:
            if 'residual_history' in detailed_info:
                self.residual_histories.append({
                    'episode': episode,
                    'residual_history': detailed_info['residual_history']
                })

            if 'tile_actions_history' in detailed_info:
                self.tile_action_histories.append({
                    'episode': episode,
                    'tile_actions_history': detailed_info['tile_actions_history']
                })

            if 'performance_stats' in detailed_info:
                self.performance_stats_history.append({
                    'episode': episode,
                    **detailed_info['performance_stats']
                })

        self.logger.info(f"Episode {episode}: {stats}")

    def log_step(self, step: int, reward: float, info: Dict = None):
        """记录步骤奖励和详细统计信息"""
        self.step_rewards.append(reward)

        # 记录详细的步骤统计（如果提供）
        if info:
            step_stat = {
                'step': step,
                'reward': reward,
                **info  # 包含 iteration, iteration_cost, tile_actions 等
            }
            self.step_stats.append(step_stat)

    def save_stats(self):
        """保存统计数据"""
        stats_file = os.path.join(self.log_dir, 'training_stats.json')
        
        # 准备要保存的数据
        data_to_save = {
            'episode_stats': self.episode_stats,
            'step_rewards': self.step_rewards,
            'step_stats': self.step_stats,
            'residual_histories': self.residual_histories,
            'tile_action_histories': self.tile_action_histories,
            'performance_stats_history': self.performance_stats_history
        }
        
        # 转换所有numpy类型为JSON可序列化类型
        serializable_data = convert_to_serializable(data_to_save)
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    def plot_training_curves(self):
        """绘制训练曲线 - 增强版本包含CG特定统计"""
        if not self.episode_stats:
            return

        # 基础统计
        episodes = [s['episode'] for s in self.episode_stats]
        total_costs = [s.get('total_cost', 0) for s in self.episode_stats]
        iterations = [s.get('iterations', 0) for s in self.episode_stats]
        converged = [1 if s.get('converged', False) else 0 for s in self.episode_stats]

        # 计算有内容的图表数量，动态决定布局
        num_plots = 0
        has_reward = False
        has_residual_history = bool(self.residual_histories)
        has_tile_actions = bool(self.tile_action_histories)
        
        # 检查是否有reward数据
        for stat in self.episode_stats:
            if stat.get('mean') is not None or stat.get('total_reward') is not None or stat.get('reward') is not None:
                has_reward = True
                break
        if not has_reward and self.step_stats:
            has_reward = True  # 可以从step_stats计算
        
        # 计算需要的图表数量
        num_plots = 2  # Total Cost, Iterations
        if has_reward:
            num_plots += 1
        num_plots += 1  # Convergence Ratio
        if has_residual_history:
            num_plots += 1
        if has_tile_actions:
            num_plots += 1
        num_plots += 1  # Summary
        
        # 使用3列布局，行数根据需要的图表数量计算
        num_rows = (num_plots + 2) // 3  # 向上取整
        if num_rows == 0:
            num_rows = 1
        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_rows > 1 and axes.ndim == 1:
            axes = axes.reshape(num_rows, -1)
        fig.suptitle('CG Training Analysis Dashboard', fontsize=16, fontweight='bold')
        
        plot_idx = 0
        
        # 1. 总成本曲线
        row, col = plot_idx // 3, plot_idx % 3
        axes[row, col].plot(episodes, total_costs, 'b-', linewidth=2, marker='o', markersize=3)
        axes[row, col].set_title('Total Cost per Episode', fontweight='bold')
        axes[row, col].set_xlabel('Episode')
        axes[row, col].set_ylabel('Total Cost')
        axes[row, col].grid(True, alpha=0.3)
        plot_idx += 1

        # 2. 迭代次数曲线
        row, col = plot_idx // 3, plot_idx % 3
        axes[row, col].plot(episodes, iterations, 'r-', linewidth=2, marker='s', markersize=3)
        axes[row, col].set_title('Iterations per Episode', fontweight='bold')
        axes[row, col].set_xlabel('Episode')
        axes[row, col].set_ylabel('Iterations')
        axes[row, col].grid(True, alpha=0.3)
        plot_idx += 1

        # 3. 奖励曲线（按 episode）
        episode_rewards = []
        episode_indices = []
        
        # 尝试从 episode_stats 中获取 reward
        for stat in self.episode_stats:
            # pfrl 的 eval_stats 通常包含 'mean' 字段表示平均 reward
            # 也可能包含 'total_reward' 或其他 reward 相关字段
            reward = stat.get('mean', stat.get('total_reward', stat.get('reward', None)))
            if reward is not None:
                episode_rewards.append(reward)
                episode_indices.append(stat.get('episode', len(episode_rewards) - 1))
        
        # 如果 episode_stats 中没有 reward，从 step_stats 中按 episode 计算
        if not episode_rewards and self.step_stats:
            # 按 episode 分组计算总 reward
            # 通过 iteration 字段来推断 episode 边界（每个 episode 的 iteration 从 0 或 1 开始）
            current_episode_reward = 0
            current_episode_idx = 0
            last_iteration = None
            
            for step_stat in self.step_stats:
                iteration = step_stat.get('iteration', 0)
                reward = step_stat.get('reward', 0)
                
                # 如果 iteration 重置（变小或为 0/1），说明开始了新的 episode
                if last_iteration is not None and iteration <= last_iteration:
                    if current_episode_reward != 0:
                        episode_rewards.append(current_episode_reward)
                        episode_indices.append(current_episode_idx)
                        current_episode_idx += 1
                    current_episode_reward = reward
                else:
                    current_episode_reward += reward
                
                last_iteration = iteration
            
            # 添加最后一个 episode 的 reward
            if current_episode_reward != 0:
                episode_rewards.append(current_episode_reward)
                episode_indices.append(current_episode_idx)
        
        if episode_rewards:
            row, col = plot_idx // 3, plot_idx % 3
            # 如果没有 episode 索引，使用默认的连续索引
            if not episode_indices:
                episode_indices = list(range(len(episode_rewards)))
            
            # 绘制 episode reward 曲线
            axes[row, col].plot(episode_indices, episode_rewards, 'purple', linewidth=2, marker='o', markersize=3)
            
            # 可选：添加平滑曲线
            if len(episode_rewards) > 10:
                window_size = min(10, len(episode_rewards) // 5)
                rewards_smooth = []
                smooth_indices = []
                for i in range(window_size, len(episode_rewards) + 1):
                    rewards_smooth.append(sum(episode_rewards[i-window_size:i]) / window_size)
                    # 使用对应位置的 episode 索引
                    smooth_indices.append(episode_indices[i-1])
                if smooth_indices:
                    axes[row, col].plot(smooth_indices, rewards_smooth,
                                       'orange', linewidth=2, linestyle='--', label=f'{window_size}-episode avg')
                    axes[row, col].legend()
            
            axes[row, col].set_title('Reward per Episode', fontweight='bold')
            axes[row, col].set_xlabel('Episode')
            axes[row, col].set_ylabel('Reward')
            axes[row, col].grid(True, alpha=0.3)
            plot_idx += 1

        # 4. 收敛比率
        row, col = plot_idx // 3, plot_idx % 3
        conv_ratios = [s.get('convergence_ratio', 0) for s in self.episode_stats]
        axes[row, col].plot(episodes, conv_ratios, 'brown', linewidth=2, marker='*', markersize=4)
        axes[row, col].set_title('Convergence Ratio per Episode', fontweight='bold')
        axes[row, col].set_xlabel('Episode')
        axes[row, col].set_ylabel('Convergence Ratio')
        axes[row, col].set_yscale('log')
        axes[row, col].grid(True, alpha=0.3)
        plot_idx += 1

        # 5. 残差历史示例（最近几个episode）
        if self.residual_histories:
            row, col = plot_idx // 3, plot_idx % 3
            axes[row, col].set_title('Residual History (Recent Episodes)', fontweight='bold')
            axes[row, col].set_xlabel('Iteration')
            axes[row, col].set_ylabel('Residual Norm (log scale)')
            axes[row, col].set_yscale('log')

            # 显示最近5个episode的残差历史
            recent_histories = self.residual_histories[-5:] if len(self.residual_histories) >= 5 else self.residual_histories
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, hist in enumerate(recent_histories):
                residual_history = hist.get('residual_history', [])
                if residual_history:
                    iters = range(len(residual_history))
                    axes[row, col].plot(iters, residual_history,
                                       color=colors[i % len(colors)],
                                       linewidth=1.5,
                                       label=f'Episode {hist["episode"]}',
                                       alpha=0.8)
            axes[row, col].legend(fontsize='small')
            axes[row, col].grid(True, alpha=0.3)
            plot_idx += 1

        # 6. Tile 动作分布（如果有tile动作历史）
        if self.tile_action_histories:
            row, col = plot_idx // 3, plot_idx % 3
            axes[row, col].set_title('Tile Action Distribution', fontweight='bold')
            axes[row, col].set_xlabel('Precision Level')
            axes[row, col].set_ylabel('Frequency')

            # 精度名称映射
            precision_names = ['fp64', 'fp32', 'fp16', 'fp8']

            # 收集所有动作
            all_actions = []
            for hist in self.tile_action_histories[-10:]:  # 最近10个episode
                actions_history = hist.get('tile_actions_history', [])
                for actions in actions_history:
                    all_actions.extend(actions)

            if all_actions:
                # 统计每种精度的使用频率
                unique_actions, counts = np.unique(all_actions, return_counts=True)
                axes[row, col].bar(unique_actions, counts, color='skyblue', alpha=0.7, edgecolor='black')
                
                # 设置 x 轴刻度位置和标签
                max_action = max(unique_actions) if len(unique_actions) > 0 else 0
                xticks = range(max_action + 1)
                xticklabels = [precision_names[i] if i < len(precision_names) else f'Action {i}' 
                              for i in xticks]
                axes[row, col].set_xticks(xticks)
                axes[row, col].set_xticklabels(xticklabels, rotation=45, ha='right')
                axes[row, col].grid(True, alpha=0.3, axis='y')
            plot_idx += 1

        # 7. 总结统计
        row, col = plot_idx // 3, plot_idx % 3
        axes[row, col].axis('off')
        summary_text = f"""
        Training Summary:

        Total Episodes: {len(episodes)}
        Converged Episodes: {sum(converged)}
        Convergence Rate: {sum(converged)/len(converged)*100:.1f}%

        Avg Iterations: {np.mean(iterations):.1f}
        Avg Total Cost: {np.mean(total_costs):.2f}

        Best Performance:
        Min Iterations: {min(iterations) if iterations else 'N/A'}
        Min Cost: {min(total_costs):.3f}
        """
        axes[row, col].text(0.05, 0.95, summary_text, transform=axes[row, col].transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(plot_idx + 1, num_rows * 3):
            row, col = i // 3, i % 3
            if row < num_rows:
                axes[row, col].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'cg_training_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saving comprehensive CG training analysis to {plot_path}")
        plt.close()

        # 额外创建残差收敛详细图
        self._plot_residual_convergence_details()
        # 性能分析图已移除（不需要 time distribution 和 cache performance）

    def _plot_residual_convergence_details(self):
        """绘制残差收敛详细分析图"""
        if not self.residual_histories:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Residual Convergence Analysis', fontsize=14, fontweight='bold')

        # 1. 多个episode的残差收敛对比
        axes[0, 0].set_title('Residual Convergence Across Episodes', fontweight='bold')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Residual Norm (log scale)')
        axes[0, 0].set_yscale('log')

        # 显示不同阶段的episode
        total_episodes = len(self.residual_histories)
        if total_episodes > 0:
            # 早期episode
            if total_episodes >= 3:
                early_idx = min(2, total_episodes - 1)
                early_hist = self.residual_histories[early_idx]
                residual_history = early_hist.get('residual_history', [])
                if residual_history:
                    axes[0, 0].plot(range(len(residual_history)), residual_history,
                                   'red', linewidth=2, label=f'Early (Ep {early_hist["episode"]})', alpha=0.8)

            # 中期episode
            mid_idx = total_episodes // 2
            if mid_idx < total_episodes:
                mid_hist = self.residual_histories[mid_idx]
                residual_history = mid_hist.get('residual_history', [])
                if residual_history:
                    axes[0, 0].plot(range(len(residual_history)), residual_history,
                                   'orange', linewidth=2, label=f'Mid (Ep {mid_hist["episode"]})', alpha=0.8)

            # 晚期episode
            late_hist = self.residual_histories[-1]
            residual_history = late_hist.get('residual_history', [])
            if residual_history:
                axes[0, 0].plot(range(len(residual_history)), residual_history,
                               'green', linewidth=2, label=f'Late (Ep {late_hist["episode"]})', alpha=0.8)

        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 收敛速度分布
        axes[0, 1].set_title('Convergence Speed Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Convergence Speed (log scale)')
        axes[0, 1].set_yscale('log')

        conv_speeds = []
        conv_episodes = []
        for hist in self.residual_histories:
            residual_history = hist.get('residual_history', [])
            if len(residual_history) > 1:
                # 计算收敛速度：初始残差/最终残差
                speed = residual_history[0] / residual_history[-1] if residual_history[-1] > 0 else 0
                if speed > 1:  # 只记录有意义的收敛
                    conv_speeds.append(speed)
                    conv_episodes.append(hist['episode'])

        if conv_speeds:
            axes[0, 1].plot(conv_episodes, conv_speeds, 'blue', linewidth=2, marker='o', markersize=4)
            axes[0, 1].axhline(y=np.mean(conv_speeds), color='red', linestyle='--',
                              label=f'Mean: {np.mean(conv_speeds):.2f}')
            axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 残差减少模式分析
        axes[1, 0].set_title('Residual Reduction Pattern', fontweight='bold')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Residual Reduction Rate')

        if self.residual_histories:
            # 使用最新的几个episode分析残差减少模式
            recent_histories = self.residual_histories[-5:] if len(self.residual_histories) >= 5 else self.residual_histories

            for hist in recent_histories:
                residual_history = hist.get('residual_history', [])
                if len(residual_history) > 2:
                    # 计算残差减少率
                    reductions = []
                    for i in range(1, len(residual_history)):
                        if residual_history[i-1] > 0:
                            reduction = (residual_history[i-1] - residual_history[i]) / residual_history[i-1]
                            reductions.append(reduction)

                    if reductions:
                        axes[1, 0].plot(range(1, len(reductions)+1), reductions,
                                       linewidth=1.5, alpha=0.7, label=f'Ep {hist["episode"]}')

        axes[1, 0].legend(fontsize='small')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 收敛时间分布
        axes[1, 1].set_title('Time to Convergence Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Iterations to Converge')
        axes[1, 1].set_ylabel('Frequency')

        convergence_iters = []
        for stat in self.episode_stats:
            if stat.get('converged', False):
                iters = stat.get('iterations', 0)
                if iters > 0:
                    convergence_iters.append(iters)

        if convergence_iters:
            axes[1, 1].hist(convergence_iters, bins=min(20, len(set(convergence_iters))),
                           alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].axvline(np.mean(convergence_iters), color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(convergence_iters):.1f}')
            axes[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'residual_convergence_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saving residual convergence analysis to {plot_path}")
        plt.close()

    def _plot_performance_analysis(self):
        """绘制性能分析图"""
        if not self.performance_stats_history:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Performance Analysis Dashboard', fontsize=14, fontweight='bold')

        episodes = [s['episode'] for s in self.performance_stats_history]

        # 1. 时间分布趋势
        times_spmv = [s.get('spmv_time', 0) for s in self.performance_stats_history]
        times_cg_math = [s.get('cg_math_time', 0) for s in self.performance_stats_history]
        times_matrix_ext = [s.get('matrix_extraction_time', 0) for s in self.performance_stats_history]

        axes[0, 0].stackplot(episodes, times_spmv, times_cg_math, times_matrix_ext,
                           labels=['SpMV Time', 'CG Math Time', 'Matrix Extraction'],
                           alpha=0.8, colors=['lightblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Time Distribution Over Episodes', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].legend(loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 缓存效率
        cache_hits = [s.get('cache_hits', 0) for s in self.performance_stats_history]
        cache_misses = [s.get('cache_misses', 0) for s in self.performance_stats_history]

        axes[0, 1].plot(episodes, cache_hits, 'green', linewidth=2, label='Cache Hits', marker='o')
        axes[0, 1].plot(episodes, cache_misses, 'red', linewidth=2, label='Cache Misses', marker='s')

        # 添加缓存命中率
        cache_total = [h + m for h, m in zip(cache_hits, cache_misses)]
        cache_hit_rate = [h / t * 100 if t > 0 else 0 for h, t in zip(cache_hits, cache_total)]

        ax2 = axes[0, 1].twinx()
        ax2.plot(episodes, cache_hit_rate, 'blue', linewidth=2, linestyle='--', label='Hit Rate (%)')
        ax2.set_ylabel('Cache Hit Rate (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        axes[0, 1].set_title('Cache Performance', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cache Operations')
        axes[0, 1].legend(loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 迭代计数和总时间
        iter_counts = [s.get('iteration_count', 0) for s in self.performance_stats_history]
        total_times = [s.get('total_time', 0) for s in self.performance_stats_history]

        axes[0, 2].plot(episodes, iter_counts, 'purple', linewidth=2, label='Iteration Count', marker='d')
        ax3 = axes[0, 2].twinx()
        ax3.plot(episodes, total_times, 'orange', linewidth=2, linestyle='--', label='Total Time')
        ax3.set_ylabel('Total Time (s)', color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')

        axes[0, 2].set_title('Iteration Count vs Total Time', fontweight='bold')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Iteration Count', color='purple')
        axes[0, 2].tick_params(axis='y', labelcolor='purple')

        # 合并图例
        lines1, labels1 = axes[0, 2].get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        axes[0, 2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 时间效率分析（每迭代的时间）
        if iter_counts and total_times:
            time_per_iter = [t / max(i, 1) for t, i in zip(total_times, iter_counts)]
            axes[1, 0].plot(episodes, time_per_iter, 'darkblue', linewidth=2, marker='h')
            axes[1, 0].set_title('Time per Iteration', fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Time per Iteration (seconds)')
            axes[1, 0].grid(True, alpha=0.3)

        # 5. SpMV vs CG Math 时间比例
        spmv_ratios = []
        for s in self.performance_stats_history:
            total_comp_time = s.get('spmv_time', 0) + s.get('cg_math_time', 0)
            if total_comp_time > 0:
                spmv_ratio = s.get('spmv_time', 0) / total_comp_time * 100
                spmv_ratios.append(spmv_ratio)

        if spmv_ratios:
            axes[1, 1].plot(episodes[:len(spmv_ratios)], spmv_ratios, 'red', linewidth=2, marker='*')
            axes[1, 1].set_title('SpMV Time Ratio in Computation', fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('SpMV Time Ratio (%)')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(True, alpha=0.3)

        # 6. 性能总结统计
        axes[1, 2].axis('off')
        if self.performance_stats_history:
            latest = self.performance_stats_history[-1]

            # 计算平均性能
            avg_spmv = np.mean([s.get('spmv_time', 0) for s in self.performance_stats_history])
            avg_cg_math = np.mean([s.get('cg_math_time', 0) for s in self.performance_stats_history])
            avg_total = np.mean([s.get('total_time', 0) for s in self.performance_stats_history])
            avg_iters = np.mean([s.get('iteration_count', 0) for s in self.performance_stats_history])

            summary_text = f"""
            Performance Summary:

            Latest Episode:
            • Total Time: {latest.get('total_time', 0):.3f}s
            • SpMV Time: {latest.get('spmv_time', 0):.3f}s
            • CG Math Time: {latest.get('cg_math_time', 0):.3f}s
            • Iterations: {latest.get('iteration_count', 0)}

            Training Averages:
            • Avg Total Time: {avg_total:.3f}s
            • Avg SpMV Time: {avg_spmv:.3f}s
            • Avg CG Math: {avg_cg_math:.3f}s
            • Avg Iterations: {avg_iters:.1f}

            Cache Performance:
            • Latest Hit Rate: {(latest.get('cache_hits', 0) / max(latest.get('cache_hits', 0) + latest.get('cache_misses', 0), 1) * 100):.1f}%
            """
            axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'performance_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saving performance analysis to {plot_path}")
        plt.close()
