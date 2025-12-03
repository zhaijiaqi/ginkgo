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

        # 创建主图 - 4x4布局
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('CG Training Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. 总成本曲线
        axes[0, 0].plot(episodes, total_costs, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_title('Total Cost per Episode', fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Cost')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 迭代次数曲线
        axes[0, 1].plot(episodes, iterations, 'r-', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_title('Iterations per Episode', fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Iterations')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 收敛率
        if len(converged) > 10:
            window_size = 10
            conv_rate = []
            for i in range(window_size, len(converged) + 1):
                conv_rate.append(sum(converged[i-window_size:i]) / window_size)
            axes[0, 2].plot(range(window_size, len(episodes) + 1), conv_rate,
                           'g-', linewidth=2, marker='^', markersize=3)
        axes[0, 2].set_title('Convergence Rate (10-episode window)', fontweight='bold')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Convergence Rate')
        axes[0, 2].set_ylim(0, 1.1)
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 奖励曲线
        if self.step_rewards:
            window_size = 100
            if len(self.step_rewards) > window_size:
                rewards_smooth = []
                for i in range(window_size, len(self.step_rewards) + 1):
                    rewards_smooth.append(sum(self.step_rewards[i-window_size:i]) / window_size)
                axes[0, 3].plot(range(window_size, len(self.step_rewards) + 1), rewards_smooth,
                               'purple', linewidth=2)
        axes[0, 3].set_title('Average Reward (100-step window)', fontweight='bold')
        axes[0, 3].set_xlabel('Step')
        axes[0, 3].set_ylabel('Average Reward')
        axes[0, 3].grid(True, alpha=0.3)

        # 5. 平均tile成本
        avg_tile_costs = [s.get('avg_tile_cost', 0) for s in self.episode_stats]
        axes[1, 0].plot(episodes, avg_tile_costs, 'orange', linewidth=2, marker='d', markersize=3)
        axes[1, 0].set_title('Average Tile Cost per Episode', fontweight='bold')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Avg Tile Cost')
        axes[1, 0].grid(True, alpha=0.3)

        # 6. 收敛比率
        conv_ratios = [s.get('convergence_ratio', 0) for s in self.episode_stats]
        axes[1, 1].plot(episodes, conv_ratios, 'brown', linewidth=2, marker='*', markersize=4)
        axes[1, 1].set_title('Convergence Ratio per Episode', fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Convergence Ratio')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        # 7. 残差历史示例（最近几个episode）
        if self.residual_histories:
            axes[1, 2].set_title('Residual History (Recent Episodes)', fontweight='bold')
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Residual Norm (log scale)')
            axes[1, 2].set_yscale('log')

            # 显示最近5个episode的残差历史
            recent_histories = self.residual_histories[-5:] if len(self.residual_histories) >= 5 else self.residual_histories
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, hist in enumerate(recent_histories):
                residual_history = hist.get('residual_history', [])
                if residual_history:
                    iters = range(len(residual_history))
                    axes[1, 2].plot(iters, residual_history,
                                   color=colors[i % len(colors)],
                                   linewidth=1.5,
                                   label=f'Episode {hist["episode"]}',
                                   alpha=0.8)
            axes[1, 2].legend(fontsize='small')
            axes[1, 2].grid(True, alpha=0.3)

        # 8. 性能统计 - 时间分布
        if self.performance_stats_history:
            latest_perf = self.performance_stats_history[-1]
            spmv_time = max(0, latest_perf.get('spmv_time', 0) or 0)
            cg_math_time = max(0, latest_perf.get('cg_math_time', 0) or 0)
            matrix_ext_time = max(0, latest_perf.get('matrix_extraction_time', 0) or 0)
            total_time = max(0, latest_perf.get('total_time', 0) or 0)
            
            other_time = max(0, total_time - spmv_time - cg_math_time - matrix_ext_time)
            
            times = [spmv_time, cg_math_time, matrix_ext_time, other_time]
            labels = ['SpMV', 'CG Math', 'Matrix Ext', 'Other']
            
            # 过滤掉0值和NaN值
            filtered_times = []
            filtered_labels = []
            for t, l in zip(times, labels):
                if t > 0 and not (isinstance(t, float) and (np.isnan(t) or np.isinf(t))):
                    filtered_times.append(t)
                    filtered_labels.append(l)
            
            if filtered_times and sum(filtered_times) > 0:
                axes[1, 3].pie(filtered_times, labels=filtered_labels, autopct='%1.1f%%', startangle=90)
                axes[1, 3].set_title('Time Distribution (Latest Episode)', fontweight='bold')
            else:
                axes[1, 3].text(0.5, 0.5, 'No time data', ha='center', va='center', transform=axes[1, 3].transAxes)
                axes[1, 3].set_title('Time Distribution (Latest Episode)', fontweight='bold')

        # 9. 缓存性能
        if self.performance_stats_history:
            cache_hits = [s.get('cache_hits', 0) for s in self.performance_stats_history]
            cache_misses = [s.get('cache_misses', 0) for s in self.performance_stats_history]
            episodes_perf = [s['episode'] for s in self.performance_stats_history]

            axes[2, 0].plot(episodes_perf, cache_hits, 'cyan', linewidth=2, label='Cache Hits', marker='o')
            axes[2, 0].plot(episodes_perf, cache_misses, 'magenta', linewidth=2, label='Cache Misses', marker='s')
            axes[2, 0].set_title('Cache Performance', fontweight='bold')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Count')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

        # 10. Tile 动作分布（如果有tile动作历史）
        if self.tile_action_histories:
            axes[2, 1].set_title('Tile Action Distribution', fontweight='bold')
            axes[2, 1].set_xlabel('Precision Level')
            axes[2, 1].set_ylabel('Frequency')

            # 收集所有动作
            all_actions = []
            for hist in self.tile_action_histories[-10:]:  # 最近10个episode
                actions_history = hist.get('tile_actions_history', [])
                for actions in actions_history:
                    all_actions.extend(actions)

            if all_actions:
                # 统计每种精度的使用频率
                unique_actions, counts = np.unique(all_actions, return_counts=True)
                axes[2, 1].bar(unique_actions, counts, color='skyblue', alpha=0.7, edgecolor='black')
                axes[2, 1].set_xticks(range(max(unique_actions)+1))
                axes[2, 1].grid(True, alpha=0.3, axis='y')

        # 11. 每步成本趋势
        if self.step_stats:
            steps = [s['step'] for s in self.step_stats]
            iteration_costs = [s.get('iteration_cost', 0) for s in self.step_stats]

            if len(steps) > 50:  # 只显示最近的点以避免拥挤
                recent_idx = len(steps) - 50
                steps = steps[recent_idx:]
                iteration_costs = iteration_costs[recent_idx:]

            axes[2, 2].scatter(steps, iteration_costs, alpha=0.6, color='coral', s=20)
            axes[2, 2].set_title('Iteration Cost per Step', fontweight='bold')
            axes[2, 2].set_xlabel('Training Step')
            axes[2, 2].set_ylabel('Iteration Cost')
            axes[2, 2].grid(True, alpha=0.3)

        # 12. 残差收敛趋势
        if self.step_stats:
            steps = [s['step'] for s in self.step_stats]
            residual_norms = [s.get('residual_norm', 0) for s in self.step_stats]

            if len(steps) > 50:  # 只显示最近的点
                recent_idx = len(steps) - 50
                steps = steps[recent_idx:]
                residual_norms = residual_norms[recent_idx:]

            axes[2, 3].scatter(steps, residual_norms, alpha=0.6, color='darkgreen', s=20)
            axes[2, 3].set_title('Residual Norm per Step', fontweight='bold')
            axes[2, 3].set_xlabel('Training Step')
            axes[2, 3].set_ylabel('Residual Norm')
            axes[2, 3].set_yscale('log')
            axes[2, 3].grid(True, alpha=0.3)

        # 13. 初始vs最终残差
        initial_residuals = [s.get('initial_residual', 0) for s in self.episode_stats if s.get('initial_residual', 0) > 0]
        final_residuals = [s.get('final_residual', 0) for s in self.episode_stats if s.get('final_residual', 0) > 0]

        if initial_residuals and final_residuals:
            min_len = min(len(initial_residuals), len(final_residuals))
            initial_residuals = initial_residuals[:min_len]
            final_residuals = final_residuals[:min_len]
            episodes_res = episodes[:min_len]

            axes[3, 0].plot(episodes_res, initial_residuals, 'red', linewidth=2, label='Initial', marker='o')
            axes[3, 0].plot(episodes_res, final_residuals, 'blue', linewidth=2, label='Final', marker='s')
            axes[3, 0].set_title('Initial vs Final Residual', fontweight='bold')
            axes[3, 0].set_xlabel('Episode')
            axes[3, 0].set_ylabel('Residual Norm')
            axes[3, 0].set_yscale('log')
            axes[3, 0].legend()
            axes[3, 0].grid(True, alpha=0.3)

        # 14. 训练效率指标
        if len(episodes) > 1:
            # 计算收敛速度（每episode减少的残差）
            conv_speeds = []
            for i in range(1, len(episodes)):
                if (self.episode_stats[i].get('initial_residual', 0) > 0 and
                    self.episode_stats[i].get('final_residual', 0) > 0):
                    speed = (self.episode_stats[i]['initial_residual'] /
                           self.episode_stats[i]['final_residual'])
                    conv_speeds.append(speed)

            if conv_speeds:
                axes[3, 1].plot(episodes[1:len(conv_speeds)+1], conv_speeds,
                               'darkblue', linewidth=2, marker='d')
                axes[3, 1].set_title('Convergence Speed', fontweight='bold')
                axes[3, 1].set_xlabel('Episode')
                axes[3, 1].set_ylabel('Initial/Final Residual Ratio')
                axes[3, 1].set_yscale('log')
                axes[3, 1].grid(True, alpha=0.3)

        # 15. 成本效率分析
        if len(total_costs) > 0 and len(iterations) > 0:
            cost_per_iter = [c / max(i, 1) for c, i in zip(total_costs, iterations)]
            axes[3, 2].plot(episodes, cost_per_iter, 'purple', linewidth=2, marker='h')
            axes[3, 2].set_title('Cost per Iteration', fontweight='bold')
            axes[3, 2].set_xlabel('Episode')
            axes[3, 2].set_ylabel('Cost/Iteration')
            axes[3, 2].grid(True, alpha=0.3)

        # 16. 总结统计
        axes[3, 3].axis('off')
        summary_text = f"""
        Training Summary:

        Total Episodes: {len(episodes)}
        Converged Episodes: {sum(converged)}
        Convergence Rate: {sum(converged)/len(converged)*100:.1f}%

        Avg Iterations: {np.mean(iterations):.1f}
        Avg Total Cost: {np.mean(total_costs):.2f}
        Avg Tile Cost: {np.mean(avg_tile_costs):.3f}

        Best Performance:
        Min Iterations: {min(iterations) if iterations else 'N/A'}
        Min Cost: {min(total_costs):.3f}
        """
        axes[3, 3].text(0.05, 0.95, summary_text, transform=axes[3, 3].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'cg_training_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saving comprehensive CG training analysis to {plot_path}")
        plt.close()

        # 额外创建残差收敛详细图
        self._plot_residual_convergence_details()
        # 额外创建性能分析图
        self._plot_performance_analysis()

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
