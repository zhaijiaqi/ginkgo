"""
Training Logger - 训练过程日志记录和统计
"""

import os
import logging
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt


class TrainingLogger:
    """
    训练日志记录器
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

        # 训练统计
        self.episode_stats = []
        self.step_rewards = []

    def log_episode(self, episode: int, stats: Dict):
        """记录 episode 统计"""
        stats['episode'] = episode
        self.episode_stats.append(stats)

        self.logger.info(f"Episode {episode}: {stats}")

    def log_step(self, step: int, reward: float):
        """记录步骤奖励"""
        self.step_rewards.append(reward)

    def save_stats(self):
        """保存统计数据"""
        stats_file = os.path.join(self.log_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump({
                'episode_stats': self.episode_stats,
                'step_rewards': self.step_rewards
            }, f, indent=2)

    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.episode_stats:
            return

        episodes = [s['episode'] for s in self.episode_stats]
        costs = [s.get('total_cost', 0) for s in self.episode_stats]
        errors = [s.get('total_error', 0) for s in self.episode_stats]
        converged = [1 if s.get('converged', False) else 0 for s in self.episode_stats]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 成本曲线
        axes[0, 0].plot(episodes, costs)
        axes[0, 0].set_title('Total Cost per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cost')

        # 误差曲线
        axes[0, 1].plot(episodes, errors)
        axes[0, 1].set_title('Total Error per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Error')

        # 收敛率
        if len(converged) > 10:
            window_size = 10
            conv_rate = []
            for i in range(window_size, len(converged) + 1):
                conv_rate.append(sum(converged[i-window_size:i]) / window_size)
            axes[1, 0].plot(range(window_size, len(episodes) + 1), conv_rate)
        axes[1, 0].set_title('Convergence Rate (10-episode window)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Rate')

        # 奖励曲线
        if self.step_rewards:
            window_size = 100
            if len(self.step_rewards) > window_size:
                rewards_smooth = []
                for i in range(window_size, len(self.step_rewards) + 1):
                    rewards_smooth.append(sum(self.step_rewards[i-window_size:i]) / window_size)
                axes[1, 1].plot(range(window_size, len(self.step_rewards) + 1), rewards_smooth)
        axes[1, 1].set_title('Average Reward (100-step window)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Reward')

        plt.tight_layout()
        print(f"saving training curves to {os.path.join(self.log_dir, 'training_curves.png')}")
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()
