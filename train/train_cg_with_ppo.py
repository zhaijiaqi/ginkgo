"""
训练脚本：使用 PPO 训练 CG 混合精度控制代理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pfrl


# 导入项目模块
from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import PPOAgentFactory, CGPPOAgent

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
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()


def train_cg_ppo(config: Dict):
    """
    训练 CG PPO 代理的主函数 - 使用 pfrl.experiments.train_agent_with_evaluation

    Args:
        config: 训练配置
    """
    print("=== 开始 CG PPO 训练 (使用 pfrl.experiments.train_agent_with_evaluation) ===")

    # 创建环境配置
    env_config = {
        'max_iter': config.get('cg', {}).get('max_iter', 100),
        'stop_tol': config.get('cg', {}).get('stop_tol', 1e-10),
        'matrix_size': config.get('cg', {}).get('matrix_size', 1024),  # 当使用真实矩阵时会被覆盖
        'matrix_name': config.get('cg', {}).get('matrix_name', 'Muu'),  # 矩阵名称
        'matrix_data_dir': config.get('cg', {}).get('matrix_data_dir', '~/data/matrix'),
        'matrix_set_csv': config.get('cg', {}).get('matrix_set_csv', 'matrix_set.csv'),
        'tilesize': config.get('spmv', {}).get('tilesize', 32),
        'precision_cost_table': config.get('spmv', {}).get('precision_cost_table', {
            'fp64': 1.0, 'fp32': 0.7, 'tf32': 0.55,
            'fp16': 0.35, 'bf16': 0.33, 'fp8': 0.15
        }),
        'reward': config.get('reward'),
        'normalize_state': config.get('env', {}).get('normalize_state', True)
    }
    env = CGEnvironment(env_config)
    state_dim = env.get_state_dim()
    action_size = env.get_action_space_size()

    print(f"环境状态维度: {state_dim}")
    print(f"动作空间大小: {action_size}")

    # 创建 CG PPO 代理
    cg_agent = PPOAgentFactory(config).create_agent(state_dim, action_size)

    # 使用适配器包装为 pfrl 兼容的代理
    from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent
    agent = PfrlCompatibleCGPPOAgent(cg_agent)

    # 创建日志记录器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    matrix_name = env_config.get('matrix_name', 'Muu')
    log_dir = os.path.join('log', f'{matrix_name}_{timestamp}')
    logger = TrainingLogger(log_dir)

    # 训练参数
    train_config = config.get('train', {})
    total_steps = train_config.get('total_steps', 10000)
    eval_interval = train_config.get('eval_interval', 50)
    save_interval = train_config.get('save_interval', 50)
    log_interval = train_config.get('log_interval', 10)

    print(f"总训练步数: {total_steps}")
    print(f"评估间隔: {eval_interval}")
    print(f"保存间隔: {save_interval}")

    # 自定义评估钩子，用于记录详细的训练统计
    class TrainingStatsHook:
        def __init__(self, training_logger, env, eval_interval_steps):
            self.training_logger = training_logger
            self.env = env
            self.eval_interval_steps = eval_interval_steps
            self.step_count = 0
            self.episode_count = 0
            self.support_train_agent = True  # pfrl 需要的属性

        def __call__(self, env, agent, step):
            self.step_count += 1

            # 记录步骤奖励（需要从环境中获取）
            if hasattr(env, 'last_reward'):
                self.training_logger.log_step(self.step_count, env.last_reward)

            # 定期日志
            if self.step_count % log_interval == 0:
                print(f"Step {self.step_count}: training in progress...")

    # 创建训练统计钩子
    training_hook = TrainingStatsHook(logger, env, eval_interval)

    # 自定义评估钩子，用于记录episode信息
    class EvalHook:
        def __init__(self, training_hook):
            self.training_hook = training_hook
            self.support_train_agent = True  # pfrl 需要的属性

        def __call__(self, env, agent, step, eval_stats, **kwargs):
            """评估钩子 - 记录episode统计"""
            print(f"评估结果 (Step {step}): {eval_stats}")

    eval_hook = EvalHook(training_hook)

    # 使用 pfrl.experiments.train_agent_with_evaluation
    try:
        trained_agent, eval_stats_history = pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=total_steps,
            eval_n_steps=None,  # 不限制每次评估的步数
            eval_n_episodes=1,  # 每次评估运行1个episode
            eval_interval=eval_interval,
            outdir=log_dir,
            checkpoint_freq=save_interval,  # 定期保存检查点
            step_hooks=[training_hook],  # 步骤钩子
            evaluation_hooks=[eval_hook],  # 评估钩子
            save_best_so_far_agent=True,  # 保存最佳代理
            use_tensorboard=False,  # 不使用tensorboard
            logger=logger.logger  # 使用我们的日志记录器
        )

        print("pfrl 训练完成!")

    except Exception as e:
        import traceback
        print(f"pfrl 训练过程中出现错误: {e}")
        print("完整的错误追踪:")
        traceback.print_exc()
        return

    # 保存最终统计和模型
    logger.save_stats()
    logger.plot_training_curves()

    final_model_path = os.path.join(log_dir, 'final_model.pt')
    agent.save(final_model_path)

    print(f"训练完成! 最终模型保存至: {final_model_path}")
    print(f"训练日志保存至: {log_dir}")

    return log_dir


def main():
    """主函数"""
    # 加载配置
    config_path = 'config/default.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 开始训练
    log_dir = train_cg_ppo(config)

    print(f"\n🎉 训练完成! 结果保存至: {log_dir}")


if __name__ == "__main__":
    main()
