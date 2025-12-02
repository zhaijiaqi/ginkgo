"""
训练脚本：使用 PPO 训练 CG 混合精度控制代理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt


# 导入项目模块
from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import PPOAgentFactory, create_cg_model


class MockPPOAgent:
    """
    模拟 PPO 代理，用于测试训练流程
    在真实实现中，这将被替换为真正的 pfrl PPO 代理
    """

    def __init__(self, state_dim: int, action_size: int):
        self.state_dim = state_dim
        self.action_size = action_size
        self.model = create_cg_model(state_dim, action_size, (64, 64))

        # 简单的 epsilon-greedy 策略
        self.epsilon = 0.1

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits, _ = self.model(obs_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

                # 直接在 tensor 上检查，无需转到 CPU/NumPy
                if (
                    torch.any(torch.isnan(probs))
                    or torch.any(torch.isinf(probs))
                    or torch.any(probs < 0)
                ):
                    return np.random.randint(self.action_size)
                else:
                    return torch.multinomial(probs, 1).item()

    def observe(self, obs, action, reward, next_obs, done):
        """观察经验 (暂时不更新)"""
        pass

    def save(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))


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
    训练 CG PPO 代理的主函数

    Args:
        config: 训练配置
    """
    print("=== 开始 CG PPO 训练 ===")

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
        'error_coeff_table': config.get('spmv', {}).get('error_coeff_table', {
            'fp64': 1e-15, 'fp32': 1e-7, 'tf32': 1e-4,
            'fp16': 1e-3, 'bf16': 5e-4, 'fp8': 1e-2
        }),
        'reward': config.get('reward'),
        'normalize_state': config.get('env', {}).get('normalize_state', True)
    }
    env = CGEnvironment(env_config)

    state_dim = env.get_state_dim()
    action_size = env.get_action_space_size()

    print(f"环境状态维度: {state_dim}")
    print(f"动作空间大小: {action_size}")

    # 创建代理 (暂时使用模拟代理)
    agent = MockPPOAgent(state_dim, action_size)

    # 创建日志记录器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # DEBUG: 打印 matrix_name 信息
    matrix_name = env_config.get('matrix_name', 'Muu')
    log_dir = os.path.join('log', f'{matrix_name}_{timestamp}')
    logger = TrainingLogger(log_dir)

    # 训练参数
    train_config = config.get('train', {})
    total_steps = train_config.get('total_steps', 10000)
    eval_interval = train_config.get('eval_interval', 1000)
    save_interval = train_config.get('save_interval', 5000)
    log_interval = train_config.get('log_interval', 100)

    print(f"总训练步数: {total_steps}")
    print(f"评估间隔: {eval_interval}")
    print(f"保存间隔: {save_interval}")

    # 训练循环
    step_count = 0
    episode_count = 0

    # 新增：以episode为单位的定期评估和保存，转换interval为以episode为单位
    eval_episode_interval = max(1, eval_interval // (step_count // max(1, episode_count)) if episode_count > 0 else eval_interval)
    save_episode_interval = max(1, save_interval // (step_count // max(1, episode_count)) if episode_count > 0 else save_interval)
    # 为保证行为和设置一致，我们直接以episode为准，interval等于之前的step-based interval除以平均每集步数
    # 但首次需要先采集至少一个episode，下面逻辑中延后计算

    episodes_since_last_eval = 0
    episodes_since_last_save = 0

    avg_steps_per_episode = 1  # 初始化，第一轮后更新

    while step_count < total_steps:
        # 开始一个 episode
        episode_reward = 0
        episode_steps = 0
        done = False

        # 重置环境
        obs = env.reset(seed=episode_count)

        while not done and step_count < total_steps:
            # 选择动作
            action = agent.act(obs)

            # 执行动作
            next_obs, reward, done, info = env.step(action)

            # 记录经验
            agent.observe(obs, action, reward, next_obs, done)

            # 更新统计
            episode_reward += reward
            episode_steps += 1
            step_count += 1

            # 记录步骤
            logger.log_step(step_count, reward)

            # 定期日志
            if step_count % log_interval == 0:
                print(f"Step {step_count}: episode {episode_count}, reward {reward:.3f}")

            obs = next_obs

        # Episode 结束
        episode_count += 1
        episodes_since_last_eval += 1
        episodes_since_last_save += 1

        episode_info = env.get_episode_info()
        episode_stats = {
            'steps': episode_steps,
            'total_reward': episode_reward,
            'avg_reward': episode_reward / episode_steps,
            **episode_info
        }

        logger.log_episode(episode_count, episode_stats)

        print(f"Episode {episode_count} 完成: total_reward {episode_stats.get('total_reward', 0):.3f}, iterations {episode_stats.get('iterations', 0)}, converged {episode_stats.get('converged', False)}")

        # 更新平均每集步数，用于后续动态间隔（episode为单位）
        avg_steps_per_episode = ((avg_steps_per_episode * (episode_count - 1)) + episode_steps) / episode_count

        # 用step-based的间隔除以平均每集步数，得到约等于原本step间隔的episode数
        # 保证至少为1
        eval_episode_interval = max(1, int(eval_interval / avg_steps_per_episode))
        save_episode_interval = max(1, int(save_interval / avg_steps_per_episode))

        # 定期评估（以episode为单位）
        if episodes_since_last_eval >= eval_episode_interval:
            eval_stats = evaluate_agent(env, agent, num_episodes=1)
            print(f"评估结果 (Episode {episode_count}): {eval_stats}")
            episodes_since_last_eval = 0

        # 定期保存（以episode为单位）
        if episodes_since_last_save >= save_episode_interval:
            model_path = os.path.join(log_dir, f'model_episode_{episode_count}.pt')
            agent.save(model_path)
            print(f"模型已保存: {model_path}")
            episodes_since_last_save = 0

    # 保存最终统计和模型
    logger.save_stats()
    logger.plot_training_curves()

    final_model_path = os.path.join(log_dir, 'final_model.pt')
    agent.save(final_model_path)

    print(f"训练完成! 最终模型保存至: {final_model_path}")
    print(f"训练日志保存至: {log_dir}")

    return log_dir


def evaluate_agent(env: CGEnvironment, agent: MockPPOAgent, num_episodes: int = 10) -> Dict:
    """
    评估代理性能

    Args:
        env: CG 环境
        agent: PPO 代理
        num_episodes: 评估的 episode 数量

    Returns:
        评估统计字典
    """
    print(f"开始评估代理性能 (num_episodes: {num_episodes})...")
    total_rewards = []
    costs = []
    errors = []
    converged_count = 0

    for episode in range(num_episodes):
        obs = env.reset(seed=episode + 1000)  # 使用不同的种子
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs

        episode_info = env.get_episode_info()
        total_rewards.append(episode_reward)
        costs.append(episode_info['total_cost'])
        errors.append(episode_info['total_error'])

        if episode_info['converged']:
            converged_count += 1

    return {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_cost': np.mean(costs),
        'avg_error': np.mean(errors),
        'convergence_rate': converged_count / num_episodes,
        'num_episodes': num_episodes
    }


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
