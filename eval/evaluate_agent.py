"""
评估脚本：评估训练好的 CG PPO 代理性能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Optional
import argparse

# 导入项目模块
from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import create_cg_model


class MockPPOAgent:
    """
    模拟 PPO 代理，用于加载和评估训练好的模型
    """

    def __init__(self, state_dim: int, action_size: int, model_path: Optional[str] = None):
        self.state_dim = state_dim
        self.action_size = action_size
        self.model = create_cg_model(state_dim, action_size, (64, 64))

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """
        选择动作

        Args:
            obs: 观测状态
            deterministic: 是否使用确定性策略 (选择概率最高的动作)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits, _ = self.model(obs_tensor)

            if deterministic:
                return torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                return torch.multinomial(probs, 1).item()

    def load(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def get_action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """获取所有动作的概率"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits, _ = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            return probs.numpy()


def evaluate_agent_performance(
    env: CGEnvironment,
    agent: MockPPOAgent,
    num_episodes: int = 100,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    评估代理性能

    Args:
        env: CG 环境
        agent: PPO 代理
        num_episodes: 评估 episode 数量
        deterministic: 是否使用确定性策略

    Returns:
        评估结果字典
    """

    print(f"开始评估代理性能 ({num_episodes} 个 episodes)...")

    episode_results = []

    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"评估进度: {episode}/{num_episodes}")

        # 重置环境
        obs = env.reset(seed=episode + 2000)  # 使用固定种子保证可重现
        episode_reward = 0
        step_count = 0
        done = False

        # 记录动作分布
        actions_taken = []
        tile_precisions = []

        while not done:
            # 选择动作
            action = agent.act(obs, deterministic=deterministic)
            actions_taken.append(action)

            # 执行动作
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # 记录 tile 精度选择
            precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
            tile_precisions.append(precision_names[action])

            obs = next_obs

        # 获取 episode 信息
        episode_info = env.get_episode_info()

        episode_result = {
            'episode': episode,
            'total_reward': episode_reward,
            'steps': step_count,
            'converged': episode_info['converged'],
            'final_residual': episode_info['final_residual'],
            'iterations': episode_info['iterations'],
            'total_cost': episode_info['total_cost'],
            'total_error': episode_info['total_error'],
            'residual_history': episode_info['residual_history'],
            'actions_taken': actions_taken,
            'tile_precisions': tile_precisions
        }

        episode_results.append(episode_result)

    # 计算汇总统计
    converged_episodes = [r for r in episode_results if r['converged']]
    convergence_rate = len(converged_episodes) / num_episodes

    summary_stats = {
        'num_episodes': num_episodes,
        'convergence_rate': convergence_rate,
        'avg_total_reward': np.mean([r['total_reward'] for r in episode_results]),
        'std_total_reward': np.std([r['total_reward'] for r in episode_results]),
        'avg_iterations': np.mean([r['iterations'] for r in episode_results]),
        'avg_cost': np.mean([r['total_cost'] for r in episode_results]),
        'avg_error': np.mean([r['total_error'] for r in episode_results]),
        'avg_final_residual': np.mean([r['final_residual'] for r in episode_results if r['final_residual']]),
    }

    # 动作分布分析
    all_actions = []
    precision_counts = {'fp64': 0, 'fp32': 0, 'tf32': 0, 'fp16': 0, 'bf16': 0, 'fp8': 0}

    for result in episode_results:
        all_actions.extend(result['actions_taken'])
        for precision in result['tile_precisions']:
            precision_counts[precision] += 1

    action_distribution = np.bincount(all_actions, minlength=6) / len(all_actions)
    precision_distribution = {k: v / sum(precision_counts.values()) for k, v in precision_counts.items()}

    summary_stats.update({
        'action_distribution': action_distribution.tolist(),
        'precision_distribution': precision_distribution
    })

    return {
        'summary_stats': summary_stats,
        'episode_results': episode_results
    }


def plot_evaluation_results(eval_results: Dict, save_dir: str):
    """
    绘制评估结果图表

    Args:
        eval_results: 评估结果
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    summary = eval_results['summary_stats']
    episodes = eval_results['episode_results']

    # 1. 收敛分析
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 收敛率
    converged = [1 if r['converged'] else 0 for r in episodes]
    axes[0, 0].plot(converged, 'o', alpha=0.6)
    axes[0, 0].set_title('Convergence per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Converged (1/0)')
    axes[0, 0].set_ylim(-0.1, 1.1)

    # 迭代次数分布
    iterations = [r['iterations'] for r in episodes]
    axes[0, 1].hist(iterations, bins=20, alpha=0.7)
    axes[0, 1].set_title('Iterations Distribution')
    axes[0, 1].set_xlabel('Iterations')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(summary['avg_iterations'], color='red', linestyle='--', label=f'Avg: {summary["avg_iterations"]:.1f}')
    axes[0, 1].legend()

    # 最终残差分布 (对收敛的 episodes)
    converged_residuals = [r['final_residual'] for r in episodes if r['converged'] and r['final_residual']]
    if converged_residuals:
        axes[0, 2].hist(converged_residuals, bins=20, alpha=0.7)
        axes[0, 2].set_title('Final Residuals (Converged)')
        axes[0, 2].set_xlabel('Final Residual')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].axvline(np.mean(converged_residuals), color='red', linestyle='--',
                          label=f'Avg: {np.mean(converged_residuals):.2e}')
        axes[0, 2].legend()
        axes[0, 2].set_yscale('log')

    # 成本 vs 误差散点图
    costs = [r['total_cost'] for r in episodes]
    errors = [r['total_error'] for r in episodes]
    scatter = axes[1, 0].scatter(costs, errors, c=converged, alpha=0.6, cmap='viridis')
    axes[1, 0].set_title('Cost vs Error')
    axes[1, 0].set_xlabel('Total Cost')
    axes[1, 0].set_ylabel('Total Error')
    plt.colorbar(scatter, ax=axes[1, 0], label='Converged')

    # 动作分布
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    action_dist = summary['action_distribution']
    axes[1, 1].bar(precision_names, action_dist)
    axes[1, 1].set_title('Action Distribution')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 奖励分布
    rewards = [r['total_reward'] for r in episodes]
    axes[1, 2].hist(rewards, bins=20, alpha=0.7)
    axes[1, 2].set_title('Total Reward Distribution')
    axes[1, 2].set_xlabel('Total Reward')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].axvline(summary['avg_total_reward'], color='red', linestyle='--',
                       label=f'Avg: {summary["avg_total_reward"]:.2f}')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 残差收敛曲线示例 (前几个 episodes)
    plt.figure(figsize=(12, 8))

    for i, episode in enumerate(episodes[:min(6, len(episodes))]):
        plt.subplot(2, 3, i+1)
        residuals = episode['residual_history']
        plt.plot(residuals, 'b-', alpha=0.7)
        plt.title(f'Episode {episode["episode"]} (Conv: {episode["converged"]})')
        plt.xlabel('CG Iteration')
        plt.ylabel('Residual Norm')
        plt.yscale('log')
        if residuals:
            plt.axhline(1e-6, color='red', linestyle='--', alpha=0.5, label='Tolerance')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'residual_convergence_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_report(eval_results: Dict, save_dir: str):
    """
    保存评估报告

    Args:
        eval_results: 评估结果
        save_dir: 保存目录
    """
    summary = eval_results['summary_stats']

    report = f"""
# CG PPO Agent Evaluation Report

## Summary Statistics
- Episodes evaluated: {summary['num_episodes']}
- Convergence rate: {summary['convergence_rate']:.3f}
- Average total reward: {summary['avg_total_reward']:.3f} ± {summary['std_total_reward']:.3f}
- Average iterations: {summary['avg_iterations']:.1f}
- Average cost: {summary['avg_cost']:.3f}
- Average error: {summary['avg_error']:.6f}
- Average final residual: {summary['avg_final_residual']:.2e}

## Precision Selection Distribution
"""

    for precision, freq in summary['precision_distribution'].items():
        report += f"- {precision}: {freq:.3f}\n"

    report += "\n## Action Distribution\n"
    precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    for i, freq in enumerate(summary['action_distribution']):
        report += f"- {precision_names[i]}: {freq:.3f}\n"

    # 保存报告
    with open(os.path.join(save_dir, 'evaluation_report.md'), 'w') as f:
        f.write(report)

    # 保存完整结果
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Evaluate CG PPO Agent')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, default='config/default.yaml', help='Config file path')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    parser.add_argument('--save_dir', type=str, help='Directory to save results')

    args = parser.parse_args()

    # 加载配置
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

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

    # 创建代理并加载模型
    agent = MockPPOAgent(state_dim, action_size, args.model_path)

    # 评估
    eval_results = evaluate_agent_performance(
        env, agent, args.num_episodes, args.deterministic
    )

    # 保存结果
    save_dir = args.save_dir or f'eval_results_{os.path.basename(args.model_path).replace(".pt", "")}'
    os.makedirs(save_dir, exist_ok=True)

    # 生成图表和报告
    plot_evaluation_results(eval_results, save_dir)
    save_evaluation_report(eval_results, save_dir)

    print(f"\n🎉 评估完成! 结果保存至: {save_dir}")
    print(f"收敛率: {eval_results['summary_stats']['convergence_rate']:.3f}")
    print(f"平均奖励: {eval_results['summary_stats']['avg_total_reward']:.3f}")


if __name__ == "__main__":
    # 如果直接运行，进行简单测试
    if len(sys.argv) == 1:
        print("运行简单评估测试...")

        # 加载默认配置
        with open('config/default.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # 创建环境
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

        # 创建随机代理进行测试
        agent = MockPPOAgent(env.get_state_dim(), env.get_action_space_size())

        # 评估
        eval_results = evaluate_agent_performance(env, agent, num_episodes=10)

        # 保存到测试目录
        test_dir = 'eval_test_results'
        os.makedirs(test_dir, exist_ok=True)

        plot_evaluation_results(eval_results, test_dir)
        save_evaluation_report(eval_results, test_dir)

        print(f"测试评估完成! 结果保存至: {test_dir}")
    else:
        main()
