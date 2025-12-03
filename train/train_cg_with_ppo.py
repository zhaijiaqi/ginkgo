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
import time


# 导入项目模块
from env.cg_env import CGEnvironment
from agent.ppo_agent_factory import PPOAgentFactory, CGPPOAgent
from utils import TrainingLogger, TrainingStatsHook, EvalHook, convert_to_serializable


def create_env_config(config: Dict) -> Dict:
    """
    从训练配置创建环境配置

    Args:
        config: 训练配置字典

    Returns:
        环境配置字典
    """
    return {
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


def extract_train_config(config: Dict) -> Dict:
    """
    从训练配置中提取训练参数

    Args:
        config: 训练配置字典

    Returns:
        训练参数字典
    """
    train_config = config.get('train', {})
    return {
        'total_steps': train_config.get('total_steps', 10000),
        'eval_interval': train_config.get('eval_interval', 50),
        'save_interval': train_config.get('save_interval', 50),
        'log_interval': train_config.get('log_interval', 10)
    }


def train_cg_ppo(config: Dict):
    """
    训练 CG PPO 代理的主函数 - 使用 pfrl.experiments.train_agent_with_evaluation

    Args:
        config: 训练配置
    """
    print("=== 开始 CG PPO 训练 (使用 pfrl.experiments.train_agent_with_evaluation) ===")

    # 创建环境配置
    env_config = create_env_config(config)
    env = CGEnvironment(env_config)
    state_dim = env.get_state_dim()
    action_size = env.get_action_space_size()


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
    train_params = extract_train_config(config)
    # 创建训练统计钩子
    training_hook = TrainingStatsHook(logger, env, train_params['eval_interval'])
    # 创建评估钩子
    eval_hook = EvalHook(training_hook)

    # 使用 pfrl.experiments.train_agent_with_evaluation
    try:
        trained_agent, eval_stats_history = pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=train_params['total_steps'],
            eval_n_steps=None,  # 不限制每次评估的步数
            eval_n_episodes=2,  # 每次评估运行1个episode
            eval_interval=train_params['eval_interval'],
            outdir=log_dir,
            checkpoint_freq=train_params['save_interval'],  # 定期保存检查点
            step_hooks=[training_hook],  # 步骤钩子
            evaluation_hooks=[eval_hook],  # 评估钩子
            save_best_so_far_agent=True,  # 保存最佳代理
            use_tensorboard=False,  # 不使用tensorboard
            logger=logger.logger  # 使用我们的日志记录器
        )

        print("pfrl 训练完成!")
        print(f"评估统计历史记录了 {len(eval_stats_history)} 次评估")

    except Exception as e:
        import traceback
        print(f"pfrl 训练过程中出现错误: {e}")
        print("完整的错误追踪:")
        traceback.print_exc()
        return

    # 保存最终统计和模型
    logger.save_stats()
    logger.plot_training_curves()

    # 保存评估统计历史
    eval_stats_path = os.path.join(log_dir, 'eval_stats_history.json')
    with open(eval_stats_path, 'w') as f:
        serializable_stats = convert_to_serializable(eval_stats_history)
        json.dump(serializable_stats, f, indent=2)
    print(f"评估统计历史保存至: {eval_stats_path}")

    final_model_path = os.path.join(log_dir, 'final_model.pt')
    trained_agent.save(final_model_path)  # 使用训练后的代理保存模型

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
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"训练时间: {time_end - time_start} 秒")
