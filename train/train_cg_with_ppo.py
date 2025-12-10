"""
训练脚本：使用 PPO 训练 CG 混合精度控制代理
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# 显式导入本地utils模块
import importlib.util
utils_spec = importlib.util.spec_from_file_location("utils", os.path.join(project_root, "utils", "__init__.py"))
utils = importlib.util.module_from_spec(utils_spec)
sys.modules["utils"] = utils
utils_spec.loader.exec_module(utils)

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
            'fp64': 1.0, 'fp32': 0.5, 'fp16': 0.25, 'fp8': 0.125
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
        'eval_interval': train_config.get('eval_interval', 1000),
        'save_interval': train_config.get('save_interval', 1000),
        'log_interval': train_config.get('log_interval', 1000)
    }


class DoublePrecisionWrapperAgent:
    """包装 PPO 代理，在第一个 episode 强制使用双精度动作"""

    def __init__(self, ppo_agent, num_tiles: int, force_double_precision_episodes: int = 1):
        self.ppo_agent = ppo_agent
        self.num_tiles = num_tiles
        self.force_double_precision_episodes = force_double_precision_episodes
        self.episode_count = 0
        self.is_first_episode = True
        self.first_episode_stats = None
        self.first_episode_started = False

        # 添加 tile_agents 属性，指向底层代理的 tile_agents
        self.tile_agents = ppo_agent.tile_agents if hasattr(ppo_agent, 'tile_agents') else []

    def act(self, obs):
        """在第一个 episode 强制使用双精度动作"""
        if self.is_first_episode:
            # 第一个 episode 强制使用 fp64 (动作 0)
            return [0] * self.num_tiles
        else:
            # 后续 episodes 使用 PPO 代理的动作
            return self.ppo_agent.act(obs)

    def observe(self, obs, reward, done, reset):
        """观察转换，第一个 episode 不记录到 PPO 缓冲区"""
        # 只有在非第一个 episode 时才记录数据到 PPO 代理
        if not self.is_first_episode:
            # 正常观察
            self.ppo_agent.observe(obs, reward, done, reset)

            if reset:
                self.episode_count += 1
        elif done and self.is_first_episode:
            # 第一个 episode 结束，切换到正常模式
            self.is_first_episode = False
            self.episode_count += 1

    def start_first_episode(self):
        """标记第一个 episode 开始"""
        self.first_episode_started = True

    def save(self, path):
        """保存 PPO 代理"""
        return self.ppo_agent.save(path)

    def load(self, path):
        """加载 PPO 代理"""
        return self.ppo_agent.load(path)

    def eval_mode(self):
        """切换到评估模式"""
        return self.ppo_agent.eval_mode()

    def get_statistics(self):
        """获取统计信息"""
        return self.ppo_agent.get_statistics()

    @property
    def training(self):
        """是否处于训练模式"""
        return self.ppo_agent.training

    @property
    def saved_attributes(self):
        """需要保存的属性"""
        return self.ppo_agent.saved_attributes


def run_double_precision_episode_with_agent(env: CGEnvironment, agent) -> Dict:
    """
    使用给定的代理运行双精度 episode 来确定合适的 max_iter

    Args:
        env: CG 环境
        agent: 代理（可以是 DoublePrecisionAgent 或其他代理）

    Returns:
        包含收敛 iterations 和 compute_cost 的字典
    """
    print("=== 运行双精度 episode 来确定合适的 max_iter ===")

    # 如果是包装器代理，标记第一个 episode 开始
    if hasattr(agent, 'start_first_episode'):
        agent.start_first_episode()

    # 重置环境开始 episode
    obs = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    print(f"开始双精度 episode，矩阵大小: {env.matrix_size}x{env.matrix_size}")

    while not done:
        # 选择动作
        actions = agent.act(obs)

        # 执行一步
        next_obs, reward, done, info = env.step(actions)
        total_reward += reward

        # 观察转换
        agent.observe(obs, reward, done, done)

        obs = next_obs
        step_count += 1

        if step_count % 10 == 0:
            print(f"双精度步骤 {step_count}: 残差 = {info['residual_norm']:.6e}")

    # 获取 episode 统计信息
    episode_info = env.get_episode_info()

    result = {
        'iterations': episode_info['iterations'],
        'compute_cost': episode_info['total_cost'],
        'final_residual': episode_info['final_residual'],
        'converged': episode_info['converged'],
        'avg_tile_cost': episode_info['avg_tile_cost']
    }

    print("=== 双精度 episode 完成 ===")
    print(f"收敛迭代次数: {result['iterations']}")
    print(f"总计算成本: {result['compute_cost']:.6f}")
    print(f"最终残差: {result['final_residual']:.6e}")
    print(f"是否收敛: {result['converged']}")

    return result


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

    # 计算 tile 数量
    tilesize = env.spmv_sim.tilesize
    num_tiles = (env.matrix_size + tilesize - 1) // tilesize

    # 使用包装器包装 PPO 代理，在第一个 episode 强制使用双精度
    wrapped_agent = DoublePrecisionWrapperAgent(cg_agent, num_tiles)

    # 确保 PPO 代理处于训练模式
    for tile_agent in cg_agent.tile_agents:
        if hasattr(tile_agent, 'training'):
            tile_agent.training = True

    # 使用适配器包装为 pfrl 兼容的代理
    from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent
    agent = PfrlCompatibleCGPPOAgent(wrapped_agent)

    # 运行双精度 episode 来确定合适的 max_iter（这个 episode 的数据会进入 PPO 的缓冲区）
    dp_result = run_double_precision_episode_with_agent(env, wrapped_agent)

    # 更新 max_iter 为收敛 iterations 的 1.5 倍
    original_max_iter = config['cg']['max_iter']
    new_max_iter = int(dp_result['iterations'] * 1.5)
    config['cg']['max_iter'] = max(new_max_iter, 10)  # 至少设置为 10

    print(f"更新 max_iter: {original_max_iter} -> {config['cg']['max_iter']}")
    print(f"双精度基准计算成本: {dp_result['compute_cost']:.6f}")

    # 重新创建环境（使用更新后的 max_iter）
    env_config = create_env_config(config)
    env = CGEnvironment(env_config)

    # 创建日志记录器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    matrix_name = env_config.get('matrix_name')
    if matrix_name == 'None':
        matrix_size = env_config.get('matrix_size', 'unknownsize')
        matrix_identifier = f"size{matrix_size}"
    else:
        matrix_identifier = matrix_name
    log_dir = os.path.join('log', f'{matrix_identifier}_tilesize{tilesize}_{timestamp}')
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
            eval_n_episodes=2,  # 每次评估运行10个episode
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

    # 进行最终性能评估
    final_evaluation(log_dir, config)

    return log_dir


def final_evaluation(log_dir: str, config: Dict):
    """
    最终性能评估：比较双精度 baseline 和训练后模型的 compute_cost

    Args:
        log_dir: 日志目录
        config: 训练配置
    """
    from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent

    print("\n" + "="*60)
    print("🎯 最终性能评估：计算成本对比")
    print("="*60)

    # 创建环境配置
    env_config = create_env_config(config)
    env = CGEnvironment(env_config)

    # 计算 tile 数量
    tilesize = env.spmv_sim.tilesize
    num_tiles = (env.matrix_size + tilesize - 1) // tilesize

    print(f"评估环境: 矩阵大小 {env.matrix_size}x{env.matrix_size}, tile 数量 {num_tiles}")

    # 1. 双精度 baseline 评估
    print("\n📊 运行双精度 baseline 评估...")
    dp_agent = DoublePrecisionWrapperAgent(num_tiles)
    dp_result = run_double_precision_episode_with_agent(env, dp_agent)

    # 2. 训练后模型评估
    print("\n🤖 运行训练后模型评估...")
    # 加载最佳模型
    best_model_path = os.path.join(log_dir, 'best')
    cg_agent = PPOAgentFactory(config).create_agent(env.get_state_dim(), env.get_action_space_size())

    # 加载模型
    cg_agent.load(best_model_path)

    # 创建评估代理（正常使用训练后的策略）
    pfrl_agent = PfrlCompatibleCGPPOAgent(cg_agent)

    # 运行评估 episode
    obs = env.reset()
    done = False
    trained_cost = 0.0
    step_count = 0

    while not done and step_count < env.max_iter:
        inference_start_time = time.time()
        actions = pfrl_agent.act(obs)
        inference_end_time = time.time()
        print(f"inference 时间: {(inference_end_time - inference_start_time) * 1000:.3f} ms")
        iteration_start_time = time.time()
        next_obs, reward, done, info = env.step(actions)
        iteration_end_time = time.time()
        print(f"iteration 时间: {(iteration_end_time - iteration_start_time) * 1000:.3f} ms")
        trained_cost += info['iteration_cost']
        obs = next_obs
        step_count += 1

    # 获取训练后模型的结果
    trained_result = env.get_episode_info()

    # 3. 性能对比
    dp_cost = dp_result['compute_cost']
    trained_total_cost = trained_result['total_cost']

    improvement = (dp_cost - trained_total_cost) / dp_cost * 100

    print("\n" + "="*60)
    print("📈 性能评估结果")
    print("="*60)
    print(f"双精度 baseline 计算成本: {dp_cost:.6f}")
    print(f"训练后模型计算成本:     {trained_total_cost:.6f}")
    print(f"性能提升:                {improvement:.2f}%")
    print(f"成本减少:                {dp_cost - trained_total_cost:.6f}")

    # 保存评估结果
    eval_result = {
        'double_precision_cost': dp_cost,
        'trained_model_cost': trained_total_cost,
        'performance_improvement_percent': improvement,
        'cost_reduction': dp_cost - trained_total_cost,
        'dp_iterations': dp_result['iterations'],
        'trained_iterations': trained_result['iterations']
    }

    eval_result_path = os.path.join(log_dir, 'final_performance_evaluation.json')
    with open(eval_result_path, 'w') as f:
        json.dump(eval_result, f, indent=2)

    print(f"详细评估结果已保存至: {eval_result_path}")

    if improvement > 0:
        print("\n🎉 恭喜！训练成功实现了性能提升！")
    else:
        print("\n⚠️ 注意：训练后模型的成本高于双精度 baseline，可能需要进一步优化。")


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
