"""
训练脚本：使用 PPO 训练 CG 混合精度控制代理
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import copy
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
        'normalize_state': config.get('env', {}).get('normalize_state', True),
        'use_torch_state': config.get('env', {}).get('use_torch_state', True)
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
    max_iter = config.get('cg', {}).get('max_iter', 1000)
    total_episodes = train_config.get('total_episodes', 100)
    total_steps = total_episodes * max_iter

    return {
        'total_steps': total_steps,
        'total_episodes': total_episodes,
        'eval_interval': train_config.get('eval_interval', 1000),
        'save_interval': train_config.get('save_interval', 1000),
        'eval_n_episodes': train_config.get('eval_n_episodes', 5)
    }


class DoublePrecisionWrapperAgent:
    """包装 PPO 代理，实现概率逐渐衰减的精度选择策略"""

    def __init__(self, ppo_agent=None, num_tiles: int = None, total_episodes: int = 1000):
        self.ppo_agent = ppo_agent
        self.num_tiles = num_tiles
        self.total_episodes = total_episodes  # 总训练episode数
        self.episode_count = 0
        self.is_force_dp_episode = True  # 第一个episode强制使用双精度
        self.force_episode_stats = None
        self.force_episode_started = False

        # 如果没有提供ppo_agent，说明这是一个纯双精度代理
        if ppo_agent is None:
            self.is_force_dp_episode = True  # 总是强制双精度

        # 添加 tile_agents 属性，指向底层代理的 tile_agents
        self.tile_agents = ppo_agent.tile_agents if hasattr(ppo_agent, 'tile_agents') and ppo_agent is not None else []

    def _get_force_dp_probability(self, episode_num: int) -> float:
        """
        计算当前episode强制使用双精度的概率

        Args:
            episode_num: 当前episode编号（从1开始）

        Returns:
            强制使用双精度的概率 (0.0 到 1.0)
        """
        if episode_num == 1:
            # 第一个episode强制使用双精度
            return 1.0

        # phase_1_end = int(self.total_episodes * 0.1)   # 前10%的episode
        # phase_2_end = int(self.total_episodes * 0.3)   # 前30%的episode

        # if episode_num <= phase_1_end:
            # 前10%的episode: 100%概率强制使用双精度
            # return 0.5
        # elif episode_num <= phase_2_end:
        #     # 从10%到30%的episode: 概率从50%线性衰减到0%
        #     progress = (episode_num - phase_1_end) / (phase_2_end - phase_1_end)
        #     return 0.5 - 0.5 * progress
        else:
            # 30%之后的episode: 10%概率强制双精度，完全由agent决定
            return 0

    def act(self, obs):
        
        """根据概率策略选择是否强制使用双精度动作"""
        if self.ppo_agent is None:
            # 纯双精度代理总是使用 fp64 (动作 0)
            return [0] * self.num_tiles

        # 兼容：某些配置/初始化异常可能导致 tile_agents 为空，避免直接索引崩溃
        is_training = False
        try:
            if hasattr(self.ppo_agent, "training"):
                is_training = bool(self.ppo_agent.training)
        except Exception:
            is_training = False
        try:
            if self.tile_agents and hasattr(self.tile_agents[0], "training"):
                is_training = bool(self.tile_agents[0].training)
        except Exception:
            pass

        if self.is_force_dp_episode and is_training:
            # 当前episode被确定为强制双精度episode
            self.ppo_agent.act(obs) # 假装使用PPO代理选择动作，实际上不使用
            return [0] * self.num_tiles
        else:
            # 使用 PPO 代理的动作
            return self.ppo_agent.act(obs)

    def observe(self, obs, reward, done, reset):
        """观察转换，强制双精度episode不记录到PPO缓冲区"""
        # 如果是纯双精度代理，不需要记录任何观察
        if self.ppo_agent is None:
            return
        
        # 如果是强制双精度episode且done，不记录观察，更新计数与标志
        if self.is_force_dp_episode and done:
            self.episode_count += 1
            self.is_force_dp_episode = False
            # 决定下一个episode是否强制双精度
            if np.random.random() < self._get_force_dp_probability(self.episode_count + 1):
                self.is_force_dp_episode = True
            return 

        # 观察转换
        self.ppo_agent.observe(obs, reward, done, reset)

        # episode结束时，更新计数和强制双精度标志
        if done:
            self.episode_count += 1
            self.is_force_dp_episode = False
            if np.random.random() < self._get_force_dp_probability(self.episode_count + 1):
                self.is_force_dp_episode = True


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

    # 如果是包装器代理，确保强制双精度episode模式
    if hasattr(agent, 'is_force_dp_episode'):
        agent.is_force_dp_episode = True

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
            print(f"双精度步骤 {step_count}: 相对残差 = {info['residual_norm_relative']:.6e}")

    # 获取 episode 统计信息
    episode_info = env.get_episode_info()

    result = {
        'iterations': episode_info['iterations'],
        'compute_cost': episode_info['total_cost'],
        'final_residual': episode_info['final_residual'],
        'converged': episode_info['converged'],
        'avg_tile_cost': episode_info['avg_tile_cost'],
        'step_rewards': episode_info['step_rewards']
    }

    print("=== 双精度 episode 完成 ===")
    print(f"收敛迭代次数: {result['iterations']}")
    print(f"总计算成本: {result['compute_cost']:.6f}")
    print(f"最终残差: {result['final_residual']:.6e}")
    print(f"是否收敛: {result['converged']}")
    print(f"总奖励: {total_reward}")
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

    # 加载初始 x_0
    matrix_name = config.get('cg', {}).get('matrix_name')
    matrix_size = config.get('cg', {}).get('matrix_size')

    # 获取训练参数
    train_params = extract_train_config(config)
    total_episodes = train_params['total_episodes']

    # 使用包装器包装 PPO 代理，实现概率逐渐衰减的精度选择策略
    wrapped_agent = DoublePrecisionWrapperAgent(cg_agent, num_tiles, total_episodes=total_episodes)

    # 确保 PPO 代理处于训练模式
    for tile_agent in cg_agent.tile_agents:
        if hasattr(tile_agent, 'training'):
            tile_agent.training = True

    # 使用适配器包装为 pfrl 兼容的代理
    from agent.ppo_agent_factory import PfrlCompatibleCGPPOAgent
    agent = PfrlCompatibleCGPPOAgent(wrapped_agent)

    # 运行初始双精度 episode 来确定合适的 max_iter
    dp_result = run_double_precision_episode_with_agent(env, wrapped_agent)

    # 更新 max_iter 为收敛 iterations 的 2 倍
    original_max_iter = config['cg']['max_iter']
    new_max_iter = int(dp_result['iterations'] * 5)
    config['cg']['max_iter'] = max(new_max_iter, 10)  # 至少设置为 10
    # 更新 eval_interval 和 save_interval 和 total_steps
    eval_interval_episode = config['train']['eval_interval_episode']
    save_interval_episode = config['train']['save_interval_episode']
    train_params['eval_interval'] = eval_interval_episode * config['cg']['max_iter']
    train_params['save_interval'] = save_interval_episode * config['cg']['max_iter']
    train_params['total_steps'] = train_params['total_episodes'] * config['cg']['max_iter']
    

    print(f"更新 max_iter: {original_max_iter} -> {config['cg']['max_iter']}")
    print(f"双精度基准计算成本: {dp_result['compute_cost']:.6f}")
    print("精度选择策略: 前10%的episode 50%概率使用fp64，后续逐渐减少到0%")

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
    # 创建训练统计钩子
    training_hook = TrainingStatsHook(logger, env, train_params['eval_interval'])
    # 创建评估钩子
    eval_hook = EvalHook(training_hook)
    
    print("="*80)
    print("训练参数如下：")
    print(json.dumps(train_params, indent=2, ensure_ascii=False))
    print("="*80)

    # 使用 pfrl.experiments.train_agent_with_evaluation
    try:
        trained_agent, eval_stats_history = pfrl.experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=train_params['total_steps'],
            eval_n_steps=None,  # 不限制每次评估的步数
            eval_n_episodes=train_params['eval_n_episodes'],  # 每次评估运行eval_n_episodes个episode
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

    # 加载初始 x_0
    matrix_name = config.get('cg', {}).get('matrix_name')
    matrix_size = config.get('cg', {}).get('matrix_size')

    # 1. 双精度 baseline 评估
    print("\n📊 运行双精度 baseline 评估...")
    dp_agent = DoublePrecisionWrapperAgent(num_tiles=num_tiles)  # 纯双精度代理
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
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CG PPO训练脚本')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='配置文件路径 (默认: config/default.yaml)')
    parser.add_argument('--single-matrix', action='store_true',
                       help='单矩阵训练模式：只训练配置文件中指定的矩阵')
    parser.add_argument('--matrix-name', type=str,
                       help='指定要训练的矩阵名称（覆盖配置文件中的设置）')
    parser.add_argument('--gpu', type=int,
                       help='指定使用的GPU设备编号 (默认: 0, 自动选择)')

    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {args.config}")
        return

    # 根据命令行参数设置GPU
    if args.gpu is not None:
        config['ppo']['gpu'] = args.gpu
        print(f"使用指定的GPU设备: {args.gpu}")
    else:
        gpu_setting = config['ppo'].get('gpu', 0)
        if gpu_setting == -1:
            print("GPU设置为自动选择模式")
        else:
            print(f"使用配置文件中的GPU设备: {gpu_setting}")

    # 单矩阵训练模式
    if args.single_matrix or args.matrix_name:
        if args.matrix_name:
            # 使用命令行指定的矩阵名称
            config['cg']['matrix_name'] = args.matrix_name

        matrix_name = config['cg']['matrix_name']
        print(f"🚀 单矩阵训练模式: {matrix_name}")

        try:
            log_dir = train_cg_ppo(config)
            print(f"✅ 矩阵 {matrix_name} 训练完成! 结果保存至: {log_dir}")
        except Exception as e:
            print(f"❌ 矩阵 {matrix_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
        return

    # 批量训练模式（原来的逻辑）
    print("🔄 批量训练模式：从cg_results.csv读取所有矩阵")

    # 从 cg_results.csv 读取矩阵名称列表
    matrix_csv_path = 'cg_results.csv'
    matrix_names = []

    try:
        with open(matrix_csv_path, 'r') as csv_file:
            # 跳过第一行标题
            next(csv_file)
            for line in csv_file:
                if line.strip():  # 跳过空行
                    parts = line.strip().split(',')
                    if len(parts) >= 3:  # 确保有足够的列
                        matrix_names.append(parts[2])  # Name 列是第3列（索引2）
    except FileNotFoundError:
        print(f"警告: 找不到文件 {matrix_csv_path}，将使用配置文件中的单个矩阵")
        matrix_names = [config['cg']['matrix_name']]

    print(f"📋 发现 {len(matrix_names)} 个矩阵需要训练:")
    for i, name in enumerate(matrix_names, 1):
        print(f"  {i}. {name}")

    # 对每个矩阵进行训练
    all_log_dirs = []
    for i, matrix_name in enumerate(matrix_names, 1):
        print(f"\n{'='*60}")
        print(f"🚀 开始训练矩阵 {i}/{len(matrix_names)}: {matrix_name}")
        print(f"{'='*60}")

        # 创建当前矩阵的配置副本
        current_config = copy.deepcopy(config)
        current_config['cg']['matrix_name'] = matrix_name

        # 开始训练
        try:
            log_dir = train_cg_ppo(current_config)
            all_log_dirs.append(log_dir)
            print(f"✅ 矩阵 {matrix_name} 训练完成! 结果保存至: {log_dir}")
        except Exception as e:
            print(f"❌ 矩阵 {matrix_name} 训练失败: {e}")
            continue

    print(f"\n🎉 所有矩阵训练完成! 共训练了 {len(all_log_dirs)}/{len(matrix_names)} 个矩阵")
    if all_log_dirs:
        print("训练结果保存目录:")
        for log_dir in all_log_dirs:
            print(f"  - {log_dir}")


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print(f"训练时间: {time_end - time_start} 秒")
