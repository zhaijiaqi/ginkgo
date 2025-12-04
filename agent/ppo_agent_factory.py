"""
PPO Agent Factory - 使用 pfrl 实现 PPO 代理
提供策略网络和价值网络，用于 CG 混合精度控制
"""

import torch
import torch.nn as nn
import pfrl
from pfrl.agents import PPO
import numpy as np
from typing import Dict, Any, Optional, List
import os
import logging
import time

from models import create_cg_model


# 为了向后兼容，保留这个函数
def create_cg_model(state_dim: int, action_size: int, hidden_sizes=(64, 64)):
    """创建CG模型的向后兼容函数"""
    from models.cg_model import create_cg_model as _create_cg_model
    return _create_cg_model(state_dim, action_size, hidden_sizes)


class CGPPOAgent:
    """
    为CG环境定制的PPO代理
    使用多个独立的PPO子代理，每个tile一个
    """

    def __init__(self, num_tiles: int, tile_state_dim: int, action_size: int, **ppo_kwargs):
        self.num_tiles = num_tiles
        self.tile_state_dim = tile_state_dim
        self.action_size = action_size

        # 创建共享的模型（单个tile的状态维度）
        shared_model = self._create_single_tile_model(tile_state_dim, action_size)

        # 创建多个代理实例，但共享相同的模型参数
        self.tile_agents = []
        for _ in range(num_tiles):
            # 创建PPO代理的参数
            agent_kwargs = ppo_kwargs.copy()
            agent_kwargs['model'] = shared_model  # 所有代理共享同一个模型
            agent_kwargs['optimizer'] = torch.optim.Adam(shared_model.parameters(), lr=ppo_kwargs.get('lr', 3e-4))
            agent_kwargs.pop('lr', None)

            # 创建代理实例
            agent = PPO(**agent_kwargs)
            self.tile_agents.append(agent)

    def _create_single_tile_model(self, tile_state_dim: int, action_size: int):
        """创建单个tile的模型"""
        # 从ppo_kwargs中提取隐藏层大小，如果没有则使用默认值
        hidden_sizes = [64, 64]  # 默认隐藏层大小

        def make_policy_network():
            layers = []
            prev_size = tile_state_dim

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, action_size))
            policy_net = nn.Sequential(*layers)

            # 初始化权重
            for layer in policy_net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

            return policy_net

        def make_value_network():
            layers = []
            prev_size = tile_state_dim

            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))
            value_net = nn.Sequential(*layers)

            # 初始化权重
            for layer in value_net:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

            return value_net

        policy = make_policy_network()
        value = make_value_network()

        # 使用 SoftmaxCategoricalHead 将策略输出转换为分类分布
        policy_with_dist = nn.Sequential(policy, pfrl.policies.SoftmaxCategoricalHead())

        model = pfrl.nn.Branched(policy_with_dist, value)
        return model

    def act(self, obs):
        """为每个tile独立选择动作（参数共享的代理）"""
        actions = []

        for tile_idx in range(self.num_tiles):
            # 提取当前tile的状态
            start_idx = tile_idx * self.tile_state_dim
            end_idx = start_idx + self.tile_state_dim
            tile_obs = obs[start_idx:end_idx]

            # 使用对应代理为当前tile选择动作
            # inference_start_time = time.time()
            action = self.tile_agents[tile_idx].act(tile_obs)
            # inference_end_time = time.time()
            # print(f"inference 时间: {(inference_end_time - inference_start_time) * 1000:.3f} ms")
            actions.append(action)

        return actions

    def observe(self, obs, reward, done, reset):
        """观察多tile环境的转换（参数共享的代理）"""
        # 为每个代理调用observe，每个代理观察对应tile的状态
        for tile_idx in range(self.num_tiles):
            # 提取当前tile的状态
            start_idx = tile_idx * self.tile_state_dim
            end_idx = start_idx + self.tile_state_dim
            tile_obs = obs[start_idx:end_idx] if len(obs) > 0 else obs

            # 为对应代理调用observe
            self.tile_agents[tile_idx].observe(tile_obs, reward, done, reset)

    def save(self, path: str):
        """保存参数共享的代理模型"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存参数共享的代理
        saved_data = {
            'num_tiles': self.num_tiles,
            'tile_state_dim': self.tile_state_dim,
            'action_size': self.action_size,
        }

        # 保存第一个代理（所有代理共享相同模型，所以只需要保存一个）
        self.tile_agents[0].save(path + "_shared")
        saved_data['shared_agent_path'] = path + "_shared"

        # 保存元数据
        import torch
        torch.save(saved_data, path)

    def load(self, path: str):
        """加载参数共享的代理模型"""
        import torch
        saved_data = torch.load(path)

        # 验证参数匹配
        assert saved_data['num_tiles'] == self.num_tiles
        assert saved_data['tile_state_dim'] == self.tile_state_dim
        assert saved_data['action_size'] == self.action_size

        # 加载共享模型（加载到一个代理，然后所有代理都会共享相同的参数）
        agent_path = saved_data['shared_agent_path']
        self.tile_agents[0].load(agent_path)

    def eval_mode(self):
        """切换到评估模式"""
        for tile_agent in self.tile_agents:
            tile_agent.eval_mode()




class PPOAgentFactory:
    """
    PPO Agent 工厂类
    创建和配置用于 CG 混合精度控制的 PPO 代理
    """

    def __init__(self, yaml_config: Dict):
        """
        初始化工厂

        Args:
            config: PPO 配置参数
        """
        self.config = yaml_config.get('ppo', {})

    def create_agent(self, state_dim: int, action_size: int) -> CGPPOAgent:
        """
        创建为CG环境定制的PPO代理

        Args:
            state_dim: 状态维度
            action_size: 动作空间大小

        Returns:
            配置好的 CGPPOAgent 代理
        """
        # 计算tile数量
        # 假设每个tile的状态维度 = tilesize + 1（迭代索引）
        tilesize = self.config.get('tilesize', 32)  # 从配置中读取tilesize
        tile_state_dim = tilesize + 1
        num_tiles = state_dim // tile_state_dim

        # 网络参数（传递给CGPPOAgent的子代理）

        # PPO 参数（传递给子代理）
        
        print("self.update_interval: ", self.config.get('update_interval', 2048))

        # 创建 CG PPO 代理
        # 参数解释:
        # num_tiles: tile 的数量，每个 tile 拥有独立的子 PPO 代理
        # tile_state_dim: 每个 tile 的状态维度（通常 = tilesize + 1，额外一维表示迭代步）
        # action_size: 动作空间大小，表示支持多少种混合精度动作
        # lr: 学习率 (learning rate)，用于优化代理神经网络
        # gpu: 使用的 GPU 设备编号（int），为 None 时使用 CPU
        # gamma: 折扣因子 (discount factor)，控制未来奖励的当前价值
        # lambd: GAE (Generalized Advantage Estimation) 的 lambda 值，调节 bias-variance
        # phi: 状态预处理函数，通常用于将状态转换为 np.float32 类型
        # value_func_coef: value function（值函数）损失的权重
        # entropy_coef: 熵奖励的权重，鼓励策略探索
        # update_interval: 模型参数更新的步数间隔（收集多少步经验再更新一次网络）
        # minibatch_size: 每个更新周期中用于梯度下降的小批量样本数
        # epochs: 每个 update_interval 下，所有数据被训练的轮数
        # clip_eps: PPO 策略损失裁剪参数 ε，限制策略更新幅度
        # clip_eps_vf: PPO value function 损失的裁剪参数（一般为None）
        # standardize_advantages: 是否对优势函数（Advantage）标准化
        # act_deterministically: 选择动作时是否用确定性策略（测试时常用）

        agent = CGPPOAgent(
            num_tiles=num_tiles,
            tile_state_dim=tile_state_dim,
            action_size=action_size,
            lr=float(self.config.get('learning_rate', 3e-4)),              # 学习率
            gpu=self.config.get('gpu', 0),                                 # GPU设备编号
            gamma=self.config.get('gamma', 0.99),                          # 折扣因子
            lambd=self.config.get('lambda', 0.95),                         # GAE lambda
            phi=lambda x: np.asarray(x, dtype=np.float32),                 # 状态预处理
            value_func_coef=self.config.get('value_coef', 0.5),            # 值函数损失系数
            entropy_coef=self.config.get('entropy_coef', 0.01),            # 熵奖励系数
            update_interval=self.config.get('update_interval', 1),         # 参数更新间隔
            minibatch_size=self.config.get('minibatch_size', 64),          # 小批量样本数
            epochs=self.config.get('n_epochs', 10),                        # 每 update 的 epoch 数
            clip_eps=self.config.get('clip_eps', 0.2),                     # 策略裁剪参数
            clip_eps_vf=self.config.get('clip_eps_vf', None),              # 值函数裁剪参数
            standardize_advantages=self.config.get('standardize_advantages', True),  # 优势归一化
            act_deterministically=self.config.get('act_deterministically', False),   # 行为是否确定性
        )
        return agent


class PfrlCompatibleCGPPOAgent:
    """
    将 CGPPOAgent 适配为 pfrl 兼容的 agent 接口

    这个适配器使得多 tile 的 CGPPOAgent 能够与 pfrl.experiments.train_agent_with_evaluation 一起使用
    """

    def __init__(self, cg_ppo_agent: CGPPOAgent):
        self.cg_ppo_agent = cg_ppo_agent
        self.logger = logging.getLogger(__name__)

    def act(self, obs):
        """选择动作"""
        return self.cg_ppo_agent.act(obs)

    def batch_act(self, batch_obs):
        """批量选择动作"""
        actions = []
        for obs in batch_obs:
            action = self.act(obs)
            actions.append(action)
        return actions

    def observe(self, obs, reward, done, reset):
        """观察环境转换"""
        return self.cg_ppo_agent.observe(obs, reward, done, reset)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        """批量观察环境转换"""
        for obs, reward, done, reset in zip(batch_obs, batch_reward, batch_done, batch_reset):
            self.observe(obs, reward, done, reset)

    def save(self, filename):
        """保存模型"""
        # CGPPOAgent 的 save 方法
        return self.cg_ppo_agent.save(filename)

    def load(self, filename):
        """加载模型"""
        # CGPPOAgent 的 load 方法
        return self.cg_ppo_agent.load(filename)

    def get_statistics(self):
        """获取统计信息"""
        # 获取第一个代理的统计信息（所有代理共享相同模型）
        try:
            stats = self.cg_ppo_agent.tile_agents[0].get_statistics()
            # pfrl 返回的是列表，转换为字典
            if isinstance(stats, list):
                merged_stats = {}
                for j, stat in enumerate(stats):
                    merged_stats[f"shared_stat_{j}"] = stat
                return merged_stats
            elif isinstance(stats, dict):
                return stats
            else:
                return {"shared_stats": stats}
        except Exception as e:
            self.logger.warning(f"无法获取共享代理的统计信息: {e}")
            return {}

    def eval_mode(self):
        """切换到评估模式，返回上下文管理器"""
        # pfrl期望eval_mode返回一个上下文管理器
        from contextlib import contextmanager

        @contextmanager
        def eval_context():
            # 切换到评估模式
            for tile_agent in self.cg_ppo_agent.tile_agents:
                tile_agent.eval_mode()

            try:
                yield
            finally:
                # pfrl会在需要时自动处理训练模式的恢复
                pass

        return eval_context()

    @property
    def training(self):
        """是否处于训练模式"""
        # 返回第一个代理的训练状态（所有代理状态应该一致）
        return self.cg_ppo_agent.tile_agents[0].training if self.cg_ppo_agent.tile_agents else False

    @property
    def saved_attributes(self):
        """需要保存的属性"""
        # 由于我们使用自定义的 save/load 方法，返回空列表
        return []
