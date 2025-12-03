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


class CGPPOAgent:
    """
    为CG环境定制的PPO代理
    使用多个独立的PPO子代理，每个tile一个
    """

    def __init__(self, num_tiles: int, tile_state_dim: int, action_size: int, **ppo_kwargs):
        self.num_tiles = num_tiles
        self.tile_state_dim = tile_state_dim
        self.action_size = action_size
        self.tile_agents = []

        # 为每个tile创建独立的PPO代理
        for _ in range(num_tiles):
            # 创建单个tile的模型
            model = self._create_single_tile_model(tile_state_dim, action_size)

            # 创建PPO代理的参数
            agent_kwargs = ppo_kwargs.copy()
            agent_kwargs['model'] = model
            # 为每个子代理创建独立的optimizer
            agent_kwargs['optimizer'] = torch.optim.Adam(model.parameters(), lr=ppo_kwargs.get('lr', 3e-4))
            # 移除PPO不接受的参数
            agent_kwargs.pop('lr', None)

            # 创建子代理
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
        """为每个tile独立选择动作"""
        actions = []

        for tile_idx in range(self.num_tiles):
            # 提取当前tile的状态
            start_idx = tile_idx * self.tile_state_dim
            end_idx = start_idx + self.tile_state_dim
            tile_obs = obs[start_idx:end_idx]

            # 使用对应的子代理选择动作
            action = self.tile_agents[tile_idx].act(tile_obs)
            actions.append(action)

        return actions

    def observe(self, obs, reward, done, reset):
        """观察多tile环境的转换"""
        # 为每个子代理调用observe
        for tile_idx in range(self.num_tiles):
            # 提取当前tile的状态
            start_idx = tile_idx * self.tile_state_dim
            end_idx = start_idx + self.tile_state_dim
            tile_obs = obs[start_idx:end_idx] if len(obs) > 0 else obs

            # 为子代理调用observe
            self.tile_agents[tile_idx].observe(tile_obs, reward, done, reset)




class PPOAgentFactory:
    """
    PPO Agent 工厂类
    创建和配置用于 CG 混合精度控制的 PPO 代理
    """

    def __init__(self, config: Dict):
        """
        初始化工厂

        Args:
            config: PPO 配置参数
        """
        self.config = config

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
        tilesize = 32  # 默认tilesize，需要与环境保持一致
        tile_state_dim = tilesize + 1
        num_tiles = state_dim // tile_state_dim

        # 网络参数（传递给CGPPOAgent的子代理）

        # PPO 参数（传递给子代理）

        # 创建 CG PPO 代理
        agent = CGPPOAgent(
            num_tiles=num_tiles,
            tile_state_dim=tile_state_dim,
            action_size=action_size,
            lr=self.config.get('learning_rate', 3e-4),
            gpu=self.config.get('gpu', 0),  # -1 表示 CPU
            gamma=self.config.get('gamma', 0.99),
            lambd=self.config.get('lambda', 0.95),  # GAE lambda
            phi=lambda x: np.asarray(x, dtype=np.float32),  # 状态预处理
            value_func_coef=self.config.get('value_coef', 0.5),
            entropy_coef=self.config.get('entropy_coef', 0.01),
            update_interval=self.config.get('update_interval', 2048),
            minibatch_size=self.config.get('minibatch_size', 64),
            epochs=self.config.get('n_epochs', 10),
            clip_eps=self.config.get('clip_eps', 0.2),
            clip_eps_vf=self.config.get('clip_eps_vf', None),
            standardize_advantages=self.config.get('standardize_advantages', True),
            act_deterministically=self.config.get('act_deterministically', False),
        )
        return agent

    def create_config_from_yaml(self, yaml_config: Dict) -> Dict:
        """
        从 YAML 配置创建 PPO 配置

        Args:
            yaml_config: YAML 配置字典

        Returns:
            PPO 配置字典
        """
        ppo_config = yaml_config.get('ppo', {})

        return {
            'learning_rate': ppo_config.get('learning_rate', 3e-4),
            'gamma': ppo_config.get('gamma', 0.99),
            'lambda': ppo_config.get('lambda', 0.95),
            'value_coef': ppo_config.get('value_coef', 0.5),
            'entropy_coef': ppo_config.get('entropy_coef', 0.01),
            'batch_size': ppo_config.get('batch_size', 64),
            'n_epochs': ppo_config.get('n_epochs', 10),
            'clip_eps': ppo_config.get('clip_range', 0.2),
            'update_interval': ppo_config.get('update_interval', 2048),
            'gpu': ppo_config.get('gpu', -1),
            'policy_hidden_sizes': ppo_config.get('policy_hidden_sizes', (64, 64)),
            'value_hidden_sizes': ppo_config.get('value_hidden_sizes', (64, 64)),
            'standardize_advantages': ppo_config.get('standardize_advantages', True),
            'act_deterministically': ppo_config.get('act_deterministically', False),
        }
