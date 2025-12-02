"""
PPO Agent Factory - 使用 pfrl 实现 PPO 代理
提供策略网络和价值网络，用于 CG 混合精度控制
"""

import torch
import torch.nn as nn
import pfrl
from pfrl.agents import PPO
import numpy as np
from typing import Dict, Any, Optional
import os



def create_cg_model(state_dim: int, action_size: int, hidden_sizes: tuple = (64, 64)):
    """
    创建用于 CG 混合精度控制的 pfrl 模型

    Args:
        state_dim: 状态维度
        action_size: 动作空间大小
        hidden_sizes: 隐藏层大小

    Returns:
        pfrl Branched 模型
    """

    def make_policy_network():
        """创建策略网络"""
        layers = []
        prev_size = state_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            prev_size = hidden_size

        # 策略头
        layers.append(nn.Linear(prev_size, action_size))

        policy_net = nn.Sequential(*layers)

        # 初始化权重
        for layer in policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        return policy_net

    def make_value_network():
        """创建价值网络"""
        layers = []
        prev_size = state_dim

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

    model = pfrl.nn.Branched(policy, value)
    return model


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

    def create_agent(self, state_dim: int, action_size: int) -> PPO:
        """
        创建 PPO 代理

        Args:
            state_dim: 状态维度
            action_size: 动作空间大小

        Returns:
            配置好的 PPO 代理
        """
        # 网络参数
        policy_hidden_sizes = self.config.get('policy_hidden_sizes', (64, 64))
        value_hidden_sizes = self.config.get('value_hidden_sizes', (64, 64))

        # 创建模型
        model = create_cg_model(state_dim, action_size, policy_hidden_sizes)

        # PPO 参数
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 3e-4))

        # 创建 PPO 代理
        agent = PPO(
            model=model,
            optimizer=optimizer,
            gpu=self.config.get('gpu', 1),  # -1 表示 CPU
            gamma=self.config.get('gamma', 0.99),
            lambd=self.config.get('lambda', 0.95),  # GAE lambda
            phi=lambda x: x.astype(np.float32, copy=False),  # 状态预处理
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
