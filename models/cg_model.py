"""
CG Model - CG混合精度控制的神经网络模型
定义策略网络和价值网络，用于PPO代理
"""

import torch
import torch.nn as nn
import pfrl
import numpy as np
from typing import Tuple


def create_cg_model(state_dim: int, action_size: int, hidden_sizes: Tuple[int, ...] = (64, 64)):
    """
    创建用于CG混合精度控制的神经网络模型

    Args:
        state_dim: 状态维度
        action_size: 动作空间大小
        hidden_sizes: 隐藏层大小的元组，默认(64, 64)

    Returns:
        pfrl.nn.Branched: 包含策略网络和价值网络的分支模型
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

    # 使用 SoftmaxCategoricalHead 将策略输出转换为分类分布
    policy_with_dist = nn.Sequential(policy, pfrl.policies.SoftmaxCategoricalHead())

    model = pfrl.nn.Branched(policy_with_dist, value)
    return model
