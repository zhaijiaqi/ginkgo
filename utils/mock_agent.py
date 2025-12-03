"""
Mock PPO Agent - 模拟PPO代理，用于加载和评估训练好的模型
"""

import os
import torch
import numpy as np
from typing import Optional

from models import create_cg_model


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
            policy_dist, _ = self.model(obs_tensor)

            if deterministic:
                # 对于确定性策略，选择概率最高的动作
                return policy_dist.logits.argmax(dim=-1).item()
            else:
                # 对于随机策略，从分布中采样
                return policy_dist.sample().item()

    def load(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def save(self, model_path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), model_path)
