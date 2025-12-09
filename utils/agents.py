"""
Agents - 代理类定义
"""


class DoublePrecisionAgent:
    """总是选择双精度 (fp64) 的简单代理"""

    def __init__(self, num_tiles: int):
        self.num_tiles = num_tiles

    def act(self, obs):
        """总是返回 fp64 动作 (0)"""
        return [0] * self.num_tiles  # 0 = fp64

    def observe(self, obs, reward, done, reset):
        """什么都不做"""
        pass

    def eval_mode(self):
        """评估模式"""
        return self

    def save(self, path):
        """什么都不做（兼容训练脚本）"""
        pass

    def load(self, path):
        """什么都不做（兼容训练脚本）"""
        pass

