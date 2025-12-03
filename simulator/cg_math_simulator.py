"""
CG Math Simulator - 提供 CG 算法所需的线性代数运算
基于 numpy 实现，提供高效的向量化数学运算，为 RL 环境提供精确的数学运算支持
"""

import math
from typing import List, Tuple
import numpy as np


class CGMathSimulator:
    """
    提供 CG (Conjugate Gradient) 算法所需的数学运算
    使用 numpy 实现高效的向量化运算，同时保持接口兼容性
    """

    @staticmethod
    def vector_dot(x: List[float], y: List[float]) -> float:
        """
        计算两个向量的点积: sum(x_i * y_i)

        Args:
            x: 第一个向量
            y: 第二个向量

        Returns:
            点积结果

        Raises:
            ValueError: 如果向量长度不匹配
        """
        if len(x) != len(y):
            raise ValueError(f"向量长度不匹配: len(x)={len(x)}, len(y)={len(y)}")

        # 使用 numpy 进行高效的点积运算
        return float(np.dot(x, y))

    @staticmethod
    def vector_norm(x: List[float], p: int = 2) -> float:
        """
        计算向量的 p-范数

        Args:
            x: 输入向量
            p: 范数阶数 (1, 2, 或 inf)

        Returns:
            向量范数

        Raises:
            ValueError: 如果 p 不是支持的范数类型
        """
        x_np = np.array(x)
        if p == 1:
            # L1 范数
            return float(np.linalg.norm(x_np, ord=1))
        elif p == 2:
            # L2 范数
            return float(np.linalg.norm(x_np, ord=2))
        elif p == float('inf'):
            # L-inf 范数
            return float(np.linalg.norm(x_np, ord=np.inf))
        else:
            raise ValueError(f"不支持的范数类型: p={p}")

    @staticmethod
    def vector_saxpy(a: float, x: List[float], y: List[float]) -> List[float]:
        """
        执行 SAXPY 操作: y = a*x + y

        Args:
            a: 标量
            x: 第一个向量
            y: 第二个向量 (会被修改)

        Returns:
            计算结果向量

        Raises:
            ValueError: 如果向量长度不匹配
        """
        if len(x) != len(y):
            raise ValueError(f"向量长度不匹配: len(x)={len(x)}, len(y)={len(y)}")

        # 使用 numpy 进行高效的 SAXPY 运算
        result = np.array(y) + a * np.array(x)
        return result.tolist()

    @staticmethod
    def vector_copy(x: List[float]) -> List[float]:
        """
        复制向量

        Args:
            x: 输入向量

        Returns:
            向量副本
        """
        return np.array(x).copy().tolist()

    @staticmethod
    def vector_scale(a: float, x: List[float]) -> List[float]:
        """
        向量标量乘法: result = a * x

        Args:
            a: 标量
            x: 输入向量

        Returns:
            缩放后的向量
        """
        return (a * np.array(x)).tolist()

    @staticmethod
    def vector_add(x: List[float], y: List[float]) -> List[float]:
        """
        向量加法: result = x + y

        Args:
            x: 第一个向量
            y: 第二个向量

        Returns:
            相加结果向量

        Raises:
            ValueError: 如果向量长度不匹配
        """
        if len(x) != len(y):
            raise ValueError(f"向量长度不匹配: len(x)={len(x)}, len(y)={len(y)}")

        return (np.array(x) + np.array(y)).tolist()

    @staticmethod
    def vector_sub(x: List[float], y: List[float]) -> List[float]:
        """
        向量减法: result = x - y

        Args:
            x: 第一个向量
            y: 第二个向量

        Returns:
            相减结果向量

        Raises:
            ValueError: 如果向量长度不匹配
        """
        if len(x) != len(y):
            raise ValueError(f"向量长度不匹配: len(x)={len(x)}, len(y)={len(y)}")

        return (np.array(x) - np.array(y)).tolist()


class CGResidualTracker:
    """
    CG 残差跟踪器
    跟踪 CG 算法的收敛过程和残差历史
    """

    def __init__(self):
        self.residual_history = []
        self.iteration_count = 0

    def reset(self):
        """重置跟踪器"""
        self.residual_history = []
        self.iteration_count = 0

    def record_residual(self, residual_norm: float):
        """
        记录当前迭代的残差范数

        Args:
            residual_norm: 当前残差的 L2 范数
        """
        self.residual_history.append(residual_norm)
        self.iteration_count += 1

    def get_convergence_info(self) -> dict:
        """
        获取收敛信息

        Returns:
            包含收敛统计信息的字典
        """
        if not self.residual_history:
            return {
                'iterations': 0,
                'final_residual': None,
                'initial_residual': None,
                'convergence_ratio': None
            }

        return {
            'iterations': self.iteration_count,
            'final_residual': self.residual_history[-1],
            'initial_residual': self.residual_history[0] if self.residual_history else None,
            'convergence_ratio': self.residual_history[-1] / self.residual_history[0] if len(self.residual_history) > 1 else None,
            'residual_history': self.residual_history.copy()
        }

    def is_converged(self, tolerance: float) -> bool:
        """
        检查是否收敛

        Args:
            tolerance: 收敛容忍度

        Returns:
            是否达到收敛条件
        """
        return len(self.residual_history) > 0 and self.residual_history[-1] < tolerance
