#!/usr/bin/env python3
"""
轻量级精度选择器：基于 sub_p 特征和迭代次数的规则函数
根据 profileP.py 分析发现的模式，实现一个轻量级的精度选择策略
"""

import numpy as np
from typing import List, Dict, Optional


def extract_sub_p_features(sub_p: np.ndarray) -> Dict[str, float]:
    """
    提取 sub_p 的特征统计信息
    
    Args:
        sub_p: 子向量（可以是 numpy 数组或列表）
        
    Returns:
        特征字典
    """
    # 确保 sub_p 是 numpy 数组
    if not isinstance(sub_p, np.ndarray):
        sub_p = np.array(sub_p)
    
    # 确保是浮点数类型
    sub_p = sub_p.astype(np.float64)
    
    features = {
        'mean': float(np.mean(sub_p)),
        'std': float(np.std(sub_p)),
        'min': float(np.min(sub_p)),
        'max': float(np.max(sub_p)),
        'l1_norm': float(np.linalg.norm(sub_p, ord=1)),
        'l2_norm': float(np.linalg.norm(sub_p, ord=2)),
        'max_abs': float(np.max(np.abs(sub_p))),
        'mean_abs': float(np.mean(np.abs(sub_p))),
        'range': float(np.max(sub_p) - np.min(sub_p)),
        'size': len(sub_p)
    }
    
    # 计算非零元素比例
    non_zero_count = np.count_nonzero(sub_p)
    features['non_zero_ratio'] = float(non_zero_count / len(sub_p)) if len(sub_p) > 0 else 0.0
    
    # 计算正负元素比例
    positive_count = np.count_nonzero(sub_p > 0)
    negative_count = np.count_nonzero(sub_p < 0)
    features['positive_ratio'] = float(positive_count / len(sub_p)) if len(sub_p) > 0 else 0.0
    features['negative_ratio'] = float(negative_count / len(sub_p)) if len(sub_p) > 0 else 0.0
    
    return features


class LightweightPrecisionSelector:
    """
    轻量级精度选择器
    
    基于 profileP.py 分析发现的模式：
    - 低精度组（fp16, bf16, fp8）选择的 sub_p 具有更大的 L2 范数、L1 范数、最大绝对值等
    - 低精度组在早期迭代中使用更多（平均迭代 68.9 vs 78.1）
    """
    
    def __init__(self,
                 l2_norm_threshold: float = 0.01,
                 max_abs_threshold: float = 0.005,
                 early_iter_threshold: int = 70,
                 use_iteration_factor: bool = True):
        """
        初始化精度选择器
        
        Args:
            l2_norm_threshold: L2 范数阈值，超过此值倾向于使用低精度
            max_abs_threshold: 最大绝对值阈值，超过此值倾向于使用低精度
            early_iter_threshold: 早期迭代阈值，小于此迭代次数时更倾向于使用低精度
            use_iteration_factor: 是否使用迭代次数因子
        """
        self.l2_norm_threshold = l2_norm_threshold
        self.max_abs_threshold = max_abs_threshold
        self.early_iter_threshold = early_iter_threshold
        self.use_iteration_factor = use_iteration_factor
        
        # 精度代码映射：0=fp64, 1=fp32, 2=tf32, 3=fp16, 4=bf16, 5=fp8
        self.precision_names = ['fp64', 'fp32', 'tf32', 'fp16', 'bf16', 'fp8']
    
    def select_precision(self, sub_p: np.ndarray, iteration: int) -> int:
        """
        为给定的 sub_p 选择精度

        Args:
            sub_p: 子向量
            iteration: 当前迭代次数

        Returns:
            精度代码 (0-5)
        """
        # 提取特征
        features = extract_sub_p_features(sub_p)

        l2_norm = features['l2_norm']
        max_abs = features['max_abs']
        l1_norm = features['l1_norm']

        # 计算特征分数（归一化到0-1范围）
        # 基于profileP的结果，低精度组的特征值更大
        l2_score = min(1.0, l2_norm / 0.5)  # 假设0.5是较大的L2范数值
        max_abs_score = min(1.0, max_abs / 0.2)  # 假设0.2是较大的最大绝对值
        l1_score = min(1.0, l1_norm / 5.0)  # 假设5.0是较大的L1范数值

        # 综合特征分数
        feature_score = (l2_score * 0.4 + max_abs_score * 0.4 + l1_score * 0.2)

        # 迭代因子：早期迭代更倾向低精度，后期更倾向高精度
        if self.use_iteration_factor:
            if iteration < 30:
                iteration_bias = 0.8  # 非常早期，强倾向低精度
            elif iteration < self.early_iter_threshold:
                iteration_bias = 0.6  # 早期，倾向低精度
            elif iteration < 120:
                iteration_bias = 0.2  # 中期，中性
            else:
                iteration_bias = -0.3  # 后期，倾向高精度
        else:
            iteration_bias = 0.0

        # 最终分数：特征分数 + 迭代偏差
        final_score = feature_score + iteration_bias

        # 确保分数在合理范围内
        final_score = max(-1.0, min(2.0, final_score))

        # 根据最终分数选择精度
        # 分数越高，使用越低精度
        if final_score > 1.2:
            return 5, final_score  # fp8
        elif final_score > 0.8:
            return 4, final_score  # bf16
        elif final_score > 0.4:
            return 3, final_score  # fp16
        elif final_score > 0.0:
            return 2, final_score  # tf32
        elif final_score > -0.3:
            return 1, final_score  # fp32
        else:
            return 0, final_score  # fp64
    
    def select_precisions(self, p_vector: np.ndarray, iteration: int, tilesize: int) -> List[int]:
        """
        为整个 p 向量的所有 tiles 选择精度
        
        Args:
            p_vector: 完整的 p 向量
            iteration: 当前迭代次数
            tilesize: tile 大小
            
        Returns:
            精度代码列表，每个元素对应一个 tile
        """
        matrix_size = len(p_vector)
        num_tiles = (matrix_size + tilesize - 1) // tilesize
        
        actions = []
        scores = []
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tilesize
            end_idx = min(start_idx + tilesize, matrix_size)
            sub_p = p_vector[start_idx:end_idx]
            
            # 如果最后一个 tile 不足 tilesize，补0
            if len(sub_p) < tilesize:
                sub_p = np.concatenate([sub_p, np.zeros(tilesize - len(sub_p))])
            
            precision, score = self.select_precision(sub_p, iteration)
            actions.append(precision)
            scores.append(score)
        print(f"scores: {[round(s, 2) for s in scores]}")
        return actions


class AdaptiveLightweightPrecisionSelector(LightweightPrecisionSelector):
    """
    自适应轻量级精度选择器
    
    根据残差范数动态调整阈值，在收敛过程中逐渐提高精度
    """
    
    def __init__(self, 
                 l2_norm_threshold: float = 0.1,
                 max_abs_threshold: float = 0.05,
                 early_iter_threshold: int = 70,
                 use_iteration_factor: bool = True,
                 residual_adaptation: bool = True):
        """
        初始化自适应精度选择器
        
        Args:
            residual_adaptation: 是否根据残差范数自适应调整
            其他参数同 LightweightPrecisionSelector
        """
        super().__init__(l2_norm_threshold, max_abs_threshold, 
                        early_iter_threshold, use_iteration_factor)
        self.residual_adaptation = residual_adaptation
        self.initial_residual_norm = None
    
    def select_precision(self, sub_p: np.ndarray, iteration: int, 
                        residual_norm: Optional[float] = None) -> int:
        """
        为给定的 sub_p 选择精度（带残差自适应）

        Args:
            sub_p: 子向量
            iteration: 当前迭代次数
            residual_norm: 当前残差范数（直接基于当前残差决策）

        Returns:
            精度代码 (0-5)
        """
        # 基础精度选择
        base_precision, base_score = super().select_precision(sub_p, iteration)

        # 仅根据当前残差norm进行自适应决策，不依赖初始残差
        if self.residual_adaptation and residual_norm is not None:
            # 直接根据绝对残差norm判断
            # 提高精度的阈值可根据实际需求微调
            if residual_norm < 1e-9:
                # 非常小的残差，使用高精度
                if base_precision >= 3:  # 如果是低精度，提升到 tf32
                    return 2  # tf32
                elif base_precision == 2:  # 如果是 tf32，提升到 fp32
                    return 1  # fp32
                elif base_precision == 1:  # 如果是 fp32，提升到 fp16
                    return 0  # fp16
            elif residual_norm < 1e-8:
                # 较小的残差，适度提高精度
                if base_precision >= 4:  # 如果是 fp8 或 bf16，提升到 fp16
                    return 3  # fp16
                elif base_precision == 3:  # 如果是 fp16，提升到 tf32
                    return 2  # tf32

        return base_precision
    
    def select_precisions(self, p_vector: np.ndarray, iteration: int, 
                         tilesize: int, residual_norm: Optional[float] = None) -> List[int]:
        """
        为整个 p 向量的所有 tiles 选择精度（带残差自适应）
        
        Args:
            p_vector: 完整的 p 向量
            iteration: 当前迭代次数
            tilesize: tile 大小
            residual_norm: 当前残差范数（可选）
            
        Returns:
            精度代码列表，每个元素对应一个 tile
        """
        matrix_size = len(p_vector)
        num_tiles = (matrix_size + tilesize - 1) // tilesize
        
        actions = []
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tilesize
            end_idx = min(start_idx + tilesize, matrix_size)
            sub_p = p_vector[start_idx:end_idx]
            
            # 如果最后一个 tile 不足 tilesize，补0
            if len(sub_p) < tilesize:
                sub_p = np.concatenate([sub_p, np.zeros(tilesize - len(sub_p))])
            
            precision = self.select_precision(sub_p, iteration, residual_norm)
            actions.append(precision)
        
        return actions

