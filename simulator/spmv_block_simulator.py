"""
SpMV Block Simulator - 模拟 tile-based SpMV 的混合精度计算
提供精度选择对计算结果、成本和误差的影响模型

支持真实的混合精度SpMV计算：
- 矩阵数据按块存储在不同精度
- 计算时进行精度转换和累加
- 模拟真实的量化误差
"""

import math
import random
import struct
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
import numpy as np


class Precision(Enum):
    """支持的精度类型"""
    FP64 = 0
    FP32 = 1
    FP16 = 2
    FP8 = 3


class SparseMatrix:
    """
    简单的稀疏矩阵表示（COO格式）
    用于SpMV模拟
    """

    def __init__(self, rows: int, cols: int, nnz: int = 0):
        """
        初始化稀疏矩阵

        Args:
            rows: 矩阵行数
            cols: 矩阵列数
            nnz: 非零元素数量（预分配）
        """
        self.rows = rows
        self.cols = cols
        self.nnz = nnz

        if nnz > 0:
            self.row_indices: np.ndarray = np.empty(nnz, dtype=np.int32)
            self.col_indices: np.ndarray = np.empty(nnz, dtype=np.int32)
            self.values: np.ndarray = np.empty(nnz, dtype=np.float64)
            self._current_idx = 0  # 当前添加位置
        else:
            self.row_indices: np.ndarray = np.array([], dtype=np.int32)
            self.col_indices: np.ndarray = np.array([], dtype=np.int32)
            self.values: np.ndarray = np.array([], dtype=np.float64)
            self._current_idx = 0

    def add_element(self, row: int, col: int, value: float):
        """添加矩阵元素（高效版本，避免np.append）"""
        if self._current_idx >= len(self.row_indices):
            raise RuntimeError(f"超出预分配的非零元素数量: {self._current_idx} >= {len(self.row_indices)}")

        self.row_indices[self._current_idx] = row
        self.col_indices[self._current_idx] = col
        self.values[self._current_idx] = value
        self._current_idx += 1

    def add_elements_batch(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray):
        """批量添加矩阵元素（高效版本）"""
        n_new = len(rows)
        if self._current_idx + n_new > len(self.row_indices):
            raise RuntimeError(f"超出预分配的非零元素数量: {self._current_idx + n_new} > {len(self.row_indices)}")

        self.row_indices[self._current_idx:self._current_idx + n_new] = rows
        self.col_indices[self._current_idx:self._current_idx + n_new] = cols
        self.values[self._current_idx:self._current_idx + n_new] = values
        self._current_idx += n_new

    def finalize(self):
        """完成矩阵构造，确保nnz正确"""
        # 如果通过add_element添加的元素少于预分配空间，使用_current_idx
        # 否则假设数组已被直接填充，使用数组长度
        if self._current_idx > 0:
            self.nnz = self._current_idx
            # 截断未使用的预分配空间
            if self._current_idx < len(self.row_indices):
                self.row_indices = self.row_indices[:self._current_idx]
                self.col_indices = self.col_indices[:self._current_idx]
                self.values = self.values[:self._current_idx]
        else:
            # 假设数组已被直接填充
            actual_nnz = len(self.row_indices)
            self.nnz = actual_nnz
            self._current_idx = actual_nnz

    @property
    def shape(self):
        """返回矩阵形状"""
        return (self.rows, self.cols)

    @property
    def has_explicit_zeros(self):
        """检查是否存在显式的零元素"""
        return np.any(self.values == 0.0)


class PrecisionConverter:
    """
    精度转换器 - 处理不同精度之间的量化/反量化
    """

    # 精度格式规格
    PRECISION_SPECS = {
        'fp64': {'exponent': 11, 'mantissa': 52, 'bias': 1023},
        'fp32': {'exponent': 8, 'mantissa': 23, 'bias': 127},
        'fp16': {'exponent': 5, 'mantissa': 10, 'bias': 15},
        'fp8': {'exponent': 4, 'mantissa': 3, 'bias': 7},     # FP8 E4M3
    }

    @staticmethod
    def quantize_to_precision(values, precision_name: str):
        if precision_name == 'fp64':
            return np.asarray(values, dtype=np.float64)

        spec = PrecisionConverter.PRECISION_SPECS[precision_name]
        mantissa_bits = spec['mantissa']

        v = np.asarray(values, dtype=np.float64)

        # 使用向量化操作同时处理所有情况
        is_zero = (v == 0.0)
        is_special = np.isnan(v) | np.isinf(v)
        normal_mask = ~(is_zero | is_special)

        # 初始化输出数组
        out = np.empty_like(v)

        # 处理零值和特殊值
        out[is_zero] = 0.0
        out[is_special] = v[is_special]

        if np.any(normal_mask):
            # 只处理正常值
            vn = v[normal_mask]
            sign = np.sign(vn)
            av = np.abs(vn)

            # 使用向量化log2和指数计算
            exponent = np.floor(np.log2(av))
            mantissa = av / (2.0 ** exponent)

            if mantissa_bits > 0:
                scale = 2.0 ** mantissa_bits
                mantissa_q = np.round(mantissa * scale) / scale
            else:
                # mantissa_bits == 0: 类似 FP16 subnormal removal
                mantissa_q = np.where(mantissa >= 0.5, 1.0, 0.0)

            # 重组值
            out[normal_mask] = sign * mantissa_q * (2.0 ** exponent)

        return out


class SpMVBlockSimulator:
    """
    SpMV Tile 模拟器
    模拟真实的混合精度矩阵-向量乘法计算
    """

    def __init__(self, config: Dict):
        """
        初始化模拟器

        Args:
            config: 配置字典，包含精度相关的参数
        """
        self.tilesize = config.get('tilesize', 32)

        # 精度成本表 (相对 fp64 的成本)
        self.precision_cost_table = config.get('precision_cost_table', {
            'fp64': 1.0,
            'fp32': 0.5,
            'fp16': 0.25,
            'fp8': 0.125
        })

        # 随机数种子，用于重现性
        self.random_seed = config.get('random_seed', 42)
        random.seed(self.random_seed)

        # 混合精度计算模式
        self.mixed_precision_mode = config.get('mixed_precision_mode', True)

        # 向量量化缓存（性能优化）
        self._vector_quantization_cache = {}  # (vector_id, precision_name) -> quantized_vector
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def _get_quantized_vector(self, vector: List[float], precision_name: str) -> np.ndarray:
        """
        获取量化向量，使用缓存避免重复计算

        Args:
            vector: 输入向量
            precision_name: 精度名称

        Returns:
            量化后的向量
        """
        # 使用向量内容的hash作为标识符（简单实现）
        vector_tuple = tuple(vector)  # 转换为不可变类型用于hash
        cache_key = (hash(vector_tuple), precision_name)

        if cache_key in self._vector_quantization_cache:
            self._cache_hit_count += 1
            return self._vector_quantization_cache[cache_key]
        else:
            self._cache_miss_count += 1
            quantized = PrecisionConverter.quantize_to_precision(vector, precision_name)
            self._vector_quantization_cache[cache_key] = quantized
            return quantized

    def _get_precision_name(self, action: int) -> str:
        """
        将动作转换为精度名称

        Args:
            action: 精度动作 (0-5)

        Returns:
            精度名称字符串

        Raises:
            ValueError: 如果动作无效
        """
        precision_map = {
            0: 'fp64',
            1: 'fp32',
            2: 'fp16',
            3: 'fp8'
        }

        if action not in precision_map:
            raise ValueError(f"无效的精度动作: {action}，必须在 0-5 范围内")

        return precision_map[action]

    def _compute_exact_spmv_block(self, matrix_block: SparseMatrix, full_vector: List[float], precision_name: str,
                                  accumulation_precision: str = 'fp64') -> np.ndarray:
        """
        计算精确的 SpMV block 结果

        Args:
            matrix_block: 该tile的矩阵块（稀疏格式）
            full_vector: 完整的输入向量
            accumulation_precision: 累加使用的精度

        Returns:
            精确的 SpMV 结果向量（该tile贡献的部分）
        """
        if matrix_block.cols != len(full_vector):
            raise ValueError(f"矩阵列数 {matrix_block.cols} 与向量长度 {len(full_vector)} 不匹配")

        # 预先批量量化矩阵元素和向量，避免逐元素调用
        quantized_matrix_values = PrecisionConverter.quantize_to_precision(matrix_block.values, precision_name)
        quantized_vector = self._get_quantized_vector(full_vector, precision_name)

        # 使用向量化操作计算所有非零元素的贡献
        valid_mask = matrix_block.col_indices < len(quantized_vector)
        if not np.any(valid_mask):
            return np.zeros(matrix_block.rows, dtype=np.float64)

        valid_rows = matrix_block.row_indices[valid_mask]
        valid_cols = matrix_block.col_indices[valid_mask]
        valid_values = quantized_matrix_values[valid_mask]

        # 计算所有有效元素的乘积
        products = valid_values * quantized_vector[valid_cols]

        # 使用numpy的bincount进行高效累加
        result = np.bincount(valid_rows, weights=products, minlength=matrix_block.rows).astype(np.float64)

        return result

    def simulate_spmv_block(self, matrix_block: SparseMatrix, full_vector: List[float],
                           precision_action: int, accumulation_precision: str = 'fp64') -> Tuple[np.ndarray, float]:
        """
        模拟单个 SpMV tile 的混合精度计算

        Args:
            matrix_block: 该tile的矩阵块
            full_vector: 完整的输入向量
            precision_action: 精度选择动作 (0-5)
            accumulation_precision: 累加使用的精度

        Returns:
            Tuple of (partial_result, cost)
        """
        precision_name = self._get_precision_name(precision_action)

        # 计算精确结果（在累加精度下）
        quantized_result = self._compute_exact_spmv_block(matrix_block, full_vector, precision_name, accumulation_precision)

        # 计算成本
        cost = self.precision_cost_table[precision_name]

        return quantized_result, cost

    def simulate_full_spmv(self, matrix: SparseMatrix, p_vector: List[float],
                          actions_for_all_tiles: List[int], accumulation_precision: str = 'fp64') -> Tuple[np.ndarray, float]:
        """
        模拟完整 SpMV 的混合精度 tile-wise 计算和聚合

        Args:
            matrix: 完整的稀疏矩阵
            p_vector: 完整的 p 向量
            actions_for_all_tiles: 每个 tile 的精度动作列表
            accumulation_precision: 累加使用的精度

        Returns:
            Tuple of (Ap_vector, total_cost)
            - Ap_vector: 聚合后的完整 Ap 向量
            - total_cost: 所有 tiles 的总成本
        """
        if matrix.cols != len(p_vector):
            raise ValueError(f"矩阵列数 {matrix.cols} 与向量长度 {len(p_vector)} 不匹配")

        # 将矩阵按行分块
        num_tiles = (matrix.rows + self.tilesize - 1) // self.tilesize  # 向上取整
        if len(actions_for_all_tiles) != num_tiles:
            raise ValueError(f"动作数量 {len(actions_for_all_tiles)} 与 tile 数量 {num_tiles} 不匹配")

        total_cost = 0.0

        # 初始化结果向量
        Ap_vector = np.zeros(matrix.rows, dtype=np.float64)

        # 预先计算每个元素属于哪个tile
        tile_indices = matrix.row_indices // self.tilesize

        # 为每个tile处理
        for tile_idx in range(num_tiles):
            # 找到属于该tile的元素
            tile_mask = tile_indices == tile_idx
            if not np.any(tile_mask):
                # 该tile为空，跳过
                total_cost += self.precision_cost_table[self._get_precision_name(actions_for_all_tiles[tile_idx])]
                continue

            start_row = tile_idx * self.tilesize
            end_row = min((tile_idx + 1) * self.tilesize, matrix.rows)
            tile_rows = end_row - start_row

            # 创建tile矩阵
            tile_nnz = int(np.sum(tile_mask))
            tile_matrix = SparseMatrix(tile_rows, matrix.cols, tile_nnz)
            tile_matrix.row_indices[:] = matrix.row_indices[tile_mask] - start_row
            tile_matrix.col_indices[:] = matrix.col_indices[tile_mask]
            tile_matrix.values[:] = matrix.values[tile_mask]
            tile_matrix.finalize()

            # 获取该 tile 的精度动作
            precision_action = actions_for_all_tiles[tile_idx]

            # 模拟该 tile 的 SpMV 计算
            partial_result, cost = self.simulate_spmv_block(
                tile_matrix, p_vector, precision_action, accumulation_precision)

            # 批量量化部分结果，应用累加精度
            quantized_partial_results = PrecisionConverter.quantize_to_precision(
                partial_result, accumulation_precision)

            # 聚合到完整结果向量
            Ap_vector[start_row:end_row] += quantized_partial_results

            # 累加成本
            total_cost += cost

        return Ap_vector, total_cost



# ------------------------------------------------------------------------------------------------
# 测试代码
def create_large_test_matrix(size: int = 1024, density: float = 0.01) -> SparseMatrix:
    """
    创建一个大的测试稀疏矩阵

    Args:
        size: 矩阵尺寸 (size x size)
        density: 矩阵密度 (非零元素比例)

    Returns:
        稀疏矩阵
    """
    np.random.seed(42)  # 确保可重现性

    # 计算期望的非零元素数量
    expected_nnz = int(size * size * density)

    # 额外空间用于对角线元素
    diagonal_nnz = int(size * 0.8)  # 80% 的行添加对角线元素
    total_nnz = expected_nnz + diagonal_nnz

    matrix = SparseMatrix(size, size, total_nnz)

    # 使用numpy批量生成随机非零元素
    rows = np.random.randint(0, size, expected_nnz)
    cols = np.random.randint(0, size, expected_nnz)
    values = np.random.uniform(-1.0, 1.0, expected_nnz)

    # 批量添加到矩阵
    matrix.add_elements_batch(rows.astype(np.int32), cols.astype(np.int32), values.astype(np.float64))

    # 确保每行至少有一个元素（避免全零行）
    diagonal_mask = np.random.random(size) < 0.8  # 80% 的行添加对角线元素
    if np.any(diagonal_mask):
        diagonal_indices = np.where(diagonal_mask)[0]
        diagonal_values = 2.0 + np.random.uniform(-0.5, 0.5, len(diagonal_indices))

        # 添加对角线元素
        matrix.add_elements_batch(diagonal_indices.astype(np.int32),
                                  diagonal_indices.astype(np.int32),
                                  diagonal_values.astype(np.float64))

    matrix.finalize()
    return matrix


def validate_spmv_accuracy(matrix_size: int = 1024):
    """
    验证 SpMVBlockSimulator 的计算准确性

    Args:
        matrix_size: 测试矩阵的尺寸
    """
    print(f"开始验证 SpMVBlockSimulator 计算准确性 (矩阵大小: {matrix_size}x{matrix_size})")

    # 创建配置
    config = {
        'tilesize': 32,  # 使用默认的 tile 大小
        'mixed_precision_mode': True,
        'random_seed': 42
    }

    simulator = SpMVBlockSimulator(config)

    # 创建测试矩阵和向量
    print("生成测试矩阵和向量...")
    matrix = create_large_test_matrix(matrix_size, density=0.005)  # 0.5% 密度
    np.random.seed(42)
    p_vector = np.random.uniform(-1.0, 1.0, matrix_size).tolist()

    print(f"矩阵信息: {matrix.rows}x{matrix.cols}, 非零元素: {len(matrix.values)}")
    print(f"向量长度: {len(p_vector)}")

    # 计算精确结果 (使用 fp64 全精度)
    print("计算精确结果 (fp64)...")
    # 使用numpy向量化计算稀疏矩阵-向量乘法
    p_vector_np = np.array(p_vector, dtype=np.float64)
    valid_mask = matrix.col_indices < len(p_vector)

    if np.any(valid_mask):
        valid_rows = matrix.row_indices[valid_mask]
        valid_cols = matrix.col_indices[valid_mask]
        valid_values = matrix.values[valid_mask]

        # 使用bincount进行高效累加
        exact_result = np.bincount(valid_rows, weights=valid_values * p_vector_np[valid_cols],
                                   minlength=matrix.rows).astype(np.float64)
    else:
        exact_result = np.zeros(matrix.rows, dtype=np.float64)

    # 计算精确结果的 L2 范数，用于相对误差计算
    exact_norm = np.linalg.norm(exact_result)
    print(f"精确结果 L2 范数: {exact_norm:.6f}")

    # 测试不同精度
    precisions_to_test = [
        ('fp64', 0),
        ('fp32', 1),
        ('fp16', 2),
        ('fp8', 3)
    ]

    results_summary = []

    print("\n测试不同精度下的计算结果:")
    print("-" * 80)

    for precision_name, action in precisions_to_test:
        # 为所有 tiles 使用相同的精度
        num_tiles = (matrix.rows + simulator.tilesize - 1) // simulator.tilesize
        actions = [action] * num_tiles

        # 运行模拟
        sim_result, cost = simulator.simulate_full_spmv(matrix, p_vector, actions, 'fp64')

        # 计算与精确结果的差异
        diff_norm = np.linalg.norm(np.array(exact_result) - np.array(sim_result))
        relative_error = diff_norm / exact_norm if exact_norm > 1e-20 else diff_norm

        results_summary.append({
            'precision': precision_name,
            'actual_relative_error': relative_error,
            'cost': cost
        })
        
        # print(f"模拟结果: {result}")
        print(f"精度: {precision_name}")
        print(f"模拟结果 L2 范数: {np.linalg.norm(sim_result):.6f}")
        print(f"相对误差: {relative_error*100:.6f}%")
        print(f"成本: {cost:.6f}")

    print("-" * 80)

    # 验证结果合理性
    print("\n验证结果合理性:")

    # fp64 应该非常准确
    fp64_result = next(r for r in results_summary if r['precision'] == 'fp64')
    if fp64_result['actual_relative_error'] > 1e-12:
        print(f"⚠️ 警告: fp64 相对误差过高: {fp64_result['actual_relative_error']:.2e}")
    else:
        print("✓ fp64 精度验证通过")

    # 检查误差随精度降低而增加的趋势
    prev_error = 0.0
    for sim_result in results_summary:
        error = sim_result['actual_relative_error']
        if prev_error > 0 and error < prev_error * 0.1:  # 误差应该随精度降低而显著增加
            print(f"⚠️ 警告: {sim_result['precision']} 的误差 ({error:.2e}) 比前一个精度低太多")
        prev_error = error

    # 检查成本计算
    fp64_cost = next(r for r in results_summary if r['precision'] == 'fp64')['cost']
    fp8_cost = next(r for r in results_summary if r['precision'] == 'fp8')['cost']
    if fp8_cost >= fp64_cost:
        print(f"⚠️ 警告: fp8 成本 ({fp8_cost:.3f}) 不应该高于 fp64 成本 ({fp64_cost:.3f})")
    else:
        print("✓ 成本计算验证通过")

    print("\n✓ SpMVBlockSimulator 计算准确性验证完成")


if __name__ == "__main__":
    validate_spmv_accuracy(10240)
