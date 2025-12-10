"""
CG Environment - PPO-Guided Mixed-Precision CG 求解环境
提供完整的 RL 环境接口，用于训练 PPO 代理学习精度选择策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math
import random
import csv
import os
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.cg_math_simulator import CGMathSimulator, CGResidualTracker
from simulator.spmv_block_simulator import SpMVBlockSimulator, SparseMatrix

from kernels.spmv_kernels import bsr_spmv_mixed
from kernels.coo2bsr_kernels import coo2bsr


try:
    from scipy.io import mmread
    from scipy.sparse import csr_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class CGEnvironment:
    """
    CG (Conjugate Gradient) 强化学习环境

    每个 episode 对应一次完整的 CG 求解过程
    每个 step 对应一次 tile 粒度精度选择的 CG 迭代
    """

    def __init__(self, config: Dict):
        """
        初始化 CG 环境

        Args:
            config: 环境配置参数
        """
        self.config = config

        # CG 参数
        self.max_iter = int(config.get('max_iter', 1000))
        self.stop_tol = float(config.get('stop_tol', 1e-6))

        # 矩阵配置
        self.matrix_name = config.get('matrix_name', None)  # 矩阵名称，如果为None则使用随机生成
        self.matrix_data_dir = config.get('matrix_data_dir', '~/data/matrix')
        self.matrix_set_csv = config.get('matrix_set_csv', 'matrix_set.csv')
        self.matrix_size = int(config.get('matrix_size', 1024))  # 矩阵大小，默认1024

        # 奖励权重
        reward_config = config.get('reward', {})
        self.w1 = reward_config.get('w1', 1.0)   # 数值误差权重
        self.w2 = reward_config.get('w2', 0.1)   # 计算成本权重
        self.w3 = reward_config.get('w3', 10.0)  # 收敛奖励权重

        # 环境参数
        self.normalize_state = config.get('normalize_state', True)

        # 初始化模拟器
        spmv_config = {
            'tilesize': config.get('tilesize', 32),
            'precision_cost_table': config.get('precision_cost_table'),
            'random_seed': config.get('random_seed', 42)
        }
        self.math_sim = CGMathSimulator()
        self.spmv_sim = SpMVBlockSimulator(spmv_config)
        self.residual_tracker = CGResidualTracker()

        # 环境状态
        self.current_iteration = 0
        self.current_tile_idx = 0
        self.episode_done = False

        # 保存上一个episode的信息（用于评估钩子获取）
        self.last_episode_info = None

        # CG 变量
        self.x = None  # 解向量
        self.r = None  # 残差向量
        self.p = None  # 搜索方向
        self.Ap = None  # A*p 向量
        self.b = None  # 右端项
        self.b_norm = None  # 右端项范数

        # 矩阵数据缓存
        self.A_matrix = None  # scipy csr_matrix
        self.A_sparse = None  # 我们的 SparseMatrix 格式
        self.A_diagonal = None  # 对角线元素（用于兼容现有代码）
        self.matrix_loaded = False  # 标记矩阵是否已加载

        # 预分组的tile数据（性能优化）
        self.tile_blocks = None  # 按tile分组的矩阵块列表，每个元素是SparseMatrix
        
        # BSR 格式数据（用于 TileLang kernel）
        self.bsr_data = None  # BSR 数据块
        self.bsr_indices = None  # BSR 列索引
        self.bsr_indptr = None  # BSR 行指针
        self.bsr_R = None  # BSR 行块大小
        self.bsr_C = None  # BSR 列块大小
        
        # fp64 基准时间测量
        self.fp64_baseline_time = None  # fp64 计算 Ap 的基准时间（毫秒）

        # 轨迹跟踪
        self.episode_cpt_cost = 0.0
        self.tile_actions = []  # 当前迭代的所有 tile 动作
        self.step_rewards = []  # 每步奖励

        # 最后一步信息（用于钩子访问）
        self.last_reward = 0.0
        self.last_info = {}

        # 性能分析
        self.performance_stats = {
            'total_time': 0.0,
            'matrix_extraction_time': 0.0,
            'spmv_time': 0.0,
            'cg_math_time': 0.0,
            'iteration_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.precision_cost_table = self.config.get('precision_cost_table', {
            'fp64': 1.0,
            'fp32': 0.5,
            'fp16': 0.25,
            'fp8': 0.125
        })

        # 动作空间：6 种精度选择
        self.action_space_n = len(self.precision_cost_table.keys())
      
        # 重置环境
        self.reset()

        # 状态空间维度
        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize  # 向上取整计算 tile 数量
        # 计算实际状态维度：考虑最后一个 tile 可能不是完整 tilesize
        self.state_dim = num_tiles * (tilesize + 1)   # 每个 tile 的状态维度 = tilesize + 1（迭代索引）

    def _load_matrix_info(self) -> Dict[str, Any]:
        """
        从 matrix_set.csv 读取矩阵信息

        Returns:
            矩阵信息字典
        """
        matrix_info = {}

        # 读取CSV文件
        with open(self.matrix_set_csv, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                name = row['Name']
                matrix_info[name] = {
                    'id': int(row['id']),
                    'group': row['Group'],
                    'name': name,
                    'rows': int(row['rows']),
                    'cols': int(row['cols']),
                    'entries': int(row['entries'])
                }

        return matrix_info

    def _load_matrix_market(self, matrix_name: str) -> csr_matrix:
        """
        加载 Matrix Market 格式的稀疏矩阵

        Args:
            matrix_name: 矩阵名称（不含.mtx扩展名）

        Returns:
            CSR格式的稀疏矩阵
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for loading Matrix Market files. Please install scipy.")

        matrix_path = os.path.expanduser(os.path.join(self.matrix_data_dir, f"{matrix_name}.mtx"))

        if not os.path.exists(matrix_path):
            raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

        # 读取Matrix Market文件
        A = mmread(matrix_path)

        # 转换为CSR格式
        if not isinstance(A, csr_matrix):
            A = A.tocsr()

        return A

    def _generate_problem(self) -> Tuple[List[float], np.ndarray]:
        """
        生成 CG 测试问题 Ax = b

        如果指定了矩阵名称，则从文件中加载真实的矩阵，
        否则生成一个简单的对角占优矩阵来确保收敛。

        Returns:
            (A_diagonal, b) - 对角线元素和右端项
        """
        if self.matrix_loaded:
            # 矩阵已加载，直接返回缓存的数据
            return self.A_diagonal, self.b

        if self.matrix_name is not None:
            # 加载真实的矩阵
            try:
                self.A_matrix = self._load_matrix_market(self.matrix_name)
                self.matrix_size = self.A_matrix.shape[0]
                self.matrix_nnz = self.A_matrix.nnz

                # 提取对角线元素用于兼容现有代码
                A_diagonal = self.A_matrix.diagonal().tolist()

                # 生成右端项 b
                b = np.array([random.gauss(0, 1) for _ in range(self.matrix_size)])

                print(f"Loaded matrix '{self.matrix_name}' with size {self.matrix_size}x{self.matrix_size}, nnz {self.matrix_nnz}")

                # 将矩阵转换为我们的 SparseMatrix 格式
                self.A_sparse = self._csr_to_sparse_matrix(self.A_matrix)

                # 预先计算tile块（性能优化）
                self._precompute_tile_blocks()
                
                # 转换为 BSR 格式（用于 TileLang kernel）
                self.bsr_data, self.bsr_indices, self.bsr_indptr, self.bsr_R, self.bsr_C = self._convert_matrix_to_bsr()

                self.matrix_loaded = True

            except Exception as e:
                print(f"Failed to load matrix '{self.matrix_name}': {e}")
                print("Falling back to random matrix generation")
                self.matrix_name = None
                A_diagonal, b = self._generate_random_problem()
                self.matrix_loaded = True
                return A_diagonal, b
        else:
            return self._generate_random_problem()

        return A_diagonal, b

    def _generate_random_problem(self) -> Tuple[List[float], np.ndarray]:
        """
        生成一个随机的 CG 测试问题（对角占优矩阵）

        Returns:
            (A_diagonal, b) - 对角线元素和右端项
        """
        # 生成对角占优矩阵的对角线元素
        A_diagonal = []
        for i in range(self.matrix_size):
            # 确保对角占优：对角线元素 > 非对角线元素之和
            diagonal = 2.0 + 0.5 * random.random() + i * 0.01
            A_diagonal.append(diagonal)

        # 生成右端项 b
        b = np.array([random.gauss(0, 1) for _ in range(self.matrix_size)])

        # 为随机矩阵创建稀疏矩阵表示（对角矩阵）
        self.A_sparse = SparseMatrix(self.matrix_size, self.matrix_size, self.matrix_size)
        for i in range(self.matrix_size):
            self.A_sparse.add_element(i, i, A_diagonal[i])
        self.A_sparse.finalize()

        # 预先计算tile块（性能优化）
        self._precompute_tile_blocks()

        self.matrix_loaded = True
        return A_diagonal, b

    def _csr_to_sparse_matrix(self, csr_mat) -> SparseMatrix:
        """
        将 scipy CSR 矩阵转换为我们的 SparseMatrix 格式

        Args:
            csr_mat: scipy csr_matrix

        Returns:
            SparseMatrix 实例
        """
        # 将 CSR 格式转换为 COO 格式
        csr_mat_coo = csr_mat.tocoo()
        nnz = len(csr_mat_coo.data)

        sparse_mat = SparseMatrix(csr_mat.shape[0], csr_mat.shape[1], nnz)

        # 批量添加元素
        sparse_mat.add_elements_batch(
            csr_mat_coo.row.astype(np.int32),
            csr_mat_coo.col.astype(np.int32),
            csr_mat_coo.data.astype(np.float64)
        )
        sparse_mat.finalize()

        return sparse_mat

    def _precompute_tile_blocks(self):

        """
        预先计算所有tile的矩阵块（性能优化）
        在矩阵加载后调用，避免每次迭代都重新提取
        """
        if self.A_sparse is None:
            return

        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize
        self.tile_blocks = []

        # 为每个tile预先提取矩阵块
        for tile_idx in range(num_tiles):
            start_row = tile_idx * tilesize
            end_row = min((tile_idx + 1) * tilesize, self.matrix_size)
            tile_block = self._extract_matrix_block(start_row, end_row)
            self.tile_blocks.append(tile_block)
            
            
    def _convert_matrix_to_bsr(self):
        """
        将 COO 格式矩阵转换为 BSR 格式

        使用 tilesize 作为块大小，使行块和列块数量与 RL 环境的 tile 数量匹配

        Returns:
            BSR 格式的数据: (data, indices, indptr, R, C)
        """
        if self.A_sparse is None:
            raise RuntimeError("矩阵未加载")

        # 使用 tilesize 作为 BSR 块大小
        tilesize = self.spmv_sim.tilesize
        R, C = tilesize, tilesize

        # 准备 COO 数据，转换为 float32
        row = self.A_sparse.row_indices.astype(np.int32)
        col = self.A_sparse.col_indices.astype(np.int32)
        val = self.A_sparse.values.astype(np.float32)
        shape = (self.A_sparse.rows, self.A_sparse.cols)
        blocksize = (R, C)

        # 使用 TileLang coo2bsr kernel 进行转换
        data_bsr, indices_bsr, indptr_bsr = coo2bsr(row, col, val, shape, blocksize)

        # 转换为 float64 以匹配 bsr_spmv_mixed kernel 的期望
        data_bsr = data_bsr.astype(np.float64)

        return data_bsr, indices_bsr, indptr_bsr, R, C

    def _extract_matrix_block(self, start_row: int, end_row: int) -> SparseMatrix:
        """
        提取矩阵的一个行块（优化版本，使用向量化操作）

        Args:
            start_row: 开始行索引
            end_row: 结束行索引（不包含）

        Returns:
            该行块的 SparseMatrix
        """
        if self.A_sparse is None:
            raise RuntimeError("矩阵未加载或转换失败")

        # 使用向量化操作找到属于该行块的元素
        mask = (self.A_sparse.row_indices >= start_row) & (self.A_sparse.row_indices < end_row)
        block_nnz = np.sum(mask)

        if block_nnz == 0:
            # 空块
            block = SparseMatrix(end_row - start_row, self.A_sparse.cols, 0)
            block.finalize()
            return block

        block = SparseMatrix(end_row - start_row, self.A_sparse.cols, block_nnz)

        # 批量提取和调整行索引
        block.row_indices[:] = self.A_sparse.row_indices[mask] - start_row
        block.col_indices[:] = self.A_sparse.col_indices[mask]
        block.values[:] = self.A_sparse.values[mask]
        block.finalize()

        return block

    def _compute_exact_residual(self, x: np.ndarray, A_diagonal: List[float], b: np.ndarray) -> np.ndarray:
        """
        计算精确残差 r = b - A*x

        Args:
            x: 当前解向量
            A_diagonal: 矩阵对角线（仅在随机矩阵时使用）
            b: 右端项

        Returns:
            残差向量 (numpy数组)
        """
        if self.A_matrix is not None:
            # 使用真实的稀疏矩阵计算 A*x
            Ax_np = self.A_matrix.dot(x)
            Ax = Ax_np
        else:
            # 使用对角线近似计算 A*x（兼容随机矩阵）
            Ax = np.array(A_diagonal) * x

        # 计算 r = b - A*x
        r = self.math_sim.vector_sub(b, Ax)
        return r

    def _extract_state_features(self, sub_p: np.ndarray, iteration: int) -> List[float]:
        """
        提取状态特征向量

        Args:
            sub_p: tile 对应的子向量
            iteration: 当前 CG 迭代次数

        Returns:
            状态特征向量
        """
        features = []

        # 原始子向量
        if hasattr(sub_p, 'tolist'):
            features.extend(sub_p.tolist())
        else:
            features.extend(sub_p)

        # 迭代索引（归一化）
        norm_iter = iteration / self.max_iter
        features.append(norm_iter)

        # 标准化
        if self.normalize_state:
            # 对向量部分进行标准化（保留统计特征的尺度）
            vector_part = features[:len(sub_p)]
            if vector_part:
                vec_mean = sum(vector_part) / len(vector_part)
                vec_std = math.sqrt(sum((v - vec_mean)**2 for v in vector_part) / len(vector_part))
                if vec_std > 0:
                    features[:len(sub_p)] = [(v - vec_mean) / vec_std for v in vector_part]

        return features
    
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
    
    def _measure_fp64_baseline_time(self):
        """
        测量 fp64 精度的 SpMV 计算时间作为基准
        """
        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize

        # 创建全 fp64 精度的动作数组（动作 0 对应 fp64）
        fp64_actions = np.zeros(num_tiles, dtype=np.int32)

        # 预热10次
        for _ in range(10):
            _ = bsr_spmv_mixed(
                self.bsr_data, fp64_actions, self.bsr_indices, self.bsr_indptr,
                self.p, self.bsr_R, self.bsr_C, device="cuda"
            )

        # 正式测量多次取平均
        num_measurements = 30
        total_time = 0.0

        for _ in range(num_measurements):
            time_start = time.time()
            Ap_baseline = bsr_spmv_mixed(
                self.bsr_data, fp64_actions, self.bsr_indices, self.bsr_indptr,
                self.p, self.bsr_R, self.bsr_C, device="cuda"
            )
            time_end = time.time()
            total_time += (time_end - time_start) * 1000  # 转换为毫秒

        self.fp64_baseline_time = total_time / num_measurements
        print(f"fp64 基准时间已测量: {self.fp64_baseline_time:.2f} ms")

    
    def get_state_features(self, p: np.ndarray, iteration: int) -> List[float]:
        """
        获取状态特征向量
        """
        # 返回所有 tiles 的状态
        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize
        state = []
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tilesize
            end_idx = start_idx + tilesize
            sub_p = self.p[start_idx:end_idx]
            # 如果最后一个 tile 不足 tilesize，则补0
            if len(sub_p) < tilesize:
                sub_p = np.concatenate([sub_p, np.zeros(tilesize - len(sub_p))])
            tile_state = self._extract_state_features(sub_p, self.current_iteration)
            state.extend(tile_state)
        return state

    def reset(self, seed: Optional[int] = None) -> List[float]:
        """
        重置环境，开始新的 CG 求解 episode

        Args:
            seed: 随机种子

        Returns:
            初始状态特征向量
        """
        # 在重置之前，保存当前episode的信息（如果有数据）
        if (hasattr(self, 'residual_tracker') and self.residual_tracker.residual_history and
            (self.episode_cpt_cost > 0 or self.step_rewards)):
            self.last_episode_info = self.get_episode_info()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 重置状态
        self.current_iteration = 0
        self.current_tile_idx = 0
        self.episode_done = False

        # 生成新的问题（可能更新matrix_size）
        A_diagonal, self.b = self._generate_problem()
        self.A_diagonal = A_diagonal  # 缓存对角线元素
        
        self.b_norm = self.math_sim.vector_norm(self.b)

        # CG 初始化（确保使用更新后的matrix_size）
        self.x = np.ones(self.matrix_size)  # x0 = 1
        self.r = self._compute_exact_residual(self.x, A_diagonal, self.b)  # r0 = b - A*x0
        self.p = self.r.copy()  # p0 = r0

        # 记录初始残差
        initial_residual_norm = self.math_sim.vector_norm(self.r)
        self.residual_tracker.reset()
        self.residual_tracker.record_residual(initial_residual_norm)

        # 重置轨迹跟踪
        self.episode_cpt_cost = 0.0
        self.tile_actions = []
        self.step_rewards = []

        # 重置最后一步信息
        self.last_reward = 0.0
        self.last_info = {}
        
        # # 测量 fp64 基准时间（如果尚未测量）
        # if self.fp64_baseline_time is None and self.bsr_data is not None:
        #     self._measure_fp64_baseline_time()

        # 重置性能统计
        self.performance_stats = {
            'total_time': 0.0,
            'matrix_extraction_time': 0.0,
            'spmv_time': 0.0,
            'cg_math_time': 0.0,
            'iteration_count': 0,
            'cache_hits': self.spmv_sim._cache_hit_count if hasattr(self.spmv_sim, '_cache_hit_count') else 0,
            'cache_misses': self.spmv_sim._cache_miss_count if hasattr(self.spmv_sim, '_cache_miss_count') else 0
        }
        
        return self.get_state_features(self.p, self.current_iteration)

    def step(self, actions: List[int]) -> Tuple[List[float], float, bool, Dict]:
        """
        执行一步：为当前迭代的所有 tiles 选择精度，完成一次完整的 CG 迭代

        Args:
            actions: 精度选择动作列表，每个元素对应一个 tile 的精度 (0-5)

        Returns:
            (next_state, reward, done, info)
        """
        if self.episode_done:
            raise RuntimeError("Episode 已结束，请调用 reset()")

        iteration_start_time = time.time()

        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize  # 向上取整计算 tile 数量

        # 验证 actions 的长度
        if len(actions) != num_tiles:
            raise ValueError(f"actions 长度 {len(actions)} 与 tile 数量 {num_tiles} 不匹配")

        # 记录当前迭代的所有 tile 动作
        self.tile_actions = actions.copy()

        # 初始化 Ap 向量（使用numpy数组提高性能）
        self.Ap = np.zeros(self.matrix_size, dtype=np.float64)

        spmv_start_time = time.time()

        # 使用 TileLang BSR SpMV kernel 进行混合精度计算
        if self.bsr_data is None:
            raise RuntimeError("BSR 数据未初始化")

        # 将 actions 转换为 numpy 数组
        actions_np = np.array(actions, dtype=np.int32)

        # 使用 TileLang kernel 执行 SpMV 计算
        # time_start = time.time()
        self.Ap = bsr_spmv_mixed(
            self.bsr_data, actions_np, self.bsr_indices, self.bsr_indptr,
            self.p, self.bsr_R, self.bsr_C, device="cuda"
        )

        # 裁剪 self.Ap 到实际矩阵大小，去掉 padding 的计算结果
        if len(self.Ap) > self.matrix_size:
            self.Ap = self.Ap[:self.matrix_size]
        
        # time_end = time.time()
        # iteration_compute_time = (time_end - time_start)*1000
        # print(f"TileLang BSR SpMV 计算时间: {iteration_compute_time:.2f} ms")

            
            
        # 真实计算总成本（相对于 fp64 基准时间的百分比）
        # if self.fp64_baseline_time is not None and self.fp64_baseline_time > 0:
        #     iteration_cpt_cost = (iteration_compute_time / self.fp64_baseline_time) * 100  # 百分比
        #     print(f"相对于 fp64 基准时间的计算成本: {iteration_cpt_cost:.1f}%")
        # else:
        #     iteration_cpt_cost = iteration_compute_time  # 如果基准时间不可用，使用绝对时间
        # 累加到 episode 总成本

        # 计算每个 tile 的计算成本，并累加得到迭代总成本
        iteration_cpt_cost = 0.0
        for action in actions:
            # action为int，对应精度编号，查表获得成本
            action_name = self._get_precision_name(action)
            cost = self.precision_cost_table.get(action_name, 1.0)
            iteration_cpt_cost += cost
        self.episode_cpt_cost += iteration_cpt_cost

        spmv_end_time = time.time()
        self.performance_stats['spmv_time'] += (spmv_end_time - spmv_start_time)
        # print(f"spmv_time: {(spmv_end_time - spmv_start_time)*1000:.3f} ms")

        # 完成一次 CG 迭代
        cg_math_start_time = time.time()
        converged, residual_norm, iter_done = self._complete_cg_iteration()
        cg_math_end_time = time.time()
        self.performance_stats['cg_math_time'] += (cg_math_end_time - cg_math_start_time)

        # 计算当前迭代的奖励
        iteration_reward = self.w1 * (-math.log(residual_norm/self.b_norm, 10)) - self.w2 * (iteration_cpt_cost/num_tiles) + self.w3 * converged

        if self.current_iteration % 10 == 0 or converged:
            print("")
            print("================================================")
            print(f"当前迭代次数: {self.current_iteration}")
            print(f"当前迭代残差: {residual_norm}")
            print(f"当前迭代每tile平均计算成本: {iteration_cpt_cost/num_tiles}")
            print(f"当前迭代是否收敛: {converged}")
            print(f"残差下降奖励: {self.w1 * (-math.log(residual_norm/self.b_norm, 10))}")
            print(f"计算成本奖励: {-self.w2 * (iteration_cpt_cost/num_tiles)}")
            print(f"收敛奖励: {self.w3 * converged}")
            print(f"总奖励: {iteration_reward}")
            # 统计每种精度选择的数量
            from collections import Counter
            precisions_to_test = [
                ('fp64', 0),
                ('fp32', 1),
                ('fp16', 2),
                ('fp8', 3)
            ]
            precision_code_to_name = {code: name for name, code in precisions_to_test}
            precision_counts = Counter(int(a) for a in actions)
            print("每种精度选择数量:")
            for precision_code, count in sorted(precision_counts.items()):
                precision_name = precision_code_to_name.get(precision_code, f"未知({precision_code})")
                print(f"  精度 {precision_name}: {count} 个")

        # 检查是否结束
        done = iter_done
        if not iter_done:
            # 开始新的迭代
            self.current_iteration += 1
            if self.current_iteration >= self.max_iter:
                done = True

        self.step_rewards.append(iteration_reward)

        # 更新性能统计
        iteration_end_time = time.time()
        self.performance_stats['total_time'] += (iteration_end_time - iteration_start_time)
        self.performance_stats['iteration_count'] += 1
        if hasattr(self.spmv_sim, '_cache_hit_count'):
            self.performance_stats['cache_hits'] = self.spmv_sim._cache_hit_count
            self.performance_stats['cache_misses'] = self.spmv_sim._cache_miss_count

        # 标记episode结束
        if done:
            self.episode_done = True

        # 准备下一个状态
        if not done:
            # 为下一个迭代的所有 tiles 提取状态
            next_state = self.get_state_features(self.p, self.current_iteration)
        else:
            # episode结束，返回重置后的状态
            # 注意：在评估模式下，pfrl会在评估钩子之后才调用reset，
            # 所以评估钩子应该能在reset之前获取episode信息
            next_state = self.reset()

        info = {
            'iteration': self.current_iteration,
            'iteration_cost': iteration_cpt_cost,
            'episode_cost': self.episode_cpt_cost,
            'tile_actions': self.tile_actions.copy(),
            'converged': converged,
            'residual_norm': residual_norm,
            'performance_stats': self.performance_stats.copy()
        }

        # 保存最后一步信息供钩子访问
        self.last_reward = iteration_reward
        self.last_info = info.copy()

        return next_state, iteration_reward, done, info


    def _complete_cg_iteration(self) -> Tuple[float, bool]:
        """
        完成一次完整的 CG 迭代

        Returns:
            (iteration_reward, done) - 迭代奖励和是否结束
        """
        # 计算 alpha = (r^T r) / (p^T Ap)
        r_dot_r = self.math_sim.vector_dot(self.r, self.r)
        p_dot_Ap = self.math_sim.vector_dot(self.p, self.Ap)

        # if p_dot_Ap <= 1e-20:  # 使用更小的阈值来检测数值问题
        #     # Ap 与 p 不正交或数值不稳定，算法发散
        #     return -self.w3, True

        alpha = r_dot_r / p_dot_Ap

        # 添加数值稳定性检查：防止alpha过大导致的数值爆炸
        # if abs(alpha) > 1e6:
        #     # alpha过大，可能是数值不稳定
        #     return -self.w3, True

        # 更新解: x = x + alpha * p
        alpha_p = self.math_sim.vector_scale(alpha, self.p)
        self.x = self.math_sim.vector_saxpy(alpha, self.p, self.x)

        # 更新残差: r = r - alpha * Ap
        alpha_Ap = self.math_sim.vector_scale(alpha, self.Ap)
        self.r = self.math_sim.vector_sub(self.r, alpha_Ap)

        # 计算残差范数并记录
        residual_norm = self.math_sim.vector_norm(self.r)
        self.residual_tracker.record_residual(residual_norm)

        # 检查收敛
        converged = residual_norm < self.stop_tol

        # 计算 beta = (r^T r) / (old_r_dot_r)
        old_r_dot_r = r_dot_r
        new_r_dot_r = residual_norm ** 2
        if old_r_dot_r > 1e-20:  # 使用更小的阈值
            beta = new_r_dot_r / old_r_dot_r
            # 防止beta过大
            if abs(beta) > 1e4:
                beta = 1e4 * (1.0 if beta > 0 else -1.0)
        else:
            beta = 0.0

        # 更新搜索方向: p = r + beta * p
        beta_p = self.math_sim.vector_scale(beta, self.p)
        self.p = self.math_sim.vector_add(self.r, beta_p)

        # 重置 Ap 为下一次迭代
        self.Ap = None

        done = converged or (self.current_iteration >= self.max_iter - 1)

        return converged, residual_norm, done

    def get_action_space_size(self) -> int:
        """获取动作空间大小"""
        return self.action_space_n

    def get_state_dim(self) -> int:
        """获取状态维度"""
        return self.state_dim

    def get_episode_info(self) -> Dict:
        """
        获取当前 episode 的完整信息

        Returns:
            episode 统计信息字典
        """
        # 如果当前episode数据已被重置，尝试从last_episode_info获取
        if (not hasattr(self, 'residual_tracker') or 
            not self.residual_tracker.residual_history or
            (self.episode_cpt_cost == 0 and not self.step_rewards)):
            if hasattr(self, 'last_episode_info') and self.last_episode_info:
                return self.last_episode_info.copy()

        convergence_info = self.residual_tracker.get_convergence_info()

        # 确保residual_history包含所有记录的残差值
        residual_history = convergence_info.get('residual_history', [])
        if not residual_history and convergence_info.get('initial_residual') is not None:
            # 如果history为空但initial_residual存在，至少包含初始残差
            residual_history = [convergence_info['initial_residual']]
            if convergence_info.get('final_residual') is not None:
                residual_history.append(convergence_info['final_residual'])

        # 计算收敛状态：如果final_residual存在且小于容忍度，则收敛
        final_residual = convergence_info.get('final_residual')
        converged = 1 if (final_residual is not None and final_residual < self.stop_tol) else 0

        # 确保total_cost和avg_tile_cost正确计算
        total_cost = self.episode_cpt_cost if self.episode_cpt_cost > 0 else 0.0
        num_steps = len(self.step_rewards) if self.step_rewards else 1
        avg_tile_cost = total_cost / max(1, num_steps)

        return {
            'iterations': convergence_info.get('iterations', 0),
            'final_residual': final_residual,
            'initial_residual': convergence_info.get('initial_residual'),
            'convergence_ratio': convergence_info.get('convergence_ratio'),
            'converged': converged,  # 使用0/1而不是False/True
            'total_cost': total_cost,
            'avg_tile_cost': avg_tile_cost,
            'residual_history': residual_history,
            'step_rewards': self.step_rewards.copy() if self.step_rewards else []
        }