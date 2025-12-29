"""
CG Environment - PPO-Guided Mixed-Precision CG 求解环境
提供完整的 RL 环境接口，用于训练 PPO 代理学习精度选择策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import math
import random
import csv
import os
import time
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.cg_math_simulator import CGResidualTracker
from simulator.spmv_block_simulator import SpMVBlockSimulator, SparseMatrix

from kernels.spmv_kernels import bsr_spmv_mixed, bsr_spmv_mixed_prequant
from kernels.coo2bsr_kernels import coo2bsr
from kernels.fused_cg_kernel import fused_cg_step

# Optional: BCSC + pre-quantization utilities (new path for future kernel refactor)
if TYPE_CHECKING:
    from kernels.bcsc_prequant import BCSCMatrix as _BCSCMatrix
else:
    _BCSCMatrix = Any  # runtime fallback

try:
    from kernels.bcsc_prequant import (
        build_bcsc_from_coo,
        quantize_bcsc_tiles,
        spmv_bcsc_mixed_ref_prequant,
    )

    BCSC_AVAILABLE = True
except Exception:
    BCSC_AVAILABLE = False

try:
    # Optional: TileLang BCSC kernel (CUDA)
    from kernels.bcsc_spmv_kernels import bcsc_spmv_mixed_prequant

    BCSC_TL_AVAILABLE = True
except Exception:
    BCSC_TL_AVAILABLE = False


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
        self.w4 = reward_config.get('w4', 100.0)  # 发散惩罚权重
        self.w5 = reward_config.get('w5', 0.3)  # 高精度鼓励权重
        # 环境参数
        self.normalize_state = config.get('normalize_state', True)
        self.verbose = bool(config.get('verbose', True))
        # 默认启用 torch 常驻状态（去掉 numpy 路径）
        self.use_torch_state = bool(config.get('use_torch_state', True))
        self.torch_device = str(config.get('torch_device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.torch_dtype = torch.float64

        # BCSC + pre-quantization (optional)
        self.use_bcsc_prequant = bool(config.get('use_bcsc_prequant', False))
        # Where to store BCSC tensors. Default: same as torch_device (so later kernels can reuse).
        self.bcsc_device = str(config.get('bcsc_device', self.torch_device))
        # Use TensorCore path for bf16 actions (TileLang MMA). Default off.
        self.bcsc_use_tensorcore = bool(config.get('bcsc_use_tensorcore', False))
        # Optional self-check after building BCSC (can be expensive on huge matrices)
        self.bcsc_self_check = bool(config.get('bcsc_self_check', False))
        # SpMV implementation:
        # - "bsr": current TileLang BSR kernel (default; runtime A+X quant as in existing kernel)
        # - "bsr_prequant": TileLang BSR kernel using pre-quantized A tiles + per-tile a_scale (x still quantized online)
        # - "bcsc_ref": BCSC reference SpMV (debug/correctness; slower, but no TileLang dependency)
        # - "bcsc_prequant": TileLang BCSC kernel using pre-quantized A tiles (requires CUDA)
        self.spmv_impl = str(config.get('spmv_impl', 'bcsc_prequant'))

        # BSR A-tile pre-quantization (optional; keeps BSR format but moves A scaling to preprocessing)
        self.use_bsr_prequant = bool(config.get('use_bsr_prequant', False))
        self.bsr_prequant_device = str(config.get('bsr_prequant_device', self.torch_device))

        # 初始化模拟器
        spmv_config = {
            'tilesize': config.get('tilesize', 32),
            'random_seed': config.get('random_seed', 42),
        }
        # Only forward precision_cost_table if user provided a real dict.
        # Passing None breaks SpMVBlockSimulator's expectations.
        pct = config.get('precision_cost_table', None)
        if isinstance(pct, dict):
            spmv_config['precision_cost_table'] = pct
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
        self.b_cpu = None  # 右端项（CPU numpy 缓存，用于 _generate_problem 复用）
        self.b_norm = None  # 右端项范数
        self.initial_residual_norm = None  # 初始残差

        # 矩阵数据缓存
        self.A_matrix = None  # scipy csr_matrix
        self.A_sparse = None  # 我们的 SparseMatrix 格式
        self.matrix_loaded = False  # 标记矩阵是否已加载

        # 预分组的tile数据（性能优化）
        self.tile_blocks = None  # 按tile分组的矩阵块列表，每个元素是SparseMatrix
        
        # BSR 格式数据（用于 TileLang kernel）
        self.bsr_data = None  # BSR 数据块
        self.bsr_indices = None  # BSR 列索引
        self.bsr_indptr = None  # BSR 行指针
        self.bsr_R = None  # BSR 行块大小
        self.bsr_C = None  # BSR 列块大小

        # Torch 版本的 BSR 数据缓存（避免每步 numpy->torch / H2D）
        self.bsr_data_t = None
        self.bsr_indices_t = None
        self.bsr_indptr_t = None

        # Optional: BSR pre-quant buffers (same tile order as bsr_data)
        self.bsr_fp32_q_t = None
        self.bsr_a_scale_fp32_t = None
        self.bsr_bf16_q_t = None
        self.bsr_a_scale_bf16_t = None

        # BCSC (column-compressed blocks) + pre-quantized tiles
        self.bcsc: Optional[_BCSCMatrix] = None
        
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
        
        # 精度成本表（仅保留 fp64/fp32/bf16；bf16 成本与旧 fp16 一致）
        raw_cost = self.config.get('precision_cost_table', {
            'fp64': 1.0,
            'fp32': 0.5,
            'bf16': 0.25,
        })
        # 兼容旧配置：若还提供 fp16/fp8，则迁移到 bf16 并丢弃多余精度
        if 'bf16' not in raw_cost and 'fp16' in raw_cost:
            raw_cost = dict(raw_cost)
            raw_cost['bf16'] = raw_cost.get('fp16', 0.25)
        # 强制只保留三种
        self.precision_cost_table = {
            'fp64': float(raw_cost.get('fp64', 1.0)),
            'fp32': float(raw_cost.get('fp32', 0.5)),
            'bf16': float(raw_cost.get('bf16', 0.25)),
        }

        # 动作空间：3 种精度选择（0=fp64, 1=fp32, 2=bf16）
        self.action_space_n = 3
      
        # 重置环境
        self.reset()

        # 状态空间维度（必须与 reset()/step() 返回的 obs 长度一致）
        # 当前实现：观测 = 所有 tile 的 p 子向量拼接；每个 tile 固定 tilesize 维（最后一个 tile 不足会补 0）。
        tilesize = int(self.spmv_sim.tilesize)
        self.tile_state_dim = tilesize
        self.num_tiles = (int(self.matrix_size) + tilesize - 1) // tilesize  # 向上取整计算 tile 数量
        self.state_dim = int(self.num_tiles * self.tile_state_dim)

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

    def _generate_problem(self) -> Tuple:
        """
        生成 CG 测试问题 Ax = b

        如果指定了矩阵名称，则从文件中加载真实的矩阵，
        否则生成一个简单的对角占优矩阵来确保收敛。

        Returns:
            (A_matrix, b) - 稀疏矩阵和右端项
        """
        if self.matrix_loaded:
            # 矩阵已加载，直接返回缓存的数据
            return self.A_matrix, self.b_cpu

        if self.matrix_name is not None:
            # 加载真实的矩阵
            try:
                self.A_matrix = self._load_matrix_market(self.matrix_name)
                self.matrix_size = self.A_matrix.shape[0]
                self.matrix_nnz = self.A_matrix.nnz

                # 生成右端项 b（A 的列值相加）
                b = np.array(self.A_matrix.sum(axis=0)).flatten()

                print(f"Loaded matrix '{self.matrix_name}' with size {self.matrix_size}x{self.matrix_size}, nnz {self.matrix_nnz}")

                # 将矩阵转换为我们的 SparseMatrix 格式
                self.A_sparse = self._csr_to_sparse_matrix(self.A_matrix)

                # 预先计算tile块（性能优化）
                self._precompute_tile_blocks()
                
                # 转换为 BSR 格式（用于 TileLang kernel）
                self.bsr_data, self.bsr_indices, self.bsr_indptr, self.bsr_R, self.bsr_C = self._convert_matrix_to_bsr()
                self._maybe_cache_bsr_torch()

                # Optional: build BCSC + pre-quantize A tiles
                self._maybe_build_bcsc_prequant()

                # Optional: build BSR pre-quant buffers
                self._maybe_build_bsr_prequant()

                self.matrix_loaded = True
                self.b_cpu = b  # 将 b 缓存为 CPU numpy，便于下次复用
                self.b = b

            except Exception as e:
                print(f"Failed to load matrix '{self.matrix_name}': {e}")
                print("Falling back to random matrix generation")
                self.matrix_name = None
                return self._generate_random_problem()
        else:
            return self._generate_random_problem()

        return self.A_matrix, self.b

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

        # 生成右端项 b，b 初始化为：每个A对应列元素之和（对于对角矩阵就是对角线元素的拷贝）
        # 随机对角矩阵：b 直接取对角线（避免依赖 self.A_matrix）
        b = np.array(A_diagonal, dtype=np.float64)

        # 为随机矩阵创建稀疏矩阵表示（对角矩阵）
        self.A_sparse = SparseMatrix(self.matrix_size, self.matrix_size, self.matrix_size)
        for i in range(self.matrix_size):
            self.A_sparse.add_element(i, i, A_diagonal[i])
        self.A_sparse.finalize()

        # 预先计算tile块（性能优化）
        self._precompute_tile_blocks()

        # 转换为 BSR 格式（用于 TileLang kernel）
        self.bsr_data, self.bsr_indices, self.bsr_indptr, self.bsr_R, self.bsr_C = self._convert_matrix_to_bsr()
        self._maybe_cache_bsr_torch()

        # Optional: build BCSC + pre-quantize A tiles
        self._maybe_build_bcsc_prequant()

        # Optional: build BSR pre-quant buffers
        self._maybe_build_bsr_prequant()

        self.matrix_loaded = True
        self.b_cpu = b
        return A_diagonal, b

    def _maybe_build_bcsc_prequant(self):
        """
        Build BCSC representation (column-compressed blocks) and pre-quantize A tiles.

        This is a preprocessing step meant to be reused by future BCSC-based TileLang kernels.
        It does NOT change the current SpMV path (still uses BSR kernel in step()).
        """
        self.bcsc = None
        if not self.use_bcsc_prequant:
            return
        if not BCSC_AVAILABLE:
            raise RuntimeError(
                "use_bcsc_prequant=True but kernels.bcsc_prequant import failed. "
                "Please ensure dependencies (torch) are available."
            )
        if self.A_sparse is None:
            raise RuntimeError("矩阵未加载或转换失败 (A_sparse is None)")

        tilesize = int(self.spmv_sim.tilesize)
        dev = torch.device(self.bcsc_device)

        # Build from our internal COO (SparseMatrix)
        row = np.asarray(self.A_sparse.row_indices, dtype=np.int64)
        col = np.asarray(self.A_sparse.col_indices, dtype=np.int64)
        val = np.asarray(self.A_sparse.values, dtype=np.float64)
        shape = (int(self.A_sparse.rows), int(self.A_sparse.cols))

        bcsc = build_bcsc_from_coo(row, col, val, shape, tilesize=tilesize, device=dev)
        bcsc = quantize_bcsc_tiles(bcsc)
        # If we intend to run the TileLang BCSC kernel, ensure payloads are on CUDA once.
        if self.spmv_impl == "bcsc_prequant":
            if not torch.cuda.is_available():
                raise RuntimeError("spmv_impl=bcsc_prequant requires CUDA, but torch.cuda.is_available() is False.")
            bcsc = bcsc.to("cuda")
        self.bcsc = bcsc

        if self.verbose:
            print(
                f"[BCSC] built: n_br={bcsc.n_br} n_bc={bcsc.n_bc} nnzb={bcsc.nnzb} device={self.bcsc_device}"
            )

        if self.bcsc_self_check:
            self._bcsc_self_check()

    def _bcsc_self_check(self, *, seed: int = 0):
        """
        Optional correctness sanity check for BCSC reference SpMV (actions all-0).

        This is mainly for development/debugging and can be slow for very large matrices.
        """
        if self.bcsc is None:
            return
        bcsc = self.bcsc
        rng = np.random.default_rng(seed)
        x_np = rng.standard_normal((bcsc.N,), dtype=np.float64)
        x_t = torch.as_tensor(x_np, dtype=torch.float64, device=bcsc.A_fp64.device)
        actions0 = torch.zeros((bcsc.n_bc,), dtype=torch.int32, device=bcsc.A_fp64.device)
        y_bcsc = spmv_bcsc_mixed_ref_prequant(bcsc, actions0, x_t).detach().cpu().numpy()

        # Compare with scipy dense-ish reference when available; otherwise fall back to COO sum (slow).
        if self.A_matrix is not None and SCIPY_AVAILABLE:
            y_ref = (self.A_matrix @ x_np).astype(np.float64)
        else:
            # COO reference
            y_ref = np.zeros((bcsc.M,), dtype=np.float64)
            y_ref[self.A_sparse.row_indices] += self.A_sparse.values * x_np[self.A_sparse.col_indices]

        max_abs = float(np.max(np.abs(y_bcsc - y_ref))) if y_ref.size > 0 else 0.0
        if not np.allclose(y_bcsc, y_ref, atol=1e-9, rtol=1e-9):
            raise AssertionError(f"BCSC self-check failed: max_abs_diff={max_abs:.3e}")
        if self.verbose:
            print(f"[BCSC] self-check passed (actions=all0), max_abs_diff={max_abs:.3e}")

    def _maybe_cache_bsr_torch(self):
        """
        若启用 use_torch_state，则把 BSR 结构一次性搬到 torch_device，避免每步重复 H2D/D2H。
        """
        if not self.use_torch_state:
            self.bsr_data_t = None
            self.bsr_indices_t = None
            self.bsr_indptr_t = None
            return
        if self.bsr_data is None or self.bsr_indices is None or self.bsr_indptr is None:
            return
        dev = torch.device(self.torch_device)
        # data: float64, indices/indptr: int32
        self.bsr_data_t = torch.as_tensor(np.asarray(self.bsr_data, dtype=np.float64), device=dev, dtype=torch.float64)
        self.bsr_indices_t = torch.as_tensor(np.asarray(self.bsr_indices, dtype=np.int32), device=dev, dtype=torch.int32)
        self.bsr_indptr_t = torch.as_tensor(np.asarray(self.bsr_indptr, dtype=np.int32), device=dev, dtype=torch.int32)

    def _maybe_build_bsr_prequant(self):
        """
        Pre-quantize BSR tiles (A only) in preprocessing stage.

        This keeps the existing BSR structure (indices/indptr), but stores:
          - bsr_fp32_q_t + bsr_a_scale_fp32_t
          - bsr_bf16_q_t + bsr_a_scale_bf16_t

        so SpMV can skip per-tile A dynamic range scanning at runtime.
        """
        # reset (avoid stale buffers if matrix changes)
        self.bsr_fp32_q_t = None
        self.bsr_a_scale_fp32_t = None
        self.bsr_bf16_q_t = None
        self.bsr_a_scale_bf16_t = None

        if not self.use_bsr_prequant:
            return
        if self.bsr_data_t is None:
            # Ensure BSR is cached as torch first
            self._maybe_cache_bsr_torch()
        if self.bsr_data_t is None:
            raise RuntimeError("use_bsr_prequant=True but bsr_data_t is None")

        dev = torch.device(self.bsr_prequant_device)
        A = self.bsr_data_t.to(device=dev, dtype=torch.float64)

        # Match kernel constants
        qmax = torch.tensor(1.8405e19, dtype=torch.float64, device=dev)
        eps = torch.tensor(1e-12, dtype=torch.float64, device=dev)

        a_max_abs = torch.amax(torch.abs(A), dim=(1, 2))  # [nnzb]
        a_scale = qmax / (a_max_abs + eps)  # [nnzb]

        A_scaled = A * a_scale[:, None, None]
        A_clipped = torch.clamp(A_scaled, min=-qmax.item(), max=qmax.item())

        self.bsr_fp32_q_t = A_clipped.to(torch.float32)
        self.bsr_a_scale_fp32_t = a_scale.to(torch.float64)
        self.bsr_bf16_q_t = A_clipped.to(torch.bfloat16)
        self.bsr_a_scale_bf16_t = a_scale.to(torch.float64)

        if self.verbose:
            print(
                f"[BSR-PREQUANT] built: nnzb={int(A.shape[0])} R={self.bsr_R} C={self.bsr_C} device={self.bsr_prequant_device}"
            )

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

    def _compute_exact_residual(self, x: np.ndarray, A_matrix: csr_matrix, b: np.ndarray) -> np.ndarray:
        """
        计算精确残差 r = b - A*x

        Args:
            x: 当前解向量
            A_matrix: 矩阵
            b: 右端项

        Returns:
            残差向量 (numpy数组)
        """

        # 兼容：x 可能是 torch.Tensor
        if torch.is_tensor(x):
            x = x.detach().to("cpu").numpy()
        Ax_np = A_matrix.dot(x)
        Ax = Ax_np
        # 计算 r = b - A*x
        r = (np.asarray(b) - np.asarray(Ax)).astype(np.float64)
        return r

    def _extract_state_features(self, sub_p: np.ndarray) -> List[float]:
        """
        提取状态特征向量

        Args:
            sub_p: tile 对应的子向量

        Returns:
            状态特征向量
        """
        features: List[float] = []

        # 原始子向量
        if hasattr(sub_p, 'tolist'):
            features.extend(sub_p.tolist())
        else:
            features.extend(sub_p)
        # 标准化
        if self.normalize_state:
            arr = np.asarray(features, dtype=np.float32)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std < 1e-12:
                std = 1e-12
            arr = (arr - mean) / std
            features = arr.tolist()

        return features
    
    def _get_precision_name(self, action: int) -> str:
        """
        将动作转换为精度名称

        Args:
            action: 精度动作 (0-2)

        Returns:
            精度名称字符串

        Raises:
            ValueError: 如果动作无效
        """
        precision_map = {
            0: 'fp64',
            1: 'fp32',
            2: 'bf16',
        }

        if action not in precision_map:
            raise ValueError(f"无效的精度动作: {action}，必须在 0-2 范围内")

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

    
    def get_state_features(self, p: np.ndarray) -> List[float]:
        """
        获取状态特征向量
        """
        # 若 p 是 torch.Tensor（可能在 GPU），先一次性搬到 CPU，避免每个 tile 单独触发拷贝
        if torch.is_tensor(self.p):
            p_arr = self.p.detach().to(device="cpu").numpy()
        else:
            p_arr = self.p
        # 返回所有 tiles 的状态
        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize
        state = []
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tilesize
            end_idx = start_idx + tilesize
            sub_p = p_arr[start_idx:end_idx]
            # 如果最后一个 tile 不足 tilesize，则补0
            if len(sub_p) < tilesize:
                sub_p = np.concatenate([sub_p, np.zeros(tilesize - len(sub_p))])
            tile_state = self._extract_state_features(sub_p)
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
        self.A_matrix, b_cpu = self._generate_problem()

        if not self.use_torch_state:
            raise RuntimeError("当前版本已移除 numpy 状态路径，请启用 use_torch_state=True（默认）")

        dev = torch.device(self.torch_device)
        b_t = torch.as_tensor(np.asarray(b_cpu, dtype=np.float64), device=dev, dtype=self.torch_dtype)
        # 默认也把 b 常驻为 torch（减少后续类型分支）
        self.b = b_t
        self.b_cpu = np.asarray(b_cpu, dtype=np.float64)
        self.b_norm = float(torch.linalg.vector_norm(b_t, ord=2).item())

        # CG 初始化：x0=0, r0=b, p0=r0（reset 时 x 恒为 0，因此等价于 exact residual）
        self.x = torch.zeros((self.matrix_size,), device=dev, dtype=self.torch_dtype)
        self.r = b_t.clone()
        self.p = self.r.clone()
        self.initial_residual_norm = float(torch.linalg.vector_norm(self.r, ord=2).item())
        self.residual_tracker.reset()
        self.residual_tracker.record_residual(self.initial_residual_norm)

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
        
        # matrix_size 可能在 _generate_problem() 后变化，确保维度元信息与观测一致
        tilesize = int(self.spmv_sim.tilesize)
        self.tile_state_dim = tilesize
        self.num_tiles = (int(self.matrix_size) + tilesize - 1) // tilesize
        self.state_dim = int(self.num_tiles * self.tile_state_dim)

        return self.get_state_features(self.p)

    def step(self, actions: List[int]) -> Tuple[List[float], float, bool, Dict]:
        """
        执行一步：为当前迭代的所有 tiles 选择精度，完成一次完整的 CG 迭代

        Args:
            actions: 精度选择动作列表，每个元素对应一个 tile 的精度 (0-2)

        Returns:
            (next_state, reward, done, info)
        """
        if self.episode_done:
            raise RuntimeError("Episode 已结束，请调用 reset()")

        iteration_start_time = time.time()

        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize  # 向上取整计算 tile 数量

        # 验证 actions 的长度
        # For BCSC implementations, actions are per block-column (n_bc), not per row-block (num_tiles)
        # For square matrices, num_tiles == n_bc, but we need to handle the mismatch
        if self.spmv_impl in ("bcsc_ref", "bcsc_prequant"):
            if self.bcsc is not None:
                expected_len = self.bcsc.n_bc
                if len(actions) != expected_len:
                    raise ValueError(
                        f"BCSC actions 长度 {len(actions)} 与 block-column 数量 {expected_len} 不匹配 "
                        f"(num_tiles={num_tiles}, n_bc={expected_len})"
                    )
        else:
            if len(actions) != num_tiles:
                raise ValueError(f"actions 长度 {len(actions)} 与 tile 数量 {num_tiles} 不匹配")

        # 记录当前迭代的所有 tile 动作
        self.tile_actions = actions.copy()

        # 初始化 Ap 向量：直接使用 torch 输出
        self.Ap = None

        spmv_start_time = time.time()

        dev = torch.device(self.torch_device)
        actions_t = torch.as_tensor(actions, dtype=torch.int32, device=dev)
        if self.spmv_impl == "bcsc_ref":
            # print("-----------use bcsc_ref----------")
            if self.bcsc is None:
                # Try build on-demand (in case matrix loaded before flags changed)
                self._maybe_build_bcsc_prequant()
            if self.bcsc is None:
                raise RuntimeError("spmv_impl=bcsc_ref requires BCSC prequant data, but bcsc is None.")

            # Ensure tensors on same device as BCSC
            bcsc_dev = self.bcsc.A_fp64.device
            actions_bc = actions_t.to(device=bcsc_dev)
            p_bc = self.p.to(device=bcsc_dev)
            try:
                self.Ap = spmv_bcsc_mixed_ref_prequant(self.bcsc, actions_bc, p_bc, trim_to_M=False)
                if self.Ap is None:
                    raise RuntimeError("spmv_bcsc_mixed_ref_prequant returned None")
                # Move Ap back to torch_device to match self.p, self.r, self.x
                if self.Ap.device != dev:
                    self.Ap = self.Ap.to(device=dev)
            except Exception as e:
                raise RuntimeError(f"spmv_bcsc_mixed_ref_prequant failed: {e}") from e
        elif self.spmv_impl == "bcsc_prequant":
            # print("-----------use bcsc_prequant----------")
            if not self.use_bcsc_prequant:
                # Allow selecting spmv_impl without pre-building buffers (build on-demand)
                self.use_bcsc_prequant = True
                self._maybe_build_bcsc_prequant()
            if self.bcsc is None:
                raise RuntimeError("spmv_impl=bcsc_prequant requires BCSC prequant data, but bcsc is None.")
            if not BCSC_TL_AVAILABLE:
                raise RuntimeError(
                    "spmv_impl=bcsc_prequant requires kernels.bcsc_spmv_kernels (TileLang CUDA) but import failed."
                )
            if not torch.cuda.is_available():
                raise RuntimeError("spmv_impl=bcsc_prequant requires CUDA, but torch.cuda.is_available() is False.")

            # Run on CUDA; keep BCSC on CUDA persistently (avoid per-step H2D copies).
            if self.bcsc.A_fp64.device.type != "cuda":
                self.bcsc = self.bcsc.to("cuda")
            actions_bc = actions_t.to(device="cuda")
            p_bc = self.p.to(device="cuda")
            self.Ap = bcsc_spmv_mixed_prequant(
                self.bcsc,
                actions_bc,
                p_bc,
                device="cuda",
                use_tensorcore_bf16=self.bcsc_use_tensorcore,
                return_torch=True,
            )
            # Move Ap back to torch_device to match self.p, self.r, self.x
            if self.Ap.device != dev:
                self.Ap = self.Ap.to(device=dev)
        elif self.spmv_impl == "bsr_prequant":
            # print("-----------use bsr_prequant----------")
            if not self.use_bsr_prequant:
                # Allow selecting spmv_impl without pre-building buffers (build on-demand)
                self.use_bsr_prequant = True
                self._maybe_build_bsr_prequant()
            if (
                self.bsr_fp32_q_t is None
                or self.bsr_a_scale_fp32_t is None
                or self.bsr_bf16_q_t is None
                or self.bsr_a_scale_bf16_t is None
            ):
                raise RuntimeError("spmv_impl=bsr_prequant requires pre-quant BSR buffers, but they are missing.")
            self.Ap = bsr_spmv_mixed_prequant(
                self.bsr_data_t if self.bsr_data_t is not None else self.bsr_data,
                self.bsr_fp32_q_t,
                self.bsr_a_scale_fp32_t,
                self.bsr_bf16_q_t,
                self.bsr_a_scale_bf16_t,
                actions_t,
                self.bsr_indices_t if self.bsr_indices_t is not None else self.bsr_indices,
                self.bsr_indptr_t if self.bsr_indptr_t is not None else self.bsr_indptr,
                self.p,
                self.bsr_R,
                self.bsr_C,
                device=self.torch_device,
                return_torch=True,
            )
        else:
            # print("-----------use bsr----------")
            # Default: use TileLang BSR SpMV kernel
            if self.bsr_data is None:
                raise RuntimeError("BSR 数据未初始化")
            self.Ap = bsr_spmv_mixed(
                self.bsr_data_t if self.bsr_data_t is not None else self.bsr_data,
                actions_t,
                self.bsr_indices_t if self.bsr_indices_t is not None else self.bsr_indices,
                self.bsr_indptr_t if self.bsr_indptr_t is not None else self.bsr_indptr,
                self.p,
                self.bsr_R,
                self.bsr_C,
                device=self.torch_device,
                return_torch=True,
            )
        if self.Ap is None:
            raise RuntimeError(f"SpMV implementation '{self.spmv_impl}' did not set self.Ap (still None after execution)")
        if int(self.Ap.shape[0]) > self.matrix_size:
            self.Ap = self.Ap[: self.matrix_size]
        ap_shape = tuple(int(x) for x in self.Ap.shape)

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
        residual_norm, prev_residual_norm, converged, done, diverged = self._complete_cg_iteration()
        cg_math_end_time = time.time()
        self.performance_stats['cg_math_time'] += (cg_math_end_time - cg_math_start_time)

        # # ===== 新的 reward 计算逻辑 =====
        # # 1. 残差下降奖励 (R_progress)：step-wise log 残差下降
        # progress_reward = self.w1 * math.log(prev_residual_norm / residual_norm)

        # # 2. 计算代价惩罚 (R_cost)：鼓励使用低精度
        # cost_penalty = -self.w2 * (iteration_cpt_cost / num_tiles)

        # # 3. 收敛终止奖励 (R_converge)：只在真正收敛时给予
        # convergence_reward = self.w3 * converged

        # # 4. 数值失败惩罚 (R_failure)：检测多种失败情况
        # failure_penalty = 0.0
        # if residual_norm > 2.0 * self.initial_residual_norm or residual_norm > prev_residual_norm * 1.2: # 持续的残差上升惩罚
        #     failure_penalty = -self.w4
        # if not np.isfinite(residual_norm): # 检查 NaN/Inf
        #     failure_penalty = -self.w4 * 5
             
        # # 5. 发散惩罚
        # diverged_penalty = -self.max_iter * 10 if diverged else 0

        # # 计算总奖励
        # iteration_reward = progress_reward + cost_penalty + convergence_reward + failure_penalty + diverged_penalty

        
        progress_reward = self.w1 * math.log(prev_residual_norm / residual_norm)
        cost_penalty = -self.w2 * (iteration_cpt_cost / num_tiles)
        convergence_reward = self.w3 * converged
        
        iteration_reward = progress_reward + cost_penalty + convergence_reward


        if self.current_iteration % 100 == 0 or done:
            if not self.verbose:
                # 静默模式：跳过大量调试输出（用于基准测试避免 I/O 干扰）
                pass
            else:
                print("")
                print("================================================")
                print(f"当前迭代次数: {self.current_iteration}")
                print(f"当前迭代残差: {residual_norm}")
                print(f"当前相对残差：{residual_norm/self.b_norm}")
                print(f"当前迭代每tile平均计算成本: {iteration_cpt_cost/num_tiles}")
                print(f"当前迭代是否收敛: {converged}")
                print(f"1.残差下降奖励 (R_progress): {progress_reward}")
                print(f"2.计算代价惩罚 (R_cost):     {cost_penalty}")
                # print(f"3.数值失败惩罚 (R_failure):  {failure_penalty}")
                # print(f"4.发散惩罚     (R_diverged):{diverged_penalty}")
                print(f"5.收敛终止奖励 (R_converge): {convergence_reward}")
                print(f"总奖励:                 : {iteration_reward}")
                # 统计每种精度选择的数量
                from collections import Counter
                precisions_to_test = [
                    ('fp64', 0),
                    ('fp32', 1),
                    ('bf16', 2),
                ]
                precision_code_to_name = {code: name for name, code in precisions_to_test}
                precision_counts = Counter(int(a) for a in actions)
                print("每种精度选择数量:")
                for precision_code, count in sorted(precision_counts.items()):
                    precision_name = precision_code_to_name.get(precision_code, f"未知({precision_code})")
                    print(f"  精度 {precision_name}: {count} 个")

        # 检查是否结束
        if not done:
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
            next_state = self.get_state_features(self.p)
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
            'residual_norm_relative': residual_norm/self.b_norm,
            'spmv_impl': self.spmv_impl,
            'Ap_shape': ap_shape,
            'performance_stats': self.performance_stats.copy()
        }

        # 保存最后一步信息供钩子访问
        self.last_reward = iteration_reward
        self.last_info = info.copy()

        return next_state, iteration_reward, done, info


    def _complete_cg_iteration(self) -> Tuple[float, float, bool, bool, bool]:
        """
        完成一次完整的 CG 迭代（使用融合内核版本）

        Returns:
            (residual_norm, prev_residual_norm, converged, done, diverged)
        """
        if not (torch.is_tensor(self.r) and torch.is_tensor(self.p) and torch.is_tensor(self.x) and torch.is_tensor(self.Ap)):
            raise RuntimeError("当前版本默认使用 torch 常驻状态：x/r/p/Ap 必须为 torch.Tensor。")

        r_t = self.r
        p_t = self.p
        x_t = self.x
        Ap_t = self.Ap

        # prev_residual_norm = ||r||
        prev_residual_norm = float(torch.linalg.vector_norm(r_t, ord=2).item())

        # 使用融合内核完成 CG 迭代步骤
        # fused_cg_step 会修改 r, p, x 的值，并返回 stats: [||r||², ||r_new||², aj]
        # 其中 mu 对应 Ap
        if not torch.cuda.is_available():
            raise RuntimeError("融合 CG 内核需要 CUDA，但 CUDA 不可用")
        
        # 确保所有 tensor 都在 CUDA 上
        if r_t.device.type != "cuda":
            raise RuntimeError(f"融合 CG 内核需要 tensor 在 CUDA 上，但 r 在 {r_t.device} 上")
        
        device = str(r_t.device)
        stats = fused_cg_step(r_t, Ap_t, p_t, x_t, device=device, block_size=256)
        
        # 提取统计信息
        r_dot_r_old = float(stats[0].item())  # ||r||² (旧)
        r_dot_r_new = float(stats[1].item())  # ||r_new||² (新)
        aj = float(stats[2].item())  # alpha (aj)
        
        # 检查发散：如果 aj == 0，说明 p_dot_Ap <= 1e-307（fused kernel 在分母过小时返回 0.0）
        diverged = False
        if abs(aj) < 1e-300:  # 接近 0，表示分母过小（fused kernel 返回 0.0 当 p_dot_Ap <= 1e-307）
            print("⚠️  WARNING: Ap is not orthogonal to p or numerical instability detected. The algorithm has diverged.")
            diverged = True
            done = True
            converged = False
            current_residual_norm = prev_residual_norm
            return current_residual_norm, prev_residual_norm, converged, done, diverged

        # residual_norm = ||r_new|| = sqrt(||r_new||²)
        residual_norm = float(np.sqrt(r_dot_r_new))
        self.residual_tracker.record_residual(residual_norm)

        converged = (residual_norm / self.b_norm) < self.stop_tol

        # 重置 Ap 为下一次迭代
        self.Ap = None

        done = converged or (self.current_iteration >= self.max_iter - 1)

        # torch 常驻：保持 torch（fused_cg_step 已经原地修改了 r, p, x）
        self.x = x_t
        self.r = r_t
        self.p = p_t

        return residual_norm, prev_residual_norm, converged, done, diverged

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