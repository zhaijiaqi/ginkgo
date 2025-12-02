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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.cg_math_simulator import CGMathSimulator, CGResidualTracker
from simulator.spmv_block_simulator import SpMVBlockSimulator, SparseMatrix

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

        # 轨迹跟踪
        self.episode_cpt_cost = 0.0
        self.tile_actions = []  # 当前迭代的所有 tile 动作
        self.step_rewards = []  # 每步奖励

        # 动作空间：6 种精度选择
        self.action_space_n = 6
      
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

    def _generate_problem(self) -> Tuple[List[float], List[float]]:
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

                # 提取对角线元素用于兼容现有代码
                A_diagonal = self.A_matrix.diagonal().tolist()

                # 生成右端项 b：A 的每一列元素之和
                b = self.A_matrix.sum(axis=1).A1.tolist()

                print(f"Loaded matrix '{self.matrix_name}' with size {self.matrix_size}x{self.matrix_size}")

                # 将矩阵转换为我们的 SparseMatrix 格式
                self.A_sparse = self._csr_to_sparse_matrix(self.A_matrix)

                self.matrix_loaded = True

            except Exception as e:
                print(f"Failed to load matrix '{self.matrix_name}': {e}")
                print("Falling back to random matrix generation")
                self.matrix_name = None
                return self._generate_random_problem()
        else:
            return self._generate_random_problem()

        return A_diagonal, b

    def _generate_random_problem(self) -> Tuple[List[float], List[float]]:
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
        b = [random.gauss(0, 1) for _ in range(self.matrix_size)]

        # 为随机矩阵创建稀疏矩阵表示（对角矩阵）
        self.A_sparse = SparseMatrix(self.matrix_size, self.matrix_size)
        for i in range(self.matrix_size):
            self.A_sparse.add_element(i, i, A_diagonal[i])

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
        sparse_mat = SparseMatrix(csr_mat.shape[0], csr_mat.shape[1])

        # 将 CSR 格式转换为 COO 格式并添加到 SparseMatrix
        csr_mat_coo = csr_mat.tocoo()
        for i, (row, col, val) in enumerate(zip(csr_mat_coo.row, csr_mat_coo.col, csr_mat_coo.data)):
            sparse_mat.add_element(row, col, val)

        return sparse_mat

    def _extract_matrix_block(self, start_row: int, end_row: int) -> SparseMatrix:
        """
        提取矩阵的一个行块

        Args:
            start_row: 开始行索引
            end_row: 结束行索引（不包含）

        Returns:
            该行块的 SparseMatrix
        """
        if self.A_sparse is None:
            raise RuntimeError("矩阵未加载或转换失败")

        block = SparseMatrix(end_row - start_row, self.A_sparse.cols)

        # 提取该行块的所有非零元素
        for row, col, val in zip(self.A_sparse.row_indices,
                                self.A_sparse.col_indices,
                                self.A_sparse.values):
            if start_row <= row < end_row:
                block.add_element(row - start_row, col, val)

        return block

    def _compute_exact_residual(self, x: List[float], A_diagonal: List[float], b: List[float]) -> List[float]:
        """
        计算精确残差 r = b - A*x

        Args:
            x: 当前解向量
            A_diagonal: 矩阵对角线（仅在随机矩阵时使用）
            b: 右端项

        Returns:
            残差向量
        """
        if self.A_matrix is not None:
            # 使用真实的稀疏矩阵计算 A*x
            x_np = np.array(x)
            Ax_np = self.A_matrix.dot(x_np)
            Ax = Ax_np.tolist()
        else:
            # 使用对角线近似计算 A*x（兼容随机矩阵）
            Ax = []
            for i in range(len(x)):
                ax_i = A_diagonal[i] * x[i]
                Ax.append(ax_i)

        # 计算 r = b - A*x
        r = self.math_sim.vector_sub(b, Ax)
        return r

    def _extract_state_features(self, sub_p: List[float], iteration: int) -> List[float]:
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
        features.extend(sub_p)

        # # 统计特征
        # l1_norm = self.math_sim.vector_norm(sub_p, p=1)
        # l2_norm = self.math_sim.vector_norm(sub_p, p=2)
        # max_abs = self.math_sim.vector_norm(sub_p, p=float('inf'))

        # features.extend([l1_norm, l2_norm, max_abs])

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
    
    def get_state_features(self, p: List[float], iteration: int) -> List[float]:
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
                sub_p += [0.0] * (tilesize - len(sub_p))
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
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 重置状态
        self.current_iteration = 0
        self.current_tile_idx = 0
        self.episode_done = False

        # 生成新的问题（可能更新matrix_size）
        A_diagonal, self.b = self._generate_problem()
        
        self.b_norm = self.math_sim.vector_norm(self.b)

        # CG 初始化（确保使用更新后的matrix_size）
        self.x = [0.0] * self.matrix_size  # x0 = 0
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

        tilesize = self.spmv_sim.tilesize
        num_tiles = (self.matrix_size + tilesize - 1) // tilesize  # 向上取整计算 tile 数量

        # 验证 actions 的长度
        if len(actions) != num_tiles:
            raise ValueError(f"actions 长度 {len(actions)} 与 tile 数量 {num_tiles} 不匹配")

        # 记录当前迭代的所有 tile 动作
        self.tile_actions = actions.copy()

        # 初始化 Ap 向量
        self.Ap = [0.0] * self.matrix_size
        iteration_cost = 0.0

        # 为当前迭代的所有 tiles 执行 SpMV 计算
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tilesize
            actual_tilesize = min(tilesize, self.matrix_size - start_idx)
            end_idx = start_idx + actual_tilesize

            # 提取当前 tile 的矩阵块
            matrix_block = self._extract_matrix_block(start_idx, end_idx)

            # 模拟该 tile 的 SpMV 计算
            action = actions[tile_idx]
            partial_result, cpt_cost = self.spmv_sim.simulate_spmv_block(matrix_block, self.p, action)

            # 累加到 Ap 向量
            for i in range(actual_tilesize):
                self.Ap[start_idx + i] += partial_result[i]

            # 累加计算成本
            iteration_cost += cpt_cost

        # 累加到 episode 总成本
        self.episode_cpt_cost += iteration_cost

        # 完成一次 CG 迭代
        converged, residual_norm, iter_done = self._complete_cg_iteration()

        # 计算当前迭代的奖励
        iteration_reward = self.w1 * (-math.log(residual_norm/self.b_norm, 10)) - self.w2 * (iteration_cost/num_tiles) + self.w3 * converged

        print("")
        print("================================================")
        print(f"当前迭代次数: {self.current_iteration}")
        print(f"当前迭代残差: {residual_norm}")
        print(f"当前迭代每tile平均计算成本: {iteration_cost/num_tiles}")
        print(f"当前迭代是否收敛: {converged}")
        print(f"残差下降奖励: {self.w1 * (-math.log(residual_norm/self.b_norm, 10))}")
        print(f"计算成本奖励: {-self.w2 * (iteration_cost/num_tiles)}")
        print(f"收敛奖励: {self.w3 * converged}")
        print(f"总奖励: {iteration_reward}")

        # 检查是否结束
        done = iter_done
        if not iter_done:
            # 开始新的迭代
            self.current_iteration += 1
            if self.current_iteration >= self.max_iter:
                done = True

        self.step_rewards.append(iteration_reward)

        # 准备下一个状态
        if not done:
            # 为下一个迭代的所有 tiles 提取状态
            next_state = self.get_state_features(self.p, self.current_iteration)
        else:
            next_state = []

        info = {
            'iteration': self.current_iteration,
            'iteration_cost': iteration_cost,
            'episode_cost': self.episode_cpt_cost,
            'tile_actions': self.tile_actions.copy(),
            'converged': converged,
            'residual_norm': residual_norm
        }

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
        convergence_info = self.residual_tracker.get_convergence_info()

        return {
            'iterations': convergence_info['iterations'],
            'final_residual': convergence_info['final_residual'],
            'initial_residual': convergence_info['initial_residual'],
            'convergence_ratio': convergence_info['convergence_ratio'],
            'converged': convergence_info['final_residual'] < self.stop_tol if convergence_info['final_residual'] else False,
            'total_cost': self.episode_cpt_cost,
            'avg_tile_cost': self.episode_cpt_cost / max(1, len(self.step_rewards)),
            'residual_history': convergence_info['residual_history'],
            'step_rewards': self.step_rewards.copy()
        }