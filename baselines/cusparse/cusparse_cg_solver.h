#ifndef CUSPARSE_CG_SOLVER_H
#define CUSPARSE_CG_SOLVER_H

#include <vector>
#include <string>

// CG 求解器结果结构（复用 cg_kernel 中的结构）
struct CGSolverResult {
    bool converged;
    int iterations;
    double final_residual;
    std::vector<double> residual_history;
    std::vector<double> solution;
    
    // Benchmark 统计（只保留时间占比）
    // 时间占比（只统计 spmv, dot product, axpy）
    double spmv_time_ratio;             // SpMV 时间占比
    double dot_time_ratio;              // dot product 时间占比
    double axpy_time_ratio;             // axpy 时间占比
    
    // 内部使用的时间记录（用于计算占比，不对外输出）
    std::vector<double> spmv_times;
    std::vector<double> dot_times;
    std::vector<double> axpy_times;

    // CG 求解时间（ms）：仅迭代中的 spmv/dot/axpy 之和，不含读 mtx、预处理、GPU 拷贝等
    double solve_time_ms = 0.0;
};

// CG 求解器配置
struct CGSolverConfig {
    int max_iter = 10000;
    double tol = 1e-10;
    bool verbose = false;
    int warmup = 3;           // Warmup 运行次数
    int benchmark_iters = 10; // Benchmark 运行次数
};

/**
 * 基于 cuSPARSE 和 cuBLAS 的 CG 求解器
 * 使用标准 CSR 矩阵格式
 * 
 * @param matrix_file: 矩阵文件路径 (.mtx)
 * @param config: CG 求解器配置
 * @param b: 右端向量（可选，如果为 nullptr 则使用 A * ones）
 * @param x0: 初始猜测（可选，如果为 nullptr 则使用零向量）
 * @return: CG 求解器结果
 */
CGSolverResult solve_cg_cusparse(
    const std::string& matrix_file,
    const CGSolverConfig& config = CGSolverConfig(),
    const std::vector<double>* b = nullptr,
    const std::vector<double>* x0 = nullptr
);

/**
 * 从已加载的 CSR 矩阵求解 CG
 * 
 * @param m, n: 矩阵维度
 * @param nnz: 非零元素数
 * @param csr_ptr, csr_colidx, csr_val: CSR 格式矩阵数据
 * @param b: 右端向量（可选，如果为 nullptr 则使用列和）
 * @param x0: 初始猜测（可选，如果为 nullptr 则使用零向量）
 * @param config: CG 求解器配置
 * @return: CG 求解器结果
 */
CGSolverResult solve_cg_cusparse_from_csr(
    int m, int n, int nnz,
    const std::vector<int>& csr_ptr,
    const std::vector<int>& csr_colidx,
    const std::vector<double>& csr_val,
    const std::vector<double>* b = nullptr,
    const std::vector<double>* x0 = nullptr,
    const CGSolverConfig& config = CGSolverConfig()
);

/**
 * 计算 b = A * ones（CPU 端实现）
 * 
 * @param matrix_file: 矩阵文件路径 (.mtx)
 * @param m: 输出参数，矩阵行数
 * @param n: 输出参数，矩阵列数
 * @return: 计算得到的 b 向量
 */
std::vector<double> compute_b_from_matrix(const std::string& matrix_file, int& m, int& n);

#endif // CUSPARSE_CG_SOLVER_H

