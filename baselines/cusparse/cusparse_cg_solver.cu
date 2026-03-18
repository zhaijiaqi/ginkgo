#include "cusparse_cg_solver.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/time.h>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUSPARSE_ERROR(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS_ERROR(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " << err << std::endl; \
            exit(1); \
        } \
    } while(0)

CGSolverResult solve_cg_cusparse_from_csr(
    int m, int n, int nnz,
    const std::vector<int>& csr_ptr,
    const std::vector<int>& csr_colidx,
    const std::vector<double>& csr_val,
    const std::vector<double>* b,
    const std::vector<double>* x0,
    const CGSolverConfig& config
) {
    CGSolverResult result;
    result.converged = false;
    result.iterations = 0;
    result.final_residual = 0.0;
    
    // 初始化性能统计字段（只保留时间占比）
    result.spmv_time_ratio = 0.0;
    result.dot_time_ratio = 0.0;
    result.axpy_time_ratio = 0.0;
    
    // 初始化 cuSPARSE 和 cuBLAS
    cusparseHandle_t cusparse_handle = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    CHECK_CUSPARSE_ERROR(cusparseCreate(&cusparse_handle));
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));
    
    // 分配 GPU 内存
    int* d_csr_ptr = nullptr;
    int* d_csr_colidx = nullptr;
    double* d_csr_val = nullptr;
    double* d_x = nullptr;
    double* d_r = nullptr;
    double* d_p = nullptr;
    double* d_mu = nullptr;  // Ap
    double* d_b = nullptr;
    double* d_temp = nullptr;  // 临时向量，用于向量操作
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_csr_ptr, (m + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_csr_colidx, nnz * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_csr_val, nnz * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_r, m * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_p, m * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_mu, m * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, m * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp, m * sizeof(double)));
    
    // 复制矩阵数据到 GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr_ptr, csr_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr_colidx, csr_colidx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_csr_val, csr_val.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

    // 创建 cuSPARSE 矩阵描述符
    // 注意：较新的 cuSPARSE 版本不允许传入 NULL 指针，因此必须在分配完设备内存后创建
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE_ERROR(cusparseCreateCsr(
        &matA,
        m, n, nnz,
        d_csr_ptr, d_csr_colidx, d_csr_val,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    ));

    // 创建向量描述符（用于 SpMV）
    cusparseDnVecDescr_t vecX, vecMu;

    // 创建向量描述符
    CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F));
    CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecMu, m, d_mu, CUDA_R_64F));
    
    // 初始化 b（必须提供）
    if (b == nullptr) {
        std::cerr << "Error: b vector must be provided, cannot be nullptr" << std::endl;
        exit(1);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b->data(), m * sizeof(double), cudaMemcpyHostToDevice));
    
    // 计算 b_norm（使用 cuBLAS nrm2）
    double b_norm = 0.0;
    CHECK_CUBLAS_ERROR(cublasDnrm2(cublas_handle, m, d_b, 1, &b_norm));
    
    // 初始化 x0（如果未提供，使用零向量）
    if (x0 != nullptr) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_x, x0->data(), n * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CHECK_CUDA_ERROR(cudaMemset(d_x, 0, n * sizeof(double)));
    }
    
    // 计算初始残差 r = b - A*x0
    // 首先计算 mu = A*x0
    const double alpha = 1.0;
    const double beta = 0.0;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    
    // 获取缓冲区大小
    CHECK_CUSPARSE_ERROR(cusparseSpMV_bufferSize(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecMu,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize
    ));
    
    if (bufferSize > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&dBuffer, bufferSize));
    }
    
    // 执行 SpMV
    CHECK_CUSPARSE_ERROR(cusparseSpMV(
        cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecMu,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer
    ));
    
    // r = b - mu (使用 cuBLAS: r = b + (-1.0) * mu)
    CHECK_CUDA_ERROR(cudaMemcpy(d_r, d_b, m * sizeof(double), cudaMemcpyDeviceToDevice));
    const double minus_one = -1.0;
    CHECK_CUBLAS_ERROR(cublasDaxpy(cublas_handle, m, &minus_one, d_mu, 1, d_r, 1));
    
    // 初始化 p = r (使用 cuBLAS copy)
    CHECK_CUBLAS_ERROR(cublasDcopy(cublas_handle, m, d_r, 1, d_p, 1));
    
    // 计算初始残差范数（使用 cuBLAS nrm2）
    double initial_residual = 0.0;
    CHECK_CUBLAS_ERROR(cublasDnrm2(cublas_handle, m, d_r, 1, &initial_residual));
    result.residual_history.push_back(initial_residual);
    
    // 检查初始收敛
    if (initial_residual / b_norm < config.tol) {
        result.converged = true;
        result.iterations = 0;
        result.final_residual = initial_residual;
        result.solve_time_ms = 0.0;

        // 复制解
        result.solution.resize(n);
        CHECK_CUDA_ERROR(cudaMemcpy(result.solution.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
        
        // 清理
        if (dBuffer != nullptr) {
            CHECK_CUDA_ERROR(cudaFree(dBuffer));
        }
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
        CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecMu));
        CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
        CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparse_handle));
        CHECK_CUDA_ERROR(cudaFree(d_csr_ptr));
        CHECK_CUDA_ERROR(cudaFree(d_csr_colidx));
        CHECK_CUDA_ERROR(cudaFree(d_csr_val));
        CHECK_CUDA_ERROR(cudaFree(d_x));
        CHECK_CUDA_ERROR(cudaFree(d_r));
        CHECK_CUDA_ERROR(cudaFree(d_p));
        CHECK_CUDA_ERROR(cudaFree(d_mu));
        CHECK_CUDA_ERROR(cudaFree(d_b));
        CHECK_CUDA_ERROR(cudaFree(d_temp));
        CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
        
        return result;
    }
    
    if (config.verbose) {
        std::cout << "Starting CG: initial residual = " << initial_residual 
                  << ", b_norm = " << b_norm << std::endl;
    }
    
    std::vector<double> spmv_times;
    
    // 详细的算子时间统计（只统计 spmv, dot product, axpy）
    std::vector<double> dot_times;
    std::vector<double> axpy_times;
    
    // 创建 p 的向量描述符（用于迭代中的 SpMV）
    cusparseDnVecDescr_t vecP;
    CHECK_CUSPARSE_ERROR(cusparseCreateDnVec(&vecP, m, d_p, CUDA_R_64F));
    
    // CG 迭代
    int iters_done = 0;          // 实际执行的迭代次数（用于正确上报）
    bool breakdown = false;      // 数值 breakdown（例如 mu_dot_p 过小）
    for (int iter = 0; iter < config.max_iter; iter++) {
        // μ = Ap (SpMV)
        struct timeval spmv_start, spmv_end;
        gettimeofday(&spmv_start, NULL);
        
        CHECK_CUSPARSE_ERROR(cusparseSpMV(
            cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecP,
            &beta,
            vecMu,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            dBuffer
        ));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        gettimeofday(&spmv_end, NULL);
        double spmv_time = (spmv_end.tv_sec - spmv_start.tv_sec) * 1000.0 + 
                          (spmv_end.tv_usec - spmv_start.tv_usec) / 1000.0;
        spmv_times.push_back(spmv_time);
        
        // CG step: 计算 dot products 和更新向量（不统计整体时间，只统计各个算子）
        
        // 计算 r·r (使用 cuBLAS dot)
        struct timeval dot_start, dot_end;
        gettimeofday(&dot_start, NULL);
        double r_dot_r = 0.0;
        CHECK_CUBLAS_ERROR(cublasDdot(cublas_handle, m, d_r, 1, d_r, 1, &r_dot_r));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        gettimeofday(&dot_end, NULL);
        double dot_time1 = (dot_end.tv_sec - dot_start.tv_sec) * 1000.0 + 
                          (dot_end.tv_usec - dot_start.tv_usec) / 1000.0;
        dot_times.push_back(dot_time1);
        
        // 计算 μ·p (使用 cuBLAS dot)
        gettimeofday(&dot_start, NULL);
        double mu_dot_p = 0.0;
        CHECK_CUBLAS_ERROR(cublasDdot(cublas_handle, m, d_mu, 1, d_p, 1, &mu_dot_p));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        gettimeofday(&dot_end, NULL);
        double dot_time2 = (dot_end.tv_sec - dot_start.tv_sec) * 1000.0 + 
                          (dot_end.tv_usec - dot_start.tv_usec) / 1000.0;
        dot_times.push_back(dot_time2);
        
        // aj = (r·r) / (μ·p)
        double aj = 0.0;
        if (fabs(mu_dot_p) > 1e-307) {
            aj = r_dot_r / mu_dot_p;
        } else {
            std::cerr << "Warning: mu_dot_p is too small, stopping iteration" << std::endl;
            // 这里属于 breakdown：没有完成本次迭代更新
            breakdown = true;
            iters_done = iter;  // 已完成 iter 次（从0开始）
            // 用当前 r_dot_r 推一个残差范数，便于上报
            result.final_residual = (r_dot_r > 0.0) ? sqrt(r_dot_r) : result.residual_history.back();
            break;
        }
        
        // x = x + aj * p (使用 cuBLAS axpy)
        // 注意：x 是 n 维，p 是 m 维
        // 为了与原始实现保持一致，如果 m != n，只更新 x 的前 min(m, n) 个元素
        int x_update_size = (m < n) ? m : n;
        struct timeval axpy_start, axpy_end;
        gettimeofday(&axpy_start, NULL);
        CHECK_CUBLAS_ERROR(cublasDaxpy(cublas_handle, x_update_size, &aj, d_p, 1, d_x, 1));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        gettimeofday(&axpy_end, NULL);
        double axpy_time1 = (axpy_end.tv_sec - axpy_start.tv_sec) * 1000.0 + 
                           (axpy_end.tv_usec - axpy_start.tv_usec) / 1000.0;
        axpy_times.push_back(axpy_time1);
        
        // r_new = r - aj * μ (使用 cuBLAS: temp = mu, temp = aj * temp, r_new = r - temp)
        // 注意：copy 和 scal 操作不统计时间，只统计 axpy
        CHECK_CUBLAS_ERROR(cublasDcopy(cublas_handle, m, d_mu, 1, d_temp, 1));
        CHECK_CUBLAS_ERROR(cublasDscal(cublas_handle, m, &aj, d_temp, 1));
        CHECK_CUBLAS_ERROR(cublasDcopy(cublas_handle, m, d_r, 1, d_r, 1));  // 保存原始 r 到 d_r
        
        const double minus_one = -1.0;
        gettimeofday(&axpy_start, NULL);
        CHECK_CUBLAS_ERROR(cublasDaxpy(cublas_handle, m, &minus_one, d_temp, 1, d_r, 1));  // r = r - temp (r_new)
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        gettimeofday(&axpy_end, NULL);
        double axpy_time2 = (axpy_end.tv_sec - axpy_start.tv_sec) * 1000.0 + 
                           (axpy_end.tv_usec - axpy_start.tv_usec) / 1000.0;
        axpy_times.push_back(axpy_time2);
        
        // 计算 ||r_new||² (使用 cuBLAS nrm2)
        // 注意：nrm2 操作不统计时间
        double r_new_norm = 0.0;
        CHECK_CUBLAS_ERROR(cublasDnrm2(cublas_handle, m, d_r, 1, &r_new_norm));
        double r_new_dot_r_new = r_new_norm * r_new_norm;
        
        // βj = ||r_new||² / ||r||²
        double beta_j = 0.0;
        if (r_dot_r > 1e-307) {
            beta_j = r_new_dot_r_new / r_dot_r;
        } else {
            beta_j = 0.0;
        }
        
        // p = r_new + βj * p = r + βj * p (使用 cuBLAS scal + axpy)
        // 注意：scal 操作不统计时间，只统计 axpy
        CHECK_CUBLAS_ERROR(cublasDscal(cublas_handle, m, &beta_j, d_p, 1));  // p = βj * p
        
        const double one = 1.0;
        gettimeofday(&axpy_start, NULL);
        CHECK_CUBLAS_ERROR(cublasDaxpy(cublas_handle, m, &one, d_r, 1, d_p, 1));  // p = r + p
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        gettimeofday(&axpy_end, NULL);
        double axpy_time3 = (axpy_end.tv_sec - axpy_start.tv_sec) * 1000.0 + 
                           (axpy_end.tv_usec - axpy_start.tv_usec) / 1000.0;
        axpy_times.push_back(axpy_time3);

        // 注意：cuBLAS 向量操作是异步的；为了让计时反映实际 GPU 执行时间，这里需要同步
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        
        // 计算新的残差范数（用于收敛判断，已经在上面计算了）
        result.residual_history.push_back(r_new_norm);
        
        if (config.verbose && (iter + 1) % 10 == 0) {
            std::cout << "Iteration " << (iter + 1) << ": residual = " << r_new_norm << std::endl;
        }
        
        // 收敛判断（CPU 端，不计入 GPU 时间）
        bool converged = (r_new_norm / b_norm) < config.tol;
        if (converged) {
            result.converged = true;
            result.iterations = iter + 1;
            result.final_residual = r_new_norm;
            break;
        }

        iters_done = iter + 1;
    }
    
    // 总 GPU 时间：仅由各 kernel 的计时累加得到（只统计 spmv, dot product, axpy）
    double total_gpu_time = std::accumulate(spmv_times.begin(), spmv_times.end(), 0.0) +
                            std::accumulate(dot_times.begin(), dot_times.end(), 0.0) +
                            std::accumulate(axpy_times.begin(), axpy_times.end(), 0.0);
    
    if (!result.converged) {
        // 不要无条件写成 max_iter：可能是 breakdown 或提前退出
        if (result.iterations == 0) {
            result.iterations = iters_done;
        }
        if (!result.residual_history.empty() && result.final_residual == 0.0) {
            result.final_residual = result.residual_history.back();
        }
        if (breakdown && config.verbose) {
            std::cerr << "CG stopped early due to breakdown (mu_dot_p too small). "
                      << "iters_done=" << iters_done << std::endl;
        }
    }
    
    // 复制解
    result.solution.resize(n);
    CHECK_CUDA_ERROR(cudaMemcpy(result.solution.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Benchmark 统计（只计算和存储时间占比）
    // 保存时间向量用于计算占比
    result.spmv_times = spmv_times;
    result.dot_times = dot_times;
    result.axpy_times = axpy_times;
    
    // 计算时间占比（只统计 spmv, dot product, axpy）
    if (total_gpu_time > 0) {
        double spmv_total = std::accumulate(spmv_times.begin(), spmv_times.end(), 0.0);
        double dot_total = std::accumulate(dot_times.begin(), dot_times.end(), 0.0);
        double axpy_total = std::accumulate(axpy_times.begin(), axpy_times.end(), 0.0);

        result.spmv_time_ratio = (spmv_total / total_gpu_time) * 100.0;
        result.dot_time_ratio = (dot_total / total_gpu_time) * 100.0;
        result.axpy_time_ratio = (axpy_total / total_gpu_time) * 100.0;
    } else {
        result.spmv_time_ratio = 0.0;
        result.dot_time_ratio = 0.0;
        result.axpy_time_ratio = 0.0;
    }

    result.solve_time_ms = total_gpu_time;

    // 清理
    if (dBuffer != nullptr) {
        CHECK_CUDA_ERROR(cudaFree(dBuffer));
    }
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE_ERROR(cusparseDestroyDnVec(vecMu));
    CHECK_CUSPARSE_ERROR(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE_ERROR(cusparseDestroy(cusparse_handle));
    CHECK_CUDA_ERROR(cudaFree(d_csr_ptr));
    CHECK_CUDA_ERROR(cudaFree(d_csr_colidx));
    CHECK_CUDA_ERROR(cudaFree(d_csr_val));
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_r));
    CHECK_CUDA_ERROR(cudaFree(d_p));
    CHECK_CUDA_ERROR(cudaFree(d_mu));
    CHECK_CUDA_ERROR(cudaFree(d_b));
        CHECK_CUDA_ERROR(cudaFree(d_temp));
        CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
        
        return result;
}

