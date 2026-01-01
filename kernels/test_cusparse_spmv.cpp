#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// CUDA 12.0+ uses new cuSPARSE API (cusparseSpMV)
// Legacy API (cusparseDcsrmv/cusparseScsrmv) is deprecated
#define USE_NEW_CUSPARSE_API

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUSPARSE(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while(0)

// Simple Matrix Market reader
struct MatrixMarket {
    int nrows, ncols, nnz;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;
    
    bool load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open file: %s\n", filename.c_str());
            return false;
        }
        
        std::string line;
        bool header_read = false;
        bool symmetric = false;
        
        while (std::getline(file, line)) {
            // Skip comments
            if (line[0] == '%') {
                if (line.find("symmetric") != std::string::npos || 
                    line.find("hermitian") != std::string::npos) {
                    symmetric = true;
                }
                continue;
            }
            
            // Read dimensions
            if (!header_read) {
                std::istringstream iss(line);
                iss >> nrows >> ncols >> nnz;
                header_read = true;
                continue;
            }
            
            // Read data
            int row, col;
            double val;
            std::istringstream iss(line);
            iss >> row >> col >> val;
            
            // Convert from 1-based to 0-based indexing
            row--;
            col--;
            
            row_indices.push_back(row);
            col_indices.push_back(col);
            values.push_back(val);
            
            // Handle symmetric matrices
            if (symmetric && row != col) {
                row_indices.push_back(col);
                col_indices.push_back(row);
                values.push_back(val);
            }
        }
        
        // Update nnz if symmetric
        if (symmetric) {
            nnz = row_indices.size();
        }
        
        file.close();
        return true;
    }
};

// Convert COO to CSR
void coo_to_csr(const std::vector<int>& row_indices, const std::vector<int>& col_indices,
                const std::vector<double>& values, int nrows, int nnz,
                std::vector<int>& csr_row_ptr, std::vector<int>& csr_col_ind,
                std::vector<double>& csr_val) {
    // Create pairs for sorting
    std::vector<std::pair<int, int>> pairs(nnz);
    for (int i = 0; i < nnz; i++) {
        pairs[i] = {row_indices[i], i};
    }
    
    // Sort by row
    std::sort(pairs.begin(), pairs.end());
    
    // Build CSR
    csr_row_ptr.resize(nrows + 1, 0);
    csr_col_ind.resize(nnz);
    csr_val.resize(nnz);
    
    for (int i = 0; i < nnz; i++) {
        int idx = pairs[i].second;
        int row = row_indices[idx];
        csr_row_ptr[row + 1]++;
        csr_col_ind[i] = col_indices[idx];
        csr_val[i] = values[idx];
    }
    
    // Cumulative sum
    for (int i = 0; i < nrows; i++) {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }
}

template<typename T>
struct BenchmarkResult {
    double avg_time_ms;
    double throughput_gops;
};

// Forward declaration
template<typename T>
BenchmarkResult<T> benchmark_cusparse_spmv(
    cusparseHandle_t handle,
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_ind,
    const std::vector<double>& csr_val,
    const T* x_d,
    T* y_d,
    int nrows,
    int nnz,
    int n_warmup = 100,
    int n_iter = 10000,
    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT);

// Template specialization for double
template<>
BenchmarkResult<double> benchmark_cusparse_spmv<double>(
    cusparseHandle_t handle,
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_ind,
    const std::vector<double>& csr_val,
    const double* x_d,
    double* y_d,
    int nrows,
    int nnz,
    int n_warmup,
    int n_iter,
    cusparseSpMVAlg_t alg) {
    
    // Allocate device memory
    int* d_csr_row_ptr;
    int* d_csr_col_ind;
    double* d_csr_val;
    
    CHECK_CUDA(cudaMalloc(&d_csr_row_ptr, (nrows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_val, nnz * sizeof(double)));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_val, csr_val.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    // Set alpha and beta
    double alpha = 1.0;
    double beta = 0.0;
    
#ifdef USE_NEW_CUSPARSE_API
    // Use new cuSPARSE API (CUDA 12.0+)
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    
    // Create sparse matrix descriptor
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, nrows, nrows, nnz,
                                     d_csr_row_ptr, d_csr_col_ind, d_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    // Create dense vector descriptors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, nrows, const_cast<double*>(x_d), CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, nrows, y_d, CUDA_R_64F));
    
    // Get buffer size
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            alg, &bufferSize);
    
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                    alg, dBuffer));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                    alg, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUDA(cudaFree(dBuffer));
#else
    // Use legacy cuSPARSE API (CUDA < 12.0)
    cusparseMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        CHECK_CUSPARSE(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     nrows, nrows, nnz, &alpha, descr,
                                     d_csr_val, d_csr_row_ptr, d_csr_col_ind,
                                     x_d, &beta, y_d));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        CHECK_CUSPARSE(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     nrows, nrows, nnz, &alpha, descr,
                                     d_csr_val, d_csr_row_ptr, d_csr_col_ind,
                                     x_d, &beta, y_d));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
#endif
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / 1000.0 / n_iter;
    double avg_time_s = avg_time_ms / 1000.0;
    double throughput_gops = (2.0 * nnz / avg_time_s) / 1e9;
    
    CHECK_CUDA(cudaFree(d_csr_row_ptr));
    CHECK_CUDA(cudaFree(d_csr_col_ind));
    CHECK_CUDA(cudaFree(d_csr_val));
    
    return {avg_time_ms, throughput_gops};
}

// Template specialization for float
template<>
BenchmarkResult<float> benchmark_cusparse_spmv<float>(
    cusparseHandle_t handle,
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_ind,
    const std::vector<double>& csr_val,
    const float* x_d,
    float* y_d,
    int nrows,
    int nnz,
    int n_warmup,
    int n_iter,
    cusparseSpMVAlg_t alg) {
    
    // Convert matrix values to float
    std::vector<float> csr_val_float(csr_val.size());
    for (size_t i = 0; i < csr_val.size(); i++) {
        csr_val_float[i] = static_cast<float>(csr_val[i]);
    }
    
    // Allocate device memory
    int* d_csr_row_ptr;
    int* d_csr_col_ind;
    float* d_csr_val;
    
    CHECK_CUDA(cudaMalloc(&d_csr_row_ptr, (nrows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_val, nnz * sizeof(float)));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_val, csr_val_float.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    // Set alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;
    
#ifdef USE_NEW_CUSPARSE_API
    // Use new cuSPARSE API (CUDA 12.0+)
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    
    // Create sparse matrix descriptor
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, nrows, nrows, nnz,
                                     d_csr_row_ptr, d_csr_col_ind, d_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    // Create dense vector descriptors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, nrows, const_cast<float*>(x_d), CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, nrows, y_d, CUDA_R_32F));
    
    // Get buffer size
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                            alg, &bufferSize);
    
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    alg, dBuffer));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    alg, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUDA(cudaFree(dBuffer));
#else
    // Use legacy cuSPARSE API (CUDA < 12.0)
    cusparseMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        CHECK_CUSPARSE(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     nrows, nrows, nnz, &alpha, descr,
                                     d_csr_val, d_csr_row_ptr, d_csr_col_ind,
                                     x_d, &beta, y_d));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        CHECK_CUSPARSE(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     nrows, nrows, nnz, &alpha, descr,
                                     d_csr_val, d_csr_row_ptr, d_csr_col_ind,
                                     x_d, &beta, y_d));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
#endif
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / 1000.0 / n_iter;
    double avg_time_s = avg_time_ms / 1000.0;
    double throughput_gops = (2.0 * nnz / avg_time_s) / 1e9;
    
    CHECK_CUDA(cudaFree(d_csr_row_ptr));
    CHECK_CUDA(cudaFree(d_csr_col_ind));
    CHECK_CUDA(cudaFree(d_csr_val));
    
    return {avg_time_ms, throughput_gops};
}

// Template specialization for bfloat16
template<>
BenchmarkResult<__nv_bfloat16> benchmark_cusparse_spmv<__nv_bfloat16>(
    cusparseHandle_t handle,
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_ind,
    const std::vector<double>& csr_val,
    const __nv_bfloat16* x_d,
    __nv_bfloat16* y_d,
    int nrows,
    int nnz,
    int n_warmup,
    int n_iter,
    cusparseSpMVAlg_t alg) {
    
    // Convert matrix values to bfloat16
    std::vector<__nv_bfloat16> csr_val_bf16(csr_val.size());
    for (size_t i = 0; i < csr_val.size(); i++) {
        csr_val_bf16[i] = __float2bfloat16(static_cast<float>(csr_val[i]));
    }
    
    // Allocate device memory
    int* d_csr_row_ptr;
    int* d_csr_col_ind;
    __nv_bfloat16* d_csr_val;
    
    CHECK_CUDA(cudaMalloc(&d_csr_row_ptr, (nrows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_val, nnz * sizeof(__nv_bfloat16)));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_val, csr_val_bf16.data(), nnz * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    
    // Note: alpha and beta will be converted to float32 for computation
    // cuSPARSE uses float32 as compute type for bfloat16 operations
    
#ifdef USE_NEW_CUSPARSE_API
    // Use new cuSPARSE API (CUDA 12.0+)
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    
    // Create sparse matrix descriptor
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, nrows, nrows, nnz,
                                     d_csr_row_ptr, d_csr_col_ind, d_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16BF));
    
    // Create dense vector descriptors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, nrows, const_cast<__nv_bfloat16*>(x_d), CUDA_R_16BF));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, nrows, y_d, CUDA_R_16BF));
    
    // Get buffer size
    // Note: For bfloat16, cuSPARSE uses float32 as compute type
    // Matrix and vectors are bfloat16, but computation is done in float32
    float alpha_fp32 = 1.0f;
    float beta_fp32 = 0.0f;
    
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_fp32, matA, vecX, &beta_fp32, vecY, CUDA_R_32F,
                            alg, &bufferSize);
    
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_fp32, matA, vecX, &beta_fp32, vecY, CUDA_R_32F,
                                    alg, dBuffer));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_fp32, matA, vecX, &beta_fp32, vecY, CUDA_R_32F,
                                    alg, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUDA(cudaFree(dBuffer));
#else
    // Legacy API doesn't support bfloat16
    fprintf(stderr, "Error: bfloat16 requires CUDA 12.0+ with new cuSPARSE API\n");
    exit(1);
#endif
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / 1000.0 / n_iter;
    double avg_time_s = avg_time_ms / 1000.0;
    double throughput_gops = (2.0 * nnz / avg_time_s) / 1e9;
    
    CHECK_CUDA(cudaFree(d_csr_row_ptr));
    CHECK_CUDA(cudaFree(d_csr_col_ind));
    CHECK_CUDA(cudaFree(d_csr_val));
    
    return {avg_time_ms, throughput_gops};
}

// Template specialization for fp16 (half precision)
template<>
BenchmarkResult<__half> benchmark_cusparse_spmv<__half>(
    cusparseHandle_t handle,
    const std::vector<int>& csr_row_ptr,
    const std::vector<int>& csr_col_ind,
    const std::vector<double>& csr_val,
    const __half* x_d,
    __half* y_d,
    int nrows,
    int nnz,
    int n_warmup,
    int n_iter,
    cusparseSpMVAlg_t alg) {
    
    // Convert matrix values to fp16
    std::vector<__half> csr_val_fp16(csr_val.size());
    for (size_t i = 0; i < csr_val.size(); i++) {
        csr_val_fp16[i] = __float2half(static_cast<float>(csr_val[i]));
    }
    
    // Allocate device memory
    int* d_csr_row_ptr;
    int* d_csr_col_ind;
    __half* d_csr_val;
    
    CHECK_CUDA(cudaMalloc(&d_csr_row_ptr, (nrows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_col_ind, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csr_val, nnz * sizeof(__half)));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_col_ind, csr_col_ind.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_val, csr_val_fp16.data(), nnz * sizeof(__half), cudaMemcpyHostToDevice));
    
    // Note: alpha and beta will be converted to float32 for computation
    // cuSPARSE uses float32 as compute type for fp16 operations
    
#ifdef USE_NEW_CUSPARSE_API
    // Use new cuSPARSE API (CUDA 12.0+)
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    
    // Create sparse matrix descriptor
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, nrows, nrows, nnz,
                                     d_csr_row_ptr, d_csr_col_ind, d_csr_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    
    // Create dense vector descriptors
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, nrows, const_cast<__half*>(x_d), CUDA_R_16F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, nrows, y_d, CUDA_R_16F));
    
    // Get buffer size
    // Note: For fp16, cuSPARSE uses float32 as compute type
    // Matrix and vectors are fp16, but computation is done in float32
    float alpha_fp32 = 1.0f;
    float beta_fp32 = 0.0f;
    
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha_fp32, matA, vecX, &beta_fp32, vecY, CUDA_R_32F,
                            alg, &bufferSize);
    
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    // Warmup
    for (int i = 0; i < n_warmup; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_fp32, matA, vecX, &beta_fp32, vecY, CUDA_R_32F,
                                    alg, dBuffer));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_fp32, matA, vecX, &beta_fp32, vecY, CUDA_R_32F,
                                    alg, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUDA(cudaFree(dBuffer));
#else
    // Legacy API doesn't support fp16
    fprintf(stderr, "Error: fp16 requires CUDA 12.0+ with new cuSPARSE API\n");
    exit(1);
#endif
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / 1000.0 / n_iter;
    double avg_time_s = avg_time_ms / 1000.0;
    double throughput_gops = (2.0 * nnz / avg_time_s) / 1e9;
    
    CHECK_CUDA(cudaFree(d_csr_row_ptr));
    CHECK_CUDA(cudaFree(d_csr_col_ind));
    CHECK_CUDA(cudaFree(d_csr_val));
    
    return {avg_time_ms, throughput_gops};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx> [n_warmup] [n_iter]\n", argv[0]);
        fprintf(stderr, "Example: %s ~/data/matrix/matrix_name.mtx 10 100\n", argv[0]);
        return 1;
    }
    
    std::string matrix_file = argv[1];
    int n_warmup = (argc > 2) ? atoi(argv[2]) : 10;
    int n_iter = (argc > 3) ? atoi(argv[3]) : 100;
    
    // Load matrix
    MatrixMarket mm;
    if (!mm.load(matrix_file)) {
        return 1;
    }
    
    printf("[INFO] Matrix loaded:\n");
    printf("  File: %s\n", matrix_file.c_str());
    printf("  Size: %d x %d\n", mm.nrows, mm.ncols);
    printf("  Non-zeros: %d\n", mm.nnz);
    
    // Convert to CSR
    std::vector<int> csr_row_ptr, csr_col_ind;
    std::vector<double> csr_val;
    coo_to_csr(mm.row_indices, mm.col_indices, mm.values,
               mm.nrows, mm.nnz, csr_row_ptr, csr_col_ind, csr_val);
    
    // Initialize cuSPARSE
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    
    // Allocate vectors
    int n = mm.nrows;
    double* x_h = new double[n];
    double* y_h = new double[n];
    
    // Initialize x with random values
    for (int i = 0; i < n; i++) {
        x_h[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory for vectors
    double* x_d_fp64;
    double* y_d_fp64;
    float* x_d_fp32;
    float* y_d_fp32;
    __nv_bfloat16* x_d_bf16;
    __nv_bfloat16* y_d_bf16;
    __half* x_d_fp16;
    __half* y_d_fp16;
    
    CHECK_CUDA(cudaMalloc(&x_d_fp64, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&y_d_fp64, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&x_d_fp32, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y_d_fp32, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&x_d_bf16, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&y_d_bf16, n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&x_d_fp16, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&y_d_fp16, n * sizeof(__half)));
    
    // Copy x to device
    CHECK_CUDA(cudaMemcpy(x_d_fp64, x_h, n * sizeof(double), cudaMemcpyHostToDevice));
    
    std::vector<float> x_h_fp32(n);
    std::vector<__nv_bfloat16> x_h_bf16(n);
    std::vector<__half> x_h_fp16(n);
    for (int i = 0; i < n; i++) {
        x_h_fp32[i] = static_cast<float>(x_h[i]);
        x_h_bf16[i] = __float2bfloat16(static_cast<float>(x_h[i]));
        x_h_fp16[i] = __float2half(static_cast<float>(x_h[i]));
    }
    CHECK_CUDA(cudaMemcpy(x_d_fp32, x_h_fp32.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d_bf16, x_h_bf16.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x_d_fp16, x_h_fp16.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    
    printf("\n[CUSPARSE] Benchmarking cuSPARSE SpMV:\n");
    printf("  Note: cuSparse uses CSR format\n");
    printf("  %-12s %-12s %-18s %-18s\n", "Precision", "Time (ms)", "Throughput (GOP/s)", "Speedup vs fp64");
    printf("  %-12s %-12s %-18s %-18s\n", "----------", "----------", "------------------", "------------------");
    
    // Benchmark fp64
    auto result_fp64 = benchmark_cusparse_spmv<double>(
        handle, csr_row_ptr, csr_col_ind, csr_val,
        x_d_fp64, y_d_fp64, n, mm.nnz, n_warmup, n_iter);
    
    printf("  %-12s %12.3f %18.3f %18.2fx\n", "fp64", 
           result_fp64.avg_time_ms, result_fp64.throughput_gops, 1.0);
    
    // Benchmark fp32
    auto result_fp32 = benchmark_cusparse_spmv<float>(
        handle, csr_row_ptr, csr_col_ind, csr_val,
        x_d_fp32, y_d_fp32, n, mm.nnz, n_warmup, n_iter);
    
    double speedup_fp32 = result_fp64.avg_time_ms / result_fp32.avg_time_ms;
    printf("  %-12s %12.3f %18.3f %18.2fx\n", "fp32",
           result_fp32.avg_time_ms, result_fp32.throughput_gops, speedup_fp32);
    
    // Benchmark bf16
    auto result_bf16 = benchmark_cusparse_spmv<__nv_bfloat16>(
        handle, csr_row_ptr, csr_col_ind, csr_val,
        x_d_bf16, y_d_bf16, n, mm.nnz, n_warmup, n_iter);
    
    double speedup_bf16 = result_fp64.avg_time_ms / result_bf16.avg_time_ms;
    printf("  %-12s %12.3f %18.3f %18.2fx\n", "bf16",
           result_bf16.avg_time_ms, result_bf16.throughput_gops, speedup_bf16);
    
    // Benchmark fp16
    auto result_fp16 = benchmark_cusparse_spmv<__half>(
        handle, csr_row_ptr, csr_col_ind, csr_val,
        x_d_fp16, y_d_fp16, n, mm.nnz, n_warmup, n_iter);
    
    double speedup_fp16 = result_fp64.avg_time_ms / result_fp16.avg_time_ms;
    printf("  %-12s %12.3f %18.3f %18.2fx\n", "fp16",
           result_fp16.avg_time_ms, result_fp16.throughput_gops, speedup_fp16);
    
    // Summary
    printf("\n  Summary:\n");
    printf("    - cuSparse fp64: %.3f ms, %.3f GOP/s\n", 
           result_fp64.avg_time_ms, result_fp64.throughput_gops);
    printf("    - cuSparse fp32: %.3f ms, %.3f GOP/s\n",
           result_fp32.avg_time_ms, result_fp32.throughput_gops);
    printf("    - cuSparse bf16: %.3f ms, %.3f GOP/s\n",
           result_bf16.avg_time_ms, result_bf16.throughput_gops);
    printf("    - cuSparse fp16: %.3f ms, %.3f GOP/s\n",
           result_fp16.avg_time_ms, result_fp16.throughput_gops);
    printf("    - cuSparse internal speedup (fp32 vs fp64): %.2fx\n", speedup_fp32);
    printf("    - cuSparse internal speedup (bf16 vs fp64): %.2fx\n", speedup_bf16);
    printf("    - cuSparse internal speedup (fp16 vs fp64): %.2fx\n", speedup_fp16);
    
    if (speedup_fp32 < 1.1) {
        printf("    Note: cuSparse shows minimal fp32 speedup. This suggests:\n");
        printf("      - Memory bandwidth may be the bottleneck (not compute)\n");
        printf("      - Or cuSparse's fp32 path is not fully optimized for this matrix\n");
    }
    
    if (speedup_bf16 > 1.1) {
        printf("    Note: cuSparse shows significant bf16 speedup (%.2fx).\n", speedup_bf16);
        printf("      - bf16 benefits from faster compute and better memory bandwidth\n");
    }
    
    if (speedup_fp16 > 1.1) {
        printf("    Note: cuSparse shows significant fp16 speedup (%.2fx).\n", speedup_fp16);
        printf("      - fp16 benefits from faster compute and better memory bandwidth\n");
    }
    
    // Compare fp16 vs bf16
    double fp16_vs_bf16 = result_bf16.avg_time_ms / result_fp16.avg_time_ms;
    if (fp16_vs_bf16 > 1.05) {
        printf("    Note: fp16 is %.2fx faster than bf16\n", fp16_vs_bf16);
    } else if (fp16_vs_bf16 < 0.95) {
        printf("    Note: bf16 is %.2fx faster than fp16\n", 1.0 / fp16_vs_bf16);
    } else {
        printf("    Note: fp16 and bf16 have similar performance\n");
    }
    
    // Algorithm tuning for fp16 and bf16
    printf("\n  [ALGORITHM TUNING] Testing different algorithms for fp16 and bf16:\n");
    printf("    Reference: https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv\n");
    
    // Test different algorithms
    // Note: Available algorithms depend on CUDA version and GPU architecture
    // Common options: CUSPARSE_SPMV_ALG_DEFAULT
    // Some versions may support: CUSPARSE_SPMV_CSR_ALG1, CUSPARSE_SPMV_CSR_ALG2, etc.
    
    struct AlgResult {
        cusparseSpMVAlg_t alg;
        const char* alg_name;
        BenchmarkResult<__nv_bfloat16> bf16_result;
        BenchmarkResult<__half> fp16_result;
    };
    
    std::vector<AlgResult> alg_results;
    
    // Test DEFAULT algorithm (already tested above)
    AlgResult default_result;
    default_result.alg = CUSPARSE_SPMV_ALG_DEFAULT;
    default_result.alg_name = "DEFAULT";
    default_result.bf16_result = result_bf16;
    default_result.fp16_result = result_fp16;
    alg_results.push_back(default_result);
    
    // Try to test other algorithms if available
    // Note: Different CUDA versions may support different algorithms
    // We'll try common ones and handle errors gracefully
    
    cusparseSpMVAlg_t test_algs[] = {
        CUSPARSE_SPMV_ALG_DEFAULT,
        CUSPARSE_SPMV_CSR_ALG1,  // CSR algorithm 1 (usually faster)
        CUSPARSE_SPMV_CSR_ALG2,  // CSR algorithm 2
        // Note: COO algorithms are for COO format, not CSR, so we skip them
        // CUSPARSE_SPMV_COO_ALG1,
        // CUSPARSE_SPMV_COO_ALG2,
    };
    
    const char* alg_names[] = {
        "DEFAULT",
        "CSR_ALG1",
        "CSR_ALG2",
    };
    
    // Verify array sizes match
    static_assert(sizeof(test_algs) / sizeof(test_algs[0]) == sizeof(alg_names) / sizeof(alg_names[0]),
                  "test_algs and alg_names arrays must have the same size");
    
    // Test bf16 with different algorithms
    printf("\n    Testing bf16 with different algorithms:\n");
    printf("    %-15s %-12s %-18s %-18s\n", "Algorithm", "Time (ms)", "Throughput (GOP/s)", "Speedup vs fp64");
    printf("    %-15s %-12s %-18s %-18s\n", "---------------", "----------", "------------------", "------------------");
    
    double best_bf16_speedup = speedup_bf16;
    cusparseSpMVAlg_t best_bf16_alg = CUSPARSE_SPMV_ALG_DEFAULT;
    const char* best_bf16_alg_name = "DEFAULT";
    
    for (size_t i = 0; i < sizeof(test_algs) / sizeof(test_algs[0]); i++) {
        // Test if algorithm is supported by trying to get buffer size
        cusparseSpMatDescr_t test_matA;
        cusparseDnVecDescr_t test_vecX, test_vecY;
        size_t test_bufferSize = 0;
        
        // Create temporary descriptors for testing
        int* d_test_csr_row_ptr;
        int* d_test_csr_col_ind;
        __nv_bfloat16* d_test_csr_val;
        __nv_bfloat16* d_test_x;
        __nv_bfloat16* d_test_y;
        
        CHECK_CUDA(cudaMalloc(&d_test_csr_row_ptr, (n + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_test_csr_col_ind, mm.nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_test_csr_val, mm.nnz * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_test_x, n * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&d_test_y, n * sizeof(__nv_bfloat16)));
        
        CHECK_CUDA(cudaMemcpy(d_test_csr_row_ptr, csr_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_test_csr_col_ind, csr_col_ind.data(), mm.nnz * sizeof(int), cudaMemcpyHostToDevice));
        
        cusparseStatus_t status = cusparseCreateCsr(&test_matA, n, n, mm.nnz,
                                                     d_test_csr_row_ptr, d_test_csr_col_ind, d_test_csr_val,
                                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16BF);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
            CHECK_CUDA(cudaFree(d_test_csr_col_ind));
            CHECK_CUDA(cudaFree(d_test_csr_val));
            CHECK_CUDA(cudaFree(d_test_x));
            CHECK_CUDA(cudaFree(d_test_y));
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        status = cusparseCreateDnVec(&test_vecX, n, d_test_x, CUDA_R_16BF);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            CHECK_CUSPARSE(cusparseDestroySpMat(test_matA));
            CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
            CHECK_CUDA(cudaFree(d_test_csr_col_ind));
            CHECK_CUDA(cudaFree(d_test_csr_val));
            CHECK_CUDA(cudaFree(d_test_x));
            CHECK_CUDA(cudaFree(d_test_y));
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        status = cusparseCreateDnVec(&test_vecY, n, d_test_y, CUDA_R_16BF);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            CHECK_CUSPARSE(cusparseDestroyDnVec(test_vecX));
            CHECK_CUSPARSE(cusparseDestroySpMat(test_matA));
            CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
            CHECK_CUDA(cudaFree(d_test_csr_col_ind));
            CHECK_CUDA(cudaFree(d_test_csr_val));
            CHECK_CUDA(cudaFree(d_test_x));
            CHECK_CUDA(cudaFree(d_test_y));
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        float test_alpha = 1.0f;
        float test_beta = 0.0f;
        status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &test_alpha, test_matA, test_vecX, &test_beta, test_vecY, CUDA_R_32F,
                                         test_algs[i], &test_bufferSize);
        
        CHECK_CUSPARSE(cusparseDestroyDnVec(test_vecY));
        CHECK_CUSPARSE(cusparseDestroyDnVec(test_vecX));
        CHECK_CUSPARSE(cusparseDestroySpMat(test_matA));
        CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
        CHECK_CUDA(cudaFree(d_test_csr_col_ind));
        CHECK_CUDA(cudaFree(d_test_csr_val));
        CHECK_CUDA(cudaFree(d_test_x));
        CHECK_CUDA(cudaFree(d_test_y));
        
        if (status != CUSPARSE_STATUS_SUCCESS) {
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        // Algorithm is supported, run benchmark
        auto test_result = benchmark_cusparse_spmv<__nv_bfloat16>(
            handle, csr_row_ptr, csr_col_ind, csr_val,
            x_d_bf16, y_d_bf16, n, mm.nnz, n_warmup, n_iter, test_algs[i]);
        
        double test_speedup = result_fp64.avg_time_ms / test_result.avg_time_ms;
        printf("    %-15s %12.3f %18.3f %18.2fx", alg_names[i],
               test_result.avg_time_ms, test_result.throughput_gops, test_speedup);
        
        if (test_speedup > best_bf16_speedup) {
            best_bf16_speedup = test_speedup;
            best_bf16_alg = test_algs[i];
            best_bf16_alg_name = alg_names[i];
            printf("  <-- Best");
        }
        printf("\n");
    }
    
    // Test fp16 with different algorithms
    printf("\n    Testing fp16 with different algorithms:\n");
    printf("    %-15s %-12s %-18s %-18s\n", "Algorithm", "Time (ms)", "Throughput (GOP/s)", "Speedup vs fp64");
    printf("    %-15s %-12s %-18s %-18s\n", "---------------", "----------", "------------------", "------------------");
    
    double best_fp16_speedup = speedup_fp16;
    cusparseSpMVAlg_t best_fp16_alg = CUSPARSE_SPMV_ALG_DEFAULT;
    const char* best_fp16_alg_name = "DEFAULT";
    
    for (size_t i = 0; i < sizeof(test_algs) / sizeof(test_algs[0]); i++) {
        // Test if algorithm is supported by trying to get buffer size
        cusparseSpMatDescr_t test_matA;
        cusparseDnVecDescr_t test_vecX, test_vecY;
        size_t test_bufferSize = 0;
        
        // Create temporary descriptors for testing
        int* d_test_csr_row_ptr;
        int* d_test_csr_col_ind;
        __half* d_test_csr_val;
        __half* d_test_x;
        __half* d_test_y;
        
        CHECK_CUDA(cudaMalloc(&d_test_csr_row_ptr, (n + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_test_csr_col_ind, mm.nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_test_csr_val, mm.nnz * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_test_x, n * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_test_y, n * sizeof(__half)));
        
        CHECK_CUDA(cudaMemcpy(d_test_csr_row_ptr, csr_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_test_csr_col_ind, csr_col_ind.data(), mm.nnz * sizeof(int), cudaMemcpyHostToDevice));
        
        cusparseStatus_t status = cusparseCreateCsr(&test_matA, n, n, mm.nnz,
                                                     d_test_csr_row_ptr, d_test_csr_col_ind, d_test_csr_val,
                                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
            CHECK_CUDA(cudaFree(d_test_csr_col_ind));
            CHECK_CUDA(cudaFree(d_test_csr_val));
            CHECK_CUDA(cudaFree(d_test_x));
            CHECK_CUDA(cudaFree(d_test_y));
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        status = cusparseCreateDnVec(&test_vecX, n, d_test_x, CUDA_R_16F);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            CHECK_CUSPARSE(cusparseDestroySpMat(test_matA));
            CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
            CHECK_CUDA(cudaFree(d_test_csr_col_ind));
            CHECK_CUDA(cudaFree(d_test_csr_val));
            CHECK_CUDA(cudaFree(d_test_x));
            CHECK_CUDA(cudaFree(d_test_y));
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        status = cusparseCreateDnVec(&test_vecY, n, d_test_y, CUDA_R_16F);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            CHECK_CUSPARSE(cusparseDestroyDnVec(test_vecX));
            CHECK_CUSPARSE(cusparseDestroySpMat(test_matA));
            CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
            CHECK_CUDA(cudaFree(d_test_csr_col_ind));
            CHECK_CUDA(cudaFree(d_test_csr_val));
            CHECK_CUDA(cudaFree(d_test_x));
            CHECK_CUDA(cudaFree(d_test_y));
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        float test_alpha = 1.0f;
        float test_beta = 0.0f;
        status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &test_alpha, test_matA, test_vecX, &test_beta, test_vecY, CUDA_R_32F,
                                         test_algs[i], &test_bufferSize);
        
        CHECK_CUSPARSE(cusparseDestroyDnVec(test_vecY));
        CHECK_CUSPARSE(cusparseDestroyDnVec(test_vecX));
        CHECK_CUSPARSE(cusparseDestroySpMat(test_matA));
        CHECK_CUDA(cudaFree(d_test_csr_row_ptr));
        CHECK_CUDA(cudaFree(d_test_csr_col_ind));
        CHECK_CUDA(cudaFree(d_test_csr_val));
        CHECK_CUDA(cudaFree(d_test_x));
        CHECK_CUDA(cudaFree(d_test_y));
        
        if (status != CUSPARSE_STATUS_SUCCESS) {
            printf("    %-15s %-30s\n", alg_names[i], "Not supported");
            continue;
        }
        
        // Algorithm is supported, run benchmark
        auto test_result = benchmark_cusparse_spmv<__half>(
            handle, csr_row_ptr, csr_col_ind, csr_val,
            x_d_fp16, y_d_fp16, n, mm.nnz, n_warmup, n_iter, test_algs[i]);
        
        double test_speedup = result_fp64.avg_time_ms / test_result.avg_time_ms;
        printf("    %-15s %12.3f %18.3f %18.2fx", alg_names[i],
               test_result.avg_time_ms, test_result.throughput_gops, test_speedup);
        
        if (test_speedup > best_fp16_speedup) {
            best_fp16_speedup = test_speedup;
            best_fp16_alg = test_algs[i];
            best_fp16_alg_name = alg_names[i];
            printf("  <-- Best");
        }
        printf("\n");
    }
    
    // Summary of best algorithms
    printf("\n    Best algorithm summary:\n");
    printf("      - bf16: %s algorithm, speedup %.2fx vs fp64\n", best_bf16_alg_name, best_bf16_speedup);
    printf("      - fp16: %s algorithm, speedup %.2fx vs fp64\n", best_fp16_alg_name, best_fp16_speedup);
    
    if (best_bf16_speedup > best_fp16_speedup) {
        printf("      - bf16 achieves %.2fx better speedup than fp16\n", best_bf16_speedup / best_fp16_speedup);
    } else if (best_fp16_speedup > best_bf16_speedup) {
        printf("      - fp16 achieves %.2fx better speedup than bf16\n", best_fp16_speedup / best_bf16_speedup);
    } else {
        printf("      - bf16 and fp16 achieve similar speedup\n");
    }
    
    // Cleanup
    CHECK_CUDA(cudaFree(x_d_fp64));
    CHECK_CUDA(cudaFree(y_d_fp64));
    CHECK_CUDA(cudaFree(x_d_fp32));
    CHECK_CUDA(cudaFree(y_d_fp32));
    CHECK_CUDA(cudaFree(x_d_bf16));
    CHECK_CUDA(cudaFree(y_d_bf16));
    CHECK_CUDA(cudaFree(x_d_fp16));
    CHECK_CUDA(cudaFree(y_d_fp16));
    delete[] x_h;
    delete[] y_h;
    CHECK_CUSPARSE(cusparseDestroy(handle));
    
    return 0;
}

