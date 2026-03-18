#include "cusparse_cg_solver.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <vector>

// 简单的 COO 到 CSR 转换函数（避免包含 CUDA 头文件）
void convertCOOtoCSR(int m, int n, int nnz,
                     const std::vector<int>& row_indices,
                     const std::vector<int>& col_indices,
                     const std::vector<double>& values,
                     std::vector<int>& csr_ptr,
                     std::vector<int>& csr_colidx,
                     std::vector<double>& csr_val) {
    (void)n; // n 仅用于维度信息，CSR 构建时不需要显式使用
    
    // 创建索引数组用于排序
    std::vector<std::pair<std::pair<int, int>, int>> indexed_elements;
    indexed_elements.reserve(nnz);
    for (int i = 0; i < nnz; i++) {
        indexed_elements.push_back({{row_indices[i], col_indices[i]}, i});
    }
    
    // 按行优先，然后按列排序
    std::sort(indexed_elements.begin(), indexed_elements.end());
    
    // 初始化 CSR 格式
    csr_ptr.assign(m + 1, 0);
    
    // 计算每行的非零元素数量
    for (int i = 0; i < nnz; i++) {
        int row = indexed_elements[i].first.first;
        csr_ptr[row + 1]++;
    }
    
    // 计算行指针（前缀和）
    for (int i = 1; i <= m; i++) {
        csr_ptr[i] += csr_ptr[i - 1];
    }
    
    // 填充列索引和值（按行填充）
    csr_colidx.resize(nnz);
    csr_val.resize(nnz);
    
    // 保存原始行指针
    std::vector<int> csr_ptr_copy = csr_ptr;
    
    for (int i = 0; i < nnz; i++) {
        int row = indexed_elements[i].first.first;
        int col = indexed_elements[i].first.second;
        int idx = indexed_elements[i].second;
        
        int pos = csr_ptr_copy[row];
        csr_colidx[pos] = col;
        csr_val[pos] = values[idx];
        csr_ptr_copy[row]++;
    }
}

// 从 Matrix Market 格式文件读取矩阵
bool read_matrix_market(const std::string& filename,
                        int& m, int& n, int& nnz,
                        std::vector<int>& row_indices,
                        std::vector<int>& col_indices,
                        std::vector<double>& values) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    bool is_symmetric = false;
    bool is_pattern = false;
    
    // 读取头部
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            if (line.find("symmetric") != std::string::npos || 
                line.find("Hermitian") != std::string::npos) {
                is_symmetric = true;
            }
            if (line.find("pattern") != std::string::npos) {
                is_pattern = true;
            }
            continue;
        }
        
        // 读取维度信息
        std::istringstream iss(line);
        if (!(iss >> m >> n >> nnz)) {
            continue;
        }
        break;
    }
    
    // 读取数据
    row_indices.clear();
    col_indices.clear();
    values.clear();
    
    int row, col;
    double val;
    
    while (file >> row >> col) {
        row--;  // Matrix Market 使用 1-based 索引
        col--;
        
        if (!is_pattern) {
            file >> val;
        } else {
            val = 1.0;
        }
        
        row_indices.push_back(row);
        col_indices.push_back(col);
        values.push_back(val);
        
        // 如果是对称矩阵，添加对称元素
        if (is_symmetric && row != col) {
            row_indices.push_back(col);
            col_indices.push_back(row);
            values.push_back(val);
        }
    }
    
    nnz = row_indices.size();
    return true;
}

// 计算 b = A * ones（CPU 端实现）
std::vector<double> compute_b_from_matrix(const std::string& matrix_file, int& m, int& n) {
    // 读取矩阵
    int nnz;
    std::vector<int> row_indices, col_indices;
    std::vector<double> values;
    
    if (!read_matrix_market(matrix_file, m, n, nnz, row_indices, col_indices, values)) {
        m = 0;
        n = 0;
        return std::vector<double>();
    }
    
    // 转换为 CSR 格式
    std::vector<int> csr_ptr, csr_colidx;
    std::vector<double> csr_val;
    
    convertCOOtoCSR(m, n, nnz, row_indices, col_indices, values,
                    csr_ptr, csr_colidx, csr_val);
    
    // 计算 b = A * ones（CPU 端）
    std::vector<double> b(m, 0.0);
    std::vector<double> ones(n, 1.0);
    
    for (int row = 0; row < m; row++) {
        int row_start = csr_ptr[row];
        int row_end = csr_ptr[row + 1];
        
        for (int j = row_start; j < row_end; j++) {
            int col = csr_colidx[j];
            if (col < n) {
                b[row] += csr_val[j] * ones[col];
            }
        }
    }
    
    return b;
}

CGSolverResult solve_cg_cusparse(
    const std::string& matrix_file,
    const CGSolverConfig& config,
    const std::vector<double>* b,
    const std::vector<double>* x0
) {
    if (config.verbose) {
        std::cout << "Loading matrix from " << matrix_file << "..." << std::endl;
    }
    
    // 读取矩阵
    int m, n, nnz;
    std::vector<int> row_indices, col_indices;
    std::vector<double> values;
    
    if (!read_matrix_market(matrix_file, m, n, nnz, row_indices, col_indices, values)) {
        CGSolverResult result;
        result.converged = false;
        return result;
    }
    
    if (config.verbose) {
        std::cout << "Matrix size: " << m << "x" << n << ", nnz: " << nnz << std::endl;
    }
    
    // 转换为 CSR 格式
    std::vector<int> csr_ptr, csr_colidx;
    std::vector<double> csr_val;
    
    convertCOOtoCSR(m, n, nnz, row_indices, col_indices, values,
                    csr_ptr, csr_colidx, csr_val);
    
    if (config.verbose) {
        std::cout << "Converted to CSR format" << std::endl;
    }
    
    // 如果 b 未提供，计算 b = A * ones
    std::vector<double> b_computed;
    const std::vector<double>* b_ptr = b;
    if (b == nullptr) {
        int m_temp, n_temp;
        b_computed = compute_b_from_matrix(matrix_file, m_temp, n_temp);
        b_ptr = &b_computed;
    }
    
    // 调用求解器
    return solve_cg_cusparse_from_csr(
        m, n, nnz,
        csr_ptr, csr_colidx, csr_val,
        b_ptr, x0,
        config
    );
}

