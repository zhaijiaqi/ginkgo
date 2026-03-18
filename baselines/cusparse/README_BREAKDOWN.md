# CG cuSPARSE Baseline 性能分解分析

本目录包含基于 cuSPARSE baseline 的 CG 求解器性能分解分析工具。

## 功能

- 遍历 `valid_matrix_set.csv` 中的所有矩阵
- 运行基于 cuSPARSE 和 cuBLAS 的 CG 求解器
- 输出每个算子的详细性能分解（只统计以下三个主要 kernel）：
  - **SpMV** (稀疏矩阵-向量乘法) - 使用 cuSPARSE
  - **Dot product** (点积) - 使用 cuBLAS
  - **Axpy** (向量线性组合) - 使用 cuBLAS

## 使用方法

### 1. 编译代码

```bash
cd cusparse_baseline
make
```

### 2. 运行性能分解分析

```bash
cd cusparse_baseline
python3 benchmark_cg_breakdown.py
```

脚本会：
1. 读取 `/root/program/kernels/valid_matrix_set.csv` 中的矩阵列表
2. 为每个矩阵查找对应的 `.mtx` 文件（默认在 `/mnt/data/matrix` 目录）
3. 运行 CG benchmark 并收集性能数据
4. 将结果输出到 `cg_cusparse_breakdown.csv`

### 3. 单独测试单个矩阵

```bash
./test_cusparse_cg <matrix_file.mtx> [max_iter] [tol] [--benchmark] [warmup] [iters] [--json]
```

示例：
```bash
# 普通测试（显示详细输出）
./test_cusparse_cg /mnt/data/matrix/bodyy4.mtx 10000 1e-10

# Benchmark 模式（多次运行取平均）
./test_cusparse_cg /mnt/data/matrix/bodyy4.mtx 10000 1e-10 --benchmark 3 10

# JSON 输出模式（用于脚本解析）
./test_cusparse_cg /mnt/data/matrix/bodyy4.mtx 10000 1e-10 --benchmark 3 10 --json
```

## 输出格式

### CSV 输出 (`cg_cusparse_breakdown.csv`)

包含以下列：
- `matrix_id`, `matrix_name`: 矩阵标识和名称
- `rows`, `cols`, `entries`: 矩阵维度信息
- `converged`, `iterations`, `final_residual`: 收敛信息
- `spmv_time_ratio`: SpMV 时间占比（%）
- `dot_time_ratio`: Dot product 时间占比（%）
- `axpy_time_ratio`: Axpy 时间占比（%）

**注意**：统计结果只保留时间占比，不包含总时间和平均时间。

### JSON 输出

当使用 `--json` 选项时，程序会输出 JSON 格式的结果，便于脚本解析。

## 性能分解说明

CG 算法的每次迭代包含以下操作，但**只统计以下三个主要 kernel**：

1. **SpMV**: `μ = A * p` - 稀疏矩阵-向量乘法（使用 cuSPARSE）
2. **Dot products**: 
   - `r·r` - 残差向量的点积
   - `μ·p` - 用于计算步长
   - 使用 cuBLAS `cublasDdot`
3. **Axpy**: 
   - `x = x + α * p` - 更新解向量
   - `r = r - α * μ` - 更新残差向量
   - `p = r + β * p` - 更新搜索方向
   - 使用 cuBLAS `cublasDaxpy`

**注意**：其他操作（如 Nrm2、Scal、Copy）虽然仍然执行，但不统计时间，因为它们通常占用的时间很少，不是性能瓶颈。

每个算子的时间占比可以帮助识别性能瓶颈。

## 注意事项

- 矩阵文件路径：默认在 `/mnt/data/matrix` 目录，脚本会自动搜索多个可能的路径
- 超时设置：每个矩阵的测试超时时间为 10 分钟
- 内存要求：确保 GPU 有足够的内存来存储矩阵和向量
