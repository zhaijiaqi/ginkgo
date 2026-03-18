# cuSPARSE Baseline CG Solver

基于 cuSPARSE 和 cuBLAS 库的 CG（共轭梯度）求解器基准实现。

## 特性

- 使用标准 CSR 矩阵格式
- 使用 cuSPARSE 进行稀疏矩阵-向量乘法（SpMV）
- 使用 cuBLAS 进行所有向量操作（官方 API，无自定义 kernel）：
  - cublasDdot - 向量点积
  - cublasDaxpy - y = alpha * x + y
  - cublasDscal - x = alpha * x
  - cublasDnrm2 - 计算 L2 范数
  - cublasDcopy - 向量复制
- 参考 `cg_kernel` 中的实现，保持相同的收敛判定逻辑
- 支持 Matrix Market 格式（.mtx）矩阵文件

## 编译

```bash
make
```

## 运行

### 命令行参数

```bash
./test_cusparse_cg <matrix_file.mtx> [max_iter] [tol] [--benchmark] [warmup] [iters]
```

**参数说明：**
- `matrix_file.mtx`: 矩阵文件路径（Matrix Market 格式，必需）
- `max_iter`: 最大迭代次数（可选，默认 1000）
- `tol`: 收敛容差（可选，默认 1e-10）
- `--benchmark`: 启用 benchmark 模式（可选）
- `warmup`: Warmup 运行次数（仅在 --benchmark 模式下有效，默认 3）
- `iters`: Benchmark 运行次数（仅在 --benchmark 模式下有效，默认 10）

### 单次测试

运行单次 CG 求解并显示结果：

```bash
./test_cusparse_cg ../bodyy4.mtx 1000 1e-10
```

**输出内容：**
- 矩阵信息（维度、非零元素数）
- 迭代过程（如果 verbose 模式开启）
- 收敛状态和迭代次数
- 残差历史
- 性能统计（SpMV 时间、CG step 时间、总 GPU 时间等）

### Benchmark 测试

运行多次测试以获取性能统计：

```bash
# 使用默认值（warmup=3, iters=10）
./test_cusparse_cg ../bodyy4.mtx 1000 1e-10 --benchmark

# 自定义 warmup 和 iters
./test_cusparse_cg ../bodyy4.mtx 1000 1e-10 --benchmark 5 20
```

**Benchmark 流程：**
1. Warmup 阶段：运行指定次数（默认 3 次）以预热 GPU
2. Benchmark 阶段：运行指定次数（默认 10 次）并收集性能数据
3. 统计输出：显示每次迭代的详细统计和所有运行的平均值

**输出内容：**
- 每次迭代的性能统计（中位数、平均值、标准差、最小值、最大值）
- SpMV 性能（GFLOP/s）
- 所有运行的平均值汇总
- 收敛状态统计

### 批量测试

使用 `test_cuSPARSE.sh` 对矩阵列表批量运行 CG，输出与 PETSc 同格式的 CSV。

**步骤：**

1. **编译**
   ```bash
   make
   ```

2. **准备矩阵**
   - 将 `.mtx` 矩阵文件放入 `MATRIX_ROOT` 目录（默认 `/data/matrix`）
   - 确保 `valid_matrix_set.csv` 存在（或自备矩阵列表 CSV，格式：`id,Group,Name,rows,cols,entries`，第 3 列为矩阵名）

3. **执行批量测试**
   ```bash
   # 使用默认 valid_matrix_set.csv，结果追加到 cusparse_cg_a100.csv
   ./test_cuSPARSE.sh

   # 指定矩阵列表
   ./test_cuSPARSE.sh /path/to/matrix_set.csv
   ```

4. **环境变量（可选）**
   | 变量 | 默认值 | 说明 |
   |------|--------|------|
   | `OUT_CSV` | `cusparse_cg_a100.csv` | 输出 CSV 路径 |
   | `TRUNCATE_OUT_CSV` | `0` | 设为 `1` 时清空再写，否则追加 |
   | `MAX_IT` | `1000` | 最大迭代次数 |
   | `TOL` | `1e-10` | 收敛容差 |
   | `MATRIX_ROOT` | `/data/matrix` | 矩阵文件根目录 |
   | `TIMEOUT_LIMIT` | `4m` | 单矩阵超时 |
   | `MAX_MATS` | `0` | 限制测试矩阵数量，`0` 表示不限制 |

5. **示例**
   ```bash
   # 清空输出后跑前 3 个矩阵
   TRUNCATE_OUT_CSV=1 MAX_MATS=3 ./test_cuSPARSE.sh

   # 自定义矩阵目录和输出
   MATRIX_ROOT=/data/matrix OUT_CSV=my_results.csv ./test_cuSPARSE.sh
   ```

**输出格式**（与 `petsc_cg_a100.csv` 一致）：
```
<matrix_path>,n=...,nnzR=...,petsc_norm=..., iterations=..., total_time=...,
```

其中 `total_time` 仅统计 CG 迭代时间（不含读 mtx、预处理、GPU 拷贝）。

### 使用 Makefile

```bash
make test        # 单次测试（使用 ../bodyy4.mtx 1000 1e-10）
make benchmark   # Benchmark 测试（使用 ../bodyy4.mtx 1000 1e-10 --benchmark 3 10）
```

### 配置选项

在代码中可以通过 `CGSolverConfig` 结构体配置求解器：

```cpp
CGSolverConfig config;
config.max_iter = 1000;        // 最大迭代次数
config.tol = 1e-10;            // 收敛容差
config.verbose = true;         // 是否显示详细输出
config.warmup = 3;             // Warmup 次数
config.benchmark_iters = 10;   // Benchmark 运行次数
```

### 示例输出

**单次测试输出示例：**
```
============================================================
CG Solver Test (cuSPARSE Baseline)
============================================================
Loading matrix from ../bodyy4.mtx...
Matrix size: 17546x17546, nnz: 121550
Converted to CSR format
Starting CG: initial residual = 1.234e+02, b_norm = 1.234e+02
Iteration 10: residual = 5.678e-03
...
Iteration 50: residual = 1.234e-08

============================================================
CG Solver Benchmark Results (cuSPARSE Baseline)
============================================================
Matrix: ../bodyy4.mtx (17546x17546, nnz=121550)

Total GPU Time (CG solver, excluding CPU convergence check):
  Total GPU: 45.678 ms

SpMV Time (per iteration, cuSPARSE):
  Median: 0.234 ms
  Mean:   0.235 ms
  Std:    0.002 ms
  Min:    0.231 ms
  Max:    0.239 ms
  GFLOP/s: 1038.45 GFLOP/s

CG Step Time (per iteration, cuBLAS):
  Median: 0.012 ms
  Mean:   0.013 ms
  ...

Iterations:
  Total: 50
  Converged: Yes
  Final residual: 1.23e-10
  Initial residual: 1.23e+02
  Reduction factor: 1.00e+12
============================================================
```

### 注意事项

1. **矩阵格式**：仅支持 Matrix Market 格式（.mtx 文件）
2. **GPU 要求**：需要支持 CUDA 的 GPU 和相应的驱动
3. **内存**：确保 GPU 有足够内存存储矩阵和向量
4. **性能**：Benchmark 模式会运行多次，可能需要较长时间
5. **收敛**：如果矩阵条件数较大，可能需要调整 `max_iter` 或 `tol`

## 算法

实现标准的 CG 算法，使用 cuSPARSE 和 cuBLAS 的官方 API：

1. 初始化：x₀, r₀ = b - Ax₀, p₀ = r₀
2. 迭代：
   - μ = Ap (使用 cuSPARSE cusparseSpMV)
   - αⱼ = (r·r) / (μ·p) (使用 cuBLAS cublasDdot)
   - x = x + αⱼ * p (使用 cuBLAS cublasDaxpy)
   - r_new = r - αⱼ * μ (使用 cuBLAS cublasDcopy + cublasDscal + cublasDaxpy)
   - βⱼ = ||r_new||² / ||r||² (使用 cuBLAS cublasDnrm2)
   - p = r_new + βⱼ * p (使用 cuBLAS cublasDscal + cublasDaxpy)
   - r = r_new (使用 cuBLAS cublasDcopy)
3. 收敛判定：||r|| / ||b|| < tol

## 性能统计

实现会输出以下性能指标：

### 单次测试输出
- **总 GPU 时间**：整个 CG 求解的 GPU 执行时间（不包括 CPU 端收敛判断）
- **SpMV 时间**：
  - 每次迭代的时间（中位数、平均值、标准差、最小值、最大值）
  - 总时间和平均时间
  - GFLOP/s 性能指标
- **CG step 时间**：
  - 每次迭代的时间（中位数、平均值、标准差、最小值、最大值）
  - 总时间和平均时间
- **时间占比**：SpMV 和 CG step 在总时间中的占比
- **迭代信息**：总迭代次数、收敛状态、残差历史

### Benchmark 输出
除了单次测试的所有信息外，还包括：
- **聚合统计**：所有运行的平均值
  - 平均总 GPU 时间
  - 平均 SpMV 时间
  - 平均 CG step 时间
  - 平均迭代次数
  - 所有运行是否都收敛

## 依赖

- CUDA Toolkit
- cuSPARSE 库（用于 SpMV）
- cuBLAS 库（用于向量操作）

## 文件结构

- `cusparse_cg_solver.h`: 头文件，定义接口和数据结构
- `cusparse_cg_solver.cu`: CUDA 实现，使用 cuSPARSE 和 cuBLAS
- `cusparse_cg_solver.cpp`: 辅助函数实现（矩阵读取、格式转换等）
- `test_cusparse_cg.cpp`: 测试程序和 benchmark 工具
- `test_cuSPARSE.sh`: 批量测试脚本
- `valid_matrix_set.csv`: 默认矩阵列表（用于批量测试）
- `Makefile`: 编译配置
- `README.md`: 本文档

## 常见问题

### Q: 如何选择合适的 max_iter 和 tol？
A: 
- `max_iter` 应该根据矩阵大小和条件数设置，通常 1000-10000 次迭代足够
- `tol` 取决于所需的精度，通常 1e-10 到 1e-6 之间

### Q: Benchmark 结果不稳定怎么办？
A: 
- 增加 warmup 次数（例如 5-10 次）
- 增加 benchmark 运行次数（例如 20-50 次）
- 确保 GPU 没有其他任务在运行

### Q: 如何提高性能？
A: 
- 使用更快的 GPU
- 确保矩阵格式正确（CSR 格式）
- 检查是否有内存带宽瓶颈

### Q: 收敛失败怎么办？
A: 
- 检查矩阵是否对称正定（CG 算法要求）
- 尝试调整 `tol` 或 `max_iter`
- 检查矩阵条件数是否过大

## 与 cg_kernel 的对比

本实现（cusparse_baseline）与 `cg_kernel` 的主要区别：

| 特性 | cusparse_baseline | cg_kernel |
|------|------------------|-----------|
| SpMV | cuSPARSE 官方 API | 自定义 CSC kernel |
| 向量操作 | cuBLAS 官方 API | 自定义 fused kernel |
| 矩阵格式 | CSR | CSC |
| 混合精度 | 不支持 | 支持 |
| 性能优化 | 库优化 | 手动优化 |

本实现主要用于：
- 作为性能基准（baseline）
- 验证算法正确性
- 对比自定义 kernel 的性能

