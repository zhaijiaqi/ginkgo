# cuSPARSE SpMV Benchmark (C++)

这是一个用 C++ 编写的 cuSPARSE SpMV 性能基准测试程序。

## 编译

```bash
make
```

或者手动编译：

```bash
g++ -std=c++17 -O3 -Wall -I/usr/local/cuda/include \
    -o test_cusparse_spmv test_cusparse_spmv.cpp \
    -L/usr/local/cuda/lib64 -lcusparse -lcudart -lcublas
```

## 使用方法

```bash
./test_cusparse_spmv <matrix_file.mtx> [n_warmup] [n_iter]
```

参数说明：
- `matrix_file.mtx`: Matrix Market 格式的稀疏矩阵文件路径
- `n_warmup`: 预热迭代次数（默认：10）
- `n_iter`: 基准测试迭代次数（默认：100）

示例：

```bash
./test_cusparse_spmv ~/data/matrix/matrix_name.mtx 10 100
```

## 功能

- 加载 Matrix Market 格式的稀疏矩阵（.mtx 文件）
- 测试 fp64（双精度）、fp32（单精度）、bf16（bfloat16）和 fp16（half precision）性能
- 输出性能指标：
  - 平均时间（毫秒）
  - 吞吐量（GOP/s）
  - fp32、bf16 和 fp16 相对于 fp64 的加速比
  - fp16 和 bf16 的性能对比

## 输出示例

```
[INFO] Matrix loaded:
  File: ~/data/matrix/matrix_name.mtx
  Size: 1000 x 1000
  Non-zeros: 5000

[CUSPARSE] Benchmarking cuSPARSE SpMV:
  Note: cuSparse uses CSR format
  Precision    Time (ms)    Throughput (GOP/s) Speedup vs fp64   
  ------------ ------------ ------------------- ------------------
  fp64             21.247             0.056                1.00x
  fp32             21.041             0.057               1.01x
  bf16             18.500             0.064               1.15x
  fp16             18.200             0.065               1.17x

  Summary:
    - cuSparse fp64: 21.247 ms, 0.056 GOP/s
    - cuSparse fp32: 21.041 ms, 0.057 GOP/s
    - cuSparse bf16: 18.500 ms, 0.064 GOP/s
    - cuSparse fp16: 18.200 ms, 0.065 GOP/s
    - cuSparse internal speedup (fp32 vs fp64): 1.01x
    - cuSparse internal speedup (bf16 vs fp64): 1.15x
    - cuSparse internal speedup (fp16 vs fp64): 1.17x
    Note: cuSparse shows minimal fp32 speedup. This suggests:
      - Memory bandwidth may be the bottleneck (not compute)
      - Or cuSparse's fp32 path is not fully optimized for this matrix
    Note: cuSparse shows significant bf16 speedup (1.15x).
      - bf16 benefits from faster compute and better memory bandwidth
    Note: cuSparse shows significant fp16 speedup (1.17x).
      - fp16 benefits from faster compute and better memory bandwidth
    Note: fp16 is 1.02x faster than bf16
```

## 依赖

- CUDA Toolkit 12.0+（包含 cuSPARSE，支持 bfloat16 和 fp16）
- C++17 兼容的编译器（g++ 7+ 或 clang++ 5+）

注意：
- bfloat16 和 fp16 支持需要 CUDA 12.0+ 和新的 cuSPARSE API（`cusparseSpMV`）
- bfloat16 和 fp16 使用 float32 作为计算类型（混合精度），以获得更好的数值稳定性

## 注意事项

- 程序会自动处理对称矩阵（如果 Matrix Market 文件标记为 symmetric）
- Matrix Market 文件使用 1-based 索引，程序会自动转换为 0-based
- 程序会将 COO 格式转换为 CSR 格式（cuSPARSE 需要的格式）

## 与 Python 版本对比

这个 C++ 版本提供了与 Python 脚本 `test_fp32_bf16_optimization.py` 中 `benchmark_cusparse_spmv` 函数相同的功能，但：
- 性能更高（无 Python 开销）
- 支持 bfloat16 和 fp16（Python 版本中 PyTorch 的 cuSPARSE 后端不支持 bf16 和 fp16）
- 可以直接集成到 C++ 项目中
- 可以作为独立的基准测试工具使用
- 提供 fp16 和 bf16 的性能对比

注意：Python 版本使用 PyTorch 的稀疏矩阵操作，它底层调用 cuSPARSE，但 PyTorch 的 `sparse.mm` 不支持 bfloat16 和 fp16。这个 C++ 版本直接使用 cuSPARSE API，可以测试 bfloat16 和 fp16 性能，并对比两者的差异。

