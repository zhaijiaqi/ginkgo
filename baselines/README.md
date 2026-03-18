# Baselines

本目录包含 CG 求解器的各类 baseline 实现。

---

## PETSc

### 1. 编译

```bash
cd baselines/petsc
./compile_PETSc.sh
```

- 依赖：PETSc 源码、CUDA、C/C++ 编译器
- 配置：`--with-mpi=0 --with-cuda=1 --download-cusp`，单进程 + GPU
- 产物：`src/ksp/ksp/tutorials/test_cg`

### 2. 测试（批量跑矩阵）

```bash
cd baselines/petsc/src/ksp/ksp/tutorials
./test_PETSc.sh
```

- 输入：同目录 `valid_matrix_set.csv`（第 3 列为 Name，对应 `<Name>.mtx`）
- 矩阵目录：`/data/matrix`
- 输出：`petsc_cg_a100.csv`（追加写入）

### 3. 单次运行示例

```bash
./test_cg /data/matrix/bodyy4.mtx -ksp_max_it 10000 -ksp_type cg -mat_type aijcusparse -vec_type cuda -pc_type none
```

---

## cuSPARSE

### 1. 编译

```bash
cd baselines/cusparse
make
```

- 依赖：CUDA（nvcc、cuSPARSE、cuBLAS）、g++
- 产物：`test_cusparse_cg`

### 2. 测试（批量跑矩阵）

```bash
cd baselines/cusparse
./test_cuSPARSE.sh [matrix_set.csv]
```

- 不传参：默认读取本目录 `valid_matrix_set.csv`
- 矩阵目录：`MATRIX_ROOT`（默认 `/data/matrix`）
- 输出：`cusparse_cg_a100.csv`（追加写入）

### 3. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MATRIX_ROOT` | /data/matrix | 矩阵根目录 |
| `OUT_CSV` | cusparse_cg_a100.csv | 输出 CSV 路径 |
| `MAX_IT` | 1000 | 最大迭代次数 |
| `TOL` | 1e-10 | 收敛容差 |
| `TRUNCATE_OUT_CSV` | 0 | 1 时清空输出再跑 |
| `MAX_MATS` | 0 | 最多测试矩阵数 |
| `TIMEOUT_LIMIT` | 4m | 单矩阵超时 |

### 4. 单次运行示例

```bash
./test_cusparse_cg /data/matrix/bodyy4.mtx 1000 1e-10 --csv --csv_out cusparse_cg_a100.csv
```

---

## Mille-feuille

### 1. 编译

```bash
cd baselines/Mille-feuille
make cg        # CG 求解器
make cg-mix    # CG 混合精度求解器
```

- 依赖：CUDA 12.0（`/usr/local/cuda-12.0`），sm_80（A100）
- 产物：`main-cg`、`main-cg-mixed`

### 2. 测试（批量跑矩阵）

```bash
cd baselines/Mille-feuille
./test_Mille_feuille.sh [matrix_set.csv]
```

- 不传参：默认读取本目录 `valid_matrix_set.csv`
- 同时运行 `main-cg` 和 `main-cg-mixed`，分别写入两个 CSV
- 矩阵目录：`MATRIX_ROOT`（默认 `/data/matrix`）

### 3. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MATRIX_ROOT` | /data/matrix | 矩阵根目录 |
| `MAX_IT` | 10000 | 最大迭代次数 |
| `OUT_CSV_CG` | mille_feuille_cg.csv | CG 输出 |
| `OUT_CSV_MIXED` | mille_feuille_cg_mixed.csv | 混合精度输出 |
| `MAX_MATS` | 0 | 最多测试矩阵数 |
| `TIMEOUT_LIMIT` | 4m | 单矩阵超时 |

### 4. 单次运行示例

```bash
./main-cg /data/matrix/bodyy4.mtx 10000 mille_feuille_cg.csv
./main-cg-mixed /data/matrix/bodyy4.mtx 10000 mille_feuille_cg_mixed.csv
```

---

## Ginkgo

### 1. 编译

```bash
cd baselines/ginkgo
./complie_Ginkgo.sh
```

- 依赖：CMake、CUDA 12.4（路径 `/usr/local/cuda-12.4`）
- 产物：`build/examples/simple-solver/simple-solver`，安装到 `install/`

### 2. 测试（批量跑矩阵）

```bash
cd baselines/ginkgo/examples/simple-solver
./test_Ginkgo.sh [matrix_set.csv]
```

- 不传参：默认读取项目根目录 `valid_matrix_set.csv`
- 传参：指定 CSV 路径（需含 `Name` 列，对应 `<Name>.mtx`）
- 矩阵目录：`MATRIX_ROOT`（默认 `/data/matrix`）
- 输出：`ginkgo_cg_a100.csv`（追加写入）

### 3. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EXECUTOR` | cuda | 执行后端：cuda / omp / reference |
| `MATRIX_ROOT` | /data/matrix | 矩阵文件根目录 |
| `GINKGO_BUILD` | ../../build | Ginkgo 构建目录 |
| `TRUNCATE_OUT_CSV` | 0 | 1 时清空输出 CSV 再跑 |
| `MAX_MATS` | 0 | 最多测试矩阵数，0 表示不限制 |
| `TIMEOUT_LIMIT` | 4m | 单矩阵超时时间 |

### 4. 单次运行示例

```bash
cd baselines/ginkgo
./build/examples/simple-solver/simple-solver cuda /data/matrix/bodyy4.mtx 1
```

- 参数：`<executor> <matrix.mtx> <flag>`，flag=1 为 CG，flag=2 为 BiCG
