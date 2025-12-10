import os

import numpy as np
import scipy.sparse as sp
import torch
from coo2bsr_kernels import coo2bsr
from scipy.io import mmread
from spmv_kernels import make_bsr_spmv_kernel

compute_dtype = torch.float64
accumulate_dtype = torch.float64


def main():
    # Problem size (you can bump these up to 100_000 once you're happy)
    # M = N = 10000  # smaller for quick test; change to 100_000 for real workload
    # density = 5e-3
    # A_csr = build_random_csr(M, N, density=density, seed=42)
    B = 32

    # 1) load matrix and generate random vector x
    matrix_path = os.path.expanduser("~/data/matrix/Muu.mtx")
    A = mmread(matrix_path)
    A_csr = A.tocsr()
    M = A.shape[0]
    N = A.shape[1]

    dtype_map = {torch.float32: np.float32, torch.float64: np.float64}
    x_np = np.random.randn(N).astype(dtype_map[compute_dtype])

    # 2) Convert CSR -> BSR/block-ELL
    A_val_np, A_colind_np, MB, NB, max_blocks_per_row = csr_to_bsr_block_ell(A_csr, B=B)
    # check_bsr_matches_csr(A_csr, A_val_np, A_colind_np, B, MB, NB, max_blocks_per_row)

    print("CSR nnz:", A_csr.nnz)
    print("BSR shape A_val:", A_val_np.shape)
    print("BSR shape A_colind:", A_colind_np.shape)

    # 3) Reference SpMV on CPU
    y_ref = A_csr.dot(x_np)  # SciPy CSR SpMV (float64)

    # y_bsr_cpu = bsr_spmv_cpu(
    #     A_val_np, A_colind_np, B, MB, max_blocks_per_row, x_np, M, N
    # )

    # print("CPU BSR max_abs_diff:", np.max(np.abs(y_bsr_cpu - y_ref)))
    # print("CPU BSR diff at tail:", (y_bsr_cpu - y_ref)[90:])
    # 4) Move to GPU with PyTorch
    device = "cuda"

    A_val_torch = torch.from_numpy(A_val_np).to(device=device, dtype=compute_dtype)
    A_colind_torch = torch.from_numpy(A_colind_np).to(device=device, dtype=torch.int32)
    x_torch = torch.from_numpy(x_np).to(device=device, dtype=compute_dtype)
    y_torch = torch.empty(M, device=device, dtype=compute_dtype)

    dtype_map = {torch.float32: "float32", torch.float64: "float64"}
    # 5) Build kernel for fp64 values, fp64 x, fp64 accum
    spmv_kernel = bsr_spmv_a100_native(
        M=M,
        N=N,
        B=B,
        MB=MB,
        MAX_BLOCKS_PER_ROW=max_blocks_per_row,
        val_dtype=dtype_map[compute_dtype],
        x_dtype=dtype_map[compute_dtype],
        accum_dtype=dtype_map[accumulate_dtype],
    )

    spmv_kernel_1 = bsr_spmv_a100_unroll_fuse_warp(
        M=M,
        N=N,
        B=B,
        MB=MB,
        MAX_BLOCKS_PER_ROW=max_blocks_per_row,
        ROWS_PER_CTA=1,
        val_dtype=dtype_map[compute_dtype],
        x_dtype=dtype_map[compute_dtype],
        accum_dtype=dtype_map[accumulate_dtype],
    )

    # 6) Run kernel: y = A * x
    # spmv_kernel(A_val_torch, A_colind_torch, x_torch, y_torch)
    spmv_kernel_1(A_val_torch, A_colind_torch, x_torch, y_torch)

    # 7) Compare with reference
    y_gpu_np = y_torch.cpu().numpy()
    diff = y_gpu_np - y_ref
    max_abs_err = np.max(np.abs(diff))
    rel_err = max_abs_err / (np.max(np.abs(y_ref)) + 1e-12)

    print(f"max_abs_err = {max_abs_err:.3e}")
    print(f"rel_err     = {rel_err:.3e}")

    # print(f"{y_ref=}")
    # print(f"{y_gpu_np=}")

    # Sanity check: should be close to machine epsilon-ish for fp64
    if compute_dtype == torch.float64:
        if max_abs_err < 1e-10:
            print("✅ SpMV test PASSED (fp64)")
        else:
            print("⚠️ SpMV test difference is larger than expected, inspect further")

    # 8) Benchmark the kernel
    print("Running performance benchmark...")
    benchmark_spmv_kernel(
        spmv_kernel,  # or spmv_kernel_fp64
        A_val_torch,
        A_colind_torch,
        x_torch,
        y_torch,
        nnz=A_csr.nnz,
        warmup=10,
        iters=1000,
    )
    print("Running performance benchmark...")
    benchmark_spmv_kernel(
        spmv_kernel_1,  # or spmv_kernel_fp64
        A_val_torch,
        A_colind_torch,
        x_torch,
        y_torch,
        nnz=A_csr.nnz,
        warmup=10,
        iters=1000,
    )


if __name__ == "__main__":
    main()
