import numpy as np
import scipy.sparse as sp
import torch
from spmv_kernels import bsr_spmv_a100_native, bsr_spmv_a100_unroll_fuse_warp

compute_dtype = torch.float64

# -----------------------------
# 1. Generate a random CSR matrix A and vector x
# -----------------------------


def build_random_csr(M, N, density=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(
        M,
        N,
        density=density,
        format="csr",
        dtype=np.float64,
        random_state=rng,
    )
    return A


# -----------------------------
# 2. Convert CSR -> 32x32 block-ELL/BSR layout
# -----------------------------


def csr_to_bsr_block_ell(A_csr, B=32):
    """
    Convert SciPy CSR matrix A_csr into our BSR/block-ELL layout:

      A_val    : (MB, MAX_BLOCKS_PER_ROW, B, B)
      A_colind : (MB, MAX_BLOCKS_PER_ROW)

    where:
      MB = ceil(M / B)
      blocks are indexed by (block_row, block_col) in units of BxB
      A_colind[br, slot] = block_col index for that slot, or -1 if empty

    We allow partial rows/cols; out-of-range entries remain implicit zeros.
    """
    M, N = A_csr.shape
    B = int(B)

    MB = (M + B - 1) // B  # ceildiv
    NB = (N + B - 1) // B

    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data

    # First pass: figure out which block-columns appear in each block-row
    block_cols_per_row = [set() for _ in range(MB)]

    for i in range(M):
        br = i // B
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for k in range(row_start, row_end):
            j = indices[k]
            bc = j // B
            block_cols_per_row[br].add(bc)

    max_blocks_per_row = max(len(s) for s in block_cols_per_row)
    print(f"MB={MB}, NB={NB}, max_blocks_per_row={max_blocks_per_row}")

    # Allocate BSR/block-ELL arrays
    A_val = np.zeros(
        (MB, max_blocks_per_row, B, B),
        dtype=np.float64,  # we'll downcast for fp32/fp16 later if needed
    )
    A_colind = -np.ones(
        (MB, max_blocks_per_row),
        dtype=np.int32,
    )

    # Map (br, bc) -> slot index
    block_slot = {}

    for br in range(MB):
        bcols = sorted(block_cols_per_row[br])
        for slot, bc in enumerate(bcols):
            A_colind[br, slot] = bc
            block_slot[(br, bc)] = slot

    # Second pass: fill block values
    for i in range(M):
        br = i // B
        row_in_block = i % B
        row_start = indptr[i]
        row_end = indptr[i + 1]
        for k in range(row_start, row_end):
            j = indices[k]
            bc = j // B
            col_in_block = j % B
            slot = block_slot[(br, bc)]
            A_val[br, slot, row_in_block, col_in_block] += data[k]

    return A_val, A_colind, MB, NB, max_blocks_per_row


from scipy.sparse import dok_matrix


def check_bsr_matches_csr(A_csr, A_val, A_colind, B, MB, NB, max_blocks_per_row):
    """
    Reconstruct A from (A_val, A_colind) and compare with original CSR.
    """
    M, N = A_csr.shape
    A_recon = dok_matrix((M, N), dtype=np.float64)

    for br in range(MB):
        for slot in range(max_blocks_per_row):
            bc = A_colind[br, slot]
            if bc < 0:
                continue
            block = A_val[br, slot]  # (B, B)
            for r in range(B):
                row = br * B + r
                if row >= M:
                    continue
                for c in range(B):
                    col = bc * B + c
                    if col >= N:
                        continue
                    v = block[r, c]
                    if v != 0.0:
                        A_recon[row, col] += v

    A_recon_csr = A_recon.tocsr()
    diff = (A_recon_csr - A_csr).tocoo()
    if diff.nnz == 0:
        print("✅ BSR reconstruction matches CSR exactly.")
    else:
        print(f"⚠️ BSR reconstruction mismatch: {diff.nnz} differing entries")
        print("   max abs diff:", np.max(np.abs(diff.data)))


def bsr_spmv_cpu(A_val, A_colind, B, MB, max_blocks_per_row, x, M, N):
    """
    CPU reference for the same BSR SpMV logic as the GPU kernel:
      y[row] = sum_{tiles, cols} A_val * x
    """
    y = np.zeros(M, dtype=x.dtype)
    for br in range(MB):
        for lane in range(B):
            global_row = br * B + lane
            if global_row >= M:
                continue
            acc = 0.0
            for bk in range(max_blocks_per_row):
                col_block = A_colind[br, bk]
                if col_block < 0:
                    continue
                base_col = col_block * B
                for j in range(B):
                    col = base_col + j
                    if col >= N:
                        continue
                    acc += A_val[br, bk, lane, j] * x[col]
            y[global_row] = acc
    return y


# -----------------------------
# 3. benchmark
# -----------------------------


def benchmark_spmv_kernel(
    spmv_kernel,
    A_val_torch,
    A_colind_torch,
    x_torch,
    y_torch,
    nnz: int,
    warmup: int = 10,
    iters: int = 100,
):
    """
    Benchmark the given SpMV kernel and report time statistics.

    Args:
      spmv_kernel   : callable(A_val, A_colind, x, y)
      A_val_torch   : (MB, max_blocks_per_row, B, B) tensor on CUDA
      A_colind_torch: (MB, max_blocks_per_row) int32 tensor on CUDA
      x_torch       : (N,) vector on CUDA
      y_torch       : (M,) vector on CUDA
      nnz           : number of nonzeros in original CSR (for FLOP count)
      warmup        : number of warmup iterations
      iters         : number of timed iterations
    """

    # Warmup (not timed)
    for _ in range(warmup):
        spmv_kernel(A_val_torch, A_colind_torch, x_torch, y_torch)
    torch.cuda.synchronize()

    # Create events for each iteration
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # Timed loop
    for i in range(iters):
        start_events[i].record()
        spmv_kernel(A_val_torch, A_colind_torch, x_torch, y_torch)
        end_events[i].record()

    # Wait for all kernels to finish
    torch.cuda.synchronize()

    # Collect per-iteration times in ms
    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]

    times_ms = np.array(times_ms, dtype=np.float64)
    avg_ms = float(times_ms.mean())
    med_ms = float(np.median(times_ms))  # this is your "mid time"

    # FLOPs: 2 * nnz (mul + add per nonzero)
    flops = 2.0 * nnz
    med_s = med_ms * 1e-3
    gflops_med = flops / med_s / 1e9

    print(f"[Benchmark] iters={iters}, warmup={warmup}")
    print(f"  mid   time : {med_ms:.3f} ms (median)")
    print(f"  avg   time : {avg_ms:.3f} ms")
    print(f"  GFLOP/s@mid: {gflops_med:.2f} GFLOP/s")

    return {
        "times_ms": times_ms,
        "med_ms": med_ms,
        "avg_ms": avg_ms,
        "gflops_med": gflops_med,
    }


# -----------------------------
# 4. End-to-end test
# -----------------------------


def main():
    # Problem size (you can bump these up to 100_000 once you're happy)
    M = N = 10000  # smaller for quick test; change to 100_000 for real workload
    density = 5e-3
    B = 32

    # 1) Build random CSR matrix and a random vector x
    A_csr = build_random_csr(M, N, density=density, seed=42)
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
        accum_dtype=dtype_map[compute_dtype],
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
        accum_dtype=dtype_map[compute_dtype],
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
