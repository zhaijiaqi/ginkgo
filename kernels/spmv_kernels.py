from optparse import make_option
from tkinter.constants import W

import numpy as np
import tilelang
import tilelang.language as T
import torch
from kernel_utils import benchmark_kernel
from scipy.sparse import coo_matrix
import sys


@tilelang.jit(target="cuda")
def bsr_spmv_a100_native(
    M: int,
    N: int,
    B: int,
    MB: int,
    MAX_BLOCKS_PER_ROW: int,
    val_dtype: str = "float64",
    x_dtype: str = "float64",
    accum_dtype: str = "float64",
):
    """
    Safe baseline BSR SpMV kernel:

      y = A * x

    A_val    : (MB, MAX_BLOCKS_PER_ROW, B, B)   val_dtype
    A_colind : (MB, MAX_BLOCKS_PER_ROW,)        int32 (>=0 or -1)
    x        : (N,)                             x_dtype
    y        : (M,)                             accum_dtype
    """

    @T.prim_func
    def main(
        A_val: T.Tensor((MB, MAX_BLOCKS_PER_ROW, B, B), val_dtype),  # type: ignore
        A_colind: T.Tensor(
            (
                MB,
                MAX_BLOCKS_PER_ROW,
            ),
            "int32",
        ),  # type: ignore
        x: T.Tensor((N,), x_dtype),  # type: ignore
        y: T.Tensor((M,), accum_dtype), # type: ignore
    ):
        # One block-row per CTA, B threads (one warp)
        with T.Kernel(MB, threads=B) as br:
            lane = T.get_thread_binding(0)  # 0..B-1
            global_row = br * B + lane

            if global_row < M:
                acc = T.alloc_local((1,), accum_dtype)
                T.clear(acc)

                for bk in T.serial(MAX_BLOCKS_PER_ROW):
                    col_block = A_colind[br, bk]
                    if col_block >= 0:
                        base_col = col_block * B

                        for j in T.serial(B):
                            col = base_col + j
                            if col < N:
                                a_ij = A_val[br, bk, lane, j].astype(accum_dtype)
                                xj = x[col].astype(accum_dtype)
                                acc[0] += a_ij * xj

                y[global_row] = acc[0]

    return main


@tilelang.jit(target="cuda")
def bsr_spmv_a100_unroll_fuse_warp(
    M: int,
    N: int,
    B: int,
    MB: int,
    MAX_BLOCKS_PER_ROW: int,
    ROWS_PER_CTA: int = 4,
    val_dtype: str = "float64",
    x_dtype: str = "float64",
    accum_dtype: str = "float64",
):
    """
    Optimized BSR SpMV kernel for A100:

      - ROWS_PER_CTA block-rows per CTA (ROWS_PER_CTA warps).
      - Shared-memory cache of x tiles per warp.
      - Unrolled inner loop over j in [0, B).

    NOTE: For now we assume MB % ROWS_PER_CTA == 0.
          Use ROWS_PER_CTA = 1 if you want to be safe for arbitrary M.
    """

    @T.prim_func
    def main(
        A_val: T.Tensor((MB, MAX_BLOCKS_PER_ROW, B, B), val_dtype),  # type: ignore
        A_colind: T.Tensor(
            (
                MB,
                MAX_BLOCKS_PER_ROW,
            ),
            "int32",
        ),  # type: ignore
        x: T.Tensor((N,), x_dtype),  # type: ignore
        y: T.Tensor((M,), accum_dtype),  # type: ignore
    ):
        # number of CTAs: MB / ROWS_PER_CTA   (assume divisible)
        grid_x = T.ceildiv(MB, ROWS_PER_CTA)
        with T.Kernel(grid_x, threads=B * ROWS_PER_CTA) as bx:
            tid = T.get_thread_binding(0)  # 0 .. (B*ROWS_PER_CTA - 1)
            warp = tid // B  # warp index in this CTA
            lane = tid % B  # lane index in warp [0..B-1]

            br = bx * ROWS_PER_CTA + warp  # block-row index
            global_row = br * B + lane

            # is this thread associated with a real row?
            row_active = (br < MB) and (global_row < M)

            # shared memory: one x tile per warp

            acc = T.alloc_local((1,), accum_dtype)
            T.clear(acc)

            # loop over tiles in this block-row
            for bk in T.serial(MAX_BLOCKS_PER_ROW):
                # default: no tile
                col_block = A_colind[br, bk] if br < MB else T.int32(-1)
                base_col = col_block * B

                # only threads with br < MB read A_colind
                # if col_block >= 0:
                #     # cooperative load of x tile
                #     col = base_col + lane
                #     if col < N:
                #         x_sh[warp, lane] = x[col].astype(accum_dtype)
                #     else:
                #         x_sh[warp, lane] = T.cast(0, accum_dtype)
                # else:
                #     # no tile here → zero x_sh for cleanliness
                #     x_sh[warp, lane] = T.cast(0, accum_dtype)

                # all threads in CTA hit the barrier
                # T.sync_threads()

                # accumulate only if this is a real row and real tile
                if row_active and (col_block >= 0):
                    for j in T.serial(0, B):
                        col_j = base_col + j
                        if col_j < N:
                            a_ij = A_val[br, bk, lane, j].astype(accum_dtype)
                            acc[0] += a_ij * x[col_j]
                            # acc[0] += a_ij * x[warp, j]

                # sync before reusing x_sh in next bk
                # T.sync_threads()

            # write back only for real rows
            if row_active:
                y[global_row] = acc[0]

    return main


@tilelang.jit(target="cuda")
def make_bsr_spmv_kernel(n_block_rows, nnzb, R, C, N):
    """
    Build a y = A x SpMV kernel for BSR format.

    BSR representation:
      data   : (nnzb, R, C) float32
      indices: (nnzb,) int32           # block-column indices
      indptr : (n_block_rows+1,) int32 # row pointer on block-rows

    Vector shapes:
      x : (N,) float32                 # N = n_block_cols * C
      y : (n_block_rows * R,) float32
    """

    @T.prim_func
    def main(
        data: T.Tensor((nnzb, R, C), "float32"), # type: ignore
        indices: T.Tensor((nnzb,), "int32"), # type: ignore
        indptr: T.Tensor((n_block_rows + 1,), "int32"), # type: ignore
        x: T.Tensor((N,), "float32"), # type: ignore
        y: T.Tensor((n_block_rows * R,), "float64"), # type: ignore
    ):
        with T.Kernel(n_block_rows, threads=1) as (br,):
            y_local = T.alloc_local((R,), "float64")

            start = indptr[br]
            end = indptr[br + 1]
            length = end - start

            # Loop over all blocks in this block-row
            for t in T.serial(length):
                bk = start + t  # block index
                bc = indices[bk]  # block-column index
                x_base = bc * C  # starting col in x

                for rr in T.serial(R):
                    for cc in T.serial(C):
                        y_local[rr] += data[bk, rr, cc] * x[x_base + cc]

            # Write the accumulated result back to y
            row_base = br * R
            for rr in T.serial(R):
                y[row_base + rr] = y_local[rr]

    return main


def bsr_spmv(data_bsr, indices_bsr, indptr_bsr, x, R, C, device="cuda"):
    """
    data_bsr   : (nnzb, R, C) float32 (numpy)
    indices_bsr: (nnzb,) int32 (numpy)
    indptr_bsr : (n_block_rows+1,) int32 (numpy)
    x          : (N,) float32 (numpy)
    R, C       : block size
    """
    data_bsr = np.asarray(data_bsr, dtype=np.float32)
    indices_bsr = np.asarray(indices_bsr, dtype=np.int32)
    indptr_bsr = np.asarray(indptr_bsr, dtype=np.int32)
    x = np.asarray(x, dtype=np.float32)

    nnzb = data_bsr.shape[0]
    n_block_rows = indptr_bsr.shape[0] - 1
    N = x.shape[0]
    M = n_block_rows * R

    # Move to device (torch tensors)
    dev = torch.device(device)
    data_t = torch.from_numpy(data_bsr).to(dev)
    indices_t = torch.from_numpy(indices_bsr).to(dev)
    indptr_t = torch.from_numpy(indptr_bsr).to(dev)
    x_t = torch.from_numpy(x).to(dev)
    y_t = torch.zeros((M,), dtype=torch.float64, device=dev)

    # Build kernel specialized to these sizes
    spmv_kernel = make_bsr_spmv_kernel(n_block_rows, nnzb, R, C, N)

    print(f"{data_t=}, {indices_t=}, {indptr_t=}, {x_t=}")
    # Run kernel
    spmv_kernel(data_t, indices_t, indptr_t, x_t, y_t)
    print(y_t)

    # Back to numpy
    y = y_t.cpu().numpy()
    return y


@tilelang.jit(target="cuda")
def make_bsr_spmv_mixed_kernel(n_block_rows, nnzb, R, C, N):
    """
    Build a y = A x mixed SpMV kernel for BSR format.

    BSR representation:
      data   : (nnzb, R, C) float32
      actions: (N / C,) int32          # Precision each block-column
      indices: (nnzb,) int32           # block-column indices
      indptr : (n_block_rows+1,) int32 # row pointer on block-rows

    Vector shapes:
      x : (N,) float32                 # N = n_block_cols * C
      y : (n_block_rows * R,) float32
    """

    def cast_fp_like(v, mantissa_bits):
        # v: float64 的 PrimExpr
        # mantissa_bits: int32 的 PrimExpr（例如 52, 23, 10, 7）
        zero = T.Cast("float64", 0.0)
        abs_v = T.abs(v)

        # 指数 & 尾数： abs_v = mant * 2 ** exponent
        exponent = T.floor(T.log2(abs_v))
        pow2e = T.exp2(exponent)
        mantissa = abs_v / pow2e

        # mantissa_bits > 0 : 按 2**mantissa_bits 级数做 round
        scale = T.exp2(T.Cast("float64", mantissa_bits))
        mantissa_rounded = T.round(mantissa * scale) / scale

        # mantissa_bits == 0 : mantissa >= 0.5 -> 1.0 else 0.0
        mantissa_binary = T.if_then_else(
            mantissa >= T.Cast("float64", 0.5),
            T.Cast("float64", 1.0),
            T.Cast("float64", 0.0),
        )

        mantissa_q = T.if_then_else(
            mantissa_bits > 0,
            mantissa_rounded,
            mantissa_binary,
        )

        # 符号
        sign = T.if_then_else(
            v < zero,
            T.Cast("float64", -1.0),
            T.Cast("float64", 1.0),
        )

        non_zero = sign * mantissa_q * pow2e

        return T.if_then_else(abs_v == zero, zero, non_zero)

    @T.prim_func
    def main(
        data: T.Tensor((nnzb, R, C), "float64"), # type: ignore
        actions: T.Tensor(((N + C - 1) // C,), "int32"), # type: ignore
        indices: T.Tensor((nnzb,), "int32"), # type: ignore
        indptr: T.Tensor((n_block_rows + 1,), "int32"), # type: ignore
        x: T.Tensor((N,), "float64"), # type: ignore
        y: T.Tensor((n_block_rows * R,), "float64"), # type: ignore
    ):
        with T.Kernel(n_block_rows, threads=1) as (br,):
            y_local = T.alloc_local((R,), "float64")

            # Initialize y_local to zero
            for rr in T.serial(R):
                y_local[rr] = T.float64(0)

            start = indptr[br]
            end = indptr[br + 1]
            length = end - start

            # Loop over all blocks in this block-row
            for t in T.serial(length):
                bk = start + t  # block index
                bc = indices[bk]  # block-column index
                x_base = bc * C  # starting col in x
                action = actions[bc]

                mantissa_bits = T.if_then_else(
                    action == 0,
                    T.int32(52),
                    T.if_then_else(
                        action == 1,
                        T.int32(23),
                        T.if_then_else(
                            action == 2,
                            T.int32(10),
                            T.int32(7),
                        ),
                    ),
                )

                for rr in T.serial(R):
                    for cc in T.serial(C):
                        col_idx = x_base + cc
                        if col_idx < N:
                            vA = cast_fp_like(data[bk, rr, cc], mantissa_bits)
                            vx = cast_fp_like(x[col_idx], mantissa_bits)
                            y_local[rr] += vA * vx

            # Write the accumulated result back to y
            row_base = br * R
            for rr in T.serial(R):
                y[row_base + rr] = y_local[rr]

    return main


@tilelang.jit(target="cuda")
def make_bsr_spmv_mixed_kernel_warp_reduce(
    n_block_rows, nnzb, R, C, N, WARPS_PER_BLOCK=2
):
    """
    Build a y = A x quantized SpMV kernel for BSR format with true quantization.

    BSR representation:
      data   : (nnzb, R, C) float64
      actions: (N / C,) int32          # Precision level each block-column (0=float64, 1=float32, 2=float16, 3=float8)
      indices: (nnzb,) int32           # block-column indices
      indptr : (n_block_rows+1,) int32 # row pointer on block-rows

    Vector shapes:
      x : (N,) float64                 # N = n_block_cols * C
      y : (n_block_rows * R,) float64

    True Quantization:
      - Each tile (R x C) is quantized to the precision specified by actions
      - Dynamic range analysis: find max(|x|) and max(|A|) for each tile
      - Scaling: scale = max_representable_value / max_abs
      - Quantization: scale -> clip -> convert to target precision -> dequantize (divide by scale)
      - action == 0: float64 (no quantization)
      - action == 1: float32 precision with quantization
      - action == 2: float16 precision with quantization
      - action == 3: float8_e4m3 precision with quantization
    """

    def q(v, action):
        v_f32 = T.Cast("float64", T.Cast("float32", v))
        v_f16 = T.Cast("float64", T.Cast("float16", v))
        v_fp8 = T.Cast("float64", T.Cast("float8_e4m3", v))
        return T.if_then_else(
            action == 0,
            v,
            T.if_then_else(
                action == 1,
                v_f32,
                T.if_then_else(action == 2, v_f16, v_fp8),
            ),
        )


    # Max representable values for different precisions
    def get_max_value(action):
        # Return max representable value for different precisions
        return T.if_then_else(
            action == 0,  # float64 - use a large value
            T.float64(1e37),
            T.if_then_else(
                action == 1,  # float32
                T.float64(3.4e38),
                T.if_then_else(
                    action == 2,  # float16
                    T.float64(65504.0),  # FP16 max
                    T.float64(448.0),    # FP8 E4M3 max
                ),
            ),
        )

    # True quantization with scaling and clipping for low precision formats
    def quantize_value(value, scale, max_val, action):
        return T.if_then_else(
            action <= 1,  # float64 and float32: simple type conversion
            T.if_then_else(
                action == 0,
                value,  # float64 - no change
                T.Cast("float64", T.Cast("float32", value)),  # float32
            ),
            # action >= 2: true quantization for FP16 and FP8
            T.if_then_else(
                action == 2,
                dequantize_value(T.Cast("float64", T.Cast("float16",
                    T.max(T.min(scale * value, max_val), -max_val))), scale),  # FP16 with scaling
                dequantize_value(T.Cast("float64", T.Cast("float8_e4m3",
                    T.max(T.min(scale * value, max_val), -max_val))), scale),  # FP8 with scaling
            ),
        )

    # Dequantization: convert back by dividing by scale
    def dequantize_value(q_value, scale):
        return q_value / scale

    R_TILES = (R + 31) // 32
    C_TILES = (C + 31) // 32

    @T.prim_func
    def main(
        data: T.Tensor((nnzb, R, C), "float64"), # type: ignore
        actions: T.Tensor(((N + C - 1) // C,), "int32"), # type: ignore
        indices: T.Tensor((nnzb,), "int32"), # type: ignore
        indptr: T.Tensor((n_block_rows + 1,), "int32"), # type: ignore
        x: T.Tensor((N,), "float64"), # type: ignore
        y: T.Tensor((n_block_rows * R,), "float64"), # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(n_block_rows, WARPS_PER_BLOCK), threads=32 * WARPS_PER_BLOCK
        ) as bx:
            warp = T.get_warp_idx_sync()
            lane = T.get_lane_idx()
            br = bx * WARPS_PER_BLOCK + warp
            acc = T.alloc_local((R_TILES,), "float64")
            for rt in T.unroll(R_TILES):
                acc[rt] = T.float64(0)
            x_lane = T.alloc_local((1,), "float64")
            x_lane[0] = T.float64(0)


            mask = T.tvm_warp_activemask()

            if br < n_block_rows:
                start = indptr[br]
                end = indptr[br + 1]

                for bk in T.serial(start, end):
                    bc = indices[bk]
                    action = actions[bc]
                    x_base = bc * C

                    # Calculate quantization parameters for this tile (only for low precision)
                    max_val = get_max_value(action)
                    x_scale = T.float64(1.0)
                    a_scale = T.float64(1.0)

                    # Only compute scaling for low precision formats (FP16, FP8)
                    if action >= 2:
                        # Find dynamic range of x tile
                        x_max_abs = T.float64(0.0)
                        for ct_check in T.unroll(C_TILES):
                            cc_base_check = ct_check * 32
                            for cc_check in T.unroll(32):
                                cc_g_check = cc_base_check + cc_check
                                if cc_g_check < C:
                                    col_check = x_base + cc_g_check
                                    if col_check < N:
                                        x_val_check = x[col_check]
                                        x_max_abs = T.max(x_max_abs, T.abs(x_val_check))

                        # Find dynamic range of A tile
                        a_max_abs = T.float64(0.0)
                        for rr_check in T.serial(R):
                            for cc_check in T.serial(C):
                                a_val_check = data[bk, rr_check, cc_check]
                                a_max_abs = T.max(a_max_abs, T.abs(a_val_check))

                        # Calculate scaling factors: scale = max_representable / max_abs
                        x_scale = max_val / (x_max_abs + T.float64(1e-12))
                        a_scale = max_val / (a_max_abs + T.float64(1e-12))

                    for ct in T.unroll(C_TILES):
                        cc_base = ct * 32

                        x_lane[0] = T.float64(0)
                        col = x_base + cc_base + lane
                        if (cc_base + lane) < C and col < N:
                            # Apply quantization to x
                            x_val = x[col]
                            x_lane[0] = quantize_value(x_val, x_scale, max_val, action)

                        for cc in T.unroll(32):
                            cc_g = cc_base + cc
                            if cc_g < C:
                                vx = T.tvm_warp_shuffle(
                                    mask, x_lane[0], T.int32(cc), 32, 32
                                )

                                for rt in T.unroll(R_TILES):
                                    rr = rt * 32 + lane
                                    if rr < R:
                                        # Apply quantization to A
                                        a_val = data[bk, rr, cc_g]
                                        vA = quantize_value(a_val, a_scale, max_val, action)
                                        acc[rt] += vA * vx

                row_base = br * R
                for rt in T.unroll(R_TILES):
                    rr = rt * 32 + lane
                    if rr < R:
                        y[row_base + rr] = acc[rt]

    return main


def bsr_spmv_mixed(
    data_bsr, actions, indices_bsr, indptr_bsr, x, R, C, device="cuda", kernel_list=1
):
    """
    data_bsr   : (nnzb, R, C) float32 (numpy)
    indices_bsr: (nnzb,) int32 (numpy)
    indptr_bsr : (n_block_rows+1,) int32 (numpy)
    x          : (N,) float32 (numpy)
    R, C       : block size
    """
    data_bsr = np.asarray(data_bsr, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int32)
    indices_bsr = np.asarray(indices_bsr, dtype=np.int32)
    indptr_bsr = np.asarray(indptr_bsr, dtype=np.int32)
    x = np.asarray(x, dtype=np.float64)

    nnzb = data_bsr.shape[0]
    n_block_rows = indptr_bsr.shape[0] - 1
    N = x.shape[0]
    M = n_block_rows * R

    # Move to device (torch tensors)
    dev = torch.device(device)
    data_t = torch.from_numpy(data_bsr).to(dev)
    actions_t = torch.from_numpy(actions).to(dev)
    indices_t = torch.from_numpy(indices_bsr).to(dev)
    indptr_t = torch.from_numpy(indptr_bsr).to(dev)
    x_t = torch.from_numpy(x).to(dev)
    y_t = torch.zeros((M,), dtype=torch.float64, device=dev)

    # Build kernel specialized to these sizes
    spmv_kernel = (
        make_bsr_spmv_mixed_kernel(n_block_rows, nnzb, R, C, N)
        if kernel_list == 0
        else make_bsr_spmv_mixed_kernel_warp_reduce(n_block_rows, nnzb, R, C, N)
    )

    # Run kernel
    spmv_kernel(data_t, actions_t, indices_t, indptr_t, x_t, y_t)

    # Back to numpy
    y = y_t.cpu().numpy()
    return y


def test_bsr_spmv():
    print("Running BSR spmv TileLang test...")
    M, N = 4, 4
    R, C = 2, 2  # BSR tile shape
    density = 1.00

    rng = np.random.default_rng(0)
    size = M * N
    nnz = int(size * density)

    rows = rng.integers(0, M, size=nnz, dtype=np.int32)
    cols = rng.integers(0, N, size=nnz, dtype=np.int32)
    vals = rng.random(nnz, dtype=np.float64)

    A_coo = coo_matrix((vals, (rows, cols)), shape=(M, N))

    A_bsr = A_coo.tobsr(blocksize=(R, C))
    A_bsr.sort_indices()
    ref_data = A_bsr.data
    ref_indices = A_bsr.indices
    ref_indptr = A_bsr.indptr

    x = rng.standard_normal(N).astype(np.float64)

    y_ref = A_bsr @ x
    y_tl = bsr_spmv(ref_data, ref_indices, ref_indptr, x, R, C, device="cuda")

    print(y_tl)
    print(y_ref)
    assert np.allclose(y_tl, y_ref, rtol=1e-5, atol=1e-6)
    print("BSR SpMV kernel matches SciPy.")


def test_bsr_spmv_mixed():
    print("Running BSR mixed spmv TileLang test...")
    M, N = 1024, 1024
    R, C = 64, 64  # BSR tile shape
    density = 0.15

    rng = np.random.default_rng(0)
    size = M * N
    nnz = int(size * density)

    rows = rng.integers(0, M, size=nnz, dtype=np.int32)
    cols = rng.integers(0, N, size=nnz, dtype=np.int32)
    vals = rng.random(nnz, dtype=np.float64)

    A_coo = coo_matrix((vals, (rows, cols)), shape=(M, N))

    A_bsr = A_coo.tobsr(blocksize=(R, C))
    A_bsr.sort_indices()
    ref_data = A_bsr.data
    ref_indices = A_bsr.indices
    ref_indptr = A_bsr.indptr

    x = rng.standard_normal(N).astype(np.float64)

    y_ref = A_bsr @ x

    actions = np.full((N // C,), 3)
    # actions = np.random.randint(0, 2, size=N // C)
    y_tl = bsr_spmv_mixed(
        ref_data,
        actions,
        ref_indices,
        ref_indptr,
        x,
        R,
        C,
        device="cuda",
        kernel_list=1,
    )

    print(y_tl[:20])
    print(y_ref[:20])
    assert np.allclose(y_tl, y_ref, rtol=1e-5, atol=1e-6)
    print("BSR SpMV kernel matches SciPy.")


def bench_bsr_spmv_mixed():
    print("Running BSR mixed spmv TileLang benchmark...")
    M, N = 1024, 1024
    R, C = 32, 32  # BSR tile shape
    density = 0.15

    rng = np.random.default_rng(0)
    size = M * N
    nnz = int(size * density)

    rows = rng.integers(0, M, size=nnz, dtype=np.int32)
    cols = rng.integers(0, N, size=nnz, dtype=np.int32)
    vals = rng.random(nnz, dtype=np.float64)

    A_coo = coo_matrix((vals, (rows, cols)), shape=(M, N))

    A_bsr = A_coo.tobsr(blocksize=(R, C))
    A_bsr.sort_indices()
    ref_data = A_bsr.data
    ref_indices = A_bsr.indices
    ref_indptr = A_bsr.indptr

    x = np.random.randn(N).astype(np.float64)
    actions = np.full((N // C,), 3, dtype=np.int32)
    n_block_rows = ref_indptr.shape[0] - 1
    nnzb = ref_data.shape[0]

    # Move to device (torch tensors)
    dev = "cuda"
    data_t = torch.from_numpy(ref_data).to(dev)
    actions_t = torch.from_numpy(actions).to(dev)
    indices_t = torch.from_numpy(ref_indices).to(dev)
    indptr_t = torch.from_numpy(ref_indptr).to(dev)
    x_t = torch.from_numpy(x).to(dev)
    y_t = torch.zeros((M,), dtype=torch.float64, device=dev)

    # spmv_kernel = make_bsr_spmv_mixed_kernel(n_block_rows, nnzb, R, C, N)
    spmv_kernel_warp_reduce = make_bsr_spmv_mixed_kernel_warp_reduce(
        n_block_rows, nnzb, R, C, N
    )

    # benchmark_kernel(
    #     spmv_kernel,
    #     (data_t, actions_t, indices_t, indptr_t, x_t, y_t),
    #     nnz,
    #     warmup=10,
    #     iters=100,
    # )
    benchmark_kernel(
        spmv_kernel_warp_reduce,
        (data_t, actions_t, indices_t, indptr_t, x_t, y_t),
        nnz,
        warmup=10,
        iters=100,
    )
    
    
if __name__ == "__main__":
    # test_bsr_spmv()
    test_bsr_spmv_mixed()
    # bench_bsr_spmv_mixed()