import tilelang
import tilelang.language as T


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
        A_val: T.Tensor((MB, MAX_BLOCKS_PER_ROW, B, B), val_dtype),
        A_colind: T.Tensor(
            (
                MB,
                MAX_BLOCKS_PER_ROW,
            ),
            "int32",
        ),
        x: T.Tensor((N,), x_dtype),
        y: T.Tensor((M,), accum_dtype),
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
        A_val: T.Tensor((MB, MAX_BLOCKS_PER_ROW, B, B), val_dtype),
        A_colind: T.Tensor(
            (
                MB,
                MAX_BLOCKS_PER_ROW,
            ),
            "int32",
        ),
        x: T.Tensor((N,), x_dtype),
        y: T.Tensor((M,), accum_dtype),
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


@tilelang.jit(
    out_idx=[4],  # y is the 5th argument
    target="cuda",
)
def csr_spmv_a100(
    M: int,
    N: int,
    NNZ: int,
    BLOCK_ROWS: int = 128,  # threads per CTA
    val_dtype: str = "float64",
    x_dtype: str = "float64",
    accum_dtype: str = "float64",
):
    """
    y = A * x where A is in CSR:

      A_data    : (NNZ,)       val_dtype
      A_indices : (NNZ,)       int32
      A_indptr  : (M+1,)       int32
      x         : (N,)         x_dtype
      y         : (M,)         accum_dtype

    Mapping:
      - gridDim.x = ceildiv(M, BLOCK_ROWS)
      - blockDim.x = BLOCK_ROWS
      - thread t in block bx handles row = bx * BLOCK_ROWS + t
    """

    @T.prim_func
    def main(
        A_data: T.Tensor((NNZ,), val_dtype),
        A_indices: T.Tensor((NNZ,), "int32"),
        A_indptr: T.Tensor((M + 1,), "int32"),
        x: T.Tensor((N,), x_dtype),
        y: T.Tensor((M,), accum_dtype),
    ):
        grid_x = T.ceildiv(M, BLOCK_ROWS)

        with T.Kernel(grid_x, threads=BLOCK_ROWS) as bx:
            tid = T.get_thread_binding(0)  # 0..BLOCK_ROWS-1
            row = bx * BLOCK_ROWS + tid

            if row < M:
                row_start = A_indptr[row]
                row_end = A_indptr[row + 1]

                acc = T.alloc_local((1,), accum_dtype)
                T.clear(acc)

                for k in T.serial(row_start, row_end):
                    col = A_indices[k]
                    a_ik = A_data[k].astype(accum_dtype)
                    xk = x[col].astype(accum_dtype)
                    acc[0] += a_ik * xk

                y[row] = acc[0]

    return main
