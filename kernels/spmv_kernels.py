import numpy as np
import tilelang
import tilelang.language as T
import torch
from scipy.sparse import coo_matrix


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
        data: T.Tensor((nnzb, R, C), "float32"),
        indices: T.Tensor((nnzb,), "int32"),
        indptr: T.Tensor((n_block_rows + 1,), "int32"),
        x: T.Tensor((N,), "float32"),
        y: T.Tensor((n_block_rows * R,), "float64"),
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

    @T.prim_func
    def main(
        data: T.Tensor((nnzb, R, C), "float64"),
        actions: T.Tensor((N // C,), "int32"),
        indices: T.Tensor((nnzb,), "int32"),
        indptr: T.Tensor((n_block_rows + 1,), "int32"),
        x: T.Tensor((N,), "float64"),
        y: T.Tensor((n_block_rows * R,), "float64"),
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
                action = actions[bc]

                for rr in T.serial(R):
                    for cc in T.serial(C):
                        if action == 0:
                            y_local[rr] += data[bk, rr, cc] * x[x_base + cc]
                        elif action == 1:
                            y_local[rr] += data[bk, rr, cc].astype("float32") * x[
                                x_base + cc
                            ].astype("float32")
                        elif action == 2:
                            y_local[rr] += data[bk, rr, cc].astype("float16") * x[
                                x_base + cc
                            ].astype("float16")
                        elif action == 3:
                            y_local[rr] += data[bk, rr, cc].astype("float8_e4m3") * x[
                                x_base + cc
                            ].astype("float8_e4m3")

            # Write the accumulated result back to y
            row_base = br * R
            for rr in T.serial(R):
                y[row_base + rr] = y_local[rr]

    return main


def bsr_spmv_mixed(data_bsr, actions, indices_bsr, indptr_bsr, x, R, C, device="cuda"):
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
    spmv_kernel = make_bsr_spmv_mixed_kernel(n_block_rows, nnzb, R, C, N)

    # Run kernel
    spmv_kernel(data_t, actions_t, indices_t, indptr_t, x_t, y_t)

    # Back to numpy
    y = y_t.cpu().numpy()
    return y


def test_bsr_spmv():
    print("Running BSR spmv TileLang test...")
    M, N = 100, 100
    R, C = 4, 4  # BSR tile shape
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

    y_ref = A_bsr @ x
    y_tl = bsr_spmv(ref_data, ref_indices, ref_indptr, x, R, C, device="cuda")

    print(y_tl)
    print(y_ref)
    assert np.allclose(y_tl, y_ref, rtol=1e-5, atol=1e-6)
    print("BSR SpMV kernel matches SciPy.")


def test_bsr_spmv_mixed():
    print("Running BSR mixed spmv TileLang test...")
    M, N = 100, 100
    R, C = 4, 4  # BSR tile shape
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

    y_ref = A_bsr @ x

    # actions = np.full((N // C,), 3)
    actions = np.random.randint(0, 2, size=N // C)
    y_tl = bsr_spmv_mixed(
        ref_data, actions, ref_indices, ref_indptr, x, R, C, device="cuda"
    )

    print(y_tl)
    print(y_ref)
    assert np.allclose(y_tl, y_ref, rtol=1e-5, atol=1e-6)
    print("BSR SpMV kernel matches SciPy.")


if __name__ == "__main__":
    # test_bsr_spmv()
    test_bsr_spmv_mixed()
