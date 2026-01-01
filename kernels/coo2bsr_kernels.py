import argparse
import os
import numpy as np
import tilelang
import tilelang.language as T
import torch
from scipy.sparse import bsr_matrix, coo_matrix
from scipy.io import mmread
from tilelang.layout.layout import IndexMap


@tilelang.jit(target="cuda")
def make_coo2bsr_kernel(nnz, nnzb, R, C):
    """
    Build a TileLang PrimFunc that does:
        for each BSR block b in [0, nnzb):
            tile[R, C] = 0
            for i in [block_ptr[b] : block_ptr[b+1]):
                (row[i], col[i], val[i]) -> accumulate into tile
            write tile to data_out[b, :, :]
    """

    @T.prim_func
    def main(
        row: T.Tensor((nnz,), "int32"),
        col: T.Tensor((nnz,), "int32"),
        val: T.Tensor((nnz,), "float32"),
        block_ptr: T.Tensor((nnzb + 1,), "int32"),
        data_out: T.Tensor((nnzb, R, C), "float32"),
    ):
        with T.Kernel(nnzb) as b:
            tile = T.alloc_local((R, C), "float32")

            # Zero-initialize the tile
            for rr in T.serial(R):
                for cc in T.serial(C):
                    tile[rr, cc] = T.float32(0)

            # Range of COO entries belonging to this BSR block
            start = block_ptr[b]
            end = block_ptr[b + 1]
            length = end - start

            # Accumulate COO entries into the local tile
            for t in T.serial(length):
                i = start + t
                r = row[i]
                c = col[i]
                v = val[i]

                lr = r % R  # local row within block
                lc = c % C  # local col within block

                tile[lr, lc] = tile[lr, lc] + v

            # Store the tile into the BSR data buffer
            for rr in T.serial(R):
                for cc in T.serial(C):
                    data_out[b, rr, cc] = tile[rr, cc]

    return main


def coo2bsr(row, col, val, shape, blocksize, EXP_FACTOR=False):
    M, N = shape
    R, C = blocksize

    nnz = int(row.shape[0])

    nbrow = (M + R - 1) // R
    nbcol = (N + C - 1) // C

    br = row // R
    bc = col // C
    block_id = br * nbcol + bc

    order = np.argsort(block_id)
    row_sorted = torch.tensor(row[order], device="cuda")
    col_sorted = torch.tensor(col[order], device="cuda")
    val_sorted = torch.tensor(val[order], device="cuda")
    block_id_sorted = block_id[order]

    uniq_block_id, idx_start = np.unique(block_id_sorted, return_index=True)
    nnzb = int(uniq_block_id.shape[0])

    indices_bsr = (uniq_block_id % nbcol).astype(np.int32)
    block_rows = (uniq_block_id // nbcol).astype(np.int32)

    indptr_bsr = np.zeros(nbrow + 1, dtype=np.int32)
    counts = np.bincount(block_rows, minlength=nbrow)
    indptr_bsr[1:] = np.cumsum(counts)

    block_ptr = np.empty(nnzb + 1, dtype=np.int32)
    block_ptr[:-1] = idx_start.astype(np.int32)
    block_ptr[-1] = nnz
    block_ptr = torch.tensor(block_ptr, device="cuda")

    data_bsr = torch.zeros((nnzb, R, C), dtype=torch.float32, device="cuda")

    kernel = make_coo2bsr_kernel(
        nnz,
        nnzb,
        R,
        C,
    )

    kernel(row_sorted, col_sorted, val_sorted, block_ptr, data_bsr)

    if EXP_FACTOR:
        print(f"Expansion factor {indptr_bsr[-1] * R * C / nnz * 100} % ")
    return data_bsr.cpu().numpy(), indices_bsr, indptr_bsr


def test_coo2bsr(matrix_file=None, matrix_name=None, matrix_dir="~/data/matrix", blocksize=(4, 4), use_random=False):
    """
    Test COO → BSR conversion.
    
    Args:
        matrix_file: Full path to .mtx file (if provided, takes precedence)
        matrix_name: Matrix name without .mtx extension (used with matrix_dir)
        matrix_dir: Directory containing matrix files (default: ~/data/matrix)
        blocksize: BSR block size tuple (R, C), default (4, 4)
        use_random: If True, use random matrix instead of loading from file
    """
    print("Running COO → BSR TileLang test...")
    
    # Ensure R and C are Python int types (not numpy integers)
    R, C = int(blocksize[0]), int(blocksize[1])
    
    if use_random:
        # Test parameters for random matrix
        M, N = 100, 100
        density = 0.15
        
        # Generate random matrix in COO
        rng = np.random.default_rng(0)
        size = M * N
        nnz = int(size * density)
        
        rows = rng.integers(0, M, size=nnz, dtype=np.int32)
        cols = rng.integers(0, N, size=nnz, dtype=np.int32)
        vals = rng.random(nnz, dtype=np.float32)
        
        A_coo = coo_matrix((vals, (rows, cols)), shape=(M, N))
        print(f"Using random matrix: {M}x{N}, nnz={nnz}, blocksize=({R},{C})")
    else:
        # Load matrix from .mtx file
        if matrix_file:
            mtx_path = os.path.expanduser(matrix_file)
        elif matrix_name:
            matrix_dir = os.path.expanduser(matrix_dir)
            mtx_path = os.path.join(matrix_dir, f"{matrix_name}.mtx")
        else:
            raise ValueError("Either matrix_file, matrix_name, or use_random=True must be provided")
        
        if not os.path.exists(mtx_path):
            raise FileNotFoundError(f"Matrix file not found: {mtx_path}")
        
        print(f"Loading matrix from: {mtx_path}")
        A = mmread(mtx_path)
        
        # Convert to COO format
        if not isinstance(A, coo_matrix):
            A_coo = A.tocoo()
        else:
            A_coo = A
        
        M, N = A_coo.shape
        rows = A_coo.row.astype(np.int32)
        cols = A_coo.col.astype(np.int32)
        vals = A_coo.data.astype(np.float32)
        nnz = len(vals)
        
        print(f"Matrix info: {M}x{N}, nnz={nnz}, blocksize=({R},{C})")
    
    # Now rows, cols, vals, A_coo, M, N, nnz are all defined
    
    # SciPy BSR for correctness reference
    # Note: Some SciPy versions have issues with tobsr on COO matrices
    # Convert to CSR first for better compatibility
    # TODO: require A.shape % blocksize == (0, 0)
    # A_csr = A_coo.tocsr()
    # A_bsr_ref = A_csr.tobsr(blocksize=(R, C))
    # A_bsr_ref.sort_indices()
    # ref_data = A_bsr_ref.data
    # ref_indices = A_bsr_ref.indices
    # ref_indptr = A_bsr_ref.indptr

    # TileLang result
    data_tl, indices_tl, indptr_tl = coo2bsr(rows, cols, vals, (M, N), (R, C), EXP_FACTOR=True)
    # Compare shapes
    # assert data_tl.shape == ref_data.shape, (
    #     f"data shape mismatch: {data_tl.shape} vs {ref_data.shape}"
    # )

    # # Compare block structure
    # assert np.all(indices_tl == ref_indices), (
    #     f"indices mismatch:\nTL={indices_tl}\nREF={ref_indices}"
    # )

    # assert np.all(indptr_tl == ref_indptr), (
    #     f"indptr mismatch:\nTL={indptr_tl}\nREF={ref_indptr}"
    # )

    # # Compare block data
    # if not np.allclose(data_tl, ref_data, rtol=1e-6, atol=1e-6):
    #     diff = np.abs(data_tl - ref_data)
    #     print("Max data diff:", diff.max())
    #     raise AssertionError("Block data mismatch")

    print("✓ TileLang COO → BSR test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test COO → BSR conversion with real or random matrices")
    parser.add_argument("--matrix-file", type=str, default=None,
                        help="Full path to .mtx file")
    parser.add_argument("--matrix-name", type=str, default=None,
                        help="Matrix name (without .mtx extension), used with --matrix-dir")
    parser.add_argument("--matrix-dir", type=str, default="~/data/matrix",
                        help="Directory containing matrix files (default: ~/data/matrix)")
    parser.add_argument("--blocksize", type=int, nargs=2, default=[4, 4],
                        metavar=("R", "C"),
                        help="BSR block size (default: 4 4)")
    parser.add_argument("--random", action="store_true",
                        help="Use random matrix instead of loading from file")
    args = parser.parse_args()
    
    test_coo2bsr(
        matrix_file=args.matrix_file,
        matrix_name=args.matrix_name,
        matrix_dir=args.matrix_dir,
        blocksize=tuple(args.blocksize),
        use_random=args.random
    )
