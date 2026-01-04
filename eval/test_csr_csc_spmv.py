"""
Test and benchmark for CSR and CSC SpMV kernels with mixed precision quantization.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.io import mmread
from kernels.csr_prequant import build_csr_from_scipy, quantize_csr_matrix
from kernels.csc_prequant import build_csc_from_scipy, quantize_csc_matrix
from kernels.csr_spmv_kernels import csr_spmv_mixed_prequant
from kernels.csc_spmv_kernels import csc_spmv_mixed_prequant
from kernels.csc_spmv_kernels import make_csc_spmv_kernel
from kernels.kernel_utils import benchmark_kernel


def load_matrix_from_mtx(matrix_name: str, matrix_dir: str = "~/data/matrix"):
    """
    Load matrix from .mtx file.
    
    Args:
        matrix_name: Matrix name (without .mtx extension)
        matrix_dir: Directory containing matrix files (default: ~/data/matrix)
    
    Returns:
        scipy sparse matrix in CSR format
    """
    matrix_path = os.path.expanduser(os.path.join(matrix_dir, f"{matrix_name}.mtx"))
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    
    # Read Matrix Market file
    A = mmread(matrix_path)
    
    # Convert to CSR format
    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    
    return A


def random_coo(M: int, N: int, nnz: int, seed: int = 0):
    """Generate random COO matrix."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, M, size=nnz, dtype=np.int32)
    cols = rng.integers(0, N, size=nnz, dtype=np.int32)
    vals = rng.standard_normal(nnz).astype(np.float64)
    return rows, cols, vals, (M, N)


def test_csr_spmv(matrix_name=None, R=64, C=64):
    """Test CSR SpMV kernel correctness."""
    print("=" * 60)
    print("Testing CSR SpMV kernel...")
    print("=" * 60)

    if matrix_name:
        print(f"Loading matrix: {matrix_name}")
        A_csr = load_matrix_from_mtx(matrix_name)
        M, N = A_csr.shape
        print(f"  Matrix size: {M} x {N}, nnz: {A_csr.nnz}")
    else:
        # Use random matrix for testing
        M, N = 1024, 1024
        density = 0.05
        nnz = int(M * N * density)
        rows, cols, vals, shape = random_coo(M, N, nnz, seed=0)
        A_coo = coo_matrix((vals, (rows, cols)), shape=shape)
        A_csr = A_coo.tocsr()
        print(f"Using random matrix: {M} x {N}, nnz: {nnz}")

    # Build CSR matrix
    csr = build_csr_from_scipy(A_csr, R=R, C=C, device="cuda")
    csr = quantize_csr_matrix(csr)
    print(f"  Number of tiles: {csr.n_tiles}")

    # Generate random x
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N).astype(np.float64)

    # Reference result
    y_ref = A_csr @ x

    # Test with different actions
    # Actions are per block-column, not per tile
    n_bc = (csr.N + csr.C - 1) // csr.C
    for action_name, action_val in [("fp64", 0), ("fp32", 1), ("bf16", 2)]:
        print(f"\nTesting action={action_val} ({action_name})...")
        actions = np.full((n_bc,), action_val, dtype=np.int32)
        y_kernel = csr_spmv_mixed_prequant(csr, actions, x, device="cuda", return_torch=False)

        max_err = np.max(np.abs(y_kernel - y_ref))
        rel_err = np.max(np.abs(y_kernel - y_ref) / (np.abs(y_ref) + 1e-12))
        print(f"  Max absolute error: {max_err:.2e}")
        print(f"  Max relative error: {rel_err:.2e}")

        # For fp64, expect exact match (within numerical precision)
        # For fp32/bf16, expect reasonable error
        if action_val == 0:
            assert np.allclose(y_kernel, y_ref, rtol=1e-10, atol=1e-12), f"fp64 test failed"
        elif action_val == 1:
            assert np.allclose(y_kernel, y_ref, rtol=1e-2, atol=1e-2), f"fp32 test failed"
        else:  # bf16
            assert np.allclose(y_kernel, y_ref, rtol=1e2, atol=1e2), f"bf16 test failed"

    print("\n✓ CSR SpMV tests passed!")


def test_csc_spmv(matrix_name=None, R=64, C=64):
    """Test CSC SpMV kernel correctness."""
    print("\n" + "=" * 60)
    print("Testing CSC SpMV kernel...")
    print("=" * 60)

    if matrix_name:
        print(f"Loading matrix: {matrix_name}")
        A_csr = load_matrix_from_mtx(matrix_name)
        A_csc = A_csr.tocsc()
        M, N = A_csc.shape
        print(f"  Matrix size: {M} x {N}, nnz: {A_csc.nnz}")
    else:
        # Use random matrix for testing
        M, N = 1024, 1024
        density = 0.05
        nnz = int(M * N * density)
        rows, cols, vals, shape = random_coo(M, N, nnz, seed=0)
        A_coo = coo_matrix((vals, (rows, cols)), shape=shape)
        A_csc = A_coo.tocsc()
        print(f"Using random matrix: {M} x {N}, nnz: {nnz}")

    # Build CSC matrix
    csc = build_csc_from_scipy(A_csc, R=R, C=C, device="cuda")
    csc = quantize_csc_matrix(csc)
    print(f"  Number of tiles: {csc.n_tiles}")

    # Generate random x
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N).astype(np.float64)

    # Reference result
    y_ref = A_csc @ x

    # Test with different actions
    # Actions are per block-column, not per tile
    n_bc = (csc.N + csc.C - 1) // csc.C
    for action_name, action_val in [("fp64", 0), ("fp32", 1), ("bf16", 2)]:
        print(f"\nTesting action={action_val} ({action_name})...")
        actions = np.full((n_bc,), action_val, dtype=np.int32)
        
        y_kernel = csc_spmv_mixed_prequant(csc, actions, x, device="cuda", return_torch=False)        
        # Compare kernel vs scipy (for fp64 only)
        max_err = np.max(np.abs(y_kernel - y_ref))
        rel_err = np.max(np.abs(y_kernel - y_ref) / (np.abs(y_ref) + 1e-12))
        print(f"  Kernel vs scipy: max_err={max_err:.2e}, rel_err={rel_err:.2e}")

        # For fp64, expect exact match (within numerical precision)
        # For fp32/bf16, expect reasonable error
        if action_val == 0:
            assert np.allclose(y_kernel, y_ref, rtol=1e-10, atol=1e-12), f"fp64 kernel vs scipy failed"
        elif action_val == 1:
            assert np.allclose(y_kernel, y_ref, rtol=1e-2, atol=1e-2), f"fp32 kernel vs scipy failed"
        else:  # bf16
            assert np.allclose(y_kernel, y_ref, rtol=1e2, atol=1e2), f"bf16 kernel vs scipy failed"

    print("\n✓ CSC SpMV tests passed!")


def test_csc_spmv_fp64_cuda_style(matrix_name=None):
    """
    Test pure-fp64 CSC SpMV kernel that matches the provided CUDA CSC_SpMV_kernel style:
      - grid-stride over columns
      - iterate nnz within each column
      - atomicAdd into y[row]
    """
    print("\n" + "=" * 60)
    print("Testing CSC fp64 (CUDA-style) kernel...")
    print("=" * 60)

    if matrix_name:
        print(f"Loading matrix: {matrix_name}")
        A_csr = load_matrix_from_mtx(matrix_name)
        A_csc = A_csr.tocsc()
        M, N = A_csc.shape
        print(f"  Matrix size: {M} x {N}, nnz: {A_csc.nnz}")
    else:
        # Use random matrix for testing
        M, N = 1024, 1024
        density = 0.05
        nnz = int(M * N * density)
        rows, cols, vals, shape = random_coo(M, N, nnz, seed=0)
        A_coo = coo_matrix((vals, (rows, cols)), shape=shape)
        A_csc = A_coo.tocsc()
        print(f"Using random matrix: {M} x {N}, nnz: {nnz}")

    # Build CSC matrix (fp64 values only; no quantization needed)
    csc = build_csc_from_scipy(A_csc, R=64, C=64, device="cuda")

    # Generate random x
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(csc.N)).astype(np.float64)
    x_t = torch.as_tensor(x, dtype=torch.float64, device="cuda")

    # Reference result
    y_ref = A_csc @ x

    # Build and run kernel (signature: colptr, rowind, values, x, y)
    kernel = make_csc_spmv_kernel(
        int(csc.M),
        int(csc.N),
        int(csc.nnz),
    )
    y_t = torch.zeros((int(csc.M),), dtype=torch.float64, device="cuda")
    kernel(csc.colptr, csc.rowind, csc.A_fp64, x_t, y_t)
    y_kernel = y_t.detach().cpu().numpy()

    max_err = np.max(np.abs(y_kernel - y_ref))
    rel_err = np.max(np.abs(y_kernel - y_ref) / (np.abs(y_ref) + 1e-12))
    print(f"  Kernel vs scipy: max_err={max_err:.2e}, rel_err={rel_err:.2e}")
    assert np.allclose(y_kernel, y_ref, rtol=1e-10, atol=1e-12), "fp64 CUDA-style CSC kernel vs scipy failed"

    print("\n✓ CSC fp64 (CUDA-style) test passed!")


def benchmark_csr_spmv(matrix_name=None, R=64, C=64):
    """Benchmark CSR SpMV kernel performance."""
    print("\n" + "=" * 60)
    print("Benchmarking CSR SpMV kernel...")
    print("=" * 60)

    if matrix_name:
        print(f"Loading matrix: {matrix_name}")
        A_csr = load_matrix_from_mtx(matrix_name)
        M, N = A_csr.shape
        print(f"  Matrix size: {M} x {N}, nnz: {A_csr.nnz}")
    else:
        # Use random matrix for testing
        M, N = 4096, 4096
        density = 0.05
        nnz = int(M * N * density)
        rows, cols, vals, shape = random_coo(M, N, nnz, seed=0)
        A_coo = coo_matrix((vals, (rows, cols)), shape=shape)
        A_csr = A_coo.tocsr()
        print(f"Using random matrix: {M} x {N}, nnz: {nnz}")

    # Build CSR matrix
    csr = build_csr_from_scipy(A_csr, R=R, C=C, device="cuda")
    csr = quantize_csr_matrix(csr)
    print(f"  Number of tiles: {csr.n_tiles}")

    # Generate random x
    rng = np.random.default_rng(1)
    N = csr.N
    x = rng.standard_normal(N).astype(np.float64)
    x_t = torch.as_tensor(x, dtype=torch.float64, device="cuda")

    # Test different actions
    # Actions are per block-column, not per tile
    n_bc = (csr.N + csr.C - 1) // csr.C
    for action_name, action_val in [("fp64", 0), ("fp32", 1), ("bf16", 2)]:
        print(f"\nBenchmarking action={action_val} ({action_name})...")
        actions = np.full((n_bc,), action_val, dtype=np.int32)
        actions_t = torch.as_tensor(actions, dtype=torch.int32, device="cuda")

        from kernels.csr_spmv_kernels import make_csr_spmv_mixed_prequant_kernel_warp_reduce

        n_br = (csr.M + csr.R - 1) // csr.R
        kernel = make_csr_spmv_mixed_prequant_kernel_warp_reduce(
            int(csr.M),
            int(csr.N),
            int(csr.nnz),
            n_br,
            n_bc,
            int(csr.R),
            int(csr.C),
        )

        y_t = torch.zeros((int(csr.M),), dtype=torch.float64, device="cuda")

        kernel_args = (
            csr.A_fp64,
            csr.A_fp32_q,
            csr.a_scale_fp32,
            csr.A_bf16_q,
            csr.a_scale_bf16,
            actions_t,
            csr.rowptr,
            csr.colind,
            x_t,
            y_t,
        )

        benchmark_kernel(kernel, kernel_args, csr.nnz, warmup=1000, iters=10000)


def benchmark_csc_spmv(matrix_name=None, R=64, C=64):
    """Benchmark CSC SpMV kernel performance."""
    print("\n" + "=" * 60)
    print("Benchmarking CSC SpMV kernel...")
    print("=" * 60)

    if matrix_name:
        print(f"Loading matrix: {matrix_name}")
        A_csr = load_matrix_from_mtx(matrix_name)
        A_csc = A_csr.tocsc()
        M, N = A_csc.shape
        print(f"  Matrix size: {M} x {N}, nnz: {A_csc.nnz}")
    else:
        # Use random matrix for testing
        M, N = 4096, 4096
        density = 0.05
        nnz = int(M * N * density)
        rows, cols, vals, shape = random_coo(M, N, nnz, seed=0)
        A_coo = coo_matrix((vals, (rows, cols)), shape=shape)
        A_csc = A_coo.tocsc()
        print(f"Using random matrix: {M} x {N}, nnz: {nnz}")

    # Build CSC matrix
    csc = build_csc_from_scipy(A_csc, R=R, C=C, device="cuda")
    csc = quantize_csc_matrix(csc)
    print(f"  Number of tiles: {csc.n_tiles}")

    # Generate random x
    rng = np.random.default_rng(1)
    N = csc.N
    x = rng.standard_normal(N).astype(np.float64)
    x_t = torch.as_tensor(x, dtype=torch.float64, device="cuda")

    # Test different actions
    # Actions are per block-column, not per tile
    n_bc = (csc.N + csc.C - 1) // csc.C
    n_br = (csc.M + csc.R - 1) // csc.R
    for action_name, action_val in [("fp64", 0), ("fp32", 1), ("bf16", 2)]:
        print(f"\nBenchmarking action={action_val} ({action_name})...")
        actions = np.full((n_bc,), action_val, dtype=np.int32)
        actions_t = torch.as_tensor(actions, dtype=torch.int32, device="cuda")

        from kernels.csc_spmv_kernels import make_csc_spmv_mixed_prequant_kernel_warp_reduce

        kernel = make_csc_spmv_mixed_prequant_kernel_warp_reduce(
            int(csc.M),
            int(csc.N),
            int(csc.nnz),
            int(csc.n_tiles),
            int(csc.R),
            int(csc.C),
            n_bc,
            n_br,
        )

        y_t = torch.zeros((int(csc.M),), dtype=torch.float64, device="cuda")

        kernel_args = (
            csc.A_fp64,
            csc.A_fp32_q,
            csc.a_scale_fp32,
            csc.A_bf16_q,
            csc.a_scale_bf16,
            actions_t,
            csc.colptr,
            csc.rowind,
            x_t,
            y_t,
        )

        benchmark_kernel(kernel, kernel_args, csc.nnz, warmup=1000, iters=10000)


def benchmark_csc_spmv_fp64_cuda_style(matrix_name=None):
    """
    Benchmark pure-fp64 CSC SpMV kernel (CUDA-style):
      - grid-stride over columns
      - iterate nnz within each column
      - atomicAdd into y[row]
    """
    print("\n" + "=" * 60)
    print("Benchmarking CSC fp64 (CUDA-style) kernel...")
    print("=" * 60)

    if matrix_name:
        print(f"Loading matrix: {matrix_name}")
        A_csr = load_matrix_from_mtx(matrix_name)
        A_csc = A_csr.tocsc()
        M, N = A_csc.shape
        print(f"  Matrix size: {M} x {N}, nnz: {A_csc.nnz}")
    else:
        # Use random matrix for benchmarking
        M, N = 4096, 4096
        density = 0.05
        nnz = int(M * N * density)
        rows, cols, vals, shape = random_coo(M, N, nnz, seed=0)
        A_coo = coo_matrix((vals, (rows, cols)), shape=shape)
        A_csc = A_coo.tocsc()
        print(f"Using random matrix: {M} x {N}, nnz: {nnz}")

    # Build CSC (fp64 only)
    csc = build_csc_from_scipy(A_csc, R=64, C=64, device="cuda")

    # Random x
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(csc.N)).astype(np.float64)
    x_t = torch.as_tensor(x, dtype=torch.float64, device="cuda")

    # Build kernel (signature: colptr, rowind, values, x, y)
    n_bc = (int(csc.N) + int(csc.C) - 1) // int(csc.C)
    n_br = (int(csc.M) + int(csc.R) - 1) // int(csc.R)
    kernel = make_csc_spmv_kernel(
        int(csc.M),
        int(csc.N),
        int(csc.nnz),
    )

    y_t = torch.zeros((int(csc.M),), dtype=torch.float64, device="cuda")
    kernel_args = (csc.colptr, csc.rowind, csc.A_fp64, x_t, y_t)

    benchmark_kernel(kernel, kernel_args, csc.nnz, warmup=5, iters=10)


def test_multiple_matrices(matrix_names=None, R=64, C=64):
    """Test and benchmark CSR and CSC SpMV on multiple real matrices."""
    if matrix_names is None:
        # Default: test a few small matrices
        matrix_names = ["1138_bus", "494_bus", "662_bus"]
    
    print("\n" + "=" * 60)
    print("Testing and benchmarking on multiple real matrices...")
    print("=" * 60)
    
    for matrix_name in matrix_names:
        try:
            print(f"\n{'='*60}")
            print(f"Testing matrix: {matrix_name}")
            print(f"{'='*60}")
            test_csr_spmv(matrix_name=matrix_name, R=R, C=C)
            test_csc_spmv(matrix_name=matrix_name, R=R, C=C)
            benchmark_csr_spmv(matrix_name=matrix_name, R=R, C=C)
            benchmark_csc_spmv(matrix_name=matrix_name, R=R, C=C)
        except Exception as e:
            print(f"  ✗ Failed on {matrix_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("All matrix tests and benchmarks completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CSR/CSC SpMV kernels")
    parser.add_argument("--matrix_name", type=str, default=None, help="Matrix name (without .mtx)")
    parser.add_argument("--matrices", type=str, nargs="+", default=None, help="List of matrix names")
    parser.add_argument("--R", type=int, default=64, help="Tile row size")
    parser.add_argument("--C", type=int, default=64, help="Tile column size")
    parser.add_argument("--all-small", action="store_true", help="Test all small matrices (< 1MB)")
    
    args = parser.parse_args()
    
    if args.all_small:
        # Find all small matrices (< 1MB)
        matrix_dir = os.path.expanduser("~/data/matrix")
        small_matrices = []
        for fname in os.listdir(matrix_dir):
            if fname.endswith(".mtx"):
                fpath = os.path.join(matrix_dir, fname)
                if os.path.getsize(fpath) < 1024 * 1024:  # < 1MB
                    small_matrices.append(fname[:-4])  # Remove .mtx
        print(f"Found {len(small_matrices)} small matrices")
        test_multiple_matrices(matrix_names=small_matrices[:10], R=args.R, C=args.C)  # Test first 10
    elif args.matrices:
        test_multiple_matrices(matrix_names=args.matrices, R=args.R, C=args.C)
    elif args.matrix_name:
        # Test single matrix
        # test_csr_spmv(matrix_name=args.matrix_name, R=args.R, C=args.C)
        # test_csc_spmv(matrix_name=args.matrix_name, R=args.R, C=args.C)
        # test_csc_spmv_fp64_cuda_style(matrix_name=args.matrix_name)
        # benchmark_csr_spmv(matrix_name=args.matrix_name, R=args.R, C=args.C)
        # benchmark_csc_spmv(matrix_name=args.matrix_name, R=args.R, C=args.C)
        benchmark_csc_spmv_fp64_cuda_style(matrix_name=args.matrix_name)
        
    else:
        # Default: test with random matrix
        # test_csr_spmv(R=args.R, C=args.C)
        test_csc_spmv(R=args.R, C=args.C)
        test_csc_spmv_fp64_cuda_style()
        # benchmark_csr_spmv(R=args.R, C=args.C)
        benchmark_csc_spmv(R=args.R, C=args.C)
        benchmark_csc_spmv_fp64_cuda_style()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

