import numpy as np
import time
import tilelang
import tilelang.language as T
import torch
from scipy.io import mmread
from .kernel_utils import benchmark_kernel


@tilelang.jit(target="cuda")
def make_fused_cg_step_kernel(
    N,
    grid_size,
    dtype="float64",
    accum_dtype="float64",
    block_size=64,
):
    """
    Fused kernel for CG iteration step: compute aj, update x and r, then compute βj and update p.

    This kernel performs:
    1. aj = (r·r) / (μ·p)  [dot_div]
    2. x = x + aj * p      [axpy]
    3. r_new = r - aj * μ  [axpy]
    4. βj = ||r_new||² / ||r||²
    5. p = r_new + βj * p  [axpy]

    Inputs:
    - r: residual vector (N,)
    - mu: Ap vector (N,)
    - p: search direction (N,)
    - x: solution vector (N,) [in/out]
    Outputs:
    - x: updated solution
    - r: updated residual (r_new)
    - p: updated search direction
    - stats: [||r||², ||r_new||², aj] for convergence/debug
    """

    @T.prim_func
    def main(
        r: T.Tensor((N,), dtype),           # type: ignore  # residual vector [in/out]
        mu: T.Tensor((N,), dtype),          # type: ignore  # A*p vector
        p: T.Tensor((N,), dtype),           # type: ignore  # search direction [in/out]
        x: T.Tensor((N,), dtype),           # type: ignore  # solution vector [in/out]
        stats: T.Tensor((3,), accum_dtype),  # type: ignore  # [||r||², ||r_new||², aj]
    ):
        # NOTE:
        # The original version launched multiple CTAs (grid_size > 1) and only performed
        # *block-local* reductions for r·r and mu·p, which makes each block compute a
        # different aj / beta. That breaks CG math and leads to different iteration counts
        # vs the reference `pytorch_cg`.
        #
        # To make the computation mathematically identical to `pytorch_cg` without needing
        # grid-wide synchronization, we run a single CTA and let each thread iterate over
        # the vector with a stride of block_size. This ensures r·r, mu·p, and r_new·r_new
        # are true *global* dot-products.
        #
        # We also match PyTorch's tiny-denominator guards (1e-307) for aj and beta.

        # Shared memory for dot product reductions
        smem_rr = T.alloc_shared((block_size,), accum_dtype)  # r·r
        smem_mu_p = T.alloc_shared((block_size,), accum_dtype)  # μ·p
        smem_r_new_sq = T.alloc_shared((block_size,), accum_dtype)  # ||r_new||²
        smem_scalars = T.alloc_shared((2,), accum_dtype)  # [aj, βj] for broadcasting

        # Single CTA kernel (see note above). Keep `grid_size` argument for API
        # compatibility but do not use it.
        with T.Kernel(1, threads=block_size) as bx:
            tx = T.get_thread_binding(0)

            # Thread-local accumulators for dot products
            acc_rr = T.alloc_local((1,), accum_dtype)
            acc_mu_p = T.alloc_local((1,), accum_dtype)
            acc_r_new_sq = T.alloc_local((1,), accum_dtype)
            T.clear(acc_rr)
            T.clear(acc_mu_p)
            T.clear(acc_r_new_sq)

            # Compute local contributions to initial dot products (global strided loop)
            iters = (N + block_size - 1) // block_size
            for t in T.serial(iters):
                idx = t * block_size + tx
                if idx < N:
                    if dtype == accum_dtype:
                        acc_rr[0] += r[idx] * r[idx]
                        acc_mu_p[0] += mu[idx] * p[idx]
                    else:
                        rr = r[idx].astype(accum_dtype)
                        acc_rr[0] += rr * rr
                        acc_mu = mu[idx].astype(accum_dtype)
                        acc_p = p[idx].astype(accum_dtype)
                        acc_mu_p[0] += acc_mu * acc_p

            # Store to shared memory
            smem_rr[tx] = acc_rr[0]
            smem_mu_p[tx] = acc_mu_p[0]
            smem_r_new_sq[tx] = 0.0  # Initialize for later use

            # Block reduction: tree-based sum for initial dot products
            if block_size > 32:
                T.sync_threads()

            # Reduce to 32 threads
            if block_size >= 512:
                if tx < 256:
                    smem_rr[tx] += smem_rr[tx + 256]
                    smem_mu_p[tx] += smem_mu_p[tx + 256]
                T.sync_threads()
            if block_size >= 256:
                if tx < 128:
                    smem_rr[tx] += smem_rr[tx + 128]
                    smem_mu_p[tx] += smem_mu_p[tx + 128]
                T.sync_threads()
            if block_size >= 128:
                if tx < 64:
                    smem_rr[tx] += smem_rr[tx + 64]
                    smem_mu_p[tx] += smem_mu_p[tx + 64]
                T.sync_threads()

            # Warp-synchronous reduction
            if block_size >= 64:
                if tx < 32:
                    smem_rr[tx] += smem_rr[tx + 32]
                    smem_mu_p[tx] += smem_mu_p[tx + 32]
            if tx < 16:
                smem_rr[tx] += smem_rr[tx + 16]
                smem_mu_p[tx] += smem_mu_p[tx + 16]
            if tx < 8:
                smem_rr[tx] += smem_rr[tx + 8]
                smem_mu_p[tx] += smem_mu_p[tx + 8]
            if tx < 4:
                smem_rr[tx] += smem_rr[tx + 4]
                smem_mu_p[tx] += smem_mu_p[tx + 4]
            if tx < 2:
                smem_rr[tx] += smem_rr[tx + 2]
                smem_mu_p[tx] += smem_mu_p[tx + 2]
            if tx < 1:
                smem_rr[tx] += smem_rr[tx + 1]
                smem_mu_p[tx] += smem_mu_p[tx + 1]

            # Thread 0 computes aj and stores in shared memory
            if tx == 0:
                # Get reduced dot products
                r_dot_r = smem_rr[0]      # ||r||²
                mu_dot_p = smem_mu_p[0]   # μ·p

                # Save ||r||²
                stats[0] = r_dot_r

                # Compute aj = (r·r) / (μ·p), match `pytorch_cg`'s tiny-denom guard
                # denom_ok = abs(mu_dot_p) > 1e-307
                denom_ok = T.abs(mu_dot_p) > T.Cast(accum_dtype, 1e-307)
                smem_scalars[0] = T.if_then_else(
                    denom_ok,
                    r_dot_r / mu_dot_p,
                    T.Cast(accum_dtype, 0.0),
                )
                stats[2] = smem_scalars[0]

            # Sync before vector operations
            T.sync_threads()

            # First set of vector updates, and accumulate ||r_new||² in a second strided loop
            aj = smem_scalars[0]  # broadcast scalar
            T.clear(acc_r_new_sq)
            for t in T.serial(iters):
                idx = t * block_size + tx
                if idx < N:
                    # x = x + aj * p
                    x[idx] = x[idx] + aj * p[idx]

                    # r = r - aj * μ (compute r_new)
                    r_new_val = r[idx] - aj * mu[idx]
                    r[idx] = r_new_val

                    # Accumulate ||r_new||² = dot(r_new, r_new)
                    if dtype == accum_dtype:
                        acc_r_new_sq[0] += r_new_val * r_new_val
                    else:
                        rrn = r_new_val.astype(accum_dtype)
                        acc_r_new_sq[0] += rrn * rrn

            # Store ||r_new||² contributions to shared memory
            smem_r_new_sq[tx] = acc_r_new_sq[0]

            # Block reduction for ||r_new||²
            if block_size > 32:
                T.sync_threads()

            if block_size >= 512:
                if tx < 256:
                    smem_r_new_sq[tx] += smem_r_new_sq[tx + 256]
                T.sync_threads()
            if block_size >= 256:
                if tx < 128:
                    smem_r_new_sq[tx] += smem_r_new_sq[tx + 128]
                T.sync_threads()
            if block_size >= 128:
                if tx < 64:
                    smem_r_new_sq[tx] += smem_r_new_sq[tx + 64]
                T.sync_threads()

            if block_size >= 64:
                if tx < 32:
                    smem_r_new_sq[tx] += smem_r_new_sq[tx + 32]
            if tx < 16:
                smem_r_new_sq[tx] += smem_r_new_sq[tx + 16]
            if tx < 8:
                smem_r_new_sq[tx] += smem_r_new_sq[tx + 8]
            if tx < 4:
                smem_r_new_sq[tx] += smem_r_new_sq[tx + 4]
            if tx < 2:
                smem_r_new_sq[tx] += smem_r_new_sq[tx + 2]
            if tx < 1:
                smem_r_new_sq[tx] += smem_r_new_sq[tx + 1]

            # Thread 0 computes βj and stores in shared memory
            if tx == 0:
                r_new_norm_sq_val = smem_r_new_sq[0]
                stats[1] = r_new_norm_sq_val
                # βj = (r_new, r_new) / (r_old, r_old), match `pytorch_cg` guard:
                # beta = where(abs(rj_dot_rj) > 1e-307, rj_dot_rj_new / rj_dot_rj, 0)
                denom_ok_beta = T.abs(stats[0]) > T.Cast(accum_dtype, 1e-307)
                smem_scalars[1] = T.if_then_else(
                    denom_ok_beta,
                    r_new_norm_sq_val / stats[0],
                    T.Cast(accum_dtype, 0.0),
                )

            # Sync before final vector update
            T.sync_threads()

            # Final vector update: p = r_new + beta * p_old
            beta = smem_scalars[1]
            for t in T.serial(iters):
                idx = t * block_size + tx
                if idx < N:
                    p[idx] = r[idx] + beta * p[idx]

    return main


def fused_cg_step(
    r, mu, p, x,
    device="cuda",
    *,
    block_size=64,
):
    """
    Perform one fused CG iteration step.

    Args:
        r, mu, p, x: torch.Tensor vectors (modified in-place)
        device: target device
        block_size: CUDA block size

    Returns:
        stats: [||r||², ||r_new||², aj]
    """
    N = r.shape[0]
    dtype = "float64"  # Assume float64 for now

    stats = torch.zeros((3,), dtype=torch.float64, device=device)

    # Build and run kernel
    grid_size = (N + block_size - 1) // block_size
    fused_kernel = make_fused_cg_step_kernel(
        N=N,
        grid_size=grid_size,
        dtype=dtype,
        accum_dtype="float64",
        block_size=block_size,
    )

    fused_kernel(r, mu, p, x, stats)

    return stats


def fused_cg(
    A_func,
    b,
    x0,
    device="cuda",
    max_iter=1000,
    tol=1e-10,
    *,
    block_size=256,
    use_concatenated=True,
):
    """
    Fused Conjugate Gradient solver using optimized TileLang kernels for all operations except SpMV.

    Solves Ax = b using the CG algorithm with fused dot_div and AXPY operations.

    Args:
        A_func: Function that computes A*x for a given x (SpMV operation)
        b: Right-hand side vector
        x0: Initial guess (default: zeros)
        device: Target device ("cuda" or "cpu")
        max_iter: Maximum iterations
        tol: Convergence tolerance
        block_size: CUDA block size for dot_div and AXPY kernels
        use_concatenated: Use concatenated dot_div kernel

    Returns:
        Tuple of (solution x, residual norm history, convergence flag)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fused CG kernel")

    dev = torch.device(device)

    # Convert inputs to torch tensors
    if not torch.is_tensor(b):
        b = torch.from_numpy(b).to(dev, dtype=torch.float64)
    else:
        b = b.to(dev, dtype=torch.float64)

    N = b.shape[0]
    if not torch.is_tensor(x0):
        x0 = torch.from_numpy(x0).to(dev, dtype=torch.float64)
    else:
        x0 = x0.to(dev, dtype=torch.float64)

    # Initialize CG variables
    x = x0.clone()
    r = b.clone()

    # r0 = b - Ax0
    Ax0 = A_func(x0)
    r.sub_(Ax0)

    p = r.clone()

    # Convergence tracking
    residual_norms = []
    residual_norm = torch.norm(r).item()
    residual_norms.append(residual_norm)

    b_norm = torch.norm(b).item()
    converged = (residual_norm / b_norm) < tol

    if converged:
        return x, residual_norms, True

    for iteration in range(max_iter):
        # μ = Ap
        mu = A_func(p)

        # Fused CG step: compute aj, βj and update x, r, p in one kernel
        stats = fused_cg_step(r, mu, p, x, device=device, block_size=block_size)

        # Extract residual norm
        residual_norm = stats[1].item() ** 0.5
        residual_norms.append(residual_norm)

        converged = (residual_norm / b_norm) < tol

        if converged:
            return x, residual_norms, True

    return x, residual_norms, False


def pytorch_cg(
    A_func,
    b,
    x0,
    max_iter=1000,
    tol=1e-10,
):
    """
    PyTorch reference implementation of Conjugate Gradient.

    Args:
        A_func: Function that computes A*x for a given x
        b: Right-hand side vector
        x0: Initial guess
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (solution x, residual norm history, convergence flag)
    """
    if not torch.is_tensor(b):
        b = torch.from_numpy(b).to(torch.float64)
    else:
        b = b.to(torch.float64)

    if not torch.is_tensor(x0):
        x0 = torch.from_numpy(x0).to(torch.float64)
    else:
        x0 = x0.to(torch.float64)

    # Initialize CG variables
    x = x0.clone()
    r = b.clone()

    # r0 = b - Ax0
    Ax0 = A_func(x0)
    r.sub_(Ax0)

    p = r.clone()

    # Convergence tracking (keep dot products on device for numerical stability)
    residual_norms = []
    rj_dot_rj = torch.dot(r, r)
    residual_norm = torch.sqrt(rj_dot_rj).item()
    residual_norms.append(residual_norm)

    b_norm = torch.norm(b).item()

    converged = (residual_norm / b_norm) < tol

    if converged:
        return x, residual_norms, True

    for iteration in range(max_iter):
        # μ = Ap
        mu = A_func(p)

        # aj = (r·r) / (μ·p)
        mu_dot_p = torch.dot(mu, p)
        # Guard extremely small denominator to avoid NaNs
        denom_ok = torch.abs(mu_dot_p) > 1e-307
        aj = torch.where(denom_ok, rj_dot_rj / mu_dot_p, torch.zeros_like(rj_dot_rj))

        # x = x + aj * p
        x.add_(aj * p)

        # r = r - aj * mu
        r.sub_(aj * mu)

        # Check convergence
        rj_dot_rj_new = torch.dot(r, r)
        residual_norm = torch.sqrt(rj_dot_rj_new).item()
        residual_norms.append(residual_norm)

        converged = (residual_norm / b_norm) < tol

        if converged:
            return x, residual_norms, True

        # βj = (r_new·r_new) / (r_old·r_old)
        beta = torch.where(torch.abs(rj_dot_rj) > 1e-307, rj_dot_rj_new / rj_dot_rj, torch.zeros_like(rj_dot_rj))

        # p = r + beta * p
        p = r + beta * p

        # roll forward
        rj_dot_rj = rj_dot_rj_new

    return x, residual_norms, False


def test_fused_cg():
    """Test fused CG kernel against PyTorch reference."""
    print("Testing fused CG kernel...")

    if not torch.cuda.is_available():
        print("CUDA is not available; skipping test_fused_cg().")
        return

    torch.manual_seed(42)
    dev = torch.device("cuda")

    # Load matrix from file
    print("Loading matrix from ~/data/matrix/bundle1.mtx...")
    A_sparse = mmread('/home/bingxing2/home/scx7axu/data/matrix/bundle1.mtx')
    A_dense = torch.from_numpy(A_sparse.toarray()).to(dev, dtype=torch.float64)
    N = A_dense.shape[0]
    print(f"Matrix size: {N}x{N}, nnz: {A_sparse.nnz}")

    # Create right-hand side: sum of each column
    b = A_dense.sum(dim=0)  # b_i = sum_j A_ji

    # x initialized to zeros
    x0 = torch.zeros((N,), device=dev, dtype=torch.float64)

    print(f"b norm: {torch.norm(b):.6f}")
    # print(f"Condition number estimate: {torch.linalg.cond(A_dense):.2e}")

    # Test CG with this matrix
    print(f"\n=== Testing with bundle1 matrix (N={N}) ===")

    # Test PyTorch reference
    def A_func(x):
        return A_dense @ x

    x_ref, residuals_ref, converged_ref = pytorch_cg(
        A_func, b, x0, max_iter=1000, tol=1e-10
    )

    # Test fused CG
    x_fused, residuals_fused, converged_fused = fused_cg(
        A_func, b, x0, device="cuda", max_iter=1000, tol=1e-10
    )


    print(f"  Fused CG converged: {converged_fused}, iterations: {len(residuals_fused)}")
    print(f"  PyTorch CG converged: {converged_ref}, iterations: {len(residuals_ref)}")
    print(f"  Final residual (fused): {residuals_fused[-1]:.2e}")
    print(f"  Final residual (PyTorch): {residuals_ref[-1]:.2e}")

    # Check solution accuracy
    try:
        torch.testing.assert_close(x_fused, x_ref, rtol=1e-4, atol=1e-5)
        print(f"  ✓ Solutions match for N={N}")
    except AssertionError as e:
        print(f"  ✗ Solutions don't match for N={N}: {e}")
        

    # Quick performance test
    def benchmark_fused():
        return fused_cg(A_func, b, x0, device="cuda", max_iter=1000, tol=1e-10)

    def benchmark_pytorch():
        return pytorch_cg(A_func, b, x0, max_iter=1000, tol=1e-10)

    try:
        fused_result = benchmark_kernel(
            benchmark_fused, [], N*N, warmup=5, iters=30
        )
        pytorch_result = benchmark_kernel(
            benchmark_pytorch, [], N*N, warmup=5, iters=30
        )

        print(f"  Fused CG latency: {fused_result['median_time_ms']:.4f} ms")
        print(f"  PyTorch CG latency: {pytorch_result['median_time_ms']:.4f} ms")
        print(f"  Performance ratio (PyTorch/Fused): {pytorch_result['median_time_ms'] / fused_result['median_time_ms']:.2f}x")
    except Exception as e:
        print(f"  Performance test failed: {e}")

    return True


if __name__ == "__main__":
    test_fused_cg()
