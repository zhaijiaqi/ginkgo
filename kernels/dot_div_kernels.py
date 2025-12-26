from optparse import make_option

import numpy as np
import time
import tilelang
import tilelang.language as T
import torch
from kernel_utils import benchmark_kernel
from typing import Optional


def _torch_dtype_to_tilelang_dtype(dt: torch.dtype) -> str:
    if dt == torch.float16:
        return "float16"
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.float32:
        return "float32"
    if dt == torch.float64:
        return "float64"
    raise TypeError(f"Unsupported torch dtype for vector_dot: {dt}")



@tilelang.jit(target="cuda")
def make_fused_dot_div_kernel(
    N,
    grid_size,
    dtype="float64",
    accum_dtype="float64",
    block_size=256,
):
    """
    Fused kernel that computes (a·b) / (c·d) where a,b,c,d are vectors.
    Returns a scalar result.
    """

    @T.prim_func
    def main(
        a: T.Tensor((N,), dtype),      # type: ignore
        b: T.Tensor((N,), dtype),      # type: ignore
        c: T.Tensor((N,), dtype),      # type: ignore
        d: T.Tensor((N,), dtype),      # type: ignore
        result: T.Tensor((2,), accum_dtype),  # type: ignore  # [dot_ab, dot_cd]
    ):
        # Shared memory for block reduction (for both dot products)
        smem_ab = T.alloc_shared((block_size,), accum_dtype)
        smem_cd = T.alloc_shared((block_size,), accum_dtype)

        with T.Kernel(grid_size, threads=block_size) as bx:
            tx = T.get_thread_binding(0)

            # Each thread handles one element
            idx = bx * block_size + tx

            # Thread-local accumulators for both dot products
            acc_ab = T.alloc_local((1,), accum_dtype)
            acc_cd = T.alloc_local((1,), accum_dtype)
            T.clear(acc_ab)
            T.clear(acc_cd)

            # Load and compute a[idx]*b[idx] and c[idx]*d[idx] if within bounds
            if idx < N:
                if dtype == accum_dtype:
                    acc_ab[0] = a[idx] * b[idx]
                    acc_cd[0] = c[idx] * d[idx]
                else:
                    acc_ab[0] = a[idx].astype(accum_dtype) * b[idx].astype(accum_dtype)
                    acc_cd[0] = c[idx].astype(accum_dtype) * d[idx].astype(accum_dtype)

            # Store to shared memory
            smem_ab[tx] = acc_ab[0]
            smem_cd[tx] = acc_cd[0]

            # Block reduction: tree-based sum for both dot products
            if block_size > 32:
                T.sync_threads()

            # Reduce to 32 threads
            if block_size >= 512:
                if tx < 256:
                    smem_ab[tx] += smem_ab[tx + 256]
                    smem_cd[tx] += smem_cd[tx + 256]
                T.sync_threads()
            if block_size >= 256:
                if tx < 128:
                    smem_ab[tx] += smem_ab[tx + 128]
                    smem_cd[tx] += smem_cd[tx + 128]
                T.sync_threads()
            if block_size >= 128:
                if tx < 64:
                    smem_ab[tx] += smem_ab[tx + 64]
                    smem_cd[tx] += smem_cd[tx + 64]
                T.sync_threads()

            # Warp-synchronous reduction (no sync needed)
            if block_size >= 64:
                if tx < 32:
                    smem_ab[tx] += smem_ab[tx + 32]
                    smem_cd[tx] += smem_cd[tx + 32]
            if tx < 16:
                smem_ab[tx] += smem_ab[tx + 16]
                smem_cd[tx] += smem_cd[tx + 16]
            if tx < 8:
                smem_ab[tx] += smem_ab[tx + 8]
                smem_cd[tx] += smem_cd[tx + 8]
            if tx < 4:
                smem_ab[tx] += smem_ab[tx + 4]
                smem_cd[tx] += smem_cd[tx + 4]
            if tx < 2:
                smem_ab[tx] += smem_ab[tx + 2]
                smem_cd[tx] += smem_cd[tx + 2]
            if tx < 1:
                smem_ab[tx] += smem_ab[tx + 1]
                smem_cd[tx] += smem_cd[tx + 1]

            # Atomic add both dot products to global accumulators
            if tx == 0:
                T.atomic_add(result[0], smem_ab[0])  # dot_ab accumulator
                T.atomic_add(result[1], smem_cd[0])  # dot_cd accumulator

    return main


@tilelang.jit(target="cuda")
def make_fused_dot_div_kernel_concatenated(
    N,
    grid_size,
    dtype="float64",
    accum_dtype="float64",
    block_size=256,
):
    """
    Optimized fused kernel that computes (a·b) / (c·d) using concatenated inputs.
    A contains [a1, a2, ..., aN, c1, c2, ..., cN]
    B contains [b1, b2, ..., bN, d1, d2, ..., dN]
    This reduces memory accesses from 4 to 2 per thread.
    """

    @T.prim_func
    def main(
        A: T.Tensor((2*N,), dtype),      # type: ignore  # [a, c] concatenated
        B: T.Tensor((2*N,), dtype),      # type: ignore  # [b, d] concatenated
        result: T.Tensor((2,), accum_dtype),  # type: ignore  # [dot_ab, dot_cd]
    ):
        # Shared memory for block reduction (for both dot products)
        smem_ab = T.alloc_shared((block_size,), accum_dtype)
        smem_cd = T.alloc_shared((block_size,), accum_dtype)

        with T.Kernel(grid_size, threads=block_size) as bx:
            tx = T.get_thread_binding(0)

            # Each thread handles two elements: one from each dot product
            base_idx = bx * block_size + tx

            # Thread-local accumulators for both dot products
            acc_ab = T.alloc_local((1,), accum_dtype)
            acc_cd = T.alloc_local((1,), accum_dtype)
            T.clear(acc_ab)
            T.clear(acc_cd)

            # Process first element (a*b part)
            idx1 = base_idx
            if idx1 < N:
                if dtype == accum_dtype:
                    acc_ab[0] = A[idx1] * B[idx1]  # a[idx1] * b[idx1]
                else:
                    acc_ab[0] = A[idx1].astype(accum_dtype) * B[idx1].astype(accum_dtype)

            # Process second element (c*d part)
            idx2 = base_idx + N  # offset by N for c*d part
            if idx2 < 2*N:
                if dtype == accum_dtype:
                    acc_cd[0] = A[idx2] * B[idx2]  # c[idx1] * d[idx1]
                else:
                    acc_cd[0] = A[idx2].astype(accum_dtype) * B[idx2].astype(accum_dtype)

            # Store to shared memory
            smem_ab[tx] = acc_ab[0]
            smem_cd[tx] = acc_cd[0]

            # Block reduction: tree-based sum for both dot products
            if block_size > 32:
                T.sync_threads()

            # Reduce to 32 threads
            if block_size >= 512:
                if tx < 256:
                    smem_ab[tx] += smem_ab[tx + 256]
                    smem_cd[tx] += smem_cd[tx + 256]
                T.sync_threads()
            if block_size >= 256:
                if tx < 128:
                    smem_ab[tx] += smem_ab[tx + 128]
                    smem_cd[tx] += smem_cd[tx + 128]
                T.sync_threads()
            if block_size >= 128:
                if tx < 64:
                    smem_ab[tx] += smem_ab[tx + 64]
                    smem_cd[tx] += smem_cd[tx + 64]
                T.sync_threads()

            # Warp-synchronous reduction (no sync needed)
            if block_size >= 64:
                if tx < 32:
                    smem_ab[tx] += smem_ab[tx + 32]
                    smem_cd[tx] += smem_cd[tx + 32]
            if tx < 16:
                smem_ab[tx] += smem_ab[tx + 16]
                smem_cd[tx] += smem_cd[tx + 16]
            if tx < 8:
                smem_ab[tx] += smem_ab[tx + 8]
                smem_cd[tx] += smem_cd[tx + 8]
            if tx < 4:
                smem_ab[tx] += smem_ab[tx + 4]
                smem_cd[tx] += smem_cd[tx + 4]
            if tx < 2:
                smem_ab[tx] += smem_ab[tx + 2]
                smem_cd[tx] += smem_cd[tx + 2]
            if tx < 1:
                smem_ab[tx] += smem_ab[tx + 1]
                smem_cd[tx] += smem_cd[tx + 1]

            # Atomic add both dot products to global accumulators
            if tx == 0:
                T.atomic_add(result[0], smem_ab[0])  # dot_ab accumulator
                T.atomic_add(result[1], smem_cd[0])  # dot_cd accumulator

    return main


def fused_dot_div(
    a, b, c, d,
    device="cuda",
    *,
    block_size=256,
    use_concatenated=False,
):
    """
    Compute (a·b) / (c·d) using TileLang fused kernel.

    Args:
        a, b, c, d: torch.Tensor vectors of same shape and dtype
        device: target device ("cuda" or "cpu")
        block_size: CUDA block size
        use_concatenated: if True, use concatenated input version (potentially better memory access)

    Returns:
        A 0-d torch.Tensor scalar: (a·b) / (c·d)
    """
    if not all(isinstance(x, torch.Tensor) for x in [a, b, c, d]):
        raise TypeError("All inputs must be torch.Tensor")
    if not all(x.shape == a.shape for x in [b, c, d]):
        raise ValueError("All vectors must have the same shape")
    if not all(x.ndim == 1 for x in [a, b, c, d]):
        raise ValueError("All inputs must be 1D tensors")
    if not all(x.dtype == a.dtype for x in [b, c, d]):
        raise ValueError("All vectors must have the same dtype")

    N = int(a.numel())
    dtype = _torch_dtype_to_tilelang_dtype(a.dtype)
    # Use same precision for accumulation as input, or promote float16 to float32
    if a.dtype == torch.float64:
        accum_dtype = "float64"
    elif a.dtype == torch.float32:
        accum_dtype = "float32"
    elif a.dtype == torch.float16 or a.dtype == torch.bfloat16:
        accum_dtype = "float32"  # promote for better precision
    else:
        accum_dtype = "float64"  # fallback

    # Convert to torch tensors
    dev = torch.device(device)
    a_t = a.to(device=dev)
    b_t = b.to(device=dev)
    c_t = c.to(device=dev)
    d_t = d.to(device=dev)

    if not all(x.is_contiguous() for x in [a_t, b_t, c_t, d_t]):
        a_t, b_t, c_t, d_t = a_t.contiguous(), b_t.contiguous(), c_t.contiguous(), d_t.contiguous()

    # Create result array to hold [dot_ab, dot_cd]
    result_t = torch.zeros((2,), dtype=getattr(torch, accum_dtype), device=dev)

    if use_concatenated:
        # Concatenated version: A = [a, c], B = [b, d]
        A_t = torch.cat([a_t, c_t], dim=0)
        B_t = torch.cat([b_t, d_t], dim=0)

        # Build and run concatenated kernel
        grid_size = (N + block_size - 1) // block_size  # Process N elements per dot product
        fused_kernel = make_fused_dot_div_kernel_concatenated(
            N=N,
            grid_size=grid_size,
            dtype=dtype,
            accum_dtype=accum_dtype,
            block_size=block_size,
        )
        fused_kernel(A_t, B_t, result_t)
    else:
        # Original version: separate arrays
        # Build and run kernel
        grid_size = (N + block_size - 1) // block_size
        fused_kernel = make_fused_dot_div_kernel(
            N=N,
            grid_size=grid_size,
            dtype=dtype,
            accum_dtype=accum_dtype,
            block_size=block_size,
        )
        fused_kernel(a_t, b_t, c_t, d_t, result_t)

    # Compute final result: dot_ab / dot_cd
    dot_ab = result_t[0]
    dot_cd = result_t[1]
    # Avoid division by zero
    if abs(dot_cd.item()) > 1e-12:
        final_result = dot_ab / dot_cd
    else:
        final_result = torch.tensor(0.0, dtype=dot_ab.dtype, device=dot_ab.device)

    return final_result


def test_fused_dot_div():
    """Test fused (a·b)/(c·d) kernel against PyTorch reference."""
    print("Testing fused dot-div kernel...")

    if not torch.cuda.is_available():
        print("CUDA is not available; skipping test_fused_dot_div().")
        return

    torch.manual_seed(42)
    dev = torch.device("cuda")
    N = 10245
    BLOCK_SIZE = 256
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_kernel = make_fused_dot_div_kernel(
        N=N,
        grid_size=grid_size,
        dtype="float64",
        accum_dtype="float64",
        block_size=BLOCK_SIZE,
    )

    # Test with float64
    a = torch.randn((N,), device=dev, dtype=torch.float64)
    b = torch.randn((N,), device=dev, dtype=torch.float64)
    c = torch.randn((N,), device=dev, dtype=torch.float64)
    d = torch.randn((N,), device=dev, dtype=torch.float64)

    # Reference: (a·b) / (c·d)
    ref_ab = torch.dot(a, b)
    ref_cd = torch.dot(c, d)
    ref_result = ref_ab / ref_cd if ref_cd != 0 else 0.0

    # TileLang result (original version)
    tl_result = fused_dot_div(a, b, c, d, device="cuda", block_size=BLOCK_SIZE, use_concatenated=False)

    # TileLang result (concatenated version)
    tl_result_concat = fused_dot_div(a, b, c, d, device="cuda", block_size=BLOCK_SIZE, use_concatenated=True)

    print(f"Reference result: {ref_result.item()}")
    print(f"TileLang result (original): {tl_result.item()}")
    print(f"TileLang result (concatenated): {tl_result_concat.item()}")

    torch.testing.assert_close(ref_result, tl_result, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(ref_result, tl_result_concat, rtol=1e-10, atol=1e-11)
    print("test_fused_dot_div(): PASSED")

    # Test with float32
    a_f32 = a.to(dtype=torch.float32)
    b_f32 = b.to(dtype=torch.float32)
    c_f32 = c.to(dtype=torch.float32)
    d_f32 = d.to(dtype=torch.float32)

    ref_ab_f32 = torch.dot(a_f32, b_f32)
    ref_cd_f32 = torch.dot(c_f32, d_f32)
    ref_result_f32 = ref_ab_f32 / ref_cd_f32

    tl_result_f32 = fused_dot_div(a_f32, b_f32, c_f32, d_f32, device="cuda", block_size=BLOCK_SIZE, use_concatenated=False)
    tl_result_f32_concat = fused_dot_div(a_f32, b_f32, c_f32, d_f32, device="cuda", block_size=BLOCK_SIZE, use_concatenated=True)

    print(f"\nFloat32 Reference result: {ref_result_f32.item()}")
    print(f"Float32 TileLang result (original): {tl_result_f32.item()}")
    print(f"Float32 TileLang result (concatenated): {tl_result_f32_concat.item()}")

    torch.testing.assert_close(ref_result_f32, tl_result_f32, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ref_result_f32, tl_result_f32_concat, rtol=1e-4, atol=1e-5)
    print("test_fused_dot_div() float32: PASSED")

    # Benchmark TileLang version (original) using benchmark_kernel
    result_t = torch.zeros((2,), dtype=torch.float64, device=dev)  # Recreate for benchmark
    def tilelang_fused_dot_div_original():
        fused_kernel(a, b, c, d, result_t)

    tilelang_bench_result = benchmark_kernel(
        tilelang_fused_dot_div_original,
        [],  # no additional args since tensors are already created
        2 * N,  # nnz: 2 dot products, each with N operations
        warmup=30,
        iters=100,
    )

    print(f"TileLang fused dot-div latency (original): {tilelang_bench_result['median_time_ms']:.4f} ms")

    # Benchmark TileLang version (concatenated) using benchmark_kernel
    A_concat = torch.cat([a, c], dim=0)
    B_concat = torch.cat([b, d], dim=0)
    result_t_concat = torch.zeros((2,), dtype=torch.float64, device=dev)

    grid_size_concat = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_kernel_concat = make_fused_dot_div_kernel_concatenated(
        N=N,
        grid_size=grid_size_concat,
        dtype="float64",
        accum_dtype="float64",
        block_size=BLOCK_SIZE,
    )

    def tilelang_fused_dot_div_concatenated():
        fused_kernel_concat(A_concat, B_concat, result_t_concat)

    tilelang_concat_bench_result = benchmark_kernel(
        tilelang_fused_dot_div_concatenated,
        [],  # no additional args since tensors are already created
        2 * N,  # nnz: 2 dot products, each with N operations
        warmup=30,
        iters=100,
    )

    print(f"TileLang fused dot-div latency (concatenated): {tilelang_concat_bench_result['median_time_ms']:.4f} ms")
    print(f"Performance improvement: {tilelang_bench_result['median_time_ms'] / tilelang_concat_bench_result['median_time_ms']:.2f}x")

    # Compare with PyTorch reference implementation
    def pytorch_fused_dot_div(a, b, c, d):
        """PyTorch reference implementation of (a·b) / (c·d)"""
        dot_ab = torch.dot(a, b)
        dot_cd = torch.dot(c, d)
        return dot_ab / dot_cd

    # Benchmark PyTorch version using benchmark_kernel
    pytorch_bench_result = benchmark_kernel(
        pytorch_fused_dot_div,
        [a, b, c, d],
        2 * N,  # nnz: 2 dot products, each with N operations
        warmup=30,
        iters=100,
    )

    print(f"PyTorch fused dot-div latency: {pytorch_bench_result['median_time_ms']:.4f} ms")
    print(f"Performance ratio (TileLang/PyTorch): {pytorch_bench_result['median_time_ms'] / tilelang_bench_result['median_time_ms']:.2f}x")


if __name__ == "__main__":
    test_fused_dot_div()
    