import numpy as np
import time
import tilelang
import tilelang.language as T
import torch
from .kernel_utils import benchmark_kernel
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
    raise TypeError(f"Unsupported torch dtype for axpy: {dt}")


@tilelang.jit(target="cuda")
def make_axpy_kernel(
    N,
    grid_size,
    dtype="float64",
    block_size=256,
):
    """
    AXPY kernel: y = a*x + y
    Parallel implementation using shared memory for efficient loading.
    """

    @T.prim_func
    def main(
        a: T.Tensor((1,), dtype),      # type: ignore  # scalar multiplier
        x: T.Tensor((N,), dtype),      # type: ignore
        y: T.Tensor((N,), dtype),      # type: ignore
    ):
        with T.Kernel(grid_size, threads=block_size) as bx:
            tx = T.get_thread_binding(0)

            # Each thread handles one element
            idx = bx * block_size + tx

            # Compute axpy: y[idx] = a[0]*x[idx] + y[idx]
            if idx < N:
                y[idx] = a[0] * x[idx] + y[idx]

    return main


def axpy(
    a,
    x,
    y,
    device="cuda",
    *,
    block_size=256,
):
    """
    Compute AXPY: y = a*x + y

    Args:
        a: scalar (torch.Tensor or scalar)
        x, y: torch.Tensor vectors of same shape and dtype
        device: target device ("cuda" or "cpu")

    Returns:
        Modified y tensor (in-place operation)
    """
    # Convert inputs to tensors if needed
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=x.dtype, device=x.device)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=y.dtype, device=y.device)

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be torch.Tensor")
    if x.shape != y.shape:
        raise ValueError(f"Vectors must have the same shape: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")
    if x.ndim != 1:
        raise ValueError(f"Vectors must be 1D; got x.ndim={x.ndim}")
    if x.dtype != y.dtype:
        raise ValueError(f"Vectors must have the same dtype: x.dtype={x.dtype}, y.dtype={y.dtype}")
    if a.numel() != 1:
        raise ValueError(f"Scalar a must have exactly 1 element; got {a.numel()}")

    N = int(x.numel())
    dtype = _torch_dtype_to_tilelang_dtype(x.dtype)

    # Convert to torch tensors
    dev = torch.device(device)
    a_t = a.to(device=dev, dtype=x.dtype).view(1)  # Ensure shape (1,)
    x_t = x.to(device=dev)
    y_t = y.to(device=dev)

    if not x_t.is_contiguous():
        x_t = x_t.contiguous()
    if not y_t.is_contiguous():
        y_t = y_t.contiguous()

    # Build and run kernel
    grid_size = (N + block_size - 1) // block_size
    axpy_kernel = make_axpy_kernel(
        N=N,
        grid_size=grid_size,
        dtype=dtype,
        block_size=block_size,
    )
    axpy_kernel(a_t, x_t, y_t)

    return y_t


def test_axpy():
    """Test AXPY kernel against PyTorch reference."""
    print("Testing AXPY kernel...")

    if not torch.cuda.is_available():
        print("CUDA is not available; skipping test_axpy().")
        return

    torch.manual_seed(42)
    dev = torch.device("cuda")
    N = 1024
    BLOCK_SIZE = 256
    grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    axpy_kernel = make_axpy_kernel(
        N=N,
        grid_size=grid_size,
        dtype="float64",
        block_size=BLOCK_SIZE,
    )

    # Test with float64
    a_scalar = 2.5
    a = torch.tensor([a_scalar], device=dev, dtype=torch.float64)
    x = torch.randn((N,), device=dev, dtype=torch.float64)
    y = torch.randn((N,), device=dev, dtype=torch.float64)

    # Reference: PyTorch axpy
    y_ref = y.clone()
    y_ref.add_(a_scalar * x)  # y = a*x + y

    # TileLang axpy (in-place)
    y_tilelang = y.clone()
    axpy_kernel(a, x, y_tilelang)

    print(f"Reference result sample [0:5]: {y_ref[:3].tolist()}")
    print(f"TileLang result sample [0:5]: {y_tilelang[:3].tolist()}")

    torch.testing.assert_close(y_ref, y_tilelang, rtol=1e-10, atol=1e-11)
    print("test_axpy(): PASSED")

    # Test with float32
    a_f32 = torch.tensor([a_scalar], device=dev, dtype=torch.float32)
    x_f32 = x.to(dtype=torch.float32)
    y_f32 = y.to(dtype=torch.float32)

    y_ref_f32 = y_f32.clone()
    y_ref_f32.add_(a_scalar * x_f32)

    y_tilelang_f32 = y_f32.clone()
    axpy_kernel_f32 = make_axpy_kernel(
        N=N,
        grid_size=grid_size,
        dtype="float32",
        block_size=BLOCK_SIZE,
    )
    axpy_kernel_f32(a_f32, x_f32, y_tilelang_f32)

    print(f"\nFloat32 Reference result sample [0:5]: {y_ref_f32[:5].tolist()}")
    print(f"Float32 TileLang result sample [0:5]: {y_tilelang_f32[:5].tolist()}")

    torch.testing.assert_close(y_ref_f32, y_tilelang_f32, rtol=1e-4, atol=1e-5)
    print("test_axpy() float32: PASSED")

    # Benchmark TileLang version using benchmark_kernel
    def tilelang_axpy_bench():
        axpy_kernel(a, x, y_tilelang)

    tilelang_bench_result = benchmark_kernel(
        tilelang_axpy_bench,
        [],  # no additional args since tensors are already created
        N,   # nnz: N operations (multiply-add per element)
        warmup=30,
        iters=100,
    )

    print(f"TileLang AXPY latency: {tilelang_bench_result['median_time_ms']:.4f} ms")

    # Compare with PyTorch reference implementation
    def pytorch_axpy():
        y_tilelang.copy_(y_tilelang + a_scalar * x)

    pytorch_bench_result = benchmark_kernel(
        pytorch_axpy,
        [],
        N,   # nnz: N operations
        warmup=30,
        iters=100,
    )

    print(f"PyTorch AXPY latency: {pytorch_bench_result['median_time_ms']:.4f} ms")
    print(f"Performance ratio (TileLang/PyTorch): {pytorch_bench_result['median_time_ms'] / tilelang_bench_result['median_time_ms']:.2f}x")


if __name__ == "__main__":
    test_axpy()
