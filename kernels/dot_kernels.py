from optparse import make_option

import numpy as np
import time
import tilelang
import tilelang.language as T
import torch
from kernel_utils import benchmark_kernel
from typing import Optional


@tilelang.jit(target="cuda")
def make_vector_dot_kernel_single_thread(N, dtype="float64", accum_dtype="float64"):
    """
    Single-threaded vector dot product kernel for correctness.
    Uses one thread to compute the entire dot product.
    """

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),  # type: ignore
        y: T.Tensor((N,), dtype),  # type: ignore
        result: T.Tensor((1,), accum_dtype),  # type: ignore
    ):
        with T.Kernel(1, 1) as bx:
            tid = T.get_thread_binding(0)

            # Single thread computes the entire dot product
            acc = T.alloc_local((1,), accum_dtype)
            acc[0] = T.float64(0)

            for i in T.serial(N):
                x_val = x[i].astype(accum_dtype)
                y_val = y[i].astype(accum_dtype)
                acc[0] += x_val * y_val

            result[0] = acc[0]

    return main


def ceil_div(a, b):
    return (a + b - 1) // b


@tilelang.jit(target="cuda")
def make_vector_dot_kernel_parallel(
    N,
    block_N,
    dtype="float64",
    accum_dtype="float64",
    THREADS_PER_BLOCK=256,   # 建议先用 256，且必须是 2 的幂
):
    """
    Correct parallel dot (optimized & safe):
      - 每线程累加 block_N 个元素（stride = THREADS_PER_BLOCK）
      - block 内归约：shared reduction 只同步到 stride=32
      - 最后 32 个线程（warp0）用 warp-synchronous shared 继续归约（不再 sync）
      - 每个 block 只做一次 atomic_add

    说明：
      - 之前基于 `T.tvm_warp_shuffle` 的“动态 src_lane”归约在部分 TileLang 版本上可能不稳定，
        会导致 kernel 卡住；这里改为更保守的写法，仍然能显著减少 `T.sync_threads()` 次数。
    """

    # 每个 block 覆盖的元素数（每线程 block_N 个）
    ELEMS_PER_BLOCK = THREADS_PER_BLOCK * block_N

    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),          # type: ignore
        y: T.Tensor((N,), dtype),          # type: ignore
        result: T.Tensor((1,), accum_dtype),  # type: ignore
    ):
        # shared: 每线程一个槽，做 block 内归约（稳定版）
        smem = T.alloc_shared((THREADS_PER_BLOCK,), accum_dtype)

        # gridDim.x = ceildiv(N, ELEMS_PER_BLOCK), blockDim.x = THREADS_PER_BLOCK
        with T.Kernel(T.ceildiv(N, ELEMS_PER_BLOCK), threads=THREADS_PER_BLOCK) as (bx,):
            tx = T.get_thread_binding(0)

            # thread-local accumulator
            acc = T.alloc_local((1,), accum_dtype)
            T.clear(acc)

            # 当前线程起始全局 index
            base = bx * ELEMS_PER_BLOCK + tx

            # 每线程处理 block_N 个元素，stride = THREADS_PER_BLOCK
            # 注意：block_N 取大时完全 unroll 会显著增加编译开销；小 block_N unroll 通常更快
            # 这里必须用“Python if + 两段循环”，不能用三元表达式去选择 T.unroll/T.serial（TVMScript 会误解析）
            if block_N <= 16:
                for k in T.unroll(block_N):
                    idx = base + k * THREADS_PER_BLOCK
                    if idx < N:
                        # 避免 dtype==accum_dtype 时的无意义 cast
                        if dtype == accum_dtype:
                            acc[0] += x[idx] * y[idx]
                        else:
                            acc[0] += x[idx].astype(accum_dtype) * y[idx].astype(accum_dtype)
            else:
                for k in T.serial(block_N):
                    idx = base + k * THREADS_PER_BLOCK
                    if idx < N:
                        if dtype == accum_dtype:
                            acc[0] += x[idx] * y[idx]
                        else:
                            acc[0] += x[idx].astype(accum_dtype) * y[idx].astype(accum_dtype)

            # 写入 shared，做归约
            smem[tx] = acc[0]
            if THREADS_PER_BLOCK > 32:
                T.sync_threads()

            # block 归约：同步到 stride=32 即可
            if THREADS_PER_BLOCK >= 512:
                if tx < 256:
                    smem[tx] += smem[tx + 256]
                T.sync_threads()
            if THREADS_PER_BLOCK >= 256:
                if tx < 128:
                    smem[tx] += smem[tx + 128]
                T.sync_threads()
            if THREADS_PER_BLOCK >= 128:
                if tx < 64:
                    smem[tx] += smem[tx + 64]
                T.sync_threads()

            # 64->32 后进入 warp-synchronous 区域（tx<32），不再 sync
            if THREADS_PER_BLOCK >= 64:
                if tx < 32:
                    smem[tx] += smem[tx + 32]

            if tx < 16:
                smem[tx] += smem[tx + 16]
            if tx < 8:
                smem[tx] += smem[tx + 8]
            if tx < 4:
                smem[tx] += smem[tx + 4]
            if tx < 2:
                smem[tx] += smem[tx + 2]
            if tx < 1:
                smem[tx] += smem[tx + 1]

            if tx == 0:
                T.atomic_add(result[0], smem[0])

    return main


def vector_dot(
    x,
    y,
    device="cuda",
    kernel_type="parallel",
    *,
    block_N: Optional[int] = None,
    threads_per_block: int = 256,
):
    """
    Compute vector dot product using TileLang kernel.

    Args:
        x: numpy array of shape (N,)
        y: numpy array of shape (N,)
        device: target device ("cuda" or "cpu")
        kernel_type: "single" for single-threaded, "parallel" for parallel warp-based

    Returns:
        dot product as numpy scalar
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"Vector shapes must match: x.shape={x.shape}, y.shape={y.shape}")

    N = x.shape[0]
    dtype = x.dtype.name
    accum_dtype = "float64"

    # Convert to torch tensors
    dev = torch.device(device)
    x_t = torch.from_numpy(x.astype(dtype)).to(dev)
    y_t = torch.from_numpy(y.astype(dtype)).to(dev)
    result_t = torch.zeros((1,), dtype=getattr(torch, accum_dtype), device=dev)

    # Build and run kernel
    result_t.zero_()
    if kernel_type == "single":
        dot_kernel = make_vector_dot_kernel_single_thread(N, dtype, accum_dtype)
    elif kernel_type == "parallel":
        # 经验值（基于你这边 A100 环境的实测）：block_N=8、tpb=256 往往最好
        # 如果你在更大 N 上测到其它组合更优，可以显式传参覆盖。
        if block_N is None:
            block_N = 8
        dot_kernel = make_vector_dot_kernel_parallel(
            N,
            int(block_N),
            dtype,
            accum_dtype,
            THREADS_PER_BLOCK=int(threads_per_block),
        )
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}. Use 'single' or 'parallel'")
    dot_kernel(x_t, y_t, result_t)

    return result_t.cpu().numpy()[0]


def test_vector_dot():
    """Test vector dot product kernel."""
    print("Testing vector dot product kernel...")

    # Test with simple case
    x_simple = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y_simple = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    ref_simple = np.dot(x_simple, y_simple)
    print(f"Simple case - Reference result: {ref_simple}")

    for kernel_type in ["single", "parallel"]:
        try:
            tl_result = vector_dot(
                x_simple, y_simple, device="cuda", kernel_type=kernel_type
            )
            print(f"Simple case - TileLang ({kernel_type}) result: {tl_result}")
            print(f"Simple case - Match: {np.allclose(ref_simple, tl_result, rtol=1e-10, atol=1e-12)}")
        except Exception as e:
            print(f"Simple case ({kernel_type}) failed: {e}")

    # Test with random vectors
    np.random.seed(42)
    N = 1000
    x = np.random.randn(N).astype(np.float64)
    y = np.random.randn(N).astype(np.float64)
    ref_result = np.dot(x, y)
    print(f"\nRandom case - Reference result: {ref_result}")

    for kernel_type in ["single", "parallel"]:
        try:
            tl_result = vector_dot(x, y, device="cuda", kernel_type=kernel_type)
            print(f"Random case - TileLang ({kernel_type}) result: {tl_result}")
            print(f"Random case - Match: {np.allclose(ref_result, tl_result, rtol=1e-5, atol=1e-6)}")
            print(f"Random case - Absolute error: {abs(ref_result - tl_result)}")
        except Exception as e:
            print(f"Random case ({kernel_type}) failed: {e}")
            import traceback
            traceback.print_exc()


def benchmark_cpu_kernel(
    fn,
    flops: float,
    *,
    warmup: int = 10,
    iters: int = 100,
):
    """
    CPU benchmark helper (for NumPy, etc.).
    Uses time.perf_counter; reports median/avg time and GFLOP/s@median.
    """
    # Warmup
    last = None
    for _ in range(warmup):
        last = fn()

    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        last = fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)

    times_ms = np.asarray(times_ms, dtype=np.float64)
    avg_ms = float(times_ms.mean())
    med_ms = float(np.median(times_ms))

    med_s = med_ms * 1e-3
    med_gflops = float(flops / med_s / 1e9) if med_s > 0 else float("inf")

    # 防止极端情况下被解释器/编译器“认为无用”（主要是为了可读性与调试）
    if last is None:
        raise RuntimeError("benchmark_cpu_kernel: function returned None in warmup/iters")

    print(f"[Benchmark-CPU] iters={iters}, warmup={warmup}")
    print(f"  mid   time : {med_ms:.3f} ms (median)")
    print(f"  avg   time : {avg_ms:.3f} ms")
    print(f"  GFLOP/s@mid: {med_gflops:.4f} GFLOP/s")

    return {
        "median_time_ms": med_ms,
        "average_time_ms": avg_ms,
        "median_gflops": med_gflops,
    }


def bench_vector_dot():
    """Benchmark vector dot product kernel."""
    print("Benchmarking vector dot product...")

    N = 10000
    x = np.random.randn(N).astype(np.float64)
    y = np.random.randn(N).astype(np.float64)

    # NumPy reference (CPU)
    # 注意：这是 CPU 侧时间，和下面 CUDA events 的 GPU 时间不可直接横向对比（设备不同）。
    def numpy_dot():
        return np.dot(x, y)

    # Convert to torch for GPU computation
    dev = torch.device("cuda")
    x_t = torch.from_numpy(x).to(dev)
    y_t = torch.from_numpy(y).to(dev)

    # 注意：benchmark_kernel 用 CUDA events 计时；因此这里的被测函数必须避免 .cpu().item() 之类的同步拷贝

    # PyTorch reference (GPU, no D2H)
    def torch_dot():
        _ = torch.dot(x_t, y_t)

    # TileLang kernels
    result_t = torch.zeros((1,), dtype=torch.float64, device=dev)

    # Single-threaded kernel
    dot_kernel_single = make_vector_dot_kernel_single_thread(N)

    def tilelang_dot_single():
        result_t.zero_()
        dot_kernel_single(x_t, y_t, result_t)

    # Parallel kernel: sweep a few parameter combos
    def run_tilelang_parallel(block_n: int, tpb: int):
        dot_kernel_parallel = make_vector_dot_kernel_parallel(
            N, block_n, "float64", "float64", THREADS_PER_BLOCK=tpb
        )

        def _fn():
            result_t.zero_()
            dot_kernel_parallel(x_t, y_t, result_t)

        return _fn

    # Benchmark
    print("NumPy dot product (CPU):")
    benchmark_cpu_kernel(numpy_dot, flops=2.0 * N, warmup=50, iters=500)

    print("PyTorch dot product:")
    benchmark_kernel(torch_dot, (), N, warmup=50, iters=500)

    print("TileLang dot product (single-threaded):")
    benchmark_kernel(tilelang_dot_single, (), N, warmup=50, iters=500)

    print("TileLang dot product (parallel) param sweep:")
    candidates = [
        (8, 256),
        (16, 256),
        (32, 256),
        (64, 256),
        (32, 128),
        (32, 512),
    ]
    for block_n, tpb in candidates:
        print(f"  - block_N={block_n}, THREADS_PER_BLOCK={tpb}")
        fn = run_tilelang_parallel(block_n, tpb)
        benchmark_kernel(fn, (), N, warmup=50, iters=500)


if __name__ == "__main__":
    test_vector_dot()
    bench_vector_dot()