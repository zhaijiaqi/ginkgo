import numpy as np
import torch


def benchmark_kernel(
    spmv_kernel,
    kernel_args,
    nnz,
    warmup: int = 10,
    iters: int = 100,
):
    for _ in range(warmup):
        spmv_kernel(*kernel_args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record(stream=torch.cuda.current_stream())
        spmv_kernel(*kernel_args)
        end_events[i].record(stream=torch.cuda.current_stream())
    # 只在整轮结束后同步一次，避免把同步开销算进每次迭代
    torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iters)]

    times_ms = np.array(times_ms, dtype=np.float32)
    avg_ms = float(times_ms.mean())
    med_ms = float(np.median(times_ms))

    flops = 2.0 * nnz
    med_s = med_ms * 1e-3
    med_gflops = flops / med_s / 1e9

    print(f"[Benchmark] nnz={nnz}, iters={iters}, warmup={warmup}")
    print(f"  mid   time : {med_ms:.3f} ms (median)")
    print(f"  avg   time : {avg_ms:.3f} ms")
    print(f"  GFLOP/s@mid: {med_gflops:.4f} GFLOP/s")

    return {
        "median_time_ms": med_ms,
        "average_time_ms": avg_ms,
        "median_gflops": med_gflops,
    }
