#!/usr/bin/env python3
"""
Performance benchmark comparing different SpMV kernel implementations:

1. spmv_bcsc_mixed_ref_prequant - BCSC reference (Python/torch)
2. bcsc_spmv_mixed_prequant - BCSC TileLang kernel (warp_reduce or TensorCore)
3. bsr_spmv_mixed_prequant - BSR TileLang kernel (pre-quantized)
4. bsr_spmv_mixed - BSR TileLang kernel (runtime quantization)

This script helps identify performance differences between implementations.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import (  # noqa: E402
    build_bcsc_from_coo,
    quantize_bcsc_tiles,
    spmv_bcsc_mixed_ref_prequant,
)
from kernels.bcsc_spmv_kernels import bcsc_spmv_mixed_prequant  # noqa: E402
from kernels.spmv_kernels import bsr_spmv_mixed, bsr_spmv_mixed_prequant  # noqa: E402


def random_coo(M: int, N: int, nnz: int, seed: int):
    """Generate random COO matrix."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, M, size=nnz, dtype=np.int64)
    cols = rng.integers(0, N, size=nnz, dtype=np.int64)
    vals = rng.standard_normal(nnz).astype(np.float64)
    return rows, cols, vals, (M, N)


def build_bsr_from_bcsc(bcsc, device="cuda"):
    """Convert BCSC to BSR format for comparison."""
    # BSR uses row-major block organization: (nnzb, R, C)
    # BCSC uses column-major: tiles stored by column, with rowind
    R, C = bcsc.R, bcsc.C
    n_br, n_bc = bcsc.n_br, bcsc.n_bc
    nnzb = bcsc.nnzb

    # Build BSR indptr and indices from BCSC colptr and rowind
    # For each block-row, find which block-columns have non-zero tiles
    indptr = torch.zeros((n_br + 1,), dtype=torch.int32, device=device)
    indices_list = []
    tile_idx_map = []  # Maps BSR tile index to BCSC tile index k

    # Count tiles per block-row and build mapping
    # Also build mapping from BCSC tile index k to its block-column bc
    k_to_bc = {}  # Maps BCSC tile index k to block-column bc
    for bc in range(n_bc):
        start = bcsc.colptr[bc].item()
        end = bcsc.colptr[bc + 1].item()
        for k in range(start, end):
            k_to_bc[k] = bc
    
    for br in range(n_br):
        tile_count = 0
        for bc in range(n_bc):
            start = bcsc.colptr[bc].item()
            end = bcsc.colptr[bc + 1].item()
            for k in range(start, end):
                if bcsc.rowind[k].item() == br:
                    indices_list.append(bc)
                    tile_idx_map.append(k)
                    tile_count += 1
        indptr[br + 1] = indptr[br] + tile_count

    indices = torch.tensor(indices_list, dtype=torch.int32, device=device)

    # Reorganize data from BCSC order to BSR order (nnzb, R, C)
    data_fp64 = torch.zeros((nnzb, R, C), dtype=torch.float64, device=device)
    data_fp32_q = torch.zeros((nnzb, R, C), dtype=torch.float32, device=device)
    data_bf16_q = torch.zeros((nnzb, R, C), dtype=torch.bfloat16, device=device)
    a_scale_fp32 = torch.zeros((nnzb,), dtype=torch.float64, device=device)
    a_scale_bf16 = torch.zeros((nnzb,), dtype=torch.float64, device=device)

    for bsr_idx, bcsc_k in enumerate(tile_idx_map):
        data_fp64[bsr_idx] = bcsc.A_fp64[bcsc_k]
        data_fp32_q[bsr_idx] = bcsc.A_fp32_q[bcsc_k]
        data_bf16_q[bsr_idx] = bcsc.A_bf16_q[bcsc_k]
        # Scale is now per column, not per tile or block-column
        # For BSR format, we need per-tile scale, so we use the first column's scale as approximation
        # This is not perfect but maintains compatibility with BSR kernel interface
        bc = k_to_bc[bcsc_k]
        col_base = bc * bcsc.C
        a_scale_fp32[bsr_idx] = bcsc.a_scale_fp32[col_base] if col_base < bcsc.N else 1.0
        a_scale_bf16[bsr_idx] = bcsc.a_scale_bf16[col_base] if col_base < bcsc.N else 1.0

    return {
        "data_fp64": data_fp64,
        "data_fp32_q": data_fp32_q,
        "data_bf16_q": data_bf16_q,
        "a_scale_fp32": a_scale_fp32,
        "a_scale_bf16": a_scale_bf16,
        "indices": indices,
        "indptr": indptr,
    }


def benchmark_kernel(kernel_func, *args, n_warmup=10, n_iter=100, **kwargs):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(n_warmup):
        _ = kernel_func(*args, **kwargs)

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time the kernel
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = kernel_func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_iter * 1000
    return avg_time_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark different SpMV kernel implementations")
    parser.add_argument("--matrix-size", type=int, default=2048, help="Matrix size (M=N)")
    parser.add_argument("--tilesize", type=int, default=64, help="Tile size")
    parser.add_argument("--density", type=float, default=0.05, help="Matrix density")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--action", type=int, default=1, choices=[0, 1, 2], help="Action (0=fp64, 1=fp32, 2=bf16)")
    parser.add_argument("--use-tensorcore", action="store_true", help="Use TensorCore for BCSC (bf16 only)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] CUDA unavailable. This benchmark requires CUDA.")
        sys.exit(1)

    print(f"[INFO] Benchmarking SpMV kernels")
    print(f"  Matrix size: {args.matrix_size}x{args.matrix_size}")
    print(f"  Tile size: {args.tilesize}")
    print(f"  Density: {args.density}")
    print(f"  Action: {args.action} ({'fp64' if args.action == 0 else 'fp32' if args.action == 1 else 'bf16'})")
    print(f"  Warmup: {args.n_warmup}, Iterations: {args.n_iter}")
    print()

    # Build BCSC
    M = N = args.matrix_size
    nnz = int(M * N * args.density)
    row, col, data, shape = random_coo(M, N, nnz, args.seed)

    print("[INFO] Building BCSC...")
    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=args.tilesize, device="cpu")
    bcsc = quantize_bcsc_tiles(bcsc)
    bcsc_cuda = bcsc.to("cuda")

    # Generate random vector
    rng = np.random.default_rng(args.seed + 1)
    x = torch.as_tensor(rng.standard_normal((N,), dtype=np.float64), dtype=torch.float64, device="cuda")

    # Generate actions
    n_bc = bcsc.n_bc
    actions = torch.full((n_bc,), args.action, dtype=torch.int32, device="cuda")

    print("[INFO] Building BSR (for comparison)...")
    bsr_data = build_bsr_from_bcsc(bcsc_cuda, device="cuda")

    results = {}

    # 1. BCSC Reference (Python/torch)
    print("\n[1/4] Benchmarking BCSC Reference (Python/torch)...")
    try:
        x_cpu = x.cpu()
        actions_cpu = actions.cpu()
        time_ms = benchmark_kernel(
            spmv_bcsc_mixed_ref_prequant,
            bcsc,
            actions_cpu,
            x_cpu,
            trim_to_M=True,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        results["BCSC Reference"] = time_ms
        print(f"  Time: {time_ms:.3f} ms/iter")
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["BCSC Reference"] = None

    # 2. BCSC TileLang Kernel (warp_reduce or TensorCore)
    print("\n[2/4] Benchmarking BCSC TileLang Kernel...")
    try:
        time_ms = benchmark_kernel(
            bcsc_spmv_mixed_prequant,
            bcsc_cuda,
            actions,
            x,
            device="cuda",
            use_tensorcore_bf16=args.use_tensorcore and args.action == 2,
            return_torch=True,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        kernel_type = "TensorCore" if (args.use_tensorcore and args.action == 2) else "warp_reduce"
        results[f"BCSC TileLang ({kernel_type})"] = time_ms
        print(f"  Time: {time_ms:.3f} ms/iter ({kernel_type})")
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["BCSC TileLang"] = None

    # 3. BSR Pre-quantized Kernel
    print("\n[3/4] Benchmarking BSR Pre-quantized Kernel...")
    try:
        time_ms = benchmark_kernel(
            bsr_spmv_mixed_prequant,
            bsr_data["data_fp64"],
            bsr_data["data_fp32_q"],
            bsr_data["a_scale_fp32"],
            bsr_data["data_bf16_q"],
            bsr_data["a_scale_bf16"],
            actions,
            bsr_data["indices"],
            bsr_data["indptr"],
            x,
            args.tilesize,
            args.tilesize,
            device="cuda",
            return_torch=True,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        results["BSR Pre-quantized"] = time_ms
        print(f"  Time: {time_ms:.3f} ms/iter")
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["BSR Pre-quantized"] = None

    # 4. BSR Runtime Quantization Kernel
    print("\n[4/4] Benchmarking BSR Runtime Quantization Kernel...")
    try:
        # For runtime quantization, we need the original fp64 data
        # Note: bsr_spmv_mixed expects actions per block-column, same as BCSC
        time_ms = benchmark_kernel(
            bsr_spmv_mixed,
            bsr_data["data_fp64"],
            actions,
            bsr_data["indices"],
            bsr_data["indptr"],
            x,
            args.tilesize,
            args.tilesize,
            device="cuda",
            return_torch=True,
            n_warmup=args.n_warmup,
            n_iter=args.n_iter,
        )
        results["BSR Runtime Quant"] = time_ms
        print(f"  Time: {time_ms:.3f} ms/iter")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        results["BSR Runtime Quant"] = None

    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    baseline = None
    for name, time_ms in results.items():
        if time_ms is not None:
            if baseline is None:
                baseline = time_ms
                print(f"{name:30s}: {time_ms:8.3f} ms/iter (baseline)")
            else:
                speedup = baseline / time_ms
                print(f"{name:30s}: {time_ms:8.3f} ms/iter ({speedup:5.2f}x {'faster' if speedup > 1 else 'slower'})")
        else:
            print(f"{name:30s}: FAILED")
    print("=" * 60)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

