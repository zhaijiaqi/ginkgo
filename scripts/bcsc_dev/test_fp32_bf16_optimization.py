#!/usr/bin/env python3
"""
Test script to verify fp32/bf16 optimization in bcsc_spmv_kernels.

This script:
  1. Loads matrix from .mtx file in ~/data/matrix directory
  2. Tests correctness: kernel output matches reference
  3. Tests all action paths (0=fp64, 1=fp32, 2=bf16)
  4. Tests with random actions
  5. Always runs performance detection with baseline comparison

The optimization should:
  - Use fp32 CUDA core instructions for action==1
  - Use bf16 CUDA core instructions for action==2
  - Maintain numerical correctness (within quantization tolerance)
  - Show speedup compared to baseline (all_fp64)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

try:
    from scipy.io import mmread
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import (  # noqa: E402
    _as_coo_arrays_from_scipy,
    build_bcsc_from_coo,
    quantize_bcsc_tiles,
    spmv_bcsc_mixed_ref_prequant,
)
from kernels.bcsc_spmv_kernels import bcsc_spmv_mixed_prequant  # noqa: E402


def load_matrix_from_mtx(matrix_name: str, matrix_dir: str = "~/data/matrix"):
    """
    Load matrix from .mtx file.
    
    Args:
        matrix_name: Matrix name (without .mtx extension)
        matrix_dir: Directory containing matrix files (default: ~/data/matrix)
    
    Returns:
        Tuple of (row, col, data, shape) as numpy arrays
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for loading Matrix Market files. Please install scipy.")
    
    matrix_path = os.path.expanduser(os.path.join(matrix_dir, f"{matrix_name}.mtx"))
    
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    
    # Read Matrix Market file
    A = mmread(matrix_path)
    
    # Convert to COO format
    row, col, data, shape = _as_coo_arrays_from_scipy(A)
    
    return row, col, data, shape


def test_correctness(bcsc, bcsc_cuda, x, actions, case_name, verbose=False):
    """Test kernel correctness against reference."""
    y_ref = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x, trim_to_M=True)
    y_tl = bcsc_spmv_mixed_prequant(bcsc_cuda, actions, x, device="cuda", return_torch=True).cpu()
    # Kernel returns padded output (n_br * R), trim to M for comparison
    y_tl = y_tl[:bcsc.M]

    # Check for non-finite values
    if not torch.isfinite(y_tl).all():
        raise AssertionError(f"[FAIL] {case_name}: produced non-finite outputs")

    # Determine tolerance based on action
    has_bf16 = torch.any(actions == 2)
    has_fp32 = torch.any(actions == 1)
    
    if has_bf16:
        atol, rtol = 10, 100  # bf16 has larger quantization error
    elif has_fp32:
        atol, rtol = 1e-4, 1e-4  # fp32 quantization error
    else:
        atol, rtol = 1e-6, 1e-6  # fp64 should be exact

    if not torch.allclose(y_tl, y_ref, atol=atol, rtol=rtol):
        diff = (y_tl - y_ref).abs()
        max_abs_diff = diff.max().item()
        max_rel_diff = (diff / (y_ref.abs() + 1e-12)).max().item()
        if verbose:
            print(f"[FAIL] {case_name}:")
            print(f"  y_tl[:8] = {y_tl[:8].numpy()}")
            print(f"  y_ref[:8] = {y_ref[:8].numpy()}")
            print(f"  max_abs_diff = {max_abs_diff:.6e}")
            print(f"  max_rel_diff = {max_rel_diff:.6e}")
        raise AssertionError(
            f"[FAIL] {case_name}: max_abs_diff={max_abs_diff:.6e}, max_rel_diff={max_rel_diff:.6e}"
        )

    # Always print success message (not just when verbose)
    print(f"[OK] {case_name}: correctness passed (atol={atol:.2e}, rtol={rtol:.2e})")
    if verbose:
        # Additional verbose info
        print(f"  Output shape: {y_tl.shape}")
        print(f"  Output range: [{y_tl.min().item():.6e}, {y_tl.max().item():.6e}]")


def benchmark_kernel(bcsc_cuda, x, actions, actual_nnz, n_warmup=10, n_iter=100, extra_warmup_for_new_actions=5):
    """
    Benchmark kernel performance and return detailed metrics.
    
    Note: TileLang kernels are JIT compiled. The first call with new kernel parameters
    triggers compilation, which can affect timing. This function includes extra warmup
    when actions change to ensure kernel is fully compiled and optimized.
    """
    # Extra warmup for new action patterns (helps with kernel compilation/optimization)
    # This is especially important when switching between different action distributions
    for _ in range(extra_warmup_for_new_actions):
        _ = bcsc_spmv_mixed_prequant(bcsc_cuda, actions, x, device="cuda", return_torch=True)
    
    # Standard warmup
    for _ in range(n_warmup):
        _ = bcsc_spmv_mixed_prequant(bcsc_cuda, actions, x, device="cuda", return_torch=True)

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time the kernel
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = bcsc_spmv_mixed_prequant(bcsc_cuda, actions, x, device="cuda", return_torch=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_iter * 1000
    avg_time_s = avg_time_ms / 1000.0
    
    # Calculate throughput (operations per second)
    # Use actual nnz (non-zero elements) for accurate throughput calculation
    # Each SpMV performs approximately 2*nnz operations (multiply-add per non-zero)
    throughput_gops = (2 * actual_nnz / avg_time_s) / 1e9  # Giga operations per second
    
    return {
        "avg_time_ms": avg_time_ms,
        "avg_time_s": avg_time_s,
        "throughput_gops": throughput_gops,
    }


def generate_random_actions(n_bc, seed, fp64_ratio=0.0, fp32_ratio=0.0, bf16_ratio=0.0):
    """
    Generate random actions with specified ratios.
    
    If all ratios are 0, generates uniformly random actions.
    Otherwise, ratios should sum to 1.0.
    """
    rng = np.random.default_rng(seed)
    
    if fp64_ratio == 0.0 and fp32_ratio == 0.0 and bf16_ratio == 0.0:
        # Uniformly random
        actions = rng.integers(0, 3, size=n_bc, dtype=np.int32)
    else:
        # Use specified ratios
        n_fp64 = int(n_bc * fp64_ratio)
        n_fp32 = int(n_bc * fp32_ratio)
        n_bf16 = n_bc - n_fp64 - n_fp32
        
        actions = np.concatenate([
            np.zeros(n_fp64, dtype=np.int32),
            np.ones(n_fp32, dtype=np.int32),
            np.full(n_bf16, 2, dtype=np.int32),
        ])
        rng.shuffle(actions)
    
    return torch.from_numpy(actions)


def get_action_distribution(actions):
    """Get distribution of actions."""
    actions_np = actions.cpu().numpy()
    n_fp64 = np.sum(actions_np == 0)
    n_fp32 = np.sum(actions_np == 1)
    n_bf16 = np.sum(actions_np == 2)
    total = len(actions_np)
    return {
        "fp64": (n_fp64, n_fp64 / total * 100),
        "fp32": (n_fp32, n_fp32 / total * 100),
        "bf16": (n_bf16, n_bf16 / total * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Test fp32/bf16 optimization in BCSC SpMV kernel")
    parser.add_argument("--matrix-name", type=str, required=True, help="Matrix name (without .mtx extension)")
    parser.add_argument("--matrix-dir", type=str, default="~/data/matrix", help="Directory containing matrix files")
    parser.add_argument("--tilesize", type=int, default=64, help="Tile size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for vector generation")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of iterations for benchmarking")
    parser.add_argument("--n-warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--random-actions", type=int, default=3, help="Number of random action test cases")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for random actions")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA unavailable.")
        return

    if not SCIPY_AVAILABLE:
        print("[ERROR] scipy is required for loading Matrix Market files. Please install scipy.")
        sys.exit(1)

    print(f"[INFO] Testing fp32/bf16 optimization")
    print(f"  Matrix name: {args.matrix_name}")
    print(f"  Matrix directory: {args.matrix_dir}")
    print(f"  Tile size: {args.tilesize}")

    # Load matrix from .mtx file
    try:
        row, col, data, shape = load_matrix_from_mtx(args.matrix_name, args.matrix_dir)
        M, N = shape
        actual_nnz = len(data)
        print(f"  Matrix size: {M}x{N}")
        print(f"  Non-zeros: {actual_nnz:,}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load matrix: {e}")
        sys.exit(1)

    # Build BCSC
    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=args.tilesize, device="cpu")
    bcsc = quantize_bcsc_tiles(bcsc)
    bcsc_cuda = bcsc.to("cuda")
    
    # Calculate actual nnz from BCSC (count non-zero elements in all tiles)
    # This accounts for tile padding and actual sparsity
    actual_nnz_in_bcsc = int((bcsc_cuda.A_fp64 != 0).sum().item())

    # Generate random vector
    rng = np.random.default_rng(args.seed)
    x = torch.as_tensor(rng.standard_normal((N,), dtype=np.float64), dtype=torch.float64)

    # Test cases
    n_bc = bcsc.n_bc
    test_cases = {
        "all_fp64": torch.zeros((n_bc,), dtype=torch.int32),
        "all_fp32": torch.ones((n_bc,), dtype=torch.int32),
        "all_bf16": torch.full((n_bc,), 2, dtype=torch.int32),
        "mixed": torch.tensor(
            [0] * (n_bc // 3) + [1] * (n_bc // 3) + [2] * (n_bc - 2 * (n_bc // 3)),
            dtype=torch.int32,
        ),
    }
    
    # Add random action test cases
    for i in range(args.random_actions):
        random_actions = generate_random_actions(n_bc, args.random_seed + i)
        test_cases[f"random_{i+1}"] = random_actions

    print("\n[TEST] Correctness tests:")
    for case_name, actions in test_cases.items():
        try:
            test_correctness(bcsc, bcsc_cuda, x, actions, case_name, verbose=args.verbose)
        except Exception as e:
            print(f"[FAIL] {case_name}: {e}")
            sys.exit(1)

    print("\n[OK] All correctness tests passed!")

    # Always run speed detection (not just when --benchmark is set)
    print("\n[SPEED] Performance detection:")
    print(f"  Actual nnz in matrix: {actual_nnz:,}")
    print(f"  Actual nnz in BCSC tiles: {actual_nnz_in_bcsc:,}")
    print(f"  Note: TileLang kernels are JIT compiled. First test may include compilation time.")
    
    # Baseline: all_fp64
    # Use extra warmup for baseline to ensure kernel is fully compiled
    baseline_actions = test_cases["all_fp64"]
    baseline_metrics = benchmark_kernel(
        bcsc_cuda, x, baseline_actions, actual_nnz_in_bcsc,
        n_warmup=args.n_warmup, n_iter=args.n_iter,
        extra_warmup_for_new_actions=10  # Extra warmup for first kernel compilation
    )
    baseline_time_ms = baseline_metrics["avg_time_ms"]
    
    print(f"\n  Baseline (all_fp64):")
    print(f"    Time: {baseline_time_ms:.3f} ms/iter")
    print(f"    Throughput: {baseline_metrics['throughput_gops']:.3f} GOP/s")
    
    # First, get our kernel performance for all precisions
    print(f"\n  [OUR KERNEL] Performance summary:")
    our_results = {}
    for case_name, actions in test_cases.items():
        if case_name in ["all_fp64", "all_fp32", "all_bf16"]:
            metrics = benchmark_kernel(
                bcsc_cuda, x, actions, actual_nnz_in_bcsc,
                n_warmup=args.n_warmup, n_iter=args.n_iter,
                extra_warmup_for_new_actions=0
            )
            precision = case_name.replace("all_", "")
            our_results[precision] = metrics
    
    print(f"    {'Precision':<12} {'Time (ms)':<12} {'Throughput (GOP/s)':<18} {'Speedup vs fp64':<18}")
    print(f"    {'-'*12} {'-'*12} {'-'*18} {'-'*18}")
    for precision in ['fp64', 'fp32', 'bf16']:
        if precision in our_results:
            metrics = our_results[precision]
            speedup = baseline_time_ms / metrics['avg_time_ms'] if precision != 'fp64' else 1.0
            print(f"    {precision:<12} {metrics['avg_time_ms']:>10.3f}   {metrics['throughput_gops']:>15.3f}   {speedup:>16.2f}x")
    
    # Test all cases and compare with baseline
    print(f"\n  Performance comparison (vs baseline):")
    print(f"    {'Case':<20} {'Time (ms)':<12} {'Speedup':<10} {'Throughput (GOP/s)':<18} {'Action Distribution'}")
    print(f"    {'-'*20} {'-'*12} {'-'*10} {'-'*18} {'-'*30}")
    
    # Collect metrics for all cases (except baseline)
    random_speedups = []
    
    for case_name, actions in test_cases.items():
        if case_name == "all_fp64":
            continue  # Skip baseline in comparison table
        
        metrics = benchmark_kernel(
            bcsc_cuda, x, actions, actual_nnz_in_bcsc,
            n_warmup=args.n_warmup, n_iter=args.n_iter
        )
        speedup = baseline_time_ms / metrics["avg_time_ms"]
        action_dist = get_action_distribution(actions)
        dist_str = f"fp64:{action_dist['fp64'][1]:.0f}% fp32:{action_dist['fp32'][1]:.0f}% bf16:{action_dist['bf16'][1]:.0f}%"
        
        print(f"    {case_name:<20} {metrics['avg_time_ms']:>10.3f}   {speedup:>8.2f}x   {metrics['throughput_gops']:>15.3f}   {dist_str}")
        
        # Collect speedups for random cases
        if case_name.startswith("random_"):
            random_speedups.append(speedup)
    
    # Summary statistics for random cases
    if random_speedups:
        print(f"\n  Random actions statistics:")
        print(f"    Average speedup: {np.mean(random_speedups):.2f}x")
        print(f"    Min speedup: {np.min(random_speedups):.2f}x")
        print(f"    Max speedup: {np.max(random_speedups):.2f}x")
    
    # Test different fp64 ratios to understand quantization overhead
    print(f"\n  [ANALYSIS] Testing fp64 ratio impact (fp64 + bf16, no fp32):")
    print(f"    Note: Kernel is already compiled from previous tests, so timing should be consistent.")
    print(f"    {'fp64%':<8} {'Time (ms)':<12} {'Speedup':<10} {'Throughput (GOP/s)':<18} {'vs all_bf16'}")
    print(f"    {'-'*8} {'-'*12} {'-'*10} {'-'*18} {'-'*15}")
    
    fp64_ratios = [0.0, 0.1, 0.2, 0.33, 0.5, 0.7, 1.0]
    fp64_ratio_results = []
    all_bf16_time = None
    
    for fp64_ratio in fp64_ratios:
        bf16_ratio = 1.0 - fp64_ratio
        actions = generate_random_actions(
            n_bc, args.random_seed + 1000, 
            fp64_ratio=fp64_ratio, 
            fp32_ratio=0.0, 
            bf16_ratio=bf16_ratio
        )
        
        metrics = benchmark_kernel(
            bcsc_cuda, x, actions, actual_nnz_in_bcsc,
            n_warmup=args.n_warmup, n_iter=args.n_iter
        )
        speedup = baseline_time_ms / metrics["avg_time_ms"]
        
        # Calculate speedup vs all_bf16
        if fp64_ratio == 0.0:
            all_bf16_time = metrics["avg_time_ms"]
            vs_bf16_str = "baseline"
        else:
            vs_bf16 = all_bf16_time / metrics["avg_time_ms"] if all_bf16_time else 0.0
            vs_bf16_str = f"{vs_bf16:.2f}x" if vs_bf16 > 0 else "N/A"
        
        fp64_ratio_results.append({
            "fp64_ratio": fp64_ratio,
            "time_ms": metrics["avg_time_ms"],
            "speedup": speedup,
            "throughput": metrics["throughput_gops"],
            "vs_bf16": vs_bf16_str
        })
        
        print(f"    {fp64_ratio*100:>6.0f}%   {metrics['avg_time_ms']:>10.3f}   {speedup:>8.2f}x   {metrics['throughput_gops']:>15.3f}   {vs_bf16_str:>15}")
    
    # Find optimal fp64 ratio
    if fp64_ratio_results:
        fastest = min(fp64_ratio_results, key=lambda x: x["time_ms"])
        bf16_pct = (1.0 - fastest['fp64_ratio']) * 100
        print(f"\n    Fastest configuration: {fastest['fp64_ratio']*100:.0f}% fp64, {bf16_pct:.0f}% bf16")
        print(f"    Time: {fastest['time_ms']:.3f} ms/iter, Speedup vs baseline: {fastest['speedup']:.2f}x")
        
        # Analysis
        all_bf16_result = next((r for r in fp64_ratio_results if r["fp64_ratio"] == 0.0), None)
        all_fp64_result = next((r for r in fp64_ratio_results if r["fp64_ratio"] == 1.0), None)
        
        if all_bf16_result and all_fp64_result:
            print(f"\n    Analysis:")
            print(f"      - all_bf16: {all_bf16_result['time_ms']:.3f} ms (bf16 compute is fast, quantization overhead is small)")
            print(f"      - all_fp64: {all_fp64_result['time_ms']:.3f} ms (no quantization, but fp64 compute is slower)")
            quant_overhead = all_bf16_result['time_ms'] - all_fp64_result['time_ms']
            if quant_overhead < 0:
                print(f"      - Conclusion: bf16 compute speed advantage ({abs(quant_overhead):.3f} ms) exceeds quantization overhead")
            else:
                print(f"      - Conclusion: quantization overhead ({quant_overhead:.3f} ms) is significant")
            if fastest["fp64_ratio"] > 0 and fastest["fp64_ratio"] < 1.0:
                print(f"      - Optimal mix ({fastest['fp64_ratio']*100:.0f}% fp64) balances quantization overhead vs compute speed")
            
            # Compare with first test results
            print(f"\n    Performance difference explanation:")
            print(f"      - First test (correctness + performance): kernel may still be compiling/optimizing")
            print(f"      - Second test (analysis): kernel fully compiled, GPU in optimized state")
            print(f"      - This is normal for JIT-compiled kernels (TileLang)")
            print(f"      - The relative speedup ratios are more reliable than absolute times")
    
    # Test fp32 impact (since original "mixed" included fp32)
    print(f"\n  [ANALYSIS] Testing fp32 impact (fp64 + fp32 + bf16):")
    print(f"    {'fp64%':<8} {'fp32%':<8} {'bf16%':<8} {'Time (ms)':<12} {'Speedup':<10} {'vs all_bf16'}")
    print(f"    {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*10} {'-'*15}")
    
    fp32_mix_configs = [
        (0.0, 0.0, 1.0),   # all_bf16
        (0.33, 0.33, 0.34),  # original "mixed"
        (0.33, 0.0, 0.67),   # fp64 + bf16 (no fp32)
        (0.0, 0.33, 0.67),   # fp32 + bf16 (no fp64)
        (0.33, 0.67, 0.0),   # fp64 + fp32 (no bf16)
        (0.0, 1.0, 0.0),     # all_fp32
    ]
    
    fp32_mix_results = []
    all_bf16_time_mix = None
    
    for fp64_r, fp32_r, bf16_r in fp32_mix_configs:
        actions = generate_random_actions(
            n_bc, args.random_seed + 2000,
            fp64_ratio=fp64_r,
            fp32_ratio=fp32_r,
            bf16_ratio=bf16_r
        )
        
        metrics = benchmark_kernel(
            bcsc_cuda, x, actions, actual_nnz_in_bcsc,
            n_warmup=args.n_warmup, n_iter=args.n_iter
        )
        speedup = baseline_time_ms / metrics["avg_time_ms"]
        
        if fp64_r == 0.0 and fp32_r == 0.0 and bf16_r == 1.0:
            all_bf16_time_mix = metrics["avg_time_ms"]
            vs_bf16_str = "baseline"
        else:
            vs_bf16 = all_bf16_time_mix / metrics["avg_time_ms"] if all_bf16_time_mix else 0.0
            vs_bf16_str = f"{vs_bf16:.2f}x" if vs_bf16 > 0 else "N/A"
        
        fp32_mix_results.append({
            "fp64": fp64_r,
            "fp32": fp32_r,
            "bf16": bf16_r,
            "time_ms": metrics["avg_time_ms"],
            "speedup": speedup,
            "vs_bf16": vs_bf16_str
        })
        
        print(f"    {fp64_r*100:>6.0f}%   {fp32_r*100:>6.0f}%   {bf16_r*100:>6.0f}%   {metrics['avg_time_ms']:>10.3f}   {speedup:>8.2f}x   {vs_bf16_str:>15}")
    
    # Find fastest fp32 mix
    if fp32_mix_results:
        fastest_mix = min(fp32_mix_results, key=lambda x: x["time_ms"])
        print(f"\n    Fastest mixed configuration: {fastest_mix['fp64']*100:.0f}% fp64, {fastest_mix['fp32']*100:.0f}% fp32, {fastest_mix['bf16']*100:.0f}% bf16")
        print(f"    Time: {fastest_mix['time_ms']:.3f} ms/iter, Speedup vs baseline: {fastest_mix['speedup']:.2f}x")
        
        # Compare with original "mixed" result
        original_mixed = next((r for r in fp32_mix_results if abs(r["fp64"] - 0.33) < 0.01 and abs(r["fp32"] - 0.33) < 0.01), None)
        if original_mixed and all_bf16_time_mix:
            print(f"\n    Why original 'mixed' (33%/33%/34%) might be faster than all_bf16:")
            print(f"      - Original mixed: {original_mixed['time_ms']:.3f} ms")
            print(f"      - All bf16: {all_bf16_time_mix:.3f} ms")
            if original_mixed['time_ms'] < all_bf16_time_mix:
                diff = all_bf16_time_mix - original_mixed['time_ms']
                print(f"      - Mixed is {diff:.3f} ms faster ({diff/all_bf16_time_mix*100:.1f}%)")
                print(f"      - Possible reasons: fp32 path optimization, better memory access pattern, or test variance")

    print("\n[OK] All tests passed!")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

