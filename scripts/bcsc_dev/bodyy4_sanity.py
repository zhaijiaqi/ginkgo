#!/usr/bin/env python3
"""
bodyy4.mtx sanity script (non-interactive).

Default: --quick
  - Build BCSC
  - Pre-quantize tiles
  - Check invariants (colptr monotonic, indices range, finite scales)
  - Print basic stats

Optional: --spmv-check
  - Run reference SpMV on a *small sampled* action vector (random) and compare
    prequant vs runtime quant for correctness (still can be slow on huge matrices).
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import (  # noqa: E402
    build_bcsc_from_scipy_csr,
    quantize_bcsc_tiles,
    spmv_bcsc_mixed_ref_prequant,
    spmv_bcsc_mixed_ref_runtime_quant,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtx", type=str, default="/home/bingxing2/home/scx7axu/data/matrix/bodyy4.mtx")
    ap.add_argument("--tilesize", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--quick", action="store_true", help="Only build+quantize+invariants (default).")
    ap.add_argument("--spmv-check", action="store_true", help="Also compare prequant vs runtime-quant SpMV (can be slow).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.quick or args.spmv_check):
        args.quick = True

    try:
        from scipy.io import mmread
        from scipy.sparse import csr_matrix
    except Exception as e:
        print(f"[SKIP] scipy unavailable: {e}")
        return

    if not os.path.exists(args.mtx):
        print(f"[SKIP] matrix file not found: {args.mtx}")
        return

    A = mmread(args.mtx)
    if not isinstance(A, csr_matrix):
        A = A.tocsr()

    M, N = A.shape
    print(f"[INFO] Loaded {os.path.basename(args.mtx)} shape=({M},{N}) nnz={A.nnz} tilesize={args.tilesize}")

    bcsc = build_bcsc_from_scipy_csr(A, args.tilesize, device=args.device)
    print(f"[INFO] BCSC n_br={bcsc.n_br} n_bc={bcsc.n_bc} nnzb={bcsc.nnzb}")

    # Invariants
    colptr = bcsc.colptr.cpu().numpy()
    rowind = bcsc.rowind.cpu().numpy()
    assert colptr.shape == (bcsc.n_bc + 1,)
    assert colptr[0] == 0
    assert colptr[-1] == bcsc.nnzb
    assert np.all(colptr[1:] >= colptr[:-1]), "colptr must be non-decreasing"
    assert np.all((rowind >= 0) & (rowind < bcsc.n_br)), "rowind out of range"

    # Quantize
    bcsc = quantize_bcsc_tiles(bcsc)
    a32 = bcsc.a_scale_fp32.cpu().numpy() if bcsc.a_scale_fp32 is not None else np.array([])
    abf = bcsc.a_scale_bf16.cpu().numpy() if bcsc.a_scale_bf16 is not None else np.array([])
    assert np.all(np.isfinite(a32)) and np.all(a32 > 0), "a_scale_fp32 must be finite and >0"
    assert np.all(np.isfinite(abf)) and np.all(abf > 0), "a_scale_bf16 must be finite and >0"

    print(f"[INFO] a_scale_fp32: min={a32.min():.3e} max={a32.max():.3e} mean={a32.mean():.3e}")
    print(f"[INFO] a_scale_bf16: min={abf.min():.3e} max={abf.max():.3e} mean={abf.mean():.3e}")

    if args.spmv_check:
        rng = np.random.default_rng(args.seed)
        x = torch.as_tensor(rng.standard_normal((N,), dtype=np.float64), dtype=torch.float64, device=bcsc.A_fp64.device)
        actions = torch.as_tensor(rng.integers(low=0, high=3, size=(bcsc.n_bc,), dtype=np.int32), dtype=torch.int32, device=bcsc.A_fp64.device)
        y_pre = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)
        y_rt = spmv_bcsc_mixed_ref_runtime_quant(bcsc, actions, x)
        assert torch.equal(y_pre, y_rt) or torch.allclose(y_pre, y_rt, atol=0.0, rtol=0.0)
        print("[OK] bodyy4 spmv-check: prequant == runtime-quant (ref) passed.")

    print("[OK] bodyy4 sanity passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


