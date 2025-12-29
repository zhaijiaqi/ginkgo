#!/usr/bin/env python3
"""
Stage D.5: bodyy4 kernel check (sampled, non-interactive).

Goal:
  - Build BCSC for bodyy4.mtx
  - Sample a small subset of block-columns (bc) to keep runtime low
  - Run TileLang BCSC prequant kernel (CUDA) on the sampled BCSC
  - Compare to Stage C reference on the same sampled BCSC

Policy:
  - actions in {0,1}: strict allclose
  - if bf16 (2) is present: finite-only (no NaN/Inf)

Skips cleanly if:
  - scipy unavailable
  - matrix file missing
  - CUDA unavailable
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import (  # noqa: E402
    BCSCMatrix,
    build_bcsc_from_scipy_csr,
    quantize_bcsc_tiles,
    spmv_bcsc_mixed_ref_prequant,
)
from kernels.bcsc_spmv_kernels import bcsc_spmv_mixed_prequant  # noqa: E402


def _filter_bcsc_by_blockcols(bcsc: BCSCMatrix, keep_bcs: List[int]) -> BCSCMatrix:
    """
    Create a new BCSCMatrix with the same n_bc, but only tiles from `keep_bcs`.
    Other block-columns become empty (colptr entries unchanged as we advance).
    """
    keep = set(int(b) for b in keep_bcs)
    dev = bcsc.A_fp64.device

    colptr_old = bcsc.colptr.to(device="cpu", dtype=torch.int32).numpy()
    rowind_old = bcsc.rowind.to(device="cpu", dtype=torch.int32).numpy()
    A_old = bcsc.A_fp64.to(device="cpu", dtype=torch.float64).numpy()

    n_bc = int(bcsc.n_bc)
    R = int(bcsc.R)
    C = int(bcsc.C)

    new_tiles = []
    new_rowind = []
    colptr_new = np.zeros((n_bc + 1,), dtype=np.int32)

    nnzb_new = 0
    colptr_new[0] = 0
    for bc in range(n_bc):
        start = int(colptr_old[bc])
        end = int(colptr_old[bc + 1])
        if bc in keep and end > start:
            new_tiles.append(A_old[start:end])
            new_rowind.append(rowind_old[start:end])
            nnzb_new += (end - start)
        colptr_new[bc + 1] = nnzb_new

    if nnzb_new == 0:
        A_new = torch.zeros((0, R, C), dtype=torch.float64, device=dev)
        rowind_new = torch.zeros((0,), dtype=torch.int32, device=dev)
    else:
        A_new_np = np.concatenate(new_tiles, axis=0)
        rowind_new_np = np.concatenate(new_rowind, axis=0).astype(np.int32, copy=False)
        A_new = torch.as_tensor(A_new_np, dtype=torch.float64, device=dev)
        rowind_new = torch.as_tensor(rowind_new_np, dtype=torch.int32, device=dev)

    colptr_new_t = torch.as_tensor(colptr_new, dtype=torch.int32, device=dev)

    return BCSCMatrix(
        M=int(bcsc.M),
        N=int(bcsc.N),
        R=R,
        C=C,
        n_br=int(bcsc.n_br),
        n_bc=n_bc,
        nnzb=int(nnzb_new),
        colptr=colptr_new_t,
        rowind=rowind_new,
        A_fp64=A_new,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtx", type=str, default="/home/bingxing2/home/scx7axu/data/matrix/bodyy4.mtx")
    ap.add_argument("--tilesize", type=int, default=64)
    ap.add_argument("--sample-bc", type=int, default=16, help="How many non-empty block-columns to sample.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-bf16", action="store_true", help="Allow action==2 in sampled actions.")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.device != "cuda":
        print("[SKIP] bodyy4_kernel_check: only supports --device=cuda.")
        return

    if not torch.cuda.is_available():
        print("[SKIP] bodyy4_kernel_check: CUDA unavailable.")
        return

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

    # Build BCSC on CPU first (cheap), then sample and move to CUDA in kernel wrapper.
    bcsc_full = build_bcsc_from_scipy_csr(A, int(args.tilesize), device="cpu")
    print(f"[INFO] BCSC(full) n_bc={bcsc_full.n_bc} nnzb={bcsc_full.nnzb}")

    # Choose non-empty block-columns
    colptr = bcsc_full.colptr.cpu().numpy()
    nonempty_bcs = [bc for bc in range(bcsc_full.n_bc) if int(colptr[bc + 1] - colptr[bc]) > 0]
    if not nonempty_bcs:
        print("[SKIP] bodyy4_kernel_check: matrix has no non-empty block-columns at this tilesize.")
        return

    rng = np.random.default_rng(int(args.seed))
    k = int(min(max(1, args.sample_bc), len(nonempty_bcs)))
    keep_bcs = rng.choice(np.asarray(nonempty_bcs, dtype=np.int32), size=k, replace=False).tolist()
    keep_bcs = [int(b) for b in keep_bcs]
    keep_bcs.sort()
    print(f"[INFO] Sampling {len(keep_bcs)} block-columns (non-empty) for kernel check.")

    bcsc = _filter_bcsc_by_blockcols(bcsc_full, keep_bcs)
    bcsc = quantize_bcsc_tiles(bcsc)

    # Build x/actions
    x = torch.as_tensor(rng.standard_normal((N,), dtype=np.float64), dtype=torch.float64)
    if args.include_bf16:
        actions_np = rng.integers(0, 3, size=(bcsc.n_bc,), dtype=np.int32)
    else:
        actions_np = rng.integers(0, 2, size=(bcsc.n_bc,), dtype=np.int32)
    actions = torch.as_tensor(actions_np, dtype=torch.int32)

    # Reference on CPU
    y_ref = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)

    # Kernel on CUDA
    bcsc_cuda = bcsc.to("cuda")
    y_tl_full = bcsc_spmv_mixed_prequant(bcsc_cuda, actions, x, device="cuda", return_torch=True).cpu()
    # Kernel output is padded to n_br*R; trim to true M for comparison.
    y_tl = y_tl_full[: int(bcsc.M)]

    has_bf16 = bool(np.any(actions_np == 2))
    if has_bf16:
        if not torch.isfinite(y_tl).all():
            raise AssertionError("[FAIL] bodyy4_kernel_check: produced non-finite outputs (bf16 present).")
        print("[OK] bodyy4_kernel_check passed (finite-only; bf16 present).")
        return

    if not torch.allclose(y_tl, y_ref, atol=1e-6, rtol=1e-6):
        diff = (y_tl - y_ref).abs().max().item()
        print("[FAIL] y_tl[:8] =", y_tl[:8].numpy())
        print("[FAIL] y_ref[:8] =", y_ref[:8].numpy())
        raise AssertionError(f"[FAIL] bodyy4_kernel_check: max_abs_diff~={diff}")

    print("[OK] bodyy4_kernel_check passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


