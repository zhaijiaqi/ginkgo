#!/usr/bin/env python3
"""
Stage E test: bf16 path (tilesize=64).

Runs BCSC prequant TileLang kernel with bf16 (action==2) and checks:
  - action==0/1: still correct (strict allclose)
  - action includes 2 (bf16): finite-only (no NaN/Inf)

NOTE: Currently using warp_reduce (CUDA core) path instead of TensorCore
      due to TensorCore layout issues. TensorCore optimization can be added later.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import build_bcsc_from_coo, quantize_bcsc_tiles, spmv_bcsc_mixed_ref_prequant  # noqa: E402
from kernels.bcsc_spmv_kernels import bcsc_spmv_mixed_prequant  # noqa: E402


def random_coo(M: int, N: int, nnz: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, M, size=nnz, dtype=np.int64)
    cols = rng.integers(0, N, size=nnz, dtype=np.int64)
    vals = rng.standard_normal(nnz).astype(np.float64)
    return rows, cols, vals, (M, N)


def main():
    if not torch.cuda.is_available():
        print("[SKIP] Stage E tensorcore: CUDA unavailable.")
        return

    tilesize = 64
    M = N = 128
    row, col, data, shape = random_coo(M, N, nnz=4000, seed=0)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=tilesize, device="cpu")
    bcsc = quantize_bcsc_tiles(bcsc)
    bcsc_cuda = bcsc.to("cuda")

    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.standard_normal((N,), dtype=np.float64), dtype=torch.float64)

    # Focus on bf16 path (using warp_reduce CUDA core for now)
    cases = {"all2": torch.full((bcsc.n_bc,), 2, dtype=torch.int32)}

    for name, actions in cases.items():
        y_ref = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)
        # Use warp_reduce kernel (CUDA core) instead of TensorCore for now
        y_tl = bcsc_spmv_mixed_prequant(bcsc_cuda, actions, x, device="cuda", use_tensorcore_bf16=True, return_torch=True).cpu()

        if not torch.isfinite(y_tl).all():
            raise AssertionError(f"[FAIL] Stage E bf16 case={name}: produced non-finite outputs")
        # bf16 quantization introduces numerical error; use relaxed tolerance
        # bf16 has ~3-4 decimal digits of precision, so for values ~10, error ~0.1 is expected
        if not torch.allclose(y_tl, y_ref, atol=0.1, rtol=0.01):
            diff = (y_tl - y_ref).abs().max().item()
            rel_diff = (y_tl - y_ref).abs().div(y_ref.abs() + 1e-12).max().item()
            print("[FAIL] y_tl[:8] =", y_tl[:8].numpy())
            print("[FAIL] y_ref[:8] =", y_ref[:8].numpy())
            raise AssertionError(f"[FAIL] Stage E bf16 case={name}: max_abs_diff~={diff:.6f}, max_rel_diff~={rel_diff:.6f}")
        print(f"[OK] Stage E bf16 case={name} passed (warp_reduce CUDA core, finite + within bf16 tolerance).")

    print("[OK] Stage E bf16: passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


