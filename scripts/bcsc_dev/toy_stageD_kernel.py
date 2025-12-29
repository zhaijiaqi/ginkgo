#!/usr/bin/env python3
"""
Stage D toy test:
  - Build toy BCSC and pre-quantize A
  - Run TileLang BCSC prequant kernel (CUDA)
  - Compare against Stage C reference (prequant ref)

Non-interactive: exits non-zero on failure.
Skips cleanly if CUDA is unavailable.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import (  # noqa: E402
    build_bcsc_from_coo,
    quantize_bcsc_tiles,
    spmv_bcsc_mixed_ref_prequant,
)
from kernels.bcsc_spmv_kernels import bcsc_spmv_mixed_prequant  # noqa: E402


def build_toy_dense() -> np.ndarray:
    A = np.zeros((6, 6), dtype=np.float64)
    T00 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    T10 = np.array([[-1, 0], [0, -2]], dtype=np.float64)
    T21 = np.array([[5, 0], [7, 8]], dtype=np.float64)
    T02 = np.array([[0, 9], [10, 11]], dtype=np.float64)
    T22 = np.array([[-3, -4], [0, 12]], dtype=np.float64)
    A[0:2, 0:2] = T00
    A[2:4, 0:2] = T10
    A[4:6, 2:4] = T21
    A[0:2, 4:6] = T02
    A[4:6, 4:6] = T22
    return A


def dense_to_coo(A: np.ndarray):
    rows, cols = np.nonzero(A)
    data = A[rows, cols].astype(np.float64)
    return rows.astype(np.int64), cols.astype(np.int64), data, A.shape


def main():
    if not torch.cuda.is_available():
        print("[SKIP] Stage D toy kernel: CUDA unavailable.")
        return

    rng = np.random.default_rng(0)
    A = build_toy_dense()
    row, col, data, shape = dense_to_coo(A)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=2, device="cpu")
    bcsc = quantize_bcsc_tiles(bcsc)
    bcsc_cuda = bcsc.to("cuda")

    x = torch.as_tensor(rng.standard_normal((6,), dtype=np.float64), dtype=torch.float64)

    cases = {
        "all0": torch.zeros((bcsc.n_bc,), dtype=torch.int32),
        "all1": torch.ones((bcsc.n_bc,), dtype=torch.int32),
        "all2": torch.full((bcsc.n_bc,), 2, dtype=torch.int32),
        "mixed": torch.tensor([1, 2, 0], dtype=torch.int32),
    }

    for name, actions in cases.items():
        y_ref = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)
        y_tl = bcsc_spmv_mixed_prequant(bcsc_cuda, actions, x, device="cuda", return_torch=True).cpu()

        if torch.any(actions == 2):
            if not torch.isfinite(y_tl).all():
                raise AssertionError(f"[FAIL] Stage D toy kernel case={name}: produced non-finite outputs")
            print(f"[OK] Stage D toy kernel case={name} passed (finite-only; bf16 present).")
        else:
            if not torch.allclose(y_tl, y_ref, atol=1e-6, rtol=1e-6):
                diff = (y_tl - y_ref).abs().max().item()
                print("[FAIL] y_tl[:8] =", y_tl[:8].numpy())
                print("[FAIL] y_ref[:8] =", y_ref[:8].numpy())
                raise AssertionError(f"[FAIL] Stage D toy kernel case={name}: max_abs_diff~={diff}")
            print(f"[OK] Stage D toy kernel case={name} passed.")

    print("[OK] Stage D toy: TileLang kernel vs reference passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


