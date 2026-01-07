#!/usr/bin/env python3
"""
Stage A toy test:
  - Build BCSC from a 6x6 toy sparse matrix with tilesize=2
  - Assert (colptr,rowind) matches the dev doc
  - Reconstruct dense and compare with ground truth

Non-interactive: exits non-zero on failure.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import build_bcsc_from_coo, bcsc_to_dense  # noqa: E402


def build_toy_dense() -> np.ndarray:
    A = np.zeros((6, 6), dtype=np.float64)

    T00 = np.array([[1, 2], [3, 4]], dtype=np.float64)
    T10 = np.array([[-1, 0], [0, -2]], dtype=np.float64)
    T21 = np.array([[5, 0], [7, 8]], dtype=np.float64)
    T02 = np.array([[0, 9], [10, 11]], dtype=np.float64)
    T22 = np.array([[-3, -4], [0, 12]], dtype=np.float64)

    # place tiles
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
    A = build_toy_dense()
    row, col, data, shape = dense_to_coo(A)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=2, device="cpu")

    colptr = bcsc.colptr.cpu().numpy().tolist()
    rowind = bcsc.rowind.cpu().numpy().tolist()

    exp_colptr = [0, 2, 3, 5]
    exp_rowind = [0, 1, 2, 0, 2]

    assert colptr == exp_colptr, f"colptr mismatch: got {colptr}, expected {exp_colptr}"
    assert rowind == exp_rowind, f"rowind mismatch: got {rowind}, expected {exp_rowind}"

    A_rec = bcsc_to_dense(bcsc, trim=True).cpu().numpy()
    assert np.allclose(A_rec, A, atol=0.0, rtol=0.0), "BCSC->dense reconstruction mismatch"

    print("[OK] Stage A toy: build_bcsc + bcsc_to_dense passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()





