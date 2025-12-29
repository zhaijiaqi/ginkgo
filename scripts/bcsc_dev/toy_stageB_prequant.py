#!/usr/bin/env python3
"""
Stage B toy test:
  - Build BCSC from the 6x6 toy matrix (tilesize=2)
  - Pre-quantize tiles into fp32/bf16 + a_scale (kernel-aligned math)
  - Spot-check one tile (T22) for exact a_scale and quantized payloads

Non-interactive: exits non-zero on failure.
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kernels.bcsc_prequant import (  # noqa: E402
    DEFAULT_QMAX_F64,
    DEFAULT_EPS_F64,
    build_bcsc_from_coo,
    quantize_bcsc_tiles,
)


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
    A = build_toy_dense()
    row, col, data, shape = dense_to_coo(A)
    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=2, device="cpu")
    bcsc = quantize_bcsc_tiles(bcsc, qmax_fp32=DEFAULT_QMAX_F64, qmax_bf16=DEFAULT_QMAX_F64, eps=DEFAULT_EPS_F64)

    assert bcsc.A_fp32_q is not None and bcsc.a_scale_fp32 is not None
    assert bcsc.A_bf16_q is not None and bcsc.a_scale_bf16 is not None
    assert bcsc.A_fp32_q.dtype == torch.float32
    assert bcsc.A_bf16_q.dtype == torch.bfloat16
    assert bcsc.a_scale_fp32.dtype == torch.float64
    assert bcsc.a_scale_bf16.dtype == torch.float64

    # Locate T22 in our toy ordering: k=4 is (bc=2, br=2)
    k = 4
    A_tile = bcsc.A_fp64[k].cpu().numpy()
    exp_tile = np.array([[-3, -4], [0, 12]], dtype=np.float64)
    assert np.allclose(A_tile, exp_tile, atol=0.0, rtol=0.0), "T22 tile mismatch (build ordering changed?)"

    max_abs = np.max(np.abs(exp_tile))
    exp_scale = float(DEFAULT_QMAX_F64) / (float(max_abs) + float(DEFAULT_EPS_F64))

    got_scale32 = float(bcsc.a_scale_fp32[k].item())
    got_scalebf = float(bcsc.a_scale_bf16[k].item())
    assert got_scale32 == exp_scale, f"a_scale_fp32 mismatch: {got_scale32} vs {exp_scale}"
    assert got_scalebf == exp_scale, f"a_scale_bf16 mismatch: {got_scalebf} vs {exp_scale}"

    # Expected quantized-domain values: cast_lowp(clamp(scale * A))
    scaled = exp_tile * exp_scale
    scaled = np.clip(scaled, -float(DEFAULT_QMAX_F64), float(DEFAULT_QMAX_F64))
    # NOTE: some torch builds can't convert bf16 tensors to numpy; compare as torch tensors.
    exp_fp32_q_t = torch.from_numpy(scaled).to(torch.float32)
    # bf16 cast must clamp to bf16 max finite to avoid overflow to inf.
    exp_bf16_q_t = torch.from_numpy(scaled).to(torch.bfloat16)

    got_fp32_q_t = bcsc.A_fp32_q[k].cpu()
    got_bf16_q_t = bcsc.A_bf16_q[k].cpu()

    assert torch.equal(got_fp32_q_t, exp_fp32_q_t), "A_fp32_q mismatch for T22"
    assert torch.equal(got_bf16_q_t, exp_bf16_q_t), "A_bf16_q mismatch for T22"

    print("[OK] Stage B toy: quantize_bcsc_tiles passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


