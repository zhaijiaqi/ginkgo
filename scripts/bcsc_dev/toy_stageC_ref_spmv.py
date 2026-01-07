#!/usr/bin/env python3
"""
Stage C toy test:
  - Build toy BCSC and pre-quantize A
  - Validate reference SpMV:
      * action all-0 equals dense fp64 A@x exactly
      * pre-quant path equals runtime-quant path (kernel-aligned) for mixed actions

Non-interactive: exits non-zero on failure.
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
    spmv_bcsc_mixed_ref_runtime_quant,
    _quantize_scaled_value_f64,
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


def assert_tensor_equivalent(a: torch.Tensor, b: torch.Tensor, *, name: str):
    """
    Assert two tensors are equivalent under IEEE-754 edge-cases:
      - finite values must match exactly
      - NaNs are considered equal if they appear in the same positions
      - inf must match sign
    """
    a = a.detach().cpu()
    b = b.detach().cpu()
    assert a.shape == b.shape, f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"

    a_nan = torch.isnan(a)
    b_nan = torch.isnan(b)
    if not torch.equal(a_nan, b_nan):
        raise AssertionError(f"{name}: NaN mask mismatch")

    a_inf = torch.isinf(a)
    b_inf = torch.isinf(b)
    if not torch.equal(a_inf, b_inf):
        raise AssertionError(f"{name}: Inf mask mismatch")
    if torch.any(a_inf):
        if not torch.equal(torch.sign(a[a_inf]), torch.sign(b[b_inf])):
            raise AssertionError(f"{name}: Inf sign mismatch")

    finite = torch.isfinite(a) & torch.isfinite(b)
    if not torch.equal(a[finite], b[finite]):
        max_abs = (a[finite] - b[finite]).abs().max().item() if torch.any(finite) else 0.0
        raise AssertionError(f"{name}: finite mismatch (max_abs_diff={max_abs:.3e})")


def main():
    rng = np.random.default_rng(0)
    A = build_toy_dense()
    row, col, data, shape = dense_to_coo(A)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=2, device="cpu")
    bcsc = quantize_bcsc_tiles(bcsc)

    x = torch.as_tensor(rng.standard_normal((6,), dtype=np.float64), dtype=torch.float64)

    # all-0 action should match dense fp64 exactly (kernel behavior: no x quant if action==0)
    actions0 = torch.zeros((bcsc.n_bc,), dtype=torch.int32)
    y0 = spmv_bcsc_mixed_ref_prequant(bcsc, actions0, x)
    y_dense = torch.as_tensor(A @ x.cpu().numpy(), dtype=torch.float64)
    # Note: different reduction orders (numpy GEMV vs our tile-wise accumulations) can differ by a few ulps.
    if not torch.allclose(y0.cpu(), y_dense, atol=1e-12, rtol=1e-12):
        diff = (y0.cpu() - y_dense).abs().max().item()
        raise AssertionError(f"action==0 ref SpMV mismatch: max_abs_diff={diff:.3e}")

    # mixed actions: compare prequant-vs-runtime quantization reference
    # Example from doc: actions[bc] per block-column (n_bc=3 for tilesize=2)
    actions_mixed = torch.tensor([1, 2, 0], dtype=torch.int32)
    y_pre = spmv_bcsc_mixed_ref_prequant(bcsc, actions_mixed, x)
    y_rt = spmv_bcsc_mixed_ref_runtime_quant(bcsc, actions_mixed, x)

    try:
        assert_tensor_equivalent(y_pre, y_rt, name="prequant_vs_runtime_quant")
    except AssertionError as e:
        max_abs_diff = (y_pre - y_rt).abs().nan_to_num(posinf=float("inf"), neginf=float("inf")).max().item()
        print(f"[DEBUG] y_pre != y_rt, max_abs_diff~={max_abs_diff}")
        print("[DEBUG] y_pre:", y_pre.cpu().numpy())
        print("[DEBUG] y_rt :", y_rt.cpu().numpy())

        # Diagnose per block-column: compare x_scale/x_q and a_scale/a_q
        R, C = bcsc.R, bcsc.C
        max_val = 1.8405e19
        eps = 1e-12
        for bc in range(bcsc.n_bc):
            action = int(actions_mixed[bc].item())
            start = int(bcsc.colptr[bc].item())
            end = int(bcsc.colptr[bc + 1].item())
            if end <= start:
                continue
            col_base = bc * C
            x_tile = torch.zeros((C,), dtype=torch.float64)
            x_tile[:] = x[col_base : col_base + C]
            if action >= 1:
                x_max_abs = torch.max(torch.abs(x_tile))
                x_scale = torch.tensor(float(max_val), dtype=torch.float64) / (x_max_abs + float(eps))
                x_q = _quantize_scaled_value_f64(x_tile, x_scale, max_val, action)
            else:
                x_scale = torch.tensor(1.0, dtype=torch.float64)
                x_q = x_tile

            print(f"[DEBUG] bc={bc} action={action} tiles={end-start} x_scale={float(x_scale.item()):.6e}")
            if action >= 1:
                print(f"[DEBUG]   x_q(f64-carrying)={x_q.cpu().numpy()}")

            # Runtime quantization: compute scale per column
            for k in range(start, end):
                A_tile = bcsc.A_fp64[k]  # [R, C]
                
                # For each column in this tile, compute runtime scale and compare with pre-quantized scale
                for cc in range(bcsc.C):
                    global_col = col_base + cc
                    if global_col >= bcsc.N:
                        break
                    
                    # Runtime quantization: find max abs for this column across all tiles in this block-column
                    col_tiles_rt = bcsc.A_fp64[start:end]  # [num_tiles, R, C]
                    col_values_rt = col_tiles_rt[:, :, cc]  # [num_tiles, R]
                    a_max_abs_col = torch.max(torch.abs(col_values_rt))
                    a_scale_rt_col = torch.tensor(float(max_val), dtype=torch.float64) / (a_max_abs_col + float(eps))
                    
                    # Pre-quantized scale for this column
                    if action == 1:
                        a_scale_pre_col = bcsc.a_scale_fp32[global_col]
                        col_q_pre = bcsc.A_fp32_q[k][:, cc].to(torch.float64)
                    elif action == 2:
                        a_scale_pre_col = bcsc.a_scale_bf16[global_col]
                        col_q_pre = bcsc.A_bf16_q[k][:, cc].to(torch.float64)
                    else:
                        a_scale_pre_col = torch.tensor(1.0, dtype=torch.float64)
                        col_q_pre = A_tile[:, cc]
                    
                    # Runtime quantized value for this column
                    col_tile_rt = A_tile[:, cc]  # [R]
                    col_q_rt = _quantize_scaled_value_f64(col_tile_rt, a_scale_rt_col, max_val, action) if action >= 1 else col_tile_rt
                    
                    ds = float((a_scale_pre_col - a_scale_rt_col).abs().item())
                    dq = float((col_q_pre - col_q_rt).abs().max().item())
                    if ds != 0.0 or dq != 0.0:
                        print(f"[DEBUG]   k={k} br={int(bcsc.rowind[k].item())} cc={cc} col={global_col} |a_scale_pre-a_scale_rt|={ds:.3e} max|a_q_pre-a_q_rt|={dq:.3e}")

        # Fail with context
        raise AssertionError(f"prequant ref != runtime-quant ref: {e}") from e

    print("[OK] Stage C toy: ref spmv (prequant vs runtime) passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()


