#!/usr/bin/env python3
"""
Edge case tests for BCSC prequant implementation.

Tests:
  - Empty matrix (no non-zero tiles)
  - Single tile matrix
  - Boundary conditions (tilesize > matrix size)
  - Very sparse matrix (few tiles)

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
)


def test_empty_matrix():
    """Test BCSC build for empty matrix."""
    M, N = 64, 64
    tilesize = 32
    row = np.array([], dtype=np.int64)
    col = np.array([], dtype=np.int64)
    data = np.array([], dtype=np.float64)
    shape = (M, N)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=tilesize, device="cpu")
    assert bcsc.nnzb == 0, f"Empty matrix should have nnzb=0, got {bcsc.nnzb}"
    assert bcsc.colptr.shape == (bcsc.n_bc + 1,), "colptr shape should match n_bc+1"
    assert torch.all(bcsc.colptr == 0), "Empty matrix: all colptr should be 0"

    bcsc = quantize_bcsc_tiles(bcsc)
    # Scale is now per column, so it should have N elements
    assert bcsc.a_scale_fp32 is not None and len(bcsc.a_scale_fp32) == bcsc.N, f"Empty: a_scale should have N={bcsc.N} elements, got {len(bcsc.a_scale_fp32)}"

    # SpMV on empty matrix should return zeros
    x = torch.ones((N,), dtype=torch.float64)
    actions = torch.zeros((bcsc.n_bc,), dtype=torch.int32)
    y = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)
    assert torch.all(y == 0), "Empty matrix SpMV should return zeros"

    print("[OK] Empty matrix test passed")


def test_single_tile():
    """Test BCSC with single non-zero tile."""
    M, N = 32, 32
    tilesize = 32
    # Single tile at (0,0)
    row = np.array([0, 1, 15], dtype=np.int64)
    col = np.array([0, 1, 15], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    shape = (M, N)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=tilesize, device="cpu")
    assert bcsc.nnzb == 1, f"Single tile should have nnzb=1, got {bcsc.nnzb}"
    assert bcsc.n_br == 1 and bcsc.n_bc == 1, f"Single tile: n_br={bcsc.n_br}, n_bc={bcsc.n_bc}"

    bcsc = quantize_bcsc_tiles(bcsc)
    # Scale is now per column, so it should have N elements
    assert bcsc.a_scale_fp32 is not None and len(bcsc.a_scale_fp32) == bcsc.N, f"Single tile: a_scale should have N={bcsc.N} elements, got {len(bcsc.a_scale_fp32)}"

    # Test SpMV
    x = torch.ones((N,), dtype=torch.float64)
    actions = torch.zeros((bcsc.n_bc,), dtype=torch.int32)
    y = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)
    # For a single tile with elements at (0,0)=1.0, (1,1)=2.0, (15,15)=3.0,
    # and x all ones, we get: y[0]=1.0, y[1]=2.0, y[15]=3.0, others=0
    # So y[0] should be 1.0, not the sum of all data
    assert torch.allclose(y[0], torch.tensor(1.0, dtype=torch.float64), atol=1e-10), f"Single tile: y[0] should be 1.0, got {y[0].item()}"
    assert torch.allclose(y[1], torch.tensor(2.0, dtype=torch.float64), atol=1e-10), f"Single tile: y[1] should be 2.0, got {y[1].item()}"
    assert torch.allclose(y[15], torch.tensor(3.0, dtype=torch.float64), atol=1e-10), f"Single tile: y[15] should be 3.0, got {y[15].item()}"

    print("[OK] Single tile test passed")


def test_large_tilesize():
    """Test BCSC with tilesize larger than matrix dimensions."""
    M, N = 16, 16
    tilesize = 64  # Larger than matrix
    row = np.array([0, 5, 10], dtype=np.int64)
    col = np.array([0, 5, 10], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    shape = (M, N)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=tilesize, device="cpu")
    # Should still work: n_br = ceil(M/tilesize) = ceil(16/64) = 1
    assert bcsc.n_br == 1, f"Large tilesize: n_br should be 1, got {bcsc.n_br}"
    assert bcsc.n_bc == 1, f"Large tilesize: n_bc should be 1, got {bcsc.n_bc}"

    bcsc = quantize_bcsc_tiles(bcsc)
    # SpMV should still work
    x = torch.ones((N,), dtype=torch.float64)
    actions = torch.zeros((bcsc.n_bc,), dtype=torch.int32)
    y = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)
    assert torch.isfinite(y).all(), "Large tilesize: SpMV should be finite"

    print("[OK] Large tilesize test passed")


def test_very_sparse():
    """Test BCSC with very sparse matrix (few tiles)."""
    M, N = 1285, 1285
    tilesize = 32
    # Only 3 non-zero elements, all in different tiles
    row = np.array([10, 50, 100], dtype=np.int64)
    col = np.array([20, 60, 110], dtype=np.int64)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    shape = (M, N)

    bcsc = build_bcsc_from_coo(row, col, data, shape, tilesize=tilesize, device="cpu")
    assert bcsc.nnzb == 3, f"Very sparse: should have nnzb=3, got {bcsc.nnzb}"

    bcsc = quantize_bcsc_tiles(bcsc)
    # Test with mixed actions
    x = torch.ones((N,), dtype=torch.float64)
    actions = torch.tensor([0, 1, 2, 0], dtype=torch.int32)  # More actions than tiles (should handle gracefully)
    if len(actions) <= bcsc.n_bc:
        y = spmv_bcsc_mixed_ref_prequant(bcsc, actions, x)
        assert torch.isfinite(y).all(), "Very sparse: SpMV should be finite"

    print("[OK] Very sparse matrix test passed")


def main():
    print("[INFO] Running BCSC edge case tests...")
    test_empty_matrix()
    test_single_tile()
    test_large_tilesize()
    test_very_sparse()
    print("[OK] All edge case tests passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

