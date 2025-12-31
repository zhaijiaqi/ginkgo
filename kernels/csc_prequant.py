"""
CSC (Compressed Sparse Column) utilities + pre-quantization helpers.

Standard CSC format with tile-based quantization:
  colptr : (N+1,) int32   # column pointer
  rowind : (nnz,) int32   # row indices
  values : (nnz,) float64 # non-zero values

Quantization rules:
  - Actions are per block-column: actions: (n_bc,) int32 where n_bc = (N+C-1)//C
    Every C columns share the same action (block-column bc = col // C)
  - All elements in the same column share the same action
  - But each element may belong to different tiles, so a_scale is per-tile
  - a_scale: (n_tiles,) float64  # per-tile scale (each tile has its own scale)

Key rule (for action>=1):
  scale = max_val / (max_abs + eps)  # computed per tile for A, per column for x
  q_scaled(v) = cast_lowp( clamp(scale * v, [-max_val, max_val]) )
  y += lowp_mul(q_scaled(A), q_scaled(x)) / (a_scale * x_scale)

For action==0:
  no quantization; scales are 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


DEFAULT_EPS_F64 = 1e-12
# Mul-safe qmax ~= sqrt(max_finite_lowp) so lowp_mul(a_q, x_q) won't overflow.
DEFAULT_QMAX_F64 = 1.8405e19


@dataclass
class CSCMatrix:
    """Standard CSC format matrix with tile-based pre-quantization support."""
    M: int
    N: int
    nnz: int
    R: int  # tile row size
    C: int  # tile column size
    n_tiles: int  # number of non-zero tiles

    # Standard CSC structure
    colptr: torch.Tensor  # int32 [N+1]
    rowind: torch.Tensor  # int32 [nnz]

    # Values
    A_fp64: torch.Tensor  # float64 [nnz]

    # Optional pre-quantized payloads (filled by `quantize_csc_matrix`)
    # These are per-element but scales are per-tile
    A_fp32_q: Optional[torch.Tensor] = None  # float32 [nnz]
    a_scale_fp32: Optional[torch.Tensor] = None  # float64 [n_br, n_bc] - per-tile scale, 1.0 for empty tiles
    A_bf16_q: Optional[torch.Tensor] = None  # bfloat16 [nnz]
    a_scale_bf16: Optional[torch.Tensor] = None  # float64 [n_br, n_bc] - per-tile scale, 1.0 for empty tiles

    def to(self, device: torch.device | str) -> "CSCMatrix":
        dev = torch.device(device)
        return CSCMatrix(
            M=self.M,
            N=self.N,
            nnz=self.nnz,
            R=self.R,
            C=self.C,
            n_tiles=self.n_tiles,
            colptr=self.colptr.to(device=dev),
            rowind=self.rowind.to(device=dev),
            A_fp64=self.A_fp64.to(device=dev),
            A_fp32_q=None if self.A_fp32_q is None else self.A_fp32_q.to(device=dev),
            a_scale_fp32=None
            if self.a_scale_fp32 is None
            else self.a_scale_fp32.to(device=dev),
            A_bf16_q=None if self.A_bf16_q is None else self.A_bf16_q.to(device=dev),
            a_scale_bf16=None
            if self.a_scale_bf16 is None
            else self.a_scale_bf16.to(device=dev),
        )


def build_csc_from_scipy(
    csr_or_csc, *, R: int = 64, C: int = 64, device: torch.device | str = "cpu"
) -> CSCMatrix:
    """
    Build CSC from scipy sparse matrix with tile mapping.
    
    Args:
        csr_or_csc: scipy sparse matrix (CSR/CSC/COO are all accepted)
        R: tile row size
        C: tile column size
        device: target device
    """
    if hasattr(csr_or_csc, "tocsc"):
        csc_mat = csr_or_csc.tocsc()
    else:
        csc_mat = csr_or_csc.tocoo().tocsc()

    M, N = int(csc_mat.shape[0]), int(csc_mat.shape[1])
    nnz = int(csc_mat.nnz)
    dev = torch.device(device)

    colptr_t = torch.as_tensor(csc_mat.indptr, dtype=torch.int32, device=dev)
    rowind_t = torch.as_tensor(csc_mat.indices, dtype=torch.int32, device=dev)
    A_fp64_t = torch.as_tensor(csc_mat.data, dtype=torch.float64, device=dev)

    # Compute number of tile blocks
    n_br = (M + R - 1) // R
    n_bc = (N + C - 1) // C
    
    # Count unique tiles (for n_tiles field, kept for compatibility)
    if nnz == 0:
        n_tiles = 0
    else:
        colptr_np = csc_mat.indptr
        rowind_np = csc_mat.indices
        cols_np = np.repeat(np.arange(N), np.diff(colptr_np))
        br_np = rowind_np // R
        bc_np = cols_np // C
        tile_idx_np = br_np * n_bc + bc_np
        n_tiles = len(np.unique(tile_idx_np))

    return CSCMatrix(
        M=M,
        N=N,
        nnz=nnz,
        R=R,
        C=C,
        n_tiles=n_tiles,
        colptr=colptr_t,
        rowind=rowind_t,
        A_fp64=A_fp64_t,
    )


def csc_to_dense(csc: CSCMatrix) -> torch.Tensor:
    """
    Reconstruct dense matrix from CSC (for debugging/tests).
    """
    M, N = csc.M, csc.N
    out = torch.zeros((M, N), dtype=torch.float64, device=csc.A_fp64.device)

    for j in range(N):
        start = int(csc.colptr[j].item())
        end = int(csc.colptr[j + 1].item())
        for k in range(start, end):
            i = int(csc.rowind[k].item())
            out[i, j] = csc.A_fp64[k]

    return out


def quantize_csc_matrix(
    csc: CSCMatrix,
    *,
    qmax_fp32: float = DEFAULT_QMAX_F64,
    qmax_bf16: float = DEFAULT_QMAX_F64,
    eps: float = DEFAULT_EPS_F64,
) -> CSCMatrix:
    """
    Pre-quantize all non-zero values into fp32/bf16 quantized-domain values, with per-tile scaling.
    
    For each tile, compute max(|A|) and scale all elements in that tile using the same scale.
    a_scale is stored as a 2D array (n_br, n_bc), with 1.0 for empty tiles.
    """
    A = csc.A_fp64
    n_br = (csc.M + csc.R - 1) // csc.R
    n_bc = (csc.N + csc.C - 1) // csc.C
    
    if A.numel() == 0:
        csc.A_fp32_q = torch.zeros_like(A, dtype=torch.float32)
        csc.a_scale_fp32 = torch.ones((n_br, n_bc), dtype=torch.float64, device=A.device)
        csc.A_bf16_q = torch.zeros_like(A, dtype=torch.bfloat16)
        csc.a_scale_bf16 = torch.ones((n_br, n_bc), dtype=torch.float64, device=A.device)
        return csc

    # Initialize per-tile max absolute value arrays (2D: n_br x n_bc)
    a_max_abs_per_tile = torch.zeros((n_br, n_bc), dtype=torch.float64, device=A.device)
    
    # Compute max absolute value for each tile using PyTorch operations
    rowind_t = csc.rowind
    colptr_t = csc.colptr
    A_abs = torch.abs(A)
    
    # For each column, compute br and bc for all non-zeros
    br_list = []
    bc_list = []
    val_list = []
    for j in range(csc.N):
        start = int(colptr_t[j].item())
        end = int(colptr_t[j + 1].item())
        bc = j // csc.C
        if start < end:
            rows_in_col = rowind_t[start:end]
            brs = rows_in_col // csc.R
            br_list.append(brs)
            bc_list.append(torch.full((end - start,), bc, dtype=torch.int64, device=A.device))
            val_list.append(A_abs[start:end])
    
    if len(br_list) > 0:
        all_br = torch.cat(br_list)
        all_bc = torch.cat(bc_list)
        all_abs = torch.cat(val_list)
        
        # Use scatter_reduce to compute max per tile
        # Flatten tile indices: tile_flat = br * n_bc + bc
        tile_flat = all_br * n_bc + all_bc
        # Use scatter_reduce to compute max (amax = reduce="amax")
        a_max_abs_flat = torch.zeros((n_br * n_bc,), dtype=torch.float64, device=A.device)
        a_max_abs_flat.scatter_reduce_(0, tile_flat, all_abs, reduce="amax", include_self=False)
        # Reshape back to 2D
        a_max_abs_per_tile = a_max_abs_flat.view(n_br, n_bc)

    # fp32 path (mul-safe)
    qmax32 = torch.tensor(float(qmax_fp32), dtype=torch.float64, device=A.device)
    a_scale32_per_tile = qmax32 / (a_max_abs_per_tile + float(eps))
    # For empty tiles (max_abs == 0), scale should be 1.0
    a_scale32_per_tile = torch.where(
        a_max_abs_per_tile > 0,
        a_scale32_per_tile,
        torch.ones_like(a_scale32_per_tile)
    )
    
    # Expand scale to each element based on its tile
    # Reuse the same br and bc computation
    a_scale32_list = []
    for j in range(csc.N):
        start = int(colptr_t[j].item())
        end = int(colptr_t[j + 1].item())
        bc = j // csc.C
        if start < end:
            rows_in_col = rowind_t[start:end]
            brs = rows_in_col // csc.R
            scales = a_scale32_per_tile[brs, bc]
            a_scale32_list.append(scales)
    if len(a_scale32_list) > 0:
        a_scale32 = torch.cat(a_scale32_list)
    else:
        a_scale32 = torch.zeros((0,), dtype=torch.float64, device=A.device)
    
    A32_scaled = A * a_scale32
    A32_clipped = torch.clamp(A32_scaled, min=-float(qmax_fp32), max=float(qmax_fp32))
    A_fp32_q = A32_clipped.to(torch.float32)

    # bf16 path (mul-safe)
    qmaxbf = torch.tensor(float(qmax_bf16), dtype=torch.float64, device=A.device)
    a_scalebf_per_tile = qmaxbf / (a_max_abs_per_tile + float(eps))
    # For empty tiles (max_abs == 0), scale should be 1.0
    a_scalebf_per_tile = torch.where(
        a_max_abs_per_tile > 0,
        a_scalebf_per_tile,
        torch.ones_like(a_scalebf_per_tile)
    )
    
    # Expand scale to each element based on its tile
    a_scalebf_list = []
    for j in range(csc.N):
        start = int(colptr_t[j].item())
        end = int(colptr_t[j + 1].item())
        bc = j // csc.C
        if start < end:
            rows_in_col = rowind_t[start:end]
            brs = rows_in_col // csc.R
            scales = a_scalebf_per_tile[brs, bc]
            a_scalebf_list.append(scales)
    if len(a_scalebf_list) > 0:
        a_scalebf = torch.cat(a_scalebf_list)
    else:
        a_scalebf = torch.zeros((0,), dtype=torch.float64, device=A.device)
    
    Abf_scaled = A * a_scalebf
    Abf_clipped = torch.clamp(Abf_scaled, min=-float(qmax_bf16), max=float(qmax_bf16))
    A_bf16_q = Abf_clipped.to(torch.bfloat16)

    csc.A_fp32_q = A_fp32_q
    csc.a_scale_fp32 = a_scale32_per_tile  # 2D array (n_br, n_bc)
    csc.A_bf16_q = A_bf16_q
    csc.a_scale_bf16 = a_scalebf_per_tile  # 2D array (n_br, n_bc)
    return csc
