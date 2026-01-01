"""
CSR (Compressed Sparse Row) utilities + pre-quantization helpers.

Standard CSR format with tile-based quantization:
  rowptr : (M+1,) int32   # row pointer
  colind : (nnz,) int32   # column indices
  values : (nnz,) float64 # non-zero values

Quantization rules:
  - Actions are per block-column: actions: (n_bc,) int32 where n_bc = (N+C-1)//C
    Every C columns share the same action (block-column bc = col // C)
  - Elements in the same row may use different actions if they belong to different block-columns
  - But each element may belong to different tiles, so a_scale is per-tile
  - a_scale: (n_br, n_bc) float64  # per-tile scale stored as 2D array, indexed by br = row // R, bc = col // C
    Empty tiles have scale 1.0 (zero padding)

Key rule (for action>=1):
  scale = max_val / (max_abs + eps)  # computed per tile for A, per row for x
  q_scaled(v) = cast_lowp( clamp(scale * v, [-max_val, max_val]) )
  y += lowp_mul(q_scaled(A), q_scaled(x)) / (a_scale * x_scale)

For action==0:
  no quantization; scales are 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


DEFAULT_EPS_F64 = 1e-12
# Mul-safe qmax ~= sqrt(max_finite_lowp) so lowp_mul(a_q, x_q) won't overflow.
DEFAULT_QMAX_F64 = 1.8405e19


@dataclass
class CSRMatrix:
    """Standard CSR format matrix with tile-based pre-quantization support."""
    M: int
    N: int
    nnz: int
    R: int  # tile row size
    C: int  # tile column size
    n_tiles: int  # number of non-zero tiles (kept for compatibility, but not used for indexing)

    # Standard CSR structure
    rowptr: torch.Tensor  # int32 [M+1]
    colind: torch.Tensor  # int32 [nnz]

    # Values
    A_fp64: torch.Tensor  # float64 [nnz]

    # Optional pre-quantized payloads (filled by `quantize_csr_matrix`)
    # These are per-element but scales are per-tile
    A_fp32_q: Optional[torch.Tensor] = None  # float32 [nnz]
    a_scale_fp32: Optional[torch.Tensor] = None  # float64 [n_br, n_bc] - per-tile scale, 1.0 for empty tiles
    A_bf16_q: Optional[torch.Tensor] = None  # bfloat16 [nnz]
    a_scale_bf16: Optional[torch.Tensor] = None  # float64 [n_br, n_bc] - per-tile scale, 1.0 for empty tiles

    def to(self, device: torch.device | str) -> "CSRMatrix":
        dev = torch.device(device)
        return CSRMatrix(
            M=self.M,
            N=self.N,
            nnz=self.nnz,
            R=self.R,
            C=self.C,
            n_tiles=self.n_tiles,
            rowptr=self.rowptr.to(device=dev),
            colind=self.colind.to(device=dev),
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


def build_csr_from_scipy(
    csr_mat, *, R: int = 64, C: int = 64, device: torch.device | str = "cpu"
) -> CSRMatrix:
    """
    Build CSR from scipy sparse matrix with tile mapping.
    
    Args:
        csr_mat: scipy sparse matrix (CSR/COO are both accepted)
        R: tile row size
        C: tile column size
        device: target device
    """
    if hasattr(csr_mat, "tocsr"):
        csr_mat = csr_mat.tocsr()
    else:
        csr_mat = csr_mat.tocoo().tocsr()

    M, N = int(csr_mat.shape[0]), int(csr_mat.shape[1])
    nnz = int(csr_mat.nnz)
    dev = torch.device(device)

    if nnz == 0:
        rowptr_t = torch.zeros((M + 1,), dtype=torch.int32, device=dev)
        colind_t = torch.zeros((0,), dtype=torch.int32, device=dev)
        A_fp64_t = torch.zeros((0,), dtype=torch.float64, device=dev)
        return CSRMatrix(
            M=M,
            N=N,
            nnz=0,
            R=R,
            C=C,
            n_tiles=0,
            rowptr=rowptr_t,
            colind=colind_t,
            A_fp64=A_fp64_t,
        )

    rowptr_t = torch.as_tensor(csr_mat.indptr, dtype=torch.int32, device=dev)
    colind_t = torch.as_tensor(csr_mat.indices, dtype=torch.int32, device=dev)
    A_fp64_t = torch.as_tensor(csr_mat.data, dtype=torch.float64, device=dev)

    # Compute tile grid dimensions
    # Tile (br, bc) contains rows [br*R, (br+1)*R) and cols [bc*C, (bc+1)*C)
    n_br = (M + R - 1) // R
    n_bc = (N + C - 1) // C

    # Count unique tiles for n_tiles (kept for compatibility)
    rowptr_np = csr_mat.indptr
    colind_np = csr_mat.indices
    rows_np = np.repeat(np.arange(M), np.diff(rowptr_np))
    br_np = rows_np // R
    bc_np = colind_np // C
    tile_idx_np = br_np * n_bc + bc_np
    unique_tiles = np.unique(tile_idx_np)
    n_tiles = len(unique_tiles)

    return CSRMatrix(
        M=M,
        N=N,
        nnz=nnz,
        R=R,
        C=C,
        n_tiles=n_tiles,
        rowptr=rowptr_t,
        colind=colind_t,
        A_fp64=A_fp64_t,
    )


def csr_to_dense(csr: CSRMatrix) -> torch.Tensor:
    """
    Reconstruct dense matrix from CSR (for debugging/tests).
    """
    M, N = csr.M, csr.N
    out = torch.zeros((M, N), dtype=torch.float64, device=csr.A_fp64.device)

    for i in range(M):
        start = int(csr.rowptr[i].item())
        end = int(csr.rowptr[i + 1].item())
        for k in range(start, end):
            j = int(csr.colind[k].item())
            out[i, j] = csr.A_fp64[k]

    return out


def quantize_csr_matrix(
    csr: CSRMatrix,
    *,
    qmax_fp32: float = DEFAULT_QMAX_F64,
    qmax_bf16: float = DEFAULT_QMAX_F64,
    eps: float = DEFAULT_EPS_F64,
) -> CSRMatrix:
    """
    Pre-quantize all non-zero values into fp32/bf16 quantized-domain values, with per-tile scaling.
    
    For each tile, compute max(|A|) and scale all elements in that tile using the same scale.
    a_scale is stored as a 2D array (n_br, n_bc), with 1.0 for empty tiles.
    """
    A = csr.A_fp64
    n_br = (csr.M + csr.R - 1) // csr.R
    n_bc = (csr.N + csr.C - 1) // csr.C
    
    if A.numel() == 0:
        csr.A_fp32_q = torch.zeros_like(A, dtype=torch.float32)
        csr.a_scale_fp32 = torch.ones((n_br, n_bc), dtype=torch.float64, device=A.device)
        csr.A_bf16_q = torch.zeros_like(A, dtype=torch.bfloat16)
        csr.a_scale_bf16 = torch.ones((n_br, n_bc), dtype=torch.float64, device=A.device)
        return csr

    # Initialize per-tile max absolute value arrays (2D: n_br x n_bc)
    a_max_abs_per_tile = torch.zeros((n_br, n_bc), dtype=torch.float64, device=A.device)
    
    # Compute max absolute value for each tile using PyTorch operations
    rowptr_t = csr.rowptr
    colind_t = csr.colind
    A_abs = torch.abs(A)
    
    # For each row, compute br and bc for all non-zeros
    br_list = []
    bc_list = []
    val_list = []
    for i in range(csr.M):
        start = int(rowptr_t[i].item())
        end = int(rowptr_t[i + 1].item())
        br = i // csr.R
        if start < end:
            cols_in_row = colind_t[start:end]
            bcs = cols_in_row // csr.C
            br_list.append(torch.full((end - start,), br, dtype=torch.int64, device=A.device))
            bc_list.append(bcs.to(torch.int64))
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
    for i in range(csr.M):
        start = int(rowptr_t[i].item())
        end = int(rowptr_t[i + 1].item())
        br = i // csr.R
        if start < end:
            cols_in_row = colind_t[start:end]
            bcs = cols_in_row // csr.C
            scales = a_scale32_per_tile[br, bcs]
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
    for i in range(csr.M):
        start = int(rowptr_t[i].item())
        end = int(rowptr_t[i + 1].item())
        br = i // csr.R
        if start < end:
            cols_in_row = colind_t[start:end]
            bcs = cols_in_row // csr.C
            scales = a_scalebf_per_tile[br, bcs]
            a_scalebf_list.append(scales)
    if len(a_scalebf_list) > 0:
        a_scalebf = torch.cat(a_scalebf_list)
    else:
        a_scalebf = torch.zeros((0,), dtype=torch.float64, device=A.device)
    
    Abf_scaled = A * a_scalebf
    Abf_clipped = torch.clamp(Abf_scaled, min=-float(qmax_bf16), max=float(qmax_bf16))
    A_bf16_q = Abf_clipped.to(torch.bfloat16)

    csr.A_fp32_q = A_fp32_q
    csr.a_scale_fp32 = a_scale32_per_tile  # 2D array (n_br, n_bc)
    csr.A_bf16_q = A_bf16_q
    csr.a_scale_bf16 = a_scalebf_per_tile  # 2D array (n_br, n_bc)
    return csr
