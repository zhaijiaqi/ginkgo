"""
BCSC (Block Compressed Sparse Column) utilities + pre-quantization helpers.

This module is intentionally NEW (no modifications to existing files), designed to be
integrated into `env/cg_env.py` later.

It mirrors the quantization rules used by:
`kernels/spmv_kernels.py::make_bsr_spmv_mixed_kernel_warp_reduce`

Key rule (for action>=1):
  scale = max_val / (max_abs + eps)
  q_scaled(v) = cast_lowp( clamp(scale * v, [-max_val, max_val]) )  # stored/represented in lowp
  y += lowp_mul(q_scaled(A), q_scaled(x)) / (a_scale * x_scale)     # accumulate in fp64

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
BF16_MAX_FINITE_F64 = 3.3895313892515355e38


@dataclass
class BCSCMatrix:
    M: int
    N: int
    R: int
    C: int
    n_br: int
    n_bc: int
    nnzb: int

    # Column-compressed block structure
    colptr: torch.Tensor  # int32 [n_bc+1]
    rowind: torch.Tensor  # int32 [nnzb]

    # Tile payloads
    A_fp64: torch.Tensor  # float64 [nnzb, R, C]

    # Optional pre-quantized payloads (filled by `quantize_bcsc_tiles`)
    A_fp32_q: Optional[torch.Tensor] = None  # float32 [nnzb, R, C]
    a_scale_fp32: Optional[torch.Tensor] = None  # float64 [N] - per column scale
    A_bf16_q: Optional[torch.Tensor] = None  # bfloat16 [nnzb, R, C]
    a_scale_bf16: Optional[torch.Tensor] = None  # float64 [N] - per column scale

    def to(self, device: torch.device | str) -> "BCSCMatrix":
        dev = torch.device(device)
        return BCSCMatrix(
            M=self.M,
            N=self.N,
            R=self.R,
            C=self.C,
            n_br=self.n_br,
            n_bc=self.n_bc,
            nnzb=self.nnzb,
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


def _as_coo_arrays_from_scipy(csr_or_coo) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Accepts scipy CSR/COO and returns (row, col, data, shape) as numpy arrays.
    """
    coo = csr_or_coo.tocoo()
    row = np.asarray(coo.row, dtype=np.int64)
    col = np.asarray(coo.col, dtype=np.int64)
    data = np.asarray(coo.data, dtype=np.float64)
    shape = (int(coo.shape[0]), int(coo.shape[1]))
    return row, col, data, shape


def build_bcsc_from_coo(
    row: np.ndarray,
    col: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
    tilesize: int,
    *,
    device: torch.device | str = "cpu",
) -> BCSCMatrix:
    """
    Build BCSC from COO triplets.

    Ordering:
      - primary: block-column bc
      - secondary: block-row br (ascending within each bc)
    """
    M, N = int(shape[0]), int(shape[1])
    R = int(tilesize)
    C = int(tilesize)
    if R <= 0:
        raise ValueError(f"tilesize must be positive, got {R}")

    n_br = (M + R - 1) // R
    n_bc = (N + C - 1) // C

    nnz = int(data.shape[0])
    dev = torch.device(device)

    if nnz == 0:
        colptr_t = torch.zeros((n_bc + 1,), dtype=torch.int32, device=dev)
        rowind_t = torch.zeros((0,), dtype=torch.int32, device=dev)
        A_fp64_t = torch.zeros((0, R, C), dtype=torch.float64, device=dev)
        return BCSCMatrix(
            M=M,
            N=N,
            R=R,
            C=C,
            n_br=n_br,
            n_bc=n_bc,
            nnzb=0,
            colptr=colptr_t,
            rowind=rowind_t,
            A_fp64=A_fp64_t,
        )

    row_i64 = np.asarray(row, dtype=np.int64)
    col_i64 = np.asarray(col, dtype=np.int64)
    data_f64 = np.asarray(data, dtype=np.float64)

    br = (row_i64 // R).astype(np.int64, copy=False)
    bc = (col_i64 // C).astype(np.int64, copy=False)

    # Column-major tile grouping key: sort by (bc, br)
    key = bc * np.int64(n_br) + br
    order = np.argsort(key, kind="stable")

    row_o = row_i64[order]
    col_o = col_i64[order]
    data_o = data_f64[order]
    br_o = br[order]
    bc_o = bc[order]
    key_o = key[order]

    tiles: list[np.ndarray] = []
    rowind: list[int] = []
    colptr = np.zeros((n_bc + 1,), dtype=np.int32)

    cur_tile_count = 0
    cur_bc = 0
    colptr[0] = 0

    i = 0
    while i < nnz:
        bc_i = int(bc_o[i])
        br_i = int(br_o[i])
        key_i = key_o[i]

        # Fill colptr for empty columns up to bc_i
        if bc_i > cur_bc:
            for b in range(cur_bc + 1, bc_i + 1):
                colptr[b] = cur_tile_count
            cur_bc = bc_i

        tile = np.zeros((R, C), dtype=np.float64)
        j = i
        while j < nnz and key_o[j] == key_i:
            rr = int(row_o[j] - np.int64(br_i) * np.int64(R))
            cc = int(col_o[j] - np.int64(bc_i) * np.int64(C))
            # Handle duplicates by summing.
            tile[rr, cc] += float(data_o[j])
            j += 1

        tiles.append(tile)
        rowind.append(br_i)
        cur_tile_count += 1
        i = j

    # Finalize colptr for remaining columns
    for b in range(cur_bc + 1, n_bc + 1):
        colptr[b] = cur_tile_count

    A_fp64_np = np.stack(tiles, axis=0) if tiles else np.zeros((0, R, C), dtype=np.float64)
    rowind_np = np.asarray(rowind, dtype=np.int32)

    colptr_t = torch.as_tensor(colptr, dtype=torch.int32, device=dev)
    rowind_t = torch.as_tensor(rowind_np, dtype=torch.int32, device=dev)
    A_fp64_t = torch.as_tensor(A_fp64_np, dtype=torch.float64, device=dev)

    return BCSCMatrix(
        M=M,
        N=N,
        R=R,
        C=C,
        n_br=n_br,
        n_bc=n_bc,
        nnzb=int(A_fp64_t.shape[0]),
        colptr=colptr_t,
        rowind=rowind_t,
        A_fp64=A_fp64_t,
    )


def build_bcsc_from_scipy_csr(
    csr_mat,
    tilesize: int,
    *,
    device: torch.device | str = "cpu",
) -> BCSCMatrix:
    """
    Build BCSC from scipy sparse matrix (CSR/COO are both accepted).
    """
    row, col, data, shape = _as_coo_arrays_from_scipy(csr_mat)
    return build_bcsc_from_coo(row, col, data, shape, tilesize, device=device)


def bcsc_to_dense(bcsc: BCSCMatrix, *, trim: bool = True) -> torch.Tensor:
    """
    Reconstruct dense matrix from BCSC tiles (for debugging/tests).
    If trim=True, return shape [M, N]. Otherwise return [n_br*R, n_bc*C].
    """
    M, N, R, C = bcsc.M, bcsc.N, bcsc.R, bcsc.C
    full_M = bcsc.n_br * R
    full_N = bcsc.n_bc * C
    out = torch.zeros((full_M, full_N), dtype=torch.float64, device=bcsc.A_fp64.device)

    for bc in range(bcsc.n_bc):
        start = int(bcsc.colptr[bc].item())
        end = int(bcsc.colptr[bc + 1].item())
        if end <= start:
            continue
        col_base = bc * C
        for k in range(start, end):
            br = int(bcsc.rowind[k].item())
            row_base = br * R
            out[row_base : row_base + R, col_base : col_base + C] += bcsc.A_fp64[k]

    return out[:M, :N] if trim else out


def quantize_bcsc_tiles(
    bcsc: BCSCMatrix,
    *,
    qmax_fp32: float = DEFAULT_QMAX_F64,
    qmax_bf16: float = DEFAULT_QMAX_F64,
    eps: float = DEFAULT_EPS_F64,
) -> BCSCMatrix:
    """
    Pre-quantize all non-zero tiles into fp32/bf16 quantized-domain values, and store per column a_scale.
    
    For each column j (0 <= j < N), compute a_scale[j] based on the maximum absolute value
    across all tiles that contain column j.
    """
    A = bcsc.A_fp64
    if A.numel() == 0:
        bcsc.A_fp32_q = torch.zeros_like(A, dtype=torch.float32)
        bcsc.a_scale_fp32 = torch.ones((bcsc.N,), dtype=torch.float64, device=A.device)
        bcsc.A_bf16_q = torch.zeros_like(A, dtype=torch.bfloat16)
        bcsc.a_scale_bf16 = torch.ones((bcsc.N,), dtype=torch.float64, device=A.device)
        return bcsc

    # Compute per column max absolute value
    # For each column j, find max(abs(A[k, :, cc])) for all tiles k and column positions cc
    # where the global column index bc*C + cc == j
    a_scale_fp32_per_col = torch.ones((bcsc.N,), dtype=torch.float64, device=A.device)
    a_scale_bf16_per_col = torch.ones((bcsc.N,), dtype=torch.float64, device=A.device)
    
    qmax32 = torch.tensor(float(qmax_fp32), dtype=torch.float64, device=A.device)
    qmaxbf = torch.tensor(float(qmax_bf16), dtype=torch.float64, device=A.device)
    
    # For each column j, collect all values from tiles that contain this column
    col_max_abs_fp32 = torch.zeros((bcsc.N,), dtype=torch.float64, device=A.device)
    col_max_abs_bf16 = torch.zeros((bcsc.N,), dtype=torch.float64, device=A.device)
    
    for bc in range(bcsc.n_bc):
        start = int(bcsc.colptr[bc].item())
        end = int(bcsc.colptr[bc + 1].item())
        if end <= start:
            continue
        
        col_base = bc * bcsc.C
        tiles_in_col = A[start:end]  # [num_tiles_in_col, R, C]
        
        # For each column position cc in this block-column
        for cc in range(bcsc.C):
            global_col = col_base + cc
            if global_col >= bcsc.N:
                break
            
            # Find max absolute value in this column across all tiles in this block-column
            # tiles_in_col[:, :, cc] gives all values in column cc for all tiles
            col_values = tiles_in_col[:, :, cc]  # [num_tiles_in_col, R]
            col_max = torch.amax(torch.abs(col_values))
            col_max_abs_fp32[global_col] = torch.maximum(col_max_abs_fp32[global_col], col_max)
            col_max_abs_bf16[global_col] = torch.maximum(col_max_abs_bf16[global_col], col_max)
    
    # Compute scales for each column
    a_scale_fp32_per_col = qmax32 / (col_max_abs_fp32 + float(eps))
    a_scale_bf16_per_col = qmaxbf / (col_max_abs_bf16 + float(eps))

    # Quantize all tiles using per-column scales
    A_fp32_q_list = []
    A_bf16_q_list = []
    
    for bc in range(bcsc.n_bc):
        start = int(bcsc.colptr[bc].item())
        end = int(bcsc.colptr[bc + 1].item())
        if end <= start:
            continue
        
        col_base = bc * bcsc.C
        tiles_in_col = A[start:end]  # [num_tiles_in_col, R, C]
        
        # For each tile, apply per-column scaling
        # Tile shape: [R, C], we need to scale each column cc by scale[col_base + cc]
        A32_scaled_list = []
        Abf_scaled_list = []
        
        for tile_idx in range(tiles_in_col.shape[0]):
            tile = tiles_in_col[tile_idx]  # [R, C]
            tile_fp32_scaled = torch.zeros_like(tile, dtype=torch.float64)
            tile_bf16_scaled = torch.zeros_like(tile, dtype=torch.float64)
            
            for cc in range(bcsc.C):
                global_col = col_base + cc
                if global_col >= bcsc.N:
                    break
                
                scale32 = a_scale_fp32_per_col[global_col]
                scalebf = a_scale_bf16_per_col[global_col]
                
                tile_fp32_scaled[:, cc] = tile[:, cc] * scale32
                tile_bf16_scaled[:, cc] = tile[:, cc] * scalebf
            
            A32_clipped = torch.clamp(tile_fp32_scaled, min=-float(qmax_fp32), max=float(qmax_fp32))
            Abf_clipped = torch.clamp(tile_bf16_scaled, min=-float(qmax_bf16), max=float(qmax_bf16))
            A32_scaled_list.append(A32_clipped.to(torch.float32))
            Abf_scaled_list.append(Abf_clipped.to(torch.bfloat16))
        
        if len(A32_scaled_list) > 0:
            A_fp32_q_list.append(torch.stack(A32_scaled_list, dim=0))
            A_bf16_q_list.append(torch.stack(Abf_scaled_list, dim=0))
    
    if len(A_fp32_q_list) > 0:
        A_fp32_q = torch.cat(A_fp32_q_list, dim=0)
        A_bf16_q = torch.cat(A_bf16_q_list, dim=0)
    else:
        A_fp32_q = torch.zeros_like(A, dtype=torch.float32)
        A_bf16_q = torch.zeros_like(A, dtype=torch.bfloat16)

    bcsc.A_fp32_q = A_fp32_q
    bcsc.a_scale_fp32 = a_scale_fp32_per_col
    bcsc.A_bf16_q = A_bf16_q
    bcsc.a_scale_bf16 = a_scale_bf16_per_col
    return bcsc


def _quantize_scaled_value_f64(value_f64: torch.Tensor, scale_f64: torch.Tensor, max_val: float, action: int) -> torch.Tensor:
    """
    Mirror `quantize_scaled_value` in the kernel:
      - action==0: return original value (unscaled)
      - action==1: return float64(float32(clamp(scale*value)))
      - action==2: return float64(bf16(clamp(scale*value)))
    """
    if action == 0:
        return value_f64
    val = scale_f64 * value_f64
    val = torch.clamp(val, min=-float(max_val), max=float(max_val))
    if action == 1:
        return val.to(torch.float32).to(torch.float64)
    if action == 2:
        return val.to(torch.bfloat16).to(torch.float64)
    raise ValueError(f"Unsupported action {action}, expected 0/1/2")


def _lowp_mul_to_f64(a_q_f64: torch.Tensor, x_q_f64: torch.Tensor, action: int) -> torch.Tensor:
    """
    Mirror `lowp_mul_to_f64` in the kernel (elementwise):
      - action==0: f64 * f64
      - action==1: f64(f32(a) * f32(x))
      - action==2: f64(bf16(a) * bf16(x))
    """
    if action == 0:
        return a_q_f64 * x_q_f64
    if action == 1:
        # Lowp rounding in fp32, multiply in fp64 to avoid overflow in quantized domain.
        return (a_q_f64.to(torch.float32).to(torch.float64) * x_q_f64.to(torch.float32).to(torch.float64)).to(torch.float64)
    if action == 2:
        # Lowp rounding in bf16, multiply in fp64 to avoid overflow in quantized domain.
        return (a_q_f64.to(torch.bfloat16).to(torch.float64) * x_q_f64.to(torch.bfloat16).to(torch.float64)).to(torch.float64)
    raise ValueError(f"Unsupported action {action}, expected 0/1/2")


@torch.no_grad()
def spmv_bcsc_mixed_ref_prequant(
    bcsc: BCSCMatrix,
    actions: torch.Tensor,
    x: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS_F64,
    max_val: float = DEFAULT_QMAX_F64,
    trim_to_M: bool = True,
) -> torch.Tensor:
    """
    Reference SpMV using BCSC + pre-quantized A (for action 1/2), matching the kernel math.

    Notes:
      - action is per block-column (bc), length n_bc.
      - action==0 path uses A_fp64 and no x quantization (kernel behavior).
    """
    if actions.dtype != torch.int32:
        actions = actions.to(dtype=torch.int32)
    x = x.to(dtype=torch.float64, device=bcsc.A_fp64.device)

    R, C = bcsc.R, bcsc.C
    out_full = torch.zeros((bcsc.n_br * R,), dtype=torch.float64, device=bcsc.A_fp64.device)

    if bcsc.A_fp32_q is None or bcsc.a_scale_fp32 is None or bcsc.A_bf16_q is None or bcsc.a_scale_bf16 is None:
        raise RuntimeError("bcsc is missing pre-quantized tiles; call quantize_bcsc_tiles(bcsc) first.")

    for bc in range(bcsc.n_bc):
        start = int(bcsc.colptr[bc].item())
        end = int(bcsc.colptr[bc + 1].item())
        if end <= start:
            continue

        action = int(actions[bc].item())
        col_base = bc * C

        # x tile (padded)
        x_tile = torch.zeros((C,), dtype=torch.float64, device=out_full.device)
        valid = min(C, bcsc.N - col_base)
        if valid > 0:
            x_tile[:valid] = x[col_base : col_base + valid]

        if action >= 1:
            x_max_abs = torch.max(torch.abs(x_tile))
            x_scale = torch.tensor(float(max_val), dtype=torch.float64, device=out_full.device) / (
                x_max_abs + float(eps)
            )
            x_q_f64 = _quantize_scaled_value_f64(x_tile, x_scale, max_val, action)  # [C] f64-carrying
        else:
            x_scale = torch.tensor(1.0, dtype=torch.float64, device=out_full.device)
            x_q_f64 = x_tile

        for k in range(start, end):
            br = int(bcsc.rowind[k].item())
            row_base = br * R

            if action == 0:
                # No quantization on A/x; kernel sets scales=1.
                out_full[row_base : row_base + R] += torch.sum(bcsc.A_fp64[k] * x_tile[None, :], dim=1)
                continue

            if action == 1:
                # represent a_q as f64-carrying lowp value
                a_q_f64 = bcsc.A_fp32_q[k].to(torch.float64)
            elif action == 2:
                a_q_f64 = bcsc.A_bf16_q[k].to(torch.float64)
            else:
                raise ValueError(f"Unsupported action {action}, expected 0/1/2")

            # Per-column scaling: each column cc uses scale[col_base + cc]
            prod_q = _lowp_mul_to_f64(a_q_f64, x_q_f64[None, :], action)  # [R,C] f64
            
            # Dequantize per column: for each column cc, divide by (a_scale[col_base + cc] * x_scale)
            acc = torch.zeros((R,), dtype=torch.float64, device=out_full.device)
            for cc in range(C):
                global_col = col_base + cc
                if global_col >= bcsc.N:
                    break
                if action == 1:
                    a_scale_col = bcsc.a_scale_fp32[global_col]
                else:  # action == 2
                    a_scale_col = bcsc.a_scale_bf16[global_col]
                acc += prod_q[:, cc] / (a_scale_col * x_scale)
            
            out_full[row_base : row_base + R] += acc

    return out_full[: bcsc.M] if trim_to_M else out_full


@torch.no_grad()
def spmv_bcsc_mixed_ref_runtime_quant(
    bcsc: BCSCMatrix,
    actions: torch.Tensor,
    x: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS_F64,
    max_val: float = DEFAULT_QMAX_F64,
    trim_to_M: bool = True,
) -> torch.Tensor:
    """
    Slow reference that re-does runtime quantization of A per tile (like current kernel),
    useful for validating that pre-quantization is bitwise-equivalent to the kernel rule.
    """
    if actions.dtype != torch.int32:
        actions = actions.to(dtype=torch.int32)
    x = x.to(dtype=torch.float64, device=bcsc.A_fp64.device)

    R, C = bcsc.R, bcsc.C
    out_full = torch.zeros((bcsc.n_br * R,), dtype=torch.float64, device=bcsc.A_fp64.device)

    for bc in range(bcsc.n_bc):
        start = int(bcsc.colptr[bc].item())
        end = int(bcsc.colptr[bc + 1].item())
        if end <= start:
            continue

        action = int(actions[bc].item())
        col_base = bc * C

        x_tile = torch.zeros((C,), dtype=torch.float64, device=out_full.device)
        valid = min(C, bcsc.N - col_base)
        if valid > 0:
            x_tile[:valid] = x[col_base : col_base + valid]

        if action >= 1:
            x_max_abs = torch.max(torch.abs(x_tile))
            x_scale = torch.tensor(float(max_val), dtype=torch.float64, device=out_full.device) / (
                x_max_abs + float(eps)
            )
            x_q_f64 = _quantize_scaled_value_f64(x_tile, x_scale, max_val, action)
        else:
            x_scale = torch.tensor(1.0, dtype=torch.float64, device=out_full.device)
            x_q_f64 = x_tile

        for k in range(start, end):
            br = int(bcsc.rowind[k].item())
            row_base = br * R

            if action == 0:
                out_full[row_base : row_base + R] += torch.sum(bcsc.A_fp64[k] * x_tile[None, :], dim=1)
                continue

            # Runtime quantization: compute scale per column
            A_tile = bcsc.A_fp64[k]  # [R, C]
            
            # Quantize per column: for each column cc, compute scale based on that column
            a_q_f64_list = []
            a_scale_list = []
            for cc in range(C):
                global_col = col_base + cc
                if global_col >= bcsc.N:
                    break
                
                # Find max absolute value in this column across all tiles in this block-column
                col_start = int(bcsc.colptr[bc].item())
                col_end = int(bcsc.colptr[bc + 1].item())
                if col_end > col_start:
                    col_tiles = bcsc.A_fp64[col_start:col_end]  # [num_tiles, R, C]
                    col_values = col_tiles[:, :, cc]  # [num_tiles, R]
                    a_max_abs_col = torch.max(torch.abs(col_values))
                else:
                    a_max_abs_col = torch.tensor(0.0, dtype=torch.float64, device=out_full.device)
                
                a_scale_col = torch.tensor(float(max_val), dtype=torch.float64, device=out_full.device) / (
                    a_max_abs_col + float(eps)
                )
                a_scale_list.append(a_scale_col)
                
                # Quantize this column of the tile
                col_tile = A_tile[:, cc]  # [R]
                col_q = _quantize_scaled_value_f64(col_tile, a_scale_col, max_val, action)  # [R] f64-carrying
                a_q_f64_list.append(col_q)
            
            # Reconstruct quantized tile [R, C]
            if len(a_q_f64_list) > 0:
                a_q_f64 = torch.stack(a_q_f64_list, dim=1)  # [R, C]
            else:
                a_q_f64 = A_tile
            
            prod_q = _lowp_mul_to_f64(a_q_f64, x_q_f64[None, :], action)  # [R,C] f64
            
            # Dequantize per column
            acc = torch.zeros((R,), dtype=torch.float64, device=out_full.device)
            for idx, cc in enumerate(range(min(C, bcsc.N - col_base))):
                a_scale_col = a_scale_list[idx]
                acc += prod_q[:, cc] / (a_scale_col * x_scale)
            
            out_full[row_base : row_base + R] += acc

    return out_full[: bcsc.M] if trim_to_M else out_full


