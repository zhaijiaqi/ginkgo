"""
TileLang kernels for CSR (Compressed Sparse Row) SpMV.

Standard CSR format with tile-based quantization:
  rowptr : (M+1,) int32   # row pointer
  colind : (nnz,) int32   # column indices
  values : (nnz,) float64 # non-zero values
  tile_map: (nnz,) int32  # tile index for each non-zero

Tile-based quantization:
  - actions: (n_br,) int32  # per block-row action (0=fp64, 1=fp32, 2=bf16)
  - a_scale: (n_tiles,) float64  # per-tile scale

NOTE:
CSR iterates rows. Each row is computed independently, so no atomic adds needed.
"""

# pyright: reportInvalidTypeForm=false

import numpy as np
import tilelang
import tilelang.language as T
import torch

from .csr_prequant import CSRMatrix


@tilelang.jit(target="cuda")
def make_csr_spmv_mixed_prequant_kernel_warp_reduce(
    M: int,
    N: int,
    nnz: int,
    n_tiles: int,
    R: int,
    C: int,
    n_bc: int,
    WARPS_PER_BLOCK: int = 2,
):
    """
    CSR mixed SpMV with PRE-QUANTIZED A values (tile-based).

    CSR representation:
      rowptr : (M+1,) int32   # row pointer
      colind : (nnz,) int32    # column indices

    Values:
      data_fp64    : (nnz,) float64   # action==0
      data_fp32_q  : (nnz,) float32   # action==1 (quantized-domain values)
      a_scale_fp32 : (n_tiles,) float64
      data_bf16_q  : (nnz,) bfloat16  # action==2 (quantized-domain values)
      a_scale_bf16 : (n_tiles,) float64

    Tile mapping:
      tile_map : (nnz,) int32  # tile index for each non-zero

    Vector:
      actions : (n_bc,) int32  # 0=fp64, 1=fp32, 2=bf16 (per block-column, n_bc = (N+C-1)//C)
      x       : (N,)    float64
      y       : (M,)   float64
    
    Note:
      - actions are per block-column (every C columns share the same action)
      - Elements in the same row may use different actions if they belong to different block-columns
      - But each element may belong to different tiles, so a_scale is per-tile
    """

    # Mul-safe quantization constant
    def get_qmax(action):
        return T.if_then_else(
            action == 0,
            T.float64(1.0),
            T.float64(1.8405e19),
        )

    def quantize_scaled_value(value, scale, qmax, action):
        val = scale * value
        clipped_val = T.max(T.min(val, qmax), -qmax)
        if_then_else = T.if_then_else
        return if_then_else(
            action == 0,
            value,
            if_then_else(
                action == 1,
                T.Cast("float64", T.Cast("float32", clipped_val)),
                if_then_else(
                    action == 2,
                    T.Cast(
                        "float64",
                        T.Cast(
                            "bfloat16",
                            T.max(
                                T.min(clipped_val, T.float64(1.8405e19)),
                                -T.float64(1.8405e19),
                            ),
                        ),
                    ),
                    T.float64(0),
                ),
            ),
        )

    # With mul-safe quantization, true lowp multiply is safe.
    def lowp_mul_to_f64(a_q_f64, x_q_f64, action):
        if_then_else = T.if_then_else
        return if_then_else(
            action == 0,
            a_q_f64 * x_q_f64,
            if_then_else(
                action == 1,
                T.Cast("float64", T.Cast("float32", a_q_f64) * T.Cast("float32", x_q_f64)),
                if_then_else(
                    action == 2,
                    T.Cast("float64", T.Cast("bfloat16", a_q_f64) * T.Cast("bfloat16", x_q_f64)),
                    T.float64(0),
                ),
            ),
        )

    @T.prim_func
    def main(
        data_fp64: T.Tensor((nnz,), "float64"),  # type: ignore
        data_fp32_q: T.Tensor((nnz,), "float32"),  # type: ignore
        a_scale_fp32: T.Tensor((n_tiles,), "float64"),  # type: ignore
        data_bf16_q: T.Tensor((nnz,), "bfloat16"),  # type: ignore
        a_scale_bf16: T.Tensor((n_tiles,), "float64"),  # type: ignore
        actions: T.Tensor((n_bc,), "int32"),  # type: ignore
        rowptr: T.Tensor((M + 1,), "int32"),  # type: ignore
        colind: T.Tensor((nnz,), "int32"),  # type: ignore
        tile_map: T.Tensor((nnz,), "int32"),  # type: ignore
        x: T.Tensor((N,), "float64"),  # type: ignore
        y: T.Tensor((M,), "float64"),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(M, WARPS_PER_BLOCK), threads=32 * WARPS_PER_BLOCK) as bx:
            warp = T.get_warp_idx_sync()
            lane = T.get_lane_idx()
            tid = T.get_thread_binding(0)  # Thread index within block (0 to 32*WARPS_PER_BLOCK-1)
            row = bx * WARPS_PER_BLOCK + warp
            mask = T.tvm_warp_activemask()

            # Load actions into shared memory (only first warp in block does it)
            actions_smem = T.alloc_shared((n_bc,), "int32")
            # Warp reduction shared memory (each warp has its own space)
            warp_reduce_smem = T.alloc_shared((WARPS_PER_BLOCK, 32), "float64")
            if warp == 0:
                for i in T.serial((n_bc + 31) // 32):
                    idx = i * 32 + lane
                    if idx < n_bc:
                        actions_smem[idx] = actions[idx]
            T.sync_threads()

            # Per-warp scratch
            x_lane_f64 = T.alloc_local((1,), "float64")
            x_lane_f32 = T.alloc_local((1,), "float32")
            x_lane_bf16 = T.alloc_local((1,), "bfloat16")
            acc = T.alloc_local((1,), "float64")
            acc[0] = T.float64(0)  # Initialize accumulator for all threads

            if row < M:
                start = rowptr[row]
                end = rowptr[row + 1]

                # Iterate non-zeros in this row
                # For now, let's use a simpler approach: lane 0 processes all elements
                # TODO: optimize with warp-level parallelism later
                if lane == 0:
                    for k in T.serial(start, end):
                        col_idx = colind[k]
                        tile_idx = tile_map[k]
                        
                        # Get action for this column (per block-column)
                        bc = col_idx // C  # block-column index
                        action = actions_smem[bc]  # Each column has its own action

                        # Compute qmax and x_scale for this specific action
                        qmax = get_qmax(action)
                        x_scale = T.float64(1.0)
                        if action >= 1:
                            # Online x quantization: find max abs in x for this specific column
                            x_max_abs = T.abs(x[col_idx])
                            x_scale = qmax / (x_max_abs + T.float64(1e-12))

                        # Select per-tile a_scale (each element may have different tile, so different a_scale)
                        a_scale = T.if_then_else(
                            action == 1,
                            a_scale_fp32[tile_idx],
                            T.if_then_else(action == 2, a_scale_bf16[tile_idx], T.float64(1.0)),
                        )
                        inv_scale = T.if_then_else(
                            action == 0,
                            T.float64(1.0),
                            T.float64(1.0) / (a_scale * x_scale),
                        )

                        # Load x element and compute product based on action
                        if col_idx < N:
                            if action == 0:
                                x_lane_f64[0] = x[col_idx]
                                a_val = data_fp64[k]
                                prod = a_val * x_lane_f64[0]
                                acc[0] += prod
                            elif action == 1:
                                x_val_f64 = quantize_scaled_value(x[col_idx], x_scale, qmax, action)
                                x_lane_f32[0] = T.Cast("float32", x_val_f64)
                                a_f32 = data_fp32_q[k]
                                prod_f32 = a_f32 * x_lane_f32[0]
                                prod_f64 = T.Cast("float64", prod_f32)
                                acc[0] += prod_f64 * inv_scale
                            else:  # action == 2
                                x_val_f64 = quantize_scaled_value(x[col_idx], x_scale, qmax, action)
                                x_lane_bf16[0] = T.Cast("bfloat16", x_val_f64)
                                a_bf16 = data_bf16_q[k]
                                prod_bf16 = a_bf16 * x_lane_bf16[0]
                                prod_f64 = T.Cast("float64", prod_bf16)
                                acc[0] += prod_f64 * inv_scale
                        else:
                            # col_idx >= N, skip
                            pass
                
                # Write result directly (no warp reduction needed for now)
                if lane == 0:
                    y[row] = acc[0]

    return main


def csr_spmv_mixed_prequant(
    csr: CSRMatrix,
    actions,
    x,
    *,
    device: str = "cuda",
    return_torch: bool = True,
):
    """
    Python wrapper: run CSR mixed prequant kernel.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("csr_spmv_mixed_prequant currently supports device='cuda' only.")

    csr_d = csr
    if csr_d.A_fp64.device.type != dev.type:
        raise ValueError(
            f"csr_spmv_mixed_prequant expects csr tensors on {dev.type}, but got {csr_d.A_fp64.device}. "
            "Move csr once during preprocessing via csr = csr.to('cuda')."
        )
    if (
        csr_d.A_fp32_q is None
        or csr_d.a_scale_fp32 is None
        or csr_d.A_bf16_q is None
        or csr_d.a_scale_bf16 is None
    ):
        raise RuntimeError("csr is missing pre-quantized values; call quantize_csr_matrix(csr) first.")

    actions_t = actions if torch.is_tensor(actions) else torch.as_tensor(actions, dtype=torch.int32)
    actions_t = actions_t.to(device=dev, dtype=torch.int32)
    n_bc = (int(csr_d.N) + int(csr_d.C) - 1) // int(csr_d.C)
    if int(actions_t.numel()) != n_bc:
        raise ValueError(f"actions length mismatch: got {int(actions_t.numel())}, expected n_bc={n_bc} (N={int(csr_d.N)}, C={int(csr_d.C)})")

    x_t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float64)
    x_t = x_t.to(device=dev, dtype=torch.float64)
    N = int(x_t.shape[0])
    if N != int(csr_d.N):
        raise ValueError(f"x length mismatch: got N={N}, expected {int(csr_d.N)}")

    y_t = torch.zeros((int(csr_d.M),), dtype=torch.float64, device=dev)

    n_bc = (int(csr_d.N) + int(csr_d.C) - 1) // int(csr_d.C)
    kernel = make_csr_spmv_mixed_prequant_kernel_warp_reduce(
        int(csr_d.M),
        int(csr_d.N),
        int(csr_d.nnz),
        int(csr_d.n_tiles),
        int(csr_d.R),
        int(csr_d.C),
        n_bc,
    )
    kernel(
        csr_d.A_fp64,
        csr_d.A_fp32_q,
        csr_d.a_scale_fp32,
        csr_d.A_bf16_q,
        csr_d.a_scale_bf16,
        actions_t,
        csr_d.rowptr,
        csr_d.colind,
        csr_d.tile_map,
        x_t,
        y_t,
    )
    return y_t if return_torch else y_t.detach().cpu().numpy()

