"""
TileLang kernels for CSC (Compressed Sparse Column) SpMV.

Standard CSC format with tile-based quantization:
  colptr : (N+1,) int32   # column pointer
  rowind : (nnz,) int32   # row indices
  values : (nnz,) float64 # non-zero values
  tile_map: (nnz,) int32  # tile index for each non-zero

Tile-based quantization:
  - actions: (n_tiles,) int32  # per-tile action (0=fp64, 1=fp32, 2=bf16)
  - a_scale: (n_tiles,) float64  # per-tile scale

NOTE:
CSC iterates columns. Multiple columns update the same row, so we use atomic adds.
"""

# pyright: reportInvalidTypeForm=false

import numpy as np
import tilelang
import tilelang.language as T
import torch

from .csc_prequant import CSCMatrix


@tilelang.jit(target="cuda")
def make_csc_spmv_mixed_prequant_kernel_warp_reduce(
    M: int,
    N: int,
    nnz: int,
    n_tiles: int,
    R: int,
    C: int,
    n_bc: int,
    n_br: int,
    WARPS_PER_BLOCK: int = 16,
    THREADS_PER_WARP: int = 32,
):
    """
    CSC mixed SpMV with PRE-QUANTIZED A values (tile-based).

    CSC representation:
      colptr : (N+1,) int32   # column pointer
      rowind : (nnz,) int32   # row indices

    Values:
      data_fp64    : (nnz,) float64   # action==0
      data_fp32_q  : (nnz,) float32   # action==1 (quantized-domain values)
      a_scale_fp32 : (n_tiles,) float64
      data_bf16_q  : (nnz,) bfloat16  # action==2 (quantized-domain values)
      a_scale_bf16 : (n_tiles,) float64

    Tile mapping:
      a_scale is stored as 2D array (n_br, n_bc), indexed by br = row_idx // R, bc = col // C

    Vector:
      actions : (n_bc,) int32  # 0=fp64, 1=fp32, 2=bf16 (per block-column, n_bc = (N+C-1)//C)
      x       : (N,)    float64
      y       : (M,)   float64  # atomic accumulated
    
    Note:
      - actions are per block-column (every C columns share the same action)
      - All elements in the same column share the same action
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
        a_scale_fp32: T.Tensor((n_br, n_bc), "float64"),  # type: ignore
        data_bf16_q: T.Tensor((nnz,), "bfloat16"),  # type: ignore
        a_scale_bf16: T.Tensor((n_br, n_bc), "float64"),  # type: ignore
        actions: T.Tensor((n_bc,), "int32"),  # type: ignore
        colptr: T.Tensor((N + 1,), "int32"),  # type: ignore
        rowind: T.Tensor((nnz,), "int32"),  # type: ignore
        x: T.Tensor((N,), "float64"),  # type: ignore
        y: T.Tensor((M,), "float64"),  # type: ignore
    ):
        """
        kernel 的核心设计：warp-per-column，每个 warp 处理一个 column。
        warp 内的 32 个 lane 是并行的，每个 lane 处理一个 non-zero。
        一列所有 non-zero 共享 x[col], 共享 action
        WARPS_PER_BLOCK 调整每个 block 处理的 column 数量
        
        """
        
        with T.Kernel(T.ceildiv(N, WARPS_PER_BLOCK), threads=THREADS_PER_WARP * WARPS_PER_BLOCK) as bx:
            # gridSize = T.ceildiv(N, WARPS_PER_BLOCK)  一共 N/WARPS_PER_BLOCK 个 block
            # blockSize = THREADS_PER_WARP * WARPS_PER_BLOCK 每个 block 处理 THREADS_PER_WARP 个 column
            # 一个 warp 处理一个 column，一个 thread 处理一个 non-zero
            warp = T.get_warp_idx_sync() # 0 to WARPS_PER_BLOCK-1, warp index within block
            lane = T.get_lane_idx() # 0 to THREADS_PER_WARP-1, thread index within warp
            col = bx * WARPS_PER_BLOCK + warp
            # Per-warp scratch
            action_local = T.alloc_local((1,), "int32")
            x_lane_f64 = T.alloc_local((1,), "float64")
            x_lane_f32 = T.alloc_local((1,), "float32")
            x_lane_bf16 = T.alloc_local((1,), "bfloat16")

            if col < N:
                start = colptr[col]
                end = colptr[col + 1]

                # Get action for this column (per block-column)
                bc = col // C  # block-column index
                action_local[0] = actions[col//C] # All elements in this column share the same action

                # Load x[col] - all threads read the same value (coalesced access is fine)
                # Since all threads in warp process the same column, they all need x[col]
                x_col_raw = x[col]

                # Compute x_scale for this column's action
                qmax = get_qmax(action_local[0])
                x_scale = T.float64(1.0)
                if action_local[0] >= 1:
                    x_max_abs = T.abs(x_col_raw)
                    x_scale = qmax / (x_max_abs + T.float64(1e-12))

                # Quantize x[col] once based on column's action
                if action_local[0] == 0:
                    x_lane_f64[0] = x_col_raw
                elif action_local[0] == 1:
                    x_q_f64 = quantize_scaled_value(x_col_raw, x_scale, qmax, action_local[0])
                    x_lane_f32[0] = T.Cast("float32", x_q_f64)
                else:  # action == 2
                    x_q_f64 = quantize_scaled_value(x_col_raw, x_scale, qmax, action_local[0])
                    x_lane_bf16[0] = T.Cast("bfloat16", x_q_f64)

                # Iterate non-zeros in this column
                # Distribute work across warp lanes to avoid redundant computation
                num_nnz = end - start
                for k_idx in T.serial((num_nnz + (THREADS_PER_WARP-1)) // THREADS_PER_WARP):
                    k = start + k_idx * THREADS_PER_WARP + lane
                    if k < end:
                        row_idx = rowind[k]
                        # Calculate tile indices directly from row_idx and col
                        br = row_idx // R
                        bc = col // C
                        # action is the same for all elements in this column

                        # Select per-tile a_scale (each element may have different tile, so different a_scale)
                        a_scale = T.if_then_else(
                            action_local[0] == 1,
                            a_scale_fp32[br, bc],
                            T.if_then_else(action_local[0] == 2, a_scale_bf16[br, bc], T.float64(1.0)),
                        )
                        inv_scale = T.if_then_else(
                            action_local[0] == 0,
                            T.float64(1.0),
                            T.float64(1.0) / (a_scale * x_scale),
                        )

                        # Load A value and compute product based on action
                        if action_local[0] == 0:
                            # fp64: use original x value
                            a_val = data_fp64[k]
                            prod = a_val * x_lane_f64[0]
                            if row_idx < M:
                                T.atomic_add(y[row_idx], prod)
                        elif action_local[0] == 1:
                            # fp32: use fp32 quantized x
                            a_f32 = data_fp32_q[k]
                            prod_f32 = a_f32 * x_lane_f32[0]
                            prod_f64 = T.Cast("float64", prod_f32)
                            if row_idx < M:
                                T.atomic_add(y[row_idx], prod_f64 * inv_scale)
                        else:  # action == 2
                            # bf16: use bf16 quantized x
                            a_bf16 = data_bf16_q[k]
                            prod_bf16 = a_bf16 * x_lane_bf16[0]
                            prod_f64 = T.Cast("float64", prod_bf16)
                            if row_idx < M:
                                T.atomic_add(y[row_idx], prod_f64 * inv_scale)

    return main


def csc_spmv_mixed_prequant(
    csc: CSCMatrix,
    actions,
    x,
    *,
    device: str = "cuda",
    return_torch: bool = True,
):
    """
    Python wrapper: run CSC mixed prequant kernel.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("csc_spmv_mixed_prequant currently supports device='cuda' only.")

    csc_d = csc
    if csc_d.A_fp64.device.type != dev.type:
        raise ValueError(
            f"csc_spmv_mixed_prequant expects csc tensors on {dev.type}, but got {csc_d.A_fp64.device}. "
            "Move csc once during preprocessing via csc = csc.to('cuda')."
        )
    if (
        csc_d.A_fp32_q is None
        or csc_d.a_scale_fp32 is None
        or csc_d.A_bf16_q is None
        or csc_d.a_scale_bf16 is None
    ):
        raise RuntimeError("csc is missing pre-quantized values; call quantize_csc_matrix(csc) first.")

    actions_t = actions if torch.is_tensor(actions) else torch.as_tensor(actions, dtype=torch.int32)
    actions_t = actions_t.to(device=dev, dtype=torch.int32)
    n_bc = (int(csc_d.N) + int(csc_d.C) - 1) // int(csc_d.C)
    if int(actions_t.numel()) != n_bc:
        raise ValueError(f"actions length mismatch: got {int(actions_t.numel())}, expected n_bc={n_bc} (N={int(csc_d.N)}, C={int(csc_d.C)})")

    x_t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float64)
    x_t = x_t.to(device=dev, dtype=torch.float64)
    N = int(x_t.shape[0])
    if N != int(csc_d.N):
        raise ValueError(f"x length mismatch: got N={N}, expected {int(csc_d.N)}")

    y_t = torch.zeros((int(csc_d.M),), dtype=torch.float64, device=dev)

    n_bc = (int(csc_d.N) + int(csc_d.C) - 1) // int(csc_d.C)
    n_br = (int(csc_d.M) + int(csc_d.R) - 1) // int(csc_d.R)
    kernel = make_csc_spmv_mixed_prequant_kernel_warp_reduce(
        int(csc_d.M),
        int(csc_d.N),
        int(csc_d.nnz),
        int(csc_d.n_tiles),
        int(csc_d.R),
        int(csc_d.C),
        n_bc,
        n_br,
    )
    kernel(
        csc_d.A_fp64,
        csc_d.A_fp32_q,
        csc_d.a_scale_fp32,
        csc_d.A_bf16_q,
        csc_d.a_scale_bf16,
        actions_t,
        csc_d.colptr,
        csc_d.rowind,
        x_t,
        y_t,
    )
    return y_t if return_torch else y_t.detach().cpu().numpy()

