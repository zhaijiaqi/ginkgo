"""
Safe TensorCore (fp16 MMA) variant for BCSC prequant SpMV.

Root cause of Stage-E non-finite:
- Quantized-domain values are clamped to qmax ~= 1.8405e19 (mul-safe),
  but casting them to fp16 for MMA overflows fp16 (max finite 65504) -> Inf/NaN.

Fix strategy:
- Keep using fp16 MMA (widely supported) with fp32 accumulation.
- Scale down BOTH A_q and x_q *inside* MMA by a large factor S so that:
    |A_q / S| <= ~6e4 and |x_q / S| <= ~6e4
  Then scale the MMA output back by S^2 before dequantization.

This preserves the intended math up to fp16 rounding while staying finite.
"""

# pyright: reportInvalidTypeForm=false

import tilelang
import tilelang.language as T
import torch

from .bcsc_prequant import BCSCMatrix
from tilelang.intrinsics.mma_macro_generator import TensorCoreIntrinEmitter


@tilelang.jit(target="cuda")
def make_bcsc_spmv_mixed_prequant_kernel_tensorcore_bf16_safe(
    n_br: int,
    n_bc: int,
    nnzb: int,
    R: int,
    C: int,
    N: int,
    *,
    tc_scale_down: float = 3.0675e14,
):
    """
    TensorCore variant for action==2 (bf16 prequantized A):
      - Uses fp16 MMA (A/B in bf16, accumulate in fp32)
      - Scales down A_q and x_q inside MMA by tc_scale_down
      - Scales up the MMA output by tc_scale_down^2 before dequantization
    """

    def get_qmax(action):
        return T.if_then_else(
            action == 0,
            T.float64(1.0),
            T.float64(1.8405e19),
        )

    def quantize_scaled_value(value, scale, qmax, action):
        val = scale * value
        clipped_val = T.max(T.min(val, qmax), -qmax)
        return T.if_then_else(
            action == 0,
            value,
            T.if_then_else(
                action == 1,
                T.Cast("float64", T.Cast("float32", clipped_val)),
                T.if_then_else(
                    action == 2,
                    T.Cast(
                        "float64",
                        T.Cast("bfloat16", clipped_val),
                    ),
                    T.float64(0),
                ),
            ),
        )

    mma = TensorCoreIntrinEmitter(
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        accum_dtype="float64",
        a_transposed=False,
        b_transposed=True,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=16,
        chunk=16,
        reduce_k=1,
    )

    @T.prim_func
    def main(
        data_fp64: T.Tensor((nnzb, R, C), "float64"),  # type: ignore
        data_fp32_q: T.Tensor((nnzb, R, C), "float32"),  # type: ignore
        a_scale_fp32: T.Tensor((N,), "float64"),  # type: ignore
        data_bf16_q: T.Tensor((nnzb, R, C), "bfloat16"),  # type: ignore
        a_scale_bf16: T.Tensor((N,), "float64"),  # type: ignore
        actions: T.Tensor((n_bc,), "int32"),  # type: ignore
        colptr: T.Tensor((n_bc + 1,), "int32"),  # type: ignore
        rowind: T.Tensor((nnzb,), "int32"),  # type: ignore
        x: T.Tensor((N,), "float64"),  # type: ignore
        y: T.Tensor((n_br * R,), "float64"),  # type: ignore
    ):
        with T.Kernel(n_bc, threads=32) as bc:
            lane = T.get_lane_idx()
            action = actions[bc]
            x_base = bc * C

            qmax = get_qmax(action)
            x_scale = T.float64(1.0)
            if action >= 1:
                x_max_abs = T.float64(0.0)
                for cc in T.serial(C):
                    col = x_base + cc
                    if col < N:
                        xv = x[col]
                        x_max_abs = T.max(x_max_abs, T.abs(xv))
                x_scale = qmax / (x_max_abs + T.float64(1e-12))

            start = colptr[bc]
            end = colptr[bc + 1]

            A_sh = T.alloc_shared((16, 16), "bfloat16")
            B_sh = T.alloc_shared((16, 16), "bfloat16")
            C_sh = T.alloc_shared((16, 16), "float64")
            A_local = T.alloc_local((mma.warp_rows * mma.local_size_a,), "bfloat16")
            B_local = T.alloc_local((mma.warp_cols * mma.local_size_b,), "bfloat16")
            C_local = T.alloc_local(
                (mma.warp_rows * mma.warp_cols * mma.local_size_out,), "float64"
            )

            down = T.float64(tc_scale_down)
            up = down * down

            for k in T.serial(start, end):
                br = rowind[k]
                row_base = br * R

                # Note: TensorCore kernel processes multiple columns at once via MMA.
                # For per-column scaling, we use the scale of the first column in this block-column as approximation.
                a_scale_first_col = T.if_then_else(
                    action == 1,
                    a_scale_fp32[x_base],
                    T.if_then_else(action == 2, a_scale_bf16[x_base], T.float64(1.0)),
                )

                for rm_blk in T.serial((R + 15) // 16):
                    rm = rm_blk * 16

                    for i in T.serial(
                        mma.warp_rows * mma.warp_cols * mma.local_size_out
                    ):
                        C_local[i] = T.float64(0)

                    for ck_blk in T.serial((C + 15) // 16):
                        ck = ck_blk * 16

                        # ---- A tile ----
                        for t in T.serial(((16 * 16) + 31) // 32):
                            idx = t * 32 + lane
                            if idx < 256:
                                rr = idx // 16
                                cc = idx % 16
                                a_q = T.Cast(
                                    "float64", data_bf16_q[k, rm + rr, ck + cc]
                                )
                                A_sh[rr, cc] = T.Cast("bfloat16", a_q / down)

                        # ---- B tile (FIXED HERE) ----
                        # Only nn == 0 holds x; other columns are zero
                        for t in T.serial(((16 * 16) + 31) // 32):
                            idx = t * 32 + lane
                            if idx < 256:
                                kk = idx // 16
                                nn = idx % 16
                                if nn == 0:
                                    col = x_base + (ck + kk)
                                    xq = T.float64(0.0)
                                    if col < N:
                                        xq = quantize_scaled_value(
                                            x[col], x_scale, qmax, action
                                        )
                                    B_sh[kk, 0] = T.Cast("bfloat16", xq / down)
                                else:
                                    B_sh[kk, nn] = T.Cast("bfloat16", 0.0)

                        T.sync_threads()

                        mma.ldmatrix_a(A_local, A_sh, 0, rk=0)
                        mma.ldmatrix_b(B_local, B_sh, 0, rk=0)
                        mma.mma(A_local, B_local, C_local)

                        T.sync_threads()

                    mma.stmatrix(C_local, C_sh)
                    T.sync_threads()

                    for rr in T.serial(16):
                        if (rm + rr) < R:
                            if lane == 0:
                                val = T.Cast("float64", C_sh[rr, 0]) * up
                                out = T.if_then_else(
                                    action == 0,
                                    val,
                                    val / (a_scale_first_col * x_scale),
                                )
                                T.atomic_add(y[row_base + rm + rr], out)

                    T.sync_threads()

    return main



def bcsc_spmv_mixed_prequant_tensorcore_safe(
    bcsc: BCSCMatrix,
    actions,
    x,
    *,
    device: str = "cuda",
    tc_scale_down: float = 3.0675e14,
    return_torch: bool = True,
):
    """
    Wrapper for the safe bf16 TensorCore kernel (Stage E bring-up).
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("bcsc_spmv_mixed_prequant_tensorcore_safe supports device='cuda' only.")

    bcsc_d = bcsc
    if bcsc_d.A_fp64.device.type != dev.type:
        raise ValueError(
            f"Expected bcsc tensors on {dev.type}, but got {bcsc_d.A_fp64.device}. "
            "Move once via bcsc = bcsc.to('cuda')."
        )
    if (
        bcsc_d.A_fp32_q is None
        or bcsc_d.a_scale_fp32 is None
        or bcsc_d.A_bf16_q is None
        or bcsc_d.a_scale_bf16 is None
    ):
        raise RuntimeError("bcsc is missing pre-quantized tiles; call quantize_bcsc_tiles(bcsc) first.")

    actions_t = actions if torch.is_tensor(actions) else torch.as_tensor(actions, dtype=torch.int32)
    actions_t = actions_t.to(device=dev, dtype=torch.int32)
    if int(actions_t.numel()) != int(bcsc_d.n_bc):
        raise ValueError(f"actions length mismatch: got {int(actions_t.numel())}, expected n_bc={int(bcsc_d.n_bc)}")

    x_t = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float64)
    x_t = x_t.to(device=dev, dtype=torch.float64)
    if int(x_t.shape[0]) != int(bcsc_d.N):
        raise ValueError(f"x length mismatch: got N={int(x_t.shape[0])}, expected {int(bcsc_d.N)}")

    y_t = torch.zeros((int(bcsc_d.n_br) * int(bcsc_d.R),), dtype=torch.float64, device=dev)

    kernel = make_bcsc_spmv_mixed_prequant_kernel_tensorcore_bf16_safe(
        int(bcsc_d.n_br),
        int(bcsc_d.n_bc),
        int(bcsc_d.nnzb),
        int(bcsc_d.R),
        int(bcsc_d.C),
        int(bcsc_d.N),
        tc_scale_down=float(tc_scale_down),
    )
    kernel(
        bcsc_d.A_fp64,
        bcsc_d.A_fp32_q,
        bcsc_d.a_scale_fp32,
        bcsc_d.A_bf16_q,
        bcsc_d.a_scale_bf16,
        actions_t,
        bcsc_d.colptr,
        bcsc_d.rowind,
        x_t,
        y_t,
    )
    return y_t if return_torch else y_t.detach().cpu().numpy()


