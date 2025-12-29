"""
TileLang kernels for BCSC (Block Compressed Sparse Column) SpMV.

Stage D (spmv_kernel_dev.md):
  - Read BCSC (colptr/rowind) structure
  - Use pre-quantized A tiles (fp32/bf16 quantized-domain buffers + per-tile a_scale)
  - Keep online quantization for x (vector tile) to compute x_scale and x_q

NOTE:
BCSC iterates block-columns (bc). This means the output y is updated from many bc's, so
we use atomic adds when accumulating into y.
"""

# pyright: reportInvalidTypeForm=false

import numpy as np
import tilelang
import tilelang.language as T
import torch

from .bcsc_prequant import BCSCMatrix
from tilelang.intrinsics.mma_macro_generator import TensorCoreIntrinEmitter
from .bcsc_spmv_kernels_tc_safe import bcsc_spmv_mixed_prequant_tensorcore_safe


@tilelang.jit(target="cuda")
def make_bcsc_spmv_mixed_prequant_kernel_warp_reduce(
    n_br: int,
    n_bc: int,
    nnzb: int,
    R: int,
    C: int,
    N: int,
    WARPS_PER_BLOCK: int = 2,
):
    """
    BCSC mixed SpMV with PRE-QUANTIZED A tiles.

    BCSC representation (block-column compressed):
      colptr : (n_bc+1,) int32   # tile pointer per block-column
      rowind : (nnzb,)   int32   # block-row index per tile

    Tile payloads:
      data_fp64    : (nnzb, R, C) float64   # action==0
      data_fp32_q  : (nnzb, R, C) float32   # action==1 (quantized-domain values)
      a_scale_fp32 : (nnzb,)      float64
      data_bf16_q  : (nnzb, R, C) bfloat16  # action==2 (quantized-domain values)
      a_scale_bf16 : (nnzb,)      float64

    Vector:
      actions : (n_bc,) int32     # 0=fp64, 1=fp32, 2=bf16 (per block-column)
      x       : (N,)    float64
      y       : (n_br*R,) float64  # atomic accumulated
    """

    # Mul-safe quantization constant (see kernels/spmv_kernels.py).
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
                    # Mul-safe already, but keep explicit clamp before bf16 cast.
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

    R_TILES = (R + 31) // 32
    C_TILES = (C + 31) // 32

    @T.prim_func
    def main(
        data_fp64: T.Tensor((nnzb, R, C), "float64"),  # type: ignore
        data_fp32_q: T.Tensor((nnzb, R, C), "float32"),  # type: ignore
        a_scale_fp32: T.Tensor((nnzb,), "float64"),  # type: ignore
        data_bf16_q: T.Tensor((nnzb, R, C), "bfloat16"),  # type: ignore
        a_scale_bf16: T.Tensor((nnzb,), "float64"),  # type: ignore
        actions: T.Tensor((n_bc,), "int32"),  # type: ignore
        colptr: T.Tensor((n_bc + 1,), "int32"),  # type: ignore
        rowind: T.Tensor((nnzb,), "int32"),  # type: ignore
        x: T.Tensor((N,), "float64"),  # type: ignore
        y: T.Tensor((n_br * R,), "float64"),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(n_bc, WARPS_PER_BLOCK), threads=32 * WARPS_PER_BLOCK) as bx:
            warp = T.get_warp_idx_sync()
            lane = T.get_lane_idx()
            bc = bx * WARPS_PER_BLOCK + warp
            mask = T.tvm_warp_activemask()

            # Per-warp scratch - allocate all needed types upfront
            x_lane_f64 = T.alloc_local((1,), "float64")
            x_lane_f32 = T.alloc_local((1,), "float32")
            x_lane_bf16 = T.alloc_local((1,), "bfloat16")
            acc = T.alloc_local((R_TILES,), "float64")

            if bc < n_bc:
                action = actions[bc]
                x_base = bc * C

                qmax = get_qmax(action)
                x_scale = T.float64(1.0)

                # Online x quantization only when low precision is selected
                if action >= 1:
                    x_max_abs = T.float64(0.0)
                    for ct_check in T.unroll(C_TILES):
                        cc_base_check = ct_check * 32
                        for cc_check in T.unroll(32):
                            cc_g_check = cc_base_check + cc_check
                            if cc_g_check < C:
                                col_check = x_base + cc_g_check
                                if col_check < N:
                                    x_val_check = x[col_check]
                                    x_max_abs = T.max(x_max_abs, T.abs(x_val_check))
                    x_scale = qmax / (x_max_abs + T.float64(1e-12))

                start = colptr[bc]
                end = colptr[bc + 1]

                # Iterate tiles in this block-column
                for k in T.serial(start, end):
                    br = rowind[k]
                    row_base = br * R

                    # Per-tile accumulators (reset for each tile)
                    for rt in T.unroll(R_TILES):
                        acc[rt] = T.float64(0)

                    # Select per-tile a_scale (only used for action>=1)
                    # Pre-compute inverse scale to use multiplication instead of division
                    a_scale = T.if_then_else(
                        action == 1,
                        a_scale_fp32[k],
                        T.if_then_else(action == 2, a_scale_bf16[k], T.float64(1.0)),
                    )
                    # Pre-compute inverse scale for faster dequantization (multiply instead of divide)
                    inv_scale = T.if_then_else(
                        action == 0,
                        T.float64(1.0),
                        T.float64(1.0) / (a_scale * x_scale),
                    )

                    for ct in T.unroll(C_TILES):
                        cc_base = ct * 32

                        # Load x element based on action, using appropriate precision type
                        col = x_base + cc_base + lane
                        if (cc_base + lane) < C and col < N:
                            if action == 0:
                                x_lane_f64[0] = x[col]
                            elif action == 1:
                                x_val_f64 = quantize_scaled_value(x[col], x_scale, qmax, action)
                                x_lane_f32[0] = T.Cast("float32", x_val_f64)
                            else:  # action == 2
                                x_val_f64 = quantize_scaled_value(x[col], x_scale, qmax, action)
                                x_lane_bf16[0] = T.Cast("bfloat16", x_val_f64)
                        else:
                            if action == 0:
                                x_lane_f64[0] = T.float64(0)
                            elif action == 1:
                                x_lane_f32[0] = T.float32(0)
                            else:  # action == 2
                                x_lane_bf16[0] = T.Cast("bfloat16", T.float64(0))

                        for cc in T.unroll(32):
                            cc_g = cc_base + cc
                            if cc_g < C:
                                if action == 0:
                                    # fp64 path: use fp64 throughout, CUDA core uses fp64 instructions
                                    vx = T.tvm_warp_shuffle(mask, x_lane_f64[0], T.int32(cc), 32, 32)
                                    for rt in T.unroll(R_TILES):
                                        rr = rt * 32 + lane
                                        if rr < R:
                                            a_val = data_fp64[k, rr, cc_g]
                                            acc[rt] += a_val * vx
                                elif action == 1:
                                    # fp32 path: use fp32 for computation, CUDA core uses fp32 instructions
                                    vx_f32 = T.tvm_warp_shuffle(mask, x_lane_f32[0], T.int32(cc), 32, 32)
                                    for rt in T.unroll(R_TILES):
                                        rr = rt * 32 + lane
                                        if rr < R:
                                            # Direct fp32 multiplication (CUDA core uses fp32 instructions)
                                            a_f32 = data_fp32_q[k, rr, cc_g]
                                            prod_f32 = a_f32 * vx_f32
                                            # Cast to fp64 for accumulation and dequantization (use multiply instead of divide)
                                            prod_f64 = T.Cast("float64", prod_f32)
                                            acc[rt] += prod_f64 * inv_scale
                                else:  # action == 2
                                    # bf16 path: use bf16 for computation, CUDA core uses bf16 instructions
                                    # Note: warp shuffle may need fp32, so we use fp32 for shuffle then cast
                                    vx_f32_shuffle = T.Cast("float32", T.tvm_warp_shuffle(mask, T.Cast("float32", x_lane_bf16[0]), T.int32(cc), 32, 32))
                                    vx_bf16 = T.Cast("bfloat16", vx_f32_shuffle)
                                    for rt in T.unroll(R_TILES):
                                        rr = rt * 32 + lane
                                        if rr < R:
                                            # Direct bf16 multiplication (CUDA core uses bf16 instructions)
                                            a_bf16 = data_bf16_q[k, rr, cc_g]
                                            prod_bf16 = a_bf16 * vx_bf16
                                            # Cast to fp64 for accumulation and dequantization (use multiply instead of divide)
                                            prod_f64 = T.Cast("float64", prod_bf16)
                                            acc[rt] += prod_f64 * inv_scale

                    # Scatter accumulate into y
                    for rt in T.unroll(R_TILES):
                        rr = rt * 32 + lane
                        if rr < R:
                            T.atomic_add(y[row_base + rr], acc[rt])

    return main


@tilelang.jit(target="cuda")
def make_bcsc_spmv_mixed_prequant_kernel_tensorcore_bf16(
    n_br: int,
    n_bc: int,
    nnzb: int,
    R: int,
    C: int,
    N: int,
):
    """
    TensorCore variant for action==2 (bf16): use MMA (m16n8k16) with fp32 accumulation.

    Constraints (current implementation):
      - R==C==64
      - action==2 uses TensorCore path
      - action==0/1 fallback to scalar path (same semantics as warp_reduce kernel)

    Notes:
      - We compute GEMV via GEMM A(16x16)*B(16x8) where B replicates the x-vector segment
        across 8 columns, then we take column 0 of the 16x8 output.
    """

    # Mul-safe quantization constant (see kernels/spmv_kernels.py).
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

    # TensorCore MMA emitter (m16n8k16). We use fp16 MMA for robustness/compatibility,
    # casting bf16 quantized-domain values into fp16 in shared memory before ldmatrix+mmas.
    # IMPORTANT: instantiate in Python scope (kernel build time), not inside primfunc.
    mma = TensorCoreIntrinEmitter(
        a_dtype="float16",
        b_dtype="float16",
        accum_dtype="float32",
        a_transposed=False,
        b_transposed=False,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=16,  # n_dim=16 (replicate_b inside emitter)
        chunk=16,
        reduce_k=1,
    )

    @T.prim_func
    def main(
        data_fp64: T.Tensor((nnzb, R, C), "float64"),  # type: ignore
        data_fp32_q: T.Tensor((nnzb, R, C), "float32"),  # type: ignore
        a_scale_fp32: T.Tensor((nnzb,), "float64"),  # type: ignore
        data_bf16_q: T.Tensor((nnzb, R, C), "bfloat16"),  # type: ignore
        a_scale_bf16: T.Tensor((nnzb,), "float64"),  # type: ignore
        actions: T.Tensor((n_bc,), "int32"),  # type: ignore
        colptr: T.Tensor((n_bc + 1,), "int32"),  # type: ignore
        rowind: T.Tensor((nnzb,), "int32"),  # type: ignore
        x: T.Tensor((N,), "float64"),  # type: ignore
        y: T.Tensor((n_br * R,), "float64"),  # type: ignore
    ):
        # One warp per block (TensorCore MMA)
        with T.Kernel(n_bc, threads=32) as bc:
            lane = T.get_lane_idx()
            # mask = T.tvm_warp_activemask()

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

            # Shared memory tiles for MMA
            A_sh = T.alloc_shared((16, 16), "float16")
            B_sh = T.alloc_shared((16, 16), "float16")
            C_sh = T.alloc_shared((16, 16), "float32")

            # Local registers used by ptx_ldmatrix/ptx_mma. TileLang GEMM MMA uses alloc_local here.
            A_local = T.alloc_local((mma.warp_rows * mma.local_size_a,), "float16")
            B_local = T.alloc_local((mma.warp_cols * mma.local_size_b,), "float16")
            C_local = T.alloc_local((mma.warp_rows * mma.warp_cols * mma.local_size_out,), "float32")

            # Iterate tiles in this block-column
            for k in T.serial(start, end):
                br = rowind[k]
                row_base = br * R

                # Per-tile a_scale (only used for action>=1)
                a_scale = T.if_then_else(
                    action == 1,
                    a_scale_fp32[k],
                    T.if_then_else(action == 2, a_scale_bf16[k], T.float64(1.0)),
                )

                # For each 16-row block of this tile
                for rm_blk in T.serial((R + 15) // 16):
                    rm = rm_blk * 16
                    # Clear C_local
                    for i in T.serial(mma.warp_rows * mma.warp_cols * mma.local_size_out):
                        C_local[i] = T.float32(0)

                    # K loop over 64 columns in chunks of 16
                    for ck_blk in T.serial((C + 15) // 16):
                        ck = ck_blk * 16
                        # Build A_sh (16x16) and B_sh (16x8) in shared memory
                        # Cooperative load by warp lanes
                        for t in T.serial(((16 * 16) + 31) // 32):
                            idx = t * 32 + lane
                            if idx < 16 * 16:
                                rr = idx // 16
                                cc = idx % 16
                                # A is bf16 quantized-domain -> cast to fp16 for MMA
                                A_sh[rr, cc] = T.Cast("float16", data_bf16_q[k, rm + rr, ck + cc])

                        for t in T.serial(((16 * 16) + 31) // 32):
                            idx = t * 32 + lane
                            if idx < 16 * 16:
                                kk = idx // 16
                                nn = idx % 16
                                col = x_base + (ck + kk)
                                xq = T.float64(0.0)
                                if col < N:
                                    xq = quantize_scaled_value(x[col], x_scale, qmax, action)
                                # replicate into 16 columns, cast to fp16 for MMA
                                B_sh[kk, nn] = T.Cast("float16", xq)

                        T.sync_threads()

                        # ldmatrix + mma (single ki since chunk=16)
                        mma.ldmatrix_a(A_local, A_sh, 0, rk=0)
                        mma.ldmatrix_b(B_local, B_sh, 0, rk=0)
                        mma.mma(A_local, B_local, C_local)

                        T.sync_threads()

                    # Store C_local -> C_sh
                    mma.stmatrix(C_local, C_sh)
                    T.sync_threads()

                    # Take column 0 and accumulate into y
                    # y is float64, C_sh is fp32 => cast up, then dequantize.
                    for rr in T.serial(16):
                        if (rm + rr) < R:
                            # Only one lane writes each row to reduce atomics; pick lane==0 and loop rr.
                            if lane == 0:
                                val = T.Cast("float64", C_sh[rr, 0])
                                out = T.if_then_else(
                                    action == 0,
                                    val,
                                    val / (a_scale * x_scale),
                                )
                                T.atomic_add(y[row_base + rm + rr], out)

                    T.sync_threads()

            # NOTE: This kernel is intended for R=C=64 and action==2-heavy workloads.
            # action==0/1 are still supported in the math above (via quantize_scaled_value),
            # but A is sourced from bf16 buffer here. Prefer using the warp-reduce kernel
            # for mixed 0/1/2 workloads, or extend this kernel to use fp32/fp64 A for 0/1.

    return main


def bcsc_spmv_mixed_prequant(
    bcsc: BCSCMatrix,
    actions,
    x,
    *,
    device: str = "cuda",
    use_tensorcore_bf16: bool = False,
    return_torch: bool = True,
):
    """
    Python wrapper: run BCSC mixed prequant kernel.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("bcsc_spmv_mixed_prequant currently supports device='cuda' only.")

    # For performance, we REQUIRE the BCSC payloads to already live on the target device.
    # Moving large BCSC tensors every step would dominate runtime.
    bcsc_d = bcsc
    # Accept any CUDA device index (cuda vs cuda:0). Require CUDA residency.
    if bcsc_d.A_fp64.device.type != dev.type:
        raise ValueError(
            f"bcsc_spmv_mixed_prequant expects bcsc tensors on {dev.type}, but got {bcsc_d.A_fp64.device}. "
            "Move bcsc once during preprocessing via bcsc = bcsc.to('cuda')."
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
    N = int(x_t.shape[0])
    if N != int(bcsc_d.N):
        raise ValueError(f"x length mismatch: got N={N}, expected {int(bcsc_d.N)}")

    if use_tensorcore_bf16:
        # Use safe TensorCore version that handles fp16 overflow by scaling
        return bcsc_spmv_mixed_prequant_tensorcore_safe(
            bcsc_d,
            actions_t,
            x_t,
            device=device,
            return_torch=return_torch,
        )

    y_t = torch.zeros((int(bcsc_d.n_br) * int(bcsc_d.R),), dtype=torch.float64, device=dev)

    kernel = make_bcsc_spmv_mixed_prequant_kernel_warp_reduce(
        int(bcsc_d.n_br),
        int(bcsc_d.n_bc),
        int(bcsc_d.nnzb),
        int(bcsc_d.R),
        int(bcsc_d.C),
        int(bcsc_d.N),
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


def _debug_build_random_bcsc(
    *,
    M: int = 256,
    N: int = 256,
    tilesize: int = 32,
    density: float = 0.05,
    seed: int = 0,
):
    """
    Helper for ad-hoc manual debugging. Not used by tests.
    """
    try:
        from scipy.sparse import coo_matrix
    except Exception as e:
        raise RuntimeError(f"scipy unavailable: {e}") from e

    rng = np.random.default_rng(seed)
    nnz = int(M * N * float(density))
    rows = rng.integers(0, M, size=nnz, dtype=np.int32)
    cols = rng.integers(0, N, size=nnz, dtype=np.int32)
    vals = rng.standard_normal(nnz).astype(np.float64)
    A_coo = coo_matrix((vals, (rows, cols)), shape=(M, N))
    return A_coo


