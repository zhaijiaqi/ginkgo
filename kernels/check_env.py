import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda")
def copy1d(N: int, dtype: str = "float32"):
    @T.prim_func
    def kernel(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            # simple 1D loop
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < N:
                    B[idx] = A[idx]

    return kernel


N = 1024
copy_kernel = copy1d(N)

import torch

A = torch.randn(N, device="cuda", dtype=torch.float32)
B = torch.empty_like(A)
copy_kernel(A, B)
