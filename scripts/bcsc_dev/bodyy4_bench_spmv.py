#!/usr/bin/env python3
"""
Quick SpMV micro-benchmark on bodyy4.mtx for:
  - bsr (online quant)
  - bsr_prequant
  - bcsc_prequant

This is for developer iteration, not a rigorous benchmark.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.cg_env import CGEnvironment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtx-name", type=str, default="bodyy4")
    ap.add_argument("--mtx-dir", type=str, default="/home/bingxing2/home/scx7axu/data/matrix")
    ap.add_argument("--tilesize", type=int, default=64)
    ap.add_argument("--impl", type=str, default="bsr", choices=["bsr", "bsr_prequant", "bcsc_prequant"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA unavailable.")
        return

    cfg = {
        "matrix_name": args.mtx_name,
        "matrix_data_dir": args.mtx_dir,
        "tilesize": int(args.tilesize),
        "torch_device": "cuda",
        "use_torch_state": True,
        "verbose": False,
        "max_iter": 1,
    }

    if args.impl == "bsr":
        cfg["spmv_impl"] = "bsr"
    elif args.impl == "bsr_prequant":
        cfg["spmv_impl"] = "bsr_prequant"
        cfg["use_bsr_prequant"] = True
        cfg["bsr_prequant_device"] = "cuda"
    else:
        cfg["spmv_impl"] = "bcsc_prequant"
        cfg["use_bcsc_prequant"] = True
        cfg["bcsc_device"] = "cpu"  # build+quant on cpu; env will move to cuda once when needed

    env = CGEnvironment(cfg)
    env.reset()

    rng = np.random.default_rng(int(args.seed))
    actions = rng.integers(0, 2, size=((env.matrix_size + env.spmv_sim.tilesize - 1) // env.spmv_sim.tilesize,), dtype=np.int32)
    actions_list = actions.tolist()

    # Warmup
    for _ in range(int(args.warmup)):
        env.step(actions_list)
        env.current_iteration = 0
        env.episode_done = False

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(int(args.iters)):
        env.step(actions_list)
        env.current_iteration = 0
        env.episode_done = False
    torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1000.0 / float(args.iters)
    print(f"[OK] bodyy4 bench impl={args.impl} tilesize={args.tilesize} avg_step_ms={avg_ms:.3f}")


if __name__ == "__main__":
    main()


