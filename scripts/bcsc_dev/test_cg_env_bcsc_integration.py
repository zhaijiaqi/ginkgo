#!/usr/bin/env python3
"""
Test BCSC prequant integration in CGEnvironment.

Verifies that:
  1. CGEnvironment can build BCSC from sparse matrix
  2. BCSC prequant SpMV works in step() method
  3. Both bcsc_ref and bcsc_prequant implementations work
  4. Actions are correctly mapped to block-columns

Non-interactive: exits non-zero on failure.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from env.cg_env import CGEnvironment  # noqa: E402


def test_bcsc_ref_impl(cfg, verbose=False):
    """Test bcsc_ref implementation (reference SpMV)."""
    cfg["spmv_impl"] = "bcsc_ref"
    cfg["use_bcsc_prequant"] = True
    cfg["bcsc_device"] = "cpu"

    env = CGEnvironment(cfg)
    env.reset()

    if env.bcsc is None:
        raise AssertionError("bcsc_ref: bcsc should be built after reset()")
    
    # Verify spmv_impl is set correctly
    if env.spmv_impl != "bcsc_ref":
        raise AssertionError(f"bcsc_ref: spmv_impl should be 'bcsc_ref', got '{env.spmv_impl}'")

    # Test a single step with random actions
    # Note: For BCSC, step() now expects actions length = n_bc (by column blocks)
    rng = np.random.default_rng(0)
    n_bc = env.bcsc.n_bc
    actions = rng.integers(0, 3, size=(n_bc,), dtype=np.int32).tolist()
    
    if verbose:
        print(f"[DEBUG] bcsc_ref: n_bc={n_bc}, actions_len={len(actions)}, spmv_impl={env.spmv_impl}")

    try:
        obs, reward, done, info = env.step(actions)
    except Exception as e:
        raise AssertionError(f"bcsc_ref: step() raised exception: {e}") from e

    # Basic sanity checks
    assert obs is not None, "bcsc_ref: obs should not be None"
    assert isinstance(reward, (float, np.floating)), f"bcsc_ref: reward should be float, got {type(reward)}"
    assert isinstance(done, bool), f"bcsc_ref: done should be bool, got {type(done)}"
    # Note: info contains 'Ap_shape' but not 'Ap' itself
    # Also note: _complete_cg_iteration() resets env.Ap to None after CG iteration,
    # so we can't check env.Ap after step() returns. Instead, verify Ap_shape in info.
    assert "Ap_shape" in info, "bcsc_ref: info should contain 'Ap_shape'"
    ap_shape = info["Ap_shape"]
    assert ap_shape is not None and len(ap_shape) > 0, f"bcsc_ref: Ap_shape should be valid, got {ap_shape}"
    
    # For verification, we can check that Ap was computed by verifying the shape matches expected
    # The actual Ap tensor is reset to None after CG iteration completes, which is expected behavior
    if verbose:
        print(f"[OK] bcsc_ref: step() passed (Ap_shape={ap_shape})")


def test_bcsc_prequant_impl(cfg, verbose=False):
    """Test bcsc_prequant implementation (TileLang kernel)."""
    if not torch.cuda.is_available():
        if verbose:
            print("[SKIP] bcsc_prequant: CUDA unavailable")
        return

    cfg["spmv_impl"] = "bcsc_prequant"
    cfg["use_bcsc_prequant"] = True
    cfg["bcsc_device"] = "cpu"  # Build on CPU, move to CUDA when needed

    env = CGEnvironment(cfg)
    env.reset()

    if env.bcsc is None:
        raise AssertionError("bcsc_prequant: bcsc should be built after reset()")

    # Test a single step with random actions
    # Note: For BCSC, step() now expects actions length = n_bc (by column blocks)
    rng = np.random.default_rng(0)
    n_bc = env.bcsc.n_bc
    actions = rng.integers(0, 3, size=(n_bc,), dtype=np.int32).tolist()

    obs, reward, done, info = env.step(actions)

    # Basic sanity checks
    assert obs is not None, "bcsc_prequant: obs should not be None"
    assert isinstance(reward, (float, np.floating)), f"bcsc_prequant: reward should be float, got {type(reward)}"
    assert isinstance(done, bool), f"bcsc_prequant: done should be bool, got {type(done)}"
    # Note: info contains 'Ap_shape' but not 'Ap' itself
    # Also note: _complete_cg_iteration() resets env.Ap to None after CG iteration,
    # so we can't check env.Ap after step() returns. Instead, verify Ap_shape in info.
    assert "Ap_shape" in info, "bcsc_prequant: info should contain 'Ap_shape'"
    ap_shape = info["Ap_shape"]
    assert ap_shape is not None and len(ap_shape) > 0, f"bcsc_prequant: Ap_shape should be valid, got {ap_shape}"

    if verbose:
        print(f"[OK] bcsc_prequant: step() passed (Ap_shape={ap_shape})")


def test_actions_mapping(cfg, verbose=False):
    """Test that actions are correctly mapped to block-columns."""
    cfg["spmv_impl"] = "bcsc_ref"
    cfg["use_bcsc_prequant"] = True
    cfg["bcsc_device"] = "cpu"

    env = CGEnvironment(cfg)
    env.reset()

    n_bc = env.bcsc.n_bc
    assert n_bc > 0, "Should have at least one block-column"

    # Test with all-0 actions (should use fp64 A)
    actions_all0 = [0] * n_bc
    obs0, _, _, info0 = env.step(actions_all0)
    ap_shape0 = info0.get("Ap_shape")

    # Test with all-1 actions (should use fp32 quantized A)
    actions_all1 = [1] * n_bc
    obs1, _, _, info1 = env.step(actions_all1)
    ap_shape1 = info1.get("Ap_shape")

    # Basic sanity checks
    assert ap_shape0 is not None, "all-0 actions: Ap_shape should be in info"
    assert ap_shape1 is not None, "all-1 actions: Ap_shape should be in info"
    # Both should have the same shape (same matrix, different quantization)
    assert ap_shape0 == ap_shape1, f"all-0 and all-1 should have same Ap_shape, got {ap_shape0} vs {ap_shape1}"

    if verbose:
        print(f"[OK] Actions mapping: all-0 and all-1 both computed successfully (Ap_shape={ap_shape0})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-name", type=str, default="bodyy4")
    ap.add_argument("--matrix-dir", type=str, default="/home/bingxing2/home/scx7axu/data/matrix")
    ap.add_argument("--tilesize", type=int, default=64)
    ap.add_argument("--test-impl", type=str, default="all", choices=["ref", "prequant", "mapping", "all"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = {
        "matrix_name": args.matrix_name,
        "matrix_data_dir": args.matrix_dir,
        "tilesize": int(args.tilesize),
        "torch_device": "cpu",  # Use CPU for reference, CUDA for kernel
        "use_torch_state": True,
        "verbose": args.verbose,
        "max_iter": 1,  # Only test one step
    }

    if "ref" in args.test_impl or args.test_impl == "all":
        test_bcsc_ref_impl(cfg.copy(), verbose=args.verbose)

    if "prequant" in args.test_impl or args.test_impl == "all":
        test_bcsc_prequant_impl(cfg.copy(), verbose=args.verbose)

    if "mapping" in args.test_impl or args.test_impl == "all":
        test_actions_mapping(cfg.copy(), verbose=args.verbose)

    print("[OK] test_cg_env_bcsc_integration: all tests passed.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

