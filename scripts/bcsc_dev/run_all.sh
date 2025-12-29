#!/usr/bin/env bash
set -euo pipefail

# Non-interactive "one command" dev runner for spmv_kernel_dev.md stages.
# It follows the documented environment approach:
#   source ~/.rlcg_env.sh
#   module load cudnn/8.8.1.3_cuda12.x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# If we're already inside a conda environment (e.g. prompt shows "(mille)"),
# do NOT source ~/.rlcg_env.sh because it runs `conda activate ...`, which may fail
# in non-interactive shells where the `conda` shell function isn't initialized.
if [[ -z "${CONDA_DEFAULT_ENV:-}" && -f "${HOME}/.rlcg_env.sh" ]]; then
  # Best-effort enable `conda activate` in non-interactive shells.
  if command -v conda >/dev/null 2>&1; then
    # Newer conda supports printing a shell hook via the CLI.
    eval "$(conda shell.bash hook 2>/dev/null)" || true
  fi
  # shellcheck disable=SC1090
  source "${HOME}/.rlcg_env.sh"
fi

# Best-effort modules init (so `module load ...` works in non-interactive shells).
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi

if command -v module >/dev/null 2>&1; then
  module load cudnn/8.8.1.3_cuda12.x >/dev/null 2>&1 || true
fi

python3 scripts/bcsc_dev/toy_stageA_bcsc_build.py
python3 scripts/bcsc_dev/toy_stageB_prequant.py
python3 scripts/bcsc_dev/toy_stageC_ref_spmv.py
python3 scripts/bcsc_dev/toy_stageD_kernel.py

# BSR pre-quant kernel smoke (fast)
python3 scripts/bcsc_dev/bsr_prequant_smoke.py

# bodyy4 is optional; will [SKIP] if file/scipy missing
python3 scripts/bcsc_dev/bodyy4_sanity.py --quick
python3 scripts/bcsc_dev/bodyy4_kernel_check.py --sample-bc 16

echo "[OK] run_all.sh completed."


