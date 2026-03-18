#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./test_cuSPARSE.sh [matrix_set.csv]
#   不传参时默认读取本目录下的 valid_matrix_set.csv
#
# 说明：
# - 模仿 baselines/petsc/src/ksp/ksp/tutorials/test_PETSc.sh
# - 逐行读取 csv 的 Name 列（第3列），在 /data/matrix 下查找对应的 <Name>.mtx
# - 对每个找到的矩阵运行 cuSPARSE CG，并追加输出到 cusparse_cg_a100.csv

out_csv="${OUT_CSV:-cusparse_cg_a100.csv}"
max_it="${MAX_IT:-10000}"
tol="${TOL:-1e-10}"
timeout_limit="${TIMEOUT_LIMIT:-4m}"
matrix_root="${MATRIX_ROOT:-/data/matrix}"
max_mats="${MAX_MATS:-0}" # 0 表示不限制

bin_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin="${bin_dir}/test_cusparse_cg"

# 未传参时默认使用脚本同目录下的 valid_matrix_set.csv
input="${1:-${bin_dir}/valid_matrix_set.csv}"

if [[ ! -f "${input}" ]]; then
  echo "ERROR: input csv not found: ${input}" >&2
  exit 1
fi

if [[ ! -x "${bin}" ]]; then
  echo "ERROR: binary not found or not executable: ${bin}" >&2
  echo "Hint: 先在 ${bin_dir} 编译生成 test_cusparse_cg" >&2
  exit 1
fi

# 清空/创建输出文件（和 PETSc 行为不同：PETSc 直接 append；这里提供可控行为）
if [[ "${TRUNCATE_OUT_CSV:-0}" == "1" ]]; then
  : > "${out_csv}"
fi

# 读 header，然后逐行处理
{
  read -r _
  while IFS=',' read -r id group name rows cols entries; do
    [[ -z "${name}" ]] && continue
    if [[ "${max_mats}" != "0" ]]; then
      if [[ "${i:-0}" -ge "${max_mats}" ]]; then
        break
      fi
    fi
    # 优先走直接路径，避免对整个目录反复 find（会很慢）
    direct="${matrix_root}/${name}.mtx"
    if [[ -f "${direct}" ]]; then
      echo "RUN ${direct}"
      timeout -s 9 "${timeout_limit}" "${bin}" "${direct}" "${max_it}" "${tol}" --csv --csv_out "${out_csv}"
      i="$(( ${i:-0} + 1 ))"
      continue
    fi

    # 回退：如果矩阵被放在子目录里，再用 find
    while IFS= read -r mtx; do
      [[ -z "${mtx}" ]] && continue
      echo "RUN ${mtx}"
      timeout -s 9 "${timeout_limit}" "${bin}" "${mtx}" "${max_it}" "${tol}" --csv --csv_out "${out_csv}"
      i="$(( ${i:-0} + 1 ))"
      if [[ "${max_mats}" != "0" && "${i}" -ge "${max_mats}" ]]; then
        break 2
      fi
    done < <(find "${matrix_root}" -name "${name}.mtx" 2>/dev/null || true)
  done
} < "${input}"

