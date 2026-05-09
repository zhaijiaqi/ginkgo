#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./test_Ginkgo.sh [matrix_set.csv]
#   不传参时默认读取本目录下的 valid_matrix_set.csv
#
# 说明：
# - 模仿 baselines/cusparse/test_cuSPARSE.sh
# - 逐行读取 csv 的 Name 列（第3列），在 MATRIX_ROOT 下查找对应的 <Name>.mtx
# - 对每个找到的矩阵运行 Ginkgo CG（CUDA 后端），并追加输出到 ginkgo_cg_a100.csv
#
# 环境变量：
#   MATRIX_ROOT      矩阵根目录，默认 /data/matrix（或 $HOME/data/matrix）
#   MAX_IT           最大迭代次数，默认 10000
#   TOL              收敛容差，默认 1e-10
#   TIMEOUT_LIMIT    单矩阵超时，默认 4m
#   MAX_MATS         最多测试矩阵数，0 表示不限制
#   OUT_CSV          输出 csv 路径，默认 ginkgo_cg_a100.csv

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_csv="${OUT_CSV:-${script_dir}/examples/simple-solver/ginkgo_cg_a100.csv}"
max_it="${MAX_IT:-10000}"
tol="${TOL:-1e-10}"
timeout_limit="${TIMEOUT_LIMIT:-4m}"
matrix_root="${MATRIX_ROOT:-/data/matrix}"
[[ -d "${matrix_root}" ]] || matrix_root="${HOME}/data/matrix"
max_mats="${MAX_MATS:-0}"

bin="${script_dir}/build/examples/simple-solver/simple-solver"

# 未传参时默认使用脚本同目录下的 valid_matrix_set.csv
input="${1:-${script_dir}/../../cg_kernels/valid_matrix_set.csv}"
# 兼容：如果项目根目录没有，尝试当前目录
[[ -f "${input}" ]] || input="${script_dir}/valid_matrix_set.csv"

if [[ ! -f "${input}" ]]; then
  echo "ERROR: input csv not found: ${input}" >&2
  exit 1
fi

if [[ ! -x "${bin}" ]]; then
  echo "ERROR: binary not found or not executable: ${bin}" >&2
  echo "Hint: cd ${script_dir} && bash compile_Ginkgo.sh" >&2
  exit 1
fi

# 清空输出文件
: > "${out_csv}"

i=0
{
  read -r _header
  while IFS=',' read -r id group name rows cols entries _rest; do
    [[ -z "${name}" ]] && continue
    if [[ "${max_mats}" != "0" && "${i}" -ge "${max_mats}" ]]; then
      break
    fi

    # 优先走直接路径，避免对整个目录反复 find
    direct="${matrix_root}/${name}.mtx"
    if [[ -f "${direct}" ]]; then
      echo "RUN ${direct}"
      timeout -s 9 "${timeout_limit}" "${bin}" "${direct}" "${max_it}" "${tol}" \
        --csv --csv_out "${out_csv}" || true
      i=$(( i + 1 ))
      continue
    fi

    # 回退：子目录搜索
    found=0
    while IFS= read -r mtx; do
      [[ -z "${mtx}" ]] && continue
      echo "RUN ${mtx}"
      timeout -s 9 "${timeout_limit}" "${bin}" "${mtx}" "${max_it}" "${tol}" \
        --csv --csv_out "${out_csv}" || true
      found=1
      i=$(( i + 1 ))
      break
    done < <(find "${matrix_root}" -name "${name}.mtx" 2>/dev/null || true)

    if [[ "${found}" == "0" ]]; then
      echo "SKIP: ${name}.mtx not found under ${matrix_root}" >&2
    fi
  done
} < "${input}"

echo "Done. Processed ${i} matrices. Results saved to ${out_csv}"
