#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./test_Ginkgo.sh [matrix_set.csv]
#   不传参时默认读取项目根目录的 valid_matrix_set.csv
#
# 说明：
# - 模仿 baselines/cusparse/test_cuSPARSE.sh
# - 逐行读取 csv 的 Name 列（第3列），在 MATRIX_ROOT 下查找对应的 <Name>.mtx
# - 对每个找到的矩阵运行 Ginkgo simple-solver CG，并追加输出到 ginkgo_cg_a100.csv
#
# 环境变量：
#   EXECUTOR       - cuda | omp | reference，默认 cuda
#   MATRIX_ROOT    - 矩阵目录，默认 /data/matrix
#   GINKGO_BUILD   - Ginkgo build 目录，默认 ../../build
#   TRUNCATE_OUT_CSV - 1 时清空输出文件再跑，默认 0
#   MAX_MATS       - 最多测试矩阵数，0 表示不限制
#   TIMEOUT_LIMIT  - 单矩阵超时，默认 4m

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../../../../" && pwd)"

out_csv="${OUT_CSV:-ginkgo_cg_a100.csv}"
executor="${EXECUTOR:-cuda}"
matrix_root="${MATRIX_ROOT:-/data/matrix}"
ginkgo_build="${GINKGO_BUILD:-${script_dir}/../../build}"
timeout_limit="${TIMEOUT_LIMIT:-4m}"
max_mats="${MAX_MATS:-0}"

bin="${ginkgo_build}/examples/simple-solver/simple-solver"

# 未传参时默认使用项目根目录的 valid_matrix_set.csv
input="${1:-${project_root}/valid_matrix_set.csv}"

# 库路径（Ginkgo + CUDA）
export LD_LIBRARY_PATH="${script_dir}/../../install/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${input}" ]]; then
  echo "ERROR: input csv not found: ${input}" >&2
  exit 1
fi

if [[ ! -x "${bin}" ]]; then
  echo "ERROR: binary not found or not executable: ${bin}" >&2
  echo "Hint: 先编译 Ginkgo 并启用 examples (complie_Ginkgo.sh 中 GINKGO_BUILD_EXAMPLES=ON)" >&2
  exit 1
fi

# 输出写到源码目录
out_path="${script_dir}/${out_csv}"
if [[ "${TRUNCATE_OUT_CSV:-0}" == "1" ]]; then
  : > "${out_path}"
fi

cd "${script_dir}"
echo "Output: ${out_path}"
echo "Executor: ${executor}"
echo "Matrix root: ${matrix_root}"
echo "---"

i=0
{
  read -r _
  while IFS=',' read -r id group name rows cols entries; do
    [[ -z "${name}" ]] && continue
    if [[ "${max_mats}" != "0" ]] && [[ "${i}" -ge "${max_mats}" ]]; then
      break
    fi
    direct="${matrix_root}/${name}.mtx"
    if [[ -f "${direct}" ]]; then
      echo "RUN ${direct}"
      timeout -s 9 "${timeout_limit}" "${bin}" "${executor}" "${direct}" 1 || true
      i=$((i + 1))
      continue
    fi
    while IFS= read -r mtx; do
      [[ -z "${mtx}" ]] && continue
      echo "RUN ${mtx}"
      timeout -s 9 "${timeout_limit}" "${bin}" "${executor}" "${mtx}" 1 || true
      i=$((i + 1))
      if [[ "${max_mats}" != "0" ]] && [[ "${i}" -ge "${max_mats}" ]]; then
        break 2
      fi
    done < <(find "${matrix_root}" -name "${name}.mtx" 2>/dev/null || true)
  done
} < "${input}"

echo "Done. Results in ${out_path}"
