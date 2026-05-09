#!/usr/bin/env bash
# Build Ginkgo with CUDA support (A100 / SM80).
# Supports incremental builds: only re-runs cmake configure if build/ is absent.
# Disables tests and benchmarks to reduce build time.
# GINKGO_SPLIT_TEMPLATE_INSTANTIATIONS=OFF avoids a CUDA 11.5 + GCC 11
# std::function template-pack issue in the split instantiation files.
#
# Usage:
#   bash compile_Ginkgo.sh          # incremental build
#   bash compile_Ginkgo.sh clean    # wipe build/ and rebuild from scratch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${1:-}" == "clean" ]]; then
  echo "Removing build directory for clean rebuild..."
  rm -rf build
fi

if [[ ! -d build ]]; then
  mkdir build
  cd build
  cmake -G "Unix Makefiles" \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DGINKGO_BUILD_TESTS=OFF \
        -DGINKGO_BUILD_BENCHMARKS=OFF \
        -DGINKGO_SPLIT_TEMPLATE_INSTANTIATIONS=OFF \
        ..
else
  cd build
fi

cmake --build . -j4
