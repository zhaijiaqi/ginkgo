#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

rm -rf build
mkdir build
cd build
cmake -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${SCRIPT_DIR}/install" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DGINKGO_BUILD_TESTS=OFF \
  -DGINKGO_BUILD_EXAMPLES=ON \
  -DGINKGO_BUILD_BENCHMARKS=OFF \
  .. && cmake --build . -j$(nproc)
cmake --install .