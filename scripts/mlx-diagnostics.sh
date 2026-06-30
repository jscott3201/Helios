#!/usr/bin/env bash
set -euo pipefail

echo "cmake: $(command -v cmake)"
cmake --version | sed -n '1p'
echo "c++: $(command -v c++)"
c++ --version | sed -n '1p'
echo "probing MLX CMake package with GEMMA4D_REQUIRE_MLX=ON"

cmake -S native/gemma4_mlx -B target/mlx-diagnostics -DGEMMA4D_REQUIRE_MLX=ON
