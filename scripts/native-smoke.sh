#!/usr/bin/env bash
set -euo pipefail

cmake -S native/gemma4_mlx -B target/native-smoke
cmake --build target/native-smoke
