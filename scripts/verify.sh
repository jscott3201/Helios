#!/usr/bin/env bash
set -euo pipefail

cargo metadata --format-version=1 --no-deps >/dev/null
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo test -p gemma4d-ffi
./scripts/native-smoke.sh
