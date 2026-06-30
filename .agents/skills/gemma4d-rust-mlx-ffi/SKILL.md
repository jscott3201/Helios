---
name: gemma4d-rust-mlx-ffi
description: Use for Rust-to-MLX native C/C++ FFI work: opaque handles, C ABI, build.rs/CMake, MLX lifetime safety, native smoke tests, and unsafe Rust review.
---
# Gemma4D Rust/MLX FFI

## Trigger

Use for native MLX shim, Rust FFI wrappers, CMake/build.rs integration, unsafe Rust, or ABI changes.

## Rules

- Keep the ABI narrow and C-compatible.
- Rust sees opaque handles only.
- No C++ exceptions cross the ABI.
- No Rust panic crosses the ABI.
- Add lifecycle tests for every handle.
- Document ownership transfer and thread-safety assumptions.

## Checklist

- [ ] Header updated in `native/gemma4_mlx/include/`.
- [ ] Rust wrapper updated in `crates/gemma4d-ffi`.
- [ ] Build works through Cargo.
- [ ] Null/error/lifecycle tests pass.
- [ ] Unsafe blocks document invariants.
