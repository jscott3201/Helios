# Compliance Review: M01 Native MLX Loader and FFI Smoke

## Scope

- Spec/contract: `milestones/M01-native-mlx-loader.md`, `spec/03-rust-mlx-ffi-contract.md`
- Version/date: 2026-06-30
- Included areas: C ABI smoke surface, opaque handles, lifecycle/error functions, Rust safe wrappers, Cargo build.rs/CMake integration, local MLX diagnostics, verification gates.
- Excluded areas: real MLX model loading, prefill/decode execution, MTP, adapters, tokenizer parity, serving API, benchmarks/profiling.

## Traceability Matrix

| Req ID | Requirement summary | Strength | Implementation evidence | Test evidence | Status | Gap |
|---|---|---|---|---|---|---|
| M01-T01 | Implement `gemma4_runtime_version` and error retrieval. | Must | `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/runtime.cc`, `crates/gemma4d-ffi/src/lib.rs` | `runtime_version_reports_smoke_backend`, `raw_null_pointers_return_invalid_argument` in `cargo test -p gemma4d-ffi` | Complete | None |
| M01-T02 | Implement opaque handle pattern and free functions. | Must | `Gemma4Target`, `Gemma4KvCache`, `gemma4_free_target`, `gemma4_kv_free` in native header/runtime; `Target`/`KvCache` `Drop` wrappers in Rust | `target_and_kv_lifecycle_work_without_model_loading`, raw null-pointer tests | Complete | None |
| M01-T03 | Add Rust `gemma4d-ffi` safe wrappers. | Must | `crates/gemma4d-ffi/src/lib.rs` exposes `runtime_version`, `Target`, `LoadConfig`, `KvCache`, `KvPolicy`, and smoke execution wrappers | `cargo test -p gemma4d-ffi` | Complete | None |
| M01-T04 | Wire CMake/native build through Rust `build.rs`. | Must | `crates/gemma4d-ffi/build.rs`, `crates/gemma4d-ffi/Cargo.toml`, `native/gemma4_mlx/CMakeLists.txt` | `cargo test -p gemma4d-ffi`; native library found under `target/debug/build/.../libgemma4_mlx.a` | Complete | None |
| M01-T05 | Add null-pointer and lifecycle tests. | Must | Tests in `crates/gemma4d-ffi/src/lib.rs` | `cargo test -p gemma4d-ffi` passed 5 tests | Complete | None |
| M01-T06 | Add optional MLX discovery diagnostics command. | Must | `scripts/mlx-diagnostics.sh`, `Makefile` target `native-mlx-diagnostics` | `./scripts/mlx-diagnostics.sh` reported missing MLX CMake package on this host | Complete | None |
| M01-A01 | FFI smoke tests pass without loading a model. | Acceptance | Native runtime allocates smoke handles only; no MLX/model path used by tests | `cargo test -p gemma4d-ffi` passed | Complete | None |
| M01-A02 | Native library builds through Cargo on a configured Mac. | Acceptance | `build.rs` runs CMake and links `gemma4_mlx` | `cargo test -p gemma4d-ffi`; `find target/debug/build -path '*libgemma4_mlx.a'` | Complete | None |
| M01-A03 | Error paths are covered. | Acceptance | Native null checks and error buffer; Rust `Error` mapping | invalid config, null pointer, and unsupported execution tests | Complete | None |
| M01-A04 | Unsafe blocks are documented. | Acceptance | Safety comments before each Rust unsafe block in `crates/gemma4d-ffi/src/lib.rs` | `cargo clippy --workspace --all-targets --all-features -- -D warnings` passed | Complete | None |

## High-Risk Gaps

No blocker, high, or medium compliance gaps were found for M01.

## Coverage Summary

- Implemented and tested: C ABI smoke functions, last-error path, target/KV opaque handles, Rust wrappers, build.rs/CMake integration, null/error/lifecycle tests, optional MLX diagnostics command.
- Implemented but not part of the acceptance gate: `scripts/mlx-diagnostics.sh`; it currently reports MLX is not configured as a CMake package on this host.
- Not implemented: real MLX model load and real prefill/decode execution, intentionally deferred to later milestones.
- Ambiguous / needs owner decision: none for M01.

## Next Work Items

1. Start M02 only after M01 is committed, pushed, and CI has verified the native Cargo build.
