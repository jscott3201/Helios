# Compliance Review: M00 Repository Bootstrap

## Scope

- Spec/contract: `milestones/M00-repo-bootstrap.md`
- Version/date: 2026-06-30
- Included areas: repository skeleton, Rust workspace, native CMake skeleton, scripts, CI, docs/artifact layout, verification gates.
- Excluded areas: inference implementation, MLX model loading, tokenizer parity, serving API, TUI implementation, benchmarks/profiling.

## Traceability Matrix

| Req ID | Requirement summary | Strength | Implementation evidence | Test evidence | Status | Gap |
|---|---|---|---|---|---|---|
| M00-T01 | Create workspace directories and placeholder crates. | Must | `Cargo.toml`, `crates/gemma4d-*` | `cargo metadata --format-version=1 --no-deps`; `cargo test --workspace --all-targets --all-features` | Complete | None |
| M00-T02 | Add `rust-toolchain.toml` pinned to Rust 1.95.0. | Must | `rust-toolchain.toml` | `cargo metadata --format-version=1 --no-deps` reads workspace with `rust_version=1.95.0`. | Complete | None |
| M00-T03 | Add native `native/gemma4_mlx` CMake skeleton. | Must | `native/gemma4_mlx/CMakeLists.txt`, `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/smoke.cc` | `cmake -S native/gemma4_mlx -B target/native-smoke`; `make verify` native build | Complete | None |
| M00-T04 | Add `justfile` or `Makefile` for common commands. | Must | `Makefile` | `make verify` | Complete | None |
| M00-T05 | Add CI/local scripts for format, clippy, tests, native build smoke. | Must | `scripts/verify.sh`, `scripts/native-smoke.sh`, `.github/workflows/ci.yml` | `make verify` | Complete | None |
| M00-T06 | Add benchmark output directories and `.gitignore` rules. | Must | `benchmarks/`, `benchmarks/out/M00/.gitkeep`, `.gitignore` | `git check-ignore -v target benchmarks/out/example.log artifacts/example.log` | Complete | None |
| M00-T07 | Add initial decision-record template. | Must | `docs/decisions/0000-template.md` | File inspection | Complete | None |
| M00-A01 | `cargo metadata` succeeds. | Acceptance | `Cargo.toml`, crate manifests | `cargo metadata --format-version=1 --no-deps` passed | Complete | None |
| M00-A02 | `cargo test` passes for placeholder crates. | Acceptance | `crates/gemma4d-*/src/lib.rs` | `cargo test --workspace --all-targets --all-features` passed | Complete | None |
| M00-A03 | CMake configures or fails with documented MLX dependency message. | Acceptance | `native/gemma4_mlx/CMakeLists.txt`, `native/gemma4_mlx/README.md` | `cmake -S native/gemma4_mlx -B target/native-smoke` passed | Complete | None |
| M00-A04 | `just verify` or equivalent runs format/lint/test smoke commands. | Acceptance | `Makefile`, `scripts/verify.sh` | `make verify` passed | Complete | None |
| M00-A05 | No full model download required. | Acceptance | No model paths/artifacts added; native smoke does not require MLX by default | Command evidence in `docs/evidence/M00.md` | Complete | None |

## High-Risk Gaps

No blocker, high, or medium compliance gaps were found for M00.

## Coverage Summary

- Implemented and tested: M00 bootstrap workspace, scripts, native smoke, docs/artifact layout, ignore rules.
- Implemented but not tested: GitHub Actions workflow is a skeleton and has not yet run remotely.
- Not implemented: Later milestone behavior, intentionally out of scope.
- Ambiguous / needs owner decision: None for M00.

## Next Work Items

1. Start M01 native MLX loader only after this M00 commit is merged/pushed.
