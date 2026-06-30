# M00 — Repository Bootstrap

## Goal

Create the Rust/native workspace, toolchain pin, CI/test skeleton, docs directories, and benchmark artifact layout.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Create workspace directories and placeholder crates.
- [ ] Add `rust-toolchain.toml` pinned to Rust 1.95.0.
- [ ] Add native `native/gemma4_mlx` CMake skeleton.
- [ ] Add `justfile` or `Makefile` for common commands.
- [ ] Add CI/local scripts for format, clippy, tests, native build smoke.
- [ ] Add benchmark output directories and `.gitignore` rules.
- [ ] Add initial decision-record template.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M00/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] `cargo metadata` succeeds.
- [ ] `cargo test` passes for placeholder crates.
- [ ] `cmake -S native/gemma4_mlx -B target/native-smoke` configures or fails with documented MLX dependency message.
- [ ] `just verify` or equivalent runs format/lint/test smoke commands.
- [ ] No full model download required.

## Recommended Codex goal

Use `codex/goals/M00-repo-bootstrap.goal.md`.

## Recommended skills

- `$gemma4d-milestone-execution`
- `$spec-contract-compliance-review`
- `$performance-ab-benchmark-review` when this milestone touches runtime performance
- milestone-specific project skill as applicable

## Blocked stop condition

If a required external dependency, model artifact, MLX API, or machine capability is unavailable, stop with:

1. attempted paths,
2. command/source evidence,
3. minimal repro or diagnostic,
4. next input required.

## TUI bootstrap note

Optionally create an empty/stub `crates/gemma4d-tui` crate and shared provider/DTO placeholder during workspace bootstrap if it helps workspace shape. Do not implement the real Ratatui app in M00; the first real TUI milestone is M05.
