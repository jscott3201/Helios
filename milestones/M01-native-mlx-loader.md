# M01 — Native MLX Loader and FFI Smoke

## Goal

Build the narrow C ABI, Rust wrappers, version query, handle lifecycle, and native smoke tests.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Implement `gemma4_runtime_version` and error retrieval.
- [ ] Implement opaque handle pattern and free functions.
- [ ] Add Rust `gemma4d-ffi` safe wrappers.
- [ ] Wire CMake/native build through Rust `build.rs`.
- [ ] Add null-pointer and lifecycle tests.
- [ ] Add optional MLX discovery diagnostics command.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M01/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] FFI smoke tests pass without loading a model.
- [ ] Native library builds through Cargo on a configured Mac.
- [ ] Error paths are covered.
- [ ] Unsafe blocks are documented.

## Recommended Codex goal

Use `codex/goals/M01-native-mlx-loader.goal.md`.

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
