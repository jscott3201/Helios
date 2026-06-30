# M04 — Reference Parity and Benchmark Harness

## Goal

Create a repeatable harness comparing gemma4d against MLX reference and llama.cpp/HF baselines where available.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Add prompt corpus and benchmark runner.
- [ ] Add baseline command capture for MLX Python and llama.cpp when configured.
- [ ] Add token sequence diff tooling.
- [ ] Add benchmark JSONL writer.
- [ ] Add report generator from raw outputs.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M04/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] At least one reference path can be configured and compared.
- [ ] Token diffs are readable.
- [ ] Benchmark records include environment and model revisions.
- [ ] Inconclusive comparisons are labelled as such, not passed.

## Recommended Codex goal

Use `codex/goals/M04-reference-parity-harness.goal.md`.

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
