# M03 — Greedy Text Inference

## Goal

Load the target Gemma 4 12B MLX 4-bit model and run text-only greedy prefill/decode through native MLX.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Implement target model load in native shim.
- [ ] Implement prefill and decode-one FFI calls.
- [ ] Implement greedy sampler in Rust.
- [ ] Add CLI `gemma4d generate` for local smoke tests.
- [ ] Record memory for 1K/4K/8K prompts on tiny16.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M03/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] Short prompts generate deterministic token sequences.
- [ ] Chunked prefill is either implemented or explicitly deferred with tests marked pending.
- [ ] Benchmark report records TTFT/decode/memory for at least 1K and 4K.
- [ ] Failures are graceful under missing model path.

## Recommended Codex goal

Use `codex/goals/M03-greedy-text-inference.goal.md`.

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
