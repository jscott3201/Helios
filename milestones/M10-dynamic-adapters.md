# M10 — Dynamic LoRA/QLoRA Adapters

## Goal

Import, validate, load, route, and unload trusted standard LoRA/QLoRA adapters with adapter-aware KV cache keys.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Implement adapter manifest parser and schema validation.
- [ ] Import PEFT `adapter_config.json` + `adapter_model.safetensors`.
- [ ] Implement one active adapter per request.
- [ ] Add local adapter registry and trusted path policy.
- [ ] Add adapter-aware KV namespace tests.
- [ ] Add load/unload/pin endpoints or CLI commands.
- [ ] Connect adapter registry summaries to the TUI adapter page/provider model.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M10/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] Valid local adapter loads.
- [ ] Wrong base/tokenizer/template adapters rejected.
- [ ] Base output unchanged when adapter disabled.
- [ ] Adapter cache cross-contamination tests pass.
- [ ] MTP disabled with adapters unless separately verified.

## Recommended Codex goal

Use `codex/goals/M10-dynamic-adapters.goal.md`.

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
