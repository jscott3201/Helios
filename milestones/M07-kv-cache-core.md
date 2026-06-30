# M07 — In-Memory KV Cache Core

## Goal

Implement logical KV block metadata, RAM prefix cache, copy-on-write conversation forks, and exact restore tests.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Create `gemma4d-kv` block/key types.
- [ ] Add native export/import for RAM prefix blocks or native-managed logical handles.
- [ ] Implement cache namespace hashing.
- [ ] Implement RAM LRU with memory budget.
- [ ] Add restore-vs-fresh tests for 1K/4K/8K/16K.
- [ ] Expose cache byte/accounting summaries through the provider DTO used by the TUI cache page.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M07/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] Fresh prefill and RAM-restored logits/tokens match for same mode.
- [ ] Wrong model/template/hash blocks are rejected.
- [ ] Memory accounting is visible.
- [ ] No SSD dependency yet.

## Recommended Codex goal

Use `codex/goals/M07-kv-cache-core.goal.md`.

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
