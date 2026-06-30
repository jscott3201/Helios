# M06 — MTP Speculative Decoding

## Goal

Implement Gemma 4 MTP assistant loading and exact greedy speculative decoding with rollback.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Add drafter load FFI function.
- [ ] Expose last target hidden state/shared views needed by drafter.
- [ ] Implement draft block size 1, then 2.
- [ ] Implement verify/accept/rollback.
- [ ] Add MTP exactness tests against non-MTP greedy.
- [ ] Add MTP metrics and auto-disable.
- [ ] Update TUI MTP placeholder/provider payload so acceptance, rollback, and auto-disable status are visible when enabled.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M06/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] MTP block size 1 exactness passes.
- [ ] MTP block size 2 exactness passes on fixture set or auto-disables with evidence.
- [ ] Acceptance metrics are recorded.
- [ ] Adapters and compressed active KV remain disabled for MTP in this milestone.

## Recommended Codex goal

Use `codex/goals/M06-mtp-speculative-decoding.goal.md`.

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
