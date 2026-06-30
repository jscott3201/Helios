# M08 — SSD Cold Prefix Cache

## Goal

Persist inactive prefix KV blocks to SSD and restore matching prefixes before prefill.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Define persisted KV manifest and versioning.
- [ ] Implement block writer/reader and checksums.
- [ ] Add SSD index and eviction policy.
- [ ] Add restore-before-prefill path.
- [ ] Benchmark cold vs warm SSD TTFT.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M08/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] SSD-restored logits/tokens match fresh prefill for same mode.
- [ ] Corrupt block rejection test passes.
- [ ] Wrong namespace rejection test passes.
- [ ] Benchmark shows raw read/write bytes and TTFT comparison.
- [ ] No mid-decode SSD fetch.

## Recommended Codex goal

Use `codex/goals/M08-ssd-prefix-cache.goal.md`.

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
