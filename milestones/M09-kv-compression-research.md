# M09 — KV Compression Research

## Goal

Evaluate MLX affine q8/q4 and Planar/Iso-style compressed prefix cache modes under Gemma 4 12B workloads.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Implement MLX affine q8/q4 prefix cache modes.
- [ ] Add compression metadata to manifest.
- [ ] Create Planar/Iso experiment interface behind feature flag.
- [ ] Run quality comparisons: logit cosine, greedy agreement, JSON/tool fixtures.
- [ ] Run memory/speed comparisons at 16K/32K and 64K if possible.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M09/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] BF16 fallback remains default.
- [ ] q8/q4 results include quality and memory deltas.
- [ ] Planar/Iso remains experimental unless all gates pass.
- [ ] Compression never silently changes cache namespace semantics.

## Recommended Codex goal

Use `codex/goals/M09-kv-compression-research.goal.md`.

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
