# XR01 — Real-context A/B benchmark harness

## Outcome

Build a reusable A/B benchmark harness that runs Helios variants against the XR00 workload corpus and emits standardized records, summaries, reports, blockers, and decisions.

## Required work

1. Add a benchmark example such as `crates/gemma4d-bench/examples/xr01_real_context_ab.rs`.
2. Support variants through explicit config: `baseline`, `candidate`, backend mode, env vars, cache flags, MTP flags, and adapter flags.
3. Load `benchmarks/workloads/real-contexts/workloads.jsonl`.
4. Run at least a dry-run/smoke subset without requiring the 12B model.
5. Run real model trials when artifacts are available.
6. Record p50/p95/p99 decode latency, prefill, total, peak memory, active KV bytes, output token ids, and correctness gate results.
7. Write a reusable report function and decision file.

## Verification surface

- Existing P00-P10 harnesses still compile.
- XR01 dry-run passes in CI/offline conditions.
- Real-run mode is failure-closed with a clear blocker if artifacts are missing.

## Decision

`accept_candidate` when at least one dry-run and one model-available command path are documented and the evidence schema is stable.


## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR01-real-context-ab-harness/records.jsonl
benchmarks/out/XR01-real-context-ab-harness/summary.json
benchmarks/out/XR01-real-context-ab-harness/report.md
benchmarks/out/XR01-real-context-ab-harness/blockers.md
benchmarks/out/XR01-real-context-ab-harness/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
