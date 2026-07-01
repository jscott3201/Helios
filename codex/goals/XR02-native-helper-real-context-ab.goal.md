# XR02 — Native vs helper A/B on real contexts

## Outcome

Re-measure helper-backed and native-incremental backends on the real-context corpus. Decide where native should be the default, opt-in, or rejected.

## Baseline

Helper/default generation and current P00/P01/P02 evidence.

## Candidate

`GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1` native incremental path.

## Required work

1. Use XR01 harness with real workloads at 1K/4K/8K/16K.
2. Compare generated tokens and greedy logits where exact parity is expected.
3. Record p50/p95/p99 decode, prefill, total, active KV bytes, peak MLX GB.
4. Separate first-token/prefill from steady-state decode.
5. Include at least one structured output workload and one code-review workload.
6. Produce a per-family recommendation: helper default, native opt-in, native default candidate, or blocked.

## Acceptance gates

- No token mismatch for deterministic greedy comparison unless explicitly documented as expected drift.
- Native p95 decode improves or remains within 5% while reducing memory or unlocking cache behavior.
- No tiny16 memory cliff.


## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR02-native-helper-real-context-ab/records.jsonl
benchmarks/out/XR02-native-helper-real-context-ab/summary.json
benchmarks/out/XR02-native-helper-real-context-ab/report.md
benchmarks/out/XR02-native-helper-real-context-ab/blockers.md
benchmarks/out/XR02-native-helper-real-context-ab/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
