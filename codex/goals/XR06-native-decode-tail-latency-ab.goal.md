# XR06 - Native decode tail-latency A/B

## Outcome

Reduce or explain native decode p95/p99 outliers using real workloads and token-level traces.

## Required work

1. Reproduce P04-style raw decode latency spikes on real workloads.
2. Record per-token latency with token id, position, layer/eval markers where feasible, active KV bytes, peak memory, and whether MLX synchronization happened.
3. A/B candidate fixes such as fewer per-token evals, preallocated buffers, concat/slice alternatives, or avoiding unnecessary `eval` after KV append.
4. Preserve p50 and correctness while targeting p95/p99.

## Acceptance gates

- p95 or p99 improves at least 15% on a workload with reproduced tail latency.
- No p50 regression >5% across other workloads.
- Correctness and memory gates pass.

## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR06-native-decode-tail-latency-ab/records.jsonl
benchmarks/out/XR06-native-decode-tail-latency-ab/summary.json
benchmarks/out/XR06-native-decode-tail-latency-ab/report.md
benchmarks/out/XR06-native-decode-tail-latency-ab/blockers.md
benchmarks/out/XR06-native-decode-tail-latency-ab/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
