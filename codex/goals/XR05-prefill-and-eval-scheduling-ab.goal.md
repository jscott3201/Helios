# XR05 - Prefill and MLX eval scheduling A/B

## Outcome

Find prefill speed and memory improvements for native/helper paths by A/B testing chunk size, eval placement, cache clearing, and MLX synchronization boundaries on real contexts.

## Required work

1. Measure current prefill across 4K/8K/16K real workloads.
2. Compare helper prefill chunk sizes where applicable.
3. Compare native eval strategies: current per-layer/tensor eval, fewer evals, explicit eval groups, and cache clearing policies.
4. Keep output token/logit gates intact.
5. Record MLX peak memory, RSS, prefill tok/s, TTFT, and variance.

## Candidate knobs

- helper `prefill_chunk_tokens` values: 512, 1024, 2048, 4096.
- native eval grouping: current, per-layer, end-of-prefill, selective KV eval.
- `mx.clear_cache`/MLX cache clearing policy where accessible.

## Acceptance gates

- No correctness regression.
- Prefill p50 improves at least 10% on one real workload family or memory peak improves at least 5% without p95 regression.

## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR05-prefill-and-eval-scheduling-ab/records.jsonl
benchmarks/out/XR05-prefill-and-eval-scheduling-ab/summary.json
benchmarks/out/XR05-prefill-and-eval-scheduling-ab/report.md
benchmarks/out/XR05-prefill-and-eval-scheduling-ab/blockers.md
benchmarks/out/XR05-prefill-and-eval-scheduling-ab/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
