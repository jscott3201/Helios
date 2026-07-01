# XR07 - Prefix cache A/B on real reuse patterns

## Outcome

Measure RAM prefix cache value on realistic repeated-prefix conversations and tool loops, not only exact synthetic restore probes.

## Required work

1. Build workloads where a long repo/document prefix is reused with small suffix edits.
2. Compare fresh prefill vs RAM prefix restore for 4K/8K/16K.
3. Measure hit rate, warm TTFT, restore latency, continued decode parity, active KV bytes, and cache memory residency.
4. Include adapter namespace isolation case and base-only reuse case.
5. Decide whether RAM prefix cache should be enabled by default for tiny16 and under what cap.

## Acceptance gates

- Restored continuation equals fresh continuation for deterministic greedy.
- Warm TTFT improvement remains meaningful after including lookup/import overhead.
- No unsafe cross-adapter or cross-cache-mode reuse.

## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR07-prefix-cache-real-reuse-ab/records.jsonl
benchmarks/out/XR07-prefix-cache-real-reuse-ab/summary.json
benchmarks/out/XR07-prefix-cache-real-reuse-ab/report.md
benchmarks/out/XR07-prefix-cache-real-reuse-ab/blockers.md
benchmarks/out/XR07-prefix-cache-real-reuse-ab/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
