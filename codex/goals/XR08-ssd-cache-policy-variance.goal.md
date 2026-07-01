# XR08 - SSD cache policy and variance A/B

## Outcome

Decide whether SSD prefix cache should remain disabled, become opt-in, or be enabled for specific tiny16 workloads by measuring variance, IO cost, and warm prefix reuse value.

## Required work

1. Run repeated SSD restore trials over realistic prefix-reuse workloads.
2. Compare storage formats/policies already available: BF16 payload, q8 payload if available, compression off/on.
3. Measure metadata IO, payload IO, restore latency, warm TTFT, SSD bytes, corruption rejection, and memory pressure.
4. Test cache admission thresholds: minimum prefix tokens and max cache size.
5. Never allow mid-decode SSD fetch as part of this goal.

## Acceptance gates

- Restore correctness passes.
- p50 and p95 warm TTFT beat fresh prefill by a goal-defined margin at 8K/16K.
- Variance and IO cost are documented.
- Decision is profile-gated.

## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR08-ssd-cache-policy-variance/records.jsonl
benchmarks/out/XR08-ssd-cache-policy-variance/summary.json
benchmarks/out/XR08-ssd-cache-policy-variance/report.md
benchmarks/out/XR08-ssd-cache-policy-variance/blockers.md
benchmarks/out/XR08-ssd-cache-policy-variance/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
