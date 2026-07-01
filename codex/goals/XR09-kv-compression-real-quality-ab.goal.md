# XR09 - KV compression real-quality A/B

## Outcome

Extend P08 compression evidence from deterministic probes to real-context workloads, and decide the next compression mode to pursue.

## Required work

1. Re-run BF16/q8/q4 prefix payload compression on real workload families.
2. Record greedy agreement, logit delta, optional top-k agreement, warm restore latency, payload size, and active memory.
3. If q4 fails, create a concise failure analysis by family and context length.
4. Keep active compressed decode disabled unless a separate feature-flag prototype proves correctness.
5. Produce a recommended next candidate: q8 default for SSD payload, q4 rejected, Planar/Iso research, or no-go.

## Acceptance gates

- q8 or any candidate must pass deterministic continued-decode gates on real contexts.
- q4 cannot be promoted unless greedy agreement failures are resolved.
- Active memory claims require active compressed decode, not just compressed storage.

## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR09-kv-compression-real-quality-ab/records.jsonl
benchmarks/out/XR09-kv-compression-real-quality-ab/summary.json
benchmarks/out/XR09-kv-compression-real-quality-ab/report.md
benchmarks/out/XR09-kv-compression-real-quality-ab/blockers.md
benchmarks/out/XR09-kv-compression-real-quality-ab/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
