# XR13 - Novel Metal/KV optimization exploration

## Outcome

Explore high-risk/high-upside optimization ideas behind feature flags, with a
no-go report as an acceptable outcome.

## Candidate tracks

1. Fused compressed-domain attention for old global/full-attention KV.
2. Planar/IsoQuant-style K-only global-prefix compression.
3. TurboQuant-inspired compressed score estimation prototype.
4. MLX custom Metal kernel sketch for decode L=1 attention over compressed K.
5. Active compressed KV decode that avoids BF16 decompression on import.

## Required work

1. Start with a written design and A/B hypothesis.
2. Prototype the smallest measurable slice, preferably a microbenchmark or
   isolated attention kernel.
3. Compare against BF16/q8 baseline from XR09.
4. Measure correctness, memory, and latency.
5. Keep all code behind an explicit feature flag or prototype directory.

## Acceptance gates

- No default path changes.
- If no prototype is implemented, produce a no-go report explaining API/kernel
  blockers.
- If prototype exists, it must include correctness and memory tests before speed
  claims.

## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR13-novel-metal-kv-exploration/records.jsonl
benchmarks/out/XR13-novel-metal-kv-exploration/summary.json
benchmarks/out/XR13-novel-metal-kv-exploration/report.md
benchmarks/out/XR13-novel-metal-kv-exploration/blockers.md
benchmarks/out/XR13-novel-metal-kv-exploration/decision.md
```

## Completion rule

Stop only when the decision file exists with raw evidence, or `blockers.md`
explains why blocked.
