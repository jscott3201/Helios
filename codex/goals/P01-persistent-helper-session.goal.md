# P01 — Persistent Helper Session Benchmark and Load Amortization

```text
goal Add a persistent helper/session benchmark path that loads the Gemma 4 12B 4-bit target once and runs multiple 1K/4K/8K/16K workloads in one process, with explicit reset between cases. Compare this warm-session path against the M12 cold CLI path and produce benchmarks/out/P01-persistent-helper-session/{records.jsonl,summary.json,report.md}. The report must quantify model load amortization, prefill/decode timing, per-token decode latency distribution, and memory growth across repeated runs. Keep helper-backed generation output stable and keep make verify green.
```

## Outcome

A measured answer to whether current UX latency is dominated by model load, prefill, decode, or process/cargo overhead.

## Verification surface

- Warm-session benchmark artifacts.
- Cold-vs-warm comparison table.
- RSS/MLX memory growth check over repeated cases.
- `make verify`.

## Boundaries

Use the existing helper-backed target. Do not implement native graph optimization.

## Completion rule

Mark this goal complete only when the evidence artifacts exist and the verification commands have been run, or when the goal is blocked with a blocker report that lists exact commands attempted, observed output, and the next required input.

## Suggested subagents

- `codebase-mapper` for read-only mapping.
- `performance-analyst` for benchmark and variance review.
- `test-verifier` for final build/test/lint verification.
