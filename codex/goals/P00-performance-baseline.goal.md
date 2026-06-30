# P00 — Performance Baseline and Measurement Hygiene

```text
/goal Establish a trustworthy performance baseline for Helios. Add or update benchmark/reporting code that separates model load time, prefill time, per-token decode latency, total wall time, MLX active/cache/peak memory where available, process RSS, git SHA, model manifest/revision/hash, command line, and relevant environment variables for the current helper-backed path. Verify by running the existing M12 real-target matrix plus the new baseline harness on 1K/4K/8K/16K and producing benchmarks/out/P00-performance-baseline/{records.jsonl,summary.json,report.md}. Keep make verify green. Do not optimize hot-path code in this goal; only instrumentation, harnesses, and docs. If local model artifacts are unavailable, produce a dry-run harness plus an explicit blocker report listing the exact local commands required.
```

## Outcome

A reproducible performance baseline that distinguishes load, prefill, decode, memory, and command overhead.

## Verification surface

- `benchmarks/out/P00-performance-baseline/records.jsonl`
- `benchmarks/out/P00-performance-baseline/summary.json`
- `benchmarks/out/P00-performance-baseline/report.md`
- `make verify`
- Existing M12 matrix still runs or its blocker is documented.

## Boundaries

No native graph optimization, no MTP changes, no cache implementation changes.

## Completion rule

Mark this goal complete only when the evidence artifacts exist and the verification commands have been run, or when the goal is blocked with a blocker report that lists exact commands attempted, observed output, and the next required input.

## Suggested subagents

- `codebase-mapper` for read-only mapping.
- `performance-analyst` for benchmark and variance review.
- `test-verifier` for final build/test/lint verification.
