---
name: helios-real-workload-ab
description: Build and run Helios real-context A/B benchmarks with deterministic workload manifests, correctness gates, and decision artifacts.
---

# Helios real-workload A/B skill

Use this skill when a task mentions Helios performance, A/B benchmarking, real workloads, benchmark corpus, repeated-prefix testing, or tiny16 performance decisions.

## Procedure

1. Read `BENCHMARKS.md`, `docs/xr-real-workload-methodology.md`, and `docs/xr-ab-methodology.md`.
2. Identify the current baseline and candidate before editing code.
3. Prefer repo-local real contexts over repeated-token prompts.
4. Record exact commands, git SHA, model identity, env vars, and output paths.
5. Use at least `records.jsonl`, `summary.json`, `report.md`, `blockers.md`, and `decision.md` for every Goal.
6. Do not accept performance gains without correctness and memory gates.
7. Mark low-trial or high-variance results honestly.

## Decision labels

Use exactly one:

- `accept_candidate`
- `reject_candidate`
- `keep_experimental`
- `needs_more_data`
- `blocked_with_evidence`
