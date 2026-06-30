# P03 — Native Graph Correctness and Bottleneck Triage

```text
goal Triage the hand-written native Gemma 4 graph before optimizing it. Run the smallest feasible native graph probes with GEMMA4D_USE_NATIVE_GRAPH=1, compare helper-backed and native outputs on tokenizer-controlled prompts, and produce a claim inventory separating confirmed parity, numerical drift, unsupported ops, memory cliffs, and measured hotspots. Add only diagnostic instrumentation required for the triage. Produce benchmarks/out/P03-native-graph-triage/{report.md,records.jsonl,blockers.md}. Do not switch defaults to the native graph in this goal.
```

## Outcome

A current go/no-go map for native graph optimization work.

## Verification surface

- Native vs helper output/logit comparison for small tokenizer-controlled prompts.
- Peak memory and timing for each probe.
- Explicit blocker list if parity fails.
- Diagnostic code gated to measurement output only.

## Boundaries

- No default path change.
- No broad refactor.
- No optimization beyond narrow diagnostic instrumentation.
- No benchmark claim unless parity or blocker is documented.

## Completion rule

Mark this goal complete only when the evidence artifacts exist and the verification commands have been run, or when the goal is blocked with a blocker report that lists exact commands attempted, observed output, and the next required input.

## Suggested subagents

- `codebase-mapper` for read-only mapping.
- `performance-analyst` for benchmark and variance review.
- `test-verifier` for final build/test/lint verification.
