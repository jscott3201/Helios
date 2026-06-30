# P09 — Real LoRA/QLoRA Adapter Hot Path

```text
Move adapters from registry/control-plane fixtures into real inference for one trusted local rank-16 LoRA adapter. Implement adapter tensor loading and LoRA delta application for the supported Gemma 4 target modules, with adapter-aware KV namespace separation. Verify against a reference implementation or deterministic adapter fixture, measure load latency, resident bytes, hotswap latency, and generation latency with and without the adapter. Produce benchmarks/out/P09-real-lora-adapter/{records.jsonl,summary.json,report.md}. Keep MTP disabled when a standard adapter is active.
```

## Outcome

Dynamic expert adapters become a real runtime feature.

## Verification surface

- Adapter output differs from base in expected way.
- Wrong base/tokenizer/template/hash rejects.
- Adapter-aware cache namespace tests.
- Load/hotswap/residency benchmarks.

## Boundaries

One adapter active per request. No remote adapter loading. No aLoRA yet.

## Completion rule

Mark this goal complete only when the evidence artifacts exist and the verification commands have been run, or when the goal is blocked with a blocker report that lists exact commands attempted, observed output, and the next required input.

## Suggested subagents

- `codebase-mapper` for read-only mapping.
- `performance-analyst` for benchmark and variance review.
- `test-verifier` for final build/test/lint verification.
