# P04 - Incremental Native KV Decode MVP

```text
goal Implement incremental native KV decode for the hand-written Gemma 4 native graph. Prefill must materialize per-layer KV state, and decode_one must process only the next token using cached K/V with Gemma 4 sliding-window truncation and full-attention/global-layer handling. Verify greedy parity against the helper-backed path on small tokenizer-controlled prompts and show decode latency no longer grows linearly with context. Produce benchmarks/out/P04-incremental-native-kv/{records.jsonl,summary.json,report.md}. Keep helper fallback.
```

## Outcome

The opt-in native graph has a measured incremental decode path that preserves
helper parity for the selected greedy text probes and records active KV memory.

## Verification Surface

- Helper vs native generated token comparison for tokenizer-controlled prompts.
- Helper vs native greedy logit comparison within the configured tolerance.
- Native decode per-token latency by context length.
- Native active KV bytes by context length.
- Explicit blocker report if parity, latency, or runtime checks fail.

## Boundaries

- No default path switch.
- No MTP integration.
- No multimodal path.
- No adapter path changes.
- No broad multi-model abstractions.
- Helper fallback must remain available.

## Completion Rule

Mark this goal complete only when the evidence artifacts exist and the
verification commands have been run, or when the goal is blocked with a blocker
report that lists exact commands attempted, observed output, and the next
required input.

## Suggested Subagents

- `performance-analyst` for latency and variance interpretation.
- `gemma4_correctness_reviewer` for token/logit/cache parity review.
- `test-verifier` for final build/test/lint verification.
