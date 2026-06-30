# P06 - Real RAM Prefix Cache with Native Handles

```text
goal Replace RAM prefix-cache observation-only fixtures with a real native prefix-cache path that can export/import exact in-memory KV state for repeated prefixes. Cache keys must include model, tokenizer, chat template, prompt hash, adapter identity, KV layout version, and cache mode. Verify that fresh-prefill logits match restored-prefix logits for the same cache mode, and measure warm RAM TTFT improvement at 4K/8K/16K. Produce benchmarks/out/P06-real-ram-prefix-cache/{records.jsonl,summary.json,report.md}. Wrong model/adapter/cache keys must reject cleanly.
```

## Outcome

RAM prefix cache provides real native prefill avoidance.

## Verification Surface

- Native snapshot export/import through the narrow C ABI.
- Fresh prefill greedy token/logit vs restored last-step greedy token/logit.
- One continued `decode_one` after restore vs the cold-cache continuation.
- Warm RAM TTFT improvement at 4K, 8K, and 16K.
- Wrong model, adapter, and cache-mode namespace rejection before native import.
- Cache hit, miss, restore-failure, and eviction metrics in artifacts.

## Boundaries

- RAM only.
- Text-only greedy inference.
- No SSD payload persistence.
- No adapter-active snapshot import beyond namespace rejection coverage.
- No compressed active KV enablement.

## Completion Rule

Mark this goal complete only when the evidence artifacts exist and the
verification commands have been run, or when the goal is blocked with a blocker
report that lists exact commands attempted, observed output, and the next
required input.

## Suggested Subagents

- `performance-analyst` for TTFT and variance interpretation.
- `gemma4_correctness_reviewer` for restored-logit and continued-decode parity.
- `security-reliability-reviewer` for namespace rejection and handle ownership.
- `test-verifier` for final build/test/lint verification.
