# XR26 - Native greedy-logit gather A/B

## Outcome

Determine whether native one-token decode can preserve greedy token and
greedy-logit parity while replacing the extra `max(logits)` reduction with a
gather of `logits[argmax(logits)]`.

## Scope

- Baseline: `native_decode_eval_per_layer` from the XR06 decode-tail harness.
- Candidate: `native_decode_gather_greedy_logit` with
  `GEMMA4D_EXPERIMENTAL_NATIVE_GATHER_GREEDY_LOGIT=1`.
- Workload source: `benchmarks/workloads/real-contexts/workloads.jsonl`.
- Start with a smoke on `chat_short_1k_001`.
- Expand only if smoke preserves token/logit correctness.

## Required work

1. Keep default native decode behavior unchanged when the env flag is absent.
2. Add only a default-off native diagnostic path and benchmark variant.
3. Record exact commands, generated files, git SHA, deterministic workload seeds,
   context lengths, generated-token counts, p50/p95/p99 decode latency, peak MLX
   memory, active KV bytes, and blockers.
4. Compare generated tokens and greedy logits against the same-code per-layer
   baseline.
5. Keep server defaults and public API behavior unchanged.

## Acceptance gates

- Candidate generated tokens match baseline for every measured record.
- Candidate greedy logits match baseline within the XR06 tolerance.
- Candidate improves steady decode p50 or p95 by at least `5%` on the smoke
  workload without increasing peak MLX memory or active KV bytes.
- If smoke passes, run a broader real-context A/B with at least the XR06 default
  workload set.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Non-goals

- Do not enable the gather path by default.
- Do not change MTP, sampling, server defaults, adapters, or KV compression.
- Do not remove greedy-logit fields from records or public step results.

## Required artifacts

```text
benchmarks/out/XR26-native-greedy-logit-gather/smoke/records.jsonl
benchmarks/out/XR26-native-greedy-logit-gather/smoke/summary.json
benchmarks/out/XR26-native-greedy-logit-gather/smoke/report.md
benchmarks/out/XR26-native-greedy-logit-gather/smoke/blockers.md
benchmarks/out/XR26-native-greedy-logit-gather/smoke/decision.md
benchmarks/out/XR26-native-greedy-logit-gather/followup-chat-short-1k/records.jsonl
benchmarks/out/XR26-native-greedy-logit-gather/followup-chat-short-1k/summary.json
benchmarks/out/XR26-native-greedy-logit-gather/followup-chat-short-1k/report.md
benchmarks/out/XR26-native-greedy-logit-gather/followup-chat-short-1k/blockers.md
benchmarks/out/XR26-native-greedy-logit-gather/followup-chat-short-1k/decision.md
```

## Result

Decision: `reject_candidate`.

The one-trial smoke on `chat_short_1k_001` preserved token/logit correctness
but was low-N. The three-trial follow-up with `64` generated tokens also passed
all token/logit correctness checks and held memory flat, but did not satisfy the
XR06 tail gate:

- Baseline raw p50/p95/p99: `86.244/88.198/324.516 ms`.
- Candidate raw p50/p95/p99: `86.001/87.314/321.880 ms`.
- Candidate p50 regression: `-0.282%`.
- Candidate p95 improvement: `1.003%`.
- Candidate p99 improvement: `0.813%`, below the `15%` gate.
- Peak MLX: `7.321 GB` for both variants.
- Active KV: `353353728` bytes for both variants.
- Blockers: none.

No broader holdout was run because the follow-up failed the performance gate.
The env flag remains default-off.

## Completion rule

Stop when the gather candidate has same-code A/B evidence against the per-layer
baseline, or when compile/runtime/correctness/performance blockers explain why
it should not continue.
