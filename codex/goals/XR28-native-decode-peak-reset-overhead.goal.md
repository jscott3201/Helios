# XR28 - Native decode peak-reset overhead A/B

## Outcome

Determine whether resetting MLX peak-memory counters on every native one-token
decode step is measurable tail-latency overhead.

## Scope

- Baseline: `native_decode_eval_per_layer` from the XR06 decode-tail harness.
- Candidate: `native_decode_skip_peak_reset` with
  `GEMMA4D_EXPERIMENTAL_NATIVE_SKIP_DECODE_PEAK_RESET=1`.
- Workload source: `benchmarks/workloads/real-contexts/workloads.jsonl`.
- Start with `chat_short_1k_001`; expand only if correctness passes and the
  latency signal is meaningful.

## Required work

1. Keep default native decode behavior unchanged when the env flag is absent.
2. Add only a default-off diagnostic path and benchmark variant.
3. Do not change model math, KV state, public C ABI, MTP policy, or server
   defaults.
4. Record exact commands, generated files, git SHA, deterministic workload
   seeds, context lengths, generated-token counts, p50/p95/p99 decode latency,
   peak MLX memory, active KV bytes, and blockers.
5. Compare generated tokens and greedy logits against the same-code per-layer
   baseline.

## Acceptance gates

- Candidate generated tokens match baseline for every measured record.
- Candidate greedy logits match baseline within the XR06 tolerance.
- Candidate improves p95 or p99 decode latency by at least `15%` without a p50
  regression above `5%`.
- Candidate remains under the XR06 memory cliff.
- Peak-memory interpretation must note that skipping per-token reset changes the
  telemetry boundary from per-decode-step peak to prior accumulated peak.

## Non-goals

- Do not enable the skip path by default.
- Do not use this candidate to make memory improvement claims.
- Do not change prefill peak resets, block decode peak resets, MTP verifier
  telemetry, or cache snapshot metadata.

## Required artifacts

```text
benchmarks/out/XR28-native-decode-peak-reset-overhead/smoke-chat-short-1k/records.jsonl
benchmarks/out/XR28-native-decode-peak-reset-overhead/smoke-chat-short-1k/summary.json
benchmarks/out/XR28-native-decode-peak-reset-overhead/smoke-chat-short-1k/report.md
benchmarks/out/XR28-native-decode-peak-reset-overhead/smoke-chat-short-1k/blockers.md
benchmarks/out/XR28-native-decode-peak-reset-overhead/smoke-chat-short-1k/decision.md
```

## Result

Decision: `reject_candidate`.

The three-trial smoke on `chat_short_1k_001` passed token/logit correctness and
stayed under the memory cliff, but failed the XR06 tail-latency gate:

- Run ID: `xr06-1782904528-39938000`.
- Workload seed: `20260630`.
- Context: `1024/1024`.
- Generated tokens per run: `64`.
- Baseline raw p50/p95/p99: `85.764/87.220/274.589 ms`.
- Candidate raw p50/p95/p99: `85.887/88.075/322.142 ms`.
- Candidate p50 regression: `0.143%`.
- Candidate p95 improvement: `-0.981%`.
- Candidate p99 improvement: `-17.318%`.
- Peak MLX: `7.321 GB` for both variants.
- Active KV: `353353728` bytes for both variants.
- Blockers: none.

No broader holdout was run because skipping the decode peak reset did not
improve latency on the smoke workload. The env flag remains default-off.

## Completion rule

Stop when the skip-reset candidate has same-code A/B evidence against the
per-layer baseline, or when compile/runtime/correctness/performance blockers
explain why it should not continue.
