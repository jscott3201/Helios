# XR27 - Native chunked prefill A/B

## Outcome

Determine whether a default-off native chunked prefill path can reduce long
context prefill memory or TTFT without breaking same-code native correctness.

## Scope

- Baseline: `native_eval_per_layer` from the XR05 prefill/eval harness.
- Candidates:
  - `native_chunked_prefill_512` with
    `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS=512`.
  - `native_chunked_prefill_1024` with
    `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS=1024`.
- Workload source: `benchmarks/workloads/real-contexts/workloads.jsonl`.
- Start with `code_review_rust_4k_001`; expand only if correctness passes.

## Required work

1. Keep public C ABI and Rust wrapper behavior unchanged.
2. Keep default native prefill behavior unchanged when the env flag is absent.
3. Reuse private native block-decode internals only for the default-off chunked
   prefill path; do not lift the public `gemma4_decode_block` token cap.
4. Record exact commands, generated files, git SHA, deterministic workload
   seeds, context lengths, prefill p50/p95, peak MLX memory, active KV bytes,
   and blockers.
5. Compare output greedy token/logit against the same-code native baseline.

## Acceptance gates

- Candidate prefill token/logit correctness passes against `native_eval_per_layer`
  for every measured record.
- Candidate has at least three passed trials before promotion.
- Candidate improves p50 prefill by at least `10%` or peak MLX memory by at
  least `5%`.
- Candidate p95 prefill regression is no worse than `5%`.
- If the 4K smoke passes, expand to at least the XR05 default 4K/8K/16K
  workload set before recommending any policy change.

## Non-goals

- Do not enable chunked prefill by default.
- Do not change helper-backed prefill behavior.
- Do not change MTP, server defaults, adapters, KV compression, or public
  decode-block limits.

## Required artifacts

```text
benchmarks/out/XR27-native-chunked-prefill-ab/smoke-4k/records.jsonl
benchmarks/out/XR27-native-chunked-prefill-ab/smoke-4k/summary.json
benchmarks/out/XR27-native-chunked-prefill-ab/smoke-4k/report.md
benchmarks/out/XR27-native-chunked-prefill-ab/smoke-4k/blockers.md
benchmarks/out/XR27-native-chunked-prefill-ab/smoke-4k/decision.md
benchmarks/out/XR27-native-chunked-prefill-ab/followup-4k-512/records.jsonl
benchmarks/out/XR27-native-chunked-prefill-ab/followup-4k-512/summary.json
benchmarks/out/XR27-native-chunked-prefill-ab/followup-4k-512/report.md
benchmarks/out/XR27-native-chunked-prefill-ab/followup-4k-512/blockers.md
benchmarks/out/XR27-native-chunked-prefill-ab/followup-4k-512/decision.md
benchmarks/out/XR27-native-chunked-prefill-ab/sentinel-8k-512/records.jsonl
benchmarks/out/XR27-native-chunked-prefill-ab/sentinel-8k-512/summary.json
benchmarks/out/XR27-native-chunked-prefill-ab/sentinel-8k-512/report.md
benchmarks/out/XR27-native-chunked-prefill-ab/sentinel-8k-512/blockers.md
benchmarks/out/XR27-native-chunked-prefill-ab/sentinel-8k-512/decision.md
```

## Result

Decision: `reject_candidate`.

The 4K smoke passed correctness and showed the 512-token candidate as the only
promising chunk size. The three-trial 4K follow-up then failed the latency gate:

- `code_review_rust_4k_001`, seed `20260631`, context `4096/4096`.
- Baseline prefill p50/p95: `11756.074/11885.280 ms`.
- `native_chunked_prefill_512` prefill p50/p95:
  `11906.928/12706.910 ms`.
- p50 improvement: `-1.283%`.
- p95 regression: `6.913%`, above the `5%` limit.
- Peak MLX improved from `9.212 GB` to `7.458 GB`.
- Active KV stayed `402653184` bytes.
- Blockers: none.

An 8K low-N memory sentinel was run because long-context memory pressure was the
main reason to test chunked prefill. It was faster and saved memory, but failed
the logit correctness gate:

- `code_review_rust_8k_001`, seed `20260632`, context `8192/8192`.
- Baseline prefill: `28946.752 ms`; candidate prefill: `26501.713 ms`.
- Peak MLX improved from `12.763 GB` to `7.594 GB`.
- Output token matched (`100`), but output logit moved from `22.5` to `23.25`;
  delta `0.75` exceeded the `0.5` tolerance.

No 16K holdout was run after the 8K correctness failure. The env flag remains
default-off.

## Completion rule

Stop when chunked native prefill has same-code A/B evidence against the native
per-layer baseline, or when compile/runtime/correctness/performance blockers
explain why it should not continue.
