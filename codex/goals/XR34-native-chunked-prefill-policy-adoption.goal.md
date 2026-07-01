# XR34 - Native chunked prefill policy adoption

## Outcome

Adopt the XR32/XR33 256-token chunked prefill evidence as an opt-in native
runtime policy without changing defaults.

## Scope

- Add `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`.
- The policy selects 256-token chunked native prefill only when prompt length is
  at least `4096` tokens.
- Explicit `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` continues to override the
  policy.
- Add an XR05 benchmark variant for the policy path.

## Required Work

1. Keep native prefill defaults unchanged when the new env is absent.
2. Keep public C ABI unchanged.
3. Preserve existing fixed chunk-size env behavior.
4. Verify compile/format checks.
5. Run a policy smoke across short and long contexts, recording exact commands,
   artifacts, seeds, context lengths, correctness, timing, peak MLX, active KV,
   and blockers.

## Acceptance Gates

- Policy variant is byte/token/logit correct against `native_eval_per_layer`
  within XR05 tolerance for measured records.
- For a short context below the policy threshold, policy behavior does not show
  chunked-prefill memory shape.
- For contexts at or above the policy threshold, policy behavior reproduces the
  accepted chunked-prefill memory/timing shape.
- No default-on runtime behavior changes.

## Required Artifacts

```text
benchmarks/out/XR34-native-chunked-prefill-policy-adoption/policy-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `accept_candidate`.

The opt-in policy landed without changing defaults. Fixed
`GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` still takes precedence in the native
selection function; `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`
selects 256-token chunked prefill only when token count is at least `4096`.

- Run: `xr05-1782920598-675780000`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR34-native-chunked-prefill-policy-adoption/policy-smoke --trials 3 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id code_review_rust_4k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`.
- Records: `12`; passed `12`; blockers: none.

### Below Threshold

- Workload: `chat_short_1k_001`.
- Seed: `20260630`.
- Context: `1024/1024`.
- Correctness: `3/3` for baseline and policy.
- Baseline prefill p50/p95: `3330.531/3609.061 ms`.
- Policy prefill p50/p95: `3410.945/3474.591 ms`.
- p50 improvement: `-2.414%`.
- p95 regression value: `-3.726%` (policy p95 improved).
- Peak MLX: `7.321 GB` for both.
- Active KV: `352321536` bytes for both.
- Interpretation: below threshold, the policy did not take the chunked memory
  shape and correctly did not meet an adoption gate.

### At Threshold

- Workload: `code_review_rust_4k_001`.
- Seed: `20260631`.
- Context: `4096/4096`.
- Correctness: `3/3` for baseline and policy.
- Baseline prefill p50/p95: `10886.219/10893.887 ms`.
- Policy prefill p50/p95: `9781.628/9936.305 ms`.
- p50 improvement: `10.147%`.
- p95 regression value: `-8.790%` (policy p95 improved).
- Peak MLX: baseline `9.279 GB`, policy `7.300 GB`
  (`21.330%` improvement).
- Active KV: `402653184` bytes for both.
- Decision: `accept_candidate`.

## Completion Rule

Stop when the policy path has compile checks and benchmark smoke evidence, or
when correctness/runtime blockers show the policy should not land.
