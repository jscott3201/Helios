# XR68 - Scoped defer_to_logits policy candidate

## Objective

Validate a scoped/default-off `defer_to_logits` decode KV eval policy candidate
with streaming cadence checks and a memory-safe 16K sentinel path.

## Candidate Shape

The candidate is benchmark-only/default-off:

```text
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256
GEMMA4D_NATIVE_DECODE_KV_EVAL=defer_to_logits
```

The baseline uses the same memory-safe prefill policy with the current unset-env
runtime decode default:

```text
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256
GEMMA4D_NATIVE_DECODE_KV_EVAL unset
```

This isolates the decode scheduling policy while avoiding the XR67 unchunked
16K memory cliff.

## Scope

- Keep runtime defaults unchanged.
- Use XR06 native decode artifacts for p50/p95/p99, per-token cadence, peak
  MLX, RSS, and active KV.
- Run `chat_short_1k_001`, `tool_json_1k_001`,
  `code_review_rust_4k_001`, `code_review_rust_8k_001`, and a 16K sentinel.
- Treat XR06 per-token decode latencies as streaming cadence evidence.
- Run XR15 MTP side-effect probes if the candidate passes native decode gates.

## Non-Goals

- Do not enable `defer_to_logits` by default.
- Do not pursue DSpark.
- Do not change the native C ABI.
- Do not claim the policy removes KV eval work; XR67 showed it moves the work
  into the final eval synchronization lane.

## Acceptance Criteria

1. XR06 writes artifacts under
   `benchmarks/out/XR68-scoped-defer-to-logits-policy/`.
2. All selected workloads preserve generated-token/logit exactness.
3. The 16K sentinel stays below the 14 GB tiny16 peak MLX gate using the
   long-context chunked prefill policy.
4. Candidate p50/p95/p99 decode latency and streaming-cadence p50/p95/p99 do
   not regress more than 5% against the runtime-default baseline on any selected
   workload.
5. Aggregate candidate decode time improves by at least 5%.
6. XR15 MTP side-effect probe preserves exactness, selected workloads, and
   weighted acceptance.
7. The result is documented in `BENCHMARKS.md` with exact commands and artifact
   paths.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
python3 -B -m py_compile scripts/xr68_defer_to_logits_policy_report.py
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR68-scoped-defer-to-logits-policy/decode-policy-chunked-16k --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id code_review_rust_4k_001 --workload-id code_review_rust_8k_001 --workload-id benchmark_qa_16k_001 --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_eval_defer_to_logits
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR68-scoped-defer-to-logits-policy/decode-policy-chunked-16k-env-inherited --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id code_review_rust_4k_001 --workload-id code_review_rust_8k_001 --workload-id benchmark_qa_16k_001 --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_eval_defer_to_logits
python3 scripts/xr68_defer_to_logits_policy_report.py --summary benchmarks/out/XR68-scoped-defer-to-logits-policy/decode-policy-chunked-16k-env-inherited/summary.json --out-dir benchmarks/out/XR68-scoped-defer-to-logits-policy
```

## Result

Rejected.

Artifacts:

- Initial blocked run:
  `benchmarks/out/XR68-scoped-defer-to-logits-policy/decode-policy-chunked-16k/`.
- Corrected XR06 run:
  `benchmarks/out/XR68-scoped-defer-to-logits-policy/decode-policy-chunked-16k-env-inherited/`.
- XR68 decision report:
  `benchmarks/out/XR68-scoped-defer-to-logits-policy/xr68-defer-to-logits-policy-summary.{json,md}`.

The initial run showed that XR06's per-variant `EnvGuard` cleared the top-level
`GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256` value before native model
load, causing the 16K sentinel to hit the unchunked `21.986 GB` memory cliff.
XR06 now inherits the global prefill/diagnostic env keys into each variant and
captures them in the summary.

The corrected XR06 run wrote `45/45` exact records and kept the 16K sentinel
under the tiny16 gate. The current runtime default peaked at `7.929 GB` on
`benchmark_qa_16k_001`; `defer_to_logits` peaked at `8.927 GB`.

The scoped policy report rejected the candidate against the current runtime
default: aggregate decode moved `23556.308 -> 23587.824 ms` (`-0.134%`),
`benchmark_qa_16k_001` raw/cadence p99 regressed `38.015%`,
`code_review_rust_4k_001` cadence p99 regressed `9.395%`, and
`code_review_rust_8k_001` cadence p95 regressed `5.909%`. XR15 was skipped
because native latency/cadence gates failed before the MTP side-effect gate.
Runtime defaults remain unchanged.
