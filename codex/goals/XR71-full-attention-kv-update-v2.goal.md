# XR71 - Full-attention KV update v2 tail stabilization

## Objective

Refine the XR70 default-off full-attention KV update candidate so we can tell
whether the remaining variance is caused by storage-capacity policy,
slice-update overhead, visible-slice materialization, or eval synchronization.

## Current Evidence

- XR70 proved the lane is real: total XR06 decode improved
  `78020.982 -> 73634.458 ms` (`+5.622%`) versus runtime default.
- XR70 remained uneven: full-attention deferred eval improved on four of five
  rows, but `chat_short_1k_001` regressed and only `code_review_rust_8k_001`
  plus `tool_json_1k_001` cleared the XR06 candidate tail gate.
- XR70's required MTP rerun stayed `keep_experimental`: protected aggregate
  speedup was `+19.845%`, below the `25%` broad default-on gate.
- DSpark remains parked after XR60 because fixed-prefix speed and tiny16 memory
  were far from viable.

## Candidate Shape

Keep the existing default-off flag:

```text
GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_KV_UPDATE=1
```

Add a second default-off tuning/profile control:

```text
GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_KV_UPDATE_CAPACITY={exact|64|128|256|512}
```

The runtime default remains unchanged. The XR71 benchmark variants compare the
current runtime default against the explicit full-attention KV update capacity
strategies.

## Required Instrumentation

When `GEMMA4D_NATIVE_DECODE_PROFILE=1`, profile:

- total full-attention KV update time;
- capacity-growth time;
- slice-update time;
- visible-slice creation time;
- capacity-growth counts and effective capacity tokens;
- existing deferred full-attention KV eval and eval-sync fields.

## Acceptance Criteria

1. Capacity variants are explicit and default-off.
2. No public request API, server default, MTP default, adapter behavior, or
   runtime decode default changes.
3. XR06 A/B on the XR70 five-workload set passes generated-token and logit
   exactness for every candidate row.
4. Accepted candidate has no row p50/p95/p99/cadence regression over `5%`
   versus `native_decode_runtime_default`.
5. The 16K sentinel stays below the `14 GB` tiny16 peak MLX gate with
   `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`.
6. The profile artifacts explain whether the winning/failing capacity strategy
   changed `full_attention_kv_update_*`, `deferred_kv_eval_full_attention_ms`,
   or `eval_sync_ms`.
7. Rerun the XR66 Adaptive-N MTP side-effect matrix only if the native
   candidate is cleaner than XR70: no row regression over `5%` and at least
   three of five rows clear the XR06 candidate tail gate.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR71-full-attention-kv-update-v2/smoke-chat-1k-capacity --trials 1 --max-new-tokens 16 --clear-workload-ids --workload-id chat_short_1k_001 --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_kv_update_exact,native_decode_full_attention_kv_update_64,native_decode_full_attention_kv_update_128,native_decode_full_attention_kv_update_256,native_decode_full_attention_kv_update_512
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR71-full-attention-kv-update-v2/chat-1k-capacity-followup --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_kv_update_exact,native_decode_full_attention_kv_update_256,native_decode_full_attention_kv_update_512
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR71-full-attention-kv-update-v2/full-matrix-256 --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id code_review_rust_4k_001 --workload-id code_review_rust_8k_001 --workload-id benchmark_qa_16k_001 --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_kv_update_256
```

## Result

XR71 implemented the capacity/profile split without changing defaults. The
runtime default, public APIs, server behavior, MTP defaults, and adapter behavior
remain unchanged.

The initial low-N capacity smoke passed correctness for all capacity variants.
The 3-trial chat follow-up selected `256` as the only focused candidate worth a
full matrix run: it improved p99 versus `native_decode_runtime_default` by
`17.422%` on that isolated run, while `exact` and `512` did not clear the XR06
tail gate.

The final full matrix evidence is:
`benchmarks/out/XR71-full-attention-kv-update-v2/full-matrix-256/`.

- `45/45` records passed, with `45/45` correctness-passed records.
- `2835/2835` decode profile samples were captured.
- Candidate total decode improved versus runtime default
  `73947.065 -> 68583.341 ms` (`+7.253%`).
- The 16K sentinel stayed below the tiny16 gate: candidate peak MLX
  `7.929 GB`.
- Full-attention KV update overhead is not the remaining bottleneck. Candidate
  profile means are about `0.010 ms/token` for
  `full_attention_kv_update_ms`, with capacity time roughly
  `0.00023..0.00042 ms`, slice-update time `0.00439..0.00460 ms`, and
  visible-slice time `0.00411..0.00432 ms`.
- The remaining dominant lane is still deferred full-attention KV eval. The
  candidate reduced `deferred_kv_eval_full_attention_ms` means on all five
  rows, but the raw tail remained noisy:
  - `benchmark_qa_16k_001`: `71.814 -> 64.842 ms`
  - `chat_short_1k_001`: `71.243 -> 70.396 ms`
  - `code_review_rust_4k_001`: `68.028 -> 64.302 ms`
  - `code_review_rust_8k_001`: `77.842 -> 63.582 ms`
  - `tool_json_1k_001`: `63.470 -> 62.246 ms`

Strict XR71 adoption criteria are not met. The candidate is useful and remains
default-off, but `chat_short_1k_001` regressed raw p95/p99 versus runtime
default by `5.168%` and `26.805%`, and only
`code_review_rust_8k_001` cleared the candidate XR06 tail gate. Because the
native candidate is not cleaner than XR70 and does not clear at least three of
five tail gates, the XR66 Adaptive-N MTP side-effect rerun was intentionally
skipped.

Conclusion: keep the full-attention KV update capacity policy as experimental
evidence and focus the next native goal on full-attention deferred-eval tail
jitter/synchronization, not capacity growth, `slice_update`, visible-slice
materialization, or broad MTP promotion.
