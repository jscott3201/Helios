# XR70 - Full-attention KV update candidate

## Objective

Implement and evaluate a default-off full-attention active-KV update/materialization
candidate after XR69 showed that runtime-default deferred KV eval is dominated
by full-attention materialization.

## Current Evidence

- XR69 widened the env-gated decode profile ABI and split deferred KV eval into
  full-attention and sliding-window groups.
- XR69 baseline profile on the selected 1K/4K/8K/16K matrix showed
  `deferred_kv_eval_full_attention_ms` accounts for nearly all
  `deferred_kv_eval_ms`: `63.118..77.984 ms` mean, while sliding eval is only
  `0.006..0.009 ms` mean.
- `attention_kv_mutation_ms` remained only `0.254..0.272 ms` mean, so the
  obvious next target is materialization/update semantics behind the grouped
  full-attention KV barrier.
- XR52's earlier slab attempt was exact but improved native decode p50 by only
  `0.39%..1.05%`, so XR70 must prove it reduces XR69's profile bucket rather
  than only changing storage shape.

## Candidate Shape

Add a default-off native flag:

```text
GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_KV_UPDATE=1
```

When enabled, full-attention layers use a slice-update-backed active KV buffer
for decode growth instead of rebuilding full-attention KV with
`mlx::core::concatenate` each token. Sliding-window layers keep the current
chronological concatenate/slice behavior. Runtime defaults, server defaults,
MTP defaults, adapter behavior, and public request APIs remain unchanged.

## Acceptance Criteria

1. The candidate is disabled by default and selected only by explicit env flag
   or XR06 variant.
2. The candidate reduces `deferred_kv_eval_full_attention_ms` versus
   `native_decode_runtime_default` in the XR06 profile artifacts.
3. XR06 A/B passes generated-token and logit correctness on:
   - `chat_short_1k_001`
   - `tool_json_1k_001`
   - `code_review_rust_4k_001`
   - `code_review_rust_8k_001`
   - `benchmark_qa_16k_001`
4. Candidate p50/p95/p99 decode latency and streaming cadence have no
   selected-workload regression over `5%`.
5. Aggregate decode improves by at least `5%`.
6. The 16K sentinel stays below the `14 GB` tiny16 peak MLX gate with
   `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`.
7. If XR06 accepts the candidate, rerun the XR66 Adaptive-N MTP matrix on the
   faster native baseline and decide whether the broad MTP default-on gate
   closes.
8. Results are documented in `BENCHMARKS.md` with exact commands and artifact
   paths under `benchmarks/out/XR70-full-attention-kv-update/`.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_DECODE_PROFILE=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR70-full-attention-kv-update/full-matrix \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --workload-id code_review_rust_4k_001 \
  --workload-id code_review_rust_8k_001 \
  --workload-id benchmark_qa_16k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_kv_update

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_KV_UPDATE=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR70-full-attention-kv-update/xr66-mtp-candidate-full-attention-kv-update \
  --source-replay benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --adaptive-policy xr61-real-margin-v1 \
  --adaptive-zero-accept-run 3 \
  --adaptive-min-generated-tokens 12 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_KV_UPDATE=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR70-full-attention-kv-update/xr66-mtp-sequential-oracle-full-attention-kv-update \
  --source-replay benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --adaptive-policy xr61-real-margin-v1 \
  --adaptive-zero-accept-run 3 \
  --adaptive-min-generated-tokens 12 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --workload-id mtp_candidate_1k_001

python3 scripts/xr61_adaptive_n_report.py \
  --policy-candidates benchmarks/out/XR61-adaptive-n-mtp/policy-search/policy_candidates.json \
  --baseline-summary benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trace-summary benchmarks/out/XR61-adaptive-n-mtp/trace-capture-real-margins/summary.json \
  --candidate-summary benchmarks/out/XR70-full-attention-kv-update/xr66-mtp-candidate-full-attention-kv-update/summary.json \
  --holdout-summary benchmarks/out/XR61-adaptive-n-mtp/candidate-adaptive-n-v2-safe-bypass-holdouts/summary.json \
  --oracle-summary benchmarks/out/XR70-full-attention-kv-update/xr66-mtp-sequential-oracle-full-attention-kv-update/summary.json \
  --out-md benchmarks/out/XR70-full-attention-kv-update/xr70-mtp-after-native-kv-update-summary.md \
  --out-json benchmarks/out/XR70-full-attention-kv-update/xr70-mtp-after-native-kv-update-summary.json \
  --ledger-updated
```

## Result

Decision: `accept_candidate`, default-off only.

XR70 adds default-off `GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_KV_UPDATE=1`
for full-attention layers. It keeps sliding-window KV handling, public APIs,
server defaults, MTP defaults, adapter behavior, and the runtime default
unchanged.

### XR06 Result

Evidence: `benchmarks/out/XR70-full-attention-kv-update/full-matrix/`.

The full XR06 matrix wrote `45/45` passed records, `45/45` correctness-passed
records, and `2835/2835` profiled decode samples. Peak MLX stayed under the
tiny16 gate; the 16K candidate peak was `7.929 GB`.

Total measured decode time improved from runtime default `78020.982 ms` to
candidate `73634.458 ms` (`+5.622%`). Summed raw p50 improved
`363.171 -> 355.840 ms` (`+2.019%`) and summed raw p95 improved
`403.644 -> 390.384 ms` (`+3.285%`).

The candidate reduced `deferred_kv_eval_full_attention_ms` on four of five
workloads:

| Workload | Runtime full-attn mean ms | Candidate full-attn mean ms | Change |
|---|---:|---:|---:|
| `benchmark_qa_16k_001` | `80.603` | `74.261` | `+7.869%` |
| `chat_short_1k_001` | `73.665` | `76.753` | `-4.192%` |
| `code_review_rust_4k_001` | `66.838` | `64.531` | `+3.452%` |
| `code_review_rust_8k_001` | `74.884` | `64.837` | `+13.417%` |
| `tool_json_1k_001` | `77.571` | `69.853` | `+9.950%` |

The XR06 comparison accepted the candidate on `code_review_rust_8k_001`
and `tool_json_1k_001`. It did not accept `benchmark_qa_16k_001` or
`code_review_rust_4k_001` because the p95/p99 tail gate was not met, and it did
not accept `chat_short_1k_001` because that row regressed.

### MTP Rerun

Evidence:

- Candidate:
  `benchmarks/out/XR70-full-attention-kv-update/xr66-mtp-candidate-full-attention-kv-update/`
- Sequential oracle:
  `benchmarks/out/XR70-full-attention-kv-update/xr66-mtp-sequential-oracle-full-attention-kv-update/`
- Gate summary:
  `benchmarks/out/XR70-full-attention-kv-update/xr70-mtp-after-native-kv-update-summary.md`
  and `.json`

The XR66-compatible Adaptive-N MTP rerun remains `keep_experimental`. The
candidate wrote `12/12` exact records, `9/9` measured exact records, no
blockers, weighted acceptance `144/204 = 0.706`, and peak MLX `8.019 GB`. The
sequential-oracle differential passed for `9` measured records.

Protected aggregate decode moved `7474.609 -> 5991.287 ms` (`+19.845%`), below
the `25%` default-on gate and slightly lower than XR66's `+20.334%`. Selected
chat/tool lanes moved `4835.648 -> 3347.089 ms` (`+30.784%`), but that narrowed
slice is not enough for broad default-on.

MTP remains default-off/experimental.
