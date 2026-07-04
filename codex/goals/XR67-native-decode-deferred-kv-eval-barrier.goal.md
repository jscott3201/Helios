# XR67 - Native decode deferred-KV eval barrier

## Objective

Benchmark whether the XR65 runtime-default native decode path still has a
profitable deferred KV eval barrier to remove or narrow. Compare the current
unset-env runtime default against the existing explicit decode KV eval policies
without changing runtime defaults.

## Scope

- Use XR06 native decode A/B artifacts as the primary evidence source.
- Treat `native_decode_runtime_default` as the XR67 baseline because XR65 made
  grouped end-of-decode eval the runtime default.
- Compare against:
  - `native_decode_eval_end_of_decode` as an explicit-default sanity row,
  - `native_decode_eval_selective_full_attention`,
  - `native_decode_eval_defer_to_logits`,
  - `native_decode_eval_per_layer` as historical rollback provenance.
- Run selected 1K lanes plus 4K, 8K, and 16K sentinels when memory allows.
- Run an XR15 MTP side-effect probe for any candidate that appears to beat the
  runtime default.

## Non-Goals

- Do not change `decode_kv_eval_mode()` defaults in this goal.
- Do not enable MTP by default.
- Do not add broad model abstractions or expose raw MLX internals.
- Do not pursue DSpark in this goal; XR60 remains rejected for now.

## Acceptance Criteria

1. XR06 writes decode artifacts under
   `benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier/`.
2. The selected workloads include `chat_short_1k_001`,
   `tool_json_1k_001`, `code_review_rust_4k_001`,
   `code_review_rust_8k_001`, and a 16K sentinel if the tiny16 memory gate
   allows it.
3. All candidate rows preserve generated-token/logit exactness against XR06's
   reference and stay under the 14 GB peak MLX gate.
4. A candidate is only marked follow-up-worthy if it beats
   `native_decode_runtime_default` by at least 5% aggregate decode-phase time
   across the measured sentinel set while avoiding p50, p95, and p99
   regressions above 5% on every measured workload.
5. Any follow-up-worthy candidate must pass an XR15 MTP side-effect probe with
   unchanged exactness, selected workloads, and weighted acceptance against the
   current runtime default.
6. The result is documented in `BENCHMARKS.md` with exact commands, artifact
   paths, and a decision of `followup_candidate`, `reject_default_change`, or
   `blocked_with_evidence`.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier/decode-policy-ab --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id code_review_rust_4k_001 --workload-id code_review_rust_8k_001 --workload-id benchmark_qa_16k_001 --variants native_decode_eval_per_layer,native_decode_eval_end_of_decode,native_decode_runtime_default,native_decode_eval_selective_full_attention,native_decode_eval_defer_to_logits
python3 scripts/xr67_deferred_kv_eval_report.py --summary benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier/decode-policy-ab/summary.json --out-dir benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier
```

## Result

Decision: `followup_candidate`.

XR67 found a measured follow-up candidate, but not a safe broad default change
yet. The existing `native_decode_eval_defer_to_logits` mode beat the XR65
runtime default on the clean 1K/4K/8K subset, preserved token/logit exactness,
stayed below the 14 GB tiny16 memory gate, and passed the XR15 MTP side-effect
probe. The 16K sentinel is blocked under the XR06 harness because all variants
peaked at `21.986 GB`, above the tiny16 gate.

### Evidence

- `benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier/decode-policy-ab/`
- `benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier/mtp-side-effect-runtime-default/`
- `benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier/mtp-side-effect-defer-to-logits/`
- `benchmarks/out/XR67-native-decode-deferred-kv-eval-barrier/xr67-deferred-kv-eval-summary.{json,md}`

### Decode Policy A/B

The full XR06 run wrote `75/75` generated/correct records. It returned
`blocked_with_evidence` only because `benchmark_qa_16k_001` crossed the memory
gate at `21.986 GB`.

Filtering out the 16K memory-cliff row, `native_decode_eval_defer_to_logits`
was the only follow-up candidate:

| Candidate | Workloads | Aggregate speedup | Worst p50 reg | Worst p95 reg | Worst p99 reg | Peak MLX | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `native_decode_eval_defer_to_logits` | 4 | `26.370%` | `-1.172%` | `-0.682%` | `-1.840%` | `12.829 GB` | follow-up |
| `native_decode_eval_selective_full_attention` | 4 | `25.041%` | `1.237%` | `1.235%` | `27.464%` | `12.829 GB` | rejected: chat p99 regression |
| `native_decode_eval_per_layer` | 4 | `17.453%` | `14.697%` | `16.172%` | `13.153%` | `12.829 GB` | rejected: tail regressions |

`defer_to_logits` moves the cost from `deferred_kv_eval_ms` into
`eval_sync_ms`; the benefit is scheduling/queue placement, not removal of the
underlying MLX work.

### MTP Side Effect

XR15 side-effect probe passed.

| Mode | Decision | Exact records | Speedup | Accepted / attempted | Weighted acceptance | Selected workloads |
|---|---|---:|---:|---:|---:|---|
| runtime default | `keep_experimental` | `12` | `26.039%` | `312/396` | `0.787879` | `chat_short_1k_001:adaptive`, `tool_json_1k_001:adaptive` |
| `defer_to_logits` | `keep_experimental` | `12` | `25.280%` | `312/396` | `0.787879` | `chat_short_1k_001:adaptive`, `tool_json_1k_001:adaptive` |

No runtime default changed.
