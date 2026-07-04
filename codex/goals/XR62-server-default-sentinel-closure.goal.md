# XR62 - Server default sentinel closure

## Outcome

Close P2 native server/default safety and observability after XR61 P1 resolved
as `keep_experimental`.

Decision: `accept_candidate`.

## Scope

- Expose server-side native prefill default policy state through the operator
  runtime surfaces without adding a native FFI getter.
- Preserve XR51/XR53 server behavior:
  - no-arg/default config remains Stub;
  - `serve --model-path PATH` remains PersistentNative when backend is omitted;
  - explicit `--backend stub`, `--backend real-helper`, and explicit
    persistent-native remain honored;
  - explicit native prefill env overrides are not overwritten.
- Add sentinel benchmark support for comparing explicit PersistentNative against
  default model-path PersistentNative, because post-XR53 RealHelper admission
  intentionally fails closed on long unchunked contexts.

## Implementation

- `crates/gemma4d-server/src/lib.rs` now preserves the server default prefill
  policy selection as `Apply`, `SkipNativeGraphDisabled`, or
  `SkipExplicitEnvOverride`.
- `crates/gemma4d-server/src/http.rs` stores
  `persistent_backend.native_prefill_policy` in `/v1/runtime/snapshot` with
  `status`, `policy`, `reason`, and `warning`; `/v1/config` also includes the
  configured native-prefill admission hint.
- `crates/gemma4d-bench/examples/xr11_persistent_native_server_ab.rs` accepts
  `--baseline-backend real-helper|persistent-native` and reports runtime
  snapshot policy evidence.

## Evidence

All sentinel commands used:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336 --baseline-backend persistent-native
```

Per-run artifact directories:

- `benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-8k`
- `benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-16k`
- `benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-24k-low-n`

Percentiles use the XR05 ceil-rank convention. The 24K run is `low_n`
(`2` repeats) because it is the edge-context sentinel.

| Workload | Context | Repeats | Token identity | Prefill p50 ms | Prefill p95 ms | Peak MLX GB | Runtime policy |
|---|---:|---:|---|---:|---:|---:|---|
| `code_review_rust_8k_001` | `8192` | `3` | `3/3` | `20289.683 -> 20587.685` | `20463.171 -> 22733.979` | `7.402 -> 7.402` | `applied -> applied` |
| `benchmark_qa_16k_001` | `16384` | `3` | `3/3` | `43289.598 -> 41360.009` | `46118.003 -> 43759.908` | `7.639 -> 7.639` | `applied -> applied` |
| `long_repo_pack_24k_001` | `24576` | `2` | `2/2` | `60939.960 -> 61254.060` | `68970.317 -> 64120.647` | `7.859 -> 7.859` | `applied -> applied` |

All generated token IDs were `[107]` for both explicit PersistentNative and
default model-path PersistentNative.

## Verification

Passed:

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-server --all-targets
cargo test -p gemma4d-bench --example xr11_persistent_native_server_ab --no-run
GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr11_persistent_native_server_ab
```

## Result

P2 server/default closure is accepted. The default model-path server path is
observable enough to show the native prefill default was applied, and 8K/16K/24K
sentinels either ran safely and token-identically or, in this evidence set, all
ran safely under the tiny16 memory budget. This does not create a new prefill
speed claim; it closes default-path safety and observability so the next
theoretical-max work can focus on measured MTP verifier/fallback cost.
