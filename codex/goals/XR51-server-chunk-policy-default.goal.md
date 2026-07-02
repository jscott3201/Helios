# XR51 - Server chunk policy default

## Outcome

Make the accepted native long-context prefill policy the default for server-owned
native graph targets, with explicit env overrides preserved.

## Scope

- Apply `PrefillChunkPolicy::LongContext256` to the persistent-native server
  worker after `ResidentTarget::load`.
- Do not change Stub, RealHelper admission behavior, helper-backed target loads,
  the generate CLI path, public FFI shape, model math, tokenizer behavior, MTP,
  adapter policy, or cache policy.
- Preserve explicit override precedence for:
  - `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS`
  - `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY`

## Implementation

- Added server-native policy constants and selector helpers in
  `crates/gemma4d-server/src/lib.rs`.
- Added `ResidentTarget::apply_native_server_default_prefill_chunk_policy`.
- Wired the persistent-native worker in `crates/gemma4d-server/src/http.rs` to
  apply the default after target load and before marking the resident model
  ready.
- The selector returns `Some(PrefillChunkPolicy::LongContext256)` only when:
  - `GEMMA4D_USE_NATIVE_GRAPH` is enabled, and
  - neither explicit native prefill chunk env override is set.
- Rust env parsing mirrors `native/gemma4_mlx/src/runtime.cc:126-134`: unset,
  empty, `0`, `false`, `FALSE`, `off`, and `OFF` all disable the native graph
  path.
- Runtime precedence was checked in `native/gemma4_mlx/src/native_model.cc`:
  native env values are read during native model load, and the FFI setter would
  override them after load. XR51 therefore skips the setter when either explicit
  chunk env is present.
- If the post-load policy setter fails after the resident target has loaded, the
  persistent-native worker records/logs a warning and continues serving with the
  loaded resident instead of converting the worker to a permanent error state.

## Acceptance Gates

- Rung 7 identity: server-mode native token sequences match unwired native
  baseline at 1K, 4K, 8K, and 16K.
- Server A/B: prefill p50/p95 and peak MLX are captured for 1K, 4K, 8K, and
  16K using persistent-native default versus per-request unwired baseline.
- Below-threshold 1K path has no memory regression; its latency delta is
  persistence-only because the native long-context chunk policy is inert below
  `4096` prompt tokens.
- 16K path recovers XR40-level memory and p50 improvements.
- Stub and existing server fixtures remain green.
- Explicit native chunk env overrides are not overridden by the new default.

## Evidence

Percentiles use the XR05 ceil-rank convention. All server A/B runs used:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_PERSISTENT_SERVER=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --repeats 3 --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336
```

The server A/B bundles persistent-native residency plus the default chunk policy.
The 1K row is not chunk-policy speed evidence; it reflects resident server
behavior below the `4096`-token policy threshold. The larger-context policy
interpretation is corroborated by isolated opt-in policy evidence from XR35
(`code_review_rust_8k_001`) and XR40 (`benchmark_qa_16k_001`).

Per-run artifact directories:

- `benchmarks/out/XR51-server-chunk-policy-default/server-default-1k-repeats3`
- `benchmarks/out/XR51-server-chunk-policy-default/server-default-4k-repeats3`
- `benchmarks/out/XR51-server-chunk-policy-default/server-default-8k-repeats3`
- `benchmarks/out/XR51-server-chunk-policy-default/server-default-16k-repeats3`

| Workload | Context | Token identity | Prefill p50 ms | Prefill p95 ms | Wall p50 ms | Peak MLX GB | Load count |
|---|---:|---|---:|---:|---:|---|---|
| `chat_short_1k_001` | 1024 | `3/3` | `2814.225 -> 2352.410` (`+16.410%`) | `2872.084 -> 2853.423` (`+0.650%`) | `6247.649 -> 5451.065` (`+12.750%`) | `7.324 -> 7.324` | `3 -> 1` |
| `code_review_rust_4k_001` | 4096 | `3/3` | `11651.369 -> 10152.938` (`+12.861%`) | `11911.859 -> 11813.827` (`+0.823%`) | `15249.464 -> 14031.474` (`+7.987%`) | `9.216 -> 7.300` (`+20.792%`) | `3 -> 1` |
| `code_review_rust_8k_001` | 8192 | `3/3` | `31285.354 -> 22618.497` (`+27.703%`) | `31597.337 -> 25073.710` (`+20.646%`) | `34804.457 -> 26740.495` (`+23.169%`) | `12.767 -> 7.402` (`+42.027%`) | `3 -> 1` |
| `benchmark_qa_16k_001` | 16384 | `3/3` | `87387.199 -> 41711.194` (`+52.269%`) | `87871.900 -> 52217.347` (`+40.576%`) | `91021.561 -> 45925.081` (`+49.545%`) | `21.874 -> 7.638` (`+65.081%`) | `3 -> 1` |

Model identity for all XR51 server A/B runs:

- path: `artifacts/models/gemma-4-12B-it-4bit`
- config SHA-256:
  `fbc1c1cb48ed86ec98482b2d41f5a03d3991aba74b7c29a93d430761e6518a38`
- tokenizer SHA-256:
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`
- safetensors inventory SHA-256:
  `4af9af81c81dcba1edb5290573e58efc28f71c887ab25a871d3917f4240459af`

## Verification

Passed:

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-server --all-targets
cargo test -p gemma4d-ffi --lib
cargo test -p gemma4d-bench --lib
GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab
GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr11_persistent_native_server_ab
```

## Result

Decision: `accept_candidate`.

XR51 ships the server-native long-context prefill default. The code path is
scoped to persistent-native server targets with native graph enabled and does
not override explicit native chunk env settings. Server-mode evidence matches
tokens at every tested context and reproduces the expected 16K memory and
prefill wins.
