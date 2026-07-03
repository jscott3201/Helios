# XR53 - Server default backend and admission estimator

## Outcome

Default model-backed `gemma4d serve --model-path PATH` to the accepted
PersistentNative server path and replace the server admission memory estimator
with measured native memory constants.

Decision: `accept_candidate`.

## Scope

- Change only the serve CLI/config construction layer for backend selection:
  `ServerConfig::default()` remains the M11 stub.
- When `--model-path` is present and no backend flag is explicit,
  `parse_serve_options` selects `ServerBackend::PersistentNative`.
- Keep explicit opt-outs:
  - `--backend stub --model-path PATH` remains a stub config.
  - `--backend real-helper --model-path PATH` remains helper-backed.
- Retire the `GEMMA4D_EXPERIMENTAL_PERSISTENT_SERVER=1` requirement for
  PersistentNative serving.
- Replace the admission estimate based on `(prompt + max_tokens) * 4096` with
  an XR51/P04 measured-memory model.

## Admission Model

Constants live in `crates/gemma4d-server/src/http.rs` and are sourced from XR51
server A/B `summary.json` peak MLX bytes plus the P04 active-KV slope:

- base resident weight / 1K prefill peak: `7_864_036_352 B`
- decode KV slope: `16_384 B/token`
- chunked prefill slope above 1K: `31 KiB/token`
- unchunked measured points: 1K `7_864_036_352 B`, 4K
  `9_895_433_216 B`, 8K `13_708_834_816 B`, 16K `23_487_508_480 B`

The server keeps the deterministic whitespace prompt estimate but applies a
`13/10` BPE correction for admission. This is intentionally conservative for
the current real-context prompts and is documented as a temporary approximation
until a cheap tokenizer-side admission estimate exists. Explicit native chunk
envs are treated as unknown policy state, so admission uses the unchunked
worst-case constants. Unchunked prompts beyond 16K fail closed.

Estimator anchors:

| Mode | Context | Predicted bytes | Measured bytes | Error |
|---|---:|---:|---:|---:|
| unchunked | 1K | `7_864_036_352` | `7_864_036_352` | `0.000%` |
| chunked | 1K | `7_864_036_352` | `7_864_036_352` | `0.000%` |
| unchunked | 4K | `9_895_433_216` | `9_895_433_216` | `0.000%` |
| chunked | 4K | `7_961_553_920` | `7_837_993_472` | `+1.576%` |
| unchunked | 8K | `13_708_834_816` | `13_708_834_816` | `0.000%` |
| chunked | 8K | `8_091_577_344` | `7_947_432_960` | `+1.814%` |
| unchunked | 16K | `23_487_508_480` | `23_487_508_480` | `0.000%` |
| chunked | 16K | `8_351_624_192` | `8_201_657_344` | `+1.828%` |

## Evidence

The A/B harness is still
`crates/gemma4d-bench/examples/xr11_persistent_native_server_ab.rs`, but XR53
changed candidate startup to call `parse_serve_options` with `--model-path` and
no backend flag before building the runtime. Baseline remains explicit
`real-helper`.

Commands:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --out-dir benchmarks/out/XR53-server-default-backend/default-path-1k-repeats3 --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --clear-workload-ids --workload-id chat_short_1k_001 --repeats 3 --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --out-dir benchmarks/out/XR53-server-default-backend/default-path-16k-repeats3 --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --clear-workload-ids --workload-id benchmark_qa_16k_001 --repeats 3 --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336
```

Both runs passed with `decision: accept_candidate`, no blockers, token IDs
`[107]` matched on all repeats, and candidate runtime snapshots reported
`persistent_native` with the resident model loaded.

Percentiles use the XR05 ceil-rank convention.

| Workload | Context | Token identity | Prefill p50 ms | Prefill p95 ms | Wall p50 ms | Peak MLX GB | Load count |
|---|---:|---|---:|---:|---:|---|---|
| `chat_short_1k_001` | 1024 | `3/3` | `2869.853 -> 2309.716` (`+19.518%`) | `2984.829 -> 2963.324` (`+0.720%`) | `6342.319 -> 5349.421` (`+15.655%`) | `7.324 -> 7.324` | `3 -> 1` |
| `benchmark_qa_16k_001` | 16384 | `3/3` | `88657.954 -> 42268.699` (`+52.324%`) | `89421.493 -> 57144.218` (`+36.096%`) | `92429.548 -> 46741.532` (`+49.430%`) | `21.874 -> 7.638` (`+65.081%`) | `3 -> 1` |

Artifact directories:

- `benchmarks/out/XR53-server-default-backend/default-path-1k-repeats3`
- `benchmarks/out/XR53-server-default-backend/default-path-16k-repeats3`

## Verification

Passed:

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-server --all-targets
cargo test -p gemma4d-bench --example xr11_persistent_native_server_ab --no-run
```

## Result

XR53 ships the model-path default to PersistentNative, preserves stub behavior
for zero-arg/default config and explicit stub selection, removes the
PersistentNative env gate, and makes the admission guard meaningful at 16K.

XR52 slab work did not re-anchor decode baselines, so these constants remain
scoped to the accepted XR51/P04 evidence on the concat-KV mainline. Re-measuring
after any future KV storage rewrite is wave-3 hygiene, not an XR53 blocker.
