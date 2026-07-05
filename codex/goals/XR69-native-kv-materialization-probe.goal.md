# XR69 - Native KV materialization probe

## Objective

Find the next high-value native decode optimization lane after XR68 by isolating
the runtime-default deferred KV materialization cost, then decide whether a
default-off fixed-capacity or `slice_update` KV storage experiment is worth
implementing before returning to MTP.

## Current Evidence

- XR68 rejected `GEMMA4D_NATIVE_DECODE_KV_EVAL=defer_to_logits` against the
  current runtime default. It did not reduce work; it moved grouped KV eval into
  the logits synchronization lane and failed cadence gates.
- XR65 already accepted the obvious scheduling win: the runtime default is now
  grouped end-of-decode, while explicit `per_layer` remains available.
- XR65/XR68 runtime-default profiles show the dominant token cost is not the
  immediate per-layer `attention_kv_mutation_ms` bucket. In current
  runtime-default records, `attention_kv_mutation_ms` is roughly `0.25 ms`,
  while `deferred_kv_eval_ms` is roughly `63..70 ms` on the selected workloads.
- Native decode still stores per-layer KV through MLX array graph operations:
  decode appends with `mlx::core::concatenate`, sliding layers may slice the
  stored window, and the runtime default materializes the resulting arrays in
  `eval_deferred_decode_kv()`.
- MLX 0.31.2 exposes C++ `slice_update`, `put_along_axis`, and `scatter`, so a
  fixed-capacity or window-update KV representation is a plausible experiment,
  but it needs proof before touching defaults.

## Hypothesis

The remaining native decode floor is dominated by materializing append/copy
work for per-layer KV arrays, especially the `concatenate`/sliding-window store
shape. A fixed-capacity or update-based KV representation may reduce
`deferred_kv_eval_ms` and cadence tails if it avoids rebuilding the whole active
KV tensor each generated token.

## Scope

1. Add no-behavior-change profiling first:
   - Split `deferred_kv_eval_ms` by full-attention vs sliding-window layers.
   - Record evaluated array count, estimated evaluated bytes, and sequence
     length in XR06 profile artifacts.
   - Keep the existing ABI narrow; only widen profile fields if needed.
2. Run a baseline profile using the current runtime default on:
   - `chat_short_1k_001`
   - `tool_json_1k_001`
   - `code_review_rust_4k_001`
   - `code_review_rust_8k_001`
   - `benchmark_qa_16k_001` with
     `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`
3. Based on the profiler result, optionally implement one default-off candidate:
   - fixed-capacity/update-based active KV storage for sliding layers first; or
   - a narrower full-vs-sliding materialization policy if the split proves only
     one family dominates.
4. Evaluate candidate through XR06 A/B with exactness, p50/p95/p99, cadence,
   peak MLX/RSS, and active KV gates.

## Non-Goals

- Do not enable a new KV representation by default.
- Do not resume DSpark or broad MTP default-on work in this goal.
- Do not change adapter loading or remote adapter trust boundaries.
- Do not claim speedup from helper-backed paths.
- Do not accept a scheduling-only policy that merely moves the same materialized
  work to a different synchronization bucket.

## Acceptance Criteria

1. A baseline profiler artifact identifies whether full-attention, sliding
   layers, or both dominate `deferred_kv_eval_ms`.
2. Any candidate is default-off and selected only through an explicit env flag or
   harness variant.
3. Candidate generated tokens and logits remain exact against the runtime
   default on every selected workload.
4. Candidate improves aggregate decode time by at least `5%` and has no
   selected-workload p50/p95/p99 or cadence regression over `5%`.
5. The 16K sentinel stays below the `14 GB` tiny16 peak MLX gate with
   `long_context_256` prefill policy.
6. Active KV bytes remain explainable and do not hide a decompressed BF16 active
   state behind a compressed/alternate label.
7. Results are documented in `BENCHMARKS.md` with exact commands and artifact
   paths under `benchmarks/out/XR69-native-kv-materialization-probe/`.

## Verification Commands

Static/profile ABI gates:

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
```

Baseline profiler run:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR69-native-kv-materialization-probe/baseline-deferred-kv-split --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id code_review_rust_4k_001 --workload-id code_review_rust_8k_001 --workload-id benchmark_qa_16k_001 --variant native_decode_runtime_default
```

## Result

Profile slice complete; no runtime default or candidate KV representation changed.

XR69 widened the env-gated native decode profile ABI to version `7` and added
full-attention/sliding attribution for deferred decode KV eval. When profiling
is disabled, runtime-default decode still uses the prior single grouped
`mlx::core::eval` call. When profiling is enabled, the harness evaluates
full-attention and sliding arrays separately to attribute the grouped barrier.

The baseline run wrote:

```text
benchmarks/out/XR69-native-kv-materialization-probe/baseline-deferred-kv-split/{records.jsonl,summary.json,report.md,blockers.md,decision.md,profile.json,profile.md}
```

It completed with `needs_more_data`, as expected for a baseline-only profile
with no candidate variant. All selected rows passed (`15/15` records), no
blockers were recorded, and the 16K sentinel stayed under the tiny16 gate at
`7.929 GB` peak MLX with `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`.

The split profile identifies full-attention KV materialization as the dominant
lane:

| Workload | Deferred KV eval mean ms | Full-attn mean ms | Sliding mean ms | Eval arrays | Eval bytes | Eval seq len |
|---|---:|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `68.274` | `68.256` | `0.009` | `96` | `352845824` | `1056` |
| `tool_json_1k_001` | `63.136` | `63.118` | `0.009` | `96` | `352845824` | `1056` |
| `code_review_rust_4k_001` | `64.165` | `64.152` | `0.006` | `96` | `403177472` | `4128` |
| `code_review_rust_8k_001` | `78.003` | `77.984` | `0.009` | `96` | `470286336` | `8224` |
| `benchmark_qa_16k_001` | `71.018` | `71.001` | `0.009` | `96` | `604504064` | `16416` |

`attention_kv_mutation_ms` remains sub-millisecond (`0.254..0.272 ms` mean),
and sliding-window KV eval is effectively noise in this profile. The next
implementation goal should therefore target full-attention active-KV
materialization/update semantics first, not a sliding-window-only fixed-capacity
slab and not another MTP policy sweep.
