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

Pending implementation.

## Result

Pending.
