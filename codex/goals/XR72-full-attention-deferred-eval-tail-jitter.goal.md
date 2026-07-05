# XR72 - Full-attention deferred-eval tail jitter

## Objective

Diagnose and reduce native decode p95/p99 tail jitter in the grouped
full-attention deferred-eval path. XR72 is a profiling-first goal: explain the
tail source before changing kernels or promoting any XR70/XR71 candidate.

## Current Evidence

- XR69 showed runtime-default deferred KV eval is dominated by full-attention
  materialization: `63.118..77.984 ms` mean, while sliding eval is only
  `0.006..0.009 ms` mean.
- XR70's default-off full-attention KV update candidate improved total decode
  `78020.982 -> 73634.458 ms` (`+5.622%`) but remained uneven and regressed
  `chat_short_1k_001`.
- XR71's `256` capacity candidate improved total decode
  `73947.065 -> 68583.341 ms` (`+7.253%`) and kept 16K peak MLX at
  `7.929 GB`, but failed strict promotion because `chat_short_1k_001` raw
  p95/p99 regressed by `5.168%`/`26.805%`.
- XR71 profile fields show full-attention update overhead is about
  `0.010 ms/token`; capacity growth, `slice_update`, and visible-slice creation
  are not the remaining bottleneck.
- XR66/XR70 MTP side-effect evidence stays `keep_experimental`; MTP should wait
  until the native baseline tail behavior is cleaner.

## Scope

- Add profile attribution around the full-attention deferred-eval barrier in
  `native/gemma4_mlx/src/native_model.cc::eval_deferred_decode_kv`.
- If a new profile control is needed, make it explicit and profile-only, for
  example `GEMMA4D_NATIVE_DECODE_FULL_ATTENTION_PROFILE=1`.
- Preserve the existing runtime default, public request APIs, server defaults,
  MTP defaults, adapter behavior, and DSpark state.
- Reuse the XR06 real-context harness and the XR70/XR71 five-workload matrix:
  - `chat_short_1k_001`
  - `tool_json_1k_001`
  - `code_review_rust_4k_001`
  - `code_review_rust_8k_001`
  - `benchmark_qa_16k_001`
- Keep all artifacts under
  `benchmarks/out/XR72-full-attention-deferred-eval-tail-jitter/`.

## Required Instrumentation

When `GEMMA4D_NATIVE_DECODE_PROFILE=1`, profile enough detail to explain:

- full-attention eval time by layer or stable layer group;
- array collection time versus `mlx::core::eval(full_attention_arrays)` time;
- full-attention array count, bytes, sequence length, and shape stability;
- whether tails correlate with first-use/JIT/cache behavior or shape churn;
- final `eval_sync_ms` versus deferred full-attention eval time;
- p95/p99 outlier records for `chat_short_1k_001`.

The profile extension must remain ABI-safe and covered by the existing FFI
layout checks or updated layout checks.

## First Patch Scope

Start with a profile-only patch across these files:

| File | Change |
|---|---|
| `native/gemma4_mlx/include/gemma4_mlx.h` | Add fixed-size full-attention group fields to `Gemma4DecodeProfileInfo` if the data must cross the C ABI |
| `native/gemma4_mlx/src/native_model.cc` | Add a cached profile-only env flag before target load and instrument `eval_deferred_decode_kv` |
| `crates/gemma4d-ffi/src/lib.rs` | Mirror any new raw/safe profile fields and update pinned layout tests |
| `crates/gemma4d-bench/examples/xr06_native_decode_tail_latency_ab.rs` | Capture the profile env key, serialize per-token fields, aggregate group p50/p95/p99, and report `chat_short_1k_001` outliers |
| `BENCHMARKS.md` | Record final commands, artifact paths, and decision after the run |

Keep server profiling out of XR72 unless it becomes a separate follow-up; the
server path does not currently expose decode profile fields.

## Acceptance Criteria

1. Profile additions are explicit, default-off or profile-only, and do not
   change runtime behavior when profiling is disabled.
2. XR06 A/B on the five-workload matrix passes generated-token and logit
   exactness for every measured row.
3. The 16K row stays below the `14 GB` tiny16 peak MLX gate with
   `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`.
4. XR72 artifacts explain the `chat_short_1k_001` p95/p99 tails with
   per-layer/group, shape, eval, and sync evidence.
5. Any performance candidate remains default-off and is promoted only if no
   row regresses p50/p95/p99/cadence over `5%` and at least three of five rows
   clear the XR06 candidate tail gate.
6. Do not rerun or promote MTP from XR72 unless the native candidate is cleaner
   than XR71 and meets the gate in item 5.
7. `BENCHMARKS.md` records the decision and exact artifact paths after the run.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_DECODE_PROFILE=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR72-full-attention-deferred-eval-tail-jitter/smoke-chat-1k \
  --trials 1 \
  --max-new-tokens 16 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_kv_update_256

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_DECODE_PROFILE=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR72-full-attention-deferred-eval-tail-jitter/full-matrix \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --workload-id code_review_rust_4k_001 \
  --workload-id code_review_rust_8k_001 \
  --workload-id benchmark_qa_16k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_kv_update_256
```

If XR72 adds a new profile-only env flag or a new XR06 variant label, include
the exact final command and update this file with the concrete name.

## Completion Rule

Complete XR72 only when the artifacts explain the full-attention deferred-eval
tail behavior and the decision is recorded as `accept_candidate`,
`keep_experimental`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence` with exact commands, paths, gate status, and next input.
