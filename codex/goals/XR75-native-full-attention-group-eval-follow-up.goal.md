# XR75 - Native full-attention group-eval follow-up

## Objective

Test whether explicit full-attention deferred-KV group scheduling reduces the
remaining native decode p95/p99 tail, especially `chat_short_1k_001`
first-token outliers, while preserving all current runtime defaults.

## Current Evidence

- XR72 accepted grouped end-of-decode KV eval versus explicit per-layer on the
  five-workload matrix and showed collection/final sync are not the tail source.
- XR72 chat first-token outliers were dominated by full-attention deferred eval:
  host `406.584..511.937 ms`, full-attention eval `397.454..503.863 ms`,
  collection `0.010..0.308 ms`, final eval sync about `6.337..7.000 ms`.
- XR70/XR71 full-attention KV update candidates remain default-off because
  their wins were uneven and did not clear the strict promotion gate.
- XR73 accepts only explicit scoped chat/tool MTP opt-in; broad MTP promotion is
  parked until protected aggregate speed clears the release threshold.
- DSpark remains parked; it is background evidence, not the shortest path to
  the current native theoretical max.

## Scope

- Add only an explicit default-off native experiment for full-attention group
  scheduling, for example
  `GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_GROUP_EVAL=1`.
- Keep Rust/MLX FFI shape unchanged unless evidence requires more profile data.
- Reuse XR06 real-context A/B harness and the XR72/XR74 workload set:
  - `chat_short_1k_001`
  - `tool_json_1k_001`
  - `code_review_rust_4k_001`
  - `code_review_rust_8k_001`
  - `benchmark_qa_16k_001`
- Keep artifacts under
  `benchmarks/out/XR75-native-full-attention-group-eval-follow-up/`.

## Non-goals

- Do not promote XR70/XR71 full-attention KV update candidates from XR75.
- Do not promote broad MTP defaults from XR75.
- Do not revive DSpark from XR75.
- Do not change server defaults or public request APIs.

## Acceptance Criteria

1. The experiment is explicit/default-off and leaves current runtime behavior
   unchanged when the env var is unset.
2. Static gates pass:
   `cargo fmt --all --check`, `git diff --check`,
   `cargo test -p gemma4d-ffi --lib`, and
   `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.
3. Focused smoke passes generated-token/logit correctness for
   `native_decode_runtime_default` and
   `native_decode_full_attention_group_eval` on `chat_short_1k_001`.
4. A full five-workload matrix is required before any promotion decision.
5. Promotion requires no row regressing p50/p95/p99/cadence over `5%`, at least
   three of five rows clearing the XR06 candidate tail gate, and the 16K row
   staying below the `14 GB` tiny16 peak MLX gate with
   `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`.
6. If the candidate only helps first-token outliers but misses the full gate,
   record `keep_experimental` or `needs_more_data` and identify the next native
   lane from evidence.

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
  --out-dir benchmarks/out/XR75-native-full-attention-group-eval-follow-up/smoke-chat-1k \
  --trials 1 \
  --max-new-tokens 16 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_group_eval

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_DECODE_PROFILE=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR75-native-full-attention-group-eval-follow-up/chat-1k-followup \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_full_attention_group_eval
```

## Completion Rule

Complete XR75 only when the decision is recorded as `accept_candidate`,
`keep_experimental`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence` with exact commands, artifact paths, gate status, and
next input.

## Result - 2026-07-05

Status: `reject_candidate`; serial full-attention group scheduling is not the
next promotion lane.

XR75 added default-off
`GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_GROUP_EVAL=1`, wired the XR06
variant `native_decode_full_attention_group_eval`, and left runtime defaults,
server behavior, public APIs, MTP defaults, adapters, and the C/Rust profile ABI
unchanged when the env var is unset.

Evidence:

- Smoke artifacts:
  `benchmarks/out/XR75-native-full-attention-group-eval-follow-up/smoke-chat-1k/`
- Follow-up artifacts:
  `benchmarks/out/XR75-native-full-attention-group-eval-follow-up/chat-1k-followup/`
- The first sandboxed smoke attempt failed before measurement because MLX could
  not access Metal from the sandbox; escalated Metal runs completed with no
  blockers.
- Smoke wrote `3/3` correct records and `45/45` profiled samples, but rejected
  the candidate due low-N.
- The decisive chat follow-up wrote `9/9` correct records and `567/567`
  profiled samples with peak MLX `7.321 GB`.
- Runtime default remained accepted against explicit per-layer on
  `chat_short_1k_001`, but the group-eval candidate failed against runtime
  default: p50 regressed `69.655 -> 71.448 ms` (`+2.574%`), p95 regressed
  `70.191 -> 72.057 ms`, and p99 improved only
  `163.705 -> 162.719 ms` (`+0.602%`).
- No full five-workload matrix is warranted because the candidate failed the
  chat follow-up gate.

Next input: focus on non-profile first-token/full-attention tail isolation and
warm/JIT/cache behavior. Keep MTP broad promotion and DSpark parked.
