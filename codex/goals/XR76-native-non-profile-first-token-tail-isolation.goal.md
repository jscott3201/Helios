# XR76 - Native non-profile first-token tail isolation

## Objective

Measure the real native runtime-default first-token/full-attention tail without
profile-mode deferred-KV scheduling perturbation, then decide whether a
warm/JIT/cache or lower-level full-attention materialization hypothesis is worth
implementation.

## Current Evidence

- XR72 showed chat first-token outliers are dominated by full-attention eval,
  not host collection, capacity growth, or final sync.
- XR75 rejected simple serial full-attention group scheduling against runtime
  default: the `chat_short_1k_001` follow-up wrote `9/9` correct records and
  `567/567` profiled samples, but p50 regressed `69.655 -> 71.448 ms`, p95
  regressed `70.191 -> 72.057 ms`, and p99 improved only
  `163.705 -> 162.719 ms`.
- XR69/XR72 profile mode is useful for attribution, but it changes deferred KV
  eval scheduling by splitting full-attention and sliding arrays for profiling.
  XR76 must therefore compare against a non-profile runtime path.
- MTP broad promotion remains parked until protected aggregate speed clears the
  release gate; DSpark remains parked.

## Scope

- Add harness/reporting support if needed to compare non-profile runtime default
  against a profiled runtime-default variant in the same XR06 run.
- Do not change native runtime defaults.
- Do not widen the C/Rust ABI unless non-profile evidence proves a missing
  timing field is necessary.
- Use `chat_short_1k_001` first, because XR72 and XR75 identified it as the
  first-token tail lane.
- Keep artifacts under
  `benchmarks/out/XR76-native-non-profile-first-token-tail-isolation/`.

## Non-goals

- Do not promote MTP defaults from XR76.
- Do not revive DSpark from XR76.
- Do not promote XR70/XR71/XR75 candidates from XR76.
- Do not claim a performance win from profile-only timing.

## Acceptance Criteria

1. The report exposes first-token latency stats from token traces whether or not
   `GEMMA4D_NATIVE_DECODE_PROFILE` is enabled.
2. XR06 can compare `native_decode_runtime_default` and
   `native_decode_runtime_default_profiled` in one run, with profiling enabled
   only for the profiled variant.
3. XR06 can run a default-off `native_decode_runtime_default_warmup_probe`
   harness variant that performs a discarded same-shape prefill/decode before
   measuring the real record, to test warm/JIT/cache behavior without changing
   native runtime defaults.
4. Static gates pass:
   `cargo fmt --all --check`, `git diff --check`, and
   `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.
5. A focused Metal run on `chat_short_1k_001` records correctness, p50/p95/p99,
   first-token p50/p95/p99/max, peak MLX, and profile sample counts.
6. Any follow-up candidate remains default-off and requires evidence against
   the non-profile runtime-default baseline.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR76-native-non-profile-first-token-tail-isolation/profile-perturbation-chat-1k \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_runtime_default_profiled

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR76-native-non-profile-first-token-tail-isolation/warmup-probe-chat-1k \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_runtime_default_warmup_probe
```

## Completion Rule

Complete XR76 only when the non-profile first-token/profile-perturbation
evidence is recorded with exact commands and a next decision:
`needs_candidate`, `reject_profile_perturbation_hypothesis`,
`accept_candidate`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence`.

## Result - 2026-07-05

Status: `needs_candidate`; warm/JIT/cache hypothesis accepted as next native
lane, but no runtime/default policy changed.

XR76 added XR06 first-token latency aggregates from existing
`decode_token_traces`, plus two harness variants:

- `native_decode_runtime_default_profiled` enables
  `GEMMA4D_NATIVE_DECODE_PROFILE=1` only for that variant.
- `native_decode_runtime_default_warmup_probe` performs a discarded
  same-workload prefill plus one decode before measuring a fresh record.

Evidence:

- Profile-perturbation artifacts:
  `benchmarks/out/XR76-native-non-profile-first-token-tail-isolation/profile-perturbation-chat-1k/`
- Warmup-probe artifacts:
  `benchmarks/out/XR76-native-non-profile-first-token-tail-isolation/warmup-probe-chat-1k/`
- Static gates passed:
  `cargo fmt --all --check`, `git diff --check`, and
  `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.
- Profile-perturbation run wrote `9/9` correct records, `189/567` profile
  samples, no blockers, and peak MLX `7.321 GB`. Profiling did not explain the
  first-token tail: first-token p50 moved `216.004 -> 218.580 ms`, and
  p95/max moved `222.905 -> 229.669 ms` versus non-profile runtime default.
- Warmup-probe run wrote `9/9` correct records, `0/567` profile samples, no
  blockers, and peak MLX `7.321 GB`. The harness-only warmup probe passed the
  XR06 tail gate against runtime default: raw p50 regressed
  `69.778 -> 70.460 ms` (`+0.977%`), raw p95 regressed
  `70.367 -> 71.690 ms`, raw p99/max improved `177.571 -> 86.680 ms`
  (`+51.186%`), and first-token p50 improved `177.571 -> 86.680 ms`.

Claim boundary:

- The warmup probe excludes the discarded warmup cost from the measured record.
- The result is hypothesis evidence only, not a default/runtime promotion.
- MTP broad promotion and DSpark remain parked.

Next input: implement a real default-off first-token warmup policy candidate
with explicit warmup-cost accounting, shape/context guardrails, and follow-up
matrix evidence beyond `chat_short_1k_001`.
