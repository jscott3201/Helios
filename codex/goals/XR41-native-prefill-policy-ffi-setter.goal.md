# XR41 - Native prefill policy FFI setter

## Outcome

Add a narrow first-class control surface for the accepted native chunked-prefill
policy without changing runtime defaults or the `Gemma4LoadConfig` C layout.

## Scope

- Add an additive target-level C ABI setter for native prefill chunk policy.
- Add a safe Rust wrapper in `gemma4d-ffi`.
- Add a small XR05 benchmark variant that applies
  `long_context_256` through the Rust setter instead of process environment.
- Baseline for the real-context smoke: `native_eval_per_layer`.
- Candidate: setter-driven `long_context_256` policy.
- Workloads:
  - `tool_json_1k_001` (`1024/1024`), below threshold.
  - `benchmark_qa_4k_001` (`4096/4095`), boundary case below threshold.
  - `adapter_expert_4k_001` (`4096/4096`), at threshold.
- Compare baseline, env-driven policy, and setter-driven policy so the new
  setter path proves both below-threshold no-chunk behavior and at-threshold
  chunked-prefill shape without rerunning 16K.
- Do not change `Gemma4LoadConfig`, public defaults, server behavior, model
  math, tokenizer behavior, MTP behavior, or non-native paths.

## Required Work

1. Keep the existing env policy path intact and preserve explicit
   `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` precedence for env-driven loads.
2. Add an additive FFI API that can set:
   - disabled native chunking,
   - fixed chunk tokens,
   - `long_context_256`.
3. Validate null pointer and invalid fixed-size errors.
4. Add Rust wrapper tests that pass without full-model MLX execution.
5. Add a benchmark variant that calls the setter after target load and before
   prefill.
6. Run correctness/compile checks and a small real-context A/B smoke.
7. Record exact commands, generated artifacts, deterministic seed, context
   length, correctness/logit deltas, prefill timing, peak MLX, active KV bytes,
   blockers, and caveats.
8. Update `BENCHMARKS.md`.

## Acceptance Gates

- `Gemma4LoadConfig` layout remains unchanged.
- Existing env-driven policy behavior remains available.
- Setter path is default-off and changes behavior only when explicitly called.
- Rust FFI tests cover success and invalid fixed-token policy without requiring
  a full model.
- The real-context smoke is correctness-clean; below-threshold setter rows do
  not show chunked-prefill memory shape, and the at-threshold setter row shows
  the accepted 4K chunked-prefill memory shape.
- No default server/profile behavior changes in this goal.

## Required Artifacts

```text
benchmarks/out/XR41-native-prefill-policy-ffi-setter/setter-boundary-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `accept_candidate` for the additive FFI setter and setter-backed
benchmark variant.

Added a target-level C ABI setter and safe Rust wrapper for native prefill
chunk policy. `Gemma4LoadConfig` layout remains unchanged. Runtime defaults,
server behavior, profile config, model math, tokenizer behavior, MTP behavior,
and non-native paths remain unchanged.

### Source Changes

- `native/gemma4_mlx/include/gemma4_mlx.h`
  - Added `Gemma4PrefillChunkMode`, `Gemma4PrefillChunkPolicy`, and
    `gemma4_target_set_prefill_chunk_policy`.
- `native/gemma4_mlx/src/native_model.{h,cc}`
  - Added `NativeTextModel::set_prefill_chunk_policy`.
- `native/gemma4_mlx/src/runtime.cc`
  - Added C ABI validation and target-level policy application.
- `crates/gemma4d-ffi/src/lib.rs`
  - Added raw bindings, safe `PrefillChunkPolicy`, and
    `Target::set_prefill_chunk_policy`.
  - Added smoke/null/invalid fixed-token/unknown-mode tests.
- `crates/gemma4d-bench/examples/xr05_prefill_eval_scheduling_ab.rs`
  - Added `native_chunked_prefill_setter_long_context_256`, which calls the
    Rust setter after target load and before prefill.
- `references/ffi/gemma4_mlx.h`
  - Mirrored the additive C ABI declarations.

### Verification

- `cargo fmt --all --check`: passed.
- `cargo test -p gemma4d-ffi --lib`: passed, `15` passed, `1` ignored.
- `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab`:
  passed.

### Benchmark

- Run: `xr05-1782925238-17972000`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR41-native-prefill-policy-ffi-setter/setter-boundary-smoke --trials 3 --clear-workload-ids --workload-id tool_json_1k_001 --workload-id benchmark_qa_4k_001 --workload-id adapter_expert_4k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256,native_chunked_prefill_setter_long_context_256`.
- Artifacts:
  `benchmarks/out/XR41-native-prefill-policy-ffi-setter/setter-boundary-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Records: `27`; passed `27`; blockers: none.

### Workload Results

- `tool_json_1k_001`, seed `20260635`, context `1024/1024`, prompt SHA-256
  `7687cd292cf8f9be5f84f3dca2e3644a08d973a1a314facb52ac91bbed0d5e2c`.
  Setter correctness was `3/3`, peak MLX stayed `7.321 GB`, active KV stayed
  `352321536` bytes, and logit delta was `0.0`. Setter p50/p95 was
  `3967.819/9812.579 ms` vs baseline `2926.258/4982.468 ms`; this
  below-threshold timing is recorded as noisy same-path variance, not speed
  evidence.
- `benchmark_qa_4k_001`, seed `20260633`, context `4096/4095`, prompt SHA-256
  `1514934863d5ad974300a0feb490ac2dbf1ab2eadc2e7f1a1525e2c2eb3b4e42`.
  Setter correctness was `3/3`, peak MLX stayed `9.212 GB`, active KV stayed
  `402636800` bytes, and logit delta was `0.0`. Setter p50/p95 was
  `20240.162/24404.636 ms` vs baseline `14452.294/18237.993 ms`; this
  boundary workload is below the 4096-token actual-context threshold and is not
  speed evidence.
- `adapter_expert_4k_001`, seed `20260638`, context `4096/4096`, prompt
  SHA-256 `e4f055746d250beee415c30893f1baae9efce40789e70e77196b506ff5a3f3a7`.
  Setter correctness was `3/3`; baseline p50/p95 was
  `14884.449/15442.978 ms`; setter p50/p95 was
  `11424.204/12538.482 ms`; p50 improved `23.247%`; p95 improved
  `18.808%`; peak MLX improved from `9.279` to `7.300 GB` (`21.330%`);
  active KV stayed `402653184` bytes; logit delta was `0.125`.

The env-driven policy variant remained available in the same run and accepted
on the at-threshold workload: `adapter_expert_4k_001` p50/p95
`11050.729/11347.331 ms`, peak MLX `7.300 GB`.

## Completion Rule

Stop when the additive setter has tests and one real-context smoke artifact, or
when ABI/test/runtime evidence shows this should stay env-only pending a
versioned load-config redesign.
