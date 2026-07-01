# XR46 - MTP adaptive zero-accept fallback A/B

## Outcome

Determine whether an opt-in adaptive MTP fallback can preserve XR45's 1K
lazy block-prefix wins while reducing the `mtp_candidate_1k_001` regression.

## Scope

- Baseline evidence: XR45 lazy block-prefix 1K family holdout.
- Candidate: existing default-off lazy block-prefix MTP flags plus an opt-in
  benchmark/runtime-loop policy:
  - `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  - `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`
  - adaptive fallback after a sustained zero-accept run once enough output
    tokens have been generated.
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Workloads:
  - `chat_short_1k_001`
  - `tool_json_1k_001`
  - `mtp_candidate_1k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.
- Trials: `3` measured plus `1` warmup.
- This goal may change the XR15 benchmark harness only. No native C ABI,
  server default, model math, tokenizer behavior, or runtime default changes are
  in scope.

## Required Work

1. Preserve XR15 harness behavior when adaptive fallback flags are absent.
2. Add artifact fields that record adaptive settings, whether fallback fired,
   fallback decode time, reason, pass index, and generated-token position.
3. Run the candidate against the XR45 1K holdout workloads.
4. Compare exactness, acceptance, attempted tokens, MTP decode phase, fallback
   decode, peak MLX memory, active KV bytes, and selected policy outcome.
5. Update `BENCHMARKS.md`, including headline MTP rows only for stable top-line
   numbers.
6. Keep MTP disabled by default.

## Acceptance Gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- `chat_short_1k_001` and `tool_json_1k_001` must not lose their XR45 guarded
  speedup due to premature fallback.
- `mtp_candidate_1k_001` must improve over XR45's `3290.224 ms` MTP
  decode-phase p50 or explain why the adaptive gate was not useful.
- Peak MLX memory stays under the configured tiny16 gate.
- Active KV bytes stay in the expected 1K shape.
- Acceptance rate and speed must be reported separately.
- No default-on runtime, server, adapter, tokenizer, or public ABI behavior
  changes in this goal.

## Non-goals

- Do not enable MTP by default.
- Do not add broad workload-family routing or prompt classification.
- Do not use block sizes above `2`.
- Do not change native verifier semantics, drafter math, sampling behavior,
  server defaults, active KV compression, prefix cache policy, adapter policy,
  or prefill policy.

## Required Artifacts

```text
benchmarks/out/XR46-mtp-adaptive-zero-run-fallback/candidate-adaptive-zero-run/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `keep_experimental` by the XR15 harness. The adaptive zero-accept
fallback fired only on `mtp_candidate_1k_001`, reduced wasted MTP attempts, and
improved that workload versus XR45, but it still did not beat native decode.
This is useful mitigation evidence, not promotion evidence.

- Run: `xr15-1782929125`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR46-mtp-adaptive-zero-run-fallback/candidate-adaptive-zero-run --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 4 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`.
- Artifacts:
  `benchmarks/out/XR46-mtp-adaptive-zero-run-fallback/candidate-adaptive-zero-run/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Exactness: `12/12` total records and `9/9` measured records were
  byte-identical against native non-MTP baseline.
- Blockers: none. The run used approved escalation so MLX could access the Mac
  Metal device.
- Adaptive settings: zero-accept run threshold `4`, minimum generated tokens
  `12`.
- `chat_short_1k_001`: seed `20260630`, context `1024/1024`, prompt SHA-256
  `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `3013.177 ms`, MTP decode-phase p50 `2228.909 ms`, fallback decode p50
  `0.000 ms`, speedup `26.028%`, acceptance `69/96 = 0.719`, rollbacks `27`,
  peak MLX `8.002 GB`, active KV `352845824` bytes. Adaptive fallback did not
  fire.
- `tool_json_1k_001`: seed `20260635`, context `1024/1024`, prompt SHA-256
  `7687cd292cf8f9be5f84f3dca2e3644a08d973a1a314facb52ac91bbed0d5e2c`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `3174.286 ms`, MTP decode-phase p50 `2117.370 ms`, fallback decode p50
  `0.000 ms`, speedup `33.296%`, acceptance `75/96 = 0.781`, rollbacks `21`,
  peak MLX `8.002 GB`, active KV `352845824` bytes. Adaptive fallback did not
  fire.
- `mtp_candidate_1k_001`: seed `20260641`, context `1024/1024`, prompt
  SHA-256 `afc51a55b76097a09f030c835b9917b4425469ba9c758ef513cb355e10da04c6`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `2872.385 ms`, MTP decode-phase p50 `3143.500 ms`, fallback decode p50
  `1245.902 ms`, speedup `-9.439%`, acceptance `21/48 = 0.438`, rollbacks
  `27`, peak MLX `8.008 GB`, active KV `352829440` bytes.
- `mtp_candidate_1k_001` fallback behavior: adaptive fallback fired in `3/3`
  measured records at pass `10` after `16` generated tokens with reason
  `consecutive zero-accept passes 4 reached threshold 4 after 16 generated
  tokens`.
- Compared with XR45, `mtp_candidate_1k_001` attempted draft tokens dropped
  `96 -> 48`, accepted draft tokens dropped `39 -> 21`, and candidate p50
  decode phase improved `3290.224 -> 3143.500 ms`, but the workload still
  regressed against native baseline.
- Fixed block-2 and acceptance-threshold policies were rejected because
  `mtp_candidate_1k_001` still regressed.
- Net-latency-guarded policy selected only `chat_short_1k_001:block2` and
  `tool_json_1k_001:block2`, with aggregate measured-trial speedup `20.322%`
  and weighted acceptance `0.750`.
- Memory caveat: mid-run `vm_stat` samples showed sustained tiny16 pressure,
  including about `3658` to `7111` free 16 KiB pages, `632357` wired pages at
  the second sample, and `310842` to `467704` pages stored in compressor.
  Post-run memory recovered to `555345` free pages.

## Completion Rule

Stop when the adaptive fallback candidate has fresh measured evidence against
the native baseline and XR45 comparator evidence, or when blockers explain why
it cannot be judged.
