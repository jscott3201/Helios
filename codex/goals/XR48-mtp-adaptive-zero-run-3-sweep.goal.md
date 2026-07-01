# XR48 - MTP adaptive zero-run 3 sweep

## Outcome

Determine whether the middle adaptive fallback threshold
`zero-run=3,min12` preserves the `chat_short_1k_001` and `tool_json_1k_001`
wins while improving `mtp_candidate_1k_001` beyond XR46 without the aggressive
chat fallback seen in XR47.

## Scope

- Baseline evidence: XR45 lazy block-prefix 1K family holdout, XR46 adaptive
  zero-accept fallback at `zero-run=4,min12`, and XR47 adaptive threshold sweep
  at `zero-run=1,min12`.
- Candidate: existing default-off lazy block-prefix MTP flags plus existing
  XR46 benchmark harness flags:
  - `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  - `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`
  - `--adaptive-zero-accept-run 3`
  - `--adaptive-min-generated-tokens 12`
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Workloads:
  - `chat_short_1k_001`
  - `tool_json_1k_001`
  - `mtp_candidate_1k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.
- Trials: `3` measured plus `1` warmup.
- No runtime code changes are expected for this goal.

## Threshold Rationale

- XR46 `zero-run=4,min12` did not fire until pass `10` for
  `mtp_candidate_1k_001`; p50 improved versus XR45 but still regressed against
  native baseline.
- XR47 `zero-run=1,min12` fired at pass `7` for `mtp_candidate_1k_001` and
  removed the p50 regression, but it also fired at pass `11` on
  `chat_short_1k_001`, reducing the chat margin to `5.958%`.
- XR45/XR46/XR47 event traces show `chat_short_1k_001` has only two-token
  zero-accept runs after the minimum-token gate, while `mtp_candidate_1k_001`
  has a sustained zero-accept run after pass `7`. A `zero-run=3` gate should
  avoid chat fallback and still stop the weak candidate path earlier than XR46.

## Required Work

1. Run the candidate with the XR15 real-context MTP harness.
2. Record exact commands, generated files, git SHA, deterministic workload
   seeds, token lengths, exactness, fallback settings, fallback firing points,
   attempted/accepted tokens, rollback count, `draft_ms`, `verify_ms`,
   `fallback_decode_ms`, decode phase, peak MLX, active KV bytes, and blockers.
3. Compare against XR45, XR46, and XR47 for the same workload set.
4. Update `BENCHMARKS.md`, including headline MTP rows only for stable
   top-line numbers.
5. Keep MTP disabled by default.

## Acceptance Gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- `chat_short_1k_001` and `tool_json_1k_001` must still clear the `5%`
  guarded speedup threshold.
- `mtp_candidate_1k_001` must improve over XR46's `3143.500 ms` MTP
  decode-phase p50 and ideally clear the `5%` guarded speedup threshold.
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
benchmarks/out/XR48-mtp-adaptive-zero-run-3-sweep/zero-run-3-min12/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `keep_experimental` by the XR15 harness. The middle
`zero-run=3,min12` threshold avoided the XR47 chat fallback and preserved the
two strong 1K wins, but `mtp_candidate_1k_001` still regressed slightly against
native baseline. This is useful policy-shape evidence, not default-on evidence.

- Run: `xr15-1782930382`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR48-mtp-adaptive-zero-run-3-sweep/zero-run-3-min12 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`.
- Artifacts:
  `benchmarks/out/XR48-mtp-adaptive-zero-run-3-sweep/zero-run-3-min12/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Exactness: `12/12` total records and `9/9` measured records were
  byte-identical against native non-MTP baseline.
- Blockers: none. The first sandboxed run failed before benchmarking because
  MLX could not access Metal; the successful run used approved escalation so
  MLX could access the Mac Metal device.
- Adaptive settings: zero-accept run threshold `3`, minimum generated tokens
  `12`.
- `chat_short_1k_001`: seed `20260630`, context `1024/1024`, prompt SHA-256
  `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `2701.736 ms`, MTP decode-phase p50 `2115.179 ms`, fallback decode p50
  `0.000 ms`, speedup `21.710%`, acceptance `69/96 = 0.719`, rollbacks `27`,
  peak MLX `8.002 GB`, active KV `352845824` bytes. Adaptive fallback did not
  fire.
- `tool_json_1k_001`: seed `20260635`, context `1024/1024`, prompt SHA-256
  `7687cd292cf8f9be5f84f3dca2e3644a08d973a1a314facb52ac91bbed0d5e2c`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `2814.431 ms`, MTP decode-phase p50 `2116.066 ms`, fallback decode p50
  `0.000 ms`, speedup `24.814%`, acceptance `75/96 = 0.781`, rollbacks `21`,
  peak MLX `8.002 GB`, active KV `352845824` bytes. Adaptive fallback did not
  fire.
- `mtp_candidate_1k_001`: seed `20260641`, context `1024/1024`, prompt
  SHA-256 `afc51a55b76097a09f030c835b9917b4425469ba9c758ef513cb355e10da04c6`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `2880.829 ms`, MTP decode-phase p50 `2915.728 ms`, fallback decode p50
  `1329.154 ms`, speedup `-1.211%`, acceptance `21/45 = 0.467`, rollbacks
  `24`, peak MLX `8.008 GB`, active KV `352829440` bytes.
- `mtp_candidate_1k_001` fallback behavior: adaptive fallback fired in `3/3`
  measured records at pass `9` after `15` generated tokens with reason
  `consecutive zero-accept passes 3 reached threshold 3 after 15 generated
  tokens`.
- Compared with XR46, `mtp_candidate_1k_001` attempted draft tokens dropped
  `48 -> 45`, candidate p50 decode phase improved `3143.500 -> 2915.728 ms`,
  and the p50 result moved from `-9.439%` regression to `-1.211%`, but it
  still missed the no-regression target and the `5%` per-workload guard.
- Compared with XR47, `chat_short_1k_001` avoided adaptive fallback, attempted
  draft tokens returned `48 -> 96`, and p50 speedup improved from `5.958%` to
  `21.710%`.
- Fixed block-2 and acceptance-threshold policies selected all three workloads
  with `14.887%` aggregate speedup, but the net-latency guard still selected only
  `chat_short_1k_001:block2` and `tool_json_1k_001:block2` because
  `mtp_candidate_1k_001` did not clear the `5%` per-workload speed gate.

## Completion Rule

Stop when the middle adaptive threshold has fresh measured evidence against the
native baseline and XR45/XR46/XR47 comparators, or when blockers explain why it
cannot be judged.
