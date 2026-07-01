# XR45 - MTP lazy block-prefix 1K family holdout

## Outcome

Determine whether XR44's lazy block-prefix MTP speedup on
`chat_short_1k_001` generalizes across the available 1K real-context workload
families, without changing runtime code or defaults.

## Scope

- Comparator evidence: XR44
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  plus
  `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`.
- Candidate: same existing default-off lazy block-prefix MTP flags:
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  and
  `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`.
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Workloads:
  - `chat_short_1k_001`
  - `tool_json_1k_001`
  - `mtp_candidate_1k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.
- Trials: `3` measured plus `1` warmup.
- No runtime code changes are expected for this goal; use existing flags.

## Required Work

1. Preserve default `gemma4_mtp_draft_block` and `gemma4_verify_tokens`
   behavior when env flags are absent.
2. Run the candidate with the XR15 real-context MTP harness.
3. Record exact commands, generated files, git SHA, deterministic workload
   seeds, token lengths, exactness, event metrics, attempted/accepted draft
   tokens, rollback count, `draft_ms`, `verify_ms`, decode phase, peak MLX,
   active KV bytes, and blockers.
4. Compare `chat_short_1k_001` against XR44's selected-slice result.
5. Update `BENCHMARKS.md`, including headline MTP rows only for stable
   top-line numbers.
6. Keep MTP disabled by default.

## Acceptance Gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Peak MLX memory stays under the configured tiny16 gate.
- Active KV bytes stay in the expected 1K shape and do not exceed the XR44
  `chat_short_1k_001` active KV shape without explanation.
- Net-latency-guarded policy selects the candidate with at least `5%` measured
  decode-phase improvement on the selected workload set.
- No fixed block-size or acceptance-threshold policy may be promoted if any 1K
  family regresses past the configured regression gate.
- Acceptance rate and speed must be reported separately.
- No default-on runtime, server, adapter, tokenizer, or public ABI behavior
  changes in this goal.

## Non-goals

- Do not optimize runtime code in this slice.
- Do not use block sizes above `2`.
- Do not enable partial-reject repair, serial-state repair, state-only repair,
  terminal no-lookahead, or batch verify unless a separate goal scopes it.
- Do not change sampling behavior, server defaults, active KV compression,
  prefix cache policy, adapter policy, or prefill policy.
- Do not promote MTP to default-on from this evidence alone.

## Required Artifacts

```text
benchmarks/out/XR45-mtp-lazy-block-prefix-1k-family-holdout/candidate-lazy-block-prefix-1k/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `keep_experimental` by the XR15 harness. The lazy block-prefix path
generalized to `tool_json_1k_001` but not to `mtp_candidate_1k_001`, so this is
still guarded, default-off MTP policy evidence rather than broad promotion
evidence.

- Run: `xr15-1782928503`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR45-mtp-lazy-block-prefix-1k-family-holdout/candidate-lazy-block-prefix-1k --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`.
- Artifacts:
  `benchmarks/out/XR45-mtp-lazy-block-prefix-1k-family-holdout/candidate-lazy-block-prefix-1k/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Exactness: `12/12` total records and `9/9` measured records were
  byte-identical against native non-MTP baseline.
- Blockers: none. The run used approved escalation so MLX could access the Mac
  Metal device.
- `chat_short_1k_001`: seed `20260630`, context `1024/1024`, prompt SHA-256
  `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `2955.491 ms`, MTP decode-phase p50 `2340.434 ms`, speedup `20.811%`,
  draft p50 `152.666 ms`, verify p50 `2187.769 ms`, acceptance
  `69/96 = 0.719`, rollbacks `27`, peak MLX `8.002 GB`, active KV
  `352845824` bytes.
- `tool_json_1k_001`: seed `20260635`, context `1024/1024`, prompt SHA-256
  `7687cd292cf8f9be5f84f3dca2e3644a08d973a1a314facb52ac91bbed0d5e2c`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `2910.560 ms`, MTP decode-phase p50 `2231.115 ms`, speedup `23.344%`,
  draft p50 `154.620 ms`, verify p50 `2076.494 ms`, acceptance
  `75/96 = 0.781`, rollbacks `21`, peak MLX `8.002 GB`, active KV
  `352845824` bytes.
- `mtp_candidate_1k_001`: seed `20260641`, context `1024/1024`, prompt
  SHA-256 `afc51a55b76097a09f030c835b9917b4425469ba9c758ef513cb355e10da04c6`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `2952.317 ms`, MTP decode-phase p50 `3290.224 ms`, speedup `-11.445%`,
  draft p50 `161.799 ms`, verify p50 `3131.050 ms`, acceptance
  `39/96 = 0.406`, rollbacks `57`, peak MLX `8.008 GB`, active KV
  `352845824` bytes.
- Fixed block-2 and acceptance-threshold policies were rejected because
  `mtp_candidate_1k_001` regressed.
- Net-latency-guarded policy selected only `chat_short_1k_001:block2` and
  `tool_json_1k_001:block2`, with aggregate measured-trial speedup `14.680%`
  and weighted acceptance `0.750`.
- Event histograms across measured records:
  - `chat_short_1k_001`: `accepted=0:24`, `accepted=1:3`,
    `accepted=2:33`.
  - `tool_json_1k_001`: `accepted=0:18`, `accepted=1:3`,
    `accepted=2:36`.
  - `mtp_candidate_1k_001`: `accepted=0:24`, `accepted=1:33`,
    `accepted=2:3`.
- Memory caveat: mid-run `vm_stat` samples showed sustained tiny16 pressure,
  including about `3553` to `4207` free 16 KiB pages, `669502` wired pages at
  the second sample, and `296418` to `476696` pages stored in compressor.
  Post-run memory recovered to `526067` free pages.

## Completion Rule

Stop when the 1K-family lazy block-prefix candidate has fresh measured evidence
against the native baseline, or when blockers explain why it cannot be judged.
