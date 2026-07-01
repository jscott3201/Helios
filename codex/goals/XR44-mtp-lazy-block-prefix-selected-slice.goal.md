# XR44 - MTP lazy block-prefix selected-slice A/B

## Outcome

Determine whether lazy second-token drafting strengthens the only currently
promising block-prefix MTP speed path from XR43.

## Scope

- Comparator evidence: XR43
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`.
- Candidate: existing default-off block-prefix path plus lazy drafting:
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  and
  `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`.
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Workloads:
  - `chat_short_1k_001`
  - `mtp_candidate_4k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.
- Trials: `3` measured plus `1` warmup.
- No runtime code changes are expected for this goal; use existing flags.

## Required Work

1. Preserve default `gemma4_mtp_draft_block` and `gemma4_verify_tokens`
   behavior when env flags are absent.
2. Run the candidate with the XR15 real-context MTP harness.
3. Compare against XR43's same-workload block-prefix evidence.
4. Record exact commands, generated files, git SHA, deterministic workload
   seeds, token lengths, exactness, event metrics, attempted/accepted draft
   tokens, rollback count, `draft_ms`, `verify_ms`, decode phase, peak MLX,
   active KV bytes, and blockers.
5. Update `BENCHMARKS.md`, including headline MTP rows only for stable
   top-line numbers.
6. Keep MTP disabled by default.

## Acceptance Gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate attempted draft tokens decrease on first-token rejects compared
  with XR43 for the same workload shape.
- Candidate does not increase active KV bytes or peak MLX over XR43.
- Net-latency-guarded policy selects the candidate with at least `5%` measured
  decode-phase improvement.
- `chat_short_1k_001` improves over XR43's `2686.191 ms` candidate p50
  decode phase or explains why variance blocks judgment.
- `mtp_candidate_4k_001` must not be promoted unless its XR43 regression is
  eliminated.
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
benchmarks/out/XR44-mtp-lazy-block-prefix-selected-slice/candidate-lazy-block-prefix-selected/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `keep_experimental` by the XR15 harness, with a stronger
`chat_short_1k_001` block-prefix result than XR43. This is still default-off
policy evidence, not broad MTP promotion evidence.

- Run: `xr15-1782927804`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR44-mtp-lazy-block-prefix-selected-slice/candidate-lazy-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001`.
- Artifacts:
  `benchmarks/out/XR44-mtp-lazy-block-prefix-selected-slice/candidate-lazy-block-prefix-selected/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Exactness: `8/8` total records and `6/6` measured records were
  byte-identical against native non-MTP baseline.
- Blockers: none. The first sandboxed benchmark attempt failed before
  benchmarking because MLX could not access a Metal device; the same command was
  rerun with approved escalation.
- `chat_short_1k_001`: seed `20260630`, context `1024/1024`, prompt SHA-256
  `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `3138.129 ms`, MTP decode-phase p50 `2355.632 ms`, speedup `24.935%`,
  draft p50 `156.381 ms`, verify p50 `2215.613 ms`, acceptance
  `69/96 = 0.719`, rollbacks `27`, peak MLX `8.002 GB`, active KV
  `352845824` bytes.
- `chat_short_1k_001` comparison with XR43: attempted draft tokens decreased
  `120 -> 96`, accepted tokens stayed `69`, rollbacks stayed `27`, candidate
  decode-phase p50 improved `2686.191 -> 2355.632 ms`, draft p50 improved
  `173.039 -> 156.381 ms`, and verify p50 improved
  `2513.152 -> 2215.613 ms`.
- `mtp_candidate_4k_001`: seed `20260642`, context `4096/4096`, prompt
  SHA-256 `88f76c633511de568b6270b3217be53a26a5c7235862a3c23a514de2646268b3`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `10406.955 ms`, MTP decode-phase p50 `11534.612 ms`, speedup `-10.836%`,
  draft p50 `816.476 ms`, verify p50 `10708.846 ms`, acceptance
  `75/96 = 0.781`, rollbacks `21`, peak MLX `9.220 GB`, active KV
  `403177472` bytes.
- Fixed block-2 and acceptance-threshold policies were still rejected because
  `mtp_candidate_4k_001` regressed; net-latency-guarded policy selected only
  `chat_short_1k_001:block2`.
- Net-latency-guarded aggregate speedup was `5.777%`, clearing the XR44 policy
  gate but only by selecting `chat_short_1k_001`.
- Event histogram across measured records: `accepted=0:36`,
  `accepted=1:12`, `accepted=2:66`.
- Memory caveat: mid-run `vm_stat` samples showed sustained tiny16 pressure,
  including about `4015` to `4071` free 16 KiB pages and about
  `572826` to `910058` pages stored in compressor. Post-run memory recovered to
  `640010` free pages.

## Completion Rule

Stop when the lazy block-prefix candidate has fresh measured evidence against
the native baseline and XR43 comparator evidence, or when blockers explain why
it cannot be judged.
