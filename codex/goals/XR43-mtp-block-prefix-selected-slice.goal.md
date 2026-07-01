# XR43 - MTP block-prefix selected-slice confirmation

## Outcome

Confirm whether the XR24 net-latency-guarded block-prefix MTP slice still shows
real measured speedup on the workloads it selected, without changing runtime
defaults or adding new optimization code.

## Scope

- Baseline: native non-MTP greedy decode inside the XR15 harness.
- Candidate: existing default-off block-prefix MTP path:
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`.
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Workloads:
  - `chat_short_1k_001`
  - `mtp_candidate_4k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.
- Trials: `3` measured plus `1` warmup.

## Required Work

1. Preserve default `gemma4_verify_tokens` behavior when the env flag is absent.
2. Run the selected-slice candidate with the XR15 real-context MTP harness.
3. Record exact commands, generated files, git SHA, model identity,
   deterministic workload seeds, token lengths, acceptance, rollback, `draft_ms`,
   `verify_ms`, net decode phase, memory, active KV bytes, and blockers.
4. Compare generated-token exactness against native non-MTP baseline for every
   record.
5. Update `BENCHMARKS.md`, including the headline MTP table if the run produces
   a stable top-line number.
6. Keep MTP disabled by default.

## Acceptance Gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Peak MLX memory stays under the configured tiny16 gate.
- Net-latency-guarded policy selects the candidate with at least `5%` measured
  decode-phase improvement.
- No selected workload regresses past the configured regression gate.
- No default-on runtime, server, adapter, tokenizer, or public ABI behavior
  changes in this goal.

## Non-goals

- Do not optimize runtime code in this slice.
- Do not broaden MTP block size above `2`.
- Do not change sampling behavior, server defaults, active KV compression,
  prefix cache policy, adapter policy, or prefill policy.
- Do not promote MTP to default-on from this evidence alone.

## Required Artifacts

```text
benchmarks/out/XR43-mtp-block-prefix-selected-slice/candidate-block-prefix-selected/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `keep_experimental` by the XR15 harness, but not promotable beyond
default-off experimental policy evidence.

- Run: `xr15-1782927140`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR43-mtp-block-prefix-selected-slice/candidate-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001`.
- Artifacts:
  `benchmarks/out/XR43-mtp-block-prefix-selected-slice/candidate-block-prefix-selected/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Exactness: `8/8` total records and `6/6` measured records were
  byte-identical against native non-MTP baseline.
- Blockers: none. The first sandboxed benchmark attempt failed before
  benchmarking because MLX could not access a Metal device; the same command was
  rerun with approved escalation.
- `chat_short_1k_001`: seed `20260630`, context `1024/1024`, prompt SHA-256
  `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `3084.066 ms`, MTP decode-phase p50 `2686.191 ms`, speedup `12.901%`,
  draft p50 `173.039 ms`, verify p50 `2513.152 ms`, acceptance
  `69/120 = 0.575`, rollbacks `27`, peak MLX `8.002 GB`, active KV
  `352845824` bytes.
- `mtp_candidate_4k_001`: seed `20260642`, context `4096/4096`, prompt
  SHA-256 `88f76c633511de568b6270b3217be53a26a5c7235862a3c23a514de2646268b3`,
  generated `32` tokens, measured exactness `3/3`, baseline decode p50
  `4886.134 ms`, MTP decode-phase p50 `11780.432 ms`, speedup `-141.099%`,
  acceptance `75/108 = 0.694`, rollbacks `21`, peak MLX `9.244 GB`, active KV
  `403177472` bytes.
- Fixed block-2 and acceptance-threshold policies were rejected because
  `mtp_candidate_4k_001` regressed; net-latency-guarded policy selected only
  `chat_short_1k_001:block2`.
- Net-latency-guarded aggregate speedup was `4.992%`, just below the nominal
  `5%` XR43 gate despite the harness decision. Treat this as borderline
  experimental evidence, not promotion evidence.
- Event histogram across measured records: `accepted=0:36`,
  `accepted=1:12`, `accepted=2:66`.
- Memory caveat: mid-run `vm_stat` samples showed sustained tiny16 pressure,
  including about `4055` to `4271` free 16 KiB pages and about
  `922942` to `927935` pages stored in compressor.

## Completion Rule

Stop when the selected block-prefix candidate has fresh measured evidence
against the native baseline, or when blockers explain why it cannot be judged.
