# XR24 - MTP block-prefix holdout A/B

## Outcome

Determine whether XR22's fast block-prefix rollback verifier remains exact and
net-beneficial across the XR14 selected real-context holdout set, not only the
single `mtp_candidate_1k_001` workload.

## Scope

- Baseline: normal native MTP serial verifier.
- Candidate: `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`.
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Workloads:
  - `chat_short_1k_001`
  - `code_review_rust_4k_001`
  - `benchmark_qa_4k_001`
  - `mtp_candidate_1k_001`
  - `mtp_candidate_4k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.
- Trials: `3` measured plus `1` warmup.

## Required work

1. Preserve default `gemma4_verify_tokens` behavior when the env flag is absent.
2. Run the normal serial verifier and XR22 fast verifier with identical workload
   selection and trial counts.
3. Record exact commands, generated files, git SHA, model identity, deterministic
   workload seeds, token lengths, acceptance, rollback, `draft_ms`, `verify_ms`,
   net decode phase, memory, active KV bytes, and blockers.
4. Compare generated-token exactness and committed-token traces between normal
   serial MTP and the fast block-prefix verifier.
5. Keep MTP disabled by default.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate committed-token traces remain compatible with the normal serial MTP
  verifier for every measured record.
- No workload regresses past the configured regression gate.
- Peak MLX memory stays under the configured tiny16 gate.
- Accept only if net-latency-guarded policy selects MTP with at least `5%`
  measured decode-phase improvement and no correctness regressions.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Non-goals

- Do not enable MTP by default.
- Do not add new runtime optimization code.
- Do not change sampling, server defaults, adapters, active KV compression, or
  block sizes above `2`.

## Required artifacts

```text
benchmarks/out/XR24-mtp-block-prefix-holdout/baseline-normal-verify/records.jsonl
benchmarks/out/XR24-mtp-block-prefix-holdout/baseline-normal-verify/summary.json
benchmarks/out/XR24-mtp-block-prefix-holdout/baseline-normal-verify/report.md
benchmarks/out/XR24-mtp-block-prefix-holdout/baseline-normal-verify/blockers.md
benchmarks/out/XR24-mtp-block-prefix-holdout/baseline-normal-verify/decision.md
benchmarks/out/XR24-mtp-block-prefix-holdout/candidate-block-prefix-rollback/records.jsonl
benchmarks/out/XR24-mtp-block-prefix-holdout/candidate-block-prefix-rollback/summary.json
benchmarks/out/XR24-mtp-block-prefix-holdout/candidate-block-prefix-rollback/report.md
benchmarks/out/XR24-mtp-block-prefix-holdout/candidate-block-prefix-rollback/blockers.md
benchmarks/out/XR24-mtp-block-prefix-holdout/candidate-block-prefix-rollback/decision.md
```

## Completion rule

Stop when the XR22 fast verifier has measured holdout evidence against the
same-code normal verifier baseline, or when blockers explain why the holdout
cannot be safely judged.
