# XR18 - MTP in-place serial verifier A/B

## Outcome

Test whether the native MTP serial verifier can reduce overhead by avoiding the
per-pass target KV clone and `native_tokens` copy, while preserving output
exactness, acceptance behavior, memory gates, and default runtime behavior.

## Scope

- Baseline: current staged native MTP serial verifier.
- Candidate: `GEMMA4D_EXPERIMENTAL_MTP_INPLACE_VERIFY=1`.
- Workload: `mtp_candidate_1k_001`, matching the recent XR15-XR17 MTP evidence.
- Block sizes: `1` and `2`.

## Required work

1. Preserve default behavior when the env flag is absent.
2. Keep batch verify and final-projection experiments disabled unless explicitly
   requested by the run command.
3. Record same-code baseline and candidate runs with exact commands and
   artifact paths.
4. Compare `verify_ms`, net `draft_ms + verify_ms`, accepted draft count,
   rollback count, exactness, active KV bytes, and peak memory.
5. Keep MTP disabled by default.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every
  record.
- Candidate attempted/accepted draft counts and rollback behavior remain
  compatible with the baseline for the same workload and block size.
- Peak MLX memory stays under the configured tiny16 gate.
- Accept only if measured `verify_ms` improves enough to reduce net decode phase
  without more than a `5%` regression on either tested block size.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Risk note

The candidate verifies directly against live native KV and token state. If a
native incremental decode fails after mutating KV, the cache cannot be restored
by this path. The candidate must therefore remain behind the explicit
experimental env flag unless a later milestone adds a native failure-atomic
commit primitive or equivalent rollback support.

## Non-goals

- Do not enable MTP by default.
- Do not change target verification semantics.
- Do not combine this with batch verify, final-projection skip, server defaults,
  sampling, adapters, or compressed active KV.

## Required artifacts

```text
benchmarks/out/XR18-mtp-inplace-serial-verify/baseline-staged-verify/records.jsonl
benchmarks/out/XR18-mtp-inplace-serial-verify/baseline-staged-verify/summary.json
benchmarks/out/XR18-mtp-inplace-serial-verify/baseline-staged-verify/report.md
benchmarks/out/XR18-mtp-inplace-serial-verify/baseline-staged-verify/blockers.md
benchmarks/out/XR18-mtp-inplace-serial-verify/baseline-staged-verify/decision.md
benchmarks/out/XR18-mtp-inplace-serial-verify/candidate-inplace-verify/records.jsonl
benchmarks/out/XR18-mtp-inplace-serial-verify/candidate-inplace-verify/summary.json
benchmarks/out/XR18-mtp-inplace-serial-verify/candidate-inplace-verify/report.md
benchmarks/out/XR18-mtp-inplace-serial-verify/candidate-inplace-verify/blockers.md
benchmarks/out/XR18-mtp-inplace-serial-verify/candidate-inplace-verify/decision.md
```

## Completion rule

Stop when the candidate has measured-trial evidence against the same-code
baseline, or when blockers explain why the optimization cannot be safely judged.
