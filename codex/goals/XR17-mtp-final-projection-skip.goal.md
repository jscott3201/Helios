# XR17 - MTP final projection skip A/B

## Outcome

Test whether skipping the MTP assistant `post_projection` on the final draft
token reduces draft overhead without changing generated output, acceptance,
rollback behavior, memory gates, or runtime defaults.

## Scope

- Baseline: current native MTP assistant draft path with final projection.
- Candidate: `GEMMA4D_EXPERIMENTAL_MTP_SKIP_FINAL_PROJECTION=1`.
- Workload: start with `mtp_candidate_1k_001` because XR15/XR16 provide recent
  comparable evidence there.
- Block sizes: `1` and `2`.

## Required work

1. Preserve default behavior when the env flag is absent.
2. Record same-code baseline and candidate runs with exact commands and
   artifact paths.
3. Compare `draft_ms`, `verify_ms`, net `draft_ms + verify_ms`, accepted draft
   count, rollback count, exactness, and peak memory.
4. Keep MTP disabled by default.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every
  record.
- Candidate attempted/accepted draft counts and rollback behavior remain
  compatible with the baseline for the same workload and block size.
- Peak MLX memory stays under the configured tiny16 gate.
- Accept only if measured `draft_ms` improves enough to reduce net decode phase
  without more than a `5%` regression on either tested block size.
- Otherwise record `reject_candidate` or `needs_more_data`.

## Non-goals

- Do not enable MTP by default.
- Do not change target verification semantics.
- Do not combine this with batch verify, server defaults, sampling, adapters, or
  compressed active KV.

## Required artifacts

```text
benchmarks/out/XR17-mtp-final-projection-skip/baseline-final-projection/records.jsonl
benchmarks/out/XR17-mtp-final-projection-skip/baseline-final-projection/summary.json
benchmarks/out/XR17-mtp-final-projection-skip/baseline-final-projection/report.md
benchmarks/out/XR17-mtp-final-projection-skip/baseline-final-projection/blockers.md
benchmarks/out/XR17-mtp-final-projection-skip/baseline-final-projection/decision.md
benchmarks/out/XR17-mtp-final-projection-skip/candidate-skip-final-projection/records.jsonl
benchmarks/out/XR17-mtp-final-projection-skip/candidate-skip-final-projection/summary.json
benchmarks/out/XR17-mtp-final-projection-skip/candidate-skip-final-projection/report.md
benchmarks/out/XR17-mtp-final-projection-skip/candidate-skip-final-projection/blockers.md
benchmarks/out/XR17-mtp-final-projection-skip/candidate-skip-final-projection/decision.md
```

## Completion rule

Stop when the candidate has measured-trial evidence against the same-code
baseline, or when blockers explain why the optimization cannot be safely judged.
