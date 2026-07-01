# XR16 - MTP overhead optimization

## Outcome

Use XR15 as the fresh native baseline, decompose why high-acceptance MTP still
regresses decode phase, and test narrowly gated overhead optimizations with
fresh A/B evidence.

## Scope

- Baseline: repaired native MTP sequential verify from XR15.
- Candidate: opt-in native MTP overhead changes behind explicit experimental
  flags.
- Initial candidate: batched incremental target verification for block size `2`
  behind `GEMMA4D_EXPERIMENTAL_MTP_BATCH_VERIFY=1`.
- Workload priority starts with `mtp_candidate_1k_001` because XR15 showed high
  acceptance but negative net decode speed there.

## Required work

1. Establish a same-code baseline before judging any candidate.
2. Preserve temperature-0 exactness against native non-MTP output.
3. Separate `draft_ms`, `verify_ms`, and net `draft_ms + verify_ms`.
4. Keep MTP opt-in and disabled by default.
5. Record exact commands, artifact paths, git SHA, deterministic workload seeds,
   token lengths, model/drafter identities, flags, and blockers.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every record.
- Peak MLX memory stays under the configured tiny16 gate.
- Candidate does not regress any measured workload by more than `5%`.
- Accept only if guarded policy shows at least `5%` measured decode-phase
  improvement over the same-code sequential MTP baseline.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Non-goals

- Do not enable MTP by default.
- Do not change server defaults, sampling, adapters, compressed active KV, or
  block sizes above `2`.
- Do not accept replay-only evidence.

## Required artifacts

```text
benchmarks/out/XR16-mtp-overhead-optimization/baseline-sequential-block2/records.jsonl
benchmarks/out/XR16-mtp-overhead-optimization/baseline-sequential-block2/summary.json
benchmarks/out/XR16-mtp-overhead-optimization/baseline-sequential-block2/report.md
benchmarks/out/XR16-mtp-overhead-optimization/baseline-sequential-block2/blockers.md
benchmarks/out/XR16-mtp-overhead-optimization/baseline-sequential-block2/decision.md
benchmarks/out/XR16-mtp-overhead-optimization/candidate-batch-block2/records.jsonl
benchmarks/out/XR16-mtp-overhead-optimization/candidate-batch-block2/summary.json
benchmarks/out/XR16-mtp-overhead-optimization/candidate-batch-block2/report.md
benchmarks/out/XR16-mtp-overhead-optimization/candidate-batch-block2/blockers.md
benchmarks/out/XR16-mtp-overhead-optimization/candidate-batch-block2/decision.md
```

## Completion rule

Stop when the candidate has measured-trial A/B evidence against the same-code
sequential baseline, or when blockers explain why the prototype cannot safely
complete.
