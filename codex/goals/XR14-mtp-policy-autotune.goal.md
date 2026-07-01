# XR14 - MTP policy autotune replay

## Outcome

Turn the repaired XR04 native MTP evidence into a reproducible policy replay
artifact that explains which MTP block-size choices are worth taking and which
selection rules are unsafe.

## Scope

- Input is the XR04 root summary at
  `benchmarks/out/XR04-mtp-repair-and-autotune/summary.json` unless overridden.
- Baseline is native non-MTP greedy decode from the same XR04 record.
- Candidates are repaired native MTP block sizes `1` and `2`.
- Compare net generation phase as `draft_ms + verify_ms` against baseline
  `decode_ms`.
- Keep this as benchmark evidence only. Do not optimize runtime code and do not
  enable MTP by default.

## Required work

1. Replay fixed-block, acceptance-threshold, and net-latency-guarded policies.
2. Record exact source files, source run IDs, git SHAs, deterministic workload
   seeds, context token lengths, max-new-token lengths, block sizes, and model
   artifact identities.
3. Prove why acceptance-only policy is insufficient when high-acceptance
   workloads regress on net latency.
4. Produce a follow-on recommendation for a real variance A/B run if replay
   evidence is promising.

## Acceptance gates

- `records.jsonl`, `summary.json`, `report.md`, `blockers.md`, and
  `decision.md` exist under
  `benchmarks/out/XR14-mtp-policy-autotune/`.
- Every selected MTP candidate used for a speed claim is byte-identical to the
  native non-MTP baseline in the source XR04 record.
- Any proposed policy must be marked `needs_more_data` unless it has independent
  holdout or variance evidence beyond XR04 replay.
- Fixed block-size policies and acceptance-only policies must be rejected when
  they regress any replayed workload beyond the policy gate.

## Required artifacts

```text
benchmarks/out/XR14-mtp-policy-autotune/records.jsonl
benchmarks/out/XR14-mtp-policy-autotune/summary.json
benchmarks/out/XR14-mtp-policy-autotune/report.md
benchmarks/out/XR14-mtp-policy-autotune/blockers.md
benchmarks/out/XR14-mtp-policy-autotune/decision.md
```

## Completion rule

Stop when the decision file exists with raw replay evidence and the benchmark
ledger explains the resulting MTP policy status.
