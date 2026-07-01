# XR19 - MTP steady-state horizon A/B

## Outcome

Test whether native MTP remains slower when the short-run final lookahead cost
is amortized over a longer generation horizon.

## Scope

- Benchmark-only evidence slice; do not change runtime code.
- Workload: `mtp_candidate_1k_001`, matching XR15-XR18.
- Block sizes: `1` and `2`.
- Horizon: `64` generated tokens.

## Required work

1. Run the existing native MTP policy variance harness with the longer horizon.
2. Preserve temperature-0 exactness against the native non-MTP baseline.
3. Record exact commands, artifact paths, git SHA, workload seed, token lengths,
   block sizes, model/drafter identities, and blockers.
4. Compare baseline decode ms, MTP `draft_ms`, `verify_ms`, net
   `draft_ms + verify_ms`, acceptance, rollback count, and peak memory.
5. Keep MTP disabled by default.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Peak MLX memory stays under the configured tiny16 gate.
- Candidate does not regress any measured workload by more than `5%`.
- Accept only if the net-latency-guarded policy selects MTP with at least `5%`
  measured decode-phase improvement.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Non-goals

- Do not enable MTP by default.
- Do not change verifier semantics, target decode code, assistant decode code,
  server defaults, sampling, adapters, or active KV compression.
- Do not combine this run with XR16-XR18 experimental env flags.

## Required artifacts

```text
benchmarks/out/XR19-mtp-steady-state-horizon/records.jsonl
benchmarks/out/XR19-mtp-steady-state-horizon/summary.json
benchmarks/out/XR19-mtp-steady-state-horizon/report.md
benchmarks/out/XR19-mtp-steady-state-horizon/blockers.md
benchmarks/out/XR19-mtp-steady-state-horizon/decision.md
```

## Completion rule

Stop when the longer-horizon A/B evidence is recorded, or when blockers explain
why the steady-state question cannot be safely judged.
