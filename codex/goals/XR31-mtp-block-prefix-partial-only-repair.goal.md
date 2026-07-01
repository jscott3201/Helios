# XR31 - MTP block-prefix partial-only repair A/B

## Outcome

Determine whether XR22 block-prefix rollback can retain useful speed while
avoiding block-produced full-accept state drift by serially repairing only
full-block accepts.

## Scope

- Baseline: normal serial MTP verifier from XR15.
- Candidate: `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` plus
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_ONLY_REPAIR=1`.
- Start with the XR24 blocker workload `code_review_rust_4k_001`.
- Run the five-workload XR24 holdout only if the blocker workload is
  byte-identical and the timing signal justifies broader measurement.

## Required work

1. Keep MTP disabled by default.
2. Keep public C ABI unchanged.
3. Do not use lazy second draft, batch verify, direct first-reject, in-place
   verify, terminal no-lookahead, or state-only repair for the judged candidate.
4. Preserve ordinary block-prefix rollback behavior when the new env flag is
   absent.
5. Record exact commands, generated files, git SHA, deterministic workload
   seed, context length, generated-token count, draft/verify/decode-phase
   timing, attempted/accepted draft tokens, rollback count, peak MLX memory,
   active KV bytes, exactness, and blockers.

## Acceptance gates

- Candidate MTP output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate committed-token behavior is stable across measured trials and does
  not reproduce the XR24 `code_review_rust_4k_001` divergence.
- Candidate reduces selected MTP decode phase by at least `5%` versus the
  native non-MTP baseline on at least one held-out workload without any
  exactness failure.
- Peak MLX stays under the tiny16 memory cliff and active KV does not increase
  beyond prior block-prefix evidence.
- The env flag remains default-off unless a later broader holdout explicitly
  promotes it.

## Non-goals

- Do not change assistant drafting, sampling, adapters, compressed KV, server
  defaults, or workload policy defaults.
- Do not treat acceptance-rate improvement alone as success.
- Do not promote XR22 block-prefix rollback broadly without holdout exactness.

## Required artifacts

```text
benchmarks/out/XR31-mtp-block-prefix-partial-only-repair/blocker-baseline-normal/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR31-mtp-block-prefix-partial-only-repair/blocker-candidate-partial-only/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

If the blocker run passes and justifies holdout:

```text
benchmarks/out/XR31-mtp-block-prefix-partial-only-repair/holdout-baseline-normal/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR31-mtp-block-prefix-partial-only-repair/holdout-candidate-partial-only/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `blocked_with_evidence`.

The candidate reproduced the XR24 blocker on `code_review_rust_4k_001`; no
holdout was run.

- Workload: `code_review_rust_4k_001`.
- Seed: `20260631`.
- Context: `4096/4096`.
- Generated tokens per run: `32`.
- Baseline run: `xr15-1782917824`.
- Candidate run: `xr15-1782918010`.
- Measured records: `3`; warmup records: `1`.
- Baseline exact records: `4/4`.
- Candidate exact records: `0/4`.
- Candidate mismatch: generated token index `12`, native baseline token `100`,
  MTP token `8970`, on warmup and all measured trials.
- Candidate measured median baseline decode: `4115.377 ms`.
- Candidate measured median MTP decode phase: `5190.941 ms`.
- Candidate measured accepted/attempted draft tokens: `12/165`.
- Candidate measured rollbacks: `84`.
- Candidate measured event histogram: `accepted=0` for `72` events,
  `accepted=1` for `12` events, and no `accepted=2` events.
- Peak MLX: `9.244 GB`.
- Active KV: `403177472` bytes.

The sidecar review predicted this failure mode: XR24's blocker had zero
full-block accepts, so serially repairing only full accepts cannot fix
partial-reject drift. The env flag remains default-off and should not be
promoted.

## Completion rule

Stop when partial-only block-prefix repair has blocker-workload A/B evidence
and either holdout evidence or a documented reason not to expand.
