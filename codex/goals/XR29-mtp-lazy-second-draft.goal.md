# XR29 - MTP lazy second draft A/B

## Outcome

Determine whether block-2 MTP can reduce assistant draft overhead by drafting
the second token only when the first draft token matches the cached target
greedy token.

## Scope

- Baseline: eager normal serial block-2 MTP verifier from XR15.
- Candidate: `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`.
- Workload source: XR14 replay summary plus XR15 real-context workload harness.
- Start with `mtp_candidate_1k_001`; expand only if exactness and committed-token
  trace behavior match baseline.

## Required work

1. Keep MTP disabled by default.
2. Do not use block-prefix rollback, batch verify, serial-state repair, or
   state-only repair.
3. Keep public C ABI unchanged.
4. Preserve ordinary eager `gemma4_mtp_draft_block` behavior when the env flag is
   absent.
5. Record exact commands, generated files, git SHA, deterministic workload seed,
   context length, generated-token count, draft/verify/decode-phase timing,
   attempted/accepted draft tokens, rollback count, peak MLX memory, active KV
   bytes, and blockers.

## Acceptance gates

- Candidate MTP output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate committed-token behavior matches eager serial MTP. Accepted draft
  count may match; attempted draft count should decrease only on first-token
  rejects.
- Candidate reduces selected decode phase by at least `5%` without a verify
  regression above `5%`.
- Peak MLX stays under the tiny16 memory cliff and active KV does not increase.
- The env flag remains default-off unless a later broader holdout explicitly
  promotes it.

## Non-goals

- Do not change target verification, target KV rollback, block-prefix paths,
  sampling, adapters, compressed KV, or server defaults.
- Do not change public `gemma4_verify_tokens` or `gemma4_decode_block` ABI.
- Do not treat acceptance-rate improvement alone as success.

## Required artifacts

```text
benchmarks/out/XR29-mtp-lazy-second-draft/baseline-eager-block2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR29-mtp-lazy-second-draft/candidate-lazy-block2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `reject_candidate`.

The candidate preserved exactness and reduced attempted draft work, but did not
clear the decode-phase speed gate:

- Workload: `mtp_candidate_1k_001`.
- Seed: `20260641`.
- Context: `1024/1024`.
- Generated tokens per run: `32`.
- Baseline run: `xr15-1782904888`.
- Candidate run: `xr15-1782904992`.
- Measured records: `3`; warmup records: `1`.
- Exact records: `4/4` in both runs.
- Median `draft_ms`: baseline `164.201`, candidate `134.852`.
- Median `verify_ms`: baseline `2966.086`, candidate `2952.452`.
- Median MTP decode phase: baseline `3116.653`, candidate `3087.304`.
- Candidate phase improvement: about `0.94%`, below the `5%` gate.
- Attempted draft tokens across measured trials: baseline `120`, candidate `96`.
- Accepted draft tokens across measured trials: `39` for both.
- Rollbacks across measured trials: `57` for both.
- Peak MLX: `7.665 GB` for both.
- Active KV: `352845824` bytes for both.
- Blockers: none.

No broader holdout was run because verify time still dominated and total
decode-phase improvement did not meet the gate. The env flag remains default-off.

## Completion rule

Stop when lazy block-2 drafting has same-workload A/B evidence against eager
normal serial block-2 MTP, or when compile/runtime/correctness/performance
blockers explain why it should not continue.
