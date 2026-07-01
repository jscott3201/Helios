# XR30 - MTP direct first-reject verifier A/B

## Outcome

Determine whether MTP verification can avoid staged KV clone/copy work when the
first drafted token is already known to reject against the cached target greedy
token.

## Scope

- Baseline: normal serial MTP verifier from XR15.
- Candidate: `GEMMA4D_EXPERIMENTAL_MTP_DIRECT_FIRST_REJECT=1`.
- Workload source: XR14 replay summary plus XR15 real-context workload harness.
- Start with `mtp_candidate_1k_001`; expand only if exactness, committed-token
  behavior, and memory behavior match baseline.

## Required work

1. Keep MTP disabled by default.
2. Do not use block-prefix rollback, batch verify, in-place verify, lazy second
   draft, serial-state repair, or state-only repair for the judged candidate.
3. Keep public C ABI unchanged.
4. Preserve ordinary `gemma4_verify_tokens` behavior when the env flag is absent.
5. Record exact commands, generated files, git SHA, deterministic workload seed,
   context length, generated-token count, draft/verify/decode-phase timing,
   attempted/accepted draft tokens, rollback count, peak MLX memory, active KV
   bytes, and blockers.

## Acceptance gates

- Candidate MTP output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate committed-token behavior and accepted draft counts match normal
  serial MTP.
- Candidate reduces selected MTP decode phase by at least `5%` without a verify
  regression above `5%`.
- Peak MLX stays under the tiny16 memory cliff and active KV does not increase.
- The env flag remains default-off unless a later broader holdout explicitly
  promotes it.

## Non-goals

- Do not change assistant drafting, target block-prefix paths, sampling,
  adapters, compressed KV, or server defaults.
- Do not change public `gemma4_verify_tokens` or `gemma4_decode_block` ABI.
- Do not treat acceptance-rate improvement alone as success.

## Required artifacts

```text
benchmarks/out/XR30-mtp-direct-first-reject/baseline-normal-verify/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR30-mtp-direct-first-reject/candidate-direct-first-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `reject_candidate`.

The candidate preserved exact output and committed-token behavior, but did not
improve decode phase and is not failure-atomic enough for promotion:

- Workload: `mtp_candidate_1k_001`.
- Seed: `20260641`.
- Context: `1024/1024`.
- Generated tokens per run: `32`.
- Baseline run: `xr15-1782917374`.
- Candidate run: `xr15-1782917475`.
- Measured records: `3`; warmup records: `1`.
- Exact records: `4/4` in both runs.
- Median `draft_ms`: baseline `159.922`, candidate `163.534`.
- Median `verify_ms`: baseline `2772.472`, candidate `2779.443`.
- Median MTP decode phase: baseline `2931.063`, candidate `2936.736`.
- Candidate phase change: about `-0.19%`; below the `5%` improvement gate.
- Attempted draft tokens across measured trials: `120` for both.
- Accepted draft tokens across measured trials: `39` for both.
- First-reject events across measured trials: `24` for both.
- Rollbacks across measured trials: `57` for both.
- Peak MLX: `7.665 GB` for both.
- Active KV: `352845824` bytes for both.
- Harness blockers: none.
- Independent correctness review blocker: the direct branch mutates live target
  KV before decode success is known. A late decode/OOM/error can leave KV
  partially advanced while token/last-step metadata remains stale, unlike the
  normal staged verifier.

No broader holdout was run because the timing gate failed and the branch needs
cache-discard/failure-injection coverage before it can be considered
behavior-preserving. The env flag remains default-off.

## Completion rule

Stop when direct first-reject verification has same-workload A/B evidence
against normal serial MTP, or when compile/runtime/correctness/performance
blockers explain why it should not continue.
