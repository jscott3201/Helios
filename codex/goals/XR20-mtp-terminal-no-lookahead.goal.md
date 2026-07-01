# XR20 - MTP terminal no-lookahead phase accounting

## Outcome

Measure whether native MTP's remaining XR19 regression is mostly the unused
final target lookahead by adding an explicit experimental terminal verifier path
for benchmark use.

## Scope

- Baseline: normal native MTP verifier.
- Candidate: terminal no-lookahead verifier, called only when the current draft
  block can satisfy the caller's remaining generation budget.
- Workload: `mtp_candidate_1k_001`, matching XR15-XR19.
- Horizon: `64` generated tokens.
- Block sizes: `1` and `2`.

## Required work

1. Preserve `gemma4_verify_tokens` default behavior.
2. Add a separate experimental FFI/Rust wrapper for terminal no-lookahead
   verification.
3. The candidate must skip the lookahead only after committed tokens satisfy the
   terminal commit budget. If they do not, it must keep normal cache-advancing
   behavior so generation can continue.
4. Record terminal skip counts, exactness, acceptance, rollback count,
   `draft_ms`, `verify_ms`, net decode phase, peak memory, active KV bytes,
   commands, artifacts, seed, and token lengths.
5. Keep MTP disabled by default.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate does not use the terminal verifier when the cache must continue.
- Peak MLX memory stays under the configured tiny16 gate.
- Accept only if net-latency-guarded policy selects MTP with at least `5%`
  measured decode-phase improvement and no workload regression over the normal
  verifier baseline.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Risk note

When the terminal path skips the final lookahead, the native KV cache is not
prepared for continuation after the final committed token. The caller must
discard that cache. This path is only valid for generation termination and must
not become the default verifier.

## Non-goals

- Do not enable MTP by default.
- Do not implement block-prefix KV rollback.
- Do not change sampling, server defaults, adapters, active KV compression, or
  verifier behavior for non-terminal calls.
- Do not combine with XR16-XR18 experimental flags.

## Required artifacts

```text
benchmarks/out/XR20-mtp-terminal-no-lookahead/baseline-normal-verify/records.jsonl
benchmarks/out/XR20-mtp-terminal-no-lookahead/baseline-normal-verify/summary.json
benchmarks/out/XR20-mtp-terminal-no-lookahead/baseline-normal-verify/report.md
benchmarks/out/XR20-mtp-terminal-no-lookahead/baseline-normal-verify/blockers.md
benchmarks/out/XR20-mtp-terminal-no-lookahead/baseline-normal-verify/decision.md
benchmarks/out/XR20-mtp-terminal-no-lookahead/candidate-terminal-no-lookahead/records.jsonl
benchmarks/out/XR20-mtp-terminal-no-lookahead/candidate-terminal-no-lookahead/summary.json
benchmarks/out/XR20-mtp-terminal-no-lookahead/candidate-terminal-no-lookahead/report.md
benchmarks/out/XR20-mtp-terminal-no-lookahead/candidate-terminal-no-lookahead/blockers.md
benchmarks/out/XR20-mtp-terminal-no-lookahead/candidate-terminal-no-lookahead/decision.md
```

## Completion rule

Stop when terminal no-lookahead has measured A/B evidence against the same-code
normal verifier baseline, or when blockers explain why the accounting question
cannot be safely judged.
