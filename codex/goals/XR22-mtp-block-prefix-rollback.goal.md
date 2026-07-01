# XR22 - MTP block prefix rollback A/B

## Outcome

Use XR21's accepted native block decode speedup to test an exact block-2 MTP
verifier that can handle partial acceptance without falling back to fully serial
verification.

## Scope

- Baseline: normal native MTP serial verifier.
- Candidate: `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`.
- Workload: `mtp_candidate_1k_001`.
- Horizon: `64` generated tokens.
- Block size: `2`.

## Required work

1. Preserve default `gemma4_verify_tokens` behavior when the env flag is absent.
2. Capture an exact prefix KV during native block decode instead of slicing the
   post-block KV after the second draft token is appended.
3. Use block decode only when `draft_count == 2` and the first draft token is
   accepted by the already-available target `last_step`.
4. For first-token accept and second-token reject, decode the fallback token
   from the block-produced prefix KV and prove continued decode parity.
5. Record exact commands, generated files, git SHA, model identity, deterministic
   workload seed, token lengths, acceptance, rollback, `draft_ms`, `verify_ms`,
   net decode phase, memory, and blockers.
6. Keep MTP disabled by default.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate committed-token traces match the normal serial MTP verifier for the
  same workload and block size.
- Continued decode after partial rollback is byte-identical to the normal serial
  verifier.
- Peak MLX memory stays under the configured tiny16 gate.
- Accept only if net-latency-guarded policy selects MTP with at least `5%`
  measured decode-phase improvement and no correctness regressions.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Non-goals

- Do not enable MTP by default.
- Do not change sampling, server defaults, adapters, active KV compression, or
  block sizes above `2`.
- Do not use post-block KV truncation for sliding-attention rollback.

## Required artifacts

```text
benchmarks/out/XR22-mtp-block-prefix-rollback/baseline-normal-verify/records.jsonl
benchmarks/out/XR22-mtp-block-prefix-rollback/baseline-normal-verify/summary.json
benchmarks/out/XR22-mtp-block-prefix-rollback/baseline-normal-verify/report.md
benchmarks/out/XR22-mtp-block-prefix-rollback/baseline-normal-verify/blockers.md
benchmarks/out/XR22-mtp-block-prefix-rollback/baseline-normal-verify/decision.md
benchmarks/out/XR22-mtp-block-prefix-rollback/candidate-block-prefix-rollback/records.jsonl
benchmarks/out/XR22-mtp-block-prefix-rollback/candidate-block-prefix-rollback/summary.json
benchmarks/out/XR22-mtp-block-prefix-rollback/candidate-block-prefix-rollback/report.md
benchmarks/out/XR22-mtp-block-prefix-rollback/candidate-block-prefix-rollback/blockers.md
benchmarks/out/XR22-mtp-block-prefix-rollback/candidate-block-prefix-rollback/decision.md
```

## Completion rule

Stop when block-prefix rollback has measured A/B evidence against the same-code
normal verifier baseline, or when blockers explain why it cannot be safely
judged.
