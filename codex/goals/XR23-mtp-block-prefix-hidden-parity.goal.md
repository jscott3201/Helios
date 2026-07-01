# XR23 - MTP block-prefix hidden parity

## Outcome

Determine why XR22 block-prefix rollback preserves target generated tokens and
committed-token traces but changes later MTP drafter tokens and acceptance
counts.

## Scope

- Baseline: XR22 normal native MTP serial verifier.
- Candidate: XR22 `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`.
- First diagnostic axis: existing `GEMMA4D_NATIVE_DECODE_KV_EVAL` modes.
- Optional repair axis: add a default-off diagnostic state-repair flag only if
  existing eval modes cannot explain the drift.
- Workload: `mtp_candidate_1k_001`.
- Horizon: `64` generated tokens.
- Block size: `2`.

## Required work

1. Preserve default `gemma4_verify_tokens` behavior when no experimental flag is
   present.
2. Compare XR22 candidate traces across decode KV eval modes: `per_layer`,
   `end`, `selective`, and `defer`.
3. Record whether each mode preserves byte-identical target output, committed
   tokens, draft tokens, acceptance counts, rollback counts, `draft_ms`,
   `verify_ms`, net decode phase, memory, and blockers.
4. If no eval mode restores drafter-state parity, add only a narrow default-off
   diagnostic repair mode that proves whether serial-equivalent state fixes the
   drift, and measure its latency cost.
5. Keep MTP disabled by default.

## Acceptance gates

- A parity-preserving candidate must match the normal serial verifier for:
  generated tokens, committed-token trace, draft-token trace,
  `accepted_draft_tokens`, and `rollback_count`.
- Peak MLX memory stays under the configured tiny16 gate.
- Accept only if parity is restored and net-latency-guarded policy keeps at
  least `5%` measured decode-phase improvement over native non-MTP baseline.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Non-goals

- Do not enable MTP by default.
- Do not change sampling, server defaults, adapters, active KV compression, or
  block sizes above `2`.
- Do not broaden the native MLX boundary.

## Required artifacts

```text
benchmarks/out/XR23-mtp-block-prefix-hidden-parity/<variant>/records.jsonl
benchmarks/out/XR23-mtp-block-prefix-hidden-parity/<variant>/summary.json
benchmarks/out/XR23-mtp-block-prefix-hidden-parity/<variant>/report.md
benchmarks/out/XR23-mtp-block-prefix-hidden-parity/<variant>/blockers.md
benchmarks/out/XR23-mtp-block-prefix-hidden-parity/<variant>/decision.md
```

## Completion rule

Stop when the XR22 drafter-state drift has a measured explanation or when the
remaining blocker identifies what native state must be compared next.
