# XR25 - MTP state-only serial repair A/B

## Outcome

Determine whether XR23's correctness-restoring serial-state repair can recover
useful MTP speed by skipping vocabulary projection for intermediate committed
token replay while keeping serial-equivalent target KV and drafter state.

## Scope

- Baseline A: normal native MTP serial verifier.
- Baseline B: XR23 full serial-state repair with
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_SERIAL_STATE_REPAIR=1`.
- Candidate: XR25 state-only serial repair with
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1` plus the XR23
  serial repair flags.
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Primary blocker workload: `code_review_rust_4k_001`.
- Holdout workloads:
  - `chat_short_1k_001`
  - `code_review_rust_4k_001`
  - `benchmark_qa_4k_001`
  - `mtp_candidate_1k_001`
  - `mtp_candidate_4k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.

## Required work

1. Preserve default `gemma4_verify_tokens` behavior when env flags are absent.
2. Keep the new state-only repair path behind a default-off experimental flag.
3. Run at least a focused blocker-workload smoke against `code_review_rust_4k_001`.
4. If the smoke is exact and clears the primary performance gate, run a
   same-code A/B against the XR14 holdout set.
5. Record exact commands, git SHA, generated files, deterministic workload seeds,
   token lengths, acceptance, rollback, `draft_ms`, `verify_ms`, decode phase,
   memory, active KV bytes, and blockers.
6. Compare generated-token exactness and committed-token traces against normal
   serial MTP.

## Acceptance gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Candidate committed-token traces remain compatible with normal serial MTP.
- Candidate acceptance and rollback signatures match full serial-state repair.
- Candidate improves measured MTP decode phase over full serial-state repair by
  at least `5%` on the primary blocker workload.
- Peak MLX memory stays under the configured tiny16 gate.
- Do not promote MTP default behavior; record `accept_candidate`,
  `reject_candidate`, `needs_more_data`, or `blocked_with_evidence`.

## Non-goals

- Do not enable MTP by default.
- Do not add public C ABI surface area for the experimental helper.
- Do not change sampling, server defaults, adapters, active KV compression, or
  block sizes above `2`.
- Do not accept speedups without exactness evidence.

## Required artifacts

```text
benchmarks/out/XR25-mtp-state-only-serial-repair/full-serial-repair-smoke/records.jsonl
benchmarks/out/XR25-mtp-state-only-serial-repair/full-serial-repair-smoke/summary.json
benchmarks/out/XR25-mtp-state-only-serial-repair/normal-serial-smoke/records.jsonl
benchmarks/out/XR25-mtp-state-only-serial-repair/normal-serial-smoke/summary.json
benchmarks/out/XR25-mtp-state-only-serial-repair/state-only-repair-smoke/records.jsonl
benchmarks/out/XR25-mtp-state-only-serial-repair/state-only-repair-smoke/summary.json
benchmarks/out/XR25-mtp-state-only-serial-repair/<holdout-variant>/records.jsonl
benchmarks/out/XR25-mtp-state-only-serial-repair/<holdout-variant>/summary.json
benchmarks/out/XR25-mtp-state-only-serial-repair/<holdout-variant>/report.md
benchmarks/out/XR25-mtp-state-only-serial-repair/<holdout-variant>/blockers.md
benchmarks/out/XR25-mtp-state-only-serial-repair/<holdout-variant>/decision.md
```

## Completion rule

Stop when state-only serial repair has same-code A/B evidence against full
serial-state repair and normal serial MTP, or when correctness/performance
blockers explain why it should not continue.
