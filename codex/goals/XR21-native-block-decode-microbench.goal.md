# XR21 - Native block decode microbenchmark

## Outcome

Determine whether native `decode_incremental_block(2)` is faster than two
serial native `decode_incremental` calls from the same real-context prefill.

## Scope

- Add only a narrow experimental FFI wrapper needed to benchmark block decode.
- Workload: `mtp_candidate_1k_001`.
- Context: `1024` tokens.
- Block size: `2`.
- Runtime defaults and MTP policy remain unchanged.

## Required work

1. Expose an experimental native block decode call that advances a KV cache with
   committed input tokens and returns per-position greedy target tokens/logits.
2. Build a benchmark artifact comparing:
   - serial: two `decode_one` calls;
   - block: one block decode over the same two committed tokens from an
     equivalent prefilled cache.
3. Record exact commands, generated files, model identity, workload seed, token
   lengths, decode timings, greedy/logit parity, memory, and blockers.
4. Keep any block-prefix rollback or runtime policy changes out of scope.

## Acceptance gates

- Block decode greedy tokens match serial decode for every measured record.
- Block decode logits match serial logits within `0.25` absolute tolerance. A
  stricter `0.05` calibration run should be recorded if it blocks, because BF16
  block-vs-serial evaluation may differ slightly while preserving greedy
  tokens.
- Peak MLX memory stays under the configured tiny16 gate.
- Accept only if block decode p50/median is at least `10%` faster than two
  serial decode calls.
- Otherwise record `reject_candidate`, `needs_more_data`, or
  `blocked_with_evidence`.

## Non-goals

- Do not enable MTP by default.
- Do not implement KV prefix truncation or partial-accept rollback.
- Do not change normal `decode_one`, `gemma4_verify_tokens`, sampling, server
  defaults, adapters, or active KV compression.

## Required artifacts

```text
benchmarks/out/XR21-native-block-decode-microbench/records.jsonl
benchmarks/out/XR21-native-block-decode-microbench/summary.json
benchmarks/out/XR21-native-block-decode-microbench/report.md
benchmarks/out/XR21-native-block-decode-microbench/blockers.md
benchmarks/out/XR21-native-block-decode-microbench/decision.md
benchmarks/out/XR21-native-block-decode-microbench/tolerance-0p25/records.jsonl
benchmarks/out/XR21-native-block-decode-microbench/tolerance-0p25/summary.json
benchmarks/out/XR21-native-block-decode-microbench/tolerance-0p25/report.md
benchmarks/out/XR21-native-block-decode-microbench/tolerance-0p25/blockers.md
benchmarks/out/XR21-native-block-decode-microbench/tolerance-0p25/decision.md
```

## Completion rule

Stop when block-vs-serial decode has measured evidence, or when blockers explain
why the microbenchmark cannot be safely judged.
