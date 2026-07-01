# XR32 - Native chunked prefill size sweep

## Outcome

Determine whether smaller/intermediate native prefill chunk sizes can preserve
the XR27 peak-memory win without reproducing the 8K logit correctness failure.

## Scope

- Baseline: `native_eval_per_layer`.
- Candidates: benchmark variants using existing
  `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` values `256`, `384`, `512`, `768`, and
  `1024`.
- Start with the XR27 8K sentinel workload `code_review_rust_8k_001`.
- Runtime behavior and defaults must remain unchanged.

## Required work

1. Add benchmark-only variants if needed; do not change native runtime math.
2. Keep public C ABI unchanged.
3. Record exact commands, generated files, git SHA, deterministic workload seed,
   context length, chunk size, prefill timing, logit/token correctness, peak MLX
   memory, active KV bytes, and blockers.
4. Expand beyond the 8K sentinel only if at least one candidate passes
   correctness and clears either the timing or memory gate without p95
   regression evidence.

## Acceptance gates

- Candidate output greedy token and greedy logit match `native_eval_per_layer`
  within the existing XR05 tolerance.
- Candidate peak MLX memory improves by at least `5%`.
- Candidate prefill p95 does not regress by more than `5%` in a follow-up run
  before promotion.
- No default-on runtime change is made from this sweep alone.

## Non-goals

- Do not alter the native chunked prefill implementation.
- Do not lower correctness tolerance to accept a candidate.
- Do not make chunked prefill the default from low-N evidence.

## Required artifacts

```text
benchmarks/out/XR32-native-chunked-prefill-size-sweep/8k-sentinel/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

If a candidate passes the 8K sentinel:

```text
benchmarks/out/XR32-native-chunked-prefill-size-sweep/followup-8k-<chunk>/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

If the 8K follow-up passes:

```text
benchmarks/out/XR32-native-chunked-prefill-size-sweep/sentinel-16k-<chunk>/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR32-native-chunked-prefill-size-sweep/followup-16k-<chunk>/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `accept_candidate` for long-context follow-up/adoption work.

The sweep found `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS=256` as the only tested
chunk size that passed the 8K sentinel correctness gate. Larger candidates kept
the output token but failed the greedy-logit tolerance at 8K.

### 8K Sentinel

- Run: `xr05-1782918414-251935000`.
- Workload: `code_review_rust_8k_001`.
- Seed: `20260632`.
- Context: `8192/8192`.
- Baseline: `native_eval_per_layer`.
- Candidates: `256`, `384`, `512`, `768`, `1024`.
- `256`: correctness passed; output token `100`; logit delta `0.25`;
  prefill `24085.904 ms`; peak MLX `7.383 GB`; active KV `469762048`
  bytes.
- `384`: correctness failed; logit delta `0.625`; prefill
  `21173.351 ms`; peak MLX `7.374 GB`.
- `512`: correctness failed; logit delta `0.75`; prefill `22559.788 ms`;
  peak MLX `7.594 GB`.
- `768`: correctness failed; logit delta `0.75`; prefill `23698.506 ms`;
  peak MLX `7.689 GB`.
- `1024`: correctness failed; logit delta `0.625`; prefill
  `24821.129 ms`; peak MLX `8.016 GB`.

### 8K Follow-Up

- Run: `xr05-1782918631-425299000`.
- Workload: `code_review_rust_8k_001`.
- Trials: `3`.
- Correctness: `3/3` for baseline and candidate.
- Baseline prefill p50/p95: `29970.057/30331.689 ms`.
- Candidate prefill p50/p95: `21391.980/23556.116 ms`.
- p50 improvement: `28.622%`.
- p95 regression value: `-22.338%` (candidate p95 improved).
- Peak MLX: baseline `12.763 GB`, candidate `7.383 GB`
  (`42.154%` improvement).
- Active KV: `469762048` bytes for both.
- Candidate logit delta: `0.25` on all trials.
- Decision: `accept_candidate`.

### 16K Sentinel And Follow-Up

- Sentinel run: `xr05-1782918864-920312000`.
- Follow-up run: `xr05-1782919079-433531000`.
- Workload: `benchmark_qa_16k_001`.
- Seed: `20260634`.
- Context: `16384/16384`.
- Sentinel correctness: passed for baseline and candidate, logit delta `0.0`.
- Sentinel prefill: baseline `86254.188 ms`, candidate `52544.936 ms`.
- Sentinel peak MLX: baseline `21.868 GB`, candidate `7.620 GB`.
- Follow-up trials: `3`.
- Follow-up correctness: `3/3` for baseline and candidate.
- Baseline prefill p50/p95: `86548.899/87510.637 ms`.
- Candidate prefill p50/p95: `42925.841/50762.654 ms`.
- p50 improvement: `50.403%`.
- p95 regression value: `-41.993%` (candidate p95 improved).
- Peak MLX: baseline `21.868 GB`, candidate `7.620 GB`
  (`65.155%` improvement).
- Active KV: `603979776` bytes for both.
- Candidate logit delta: `0.0` on all trials.
- Decision: `accept_candidate`.

No runtime default changed. This evidence supports a later adoption goal for a
long-context native prefill policy, likely using 256-token chunks for 8K/16K
contexts only, with additional 4K/other-family guardrails before enabling by
default.

## Completion rule

Stop when the 8K sentinel sweep identifies a correctness-clean candidate for
follow-up, or when all tested chunk sizes fail correctness/performance gates.
