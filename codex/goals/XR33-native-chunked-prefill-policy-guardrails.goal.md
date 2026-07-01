# XR33 - Native chunked prefill policy guardrails

## Outcome

Establish whether the XR32 `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS=256` candidate
should be treated as a long-context-only native prefill policy candidate.

## Scope

- Baseline: `native_eval_per_layer`.
- Candidate: `native_chunked_prefill_256`.
- Workload source: `benchmarks/workloads/real-contexts/workloads.jsonl`.
- Start with a 4K guardrail to decide whether 256-token chunking should be
  excluded from shorter contexts.
- Run at least one second long-context family if 4K results do not invalidate
  the long-context policy direction.

## Required work

1. Keep runtime behavior and defaults unchanged.
2. Keep public C ABI unchanged.
3. Use the existing XR05 prefill/eval scheduling harness and XR32 benchmark
   variants.
4. Record exact commands, generated files, git SHA, deterministic workload
   seeds, context lengths, prefill p50/p95, correctness/logit deltas, peak MLX
   memory, active KV bytes, and blockers.
5. Document whether evidence supports a later adoption goal, and with what
   context threshold and exclusions.

## Acceptance Gates

- Candidate output greedy token and greedy logit match `native_eval_per_layer`
  within the existing XR05 tolerance for every measured record in a workload
  considered for inclusion.
- Candidate peak MLX memory improves by at least `5%`.
- Candidate p95 does not regress by more than `5%`.
- Candidate is not recommended for a context range where the 3-trial guardrail
  shows a p50 or p95 regression without a compelling memory cliff reason.
- No default-on runtime change is made in this goal.

## Required Artifacts

```text
benchmarks/out/XR33-native-chunked-prefill-policy-guardrails/guardrail-4k-256/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

If the long-context policy remains plausible:

```text
benchmarks/out/XR33-native-chunked-prefill-policy-guardrails/guardrail-long-repo-16k-256/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `accept_candidate` for a later native long-context prefill policy
adoption goal.

The `256` chunk candidate passed the 4K guardrail and a second 16K workload
family without changing runtime defaults.

### 4K Guardrail

- Run: `xr05-1782919758-54544000`.
- Workload: `code_review_rust_4k_001`.
- Seed: `20260631`.
- Context: `4096/4096`.
- Trials: `3`.
- Correctness: `3/3` for baseline and candidate.
- Baseline prefill p50/p95: `10923.489/11109.471 ms`.
- Candidate prefill p50/p95: `10535.555/10852.503 ms`.
- p50 improvement: `3.551%`.
- p95 regression value: `-2.313%` (candidate p95 improved).
- Peak MLX: baseline `9.212 GB`, candidate `7.281 GB`
  (`20.964%` improvement).
- Active KV: `402653184` bytes for both.
- Candidate logit delta: `0.0` on all trials.
- Decision: `accept_candidate` by memory gate.

### Long Repo 16K Guardrail

- Run: `xr05-1782919896-139768000`.
- Workload: `long_repo_pack_16k_001`.
- Seed: `20260639`.
- Context: `16384/16384`.
- Trials: `3`.
- Correctness: `3/3` for baseline and candidate.
- Baseline prefill p50/p95: `87369.038/87428.799 ms`.
- Candidate prefill p50/p95: `41889.113/50182.123 ms`.
- p50 improvement: `52.055%`.
- p95 regression value: `-42.602%` (candidate p95 improved).
- Peak MLX: baseline `21.868 GB`, candidate `7.620 GB`
  (`65.155%` improvement).
- Active KV: `603979776` bytes for both.
- Candidate logit delta: `0.125` on all trials.
- Decision: `accept_candidate`.

### Policy Boundary

Combined with XR32, `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS=256` now has
three-trial correctness and performance wins on:

- `code_review_rust_4k_001`.
- `code_review_rust_8k_001`.
- `benchmark_qa_16k_001`.
- `long_repo_pack_16k_001`.

This supports a later adoption goal for an opt-in or guarded automatic native
chunked-prefill policy. This goal made no runtime default change.

## Completion Rule

Stop when 4K threshold evidence and at least one additional long-context family
either support a scoped adoption follow-up or show why no policy should be
advanced.
