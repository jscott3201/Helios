# XR35 - Native chunked prefill policy holdouts

## Outcome

Validate the opt-in `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`
path on additional real-context holdout workloads before considering any
broader config or default adoption.

## Scope

- Baseline: `native_eval_per_layer`.
- Candidate: `native_chunked_prefill_policy_long_context_256`.
- Workload source: `benchmarks/workloads/real-contexts/workloads.jsonl`.
- Start with the 8K holdout `code_review_rust_8k_001`.
- Run a 16K holdout only if system memory pressure remains acceptable after the
  8K run.
- Do not run 24K or 32K in this goal unless explicitly restarted with fresh
  memory headroom.

## Required Work

1. Keep runtime behavior and defaults unchanged.
2. Keep public C ABI unchanged.
3. Use the existing XR05 prefill/eval scheduling harness and XR34 policy
   variant.
4. Record exact commands, generated files, git SHA, deterministic workload
   seeds, context lengths, prefill p50/p95, correctness/logit deltas, peak MLX
   memory, active KV bytes, system memory observations, and blockers.
5. Update `BENCHMARKS.md` with the decision and generated artifact paths.

## Acceptance Gates

- Candidate output greedy token and greedy logit match `native_eval_per_layer`
  within the existing XR05 tolerance for every measured record in an accepted
  holdout.
- Candidate peak MLX memory improves by at least `5%`.
- Candidate prefill p95 does not regress by more than `5%`.
- Active KV bytes do not increase.
- No default-on runtime change is made in this goal.

## Required Artifacts

```text
benchmarks/out/XR35-native-chunked-prefill-policy-holdouts/holdout-8k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

If memory remains acceptable after 8K:

```text
benchmarks/out/XR35-native-chunked-prefill-policy-holdouts/holdout-16k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `accept_candidate` for the 8K holdout only.

The policy path remained correct and cleared the memory/timing gate on
`code_review_rust_8k_001`. No runtime code, default behavior, or public C ABI
changed.

- Run: `xr05-1782921181-879661000`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR35-native-chunked-prefill-policy-holdouts/holdout-8k-policy --trials 3 --clear-workload-ids --workload-id code_review_rust_8k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`.
- Artifacts:
  `benchmarks/out/XR35-native-chunked-prefill-policy-holdouts/holdout-8k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Records: `6`; passed `6`; blockers: none.
- Workload: `code_review_rust_8k_001`.
- Seed: `20260632`.
- Prompt SHA-256:
  `24988dedab99e7a200035341ed0cc103d3f06ae84190777c8201d59b8590215e`.
- Context: `8192/8192`.
- Correctness: `3/3` for baseline and policy.
- Baseline prefill p50/p95: `30339.051/32421.743 ms`.
- Policy prefill p50/p95: `21993.044/27939.163 ms`.
- p50 improvement: `27.509%`.
- p95 regression value: `-13.826%` (policy p95 improved).
- Peak MLX: baseline `12.763 GB`, policy `7.383 GB`
  (`42.154%` improvement).
- Active KV: `469762048` bytes for both.
- Policy logit delta: `0.25` on all trials.

The 16K holdout was deferred in this pass. A mid-run `vm_stat` sample during
the 8K run showed only `3698` free 16 KiB pages, about `58 MiB`, and `641562`
wired pages, about `9.79 GiB`, while the user had also observed yellow memory
pressure and about `5 GB` swap. The post-run sample recovered to `676512` free
pages, but rerunning an unchunked 16K baseline under this pressure would add
risk without changing the 8K holdout decision.

## Completion Rule

Stop when the policy path has holdout evidence and `BENCHMARKS.md` records the
decision, or when correctness/runtime/memory pressure blockers show the policy
should not advance beyond opt-in smoke coverage.
