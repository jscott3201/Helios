# XR38 - MTP lazy draft with partial-reject repair

## Outcome

Decide whether the existing lazy-second-draft optimization becomes useful when
combined with the XR37 exact block-prefix partial-reject state-only repair.

## Scope

- Comparator: XR37
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  plus
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1`
  plus
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1`.
- Candidate: comparator plus
  `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`.
- Workload: `code_review_rust_4k_001`.
- No runtime code changes are expected for this goal; use existing flags.
- Keep defaults, public C ABI, model math, tokenizer behavior, and non-MTP paths
  unchanged.

## Required Work

1. Run the candidate with the XR15 real-context MTP harness.
2. Compare against XR37's same-workload evidence.
3. Record exact commands, generated files, git SHA, deterministic workload seed,
   context length, exactness, event metrics, attempted/accepted draft tokens,
   timing, peak MLX, active KV bytes, and blockers.
4. Update `BENCHMARKS.md` with the decision and generated artifact paths.

## Acceptance Gates

- Candidate exactness must remain `4/4` byte-identical against the native
  baseline.
- Candidate must not increase active KV bytes over XR37.
- Candidate peak MLX must stay under the prior blocker ceiling of about
  `9.244 GB`.
- Candidate attempted draft tokens should decrease on first-reject events.
- Candidate fixed block-2 decode-phase regression must materially improve over
  XR37's `-16.405%` result and clear the net speed gate before promotion.
- No default-on runtime change is made in this goal.

## Required Artifacts

```text
benchmarks/out/XR38-mtp-lazy-draft-partial-reject-repair/candidate-lazy-state-only-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `reject_candidate` for speed-policy promotion.

The combined candidate preserved exactness and reduced first-reject draft work,
but fixed block-2 MTP still regressed against native decode.

- Run: `xr15-1782922596`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR38-mtp-lazy-draft-partial-reject-repair/candidate-lazy-state-only-partial-reject --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`.
- Artifacts:
  `benchmarks/out/XR38-mtp-lazy-draft-partial-reject-repair/candidate-lazy-state-only-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Workload: `code_review_rust_4k_001`.
- Seed: `20260631`.
- Context: `4096/4096`.
- Exactness: `4/4` byte-identical.
- Blockers: none.
- Event histogram per record:
  `accepted=0:13`, `accepted=1:3`, `accepted=2:7`.
- Accepted/attempted: `17/32` per record, `51/96` across measured records.
- Attempted draft reduction vs XR37: `45 -> 32` per record.
- Rollbacks: `15` per record.
- Active KV: `403177472` bytes, unchanged from XR37.
- Peak MLX: `9.244 GB`, unchanged from XR37 and within the prior blocker
  ceiling.
- Fixed block-2 result: selected decode phase `4150.249 ms` vs native baseline
  `3795.764 ms`, speedup `-9.339%`.
- Interpretation: lazy drafting improved on XR37's `-16.405%` result but still
  missed the net speed gate, so the combined path remains default-off.

System memory pressure remained a benchmark caveat. During the run, `vm_stat`
showed `3998` free 16 KiB pages, about `62 MiB`, and `410411` compressor pages,
about `6.26 GiB`, before recovering after the process exited.

## Completion Rule

Stop when the combined candidate either preserves exactness and clears the MTP
speed gate, or shows that lazy drafting is still insufficient once block-prefix
partial-reject correctness is repaired.
