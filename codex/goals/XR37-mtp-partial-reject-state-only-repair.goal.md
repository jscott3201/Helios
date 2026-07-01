# XR37 - MTP partial-reject state-only repair

## Outcome

Decide whether the XR36 exactness repair can reduce overhead by allowing the
existing state-only serial repair mode to apply to partial-reject repairs.

## Scope

- Baseline evidence: XR36 normal baseline and XR36 partial-reject serial repair.
- Candidate:
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  plus
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1`
  plus
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1`.
- The state-only path may apply only inside the existing serial repair helper.
  For the accepted-first/rejected-second branch, it should update KV for the
  accepted draft token without producing logits, then run the full decode for
  the fallback token.
- Keep defaults, public C ABI, model math, tokenizer behavior, and non-MTP paths
  unchanged.

## Required Work

1. Reuse the existing `decode_incremental_state_only` path; do not add a new C
   ABI function.
2. Keep XR36 partial-reject repair behavior unchanged when
   `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR` is absent.
3. Run compile checks for the touched native path.
4. Run the XR15 blocker candidate on `code_review_rust_4k_001`.
5. Record exact commands, generated files, git SHA, deterministic workload seed,
   context length, exactness, event metrics, timing, peak MLX, active KV bytes,
   and blockers.
6. Update `BENCHMARKS.md` with the decision and generated artifact paths.

## Acceptance Gates

- Candidate exactness must remain `4/4` byte-identical against the native
  baseline for the blocker workload.
- Candidate must not increase active KV bytes over XR36.
- Candidate peak MLX must stay under the prior blocker ceiling of about
  `9.244 GB`.
- Candidate fixed block-2 decode-phase regression must materially improve over
  XR36's `-34.657%` result before any further MTP repair path is justified.
- No default-on runtime change is made in this goal.

## Required Artifacts

```text
benchmarks/out/XR37-mtp-partial-reject-state-only-repair/candidate-state-only-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `reject_candidate` for speed-policy promotion.

The state-only partial-reject repair preserved the XR36 exactness fix and
reduced the repair overhead, but fixed block-2 MTP still regressed the native
baseline by more than the policy gate allows.

- Compile check:
  `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-ffi`.
- Run: `xr15-1782922196`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR37-mtp-partial-reject-state-only-repair/candidate-state-only-partial-reject --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`.
- Artifacts:
  `benchmarks/out/XR37-mtp-partial-reject-state-only-repair/candidate-state-only-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Workload: `code_review_rust_4k_001`.
- Seed: `20260631`.
- Context: `4096/4096`.
- Exactness: `4/4` byte-identical.
- Blockers: none.
- Event histogram per record:
  `accepted=0:13`, `accepted=1:3`, `accepted=2:7`.
- Accepted/attempted: `17/45` per record, `51/135` across measured records.
- Rollbacks: `15` per record.
- Active KV: `403177472` bytes, unchanged from XR36.
- Peak MLX: `9.244 GB`, unchanged from XR36 and within the prior blocker
  ceiling.
- Fixed block-2 result: selected decode phase `4437.904 ms` vs native baseline
  `3812.471 ms`, speedup `-16.405%`.
- Interpretation: state-only repair materially improved on XR36's `-34.657%`
  result but still missed the `5%` speedup gate and remained slower than native.

System memory pressure remained a benchmark caveat. During the run, `vm_stat`
showed `4932` free 16 KiB pages, about `77 MiB`, and `417493` compressor pages,
about `6.37 GiB`, before recovering after the process exited.

## Completion Rule

Stop when the state-only partial-reject candidate either preserves exactness and
meaningfully reduces repair overhead, or shows the repaired block-prefix path is
still not a viable MTP speed path.
