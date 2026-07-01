# XR36 - MTP block-prefix partial-reject repair

## Outcome

Decide whether the XR31 block-prefix correctness blocker can be repaired by
serial-repairing only the accepted-first/rejected-second branch, without
repeating the already rejected full-accept-only repair.

## Scope

- Baseline: normal native MTP block-2 policy on `code_review_rust_4k_001`.
- Candidate:
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  plus
  `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1`.
- The new flag may call the existing serial repair helper only when block-prefix
  verification accepted the first draft token and rejected the second.
- Keep `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_ONLY_REPAIR` semantics
  unchanged.
- Do not change defaults, public C ABI, model math, tokenizer behavior, or
  non-MTP paths.

## Required Work

1. Add the default-off native flag for partial-reject repair.
2. Keep the existing full-accept-only repair flag independent.
3. Run format/compile checks for the touched code path.
4. Run the XR15 blocker baseline and candidate on `code_review_rust_4k_001`.
5. Record exact commands, generated files, git SHA, deterministic workload seed,
   context length, exactness, event metrics, timing, peak MLX, active KV bytes,
   and blockers.
6. Update `BENCHMARKS.md` with the decision and generated artifact paths.

## Acceptance Gates

- Candidate exactness must be `4/4` byte-identical against the native baseline
  for the blocker workload.
- Candidate must not increase active KV bytes over the normal baseline.
- Candidate peak MLX must stay under the prior blocker ceiling of about
  `9.244 GB`.
- Candidate event trace must show the partial-reject branch was exercised.
- Candidate must not be promoted if it restores exactness only by regressing MTP
  phase beyond the existing XR15/XR31 native-baseline gap.
- No default-on runtime change is made in this goal.

## Required Artifacts

```text
benchmarks/out/XR36-mtp-block-prefix-partial-reject-repair/blocker-baseline-normal/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR36-mtp-block-prefix-partial-reject-repair/blocker-candidate-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `reject_candidate` for speed-policy promotion.

The new default-off
`GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1` flag restored
byte-identical exactness on the XR31 blocker, but it paid enough serial repair
cost that fixed block-2 MTP remained slower than the native baseline. The flag
remains experimental and default-off.

- Compile check:
  `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-ffi`.
- Baseline run: `xr15-1782921632`.
- Baseline command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR36-mtp-block-prefix-partial-reject-repair/blocker-baseline-normal --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`.
- Candidate run: `xr15-1782921825`.
- Candidate command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR36-mtp-block-prefix-partial-reject-repair/blocker-candidate-partial-reject --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`.
- Workload: `code_review_rust_4k_001`.
- Seed: `20260631`.
- Context: `4096/4096`.
- Baseline exactness: `4/4`.
- Candidate exactness: `4/4`.
- Candidate blockers: none.
- Candidate event histogram per record:
  `accepted=0:13`, `accepted=1:3`, `accepted=2:7`.
- Candidate accepted/attempted: `17/45` per record, `51/135` across
  measured records.
- Candidate rollbacks: `15` per record.
- Candidate active KV: `403177472` bytes, unchanged from baseline.
- Candidate peak MLX: `9.244 GB`, unchanged from baseline and within the prior
  blocker ceiling.
- Baseline fixed block-2 policy result: selected decode phase
  `3694.889 ms` vs native baseline `3445.987 ms`, speedup `-7.223%`.
- Candidate fixed block-2 policy result: selected decode phase
  `5721.069 ms` vs native baseline `4248.623 ms`, speedup `-34.657%`.

System memory pressure remained a benchmark caveat. During the candidate run,
`vm_stat` showed `4544` free 16 KiB pages, about `71 MiB`, and `425469`
compressor pages, about `6.49 GiB`, before recovering after the process exited.

## Completion Rule

Stop when the partial-reject repair either restores exactness with usable MTP
timing evidence or produces a correctness/runtime blocker that shows block-prefix
KV commit remains unsafe for this workload.
