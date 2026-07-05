# XR81 - MTP protected aggregate overhead gap

## Objective

Use XR79's protected MTP aggregate evidence to identify the concrete overhead
that must move before broad MTP default-on can be reconsidered. Produce a
trace-backed attribution artifact and next-runtime-candidate target without
weakening exactness, oracle, default-overhead, memory, or holdout requirements.

## Current Evidence

- XR79 accepted scoped chat/tool MTP evidence only. Candidate exactness,
  sequential oracle, default-disabled overhead, 4K holdouts, protected
  `mtp_candidate_1k_001` bypass, and tiny16 memory gates passed.
- XR79 protected aggregate improved `7479.958 -> 6022.716 ms`
  (`+19.482%`), below the `25%` broad default-on gate.
- XR79 selected chat/tool lanes improved `+29.237%`, with weighted acceptance
  `144/204 = 0.706`.
- Native warmup remains out-of-request/load-time shape work only. It does not
  promote request-path warmup or broad MTP default-on.
- DSpark remains parked.

## Scope

- Analyze existing XR79 candidate and combined-report artifacts.
- Quantify the protected aggregate gate gap in milliseconds and required
  selected-lane speedup if the protected bypass remains unchanged.
- Attribute selected-lane MTP decode phase across draft, verifier forward,
  repair, fallback, and first verifier pass costs.
- Preserve the distinction between acceptance rate and net speedup.
- Keep all results as a report/artifact; do not change runtime behavior in this
  goal.

## Non-Goals

- Do not enable broad MTP default-on.
- Do not weaken `mtp_candidate_1k_001` or 4K holdout bypass behavior.
- Do not change draft/verify/rollback semantics.
- Do not promote request-path warmup.
- Do not resume DSpark.

## Acceptance Criteria

1. Add a reusable report script for XR81 MTP overhead attribution.
2. Generate `benchmarks/out/XR81-mtp-protected-aggregate-overhead-gap/`
   artifacts from XR79 candidate and combined summary inputs.
3. Report the protected aggregate gate gap and selected-lane speed target.
4. Report selected-lane component attribution for draft, verifier forward,
   verifier repair, repair fallback, fallback decode, and first verifier pass.
5. Identify the dominant next runtime-candidate target with exact artifact
   paths.
6. Update `BENCHMARKS.md` and `docs/xr-current-state-review.md` with the XR81
   result and next recommendation.
7. Verify the report script compiles and regenerates deterministic artifacts.

## Verification Commands

```text
python3 -m py_compile scripts/xr81_mtp_overhead_report.py

python3 scripts/xr81_mtp_overhead_report.py \
  --candidate-summary benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/candidate-scoped-chat-tool/summary.json \
  --combined-summary benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/xr79-warmup-aware-mtp-summary.json \
  --out-dir benchmarks/out/XR81-mtp-protected-aggregate-overhead-gap

cargo fmt --all --check
git diff --check
```

## Result - 2026-07-05

Status: `needs_runtime_candidate`. XR81 does not change runtime behavior or MTP
defaults. It converts the XR79 protected-aggregate miss into a concrete runtime
target.

Artifacts:

- Script: `scripts/xr81_mtp_overhead_report.py`
- Report:
  `benchmarks/out/XR81-mtp-protected-aggregate-overhead-gap/xr81-mtp-overhead-gap.md`
- JSON:
  `benchmarks/out/XR81-mtp-protected-aggregate-overhead-gap/xr81-mtp-overhead-gap.json`

Findings:

- Protected aggregate is still `+19.482%`, below the `25%` broad gate.
- Target decode phase for the gate is `5609.968 ms`; current protected decode
  phase is `6022.716 ms`, leaving a `412.747 ms` gap.
- If the protected bypass stays unchanged, selected chat/tool decode phase must
  move from `3527.062 ms` to `3114.315 ms`, requiring selected-lane speedup of
  `+37.518%`.
- Selected MTP component median-sums: `draft_ms=344.459`,
  `verify_forward_ms=3027.162`, `verify_repair_ms=144.963`,
  `repair_fallback_ms=144.962`, and `fallback_decode_ms=0.000`.
- `verify_forward_ms` is the dominant independent component at `85.827%` of
  selected MTP decode phase. The gate gap is only `13.635%` of verifier-forward
  median-sum.
- First verifier pass excess versus later-pass p50 is `497.460 ms`, or `1.205x`
  the current gate gap. That makes first verifier pass warm/JIT/cache behavior
  the next runtime-candidate target before changing acceptance policy.

Recommendation: pursue a scoped runtime candidate that isolates and reduces
the first verifier-forward pass cost on selected MTP rows while preserving
XR79 exactness, oracle, default-overhead, memory, and holdout gates.
