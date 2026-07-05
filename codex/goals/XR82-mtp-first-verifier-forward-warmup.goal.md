# XR82 - MTP first verifier-forward warmup

## Objective

Test whether XR81's first verifier-forward spike is a native warm/JIT/cache
effect by adding a cost-accounted, cloned-cache warmup before measured MTP
verification. Preserve live cache semantics and exactness while deciding
whether this is a viable request-path runtime candidate or only a lower-level
native warmup/materialization target.

## Current Evidence

- XR81 found the protected aggregate needs `412.747 ms` more selected-lane
  reduction to clear the `25%` broad gate if protected bypass behavior stays
  unchanged.
- XR81 attributed selected MTP decode phase mostly to `verify_forward_ms`
  (`3027.162 ms`, `85.827%` of selected phase).
- XR81 found first verifier-pass excess versus later-pass p50 of `497.460 ms`,
  enough on paper to cover the current gate gap.
- XR30 already rejected direct first-reject as a promotion path: it did not
  improve net decode phase and was not failure-atomic on live KV mutation.

## Scope

- Add a default-off XR15 harness experiment controlled by
  `GEMMA4D_EXPERIMENTAL_MTP_FIRST_VERIFY_WARMUP=1`.
- After prefill, export/import a cloned KV snapshot into a temporary cache,
  perform one uncommitted target decode on the imported cache, record its cost
  and memory, then drop the temporary cache before measured MTP starts.
- Include the warmup cost in `mtp.decode_phase_ms` so request-path claims remain
  cost-accounted.
- Compare a focused selected chat/tool baseline against the warmup candidate.
- Preserve generated-token exactness and generated-token parity against the
  baseline selected-lane run.

## Non-Goals

- Do not change native C ABI.
- Do not mutate live verifier KV during the warmup.
- Do not enable broad MTP default-on.
- Do not weaken `mtp_candidate_1k_001` or 4K holdout protections.
- Do not resume DSpark.

## Acceptance Criteria

1. Add cost-accounted XR15 fields for pre-verifier warmup timing and memory.
2. Generate selected chat/tool baseline and warmup candidate artifacts under
   `benchmarks/out/XR82-mtp-first-verifier-forward-warmup/`.
3. Add a comparison report that checks exactness, generated-token parity, first
   verifier-forward delta, preverify warmup cost, net decode-phase delta, and
   peak memory.
4. Recommend promotion only if cost-accounted selected-lane MTP decode phase
   improves by at least `5%` with exactness/parity intact.
5. Update `BENCHMARKS.md` and `docs/xr-current-state-review.md`.

## Verification Commands

```text
python3 -m py_compile scripts/xr82_first_verify_warmup_report.py
cargo fmt --all --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR82-mtp-first-verifier-forward-warmup/baseline-selected-chat-tool \
  --source-replay benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --adaptive-policy xr61-real-margin-v1 \
  --adaptive-zero-accept-run 3 \
  --adaptive-min-generated-tokens 12 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
GEMMA4D_EXPERIMENTAL_MTP_FIRST_VERIFY_WARMUP=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR82-mtp-first-verifier-forward-warmup/candidate-first-verify-warmup \
  --source-replay benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --adaptive-policy xr61-real-margin-v1 \
  --adaptive-zero-accept-run 3 \
  --adaptive-min-generated-tokens 12 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001

python3 scripts/xr82_first_verify_warmup_report.py \
  --baseline-summary benchmarks/out/XR82-mtp-first-verifier-forward-warmup/baseline-selected-chat-tool/summary.json \
  --candidate-summary benchmarks/out/XR82-mtp-first-verifier-forward-warmup/candidate-first-verify-warmup/summary.json \
  --out-dir benchmarks/out/XR82-mtp-first-verifier-forward-warmup
```

## Result - 2026-07-05

Status: `warmup_hypothesis_supported_net_rejected`.

Artifacts:

- Baseline:
  `benchmarks/out/XR82-mtp-first-verifier-forward-warmup/baseline-selected-chat-tool/`
- Candidate:
  `benchmarks/out/XR82-mtp-first-verifier-forward-warmup/candidate-first-verify-warmup/`
- Combined report:
  `benchmarks/out/XR82-mtp-first-verifier-forward-warmup/xr82-first-verify-warmup.md`
  and
  `benchmarks/out/XR82-mtp-first-verifier-forward-warmup/xr82-first-verify-warmup.json`

Evidence:

- Baseline run ID: `xr15-1783237168`.
- Candidate run ID: `xr15-1783237318`.
- Candidate exactness: `8/8` all records and `6/6` measured records.
- Generated-token parity versus baseline: `6/6`.
- Acceptance unchanged: baseline `144/204`, candidate `144/204`.
- First verifier-forward median-sum improved `801.087 -> 169.012 ms`
  (`+78.902%`).
- Cost-accounted selected MTP decode phase regressed
  `3663.085 -> 3791.790 ms` (`-3.514%`) because preverify warmup cost was
  `754.613 ms`.
- Chat improved net `1820.832 -> 1754.855 ms` (`+3.623%`) with
  `312.891 ms` warmup cost.
- Tool regressed net `1842.253 -> 2036.934 ms` (`-10.568%`) with
  `441.722 ms` warmup cost.
- Peak MLX stayed at `8.008 GB`.

Recommendation: the first verifier-forward spike is a real warm/JIT/cache
effect, but request-path preverify warmup is not a viable runtime promotion.
The next high-value native task is to move this warm cost out of the request
path or attack the lower-level first full-attention/verifier graph
materialization cost directly.
