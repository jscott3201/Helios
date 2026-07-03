# XR56 - Repair-path cost

## Outcome

Eliminate the redundant exact-prefix repair forward in the default-off native
MTP block-prefix path by reusing the first batched verify forward's prefix KV
tensors, while preserving greedy exactness and sequential-oracle commit
semantics.

Decision: pending fresh main baseline and candidate sweep.

## Scope

- Runtime behavior change:
  - Split `verify_repair_ms` into `repair_clone_ms`, `repair_forward_ms`, and
    `repair_fallback_ms` in the FFI result and XR15 harness records.
  - On partial block-prefix accepts, materialize the accepted-prefix KV by
    slicing tensors captured during the first target block forward.
  - Delete the second full block forward from `commit_prefix_repaired_state`;
    only the fallback token decode remains on that path.
- Evidence:
  - Fresh baseline leg is `main` at `fa50bd0`.
  - Candidate uses `feature/xr56-repair-cost` with
    `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` and
    `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`.
  - Sequential oracle omits block-prefix rollback and compares generated tokens
    for every swept block size.
- Non-goals:
  - Do not implement XR57 real top-k/margin instrumentation.
  - Do not enable MTP by default; a default-on flip is a wave-5 decision.
  - Do not change sequential verifier commit semantics.
  - Do not change the `BATCH_VERIFY` block-size-2 startup blocker.

## Predictions

- N=4 fixed-block speedup rises from XR55 `+7.151%` to at least `+15%`.
- The guarded optimum shifts to N=4 for at least one selected workload.
- Aggregate selected speedup rises from XR55 `+20.371%` toward the `25%`
  default-on gate. If it reaches `>=25%` with exactness intact, accept the
  candidate and flag MTP-default-on as the wave-5 headline decision.
- N=6 and N=8 improve but stay acceptance-bound; the curve should flatten, not
  invert.
- `mtp_candidate_1k_001` remains auto-disabled or unselected and receives its
  own ledger row.

## Gates

- Rung-10 greedy exactness must hold for block sizes `{1,2,3,4,6,8}`.
- Candidate generated tokens must match the sequential verifier path for every
  swept block size.
- Evidence records must include shared `capture_build_provenance()` fields and
  nonempty `build_provenance.gemma4d_env`.
- Report, per block size:
  - fixed-block speedup;
  - `verify_ms`, `verify_forward_ms`, `verify_repair_ms`;
  - `repair_clone_ms`, `repair_forward_ms`, `repair_fallback_ms`;
  - tokens per verify pass excluding auto-disabled fallback records;
  - per-slot acceptance and guarded-policy selection;
  - peak MLX memory.
- BENCHMARKS.md must include a separate `mtp_candidate_1k_001` row, disclose
  that unselected workloads enter aggregate policy math at baseline latency on
  both sides, and keep the boundary line that sequential commit semantics are
  frozen while trace/ABI plumbing may change with differential evidence.

## Commands

```text
cargo fmt --all --check
python3 -m py_compile scripts/xr55_nblock_report.py
cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
cargo test -p gemma4d-server --all-targets
cargo test -p gemma4d-bench --lib

git worktree add /private/tmp/helios-xr56-main main
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir /Users/justin/Development/Helios/benchmarks/out/XR56-repair-cost/baseline-main --source-replay /Users/justin/Development/Helios/benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR56-repair-cost/candidate-retro-prefix --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR56-repair-cost/sequential-oracle-sweep --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

python3 scripts/xr55_nblock_report.py --candidate-records benchmarks/out/XR56-repair-cost/candidate-retro-prefix/records.jsonl --candidate-summary benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json --sequential-records benchmarks/out/XR56-repair-cost/sequential-oracle-sweep/records.jsonl --require-gemma4d-env --out-md benchmarks/out/XR56-repair-cost/xr56-repair-cost-summary.md --out-json benchmarks/out/XR56-repair-cost/xr56-repair-cost-summary.json
```

## Required Evidence

- `benchmarks/out/XR56-repair-cost/baseline-main/`
- `benchmarks/out/XR56-repair-cost/candidate-retro-prefix/`
- `benchmarks/out/XR56-repair-cost/sequential-oracle-sweep/`
- `benchmarks/out/XR56-repair-cost/xr56-repair-cost-summary.{md,json}`
- `BENCHMARKS.md` ledger rows and claim-boundary update.

## Completion Rule

Stop when the repair-cost change has exactness, sequential differential,
provenance/env, memory, policy, and ledger evidence, or when blockers explain
why the milestone cannot be judged.
