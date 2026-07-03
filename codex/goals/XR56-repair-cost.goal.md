# XR56 - Repair-path cost

## Outcome

Eliminate the redundant exact-prefix repair forward in the default-off native
MTP block-prefix path by reusing the first batched verify forward's prefix KV
tensors, while preserving greedy exactness and sequential-oracle commit
semantics.

Decision: `keep_experimental`.

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

## Result

Decision: `keep_experimental`. XR56 removes the redundant accepted-prefix
repair forward, but MTP remains default-off because the guarded aggregate stayed
below the `25%` default-on gate.

- Fresh main baseline:
  `benchmarks/out/XR56-repair-cost/baseline-main/`. Run
  `xr15-1783069298`, git SHA `fa50bd0f3a640d0af2f320dd1f45563bd364d487`,
  clean dirty-diff SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
  exact `72/72`, active `GEMMA4D_*` env stamped.
- Candidate:
  `benchmarks/out/XR56-repair-cost/candidate-retro-prefix/`. Run
  `xr15-1783069895`, git SHA
  `06a2e1a7fe172acf742c6d0affa43e2bbc6f07d9`, clean dirty-diff SHA-256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
  exact `72/72`, active `GEMMA4D_*` env stamped.
- Sequential oracle:
  `benchmarks/out/XR56-repair-cost/sequential-oracle-sweep/`. Run
  `xr15-1783070446`; generated tokens matched the candidate for all `72/72`
  records.
- Final report:
  `benchmarks/out/XR56-repair-cost/xr56-repair-cost-summary.{md,json}`.
  Block coverage, workload/trial coverage, exactness, provenance with
  `gemma4d_env`, tiny16 memory, trace completeness, summary policy, and
  sequential differential gates passed.

Block sweep aggregate:

| N | Exact measured | Speedup | Acceptance | Tokens/verify | Verify ms | Verify forward ms | Verify repair ms | Repair clone ms | Repair forward ms | Repair fallback ms | Peak MLX |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `9/9` | `-7.195%` | `162/234 = 0.692` | `1.000` | `21922.8` | `21911.0` | `0.0` | `0.0` | `0.0` | `0.0` | `7.652 GB` |
| 2 | `9/9` | `+16.127%` | `165/237 = 0.696` | `1.641` | `16200.3` | `14457.6` | `1736.1` | `0.0` | `0.0` | `1736.1` | `8.009 GB` |
| 3 | `9/9` | `+20.320%` | `165/252 = 0.655` | `2.065` | `15159.6` | `13635.4` | `1518.6` | `0.0` | `0.0` | `1518.6` | `8.009 GB` |
| 4 | `9/9` | `+17.609%` | `162/270 = 0.600` | `2.783` | `15762.4` | `12268.1` | `3489.1` | `0.0` | `0.0` | `3489.1` | `8.009 GB` |
| 6 | `9/9` | `+9.013%` | `162/348 = 0.466` | `3.048` | `17739.3` | `14162.5` | `3571.9` | `0.0` | `0.0` | `3571.9` | `8.010 GB` |
| 8 | `9/9` | `+2.116%` | `162/423 = 0.383` | `3.200` | `19309.5` | `15706.0` | `3598.7` | `0.0` | `0.0` | `3598.6` | `8.010 GB` |

Guarded policy:

- `net_latency_guarded_5pct` selected `chat_short_1k_001:N=3` and
  `tool_json_1k_001:N=6`.
- Aggregate selected decode phase: `8458.990 -> 6598.914 ms`
  (`+21.989%`).
- Weighted selected acceptance: `144/204 = 0.706`.
- Peak selected MLX: `8.004 GB`.

Interpretation:

- S1 succeeded: `repair_forward_ms` is `0.0` across the measured candidate
  records, so accepted-prefix repair no longer replays a second block forward.
- Remaining `verify_repair_ms` is fallback decode (`repair_fallback_ms`).
- N=4 rose from XR55 `+7.151%` to `+17.609%`, satisfying the main repair-cost
  prediction. The guarded optimum did not shift to N=4; it selected N=3 for
  chat and N=6 for tool JSON. The tool JSON N=6 pick is within noise of N=3/N=4
  at 3 measured trials, so wave-5 should not overfit that exact block choice.
- N=6 and N=8 improved versus XR55 and remained acceptance-bound, so prediction
  4 held: the curve flattened rather than inverted.
- `mtp_candidate_1k_001` stayed unselected, auto-disabled in `18/18` measured
  records, and only reached `+0.380%` at its best fixed N.

Verification passed:

- `cargo fmt --all --check`
- `git diff --check`
- `python3 -m py_compile scripts/xr55_nblock_report.py`
- `cargo test -p gemma4d-ffi --lib`
- `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib`
- `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run`
- `cargo test -p gemma4d-bench --lib`
- `cargo test -p gemma4d-server --all-targets`
- Escalated MLX smoke:
  `benchmarks/out/XR56-repair-cost/smoke-n4-chat/`.
- Escalated MLX full baseline, candidate, and sequential-oracle commands listed
  above, with absolute model and assistant paths when run from clean
  `/private/tmp` worktrees.

## Completion Rule

Stop when the repair-cost change has exactness, sequential differential,
provenance/env, memory, policy, and ledger evidence, or when blockers explain
why the milestone cannot be judged.
