# XR55 - MTP N-block generalization

## Outcome

Generalize the default-off native MTP block-prefix experiment from block size 2
to larger draft blocks, sweep `{1,2,3,4,6,8}`, and identify the guarded policy
curve on the XR48 1K real-context holdout.

Decision: pending full sweep.

## Scope

- Runtime behavior change:
  - Native MTP trace and committed-token buffers support the larger XR55 block
    sweep.
  - `gemma4_mtp_draft_block` and native target verification accept block sizes
    above 2 up to the shared trace capacity.
  - Block-prefix verification stays default-off and exact; later partial
    accepts materialize an exact accepted-prefix KV before committing fallback.
- Evidence:
  - Baseline is post-XR54 `main` at `24186cf`, block size 2.
  - Candidate uses `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` and
    `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`.
  - Sequential oracle omits block-prefix rollback and compares generated tokens
    for every swept block size.
- Non-goals:
  - Do not enable MTP by default.
  - Do not change `KvPolicy.block_size_tokens`; it is unrelated to MTP draft
    block size.
  - Do not cite `logit_margins`; current trace margins remain diagnostic debt.

## Gates

- Greedy exactness must hold for every swept block size `{1,2,3,4,6,8}` against
  native non-MTP greedy.
- Candidate generated tokens must match the sequential verifier path for every
  swept block size.
- Evidence records must include shared fail-closed build provenance from
  `capture_build_provenance()`.
- N=8 trace completeness must fail loudly on truncation and pass with actual
  eight-token draft events.
- Report, per block size:
  - total acceptance and per-slot acceptance;
  - tokens per verify pass;
  - `draft_ms`, `verify_ms`, `verify_forward_ms`, and `verify_repair_ms`;
  - net decode-phase speedup;
  - guarded-policy selection;
  - peak MLX memory.
- Flag any draft cost per draft step above `0.1` verify-units.
- Tiny16 peak memory must remain in band.

## Commands

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR55-nblock-generalization/baseline-block2 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR55-nblock-generalization/candidate-nblock-sweep --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR55-nblock-generalization/sequential-oracle-sweep --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR55-nblock-generalization/trace-n8-chat-prefix-repair --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 1 --warmups 0 --max-new-tokens 32 --block-sizes 8 --clear-workload-ids --workload-id chat_short_1k_001

python3 scripts/xr55_nblock_report.py --candidate-records benchmarks/out/XR55-nblock-generalization/candidate-nblock-sweep/records.jsonl --candidate-summary benchmarks/out/XR55-nblock-generalization/candidate-nblock-sweep/summary.json --sequential-records benchmarks/out/XR55-nblock-generalization/sequential-oracle-sweep/records.jsonl --out-md benchmarks/out/XR55-nblock-generalization/xr55-nblock-summary.md --out-json benchmarks/out/XR55-nblock-generalization/xr55-nblock-summary.json

cargo fmt --all --check
git diff --check
cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib
cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
cargo test -p gemma4d-server --all-targets
```

## Required Evidence

- `benchmarks/out/XR55-nblock-generalization/baseline-block2/`
- `benchmarks/out/XR55-nblock-generalization/candidate-nblock-sweep/`
- `benchmarks/out/XR55-nblock-generalization/sequential-oracle-sweep/`
- `benchmarks/out/XR55-nblock-generalization/trace-n8-chat-prefix-repair/`
- `benchmarks/out/XR55-nblock-generalization/xr55-nblock-summary.{md,json}`
- `BENCHMARKS.md` ledger row and claim-boundary update.

## Result

Decision: `keep_experimental`. XR55 keeps MTP default-off and expands the
measured default-off block-prefix experiment to N>2.

- Baseline post-XR54 main: `benchmarks/out/XR55-nblock-generalization/baseline-block2/`.
  Run `xr15-1783059933`, git SHA `24186cf5bc3ade2662ff2074f048268187e77dac`,
  dirty diff SHA-256
  `d52668223612b1813a989fb408f20bbd65d4d415f8bff7ac0352e6efa7aabaf9`,
  exact `12/12`, net-latency guard selected `chat_short_1k_001:N=2`
  plus `tool_json_1k_001:N=2` with aggregate `16.038%`.
- Candidate sweep:
  `benchmarks/out/XR55-nblock-generalization/candidate-nblock-sweep/`.
  Run `xr15-1783061850`, dirty diff SHA-256
  `20334da2790c9419d28828313541563ef98cb5c5167466bfca27f7f05050fa6f`,
  runner link mtime `1783061770`, exact `72/72` total and `54/54`
  measured records.
- Sequential oracle:
  `benchmarks/out/XR55-nblock-generalization/sequential-oracle-sweep/`.
  The final report compared all `72` candidate records against this sequential
  verifier path with no missing records and no generated-token mismatches.
- N=8 trace gate:
  `benchmarks/out/XR55-nblock-generalization/trace-n8-chat-prefix-repair/`.
  Actual eight-token draft events were recorded, trace completeness passed, and
  active KV bytes matched the live committed cache.
- Final report:
  `benchmarks/out/XR55-nblock-generalization/xr55-nblock-summary.{md,json}`.
  Provenance, block coverage, workload/trial coverage, greedy exactness,
  tiny16 memory, trace completeness, full-N=8 event coverage, summary policy,
  and sequential differential gates passed.

Block sweep aggregate:

| N | Exact measured | Speedup | Acceptance | Tokens/verify | Draft ms | Verify ms | Verify forward ms | Verify repair ms | Draft-step verify units | Peak MLX |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `9/9` | `-8.266%` | `162/234 = 0.692` | `1.231` | `1200.8` | `22778.1` | `22765.4` | `0.0` | `0.053` | `7.652 GB` |
| 2 | `9/9` | `+14.168%` | `165/237 = 0.696` | `2.000` | `1253.6` | `17157.7` | `15341.2` | `1808.6` | `0.044` | `8.008 GB` |
| 3 | `9/9` | `+18.054%` | `165/252 = 0.655` | `2.400` | `1283.0` | `16100.2` | `14258.0` | `1835.7` | `0.038` | `8.357 GB` |
| 4 | `9/9` | `+7.151%` | `162/270 = 0.600` | `3.097` | `1317.1` | `18921.8` | `12798.8` | `6116.7` | `0.024` | `8.357 GB` |
| 6 | `9/9` | `-1.367%` | `162/348 = 0.466` | `3.310` | `1541.0` | `20907.0` | `14642.8` | `6258.2` | `0.018` | `8.358 GB` |
| 8 | `9/9` | `-9.353%` | `162/423 = 0.383` | `3.429` | `1743.9` | `22778.6` | `16502.8` | `6270.0` | `0.015` | `8.358 GB` |

Per-slot acceptance:

- N=1: `s1=162/234 (0.692)`.
- N=2: `s1=93/144 (0.646)`, `s2=72/93 (0.774)`.
- N=3: `s1=66/120 (0.550)`, `s2=51/66 (0.773)`,
  `s3=48/66 (0.727)`.
- N=4: `s1=60/93 (0.645)`, `s2=45/60 (0.750)`,
  `s3=39/60 (0.650)`, `s4=18/57 (0.316)`.
- N=6: `s1=54/87 (0.621)`, `s2=39/54 (0.722)`,
  `s3=33/54 (0.611)`, `s4=s5=s6=12/51 (0.235)`.
- N=8: `s1=51/84 (0.607)`, `s2=36/51 (0.706)`,
  `s3=30/51 (0.588)`, `s4=s5=s6=s7=9/48 (0.188)`,
  `s8=9/45 (0.200)`.

Guarded policy:

- `net_latency_guarded_5pct` selected `chat_short_1k_001:N=3` and
  `tool_json_1k_001:N=4`.
- Aggregate selected decode phase: `8674.797 -> 6907.671 ms`
  (`+20.371%`).
- Weighted selected acceptance: `144/198 = 0.727`.
- Peak selected MLX: `8.357 GB`.

Interpretation:

- N=3 was the best fixed block in aggregate. Fixed N=4, N=6, and N=8 are not
  broad candidates because at least two held-out workloads regress.
- `mtp_candidate_1k_001` still auto-disables in every measured record and is
  not a selected workload.
- Draft cost per draft step stayed below the `0.1` verify-unit flag threshold
  for every block. The binding cost for N>=4 is exact prefix repair, not draft
  step cost.
- Larger block verify forward cost rose smoothly rather than showing a sharp
  kernel cliff; total `verify_ms` is dominated by repair for N>=4.

Verification passed:

- `cargo fmt --all --check`
- `git diff --check`
- `python3 -m py_compile scripts/xr55_nblock_report.py`
- `cargo test -p gemma4d-ffi --lib`
- `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib`
- `cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run`
- `cargo test -p gemma4d-server --all-targets`
- `cargo test -p gemma4d-bench --lib`

## Completion Rule

Stop when the N-block sweep has exactness, sequential differential, trace,
memory, and policy evidence, or when blockers explain why the milestone cannot
be judged.
