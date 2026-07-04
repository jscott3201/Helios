# XR61 - Adaptive-N MTP policy search

## Outcome

Prove or falsify an env-gated Adaptive-N MTP policy for the tiny16 profile by
using XR56 repair-cost evidence and XR57 real margin/top-k signals, while
preserving greedy exactness, sequential-oracle commit semantics, and default-off
MTP behavior until all default-on gates pass.

Decision: `keep_experimental`.

XR61 P1 is closed as an env-gated/harness-only result. The safe-bypass
adaptive policy preserved measured exactness on the proven primary lanes and
protected 4K/protected holdouts, and the sequential-oracle differential matched
generated tokens record-by-record. It did not clear default-on gates: selected
aggregate speed was `+21.303%`, below the `25%` threshold, default-path overhead
was not remeasured, and risk review was not recorded. The earlier generic
holdout probe is retained as blocker evidence because `code_review_rust_4k_001`
failed exactness under generic adaptive `N=2`.

Result artifacts:

- `benchmarks/out/XR61-adaptive-n-mtp/candidate-adaptive-n-v2-safe-bypass/`
- `benchmarks/out/XR61-adaptive-n-mtp/candidate-adaptive-n-v2-safe-bypass-holdouts/`
- `benchmarks/out/XR61-adaptive-n-mtp/sequential-oracle-adaptive-n-v2-safe-bypass/`
- `benchmarks/out/XR61-adaptive-n-mtp/xr61-adaptive-n-summary.md`
- `benchmarks/out/XR61-adaptive-n-mtp/xr61-adaptive-n-summary.json`

## Scope

- Establish a fresh XR56-style baseline on the current branch before changing
  performance-critical MTP code.
- Capture or consume real-margin/top-k traces only under
  `GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1`.
- Add offline policy-search evidence under
  `benchmarks/out/XR61-adaptive-n-mtp/policy-search/`.
- Implement an adaptive policy in the XR15 benchmark harness only after the
  offline report identifies a causal policy that can be evaluated.
- Keep production/server defaults unchanged unless the full default-on gate is
  satisfied and separately reviewed.

## Non-goals

- Do not optimize DSpark.
- Do not make real-margin capture part of the default decode path.
- Do not change sequential verifier commit semantics.
- Do not broaden the native C ABI or expose raw MLX internals to Rust.
- Do not add native decode kernels or cache-layout rewrites without a measured
  stage-level `>5%` lane.

## Baseline

XR56 remains the starting comparator until a fresh XR61 baseline supersedes it:

- selected guarded policy: `chat_short_1k_001:N=3` and
  `tool_json_1k_001:N=6`;
- selected aggregate decode phase: `8458.990 -> 6598.914 ms`
  (`+21.989%`);
- selected acceptance: `144/204 = 0.706`;
- candidate and sequential oracle exactness: `72/72`;
- `repair_forward_ms = 0.0`; remaining repair cost is fallback decode;
- `mtp_candidate_1k_001` remains protected and unselected.

XR57 supplies real top-k and drafter margin signals only for records captured
with `GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1`.

## Required commands

Initial static gates:

```text
cargo fmt --all --check
git diff --check
python3 -m py_compile scripts/xr55_nblock_report.py scripts/xr57_trace_spotcheck.py scripts/xr61_adaptive_n_policy_search.py scripts/xr61_adaptive_n_report.py
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
```

Baseline command shape:

```text
GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR61-adaptive-n-mtp/baseline-xr56-policy \
  --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --adaptive-zero-accept-run 3 \
  --adaptive-min-generated-tokens 12 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --workload-id mtp_candidate_1k_001
```

Real-margin trace command shape:

```text
GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR61-adaptive-n-mtp/trace-capture-real-margins \
  --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --adaptive-zero-accept-run 3 \
  --adaptive-min-generated-tokens 12 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --workload-id mtp_candidate_1k_001
```

Offline policy search command:

```text
python3 scripts/xr61_adaptive_n_policy_search.py \
  --candidate-records benchmarks/out/XR61-adaptive-n-mtp/baseline-xr56-policy/records.jsonl \
  --real-margin-records benchmarks/out/XR61-adaptive-n-mtp/trace-capture-real-margins/records.jsonl \
  --out-dir benchmarks/out/XR61-adaptive-n-mtp/policy-search
```

## Required evidence

- `benchmarks/out/XR61-adaptive-n-mtp/baseline-xr56-policy/`
- `benchmarks/out/XR61-adaptive-n-mtp/trace-capture-real-margins/`
- `benchmarks/out/XR61-adaptive-n-mtp/policy-search/policy_report.md`
- `benchmarks/out/XR61-adaptive-n-mtp/policy-search/policy_candidates.json`
- `benchmarks/out/XR61-adaptive-n-mtp/policy-search/policy_features.jsonl`
- candidate, holdout, sequential-oracle, and final XR61 summary artifacts if
  implementation proceeds beyond offline analysis.

## Default-on gate

MTP can be proposed for default-on only if all of these pass:

- `100%` generated-token exactness against native greedy baseline;
- `100%` candidate-vs-sequential-oracle generated-token match;
- selected aggregate decode speedup `>=25%`;
- `mtp_candidate_1k_001` and at least one 4K holdout do not regress more than
  the configured `5%` gate;
- selected peak MLX remains below the tiny16 memory cliff;
- default path overhead with adaptive and real-margin envs disabled is `<=1%`;
- real-margin overhead is disclosed and remains env-gated if above the guard;
- all benchmark records include provenance, active `GEMMA4D_*` env, model
  identity, assistant identity, command line, and artifact paths;
- `BENCHMARKS.md` records only evidence-backed claims;
- risk review confirms no accidental FFI, adapter, tokenizer, or server
  behavior broadening.

## Completion rule

Complete XR61 when Adaptive-N MTP is accepted, kept experimental, rejected, or
blocked with evidence, and any deferred server/default or native decode work is
explicitly tied to measured limiters. If no valid tiny16 `>5%` lane remains,
stop with blocker evidence instead of speculating.

Completion result:

- Status: `keep_experimental`
- Primary candidate: `9/9` measured exact, selected
  `chat_short_1k_001:adaptive` and `tool_json_1k_001:adaptive`, aggregate
  `8315.953 -> 6544.437 ms` (`+21.303%`), weighted acceptance
  `144/204 = 0.706`, peak MLX `8.008 GB`.
- Holdout: `9/9` measured exact, selected no MTP workloads, baseline-bypassed
  `code_review_rust_4k_001`, `benchmark_qa_4k_001`, and
  `mtp_candidate_4k_001`, aggregate `+0.000%`, peak MLX `9.244 GB`.
- Oracle: compared `9` measured candidate records with no generated-token
  mismatches.
- Default path: unchanged; no server/default promotion.
- Next measured limiter: accepted tokens per verifier/fallback cost. Adaptive
  policy selection alone does not exceed XR56 or the `25%` default-on gate.
