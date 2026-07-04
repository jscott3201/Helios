# XR66 - Post-XR65 MTP revisit

## Objective

Re-run the XR61 Adaptive-N MTP frontier after XR65 changed the native decode
runtime default to grouped end-of-decode KV eval. Decide whether the faster
native baseline materially changes the MTP default-on case, while preserving
generated-token exactness, sequential-oracle agreement, tiny16 memory gates, and
default-off MTP behavior unless every default-on gate passes.

## Scope

- Use the current runtime default by leaving `GEMMA4D_NATIVE_DECODE_KV_EVAL`
  unset in the primary candidate and oracle runs.
- Reuse the XR61 safe-bypass adaptive policy, source replay, real-margin
  instrumentation, and selected/protected workload set:
  - `chat_short_1k_001`,
  - `tool_json_1k_001`,
  - `mtp_candidate_1k_001`.
- Report both aggregate views:
  - the XR61-compatible protected aggregate including
    `mtp_candidate_1k_001` baseline-bypass,
  - the narrowed selected-lane aggregate for `chat_short_1k_001` and
    `tool_json_1k_001`.
- Run the sequential-oracle differential against the new candidate records.
- Compare the result against XR61 and XR63 evidence without reusing old timing
  as current evidence.

## Non-Goals

- Do not enable MTP by default from a narrowed two-lane result.
- Do not change adaptive policy selection, verifier commit semantics, native
  FFI, server defaults, or model/tokenizer behavior.
- Do not claim DSpark progress from this run.
- Do not force `GEMMA4D_NATIVE_DECODE_KV_EVAL=per_layer` except for an explicit
  historical rollback comparison.

## Acceptance Criteria

1. Candidate run writes XR15 summary, records, report, blockers, and decision
   artifacts under `benchmarks/out/XR66-post-xr65-mtp-revisit/`.
2. Candidate records are `100%` generated-token exact against the native greedy
   baseline.
3. Candidate-vs-sequential-oracle generated tokens match for all measured
   candidate records.
4. The report states whether the XR61-compatible aggregate clears the `25%`
   default-on speedup gate and whether protected/holdout behavior remains safe.
5. Peak MLX memory stays below the tiny16 memory cliff.
6. `BENCHMARKS.md` records only evidence-backed claims and preserves default-off
   wording unless all gates pass.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR66-post-xr65-mtp-revisit/candidate-adaptive-n-post-xr65 \
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
  --workload-id tool_json_1k_001 \
  --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR66-post-xr65-mtp-revisit/sequential-oracle-adaptive-n-post-xr65 \
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
  --workload-id tool_json_1k_001 \
  --workload-id mtp_candidate_1k_001

python3 scripts/xr61_adaptive_n_report.py \
  --policy-candidates benchmarks/out/XR61-adaptive-n-mtp/policy-search/policy_candidates.json \
  --baseline-summary benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trace-summary benchmarks/out/XR61-adaptive-n-mtp/trace-capture-real-margins/summary.json \
  --candidate-summary benchmarks/out/XR66-post-xr65-mtp-revisit/candidate-adaptive-n-post-xr65/summary.json \
  --holdout-summary benchmarks/out/XR61-adaptive-n-mtp/candidate-adaptive-n-v2-safe-bypass-holdouts/summary.json \
  --oracle-summary benchmarks/out/XR66-post-xr65-mtp-revisit/sequential-oracle-adaptive-n-post-xr65/summary.json \
  --out-md benchmarks/out/XR66-post-xr65-mtp-revisit/xr66-post-xr65-mtp-summary.md \
  --out-json benchmarks/out/XR66-post-xr65-mtp-revisit/xr66-post-xr65-mtp-summary.json \
  --ledger-updated
```

## Completion Rule

Complete XR66 when the post-XR65 candidate is accepted for default-on, kept
experimental, rejected, or blocked with exact commands, artifacts, gate status,
and next required input. A default-on recommendation requires the full XR61 gate
set, not only a selected-lane speedup.

## Result

Decision: `keep_experimental`.

XR66 re-ran the XR61 Adaptive-N safe-bypass policy after XR65 made grouped
end-of-decode KV eval the native runtime default. The primary candidate left
`GEMMA4D_NATIVE_DECODE_KV_EVAL` unset, so the faster XR65 default was active.
Generated-token exactness passed, the sequential-oracle differential passed,
and peak MLX stayed under the tiny16 cliff, but the XR61-compatible protected
aggregate still missed the `25%` default-on speedup gate.

### Evidence

- Candidate:
  `benchmarks/out/XR66-post-xr65-mtp-revisit/candidate-adaptive-n-post-xr65/`
- Sequential oracle:
  `benchmarks/out/XR66-post-xr65-mtp-revisit/sequential-oracle-adaptive-n-post-xr65/`
- Gate summary:
  `benchmarks/out/XR66-post-xr65-mtp-revisit/xr66-post-xr65-mtp-summary.md`
  and `.json`

### Candidate Result

The candidate wrote `12/12` exact records, `9/9` measured exact records, and no
blockers at git `8253052386a0fcbd256d7535eff216cd86214f17`. Active env capture
included MTP adaptive-N, block-prefix rollback, lazy second draft, real margins,
required MLX, and native graph, with no `GEMMA4D_NATIVE_DECODE_KV_EVAL` entry.

| Aggregate | Baseline decode | Selected decode | Speedup | Acceptance |
|---|---:|---:|---:|---:|
| Protected XR61-compatible aggregate | `7904.818 ms` | `6297.468 ms` | `+20.334%` | `144/204 = 0.706` |
| Selected chat/tool lanes only | `5179.439 ms` | `3572.089 ms` | `+31.033%` | `144/204 = 0.706` |

Per selected lane:

| Workload | Baseline decode | MTP decode phase | Speedup | Acceptance |
|---|---:|---:|---:|---:|
| `chat_short_1k_001` | `2483.292 ms` | `1746.984 ms` | `+29.650%` | `69/96 = 0.719` |
| `tool_json_1k_001` | `2696.147 ms` | `1825.106 ms` | `+32.307%` | `75/108 = 0.694` |
| `mtp_candidate_1k_001` | `2725.379 ms` | `2725.379 ms` | `+0.000%` | baseline-bypassed |

### Oracle And Gate Status

The fresh sequential-oracle run compared `9` measured candidate records and had
no missing, extra, or mismatched generated-token records. Peak candidate MLX was
`8.008 GB`. The selected chat/tool slice remains promising, but it is too narrow
for broad default-on. The protected aggregate dropped from XR61's `+21.303%` to
XR66's `+20.334%`, so the faster native default makes the broad MTP default-on
case weaker, not stronger.

No MTP runtime/default/server behavior changed.
