# XR73 - Scoped MTP chat/tool opt-in

## Objective

Turn the repeatedly strong MTP chat/tool evidence into a safe scoped opt-in or
workload-gated path without enabling broad default-on MTP. XR73 should run after
XR72 unless the user explicitly chooses to work on MTP first.

## Current Evidence

- XR66 kept Adaptive-N MTP exact and oracle-clean after XR65 changed the native
  decode runtime default. Selected chat/tool lanes improved
  `5179.439 -> 3572.089 ms` (`+31.033%`), but the protected aggregate including
  `mtp_candidate_1k_001` was only `+20.334%`.
- XR70 reran the MTP side-effect matrix after the full-attention KV update
  candidate. Selected chat/tool lanes stayed strong at `+30.784%`, but the
  protected aggregate was only `+19.845%`, below the `25%` broad default-on
  gate.
- Holdout protection is essential: `mtp_candidate_1k_001` repeatedly prevents
  broad promotion even when selected chat/tool lanes win.
- MTP must remain disabled by default unless exactness, oracle, holdout,
  memory, no-default-overhead, and aggregate gates all pass.

## Scope

- Design and evaluate an explicit opt-in or workload-gated chat/tool MTP path.
- Preserve the existing default-off behavior for all unselected workloads.
- Preserve generated-token exactness against native greedy output.
- Preserve sequential-oracle agreement for measured MTP rows.
- Preserve `mtp_candidate_1k_001` holdout protection and 4K holdout bypass
  behavior.
- Prove default-path overhead is zero or indistinguishable when MTP is disabled.
- Keep artifacts under `benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/`.

## Non-Goals

- Do not enable broad default-on MTP from selected-lane evidence alone.
- Do not weaken holdout protection to improve aggregate numbers.
- Do not change verifier commit semantics without a separate correctness gate.
- Do not combine DSpark with this milestone.
- Do not expose untrusted remote adapters, assistant weights, or code through a
  client request.

## Acceptance Criteria

1. The opt-in or gated path is explicit, documented, and disabled on the default
   path.
2. Default-path runs with MTP disabled show no measurable overhead and no extra
   MTP side effects.
3. Candidate and sequential-oracle runs pass generated-token exactness for every
   measured row.
4. The protected aggregate including `mtp_candidate_1k_001` is reported
   alongside the narrowed selected chat/tool aggregate.
5. `mtp_candidate_1k_001` and unproven 4K holdouts remain protected or bypassed
   unless their own gates pass.
6. Peak MLX memory stays below the tiny16 memory gate.
7. Broad default-on can be recommended only if the protected aggregate clears
   `25%` and all exactness, oracle, holdout, memory, and default-overhead gates
   pass.
8. `BENCHMARKS.md` records the result and exact artifact paths.

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
  --out-dir benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/candidate-scoped-chat-tool \
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
  --out-dir benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/sequential-oracle-scoped-chat-tool \
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
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/default-overhead-mtp-disabled \
  --source-replay benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --disable-mtp \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/holdout-bypass-4k \
  --source-replay benchmarks/out/XR56-repair-cost/candidate-retro-prefix/summary.json \
  --trials 3 \
  --warmups 1 \
  --max-new-tokens 32 \
  --block-sizes 1,2,3,4,6,8 \
  --adaptive-policy xr61-real-margin-v1 \
  --adaptive-zero-accept-run 3 \
  --adaptive-min-generated-tokens 12 \
  --clear-workload-ids \
  --workload-id benchmark_qa_4k_001 \
  --workload-id code_review_rust_4k_001 \
  --workload-id mtp_candidate_4k_001

python3 scripts/xr73_scoped_mtp_report.py \
  --candidate-summary benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/candidate-scoped-chat-tool/summary.json \
  --oracle-summary benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/sequential-oracle-scoped-chat-tool/summary.json \
  --default-overhead-summary benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/default-overhead-mtp-disabled/summary.json \
  --holdout-summary benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/holdout-bypass-4k/summary.json \
  --out-dir benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in
```

## Completion Rule

Complete XR73 when the scoped MTP decision is recorded as `accept_candidate`,
`keep_experimental`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence`, with exact commands, artifacts, protected aggregate,
selected-lane aggregate, default-overhead evidence, and holdout/oracle status.

## Result - 2026-07-05

Status: `accept_candidate` for explicit scoped chat/tool opt-in only. Broad
default-on remains rejected because the protected aggregate is still below the
`25%` release gate.

XR73 added `--disable-mtp` to the XR15 benchmark as a default-overhead probe and
added `scripts/xr73_scoped_mtp_report.py` to combine scoped candidate, oracle,
default-overhead, protected aggregate, selected-lane aggregate, and holdout
status. Runtime/server defaults were not changed; MTP remains explicit,
scoped, and default-off.

Evidence:

- Candidate:
  `benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/candidate-scoped-chat-tool/`
- Sequential oracle:
  `benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/sequential-oracle-scoped-chat-tool/`
- Default-overhead probe:
  `benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/default-overhead-mtp-disabled/`
- 4K holdout probe:
  `benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/holdout-bypass-4k/`
- Combined decision:
  `benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/xr73-scoped-mtp-summary.md`
  and `benchmarks/out/XR73-scoped-mtp-chat-tool-opt-in/xr73-scoped-mtp-summary.json`

Gate status:

- Candidate run wrote `12/12` exact records with no blockers.
- Scoped selected workloads were `chat_short_1k_001:adaptive` and
  `tool_json_1k_001:adaptive`; protected `mtp_candidate_1k_001` was
  baseline-bypassed.
- Protected aggregate improved `7523.808 -> 6076.627 ms` (`+19.235%`), with
  weighted acceptance `144/204 = 0.706` and peak MLX `8.008 GB`.
- Selected chat/tool lanes alone improved `+28.820%`: chat `+30.203%` with
  `69/96` accepted/attempted and tool `+27.499%` with `75/108`.
- Sequential oracle compared `9` measured records with no missing, extra, or
  generated-token mismatches.
- Default-overhead probe wrote `12/12` exact records and `9` measured rows with
  `--disable-mtp`; it had zero attempted drafts, zero drafter/draft/verify/repair
  work, no MTP events, and `-0.000%` overhead.
- 4K holdout probe wrote `12/12` exact records and intentionally selected no MTP
  workloads for `benchmark_qa_4k_001`, `code_review_rust_4k_001`, and
  `mtp_candidate_4k_001`; all were baseline-bypassed with zero attempted drafts.
- Combined report scoped gates passed and broad default support was `false`
  because `protected_speed_ge_gate` was `false`.

Next input: XR74 should run the native default-readiness sweep over server
default wiring, admission/tokenizer guardrails, tiny16 sentinels, operator
observability, rollback/env flags, and benchmark-ledger cleanup.
