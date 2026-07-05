# XR86 - MTP protected aggregate on XR85 server prefix-warm surface

## Objective

Rerun the protected MTP aggregate on current `main` after XR85, attaching the
explicit server prefix-warm control-surface evidence to the combined report.
Decide whether the protected aggregate is now strong enough for broad MTP
default-on; if not, preserve the scoped chat/tool MTP productization path.

## Current Evidence

- XR73 and XR79 repeatedly accepted the scoped chat/tool MTP lane as exact,
  oracle-clean, default-overhead-clean, holdout-protected, and useful at about
  `+29%` selected-lane speedup.
- XR79 rejected broad MTP default-on because protected aggregate speed was
  `+19.482%`, below the `25%` gate.
- XR81 found the gap was dominated by verifier-forward cost, especially first
  verifier-pass warm/JIT/cache behavior.
- XR82 proved request-path first-verifier warmup removes most of that first
  verifier-forward spike, but the cost-accounted request-path candidate
  regressed net selected MTP phase.
- XR85 validated an explicit persistent-native server prefix-warm control
  surface and observability. It improved the first cold measured chat token but
  did not justify automatic warmup or broad native defaults.

## Scope

- Preserve the XR73/XR79 protected aggregate shape:
  - selected chat/tool adaptive MTP candidate,
  - sequential oracle,
  - MTP-disabled default-overhead probe,
  - 4K holdout bypass probe,
  - protected `mtp_candidate_1k_001` bypass.
- Extend the combined report to include XR85 server prefix-warm summary/report
  as explicit context.
- Record whether protected aggregate speed clears `25%`.
- Keep broad MTP default-off unless every protected aggregate gate passes.

## Non-Goals

- Do not implement server-side MTP execution in this slice.
- Do not enable automatic prefix warmup.
- Do not weaken `mtp_candidate_1k_001` or 4K holdout protections.
- Do not change verifier commit/rollback semantics.
- Do not resume DSpark.

## Acceptance Criteria

1. `scripts/xr73_scoped_mtp_report.py` renders XR85 server prefix-warm context
   without replacing the existing native warmup context path.
2. Candidate, oracle, default-overhead, and 4K holdout runs are generated under
   `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/`.
3. Candidate and sequential-oracle runs pass generated-token exactness for every
   measured row.
4. Default-overhead run with MTP disabled shows no draft, verify, repair,
   rollback, event, or decode-phase side effects.
5. `mtp_candidate_1k_001` and unproven 4K holdouts remain protected or bypassed
   unless their own gates pass.
6. Peak MLX memory stays below the tiny16 memory gate.
7. Broad default-on can be recommended only if protected aggregate speed is
   `>=25%` and exactness, oracle, holdout, memory, and default-overhead gates
   all pass.
8. The combined report cites XR85 server prefix-warm context and states that it
   remains explicit/off-request/default-off.
9. `BENCHMARKS.md` and `docs/xr-current-state-review.md` record the result,
   artifact paths, and next recommendation.

## Verification Commands

```text
python3 -m py_compile scripts/xr73_scoped_mtp_report.py
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
  --out-dir benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/candidate-scoped-chat-tool \
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
  --out-dir benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/sequential-oracle-scoped-chat-tool \
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
  --out-dir benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/default-overhead-mtp-disabled \
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
  --out-dir benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/holdout-bypass-4k \
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
  --title "XR86 MTP Protected Aggregate With Server Prefix-Warm Context" \
  --goal XR86-mtp-protected-aggregate-server-prefix-context \
  --candidate-summary benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/candidate-scoped-chat-tool/summary.json \
  --oracle-summary benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/sequential-oracle-scoped-chat-tool/summary.json \
  --default-overhead-summary benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/default-overhead-mtp-disabled/summary.json \
  --holdout-summary benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/holdout-bypass-4k/summary.json \
  --server-prefix-warm-summary benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/summary.json \
  --server-prefix-warm-report benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/report.md \
  --server-prefix-warm-label "XR85 server prefix warmup" \
  --out-dir benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context \
  --out-md xr86-mtp-protected-aggregate-server-prefix-context.md \
  --out-json xr86-mtp-protected-aggregate-server-prefix-context.json
```

## Completion Rule

Complete XR86 when the combined decision is recorded as `accept_candidate`,
`keep_experimental`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence`, with exact commands, artifacts, protected aggregate,
selected-lane aggregate, default-overhead evidence, holdout/oracle status,
XR85 server prefix-warm context, benchmark ledger updates, and a next-work
recommendation.

## Result - 2026-07-05

Status: `accept_candidate` for scoped MTP evidence only. Broad default-on
remains rejected because the protected aggregate is still below the `25%`
release gate.

Artifacts:

- Candidate:
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/candidate-scoped-chat-tool/`
- Sequential oracle:
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/sequential-oracle-scoped-chat-tool/`
- Default-overhead probe:
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/default-overhead-mtp-disabled/`
- 4K holdout probe:
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/holdout-bypass-4k/`
- Combined decision:
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/xr86-mtp-protected-aggregate-server-prefix-context.md`
  and
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/xr86-mtp-protected-aggregate-server-prefix-context.json`
- Overhead gap:
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/xr86-mtp-overhead-gap.md`
  and
  `benchmarks/out/XR86-mtp-protected-aggregate-server-prefix-context/xr86-mtp-overhead-gap.json`
- Server prefix-warm context:
  `benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/summary.json`

Gate status:

- Candidate run wrote `12/12` exact records with no blockers.
- Scoped selected workloads were `chat_short_1k_001:adaptive` and
  `tool_json_1k_001:adaptive`; protected `mtp_candidate_1k_001` was
  baseline-bypassed.
- Protected aggregate improved `7301.279 -> 5843.321 ms` (`+19.969%`), with
  weighted acceptance `144/204 = 0.706` and peak MLX `8.008 GB`.
- Selected chat/tool lanes alone improved `+30.138%`: chat `+29.851%` with
  `69/96` accepted/attempted and tool `+30.404%` with `75/108`.
- Sequential oracle compared `9` measured records with no missing, extra, or
  generated-token mismatches.
- Default-overhead probe wrote `12/12` exact records and `9` measured rows with
  `--disable-mtp`; it had zero attempted drafts, zero drafter/draft/verify/repair
  work, no MTP events, and `-0.000%` overhead.
- 4K holdout probe wrote `12/12` exact records and intentionally selected no
  MTP workloads for `benchmark_qa_4k_001`, `code_review_rust_4k_001`, and
  `mtp_candidate_4k_001`; all were baseline-bypassed with zero attempted drafts.
- XR85 server prefix-warm context was attached as explicit/off-request/default-off
  evidence only: two `128`-token warmups, `256` warm tokens total, and
  `4.754268 s` total warmup time. It does not justify automatic prefix warmup.
- Overhead gap remains concrete: current protected decode phase is
  `5843.321 ms`, the `25%` target is `5475.959 ms`, and the remaining gap is
  `367.362 ms`. If protected bypass stays unchanged, selected chat/tool MTP
  must move from `3379.656 -> 3012.294 ms`, requiring selected-lane speedup of
  `+37.732%`. Verifier forward remains dominant at `2898.462 ms`, `85.762%` of
  selected MTP phase.

Next recommendation:

- Productize the accepted scoped chat/tool MTP opt-in as explicit local opt-in
  with request/workload gating, observability, and rollback.
- Keep broad MTP default-on parked until protected aggregate speed clears
  `25%` with exactness, oracle, default-overhead, holdout, and memory gates.
- Keep automatic prefix warmup and DSpark parked.
