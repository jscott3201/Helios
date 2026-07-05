# XR79 - Warmup-aware MTP protected aggregate

## Objective

Rerun the protected MTP aggregate after XR78 with explicit native-tail and
warmup claim boundaries. Preserve XR73's accepted scoped chat/tool opt-in while
keeping broad MTP default-on parked unless the protected aggregate clears the
release gate.

## Current Evidence

- XR73 accepted explicit scoped chat/tool MTP opt-in only. The candidate was
  exact, oracle-clean, default-overhead-clean, and holdout-protected, but broad
  default-on stayed unsupported because the protected aggregate was only
  `+19.235%`, below the `25%` gate.
- XR78 accepted native amortized warmup for the chat first-token tail only. The
  chat raw p99/max tail improved `387.059 -> 92.292 ms`, but the warmup event
  p50 was `3843.020 ms` and remains an out-of-request/load-time shape, not a
  request-path policy.
- The 4K code XR78 row did not reproduce the native tail and did not earn a
  warmup promotion claim.
- DSpark remains parked because XR60 was exact but not speed- or memory-viable
  on tiny16.

## Scope

- Rerun the XR73 MTP matrix on current `main`.
- Preserve generated-token exactness against native greedy output.
- Preserve sequential-oracle agreement for measured MTP rows.
- Preserve `mtp_candidate_1k_001` protected holdout behavior and 4K holdout
  bypass behavior.
- Prove the default path remains clean when MTP is disabled.
- Add a combined report that includes XR78 native warmup context without
  claiming request-path warmup.
- Record whether the protected aggregate clears or misses the `25%` broad
  default-on gate.

## Non-Goals

- Do not enable broad default-on MTP unless every protected aggregate gate
  passes.
- Do not weaken holdout protection to improve aggregate numbers.
- Do not change verifier commit/rollback semantics.
- Do not promote native warmup to a server/request policy.
- Do not resume DSpark.

## Acceptance Criteria

1. Candidate, oracle, default-overhead, and 4K holdout runs are generated under
   `benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/`.
2. Candidate and sequential-oracle runs pass generated-token exactness for every
   measured row.
3. Default-overhead run with MTP disabled shows no draft, verify, repair,
   rollback, event, or decode-phase side effects.
4. Protected aggregate including `mtp_candidate_1k_001` is reported alongside
   the narrowed selected chat/tool aggregate.
5. `mtp_candidate_1k_001` and unproven 4K holdouts remain protected or bypassed
   unless their own gates pass.
6. Peak MLX memory stays below the tiny16 memory gate.
7. Broad default-on can be recommended only if protected aggregate speed is
   `>=25%` and exactness, oracle, holdout, memory, and default-overhead gates
   all pass.
8. The combined report cites XR78 native warmup context and states that native
   warmup remains out-of-request/load-time shape work.
9. `BENCHMARKS.md` and `docs/xr-current-state-review.md` record the result,
   artifact paths, and next recommendation.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
python3 -m py_compile scripts/xr73_scoped_mtp_report.py
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 \
GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 \
cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- \
  --out-dir benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/candidate-scoped-chat-tool \
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
  --out-dir benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/sequential-oracle-scoped-chat-tool \
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
  --out-dir benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/default-overhead-mtp-disabled \
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
  --out-dir benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/holdout-bypass-4k \
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
  --title "XR79 Warmup-Aware MTP Protected Aggregate" \
  --goal XR79-warmup-aware-mtp-protected-aggregate \
  --candidate-summary benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/candidate-scoped-chat-tool/summary.json \
  --oracle-summary benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/sequential-oracle-scoped-chat-tool/summary.json \
  --default-overhead-summary benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/default-overhead-mtp-disabled/summary.json \
  --holdout-summary benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/holdout-bypass-4k/summary.json \
  --native-warmup-summary benchmarks/out/XR78-native-amortized-warmup-matrix/amortized-1k-4k-trials3/summary.json \
  --native-warmup-report benchmarks/out/XR78-native-amortized-warmup-matrix/amortized-1k-4k-trials3/report.md \
  --out-dir benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate \
  --out-md xr79-warmup-aware-mtp-summary.md \
  --out-json xr79-warmup-aware-mtp-summary.json
```

## Completion Rule

Complete XR79 when the combined decision is recorded as `accept_candidate`,
`keep_experimental`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence`, with exact commands, artifacts, protected aggregate,
selected-lane aggregate, default-overhead evidence, holdout/oracle status,
native warmup context, benchmark ledger updates, and a next-work recommendation.

## Result - 2026-07-05

Status: `accept_candidate` for scoped MTP evidence only. Broad default-on remains
rejected because the protected aggregate is still below the `25%` release gate.

XR79 extended `scripts/xr73_scoped_mtp_report.py` with backwards-compatible
`--title`, `--goal`, and optional native warmup context fields, then reran the
XR73 protected MTP matrix against current `main`. Runtime/server defaults, MTP
defaults, native warmup policy, and DSpark behavior were unchanged.

Evidence:

- Candidate:
  `benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/candidate-scoped-chat-tool/`
- Sequential oracle:
  `benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/sequential-oracle-scoped-chat-tool/`
- Default-overhead probe:
  `benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/default-overhead-mtp-disabled/`
- 4K holdout probe:
  `benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/holdout-bypass-4k/`
- Combined decision:
  `benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/xr79-warmup-aware-mtp-summary.md`
  and
  `benchmarks/out/XR79-warmup-aware-mtp-protected-aggregate/xr79-warmup-aware-mtp-summary.json`
- Native warmup context:
  `benchmarks/out/XR78-native-amortized-warmup-matrix/amortized-1k-4k-trials3/summary.json`

Gate status:

- Candidate run wrote `12/12` exact records with no blockers.
- Scoped selected workloads were `chat_short_1k_001:adaptive` and
  `tool_json_1k_001:adaptive`; protected `mtp_candidate_1k_001` was
  baseline-bypassed.
- Protected aggregate improved `7479.958 -> 6022.716 ms` (`+19.482%`), with
  weighted acceptance `144/204 = 0.706` and peak MLX `8.008 GB`.
- Selected chat/tool lanes alone improved `+29.237%`: chat `+30.447%` with
  `69/96` accepted/attempted and tool `+28.031%` with `75/108`.
- Sequential oracle compared `9` measured records with no missing, extra, or
  generated-token mismatches.
- Default-overhead probe wrote `12/12` exact records and `9` measured rows with
  `--disable-mtp`; it had zero attempted drafts, zero drafter/draft/verify/repair
  work, no MTP events, and `-0.000%` overhead.
- 4K holdout probe wrote `12/12` exact records and intentionally selected no MTP
  workloads for `benchmark_qa_4k_001`, `code_review_rust_4k_001`, and
  `mtp_candidate_4k_001`; all were baseline-bypassed with zero attempted drafts.
- XR78 native warmup context is included only as claim boundary evidence: chat
  p99 tail improved `76.155%`, but the warmup event p50 was `3843.020 ms`, so
  warmup remains out-of-request/load-time shape work and does not justify a
  request-path policy.

Next recommendation:

- Productize the accepted scoped chat/tool MTP opt-in if the priority is
  shippable value.
- If the priority is broad theoretical max, attack the protected aggregate gap
  directly by reducing MTP draft/verify overhead while preserving holdout,
  oracle, default-overhead, exactness, and memory gates.
- Keep DSpark parked.
