# XR57 - Real MTP Top-K and Margins

## Outcome

Replace fake MTP score fields with real target-side top-k and drafter-side
confidence signals, while keeping the default decode path within the XR57
performance guard.

Decision: `accept_candidate`; enabled-path guard failed at N=2 (+1.916% > 1%) → env-gated per rule.

## Scope

- Runtime behavior change:
  - Add ABI-v4 draft scoring buffers and safe Rust `DraftToken` /
    `draft_block_with_scores`.
  - Capture target top-5, drafter top-1 logit, drafter top1-top2 margin, and
    real `draft_in_top_k` only when
    `GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1`.
  - Keep the default path token-only: target traces record top-1 and drafter
    score fields remain zero to protect decode latency.
  - Fix trace row reuse so cached lookahead steps copy the source row matching
    `step.sequence_len - 1`, not stale slot 0.
- Non-goals:
  - Do not implement adaptive-N decision logic.
  - Do not enable MTP by default.
  - Do not change sequential verifier commit semantics.

## Gates

- Exactness unchanged on the XR55/XR56 N=2/N=4 sweep slice.
- Default decode-phase regression at N=2 and N=4 is `<=1%` versus fresh
  `b9e2dbe` baseline evidence, accounting for local variance.
- Enabled real-margin records expose real target top-k where captured, degrade
  honestly to `trace_top_k=1` for rank-1 cached rows, and pass trace
  spot-checks with a raw-logit anchor.
- Drafter logits/margins match the XR54 PyTorch parity path within bf16
  tolerance.
- Provenance includes active `GEMMA4D_*` env stamps on benchmark legs.

## Commands

```text
cargo fmt --all --check
python3 -m py_compile scripts/xr54_drafter_pytorch_parity.py scripts/xr57_trace_spotcheck.py
cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr54_drafter_pytorch_parity --no-run

git worktree add --detach /private/tmp/helios-xr57-baseline-b9e2dbe b9e2dbe

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir /Users/justin/Development/Helios/benchmarks/out/XR57-real-margins/baseline-main-rerun-n2n4-abs --model-path /Users/justin/Development/Helios/artifacts/models/gemma-4-12B-it-4bit --assistant-model-path /Users/justin/Development/Helios/artifacts/models/gemma-4-12B-it-qat-assistant-4bit --source-replay /Users/justin/Development/Helios/benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2,4 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir /Users/justin/Development/Helios/benchmarks/out/XR57-real-margins/baseline-main-rerun2-n2n4-abs --model-path /Users/justin/Development/Helios/artifacts/models/gemma-4-12B-it-4bit --assistant-model-path /Users/justin/Development/Helios/artifacts/models/gemma-4-12B-it-qat-assistant-4bit --source-replay /Users/justin/Development/Helios/benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2,4 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir /Users/justin/Development/Helios/benchmarks/out/XR57-real-margins/baseline-main-rerun3-n2n4-abs --model-path /Users/justin/Development/Helios/artifacts/models/gemma-4-12B-it-4bit --assistant-model-path /Users/justin/Development/Helios/artifacts/models/gemma-4-12B-it-qat-assistant-4bit --source-replay /Users/justin/Development/Helios/benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2,4 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR57-real-margins/candidate-gated-tokenonly-cached-n2n4 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2,4 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR57-real-margins/candidate-real-margins-enabled-final-n2n4 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2,4 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 GEMMA4D_XR57_TARGET_LOGITS_ANCHOR_PATH=benchmarks/out/XR57-real-margins/xr57-target-logits-anchor-final.json cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR57-real-margins/candidate-real-margins-enabled-spotcheck-final --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 0 --max-new-tokens 32 --block-sizes 2,4 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001

python3 scripts/xr57_trace_spotcheck.py --records benchmarks/out/XR57-real-margins/candidate-real-margins-enabled-spotcheck-final/records.jsonl --anchor-logits benchmarks/out/XR57-real-margins/xr57-target-logits-anchor-final.json --min-events 3 --out benchmarks/out/XR57-real-margins/xr57-trace-spotcheck-final.json

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example xr54_drafter_pytorch_parity -- --out-dir benchmarks/out/XR57-real-margins/drafter-score-parity-final --workload-id mtp_candidate_1k_001 --block-size 4
```

## Result

- Default candidate:
  `benchmarks/out/XR57-real-margins/candidate-gated-tokenonly-cached-n2n4/`.
  Run `xr15-1783103456`, exact `24/24`, measured `18/18`, no blockers,
  `trace_top_k=[1]`, zero drafter score fields, and no target top-1 mismatch.
- Clean baselines:
  `baseline-main-rerun-n2n4-abs`, `baseline-main-rerun2-n2n4-abs`, and
  `baseline-main-rerun3-n2n4-abs` from `b9e2dbe`. Their fixed-block selected
  decode means were N=2 `7012.621 ms`, N=4 `6896.168 ms`.
- Default guard:
  candidate selected decode was N=2 `6940.139 ms` (`-1.034%` versus the
  three-run baseline mean) and N=4 `6949.761 ms` (`+0.777%`). Against the
  fastest single N=4 baseline the delta is `+2.945%`, so the gate is accepted
  on mean evidence with variance disclosed.
- Enabled candidate:
  `benchmarks/out/XR57-real-margins/candidate-real-margins-enabled-final-n2n4/`.
  Run `xr15-1783104010`, exact `24/24`, measured `18/18`, no blockers,
  `trace_top_k=[5]` before the review-round raw-logit anchor rerun.
- Spot-check rerun:
  `benchmarks/out/XR57-real-margins/candidate-real-margins-enabled-spotcheck-final/`.
  Run `xr15-1783108188`, exact `18/18`, measured `18/18`, no blockers,
  `trace_top_k=[5]`.
- Enabled overhead:
  N=2 `+1.916%`, N=4 `+0.925%` selected-decode delta versus the three-run main
  mean. Real capture remains env-gated.
- Trace validation:
  `benchmarks/out/XR57-real-margins/xr57-trace-spotcheck-final.json` passed
  under coverage sampling with a raw-logit top-5 anchor: `237/237` checked
  events clean, rejected-slot coverage present, accepted-slot-beyond-0 coverage
  present, full-accept sampled, and anchor top-5 ids
  `[45518,236779,236787,1904,236751]` matched a recorded target row.
- Drafter score parity:
  `benchmarks/out/XR57-real-margins/drafter-score-parity-final/` completed.
  Run `xr54-parity-1783108431`; native draft tokens
  `[236792,236865,22592,236779]` matched PyTorch pinned and incremented tokens.
  Pinned native/PyTorch drafter logits matched within `0.5`, and margins
  matched within `0.25`. The stale XR54-R two-token reference mismatch is a
  warning, not a blocker, for XR57 block-size-4 score validation.

## Claim Boundary

Pre-XR57 margin/top-k fields are historical-fake except target greedy/top-1
fields. XR57 margins and target top-k ranks beyond top-1 are real only under
`GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1` ABI-v4 records; default-path records
intentionally keep top-1/zero score fields. XR57 delivers signals only;
adaptive-N policy logic remains wave-5.
