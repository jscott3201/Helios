# XR52 - KV slab incremental decode

## Outcome

Replace full-attention KV cache growth by whole-cache concatenation with a
preallocated slab append path, reduce verifier staging clones on the live MTP
paths, and expose verifier timing splits for XR15 records.

## Scope

- Native target KV storage in `native/gemma4_mlx/src/native_model.cc`.
- Runtime verifier staging in `native/gemma4_mlx/src/runtime.cc`.
- C/Rust FFI step result shape for:
  - `verify_stage_ms`
  - `verify_forward_ms`
  - `verify_repair_ms`
- XR15 variance benchmark records.
- Stale M06 documentation language about the dead full-recompute verifier.

Out of scope: N>2 draft blocks, assistant/drafter changes, external
transformers ground-truth audit, and Metal kernel work.

## Reference Branch Implementation

The full experiment is preserved on `feature/xr52-kv-slab`. That reference
branch:

- Added per-layer logical KV slab metadata and slab materialization helpers.
- Full-attention incremental decode and block decode append new K/V with
  `mlx::core::slice_update` and attend over the logical slab slice.
- Sliding-attention layers keep the existing chronological window semantics and
  store only the newest logical window in a fixed-size slab. The sliding append
  remains window-bounded rather than context-bounded.
- Added `GEMMA4D_NATIVE_KV_CONCAT_FALLBACK=1` as a debug kill switch; the slab
  path is default.
- Changed `NativeKvState::clone()` to clone logical K/V views only, so in-memory
  snapshot export/import and any remaining staging clones do not preserve slab
  capacity.
- Removed dead `NativeTextModel::verify_draft_block` and
  `forward_verify_logits`.
- The selected block-prefix MTP verifier decodes the block against live KV when
  legacy serial-repair flags are disabled, using the existing prefix KV output
  for the partial-reject rollback state.
- The default serial verifier advances live KV for non-terminal verification
  passes; terminal no-lookahead remains staged because it can skip the final
  target decode.
- The runtime records verifier setup, target forward, and repair decode timing
  into `Gemma4StepResult`; XR15 emits both aggregate and per-event split fields.

## Evidence PR Retained Code

The salvage evidence branch intentionally retains only the pieces that are true
independent of the failed slab optimization:

- dead `NativeTextModel::verify_draft_block` / `forward_verify_logits` removal;
- M06 documentation amendments for the stale full-recompute verifier wording;
- verifier timing split instrumentation and XR15 record fields;
- XR52 benchmark ledger rows and blocked goal card.

It does not retain the slab storage rewrite, logical rollback changes, or
snapshot serialization changes.

## Acceptance Gates

- Plain native decode remains byte-identical to the pre-change path on the 1K
  family probes and a 4K probe.
- MTP greedy output equals non-MTP greedy for block sizes 1 and 2 with slab
  enabled.
- Chunked prefill identity is unchanged on a 4K spot-check.
- KV snapshot export/import/resume remains exact and reports logical
  `active_kv_bytes`.
- `cargo test -p gemma4d-ffi --lib`, bench tests, server tests, C++ smoke, and
  MLX-required native compile pass.
- Tiny16 memory remains under gate; slab slack is evaluated with peak MLX memory
  rather than `active_kv_bytes`.

## Evidence

Artifacts are under:

- `benchmarks/out/XR52-kv-slab-incremental/`

Verification passed:

```text
cargo fmt --all --check
cargo test -p gemma4d-ffi --lib
cargo test -p gemma4d-bench --lib
cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
cargo test -p gemma4d-server --all-targets
scripts/native-smoke.sh
scripts/mlx-diagnostics.sh
cmake --build target/mlx-diagnostics
git diff --check
```

Real-model evidence:

- `decode-baseline-main/`: pre-change `main` baseline at `32b2a43` for
  `chat_short_1k_001`, `tool_json_1k_001`, `mtp_candidate_1k_001`, and
  `code_review_rust_4k_001`.
- `decode-candidate-slab/`: exact bounded slab candidate. Token sequences
  matched `main` for all 12 compared records. Steady decode p50 improved only
  `0.39%..1.05%`, below the XR52 `>=5%` gate; peak MLX and active KV were
  unchanged.
- `mtp-selected-slab/`: XR48-style selected MTP holdout with verifier timing
  splits. Exactness was `12/12`; acceptance matched XR48 (`chat 69/96`,
  `tool_json 75/96`, `mtp_candidate 21/45`). Guarded aggregate speedup was
  `16.719%` versus XR48 `15.302%`, below the required `+5` point improvement,
  and `mtp_candidate_1k_001` still regressed by `-2.262%`.
- `decode-candidate-slab-rotating/`: diagnostic physical rotating-window
  experiment. It improved p50 by about `6%..7%` but failed byte-identical token
  parity in `9/12` compared records, so it was rejected and not retained.
- `mtp-selected-slab-chronological/`: diagnostic chronological rotating-window
  run. It stayed output-exact but drifted `chat_short_1k_001` MTP acceptance
  from XR48 `69/96` to `66/96`, so the rotating sliding path was removed.
- `instrumentation-on-main-smoke/`: evidence-branch smoke on concat storage
  with the timing split retained. `chat_short_1k_001` block-2 exactness was
  `1/1`, no blockers were recorded, split fields were emitted, and the maximum
  per-event `verify_ms - (verify_stage_ms + verify_forward_ms +
  verify_repair_ms)` absolute difference was `0.068 ms`.

## Result

Blocked with evidence. XR52 did not satisfy the promotion decision rule:

- baseline native decode improved by less than `5%`;
- `mtp_candidate_1k_001` did not clear the `5%` guarded MTP gate;
- aggregate guarded MTP speedup improved by only `+1.417` points versus XR48,
  not the required `+5` points.

The full reference branch is preserved for review, but the slab and snapshot
changes are not ready to ship as a default-on optimization. The evidence branch
keeps the verdict, stale-doc cleanup, dead verifier deletion, and verifier timing
instrumentation. XR53 is unblocked by the ruling because no XR52 baseline
re-anchor happened.
