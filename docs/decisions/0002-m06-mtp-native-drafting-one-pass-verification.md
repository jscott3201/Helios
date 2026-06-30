# Decision Record: M06 MTP Native Drafting With One-Pass Verification

- Status: accepted
- Date: 2026-06-30
- Milestone: M06

## Context

M06 requires Gemma 4 MTP assistant loading and exact greedy speculative decoding with rollback. The native runtime can validate target artifacts, call the MLX-LM helper path, and run an opt-in native target graph. The native graph now preserves an opaque last-hidden owner and final full-attention/sliding-attention shared KV views in the KV cache.

The public MTP assistant artifact uses a distinct `gemma4_unified_assistant` config with a 4-layer text model, `backbone_hidden_size=3840`, `num_kv_shared_layers=4`, and assistant-specific tensor names. Accepting a target-model manifest as a drafter would hide the missing native integration.

## Decision

Implement and test the M06 speculative decoding state machine, metrics, rollback, auto-disable, TUI payload, target hidden/shared-view materialization, and FFI contracts now. Strict drafter load validates the real assistant artifact shape and returns a drafter handle only for assistant manifests.

The native drafter now loads real assistant tensors and `gemma4_mtp_draft_block` executes block-size 1 and 2 assistant draft generation against the materialized target hidden/shared KV views. `gemma4_verify_tokens` runs one native target forward over the accepted prefix plus draft block, compares draft tokens against the target greedy tokens at the corresponding positions, commits matching draft tokens, commits the target greedy fallback token on first mismatch, and never appends rejected draft tokens to the cache.

Native full-attention target layers use causal masking for multi-token forwards so verification logits for prefix positions cannot attend to future draft tokens.

## Consequences

- Fixture exactness and M06 acceptance metrics are available now.
- The TUI can show MTP acceptance, rollback, and auto-disable state.
- Native Gemma 4 assistant draft generation now runs for real artifacts in the opt-in native graph.
- Real Gemma 4 target+assistant draft verification now has exact accept/rollback semantics for block size 1/2.
- KV-backed verification remains a performance follow-up, not a correctness prerequisite for M06.

## Evidence

- Commands:
  - `cargo test -p gemma4d-ffi`
  - `hf download mlx-community/gemma-4-12B-it-qat-assistant-4bit --local-dir artifacts/models/gemma-4-12B-it-qat-assistant-4bit`
  - `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_FULL_MODEL_TESTS=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo test -p gemma4d-ffi native_graph_prefills_one_token_when_explicitly_enabled -- --nocapture`
  - `cargo test -p gemma4d-engine --all-targets`
  - `cargo run -p gemma4d-engine --example mtp_fixture -- --out benchmarks/out/M06/mtp-fixture-report.json`
- Files:
  - `native/gemma4_mlx/src/model_manifest.cc`
  - `native/gemma4_mlx/src/runtime.cc`
  - `crates/gemma4d-engine/src/lib.rs`
  - `crates/gemma4d-ffi/src/lib.rs`
  - `docs/evidence/M06.md`
  - `docs/evidence/M06-compliance.md`
- References:
  - `spec/05-speculative-decoding-mtp.md`
  - `milestones/M06-mtp-speculative-decoding.md`
  - Hugging Face Transformers `Gemma4UnifiedAssistantConfig`
  - Hugging Face Transformers `SinglePositionMultiTokenCandidateGenerator`
  - `mlx-community/gemma-4-12B-it-qat-assistant-4bit` config and safetensors header
