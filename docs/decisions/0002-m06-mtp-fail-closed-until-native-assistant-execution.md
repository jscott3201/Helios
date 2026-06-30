# Decision Record: M06 MTP Fails Closed Until Native Assistant Execution Exists

- Status: accepted
- Date: 2026-06-30
- Milestone: M06

## Context

M06 requires Gemma 4 MTP assistant loading and exact greedy speculative decoding with rollback. The native runtime can validate target artifacts, call the MLX-LM helper path, and run an opt-in native target graph. The native graph now preserves an opaque last-hidden owner and final full-attention/sliding-attention shared KV views in the KV cache.

The public MTP assistant artifact uses a distinct `gemma4_unified_assistant` config with a 4-layer text model, `backbone_hidden_size=3840`, `num_kv_shared_layers=4`, and assistant-specific tensor names. Accepting a target-model manifest as a drafter would hide the missing native integration.

## Decision

Implement and test the M06 speculative decoding state machine, metrics, rollback, auto-disable, TUI payload, target hidden/shared-view materialization, and FFI contracts now. Strict drafter load validates the real assistant artifact shape and returns a drafter handle only for assistant manifests. Native assistant drafting remains fail-closed: `gemma4_mtp_draft_block` returns `GEMMA4_ERR_UNSUPPORTED_CONFIG` after confirming target hidden/shared views are present, until the assistant graph itself is implemented.

## Consequences

- Fixture exactness and M06 acceptance metrics are available now.
- The TUI can show MTP acceptance, rollback, and auto-disable state.
- Real Gemma 4 target+assistant MTP execution is still a follow-up, dependent on native assistant tensor loading and forward execution.
- The runtime will not silently run an unsupported or generic speculative path for real models.

## Evidence

- Commands:
  - `cargo test -p gemma4d-ffi`
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
  - `mlx-community/gemma-4-12B-it-qat-assistant-4bit` config and safetensors header
