# Decision Record: M10 Trusted Local PEFT Adapter Registry

- Status: accepted
- Date: 2026-06-30
- Milestone: M10

## Context

M10 requires dynamic LoRA/QLoRA adapter import, validation, loading, routing, unloading, pinning, adapter-aware KV cache keys, and TUI registry summaries. The native MLX LoRA math path is not yet stable enough to safely fuse arbitrary adapters into the target model during this milestone.

The PEFT checkpoint convention uses `adapter_config.json` plus adapter weights such as `adapter_model.safetensors`, which matches the M10 import target. Reference: <https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint>.

## Decision

Implement M10 as a trusted local PEFT registry and routing layer:

- Accept only trusted local source paths under an explicit trusted root.
- Import PEFT `adapter_config.json` and `adapter_model.safetensors`.
- Validate the adapter manifest against expected base model, base weight hash, tokenizer hash, and chat-template hash.
- Reject unsupported `modules_to_save`, tokenizer-changing adapters, and non-standard adapter types for the M10 MVP.
- Keep one active adapter per request in the routing state.
- Keep MTP disabled whenever a standard LoRA adapter is active until per-adapter exactness is separately verified.
- Include adapter identity and adapter weight hash in KV namespace and block-ID tests.
- Surface registry summaries through the TUI provider model and Adapters page.

## Consequences

- M10 proves safe import, compatibility rejection, registry state, request routing, CLI controls, TUI visibility, and cache namespace isolation without prematurely committing to native MLX adapter fusion.
- Valid adapters are loaded into registry state and validated at the safetensors header level; actual MLX tensor application remains future work.
- Adapter-specific MTP enablement remains blocked until exactness tests exist for each adapter.

## Evidence

- `crates/gemma4d-adapters/src/lib.rs`
- `crates/gemma4d-adapters/examples/m10_adapter_fixture.rs`
- `crates/gemma4d-server/src/lib.rs`
- `crates/gemma4d-kv/src/lib.rs`
- `crates/gemma4d-tui/src/app.rs`
- `crates/gemma4d-tui/src/provider.rs`
- `crates/gemma4d-tui/src/ui.rs`
- `docs/evidence/M10.md`
- `docs/evidence/M10-compliance.md`
- `benchmarks/out/M10/adapter-fixture.json` (generated and ignored)
