# M10 Compliance Matrix

## Scope

- Milestone: `milestones/M10-dynamic-lora-qlora-adapters.md`
- Spec: `spec/07-dynamic-lora-qlora-adapters.md`
- References: `spec/02-architecture.md`, `spec/06-kv-cache-offload-compression.md`, `docs/decisions/0006-m10-trusted-local-peft-adapter-registry.md`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M10-T01 | Implement adapter manifest parser and schema validation. | `AdapterManifest::from_json_str`; `AdapterManifest::validate`; unit tests. | Complete | Schema file is used as the contract; Rust validation enforces MVP safety rules. |
| M10-T02 | Import PEFT `adapter_config.json` and `adapter_model.safetensors`. | `AdapterRegistry::import_peft`; `PeftAdapterConfig`; safetensors header validator; `m10_adapter_fixture`. | Complete | Real downloaded adapters still need native shape binding later. |
| M10-T03 | Implement one active adapter per request. | `AdapterRegistry::activate_request`; fixture route report `one_active_adapter_per_request=true`. | Complete | Runtime server request integration arrives with later daemon/control work. |
| M10-T04 | Add local adapter registry and trusted path policy. | `TrustedPathPolicy`; persisted `registry.json`; untrusted path tests and fixture gate. | Complete | Trust policy is local-root based only. |
| M10-T05 | Add adapter-aware KV namespace tests. | KV tests `adapter_identity_and_weight_hash_partition_namespace_and_blocks`, `adapter_namespace_mismatch_rejects_ram_restore`, and `adapter_namespace_mismatch_rejects_ssd_restore`. | Complete | Native KV tensor identity remains future work. |
| M10-T06 | Add load/unload/pin endpoints or CLI commands. | `gemma4d adapter import|load|unload|pin|list`; server tests. | Complete | No HTTP endpoint yet; CLI satisfies M10. |
| M10-T07 | Connect adapter registry summaries to the TUI adapter page/provider model. | `AdapterSnapshot`; `MockProvider::adapter_snapshot`; `render_adapters`; TUI acceptance test. | Complete | File provider reports disabled until a live registry source is attached. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| Valid local adapter loads. | Adapter unit test; CLI test; `adapter-fixture.json` import section. | Complete. |
| Wrong base/tokenizer/template adapters rejected. | Adapter unit test; fixture rejection gates all true. | Complete. |
| Base output unchanged when adapter disabled. | `fixture_generate_token`; fixture gate `base_output_unchanged_when_disabled=true`. | Complete. |
| Adapter cache cross-contamination tests pass. | Adapter-aware KV namespace/block-ID tests; wrong-adapter RAM/SSD restore rejection. | Complete. |
| MTP disabled with adapters unless separately verified. | `AdapterRegistry::activate_request` returns `mtp_enabled=false`; fixture gate true; existing engine MTP adapter-active disable test remains in place. | Complete. |

## Spec Compliance Summary

- Compliant: trusted local PEFT import, manifest/compatibility validation, safetensors header validation, local registry state, one-active-adapter routing, load/unload/pin/list CLI, adapter-aware cache namespace tests, TUI provider/page summaries, MTP disablement with active adapters.
- Intentionally deferred: native MLX adapter math/fusion, remote adapter loading, adapter composition, multiple active adapters, aLoRA-specific KV sharing, and per-adapter MTP exactness enablement.

## Risk

The main residual risk is native integration: M10 validates registry and control-plane correctness with deterministic fixtures, but real adapter tensor application must still be validated against Gemma 4 base-model shapes and greedy parity once the native adapter path is implemented.
