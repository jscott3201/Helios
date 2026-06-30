# Compliance Review: M02 Gemma 4 Tokenizer, Chat, and Config

## Scope

- Spec/contract: `milestones/M02-gemma4-tokenizer-chat-config.md`, `spec/04-model-loading-tokenization.md`, relevant cache-key fields from `spec/06-kv-cache-offload-compression.md`
- Version/date: 2026-06-30
- Included areas: Gemma 4 12B config validation, local fixture tokenizer loading, chat prompt compiler with system role and thinking-mode input, token fixture equality, tokenizer/chat/config hash inputs, deterministic cache-key namespace hashing.
- Excluded areas: real Gemma 4 12B model load, real Hugging Face tokenizer snapshot download, greedy inference, KV tensor cache restore, server/TUI runtime behavior.

## Traceability Matrix

| Req ID | Requirement summary | Strength | Implementation evidence | Test evidence | Status | Gap |
|---|---|---|---|---|---|---|
| M02-T01 | Parse and validate Gemma 4 12B config fields. | Must | `crates/gemma4d-tokenizer/src/lib.rs`, `tests/fixtures/model/gemma4_12b_config.json` | `cargo test -p gemma4d-tokenizer` tests `validates_supported_gemma4_config` and `unsupported_configs_fail_clearly` | Complete | None |
| M02-T02 | Load tokenizer files and create tokenization fixtures. | Must | `FixtureTokenizer` in `crates/gemma4d-tokenizer/src/lib.rs`, `tests/fixtures/tokenizer/tokenizer.json`, prompt fixtures in `tests/fixtures/prompts/` | `cargo test -p gemma4d-tokenizer`; `cargo test -p gemma4d-chat` | Complete | None |
| M02-T03 | Implement chat prompt compiler with system role support. | Must | `compile_prompt`, `Role::System`, `ChatTemplateConfig` in `crates/gemma4d-chat/src/lib.rs` | `compiles_system_user_and_generation_prompt`; fixture equality tests | Complete | None |
| M02-T04 | Add hash computation for tokenizer/chat/model config. | Must | `file_sha256` and `canonical_json_sha256` in `gemma4d-tokenizer`; `template_hash` in `gemma4d-chat`; `CacheKeyInputs` and prompt hashes in `gemma4d-cache` | `file_hashes_are_stable`; `thinking_mode_changes_rendered_prompt_and_hash`; `prompt_hash_inputs_are_deterministic` | Complete | None |
| M02-T05 | Create simple chat, system prompt, Rust/Python code prompt, and long prefix fixtures. | Must | `tests/fixtures/prompts/simple_chat.json`, `system_prompt.json`, `code_rust.json`, `code_python.json`, `long_prefix_4k.json`; plus spec-requested `tool_call_shape.json` | `all_prompt_fixture_token_ids_match_reference` | Complete | None |
| M02-A01 | Fixture token IDs match reference. | Acceptance | Checked-in prompt fixture expected IDs and fixture tokenizer reference source | `cargo test -p gemma4d-chat` exact equality across all fixtures | Complete | None |
| M02-A02 | Unsupported configs fail clearly. | Acceptance | `GemmaConfig::validate` explicit field-specific errors | `unsupported_configs_fail_clearly` | Complete | None |
| M02-A03 | Cache-key hash inputs are deterministic. | Acceptance | `CacheKeyInputs::namespace_hash`, `prompt_token_prefix_hash`, `prompt_hashes` | `cargo test -p gemma4d-cache` | Complete | None |
| M02-A04 | No full 12B model load required. | Acceptance | Fixture config/tokenizer only; no model artifact paths or native model load calls | All M02 tests and `make verify` pass without model path/env; evidence ledger notes no model download/load | Complete | None |

## High-Risk Gaps

No blocker, high, or medium compliance gaps were found for M02.

## Coverage Summary

- Implemented and tested: config validation, fixture tokenizer loading, stop-token IDs, chat rendering with system role and thinking-mode input, exact prompt token fixture equality, tokenizer/chat/prompt/cache hash determinism.
- Implemented but explicitly local: fixture tokenizer/reference IDs are checked into this repo and labelled `gemma4d_fixture_tokenizer_v1`; they are not a real Hugging Face/Gemma tokenizer snapshot.
- Not implemented: full Gemma 4 tokenizer snapshot validation, real model loading, greedy generation, and reference parity against external MLX/HF paths; these belong to later milestones or require real artifacts.
- Ambiguous / needs owner decision: none for the local M02 acceptance gate.

## Next Work Items

1. Start M03 greedy text inference only after M02 is committed, pushed, and CI has verified the workspace.
