# M02 — Gemma 4 Tokenizer, Chat, and Config

## Goal

Implement Gemma 4 12B config validation, tokenizer loading, chat-template fixture tests, and cache-key hashing inputs.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Parse and validate Gemma 4 12B config fields.
- [ ] Load tokenizer files and create tokenization fixtures.
- [ ] Implement chat prompt compiler with system role support.
- [ ] Add hash computation for tokenizer/chat/model config.
- [ ] Create fixtures for simple chat, system prompt, Rust/Python code prompts, and long prefix.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M02/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] Fixture token IDs match reference.
- [ ] Unsupported configs fail clearly.
- [ ] Cache-key hash inputs are deterministic.
- [ ] No full 12B model load required.

## Recommended Codex goal

Use `codex/goals/M02-gemma4-tokenizer-chat-config.goal.md`.

## Recommended skills

- `$gemma4d-milestone-execution`
- `$spec-contract-compliance-review`
- `$performance-ab-benchmark-review` when this milestone touches runtime performance
- milestone-specific project skill as applicable

## Blocked stop condition

If a required external dependency, model artifact, MLX API, or machine capability is unavailable, stop with:

1. attempted paths,
2. command/source evidence,
3. minimal repro or diagnostic,
4. next input required.
