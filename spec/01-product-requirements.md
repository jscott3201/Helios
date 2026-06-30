# 01 — Product Requirements

## Users

Primary user: a developer/researcher running local inference on a 16GB MacBook and later larger Apple Silicon machines.

Primary workflows:

1. Local chat/completions through OpenAI-compatible APIs.
2. Coding-agent use with long shared prefixes and repeated repository context.
3. Benchmark-driven exploration of memory limits and KV/prefix cache offloading.
4. Dynamic specialist adapters such as `rust-expert` and `python-expert`.

## Functional requirements

### FR1 — Gemma 4 12B text inference

- Load a Gemma 4 12B MLX 4-bit checkpoint.
- Run text-only prompt prefill and decode.
- Support greedy generation first; sampling later.
- Support stop tokens and streaming decode after server milestone.

### FR2 — Reference parity

- Compare tokenizer output against a reference tokenizer path.
- Compare chat-template construction against reference fixtures.
- Compare greedy token sequences against MLX reference and/or llama.cpp/HF baselines where practical.

### FR3 — MTP speculative decoding

- Load target and MTP assistant.
- Implement draft/verify/accept/rollback loop.
- Require byte-identical greedy output to non-MTP mode for the same target mode.
- Record MTP acceptance rate and accepted tokens per verify pass.

### FR4 — KV/prefix cache

- Implement in-memory KV cache with explicit block metadata.
- Implement RAM prefix cache with exact restore validation.
- Implement SSD cold prefix cache with checksums and cache namespace validation.
- Later implement quantized/compressed prefix cache modes.

### FR5 — Dynamic adapters

- Import PEFT and MLX-LM LoRA/QLoRA adapters into a validated internal manifest.
- Load/unload/pin adapters dynamically from trusted local paths.
- Select one active standard LoRA adapter per request.
- Include adapter identity in KV/prefix cache keys.
- Disable MTP for adapters until per-adapter exactness tests pass.

### FR6 — Serving

- Provide an OpenAI-compatible local `/v1/chat/completions` endpoint.
- Support streaming.
- Support explicit `adapter` field and adapter-as-model alias.
- Provide health, model, adapter, metrics, runtime snapshot, cache summary, config validation, and benchmark-control endpoints.

### FR7 — Ratatui operator TUI

- Provide a keyboard-first local TUI for configuration, benchmark/profiling orchestration, logs, and runtime visibility.
- Support offline/mock/file provider modes before live server attach exists.
- Later support live dashboard, chat workbench, adapter manager, cache inspector, MTP view, benchmark runner, config editor, and logs/traces screens.
- Use provider fakes and snapshot tests so the TUI can be developed without model downloads.
- Keep all destructive actions confirmation-gated.
- Keep CLI/HTTP paths authoritative and available for every critical operation.

## Non-functional requirements

### NFR1 — Memory boundedness

The `tiny16` profile must protect the OS. It must fail gracefully before hitting system-wide memory collapse.

### NFR2 — Reproducibility

All benchmark reports must record exact model revision, adapter revision, rustc version, MLX version, macOS version, machine model, command line, prompt file, context length, and raw output path.

### NFR3 — Safety and security

- Never load arbitrary remote adapters from unauthenticated clients.
- Reject adapter manifests with mismatched base/tokenizer/template hashes unless explicitly overridden in a local trusted config.
- Keep unsafe/FFI isolated and audited.

### NFR4 — Maintainability

Avoid generic abstractions until there are at least two concrete implementations requiring the abstraction. Gemma 4 specialization is a feature, not a limitation, for this project phase.

## Ratatui operator UX

The project must provide a local TUI for the operator/developer loop.

Required initial workflows:

- inspect and validate `tiny16` runtime config,
- launch or inspect benchmark/profiling runs,
- view logs and recent errors,
- inspect runtime/provider health,
- show disabled placeholders for chat, cache, adapters, and MTP before those backends are ready.

Required later workflows:

- stream chat through the local runtime,
- select adapters and observe adapter residency,
- inspect KV/prefix/SSD cache namespaces,
- observe MTP acceptance/rollback/auto-disable behavior,
- drive the tiny16 profiling release checklist from the TUI.

### NFR5 — TUI non-interference

The TUI must not distort inference profiling. It must have bounded memory/log buffers, configurable tick rate, measured idle overhead, and a client/server boundary that lets the server continue or fail independently of the terminal UI.
