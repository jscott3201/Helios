# 02 — Architecture

## Workspace shape

```text
gemma4d/
  Cargo.toml
  rust-toolchain.toml
  crates/
    gemma4d-server/
    gemma4d-tui/
    gemma4d-engine/
    gemma4d-tokenizer/
    gemma4d-chat/
    gemma4d-sampler/
    gemma4d-kv/
    gemma4d-cache/
    gemma4d-adapters/
    gemma4d-router/
    gemma4d-ffi/
    gemma4d-bench/
    gemma4d-llama-baseline/
  native/
    gemma4_mlx/
      CMakeLists.txt
      include/gemma4_mlx.h
      src/model_loader.cc
      src/target.cc
      src/drafter.cc
      src/kv_cache.cc
      src/spec_decode.cc
      src/adapter_loader.cc
      src/cache_export.cc
      src/quant.cc
  benchmarks/
    prompts/
    out/
  docs/
```

## Rust responsibilities

- API server and streaming.
- Ratatui operator console, provider boundary, config/benchmark/log/cache/adapters/MTP UI.
- Request queue, scheduler state machine, cancellation.
- Tokenizer/chat-template integration and prompt hashing.
- Sampling policy.
- KV/cache policy, manifest, SSD index, LRU.
- Adapter registry, adapter routing, adapter trust policy.
- Memory guardrails and telemetry.
- Benchmark harness and report generation.

## Native MLX responsibilities

- Model loading from MLX checkpoint format.
- MLX array ownership and lifetime.
- Target prefill/decode/verify.
- Drafter execution for MTP.
- KV tensor layout and conversion.
- LoRA adapter math on top of quantized base linear layers.
- Optional custom Metal kernels.

## TUI/control boundary

The Ratatui TUI is a separate crate and binary. It must not call native MLX scheduler APIs directly. It communicates through provider traits:

```text
MockProvider -> deterministic tests and snapshots
FileProvider -> configs, logs, benchmark JSONL/reports
HttpProvider -> local daemon/control API after server milestone
```

The TUI can be introduced before the OpenAI server is complete because offline/mock/file modes are useful for configuration and benchmark evidence loops.

## TUI model

The TUI lives outside the MLX scheduler. It connects to `gemma4d-server` over localhost APIs, receives snapshots/events, and submits actions. The TUI must never own MLX arrays, native handles, KV tensors, or adapter tensors in the MVP.

```text
gemma4d-tui -> localhost HTTP/SSE/WebSocket -> gemma4d-server -> engine -> native MLX
```

## Scheduler model for MVP

Use one MLX scheduler thread at first.

```text
HTTP/server tasks
  -> bounded request queue
  -> single MLX scheduler thread
  -> target/drafter/cache/adapters
  -> response stream channel
```

Initial state machine:

```rust
enum JobState {
    WaitingForAdapter,
    WaitingForPrefixRestore,
    Prefilling,
    DecodingTarget,
    Drafting,
    Verifying,
    Streaming,
    Complete,
    Cancelled,
    Failed,
}
```

## Runtime profiles

`tiny16` is the default profiling target:

```text
max_active_generations = 1
max_context_tokens = 32768 initially
prefill_chunk_tokens = 512
draft_block_size = 2 after MTP passes exactness
active_kv = bf16 first
prefix_ram = q8 after correctness
ssd_prefix = q4 after RAM restore correctness
max_hot_adapters = 2
```

## Design invariant

Correctness gates come before optimization gates. Any optimization path must define a fallback and a comparison to the unoptimized reference path.
