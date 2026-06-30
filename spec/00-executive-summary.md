# 00 — Executive Summary

## Thesis

Build a narrow, evidence-driven Gemma 4 12B runtime rather than a broad LLM engine.

The system should keep Rust in charge of API, scheduling, configuration, cache policy, measurements, and safety controls, while a small native C++/MLX layer owns MLX arrays, model execution, quantized math, custom Metal kernels, and FFI lifetime boundaries.

## First working target

```text
Model: Gemma 4 12B MLX 4-bit target
Drafter: Gemma 4 12B MTP assistant, MLX 4-bit
Machine: 16GB Apple Silicon MacBook
Mode: text-only, single active generation
Correctness: greedy target output stable before optimization
Serving: OpenAI-compatible local endpoint after core inference works
Operator UX: Ratatui operator console introduced early for config/bench/log workflows, then attached live after server/control APIs are stable
```

## Architectural bias

- Build a clean Rust workspace with a narrow native MLX C ABI shim.
- Use MLX-format models for the native path first; use llama.cpp/GGUF as reference baseline, not the primary execution path.
- Use Gemma 4 12B-specific assumptions to avoid generic-engine complexity.
- Treat MTP exactness, KV cache correctness, and adapter-aware cache identity as core features.
- Treat SSD as a cold prefix/KV cache tier, not as a live dense-weight paging layer.
- Treat the Ratatui TUI as a local operator console over provider/client APIs, not as a second inference runtime or native scheduler path.

## Success definition

The project is viable when a 16GB MacBook can run Gemma 4 12B in a measured, bounded profile with:

- reproducible greedy text generation,
- reference parity harnesses,
- MTP greedy exactness relative to non-MTP target generation,
- safe memory limits,
- measurable TTFT/decode/memory telemetry,
- RAM prefix cache,
- SSD cold prefix cache,
- dynamic standard LoRA/QLoRA adapter selection with adapter-aware KV namespaces,
- a local Ratatui TUI that can drive chat, adapters, cache inspection, benchmarks, config validation, and profiling without distorting engine measurements.

## Non-goals for this package

- DiffusionGemma.
- Broad model-family support.
- Multimodal implementation.
- Multi-user high-concurrency serving.
- Remote untrusted adapter marketplace.
- Production claims without benchmark evidence.
