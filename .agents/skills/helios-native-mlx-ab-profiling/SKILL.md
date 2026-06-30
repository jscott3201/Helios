---
name: helios-native-mlx-ab-profiling
description: Profile and optimize Helios native MLX/C++ execution paths with A/B evidence, memory gates, and feature-flagged experimental kernels.
---

# Helios native MLX A/B profiling skill

Use this skill for native graph, MLX, Metal, KV cache, compressed attention, LoRA hot path, and prefill/decode optimization.

## Rules

1. Establish baseline records before changing hot-path code.
2. Separate model load, prefill, decode, eval/synchronization, cache import/export, and server overhead.
3. Keep unsafe/FFI changes narrow and tested.
4. Avoid default-on experimental kernels.
5. Record p50, p95, p99, peak MLX memory, active KV bytes, and RSS.
6. Treat no-go reports as valid outcomes for high-risk ideas.

## Red flags

- Speedup measured only on repeated-token prompts.
- Active memory reduction claimed when compressed payload is decompressed into BF16 active state.
- MTP speedup claimed without acceptance and exactness evidence.
- Server speedup measured through stub metrics.
