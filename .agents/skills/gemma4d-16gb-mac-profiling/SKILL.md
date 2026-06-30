---
name: gemma4d-16gb-mac-profiling
description: Use for tiny16 MacBook profiling: memory pressure, context-length limits, TTFT/decode metrics, cache effectiveness, MTP acceptance, and release-gate benchmark reports.
---
# Gemma4D 16GB Mac Profiling

## Trigger

Use for profiling or release validation on a 16GB Apple Silicon Mac.

## Rules

- Protect system headroom.
- Start with 1K/4K/8K/16K before 32K+.
- Record exact machine, macOS, Rust, MLX, model, and adapter revisions.
- Treat memory guard rejections as useful evidence, not failure, if graceful.

## Required metrics

TTFT, prefill tok/s, decode tok/s, peak RSS, cache bytes, SSD IO, MTP acceptance, adapter load latency, and fallback/error codes.
