---
name: gemma4d-kv-cache-offload
description: Use for Gemma4D KV cache, prefix cache, SSD cold tier, cache manifests, cache-key correctness, quantized/compressed KV experiments, and restore parity.
---
# Gemma4D KV Cache and Offload

## Trigger

Use for active KV, RAM prefix cache, SSD cold cache, cache manifests, q8/q4 KV, Planar/Iso/TurboQuant experiments, and restore correctness.

## Rules

- No mid-decode SSD fetch in MVP.
- SSD cache accelerates future prefills, not live dense weight paging.
- Cache key must include model, quantization, tokenizer, template, adapter, KV layout, KV mode, and engine versions.
- Restore must be tested against fresh prefill.
- Compression modes require quality metrics and fallback.

## Required output

- Cache mode.
- Block size.
- Namespace fields.
- Restore test result.
- TTFT/memory/cache hit evidence.
