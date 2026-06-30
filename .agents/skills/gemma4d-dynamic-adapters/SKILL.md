---
name: gemma4d-dynamic-adapters
description: Use for dynamic LoRA/QLoRA adapter import, validation, loading, routing, adapter-aware KV cache keys, aLoRA planning, and adapter memory residency.
---
# Gemma4D Dynamic Adapters

## Trigger

Use for PEFT/MLX-LM adapter import, LoRA/QLoRA routing, adapter manifests, adapter memory management, or adapter-aware cache behavior.

## Rules

- Trusted local paths only in MVP.
- Prefer safetensors.
- Reject base/tokenizer/template mismatches.
- Reject unexpected `modules_to_save` unless explicitly allowed.
- Standard LoRA KV namespaces cannot be shared across adapters.
- MTP disabled with adapters until per-adapter exactness passes.

## Required evidence

- Adapter manifest.
- Shape validation result.
- Load/unload behavior.
- Cache namespace isolation test.
- Memory and load latency.
