# 07 — Dynamic LoRA / QLoRA Adapters

## Goal

Load the Gemma 4 12B base once, then dynamically activate lightweight domain adapters per request or per conversation.

Examples:

```text
base
rust-coding-r16-v1
python-coding-r16-v1
sql-agent-r16-v1
```

## MVP adapter mode

- Standard LoRA/QLoRA adapters.
- One active adapter per request.
- Trusted local paths only.
- PEFT import first; MLX-LM import second.
- No adapter fusion in MVP.
- No arbitrary remote adapter loading.

## Adapter-aware cache rule

Standard LoRA changes activations from the point it is applied. Therefore:

```text
base KV cache != rust adapter KV cache
rust adapter KV cache != python adapter KV cache
```

Adapter identity and weight hash must be part of every standard-LoRA KV/prefix cache key.

## aLoRA later

Activated LoRA is a later milestone candidate because it allows base-prefix KV reuse before an invocation token sequence. Support it only after standard LoRA cache correctness is proven.

## Adapter manifest

See `references/schemas/adapter-manifest.schema.json`.

Important fields:

```text
adapter_id
adapter_type
base_model_id
base_weight_hash
tokenizer_hash
chat_template_hash
rank
alpha
target_modules
modules_to_save
requires_tokenizer_changes
supports_mtp
adapter_weight_hash
```

## Serving behavior

Requests may choose adapters explicitly:

```json
{
  "model": "gemma4-12b-it",
  "adapter": "rust-coding-r16-v1",
  "messages": []
}
```

or through aliases:

```json
{
  "model": "gemma4-12b-it:rust-coding",
  "messages": []
}
```

## MTP interaction

- Disable MTP when an adapter is active until per-adapter exactness tests pass.
- Track acceptance rate per adapter.
- Auto-disable MTP per adapter if acceptance falls below threshold.

## Acceptance

M10 is complete only when:

- valid PEFT adapter imports,
- mismatched base/tokenizer/template manifests are rejected,
- one adapter can be activated per request,
- unloading returns behavior to base,
- adapter-specific KV namespaces cannot cross-contaminate,
- memory usage and adapter load latency are measured.
