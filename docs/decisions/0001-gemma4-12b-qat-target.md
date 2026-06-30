# Decision Record: Gemma 4 12B QAT Target Handling

- Status: accepted
- Date: 2026-06-30
- Milestone: M03

## Context

The original spec names `mlx-community/gemma-4-12B-it-4bit` as the MVP target. Google's current Gemma 4 QAT guidance now recommends QAT checkpoints for deployments that need maximum efficiency with minimal quality compromise, and its routing table lists `-qat-q4_0-unquantized` as the source for converting to other formats such as MLX.

Hugging Face now has MLX QAT artifacts for Gemma 4 12B, including:

- `mlx-community/gemma-4-12B-it-qat-4bit`
- `mlx-community/gemma-4-12B-it-qat-OptiQ-4bit`
- `mlx-community/gemma-4-12B-it-qat-assistant-4bit`

The available MLX QAT target artifacts are not equivalent to the current all-default 4-bit artifact. Their configs use mixed precision: default 4-bit affine quantization plus per-module 8-bit overrides, especially for MLP projections. The regular QAT MLX target also carries multimodal tensors and has an indexed weight payload around 10.99 GB. The OptiQ QAT target is text-only, has 1324 tensors, and has an indexed weight payload around 8.90 GB, but it is still mixed 4/8-bit.

## Decision

Do not silently replace the M03 local model artifact before benchmark evidence exists.

For M03, keep `mlx-community/gemma-4-12B-it-4bit` as the already-downloaded baseline artifact and make the native loader QAT-ready by honoring per-module quantization overrides from `config.json`. Treat QAT MLX target selection as an explicit benchmarked follow-up: download the selected QAT target, validate load/generation, and rerun 1K/4K/8K memory measurements before changing defaults.

Prefer the text-only QAT MLX target for tiny16 evaluation if it validates:

```text
mlx-community/gemma-4-12B-it-qat-OptiQ-4bit
```

Use the QAT assistant artifact later for MTP work:

```text
mlx-community/gemma-4-12B-it-qat-assistant-4bit
```

## Consequences

- Native quantized ops must read module-specific bit widths instead of assuming every quantized tensor uses 4 bits.
- Manifest validation must allow text-only MLX artifacts with no ignored multimodal tensors.
- M03 evidence for the current baseline remains valid, but it is not evidence that a QAT artifact fits the tiny16 profile.
- A target switch requires fresh TTFT/decode/memory evidence and updated model IDs in docs/config/tests.

## Evidence

- Commands:
  - `hf download mlx-community/gemma-4-12B-it-qat-4bit config.json --local-dir /private/tmp/helios-hf-q-qconfig`
  - `hf download mlx-community/gemma-4-12B-it-qat-4bit model.safetensors.index.json --local-dir /private/tmp/helios-hf-q-qindex`
  - `hf download mlx-community/gemma-4-12B-it-qat-OptiQ-4bit config.json --local-dir /private/tmp/helios-hf-q-optiq-config`
  - `hf download mlx-community/gemma-4-12B-it-qat-OptiQ-4bit model.safetensors.index.json --local-dir /private/tmp/helios-hf-q-optiq-index`
  - `jq '{default_bits:.quantization_config.bits, group_size:.quantization_config.group_size, mode:.quantization_config.mode, overrides:(.quantization_config|to_entries|map(select(.value|type=="object"))|length), gate0:.quantization_config."language_model.model.layers.0.mlp.gate_proj"}' /private/tmp/helios-hf-q-optiq-config/config.json`
- Files:
  - `_spec/spec/04-model-loading-tokenization.md`
  - `_spec/references/configs/tiny16.toml`
  - `native/gemma4_mlx/src/model_manifest.cc`
  - `native/gemma4_mlx/src/native_model.cc`
- References:
  - Google AI for Developers, Gemma 4 model overview QAT section: `https://ai.google.dev/gemma/docs/core#qat`
  - Google blog, Gemma 4 QAT models: `https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/`
  - Hugging Face Transformers Gemma4 Unified docs: `https://huggingface.co/docs/transformers/en/model_doc/gemma4_unified`
  - Hugging Face model metadata for `mlx-community/gemma-4-12B-it-qat-4bit`
  - Hugging Face model metadata for `mlx-community/gemma-4-12B-it-qat-OptiQ-4bit`
