# 04 — Model Loading, Tokenization, and Chat Templates

## Target checkpoint plan

MVP runtime target:

```text
mlx-community/gemma-4-12B-it-4bit or local equivalent
```

Reference/baseline targets:

```text
MLX Python reference path where available
Hugging Face tokenizer/config reference fixtures
llama.cpp GGUF/QAT path for quality/performance comparison
```

## Config assumptions to validate

Gemma 4 12B-specific runtime assumptions must be loaded from config and verified by tests:

```text
architecture = Gemma4UnifiedForConditionalGeneration
layers = 48
sliding_window = 1024
context_length = 262144 maximum, but tiny16 starts at 32768
vocab = 262144
hybrid local/full attention
full/global layers have unified K/V behavior
```

Do not silently run if the loaded config diverges from the supported assumptions. Produce an explicit unsupported-config error.

## Tokenizer requirements

- Load tokenizer from local model directory or trusted downloaded snapshot.
- Test token IDs for system/user/assistant messages.
- Test stop tokens.
- Test streaming detokenization boundaries.
- Hash tokenizer files into cache keys.

## Chat template requirements

- Support native system role.
- Support thinking-mode configuration as a prompt/compiler input.
- Preserve tool-call formatting decisions in test fixtures.
- Hash chat template into KV and adapter cache namespaces.

## Fixtures

Create fixtures under:

```text
tests/fixtures/prompts/
  simple_chat.json
  system_prompt.json
  code_rust.json
  code_python.json
  long_prefix_4k.json
  tool_call_shape.json
```

Each fixture should include:

```text
messages
expected token ids or reference token-id file
expected rendered prompt text when safe to store
expected stop behavior
```

## Acceptance

M02 is complete only when tokenizer/chat/config tests pass without loading the full 12B model.
