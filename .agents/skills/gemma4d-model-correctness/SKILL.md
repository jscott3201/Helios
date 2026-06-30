---
name: gemma4d-model-correctness
description: Use for Gemma 4 12B config, tokenizer, chat-template, prompt hashing, greedy parity, fixture creation, and reference comparison work.
---
# Gemma4D Model Correctness

## Trigger

Use for tokenizer/config/chat-template/logit parity and fixture work.

## Rules

- Do not optimize until correctness fixtures pass.
- Unsupported Gemma configs must fail loudly.
- Cache keys must include tokenizer and chat-template hashes.
- Reference comparisons may be exact, approximate, or blocked, but must be labelled honestly.

## Required evidence

- Fixture names.
- Reference source/tool used.
- Token diffs or equality result.
- Config fields validated.
