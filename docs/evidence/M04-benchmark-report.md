# M04 Reference Parity Report

## Summary

- Passed: 2
- Failed: 0
- Inconclusive: 0

## Environment

| Item | Value |
|---|---|
| OS | Darwin Justins-MBP 25.6.0 Darwin Kernel Version 25.6.0: Tue Jun  9 23:08:46 PDT 2026; root:xnu-12377.160.70.501.6~2/RELEASE_ARM64_T8142 arm64 |
| Rust | rustc 1.95.0 (59807616e 2026-04-14) |
| Git commit | `01d6841bf668030bc9a5528ab230c66cdd055dd1` |

## Model

| Item | Value |
|---|---|
| Path | `artifacts/models/gemma-4-12B-it-4bit` |
| config.json FNV64 | `78b33ee49524555d` |
| tokenizer.json FNV64 | `040d5ce8cfcf8e2a` |

## Results

| Prompt | Reference | Status | Summary |
|---|---|---|---|
| `hello_smoke` | `mlx_python` | passed | tokens match (8 tokens) |
| `hello_divergent_prefix` | `mlx_python` | passed | tokens match (1 tokens) |

## Commands

```text
/opt/homebrew/opt/mlx-lm/libexec/bin/python native/gemma4_mlx/scripts/gemma4d_mlx_lm_helper.py artifacts/models/gemma-4-12B-it-4bit
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 target/debug/gemma4d generate --model-path artifacts/models/gemma-4-12B-it-4bit --token-ids 9259 --max-new-tokens 8 --json
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 target/debug/gemma4d generate --model-path artifacts/models/gemma-4-12B-it-4bit --token-ids 9259,236772,236772 --max-new-tokens 1 --json
```
