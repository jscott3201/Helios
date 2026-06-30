# Helios XR00 Real-Context Workload Corpus

This corpus contains deterministic, repo-local prompt contexts for Helios XR-phase A/B benchmarks. It replaces repeated-token-only probes with realistic prompt shapes while performing no model execution.

## Regeneration

```text
cargo run -p gemma4d-bench -- workload-corpus --model-path artifacts/models/gemma-4-12B-it-4bit --workload-dir benchmarks/workloads/real-contexts --out-dir benchmarks/out/XR00-real-workload-corpus --python /opt/homebrew/opt/mlx-lm/libexec/bin/python --seed 20260630
```

- Model tokenizer path: `artifacts/models/gemma-4-12B-it-4bit`
- Deterministic seed base: `20260630`
- Workload manifest: `benchmarks/workloads/real-contexts/workloads.jsonl`
- Evidence directory: `benchmarks/out/XR00-real-workload-corpus`

## Families

| Family | Workloads |
|---|---:|
| `chat_short` | 1 |
| `code_review_rust` | 2 |
| `benchmark_qa` | 2 |
| `tool_json` | 1 |
| `prefix_reuse_edit` | 2 |
| `adapter_expert` | 1 |
| `long_repo_pack` | 2 |
| `mtp_candidate` | 2 |

## Token Length Policy

`actual_context_tokens` is measured with the local Gemma 4 tokenizer through `mlx_lm.utils.load_tokenizer`; character counts are not used as a proxy.

## Privacy

All committed prompts are generated from repo-local files. Private user artifacts must stay under ignored `artifacts/workloads/` paths and are not part of XR00.
