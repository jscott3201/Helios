# M04 Compliance Matrix

## Scope

- Milestone: `milestones/M04-reference-parity-harness.md`
- Goal: `codex/goals/M04-reference-parity-harness.goal.md`
- Spec: `spec/10-correctness-evals-benchmarks.md`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M04-T01 | Add prompt corpus and benchmark runner. | `benchmarks/prompts/M04-corpus.tsv`; `gemma4d-bench run`; harness run wrote `benchmarks/out/M04/records.jsonl`. | Complete | None for M04 scope. |
| M04-T02 | Add baseline command capture for MLX Python and llama.cpp when configured. | MLX helper command captured in JSONL/report; `--llama-cmd TEMPLATE` support and unit test `configured_command_reference_parses_generated_tokens`. | Complete | No real llama.cpp binary configured locally; command-template path is implemented and inconclusive if unparsable. |
| M04-T03 | Add token sequence diff tooling. | `compare_tokens` in `crates/gemma4d-bench/src/lib.rs`; unit tests for match, mismatch, and length mismatch. | Complete | None. |
| M04-T04 | Add benchmark JSONL writer. | `benchmarks/out/M04/records.jsonl` from harness run; records include candidate/reference commands, tokens, metrics, environment, and model hashes. | Complete | Raw JSONL is intentionally ignored. |
| M04-T05 | Add report generator from raw outputs. | `gemma4d-bench report`; generated `docs/evidence/M04-benchmark-report.md` from raw JSONL. | Complete | None. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| At least one reference path can be configured and compared. | `--reference mlx-helper` run compared two prompt cases against MLX Python; both passed. | Complete |
| Token diffs are readable. | Unit tests assert first-mismatch and length-mismatch summaries; JSONL stores `comparison_summary` and `comparison_detail`. | Complete |
| Benchmark records include environment and model revisions. | JSONL records contain `environment` plus `model_revision` with model path, config hash, and tokenizer hash; report includes these fields. | Complete |
| Inconclusive comparisons are labelled as such, not passed. | `comparison_for` returns `inconclusive` for unavailable candidate/reference; report unit test verifies inconclusive records are counted as inconclusive. | Complete |

## Coverage Summary

- Implemented and tested: prompt corpus parsing, candidate command capture, MLX helper reference, configurable command references, token diffs, JSONL writing, report generation, inconclusive labelling.
- Implemented but not live-tested with a real external binary: llama.cpp command template.
- Not implemented: benchmark optimization, variance analysis, or later milestone eval suites.
- Ambiguous / needs owner decision: when to require a real llama.cpp/GGUF artifact in CI.
