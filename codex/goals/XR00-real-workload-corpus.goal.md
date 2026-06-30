# XR00 - Real-context workload corpus

## Outcome

Create a deterministic, repo-local real-context workload corpus for Helios A/B benchmarks. This goal produces workload files and token-length metadata only. It must not optimize inference code.

## Why

Current benchmark evidence includes useful repeated-token probes, but future performance claims need realistic prompts: code review, benchmark QA, structured tool output, prefix reuse, adapter-specialist prompts, and long repo-context packs.

## Required work

1. Add `benchmarks/workloads/real-contexts/README.md` explaining the corpus.
2. Add prompt text files under `benchmarks/workloads/real-contexts/prompts/`.
3. Add `benchmarks/workloads/real-contexts/workloads.jsonl` with stable workload metadata.
4. Add or update a small tokenizer/counting tool that fills `actual_context_tokens` for the local model artifact.
5. Include at least these families:
   - `chat_short`
   - `code_review_rust`
   - `benchmark_qa`
   - `tool_json`
   - `prefix_reuse_edit`
   - `adapter_expert`
   - `long_repo_pack`
   - `mtp_candidate`
6. Include context targets at 1K, 4K, 8K, 16K, and one optional 24K/32K edge case if tiny16 memory allows.

## Verification surface

- Corpus validates without model execution.
- Token counts are generated using the actual local tokenizer when available.
- No private artifacts are committed.
- `cargo test -p gemma4d-bench --all-targets` still passes.

## Decision

`accept_candidate` when the corpus exists, is reproducible, and covers the required families.

## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR00-real-workload-corpus/records.jsonl
benchmarks/out/XR00-real-workload-corpus/summary.json
benchmarks/out/XR00-real-workload-corpus/report.md
benchmarks/out/XR00-real-workload-corpus/blockers.md
benchmarks/out/XR00-real-workload-corpus/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
