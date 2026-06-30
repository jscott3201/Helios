# M03 Compliance Matrix

## Scope

- Milestone: `milestones/M03-greedy-text-inference.md`
- Goal: `codex/goals/M03-greedy-text-inference.goal.md`
- Specs: `spec/03-rust-mlx-ffi-contract.md`, `spec/04-model-loading-tokenization.md`, `spec/09-observability-profiling.md`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M03-T01 | Implement target model load in native shim. | Strict artifact validation, native config/tensor manifest validation in `native/gemma4_mlx/src/model_manifest.cc`, opt-in MLX tensor loading in `native/gemma4_mlx/src/native_model.cc`, plus helper startup in `native/gemma4_mlx/src/runtime.cc`; real model CLI smoke and gated native test pass. QAT target handling is recorded in `docs/decisions/0001-gemma4-12b-qat-target.md`. | Complete for M03 | QAT MLX artifacts still require download and fresh tiny16 benchmarks before becoming default. |
| M03-T02 | Implement prefill and decode-one FFI calls. | C ABI routes `gemma4_prefill`/`gemma4_decode_one` through helper IPC by default and through opt-in native BF16 full-recompute when `GEMMA4D_USE_NATIVE_GRAPH=1`; native one-token/two-token smoke passes; native `Hello` 8-token sequence matches helper; formerly divergent prefix has exact MLX-LM logits. | Complete for M03 | Native graph is still opt-in and full-recompute; chunked/KV native execution is follow-up work. |
| M03-T03 | Implement greedy sampler in Rust. | `crates/gemma4d-sampler/src/lib.rs`; `cargo test -p gemma4d-sampler`. | Complete | None for argmax sampler surface. |
| M03-T04 | Add CLI `gemma4d generate` for local smoke tests. | `crates/gemma4d-server/src/main.rs`, prompt smoke, token-id smoke, CLI tests. | Complete | None for M03 smoke surface. |
| M03-T05 | Record memory for 1K/4K/8K prompts on tiny16. | `docs/evidence/M03-benchmark-report.md`; raw ignored `benchmarks/out/M03/diagnostics.jsonl`. | Complete | Helper-backed and opt-in native BF16 full-recompute measurements are both recorded. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| Short prompts generate deterministic token sequences. | `--prompt Hello --max-new-tokens 8 --json` repeated produced `[236772, 236772, 236761, 236779, 236772, 236772, 236772, 236772]`. | Complete |
| Chunked prefill is implemented or explicitly deferred with tests marked pending. | Ignored test `chunked_prefill_matches_unchunked_for_full_model` in `gemma4d-ffi`. | Deferred |
| Benchmark report records TTFT/decode/memory for at least 1K and 4K. | `docs/evidence/M03-benchmark-report.md` records 1K, 4K, and 8K runs. | Complete |
| Failures are graceful under missing model path. | FFI and CLI tests plus expected-failure CLI command. | Complete |

## Conclusion

M03 has a verified helper-backed local MLX execution path behind the native C ABI and an opt-in native C++/MLX BF16 full-recompute graph that passes short one/two-token smoke checks, exact logits on the formerly divergent prefix, the deterministic 8-token `Hello` sequence, and 1K/4K/8K prefill measurements. The remaining engineering work is productization: decide when to replace the helper default, then add native chunked/KV execution.

The baseline M03 artifact remains `mlx-community/gemma-4-12B-it-4bit`. Google QAT guidance has been incorporated as a target-selection decision, and the native loader now parses mixed 4/8-bit quantization overrides, but QAT artifact fit/parity is not yet claimed.
