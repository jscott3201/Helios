# Benchmark Report

## Milestone

M03 Greedy Text Inference

## Question

- Baseline: M03 scaffolding with graceful missing-model failure.
- Candidate: helper-backed local MLX execution plus opt-in native C++/MLX BF16 full-recompute execution behind the C ABI.
- Workload: short prompt, repeated-token 1K/4K/8K context greedy decode.
- Metrics: TTFT, decode tok/s, MLX peak memory, helper peak RSS.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M5 |
| RAM | 17179869184 bytes |
| macOS | 26.6 build 25G5043d |
| Rust | `rustc 1.95.0 (59807616e 2026-04-14)` |
| MLX | 0.31.2 |
| MLX-LM | 0.31.3_2 |
| Baseline model | `mlx-community/gemma-4-12B-it-4bit` |
| Model revision/hash | Local ignored `artifacts/models/gemma-4-12B-it-4bit` |
| QAT target decision | `docs/decisions/0001-gemma4-12b-qat-target.md` |
| Adapter | None |

## Commands

```bash
cargo fmt --all --check
cargo test -p gemma4d-sampler
cargo test -p gemma4d-server
cargo test -p gemma4d-ffi
./scripts/mlx-diagnostics.sh
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_FULL_MODEL_TESTS=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo test -p gemma4d-ffi native_graph_prefills_one_token_when_explicitly_enabled -- --nocapture
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_TRACE_PARITY_LOGITS=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --token-ids 9259,236772,236772 --max-new-tokens 1 --json
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --token-ids 9259 --max-new-tokens 2 --json
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --prompt Hello --max-new-tokens 8 --json
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --context-tokens 1024 --repeat-token 9259 --max-new-tokens 1 --json
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --context-tokens 4096 --repeat-token 9259 --max-new-tokens 1 --json
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --context-tokens 8192 --repeat-token 9259 --max-new-tokens 1 --json
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --token-ids 9259 --max-new-tokens 1 --json
cargo run -p gemma4d-server -- generate --model-path /tmp/gemma4d-missing-generate-model-path-for-test --token-ids 1,2,3 --max-new-tokens 1
hf download mlx-community/gemma-4-12B-it-4bit config.json tokenizer.json tokenizer_config.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors --local-dir artifacts/models/gemma-4-12B-it-4bit
hf download mlx-community/gemma-4-12B-it-qat-4bit config.json --local-dir /private/tmp/helios-hf-q-qconfig
hf download mlx-community/gemma-4-12B-it-qat-4bit model.safetensors.index.json --local-dir /private/tmp/helios-hf-q-qindex
hf download mlx-community/gemma-4-12B-it-qat-OptiQ-4bit config.json --local-dir /private/tmp/helios-hf-q-optiq-config
hf download mlx-community/gemma-4-12B-it-qat-OptiQ-4bit model.safetensors.index.json --local-dir /private/tmp/helios-hf-q-optiq-index
cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --prompt Hello --max-new-tokens 8 --json
target/debug/gemma4d generate --model-path artifacts/models/gemma-4-12B-it-4bit --prompt Hello --max-new-tokens 8 --json
cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --prompt Hello --max-new-tokens 2 --json
target/debug/gemma4d generate --model-path artifacts/models/gemma-4-12B-it-4bit --repeat-token 9259 --context-tokens 1024 --max-new-tokens 16 --json
target/debug/gemma4d generate --model-path artifacts/models/gemma-4-12B-it-4bit --repeat-token 9259 --context-tokens 4096 --max-new-tokens 16 --json
target/debug/gemma4d generate --model-path artifacts/models/gemma-4-12B-it-4bit --repeat-token 9259 --context-tokens 8192 --max-new-tokens 16 --json
```

## Results

| Workload | Context | Mode | TTFT | Decode tok/s | Peak MLX Memory | Peak RSS | Notes |
|---|---:|---|---:|---:|---:|---:|---|
| Missing model path | 3 tokens | CLI generate | n/a | n/a | n/a | n/a | Passed graceful failure check. |
| Native divergent-prefix check | 3 tokens | opt-in native BF16 full recompute | 995.829 ms | n/a | 6.705 GB | n/a | Generated `[236761]`; candidate logits exactly matched MLX-LM (`236761:18`, `236772:17.875`). |
| Native prefill+decode smoke | 1 token | opt-in native BF16 full recompute | 1012.522 ms | 15.289 | 6.707 GB | n/a | Generated `[236772, 236772]`, matching helper first two tokens. |
| Native `Hello` prompt | 1 token | opt-in native BF16 full recompute | 981.518 ms | 9.198 | 6.723 GB | n/a | Generated `[236772, 236772, 236761, 236779, 236772, 236772, 236772, 236772]`, matching helper sequence. |
| Repeated token 9259 | 1024 tokens | opt-in native BF16 full recompute | 2422.635 ms | n/a | 7.209 GB | n/a | One generated token; refreshed after BF16 parity fix. |
| Repeated token 9259 | 4096 tokens | opt-in native BF16 full recompute | 10211.584 ms | n/a | 7.914 GB | n/a | One generated token; refreshed after BF16 parity fix. |
| Repeated token 9259 | 8192 tokens | opt-in native BF16 full recompute | 25492.132 ms | n/a | 10.036 GB | n/a | One generated token; refreshed after BF16 parity fix. |
| `Hello` prompt | 1 token | helper-backed target greedy | 254.174 ms | 16.606 | 6.792 GB | 4953.3 MB | Generated `[236772, 236772, 236761, 236779, 236772, 236772, 236772, 236772]`. |
| `Hello` prompt repeat | 1 token | helper-backed target greedy | 424.474 ms | 16.574 | 6.792 GB | 3645.3 MB | Same generated token sequence. |
| `Hello` prompt manifest check | 1 token | helper-backed target greedy | 644.810 ms | 16.039 | 6.792 GB | 3075.9 MB | Rebuilt native validator checked config and safetensor inventory before helper startup; generated `[236772, 236772]`. |
| Repeated token 9259 | 1024 tokens | helper-backed target greedy | 2668.492 ms | 8.450 | 8.065 GB | 3754.8 MB | 16 generated tokens. |
| Repeated token 9259 | 4096 tokens | helper-backed target greedy | 9810.521 ms | 6.940 | 9.480 GB | 4121.9 MB | 16 generated tokens. |
| Repeated token 9259 | 8192 tokens | helper-backed target greedy | 19275.786 ms | 2.890 | 9.833 GB | 5191.0 MB | 16 generated tokens. |

## Correctness Guardrails

- Greedy sampler deterministically selects the largest finite logit and breaks ties toward the lowest token id.
- Chunked-prefill equivalence is represented by an ignored pending test until native chunked/KV parity replaces the full-recompute guardrail.
- FFI smoke tests still run without loading the full model by default.
- Real-model strict loads validate local config/tokenizer/safetensor artifacts, then parse `config.json` and safetensor headers before helper startup.
- Opt-in native loads use `GEMMA4D_USE_NATIVE_GRAPH=1`, validate the loaded MLX tensor inventory, and run a BF16 full-recompute native graph. Short deterministic parity is proven for the 8-token `Hello` sequence and for exact logits on the formerly divergent prefix.

## Raw Artifacts

- Local ignored artifact: `benchmarks/out/M03/diagnostics.jsonl`

## Interpretation

The benchmark is valid for both the helper-backed local MLX path and the opt-in native C++/MLX BF16 full-recompute path: model load, prefill, decode, tokenization, and metrics all run on the downloaded Gemma 4 12B MLX 4-bit artifacts. The native path is parity-clean for the M03 short deterministic smoke sequence, but it is still a full-recompute implementation and does not yet replace the helper-backed default path.

Google's QAT docs and the current Hugging Face MLX QAT metadata indicate that QAT target artifacts should be evaluated separately before changing the M03 default. The MLX QAT artifacts use mixed 4/8-bit quantization overrides and have different weight payload sizes, so the native loader now honors per-module quantization metadata, but this report does not claim QAT tiny16 fit or parity.

## Follow-up

Decide when to make the pure C++/MLX Gemma 4 text graph the default behind the same C ABI, then add native chunked/KV execution. Download the selected QAT MLX target and rerun the 1K/4K/8K benchmark before changing default model IDs.
