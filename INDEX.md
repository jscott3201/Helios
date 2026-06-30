# Index / Table of Contents

## Start here

1. `README.md` — what this package is and how to start.
2. `AGENTS.md` — root behavior contract for Codex.
3. `SOURCES.md` — research evidence map and external references.
4. `spec/00-executive-summary.md` — concise project thesis.
5. `milestones/milestone-index.md` — implementation order.

## Specification

| File | Purpose |
|---|---|
| `spec/00-executive-summary.md` | North star, constraints, design bias. |
| `spec/01-product-requirements.md` | Functional/non-functional requirements. |
| `spec/02-architecture.md` | Workspace, runtime, scheduler, native boundary. |
| `spec/03-rust-mlx-ffi-contract.md` | C ABI, ownership, error model, build rules. |
| `spec/04-model-loading-tokenization.md` | Gemma 4 12B model/config/tokenizer/chat path. |
| `spec/05-speculative-decoding-mtp.md` | Gemma 4 MTP exactness and rollback design. |
| `spec/06-kv-cache-offload-compression.md` | Active KV, RAM prefix cache, SSD cold tier, compression. |
| `spec/07-dynamic-lora-qlora-adapters.md` | Dynamic standard LoRA/QLoRA, aLoRA later, adapter-aware KV. |
| `spec/08-serving-api-runtime.md` | OpenAI-compatible API, scheduler, runtime config. |
| `spec/09-observability-profiling.md` | Metrics, traces, benchmark records, 16GB Mac profile. |
| `spec/10-correctness-evals-benchmarks.md` | Test matrix, parity, benchmark acceptance. |
| `spec/11-security-licensing-safety.md` | Local security, adapter safety, license review. |
| `spec/12-risk-register.md` | Major risks, mitigations, kill switches. |
| `spec/13-tui-operator-ux.md` | Ratatui TUI UX, pages, event loop, provider boundary, tests. |

## Milestones

Read `milestones/milestone-index.md` first.

| Milestone | File | Theme |
|---|---|---|
| M00 | `milestones/M00-repo-bootstrap.md` | Workspace, CI, tools, project skeleton. |
| M01 | `milestones/M01-native-mlx-loader.md` | MLX C++ loader and FFI smoke tests. |
| M02 | `milestones/M02-gemma4-tokenizer-chat-config.md` | Config/tokenizer/chat-template parity. |
| M03 | `milestones/M03-greedy-text-inference.md` | Target model greedy text decode. |
| M04 | `milestones/M04-reference-parity-harness.md` | Reference comparisons and benchmark harness. |
| M05 | `milestones/M05-tui-operator-console.md` | Ratatui operator console for config, benchmarks, logs, and later live UX. |
| M06 | `milestones/M06-mtp-speculative-decoding.md` | Gemma 4 MTP exact greedy speculative decoding. |
| M07 | `milestones/M07-kv-cache-core.md` | In-memory KV block/cache design. |
| M08 | `milestones/M08-ssd-prefix-cache.md` | SSD cold prefix cache and restore. |
| M09 | `milestones/M09-kv-compression-research.md` | q8/q4, Planar/Iso, compressed-cache experiments. |
| M10 | `milestones/M10-dynamic-adapters.md` | LoRA/QLoRA adapter import, routing, cache correctness. |
| M11 | `milestones/M11-openai-server.md` | OpenAI-compatible local server and live TUI attach provider. |
| M12 | `milestones/M12-tiny16-profiling-release.md` | 16GB Mac validation, TUI-driven profiling, and release gate. |

## Codex workflow

| Path | Purpose |
|---|---|
| `codex/goals/` | Goal prompts shaped for Codex Goals. |
| `codex/prompts/` | Orchestration, review, and subagent prompts. |
| `.codex/agents/` | Custom subagents. |
| `.agents/skills/` | Skills Codex can discover and activate. |

## Reference artifacts

| Path | Purpose |
|---|---|
| `references/configs/tiny16.toml` | Initial 16GB Mac runtime config target. |
| `references/configs/tui.toml` | Initial Ratatui operator console config. |
| `references/ffi/gemma4_mlx.h` | Initial C ABI header skeleton. |
| `references/schemas/` | JSON schema drafts for KV, adapter, and TUI session manifests. |
| `references/templates/` | Benchmark, evidence, decision, TUI usability, and subagent templates. |
| `references/tui-screen-map.md` | TUI page map and page responsibilities. |
| `references/tui-keybindings.md` | Baseline TUI keybindings and safety rules. |
| `references/benchmark-matrix.md` | Standard benchmark matrix for milestones. |
| `references/acceptance-checklists.md` | Cross-cutting acceptance gates. |
