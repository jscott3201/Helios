# Sources and Evidence Map

Last reviewed: 2026-06-30.

This file is a source map for the implementation agent. Re-check volatile sources when implementing a milestone, especially model repos, MLX APIs, Codex CLI behavior, and external benchmark claims.

## Codex workflow sources

- OpenAI Codex Goals cookbook: `https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex`
  - Key use: goals should define outcome, verification surface, constraints, boundaries, iteration policy, and blocked stop condition.
- OpenAI Codex Subagents: `https://developers.openai.com/codex/subagents`
  - Key use: subagents are explicit, project-scoped TOML files can define `name`, `description`, and `developer_instructions`; keep subagents narrow.
- OpenAI Codex Skills: `https://developers.openai.com/codex/skills`
  - Key use: skills live in directories with `SKILL.md`, must include `name` and `description`, and can include references/assets/scripts.
- OpenAI Codex Prompting: `https://developers.openai.com/codex/prompting`
  - Key use: Codex performs best when work is scoped, verifiable, and broken into smaller focused steps.
- OpenAI Codex Configuration Reference: `https://developers.openai.com/codex/config-reference`
  - Key use: `model_reasoning_effort` supports `xhigh` for supported/model-dependent setups; project agents inherit the active model rather than pinning a possibly account-specific model ID.
- Microsoft Command Line, agent optimization loop: `https://commandline.microsoft.com/the-agent-optimization-loop-and-how-we-built-it-in-foundry/`
  - Key use: use traces/evals/developer review/versioned changes; better diagnosis can beat better execution.
- LangChain, Art of Loop Engineering: `https://www.langchain.com/blog/the-art-of-loop-engineering`
  - Key use: agent loop + verification loop + event/app loop + hill-climbing loop; this package emphasizes loops 1 and 2 for implementation, and loop 4 for future skill/spec improvement.

## Runtime and model sources

- MLX repository: `https://github.com/ml-explore/mlx`
  - MLX is Apple Silicon-focused, supports C++/C APIs, lazy computation, dynamic graphs, CPU/GPU, and unified memory.
- Gemma 4 model card: `https://ai.google.dev/gemma/docs/core/model_card_4`
  - Gemma 4 12B Unified has 48 layers, 1024 sliding window, 256K context, 262K vocabulary, hybrid local/global attention, and unified K/V in global layers.
- Gemma 4 MTP overview: `https://ai.google.dev/gemma/docs/mtp/overview`
  - MTP drafts multiple tokens with a smaller draft model and verifies them with the target; the drafter shares embeddings and uses target activations.
- mlx-lm LoRA/QLoRA docs: `https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md`
  - MLX-LM supports LoRA/QLoRA fine-tuning and generation with adapter paths.
- Hugging Face PEFT LoRA/aLoRA docs: `https://huggingface.co/docs/peft/developer_guides/lora`
  - aLoRA activation tokens allow pre-invocation KV cache reuse with the base model.
- Hugging Face PEFT checkpoint docs: `https://huggingface.co/docs/peft/developer_guides/checkpoint`
  - PEFT adapter checkpoints use `adapter_model.safetensors` or `.bin` plus `adapter_config.json`; prefer safetensors.

## Implementation precedent sources

- mlxcel: `https://github.com/lablup/mlxcel`
  - Rust CLI/server executing models through native MLX C++ bindings; useful reference for build strategy and Gemma-family handling.
- oMLX: `https://github.com/jundot/omlx`
  - Tiered KV cache with hot RAM, cold SSD, prefix sharing, CoW, and safetensors persistence.
- llama.cpp: `https://github.com/ggml-org/llama.cpp`
  - Baseline for Apple Metal support, GGUF/QAT validation, and quantized KV experiments.
- mistral.rs: `https://github.com/ericlbuehler/mistral.rs`
  - Rust inference reference with LoRA/X-LoRA, dynamic model loading, tool-calling, and server patterns.
- RotorQuant: `https://github.com/scrya-com/rotorquant`
  - KV compression reference for Planar/Iso/Turbo cache modes and llama.cpp integration ideas.
- TurboQuant: `https://github.com/0xSero/turboquant`
  - Algorithmic reference for compressed KV scoring and quality tradeoffs; not a direct Apple/MLX code dependency.

## TUI / terminal UX sources

- Ratatui website: `https://ratatui.rs/`
  - Key use: Ratatui is a fast/lightweight Rust TUI library with dashboard-appropriate widgets and immediate-mode rendering.
- Ratatui repository: `https://github.com/ratatui/ratatui`
  - Key use: quickstart, templates, examples, project status, and API links.
- Ratatui Component Template: `https://ratatui.rs/templates/component/`
  - Key use: Tokio async events, tracing, color-eyre, clap, component trait, and terminal-app scaffolding.
- Ratatui Async Event Stream tutorial: `https://ratatui.rs/tutorials/counter-async-app/async-event-stream/`
  - Key use: async input/events with Tokio and channels.
- Ratatui FAQ: `https://ratatui.rs/faq/`
  - Key use: Ratatui is not natively async; the app owns event loop/state/redraw; crossterm event-stream can make input async.
- Ratatui Backends docs: `https://ratatui.rs/concepts/backends/`
  - Key use: Crossterm is the default backend; TestBackend supports UI unit tests; avoid multiple semver-incompatible Crossterm versions.
- Ratatui snapshot testing recipe: `https://ratatui.rs/recipes/testing/snapshots/`
  - Key use: use insta/cargo-insta with Ratatui apps/widgets for snapshot tests.
- Crossterm terminal docs: `https://docs.rs/crossterm/latest/crossterm/terminal/index.html`
  - Key use: alternate screen and raw mode lifecycle details.


## User-provided archive

- `research_review_agent_skills_v1.zip`
  - General-purpose research/review skills and subagents. This package copies the Codex-native portions and adds Gemma4D-specific extensions.
