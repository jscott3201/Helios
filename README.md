# Gemma4D Codex Work Package

Implementation-ready specification package for a Gemma 4 12B-focused inference runtime on Apple macOS using Rust 1.95.0 and MLX.

This package is designed for a GPT-5.5 xHigh Codex workflow. The included project agents request `model_reasoning_effort = "xhigh"` and inherit your active Codex model, so set the exact GPT-5.5 Codex model ID in your user-level Codex config/profile if your installation requires one. It gives Codex a durable project contract, milestone plan, measurable acceptance gates, subagent roles, and repo-scoped skills so implementation can proceed in concrete loops rather than broad research prompts.

## North star

Build `gemma4d`: a local Apple Silicon inference runtime optimized for **Gemma 4 12B** first.

Core target:

```text
Rust 1.95.0
macOS Apple Silicon
MLX C/C++ backend behind a narrow Rust FFI layer
Gemma 4 12B MLX 4-bit text path
Gemma 4 MTP speculative decoding
16GB MacBook memory envelope
RAM prefix/KV cache
SSD cold prefix cache
KV compression experiments
Dynamic LoRA/QLoRA adapter routing
Ratatui operator TUI for configuration, benchmarks, profiling, cache/adapters/MTP visibility, and chat
OpenAI-compatible serving API
```

Deliberate exclusions for the first implementation wave:

```text
No DiffusionGemma path.
No general-purpose model runtime.
No live dense-weight SSD paging in MVP.
No multimodal runtime until the text path, MTP path, and KV/cache contracts are stable.
No arbitrary remote adapter loading by unauthenticated clients.
```

## Package layout

| Path | Purpose |
|---|---|
| `INDEX.md` | Table of contents and reading order. |
| `AGENTS.md` | Root instructions for Codex in this repo/package. |
| `.codex/` | Project-scoped Codex subagent config. |
| `.agents/skills/` | Codex skills: copied research/review skills plus project-specific Gemma4D skills. |
| `spec/` | Architecture and implementation contracts, including the Ratatui operator UX. |
| `milestones/` | Concrete milestone plans with tasks, tests, measurements, and exit criteria. |
| `tasks/` | Cross-milestone task conventions, dependency graph, measurement gates. |
| `codex/` | Goal prompts, orchestration prompts, install/use notes. |
| `references/` | Schemas, FFI skeletons, benchmark/TUI templates, configs, and shared templates from the uploaded skills package. |
| `SOURCES.md` | Source bibliography and research evidence map. |

## How to use with Codex

1. Unzip this package into an empty planning repo or the root of the implementation repo.
2. Start Codex from the package/repo root so it can discover `AGENTS.md`, `.codex/agents`, and `.agents/skills`.
3. Open `INDEX.md`, then `milestones/milestone-index.md`.
4. Start with `codex/goals/00-main-goal.md` or the specific milestone goal you want.
5. Ask Codex to spawn subagents only where the milestone calls for independent exploration, implementation, profiling, or verification.

Recommended first prompt:

```text
Read AGENTS.md, INDEX.md, spec/00-executive-summary.md, and milestones/M00-repo-bootstrap.md.
Use the milestone goal in codex/goals/M00-repo-bootstrap.goal.md.
Implement only M00. Use subagents only for read-only verification and do not widen scope.
```

## Evidence discipline

Every milestone has:

- scope boundaries,
- concrete task cards,
- verification commands,
- benchmark/profiling surfaces,
- blocker stop conditions,
- artifacts to produce,
- acceptance criteria.

Performance work must record exact commands, machine details, model revisions, context lengths, prompt sets, memory readings, and raw output paths. Correctness work must compare against known references before optimizing.

## Attached skill archive integration

The uploaded `research_review_agent_skills_v1` package was used as the base general-purpose research/review layer. This package keeps its Codex-compatible skills and agents, then adds Gemma4D-specific skills for Rust/MLX FFI, Gemma 4 model correctness, Ratatui TUI UX, MTP, KV/SSD/cache compression, LoRA/QLoRA adapters, Ratatui TUI control-plane work, and 16GB Mac profiling.
