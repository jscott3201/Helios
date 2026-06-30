# Milestone Index

## Execution order

```text
M00 repo bootstrap
M01 native MLX loader
M02 tokenizer/chat/config
M03 greedy text inference
M04 reference parity + benchmark harness
M05 Ratatui operator console
M06 MTP speculative decoding
M07 in-memory KV cache core
M08 SSD prefix cache
M09 KV compression research
M10 dynamic LoRA/QLoRA adapters
M11 OpenAI-compatible server + live TUI attach
M12 tiny16 profiling release gate
```

## Dependency rule

A milestone may begin only when all predecessor acceptance criteria are met or explicitly waived in a decision record under `docs/decisions/`.

## Why TUI is M05

The TUI arrives after the first reference/benchmark harness so it can make configuration and evidence loops usable early. It arrives before MTP/KV/adapters/server so each later subsystem can add pages, metrics, and controls incrementally instead of bolting UX on at the end.

## Parallelism rule

Codex may spawn read-only subagents to prepare later milestones, but implementation patches should stay inside the active milestone unless requested.

## Measurement-first rule

Any milestone with performance claims must produce raw output under `benchmarks/out/<milestone>/` and a summary report using `references/templates/benchmark-report.md` or `references/templates/tui-usability-report.md` when the milestone is TUI-facing.
