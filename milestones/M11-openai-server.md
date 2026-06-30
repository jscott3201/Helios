# M11 — OpenAI-Compatible Server

## Goal

Expose local chat completions, streaming, model/adapters endpoints, health, and metrics.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Add `gemma4d-server` using axum/tokio or selected stack.
- [ ] Implement `/v1/chat/completions` non-streaming then streaming.
- [ ] Implement `/v1/models`, `/v1/adapters`, `/health`, `/metrics`.
- [ ] Add request admission and memory guard errors.
- [ ] Add integration tests with small/stub backend and optional full model tests.
- [ ] Implement TUI `HttpProvider` attach path for health, metrics, adapters, cache summaries, benchmark status, and streaming chat status.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M11/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] Simple streaming chat works locally.
- [ ] Adapter selection field is parsed and routed.
- [ ] Metrics endpoint exposes core counters.
- [ ] Server binds localhost by default.
- [ ] Unsafe remote adapter loading not exposed.
- [ ] TUI can attach to the localhost server/control provider and display live health/metrics with stub backend.

## Recommended Codex goal

Use `codex/goals/M11-openai-server.goal.md`.

## Recommended skills

- `$gemma4d-milestone-execution`
- `$spec-contract-compliance-review`
- `$performance-ab-benchmark-review` when this milestone touches runtime performance
- milestone-specific project skill as applicable

## Blocked stop condition

If a required external dependency, model artifact, MLX API, or machine capability is unavailable, stop with:

1. attempted paths,
2. command/source evidence,
3. minimal repro or diagnostic,
4. next input required.
