# M05 — Ratatui Operator Console

## Goal

Introduce `gemma4d-tui`, a Ratatui-based local operator console for configuration, benchmark/profiling orchestration, logs, and future runtime/chat control.

## Scope

Implement the TUI skeleton, offline/mock provider, config validation surface, benchmark launcher surface, dashboard snapshots, and navigation model. Do not require model weights or a live daemon in this milestone. Do not integrate live chat or mutate runtime state except through explicit mock/file-provider commands.

## Why this milestone is placed here

M05 comes after greedy inference and the parity harness because the operator now needs a usable surface for configuration, test execution, profiling evidence, and later MTP/KV/adapter debugging. It comes before MTP/KV/adapters so those features can add pages and telemetry as they land.

## Tasks

- [ ] Add `crates/gemma4d-tui` with Ratatui/Crossterm/Tokio dependencies pinned through the workspace lockfile.
- [ ] Implement TUI startup/shutdown lifecycle with raw-mode and alternate-screen restoration.
- [ ] Implement event loop with input, resize, tick, render, and backend events.
- [ ] Implement `AppState`, `Action`, reducer, and page navigation.
- [ ] Implement pages: Dashboard, Config, Benchmarks, Logs, Help.
- [ ] Add disabled/placeholder pages for Chat, Cache, Adapters, and MTP with clear dependency messages.
- [ ] Implement `RuntimeProvider` trait plus `MockProvider` and `FileProvider`.
- [ ] Implement config validation view for `references/configs/tiny16.toml`-style configs.
- [ ] Implement benchmark run surface that records exact command, output directory, status, and report path.
- [ ] Add snapshot tests for 80x24 and 120x40 layouts.
- [ ] Add reducer/keybinding tests.
- [ ] Add a TUI usability smoke report template/output under `artifacts/tui/` or `benchmarks/out/M05/`.

## Measurements / evidence

- Record exact commands run.
- Store TUI screenshots/snapshots and test output under `benchmarks/out/M05/` or `artifacts/tui/`.
- Record mock dashboard render p50/p95 or equivalent timing if a stable local method exists.
- Record idle CPU observation or caveat.
- Record terminal lifecycle review evidence: normal quit, Ctrl-C, and panic/error path if feasible.

## Acceptance criteria

- [ ] `cargo test -p gemma4d-tui` passes.
- [ ] `cargo run -p gemma4d-tui -- --provider mock` launches and exits cleanly.
- [ ] Terminal state is restored on normal quit and controlled error path.
- [ ] Dashboard, Config, Benchmarks, Logs, and Help pages render in snapshot tests.
- [ ] Chat, Cache, Adapters, and MTP pages exist as dependency-aware placeholders.
- [ ] Config validation catches at least one intentionally invalid fixture.
- [ ] Benchmark surface records exact command and output path even when using mock provider.
- [ ] No MLX scheduler/native calls are made from the TUI in this milestone.

## Recommended Codex goal

Use `codex/goals/M05-tui-operator-console.goal.md`.

## Recommended skills

- `$gemma4d-milestone-execution`
- `$gemma4d-tui-operator-ux`
- `$spec-contract-compliance-review`
- `$performance-ab-benchmark-review` only for render/CPU observations

## Recommended subagents

- `tui_ux_engineer` for TUI architecture and usability review.
- `test-verifier` for independent terminal/snapshot test verification.
- `security-reliability-reviewer` for raw-mode, file-write, and command-launch safety.

## Blocked stop condition

If Ratatui/Crossterm/Tokio versions are incompatible with Rust 1.95.0 or the selected terminal environment cannot run TUI tests, stop with:

1. attempted versions/commands,
2. compatibility evidence,
3. smallest fallback that preserves the provider/reducer/page contracts,
4. next input required.
