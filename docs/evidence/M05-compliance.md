# M05 Compliance Matrix

## Scope

- Milestone: `milestones/M05-tui-operator-console.md`
- Goal: `codex/goals/M05-tui-operator-console.goal.md`
- Spec: `spec/13-tui-operator-ux.md`
- References: `references/tui-screen-map.md`, `references/tui-keybindings.md`, `references/configs/tui.toml`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M05-T01 | Add `crates/gemma4d-tui` with Ratatui/Crossterm/Tokio dependencies pinned through the lockfile. | `crates/gemma4d-tui/Cargo.toml`; `Cargo.lock`; `cargo tree -p gemma4d-tui --depth 1`. | Complete | None. |
| M05-T02 | Add terminal startup/shutdown lifecycle with restore on normal and error paths. | `crates/gemma4d-tui/src/terminal.rs`; tests `terminal_lifecycle_restores_after_normal_quit` and `terminal_lifecycle_restores_after_controlled_error`. | Complete | Real interactive TTY smoke was not available in this command runner. |
| M05-T03 | Add event loop, input, resize, tick, render, and backend events. | `terminal.rs` `AppEvent`, `next_event`, and `run_interactive`; provider events reduced into state. | Complete | None for M05 scope. |
| M05-T04 | Add `AppState`, `Action`, reducer, and page navigation. | `crates/gemma4d-tui/src/app.rs`; reducer/keybinding tests. | Complete | None. |
| M05-T05 | Render Dashboard, Config, Benchmarks, Logs, and Help pages. | `crates/gemma4d-tui/src/ui.rs`; snapshot tests; generated snapshots under `benchmarks/out/M05/snapshots/`. | Complete | None. |
| M05-T06 | Add disabled placeholders for Chat, Cache, Adapters, and MTP. | `PageId::dependency_message`; placeholder render tests and snapshots. | Complete | None. |
| M05-T07 | Implement `RuntimeProvider`, `MockProvider`, and `FileProvider`. | `crates/gemma4d-tui/src/provider.rs`; CLI provider selection. | Complete | File provider is offline/file-only until runtime daemon milestones. |
| M05-T08 | Add config validation for tiny16/TUI-style configs and one invalid fixture. | `crates/gemma4d-tui/src/config.rs`; `tests/fixtures/invalid-tiny16.toml`; validation command output. | Complete | None. |
| M05-T09 | Add benchmark launcher surface recording command, output path, status, and report path. | `BenchmarkRecord`; `MockProvider::start_benchmark`; test and CLI benchmark output. | Complete | M05 intentionally does not spawn real model benchmark processes. |
| M05-T10 | Add snapshot and reducer/keybinding tests. | `crates/gemma4d-tui/tests/m05_acceptance.rs`; `cargo test -p gemma4d-tui`. | Complete | None. |
| M05-T11 | Add TUI usability smoke evidence. | `benchmarks/out/M05/tui-usability-report.md`; `docs/evidence/M05.md`. | Complete | Idle CPU observation is caveated because the runner is noninteractive. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| `cargo test -p gemma4d-tui` passes. | 9 tests passed. | Complete |
| `cargo run -p gemma4d-tui -- --provider mock` launches/exits cleanly. | Command rendered headless snapshot and exited 0. | Complete |
| Terminal state restored on normal quit and controlled error path. | Dedicated lifecycle tests cover both paths. | Complete |
| Dashboard/Config/Benchmarks/Logs/Help pages render in snapshots. | Snapshot tests plus generated 80x24 and 120x40 files. | Complete |
| Chat/Cache/Adapters/MTP dependency-aware placeholders exist. | Placeholder snapshots and tests. | Complete |
| Config validation catches at least one intentionally invalid fixture. | Invalid fixture reports runtime headroom and model target errors. | Complete |
| Benchmark surface records exact command/output path/status/report path with mock provider. | CLI benchmark JSON shows `completed`, command, `out_dir`, and `report_path`. | Complete |
| No MLX scheduler/native calls from TUI. | `rg` check found no FFI/native/MLX call references in `crates/gemma4d-tui`. | Complete |

## Coverage Summary

- Implemented and tested: reducer navigation, keybindings, page rendering, provider boundary, config validation, benchmark records, terminal lifecycle, snapshots, and render timing.
- Implemented but caveated: interactive terminal behavior; lifecycle restore is unit-tested, while local command evidence used headless mode because stdin/stdout were not TTYs.
- Not implemented by design: live chat, live cache mutation, live adapter loading, MTP controls, daemon HTTP provider, and real benchmark process orchestration.
