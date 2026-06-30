---
name: gemma4d-tui-operator-ux
description: Use when implementing or reviewing the Gemma4D Ratatui operator console, including page architecture, async event loops, terminal lifecycle, provider boundaries, config/benchmark UX, snapshots, and safety constraints.
---

# Gemma4D TUI Operator UX Skill

## Use this skill when

- Adding or changing `crates/gemma4d-tui`.
- Designing Ratatui pages, widgets, keybindings, command palette, or navigation.
- Connecting the TUI to mock/file/http runtime providers.
- Adding TUI snapshot tests or terminal lifecycle tests.
- Reviewing TUI integration with benchmarks, profiling, adapters, cache, MTP, or server controls.

## Required reading

- `spec/13-tui-operator-ux.md`
- `milestones/M05-tui-operator-console.md`
- `references/tui-screen-map.md`
- `references/tui-keybindings.md`
- `references/configs/tui.toml`

## Hard rules

- Do not call native MLX scheduler APIs from the TUI.
- Keep TUI logic behind provider traits: mock, file, then HTTP/control provider.
- Restore terminal state on normal exit, Ctrl-C, and controlled error paths.
- Destructive actions require confirmation.
- Avoid unbounded redraw loops; use bounded frame/tick rates.
- Make snapshot renders deterministic.
- TUI pages must degrade gracefully when the daemon is not running.
- Remote adapter loading remains disabled unless explicitly trusted by local config.

## Review checklist

- Does the page show exact commands and raw output paths for benchmarks?
- Can the operator reproduce a run from the TUI output?
- Are config writes explicit and diffed before save?
- Are cache/adapters/MTP controls gated by feature readiness?
- Does every backend action have a visible pending/success/error state?
- Are keybindings documented in Help and tested in the reducer?
- Are terminal raw mode and alternate screen owned by one lifecycle object?
