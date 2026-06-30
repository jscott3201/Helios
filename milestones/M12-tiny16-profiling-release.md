# M12 — tiny16 Profiling and Release Gate

## Goal

Validate the whole runtime on a 16GB MacBook profile with measured memory, TTFT, decode, MTP, cache, and adapter behavior.

## Scope

Implement only this milestone and the minimum stubs needed for tests/builds. Do not optimize beyond the measurements requested here.

## Tasks

- [ ] Run standard benchmark matrix at 1K/4K/8K/16K/32K.
- [ ] Run MTP exactness and acceptance benchmarks if enabled.
- [ ] Run RAM and SSD prefix cache warm TTFT tests.
- [ ] Run one Rust expert adapter load/route/unload test.
- [ ] Run TUI-assisted validation of dashboard, chat, cache, adapter, benchmark, and config screens.
- [ ] Run release-readiness review and risk audit.
- [ ] Produce release report and known limitations.
- [ ] Run a TUI-driven release walkthrough covering config validation, benchmark launch, metrics review, adapter status, cache status, and report export.

## Measurements / evidence

- Record exact commands run.
- Store raw outputs under `benchmarks/out/M12/` when benchmarks or profiling are involved.
- Update a decision record if a spec assumption changes.

## Acceptance criteria

- [ ] Release report exists.
- [ ] 16K passes end-to-end.
- [ ] 32K passes or fails gracefully with memory evidence.
- [ ] TUI can be disabled or crash without killing/corrupting the server.
- [ ] All enabled features have fallback/disable paths.
- [ ] No blocker findings remain open.
- [ ] TUI release walkthrough report exists under `benchmarks/out/M12/` or `artifacts/tui/`.

## Recommended Codex goal

Use `codex/goals/M12-tiny16-profiling-release.goal.md`.

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

## TUI validation note

Use the TUI during at least one validation pass, but do not rely on the TUI as the only evidence source. Every TUI-observed result must also have a raw benchmark/report/log artifact path.
