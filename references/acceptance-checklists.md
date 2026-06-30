# Acceptance Checklists

## Correctness

- [ ] Tokenizer fixtures pass.
- [ ] Chat-template fixtures pass.
- [ ] Unsupported model configs fail clearly.
- [ ] Greedy output deterministic for fixture prompts.
- [ ] MTP greedy equals non-MTP greedy when MTP enabled.
- [ ] Cache restore equals fresh prefill for same mode.
- [ ] Adapter namespaces cannot cross-contaminate.

## Performance/profiling

- [ ] Baseline captured before candidate.
- [ ] Exact commands recorded.
- [ ] Environment recorded.
- [ ] Raw outputs retained.
- [ ] Variance/caveat noted.
- [ ] Correctness guardrails passed.

## Safety/release

- [ ] Server binds localhost by default.
- [ ] Remote adapter loading disabled by default.
- [ ] Path traversal rejected.
- [ ] Unsafe/FFI reviewed.
- [ ] License review performed before copying external code.
- [ ] Memory guard failure is graceful.

## TUI/operator UX

- [ ] Terminal restores after normal exit.
- [ ] Terminal restores after controlled error path.
- [ ] Dashboard/config/benchmark/log/help pages have deterministic snapshot tests.
- [ ] TUI provider boundary prevents direct native MLX scheduler calls.
- [ ] Config writes require explicit confirmation and show diff.
- [ ] Benchmark workflows record exact command and output path.
- [ ] Destructive actions require confirmation.
- [ ] TUI degrades gracefully when daemon is unavailable.
