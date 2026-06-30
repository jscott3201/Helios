# Measurement Gates

## Gate A — Correctness before speed

No performance optimization is accepted unless the relevant correctness tests pass before and after the change.

## Gate B — Baseline before candidate

Every benchmark must identify:

- baseline command/state,
- candidate command/state,
- workload,
- model revision,
- machine/environment,
- raw output path,
- variance/repeated samples where feasible.

## Gate C — Memory evidence for tiny16

Memory-sensitive milestones must report:

- process RSS,
- MLX/native-reported KV/model/cache bytes when available,
- macOS memory pressure observations if available,
- max context reached,
- failure mode if rejected.

## Gate D — Cache restore exactness

A cache restore is not accepted unless fresh-prefill and restored-prefill outputs match under the same mode, or the mismatch is explicitly expected for an experimental lossy compression mode and documented.

## Gate E — Adapter namespace safety

No adapter milestone is accepted unless wrong-adapter cache reuse is tested and rejected.

## Gate F — TUI reproducibility

TUI-driven benchmark/config/profiling workflows are not accepted unless the TUI shows or records:

- exact command invoked,
- config path and effective profile,
- output directory/report path,
- provider mode,
- success/error state,
- reproduction bundle path when exported.

The TUI must also restore terminal state on normal exit and controlled error paths.

## TUI gates

- [ ] Fake-server TUI tests pass without model artifacts.
- [ ] Ratatui TestBackend/insta snapshots cover MVP screens at required terminal sizes.
- [ ] Real-server smoke covers health, metrics, runtime snapshot, and one streaming chat.
- [ ] TUI idle RSS/CPU and render timing are recorded.
- [ ] TUI buffers are bounded and destructive actions are confirmation-gated.
