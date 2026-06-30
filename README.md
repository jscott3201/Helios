# Helios

Local Gemma 4 inference experiments for Apple Silicon, built around the
`gemma4d` Rust workspace and an MLX backend.

> **Status: M12 tiny16 release gate passed with known limitations.** Helios can
> run the current Gemma 4 12B 4-bit target path locally through the Rust CLI and
> C ABI / MLX-LM helper, expose a localhost OpenAI-compatible control API, and
> drive a Ratatui operator console. The hand-written native Gemma 4 graph is
> still a tracked follow-up, so current benchmark claims are for the helper-backed
> target path only.

Helios is aimed at making a 16GB MacBook a useful local inference and operator
testbed for Gemma 4. The repo focuses on measurable runtime behavior: context
length, memory pressure, first-token latency, decode throughput, prefix cache
restore behavior, speculative decoding correctness, adapter routing, server
admission guards, and TUI workflows.

## What You Get

- Rust 1.95.0 workspace with a narrow Rust-to-MLX FFI boundary.
- Local Gemma 4 12B 4-bit generation via `gemma4d-server generate`.
- Localhost-only OpenAI-compatible serving surface for chat, metrics, runtime
  snapshots, cache summaries, adapters, config validation, and benchmarks.
- Ratatui operator console for config, chat, cache, adapters, MTP, benchmark,
  dashboard, and log workflows.
- RAM prefix cache and SSD cold prefix cache fixture coverage.
- MTP speculative decoding exactness fixtures and disabled-by-default tiny16
  fallback behavior.
- Adapter import/load/route/unload control-plane coverage.
- Committed milestone evidence under `docs/evidence/`.

## Latest Benchmarks

The latest committed release report is
[`docs/evidence/M12-release-report.md`](docs/evidence/M12-release-report.md).
The raw benchmark records live under `benchmarks/out/M12/` and are intentionally
ignored by Git.

Environment for this run:

| Item | Value |
|---|---|
| Date | 2026-06-30 |
| Machine | Apple Silicon MacBook, `arm64`, Darwin 25.6.0 |
| macOS | 26.6 |
| Rust | 1.95.0 |
| MLX | `mlx=0.31.2`, `mlx_lm=0.31.3` |
| Model | `artifacts/models/gemma-4-12B-it-4bit` |
| Mode | `target_greedy_mlx_lm_helper_via_c_abi` |

M12 real target matrix:

| Context | Generated | TTFT ms | Prefill tok/s | Decode tok/s | Peak native GB | Wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 128/128 | 2080.211 | 492.258 | 15.643 | 8.065 | 12.376s |
| 4K | 128/128 | 9214.994 | 444.493 | 12.244 | 9.480 | 21.704s |
| 8K | 128/128 | 18738.957 | 437.164 | 15.277 | 9.833 | 29.236s |
| 16K | 128/128 | 41153.129 | 398.123 | 12.864 | 10.512 | 53.313s |
| 32K | 1/1 | 98323.500 | 333.267 | n/a | 11.888 | 100.958s |

The 32K row is a one-token decode memory probe, not a sustained 128-token decode
run. It exists to prove prefill plus one decode step within the tiny16 memory
envelope without pushing the machine into avoidable memory pressure.

## Quick Start

Install the Rust toolchain from `rust-toolchain.toml`, then verify the workspace:

```bash
make verify
```

Place the local MLX 4-bit model artifact at:

```text
artifacts/models/gemma-4-12B-it-4bit
```

`artifacts/` is ignored, so model weights and downloaded assets stay local.

Run a small local generation:

```bash
cargo run -p gemma4d-server -- generate \
  --model-path artifacts/models/gemma-4-12B-it-4bit \
  --context-tokens 1024 \
  --repeat-token 1 \
  --max-context-tokens 32768 \
  --max-new-tokens 16 \
  --json
```

Start the localhost server:

```bash
cargo run -p gemma4d-server -- serve \
  --bind 127.0.0.1:8080 \
  --max-context-tokens 32768 \
  --memory-budget-mb 12288
```

Attach the TUI to the server:

```bash
cargo run -p gemma4d-tui -- \
  --provider http \
  --server-url http://127.0.0.1:8080 \
  --config references/configs/tiny16.toml
```

Run the M12 real target benchmark matrix:

```bash
cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- \
  --out-dir benchmarks/out/M12/real-matrix \
  --model-path artifacts/models/gemma-4-12B-it-4bit
```

## Repository Layout

| Path | Purpose |
|---|---|
| `crates/gemma4d-server` | CLI generation path and localhost OpenAI-compatible API. |
| `crates/gemma4d-ffi` | Safe Rust wrappers around the native MLX C ABI. |
| `native/gemma4_mlx` | C/C++ MLX boundary and helper integration. |
| `crates/gemma4d-engine` | MTP and generation orchestration surfaces. |
| `crates/gemma4d-kv` | RAM and SSD prefix cache experiments. |
| `crates/gemma4d-adapters` | Adapter registry and routing control-plane code. |
| `crates/gemma4d-tui` | Ratatui operator console. |
| `crates/gemma4d-bench` | Benchmark, fixture, and release-gate runners. |
| `docs/evidence` | Committed milestone evidence and release reports. |
| `references/configs` | Local profiles, including `tiny16.toml`. |

Folders that start with `_`, including `_spec/`, are private source material and
must not be committed.

## Scope Boundaries

Current M12 evidence supports:

- Real Gemma 4 12B 4-bit target generation through the C ABI / MLX-LM helper.
- 1K, 4K, 8K, and 16K 128-token decode benchmark rows.
- 32K prefill plus one-token decode memory probe.
- Server admission and fallback behavior for localhost use.
- TUI walkthroughs and controlled TUI failure isolation.
- MTP, RAM cache, SSD cache, and adapter fixture paths.

Current M12 evidence does not claim:

- Hand-written native Gemma 4 graph performance.
- Production non-localhost serving readiness.
- Revision-pinned distributable model release status.
- Long-running 32K decode throughput.
- Multimodal Gemma runtime support.

## Documentation

- [`docs/evidence/M12.md`](docs/evidence/M12.md) - M12 implementation and
  verification summary.
- [`docs/evidence/M12-release-report.md`](docs/evidence/M12-release-report.md) -
  user-facing tiny16 release report and benchmark table.
- [`docs/evidence/M12-compliance.md`](docs/evidence/M12-compliance.md) - M12
  acceptance and compliance matrix.
- [`docs/evidence/M12-release-readiness.md`](docs/evidence/M12-release-readiness.md) -
  release-readiness review.
- [`AGENTS.md`](AGENTS.md) - repo instructions for coding agents.

## License

The workspace metadata declares `MIT OR Apache-2.0`.
