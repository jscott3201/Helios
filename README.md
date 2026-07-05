# Helios / gemma4d

Helios is the `gemma4d` Rust + MLX runtime workspace for local Gemma 4 12B
4-bit inference on Apple Silicon. The project is optimized first for a 16 GB
MacBook profile, with a narrow C/C++ native boundary around MLX and a
text-only runtime path.

## Current Status

The project has moved past the original M12 helper-backed release gate. M12 is
still useful as the historical correctness and tiny16 baseline, but the active
runtime work is now the persistent native graph path.

Current state:

- `generate` can run local Gemma 4 12B 4-bit text generation.
- `serve` provides a localhost OpenAI-compatible API with `stub`,
  `real-helper`, and `persistent-native` backends.
- `serve --model-path PATH` selects `persistent-native` when `--backend` is
  omitted.
- The native server path has measured default-wiring evidence for long-context
  prefill and native decode KV evaluation.
- The Ratatui operator console remains a local UI over provider/client APIs.
- MTP, cache, adapter, and DSpark work remains feature-flagged or experimental
  unless a benchmark row explicitly says otherwise.

`BENCHMARKS.md` is the source of truth for measured claims. Raw benchmark
outputs, traces, and release artifacts are intentionally ignored by git and
kept under `artifacts/`, `benchmarks/out/`, or `target/`.

## Evidence Snapshot

Recent native graph and server evidence:

| Area | Result | Status |
| --- | --- | --- |
| Native server prefill, 16K | `87387.199 -> 41711.194 ms`, +52.269%, peak `21.874 -> 7.638 GB` | Default path evidence |
| Native server prefill, 8K | `31285.354 -> 22618.497 ms`, +27.703% | Default path evidence |
| Server default sentinel, 16K | `43289.598 -> 41360.009 ms`, +4.457%, peak `7.639 GB` | Accepted low-N sentinel |
| Server default sentinel, 24K | `60939.960 -> 61254.060 ms`, -0.515%, peak `7.859 GB` | Accepted low-N sentinel |
| Native decode KV eval | Chat/tool p50 around `81 ms -> 70 ms` | Runtime default |
| XR71 full-attention capacity candidate | Total decode `73947.065 -> 68583.341 ms`, +7.253%, peak `7.929 GB` | Experimental, not default |
| MTP selected chat/tool lanes | Selected lanes about +31%, protected aggregate about +20% | Experimental, not default |

The largest remaining native graph bottleneck is not capacity growth or visible
slice update overhead. XR71 measured update overhead at roughly `0.010 ms/token`;
the remaining instability is full-attention deferred evaluation tail latency,
especially `chat_short_1k_001` p95/p99 behavior.

## Quick Start

Install the Rust toolchain declared by the project:

```bash
rustup toolchain install 1.95.0
```

Place the local model artifact outside git, for example:

```text
artifacts/models/gemma-4-12B-it-4bit
```

Run the verification bundle:

```bash
make verify
```

Run a native graph generation smoke:

```bash
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 \
  cargo run -p gemma4d-server -- generate \
  --model-path artifacts/models/gemma-4-12B-it-4bit \
  --prompt "Write one sentence about local inference." \
  --max-new-tokens 32
```

Start the local server on the persistent native path:

```bash
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 \
  cargo run -p gemma4d-server -- serve \
  --model-path artifacts/models/gemma-4-12B-it-4bit \
  --bind 127.0.0.1:8080 \
  --max-context-tokens 32768 \
  --memory-budget-mb 12288
```

Use `--backend real-helper` to force the helper-backed path, or `--backend stub`
for control-plane and API tests. When `--model-path` is present and
`--backend` is omitted, `serve` selects `persistent-native`.

Call the OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4d-local",
    "messages": [{"role": "user", "content": "Give me one terse status line."}],
    "max_tokens": 32
  }'
```

Run the local operator console against a running server:

```bash
cargo run -p gemma4d-tui -- \
  --provider http \
  --server-url http://127.0.0.1:8080 \
  --config references/configs/tui.toml
```

## Repository Map

| Path | Purpose |
| --- | --- |
| `crates/gemma4d-server/` | CLI, server, OpenAI-compatible API, backend routing |
| `crates/gemma4d-ffi/` | Rust FFI wrapper around the narrow native MLX boundary |
| `native/gemma4_mlx/` | C/C++ MLX graph implementation and native probes |
| `crates/gemma4d-tui/` | Ratatui operator console |
| `scripts/` | Verification, smoke, and benchmark entry points |
| `benchmarks/` | Benchmark manifests and harness outputs |
| `docs/` | Evidence, design notes, and release artifacts |
| `codex/` | Milestones, goals, and agent-oriented task contracts |

## Scope Boundaries

Supported by current evidence:

- Local Gemma 4 12B 4-bit text generation.
- Helper-backed and persistent native server paths.
- Native graph prefill and decode evaluation under explicit MLX/native flags.
- Tiny16-oriented benchmark and memory gates.
- Server admission, default backend selection, and long-context sentinels.
- Local TUI operator workflows over the provider/client boundary.

Not claimed yet:

- Multimodal inference.
- Production internet-facing serving.
- Broad MTP default-on behavior.
- DSpark speed or tiny16 viability.
- Default promotion of XR70/XR71 full-attention update candidates.
- Long-running 32K decode throughput guarantees.

## Next Work

The next high-value sequence is:

1. Native first-token warmup policy candidate. XR76 showed profile-mode
   perturbation is not the main tail source and a harness-only same-shape warmup
   probe cut `chat_short_1k_001` raw p99 from `177.571` to `86.680 ms`; the
   next step is a real default-off policy that accounts for warmup cost and
   shape scope.
2. Broader MTP promotion remains parked until protected aggregate speed clears
   the release threshold. XR73 accepts only explicit scoped chat/tool opt-in.

DSpark stays parked for now. XR60 alignment remains useful background evidence,
but it is not the shortest path to the current theoretical maximum.

## Documentation

- `BENCHMARKS.md` records benchmark commands, outputs, interpretation, and
  default/experimental decisions.
- `docs/xr-current-state-review.md` records the current next-goal review after
  XR74 readiness.
- `docs/evidence/XR74-native-default-readiness.md` records the XR74 readiness
  decision.
- `docs/evidence/` holds release and milestone evidence packets.
- `codex/milestones/` and `codex/goals/` hold milestone contracts.
- `AGENTS.md` defines repository operating constraints for Codex agents.
