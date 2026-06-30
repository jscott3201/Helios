# M12 tiny16 Release Report

This report records the M12 local release gate completed on 2026-06-30.

## Status

M12 is `ready_with_known_limitations` for the current tiny16 runtime slice. The release gate validates real Gemma 4 12B 4-bit target generation through the current C ABI / MLX-LM helper path, local serving, request admission, metrics, MTP/cache/adapter fixture paths, TUI workflow evidence, and failure isolation. The hand-written native graph remains a tracked follow-up, so this report does not claim native-graph performance.

## Environment

Raw environment capture is in `benchmarks/out/M12/real-matrix/summary.json` and `benchmarks/out/M12/release-gate.json`.

| Item | Value |
|---|---|
| Machine | `Darwin Justins-MBP 25.6.0 ... RELEASE_ARM64_T8142 arm64` |
| macOS | `ProductVersion: 26.6` |
| MLX | `mlx=0.31.2 mlx_lm=0.31.3` |
| Model | `artifacts/models/gemma-4-12B-it-4bit` |
| Profile | `tiny16` |
| Hard memory limit | 12288 MB |
| System headroom | 3072 MB |

## Real Target Matrix

The M12 real target matrix uses `gemma4d generate` against the local 4-bit model artifact. Contexts 1K through 16K use 128 generated tokens. The 32K case uses a one-token decode memory probe to protect tiny16 headroom while still proving prefill and one decode step.

| Context | Generated | Result | TTFT ms | Decode tok/s | Peak native GB | Evidence |
|---:|---:|---|---:|---:|---:|---|
| 1K | 128/128 | Passed | 2080.211 | 15.643 | 8.065 | `real-matrix/records.jsonl` |
| 4K | 128/128 | Passed | 9214.994 | 12.244 | 9.480 | `real-matrix/records.jsonl` |
| 8K | 128/128 | Passed | 18738.957 | 15.277 | 9.833 | `real-matrix/records.jsonl` |
| 16K | 128/128 | Passed | 41153.129 | 12.864 | 10.512 | `real-matrix/records.jsonl` |
| 32K | 1/1 | Passed | 98323.500 | n/a | 11.888 | `real-matrix/records.jsonl` |

The separate `release-gate.json` context matrix remains useful control-plane evidence for server admission and memory guard behavior, but it is not the real target benchmark source.

## Feature Gates

| Gate | Result | Evidence |
|---|---|---|
| MTP exactness/acceptance | Passed | `benchmarks/out/M12/mtp-fixture.json` |
| RAM prefix cache warm restore | Passed | `benchmarks/out/M12/ram-restore-matrix.json` |
| SSD prefix cache warm restore | Passed | `benchmarks/out/M12/ssd-benchmark.json` |
| Rust expert adapter load/route/unload | Passed | `benchmarks/out/M12/adapter-fixture.json` and `release-gate.json` |
| Server streaming/metrics/admission smoke | Passed | `benchmarks/out/M12/server-smoke.json` |
| TUI release walkthrough | Passed | `benchmarks/out/M12/tui-walkthrough/tui-release-walkthrough.md` |
| TUI HTTP attach walkthrough | Passed | `benchmarks/out/M12/tui-http-walkthrough/tui-release-walkthrough.md` |
| TUI failure does not kill server | Passed | `server-health-before-tui-failure.json`, `tui-fail-after-init-live-server.log`, `server-health-after-live-tui-failure.json`, `server-metrics-after-live-tui-failure.prom` |

## Fallback / Disable Paths

| Feature | Path |
|---|---|
| Sampling | Nonzero temperature rejects with `unsupported_model_config`. |
| Unknown model | Unsupported model IDs reject with `unsupported_model_config`. |
| 32K pressure | Real target 32K one-token probe passes at 11.888 GB peak native memory; server admission guard still rejects unsafe 32K stub requests with `memory_guard_rejected`. |
| MTP | `references/configs/tiny16.toml` has `speculative.enabled = false`; MTP exactness remains independently verified by fixture. |
| SSD cache | `references/configs/tiny16.toml` has `ssd_cache.enabled = false`; restore-before-prefill remains independently verified by fixture. |
| Remote adapters | Server and adapter fixtures reject remote/caller-supplied adapter sources. |
| Cache deletion | `/v1/cache/evict` returns `read_only_stub` in the local M12 path. |
| TUI disabled/crash | Server health remains `ok` after a controlled TUI failure. |

## Known Limitations

| Severity | Area | Limitation | Mitigation |
|---|---|---|---|
| Medium | Native graph | M12 validates real target generation through the current MLX-LM helper path, not the hand-written native graph. | Keep the native graph follow-up open and require fresh 1K/4K/8K/16K/32K evidence before switching the default target path. |
| Low | Model revisions | `references/configs/tiny16.toml` still uses `target_revision = "PIN_ME"`. | Pin target and drafter revisions before distributable release or model-specific benchmark claims. |
| Low | HTTP stack | The localhost server uses the M11 stdlib HTTP stack selected for offline verifiability. | Revisit a maintained HTTP stack before non-localhost serving. |

## Raw Artifacts

Raw M12 artifacts are intentionally ignored:

- `benchmarks/out/M12/release-gate.json`
- `benchmarks/out/M12/release-report.md`
- `benchmarks/out/M12/real-matrix/records.jsonl`
- `benchmarks/out/M12/real-matrix/summary.json`
- `benchmarks/out/M12/real-matrix/report.md`
- `benchmarks/out/M12/mtp-fixture.json`
- `benchmarks/out/M12/ram-restore-matrix.json`
- `benchmarks/out/M12/ssd-benchmark.json`
- `benchmarks/out/M12/adapter-fixture.json`
- `benchmarks/out/M12/server-smoke.json`
- `benchmarks/out/M12/tui-walkthrough/**`
- `benchmarks/out/M12/tui-http-walkthrough/**`
- `benchmarks/out/M12/server-health-*.json`
- `benchmarks/out/M12/server-metrics-*.prom`
- `benchmarks/out/M12/tui-fail-after-init*.log`
