# 12 — Risk Register

| Risk | Severity | Trigger | Mitigation | Owner skill/agent |
|---|---:|---|---|---|
| MLX C++/C ABI mismatch or lifetime bug | blocker | crashes, leaks, invalid tensors | tiny C ABI, FFI smoke tests, sanitizer/instruments where possible | `$gemma4d-rust-mlx-ffi`, `mlx_ffi_engineer` |
| Gemma 4 config mismatch | blocker | wrong logits, unsupported checkpoint | strict config validation and unsupported-config error | `$gemma4d-model-correctness` |
| MTP divergence | high | MTP greedy differs from target greedy | exactness suite, auto-disable MTP | `$gemma4d-mtp-speculative-decoding` |
| 16GB memory cliff | high | system memory pressure/OOM | hard memory guard, conservative context, telemetry | `$gemma4d-16gb-mac-profiling` |
| KV restore corruption | high | wrong logits after restore | explicit manifest, checksums, restore parity tests | `$gemma4d-kv-cache-offload` |
| Cross-adapter cache contamination | high | Rust adapter uses Python/base KV | adapter hash in cache key, rejection tests | `$gemma4d-dynamic-adapters` |
| SSD cache latency spikes | medium | slow decode or stalls | restore before prefill only in MVP, no mid-decode SSD fetch | performance analyst |
| Compression quality regression | medium | logit/token drift, tool JSON failures | compare per mode, global-only first, fallback to BF16/q8 | `$gemma4d-kv-cache-offload` |
| Scope creep into general engine | medium | broad model traits before Gemma stable | milestone boundaries and AGENTS.md constraints | milestone execution skill |
| TUI corrupts terminal state or distorts profiling | medium | panic leaves raw mode, high UI CPU/RSS, unbounded logs | mock/file provider tests, snapshot tests, panic cleanup, bounded buffers, overhead records | `$gemma4d-tui-operator-ux`, `tui_ux_engineer` |
| License contamination | high | code copied from incompatible repo | license review before copying | security-reliability-reviewer |

## TUI terminal lifecycle risk

Risk: raw-mode or alternate-screen failures can leave the user's terminal in a broken state, or a busy redraw loop can add CPU pressure while profiling inference.

Mitigation:

- one lifecycle owner for raw mode and alternate screen,
- restore on normal exit, Ctrl-C, and controlled error path,
- bounded frame/tick rates,
- deterministic snapshot tests,
- mock/file provider first so UI bugs cannot corrupt native MLX state.
