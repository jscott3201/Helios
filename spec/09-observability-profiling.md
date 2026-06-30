# 09 — Observability and Profiling

## Required metrics

Expose Prometheus-style metrics and write benchmark JSONL records.

### Runtime

```text
gemma4d_requests_total
gemma4d_active_generations
gemma4d_queue_depth
gemma4d_errors_total{code}
gemma4d_memory_process_rss_bytes
gemma4d_memory_guard_rejections_total
```

### Inference

```text
gemma4d_prefill_tokens_total
gemma4d_decode_tokens_total
gemma4d_prefill_seconds
gemma4d_decode_seconds
gemma4d_ttft_seconds
gemma4d_tokens_per_second
```

### MTP

```text
gemma4d_mtp_attempted_tokens_total
gemma4d_mtp_accepted_tokens_total
gemma4d_mtp_acceptance_rate
gemma4d_mtp_rollbacks_total
gemma4d_mtp_auto_disabled_total
```

### Cache

```text
gemma4d_kv_active_bytes
gemma4d_prefix_cache_hits_total{tier}
gemma4d_prefix_cache_misses_total
gemma4d_ssd_cache_read_bytes_total
gemma4d_ssd_cache_write_bytes_total
gemma4d_cache_restore_failures_total
```

### TUI

```text
gemma4d_tui_render_seconds
gemma4d_tui_backend_events_total
gemma4d_tui_actions_total{action}
gemma4d_tui_provider_errors_total{provider}
```

### Adapters

```text
gemma4d_adapters_loaded
gemma4d_adapter_load_seconds
gemma4d_adapter_resident_bytes
gemma4d_adapter_evictions_total
gemma4d_adapter_requests_total{adapter_id}
```

## Benchmark record format

Every benchmark writes one JSONL record per run:

```json
{
  "timestamp": "2026-06-30T00:00:00Z",
  "machine": "MacBook ...",
  "macos": "...",
  "rustc": "1.95.0",
  "mlx_version": "...",
  "model_id": "...",
  "model_revision": "...",
  "adapter_id": null,
  "context_tokens": 16384,
  "generated_tokens": 128,
  "mode": "target_greedy",
  "kv_mode": "bf16",
  "mtp_enabled": false,
  "ttft_ms": 0,
  "prefill_tps": 0,
  "decode_tps": 0,
  "peak_rss_mb": 0,
  "raw_output_path": "benchmarks/out/..."
}
```

## Profiling tools

Use what is available locally, but record versions and limitations:

- Rust `cargo test`, `cargo bench`, `criterion` where useful.
- macOS Activity Monitor / `vm_stat` / `memory_pressure` snapshots.
- Instruments for allocation and Metal profiling when needed.
- MLX/native logs for tensor/cache sizes.
- llama.cpp baseline commands for comparison.

## Reporting

Every performance milestone produces a report using `references/templates/benchmark-report.md`.

## TUI-facing profiling

The TUI must make benchmark evidence reproducible by showing exact commands, output paths, model/config revisions, and report locations. TUI-specific reports should use `references/templates/tui-usability-report.md` and include launch/quit, config validation, benchmark command preview, benchmark stop, and log filtering workflows.

### TUI

```text
gemma4d_tui_frames_total
gemma4d_tui_render_seconds
gemma4d_tui_event_lag_seconds
gemma4d_tui_server_disconnects_total
gemma4d_tui_stream_deltas_total
gemma4d_tui_bounded_log_drops_total
gemma4d_tui_destructive_confirmations_total
```

## TUI overhead records

M11 records separate TUI overhead evidence:

```json
{
  "timestamp": "2026-06-30T00:00:00Z",
  "terminal": "Apple Terminal/iTerm/etc",
  "size": "120x40",
  "tick_ms": 250,
  "idle_rss_mb": 0,
  "idle_cpu_percent": 0,
  "render_p50_ms": 0,
  "render_p95_ms": 0,
  "streaming_chat_added_latency_ms": 0,
  "snapshot_suite": "passed",
  "raw_output_path": "benchmarks/out/M11/..."
}
```
