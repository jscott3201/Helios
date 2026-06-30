# TUI Screen Map

## Navigation

```text
Dashboard   Config   Benchmarks   Chat   Cache   Adapters   MTP   Logs   Help
```

## Dashboard

- Runtime health and daemon/provider mode.
- Model target/drafter load state.
- Active adapter and routing mode.
- Memory gauges: process RSS, native/MLX bytes, KV bytes, adapter bytes.
- Throughput: TTFT, prefill tok/s, decode tok/s.
- Cache summary: RAM/SSD hit rate, bytes, restore failures.
- MTP summary: enabled, block size, acceptance, rollbacks.
- Recent errors.

## Config

- Current profile path and effective settings.
- Validation results and warnings.
- Derived memory budget for tiny16.
- Diff-before-save modal.

## Benchmarks

- Benchmark profile list.
- Active run progress.
- Exact command preview/copy.
- Output directory and latest report.
- Cold/warm TTFT and memory summaries when available.

## Chat

- Placeholder in M05.
- Live after server/control provider integration.
- Composer, streaming response, adapter selector, MTP status, token count, stop button.

## Cache

- Placeholder in M05.
- Later: namespaces, RAM/SSD blocks, hit/miss/evict counters, compression modes, flush confirmation.

## Adapters

- Placeholder in M05.
- Later: manifest validation, hot/warm/cold residency, load/unload/pin, cache namespace, MTP acceptance.

## MTP

- Placeholder in M05.
- Later: drafter compatibility, exactness status, acceptance rate, rollback details, auto-disable reason.

## Logs

- Live or file-backed log tail.
- Level/component filters.
- Search and copy event details.

## Help

- Keybindings.
- Current provider mode.
- Page readiness/dependency map.
