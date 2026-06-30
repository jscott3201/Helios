# 08 — Serving API and Runtime

## Initial API

Implement an OpenAI-compatible subset:

```text
GET  /health
GET  /v1/models
GET  /v1/adapters
POST /v1/adapters/load
POST /v1/adapters/unload
POST /v1/chat/completions
GET  /metrics
```

## `/v1/chat/completions`

Required request fields:

```text
model
messages
stream optional
temperature optional, default 0 initially
max_tokens optional
adapter optional
```

MVP limitations:

```text
single active generation
temperature=0 only until sampler milestone
MTP only when enabled and exactness-proven
no remote images/audio/video
```

## Runtime config

Use TOML. See `references/configs/tiny16.toml`.

Priority order:

```text
CLI flags > environment > config file > profile defaults
```

## Scheduler admission

Reject or queue requests when:

- active generation is already running and queue is full,
- requested context exceeds profile limit,
- requested adapter is unavailable/untrusted,
- memory guard predicts unsafe allocation,
- MTP requested but not verified for active target/adapter/KV mode.

## Error model

Return structured JSON errors with stable codes:

```text
unsupported_model_config
context_too_large
adapter_not_loaded
adapter_manifest_mismatch
memory_guard_rejected
mtp_not_verified
cache_restore_failed
native_backend_error
```

## TUI attach/control surface

The OpenAI-compatible API remains the external serving contract. The Ratatui TUI may use the same endpoints where suitable, but it should access local operational data through a typed provider/client layer so the UI can also operate in mock and file modes.

M11 should add or expose enough localhost-only control data for the TUI to show:

```text
health
metrics snapshot
adapter list/status
cache namespace summaries
benchmark run status
recent logs/events
streaming chat status
```

Do not expose remote adapter loading, cache deletion, or benchmark execution beyond localhost without an explicit security review and trusted local configuration.

## Control endpoints for TUI

M11 requires additional local control endpoints. They may begin as read-only or stubbed endpoints, but the schema must be stable and shared with `gemma4d-tui`.

```text
GET  /v1/runtime/snapshot
GET  /v1/runtime/events
GET  /v1/config
POST /v1/config/validate
POST /v1/config/apply
GET  /v1/cache/summary
POST /v1/cache/evict
POST /v1/benchmarks/run
GET  /v1/benchmarks/runs/{id}
```

`/v1/runtime/events` should use SSE or WebSocket. The initial implementation can use polling if it is clearly marked and measured, but the API shape should not require the TUI to scrape logs.

## TUI client boundary

`gemma4d-tui` is a client of these endpoints. It must not use the native MLX FFI, model handles, adapter handles, or KV tensors directly in the MVP.
