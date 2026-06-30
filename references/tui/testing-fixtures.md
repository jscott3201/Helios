# TUI Testing Fixtures

## Mock provider profiles

### `fake_healthy_idle`

- connected provider,
- model loaded,
- no active generation,
- no adapters hot,
- empty cache.

### `fake_streaming_chat`

- user submits request,
- mock provider emits token deltas every tick,
- final completion summary includes token count and timings.

### `fake_memory_warning`

- RSS near hard memory limit,
- memory guard warning displayed,
- benchmark run disabled.

### `fake_adapter_loaded`

- `rust-expert` hot and pinned,
- `python-expert` cold,
- adapter manifest selected.

### `fake_cache_restore_failed`

- SSD cache block checksum failure,
- cache inspector displays failure and safe eviction option.

## Required key-sequence tests

```text
? -> help overlay -> Esc closes
2 -> chat -> type prompt -> Enter submits -> deltas render -> completion summary
3 -> adapter manager -> select adapter -> load confirmation -> state updates
4 -> cache inspector -> evict -> confirmation required -> cancel preserves cache
6 -> config -> edit field -> validate -> save draft
```
