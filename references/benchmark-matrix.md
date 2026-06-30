# Benchmark Matrix

## Prompt categories

| Category | Purpose | Files |
|---|---|---|
| simple_chat | smoke and TTFT | `benchmarks/prompts/simple_chat.jsonl` |
| rust_code | Rust expert and code reasoning | `benchmarks/prompts/rust_code.jsonl` |
| python_code | Python expert routing | `benchmarks/prompts/python_code.jsonl` |
| long_prefix | cache/SSD warm TTFT | `benchmarks/prompts/long_prefix_*.jsonl` |
| tool_shape | JSON/tool-call formatting stability | `benchmarks/prompts/tool_shape.jsonl` |

## Context lengths

```text
1K, 4K, 8K, 16K, 32K, 64K if memory allows
```

## Modes

```text
target_greedy_bf16_kv
target_mtp_bf16_kv
ram_prefix_bf16
ssd_prefix_bf16
prefix_q8
prefix_q4
adapter_rust_no_mtp
adapter_python_no_mtp
```

## Metrics

```text
TTFT
prefill tok/s
decode tok/s
peak RSS
KV active bytes
cache hit/miss
SSD read/write MB
MTP acceptance rate
adapter load latency
error/fallback path
```

## TUI/operator benchmarks

| Milestone | Scenario | Evidence |
|---|---|---|
| M05 | Mock dashboard render 80x24 and 120x40 | snapshot tests + timing if stable |
| M05 | Config validation workflow | test output + TUI usability report |
| M05 | Benchmark command preview/run/stop mock workflow | exact command/output path recorded |
| M11 | TUI HttpProvider attach to local server | integration test or manual evidence |
| M12 | TUI-driven tiny16 profiling walkthrough | report under `benchmarks/out/M12/` |

## TUI modes

```text
tui_fake_server_dashboard
tui_real_server_health_metrics
tui_streaming_chat
tui_adapter_manager_snapshot
tui_cache_inspector_snapshot
tui_config_editor_snapshot
tui_idle_overhead
tui_render_overhead
```

## TUI metrics

```text
TUI idle RSS
TUI idle CPU
render p50/p95
server event lag
streaming token render lag
snapshot test pass/fail
bounded log drops
confirmation-gated destructive action count
```
