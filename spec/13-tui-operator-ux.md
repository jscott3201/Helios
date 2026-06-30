# 13 — Ratatui Operator TUI / UX

## Purpose

Add `gemma4d-tui` as the primary local operator experience for configuration, model/runtime inspection, benchmark execution, adapter management, cache inspection, profiling, and eventually chat. The TUI is not a replacement for the local OpenAI-compatible API; it is a first-class client/supervisor that gives the developer a fast feedback loop while building and profiling the 16GB MacBook runtime.

## Research basis

Ratatui is a Rust TUI library for fast, lightweight, rich terminal interfaces, with widgets suitable for dashboards, tables, charts, gauges, sparklines, progress bars, and interactive terminal experiences. Ratatui is immediate-mode and expects the application to own state, input handling, and redraw policy; async I/O should be handled by the application using Tokio/Crossterm event streams where useful. Crossterm provides raw mode and alternate-screen support, which is the right terminal backend for the first macOS-focused implementation.

Sources to re-check before implementation:

- `https://ratatui.rs/`
- `https://github.com/ratatui/ratatui`
- `https://ratatui.rs/templates/component/`
- `https://ratatui.rs/tutorials/counter-async-app/async-event-stream/`
- `https://ratatui.rs/faq/`
- `https://docs.rs/crossterm/latest/crossterm/terminal/index.html`

## Design decision

Implement the TUI as a separate crate/binary:

```text
crates/gemma4d-tui/
```

The TUI must communicate with the runtime through stable Rust client/control abstractions rather than calling the native MLX scheduler directly. This protects the inference engine from UI lifecycle bugs and allows the TUI to work in two modes:

```text
offline mode:
  inspect/edit config, validate manifests, browse benchmark output, launch local tests.

attach mode:
  connect to a running gemma4d daemon/control endpoint for live chat, metrics, cache, adapter, and profiling views.
```

Early milestones should implement offline mode first. Attach mode can initially use an in-process mock provider or local JSONL/metrics files, then move to the real server/control API when M11 lands.

## Workspace additions

```text
crates/
  gemma4d-tui/
    src/
      main.rs
      app.rs
      event.rs
      action.rs
      ui.rs
      pages/
        dashboard.rs
        config.rs
        benchmarks.rs
        logs.rs
        chat.rs
        cache.rs
        adapters.rs
        mtp.rs
        help.rs
      widgets/
        memory_gauge.rs
        metric_sparkline.rs
        status_bar.rs
        keymap.rs
      client/
        mod.rs
        mock.rs
        file_provider.rs
        http_provider.rs
      theme.rs
      keybindings.rs
      snapshot.rs
```

Recommended dependencies:

```toml
ratatui = "0.30"
crossterm = { version = "0.29", features = ["event-stream"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "time"] }
tokio-util = "0.7"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
color-eyre = "0.6"
clap = { version = "4", features = ["derive"] }
```

Pin exact versions during M05 based on the current lockfile and Rust 1.95.0 compatibility.

## Event model

Use an Elm/TEA-inspired architecture with explicit actions and one mutable `AppState`. Do not let pages own long-lived engine handles.

```rust
pub enum AppEvent {
    Input(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),
    Tick,
    Render,
    Backend(BackendEvent),
    Shutdown,
}

pub enum Action {
    Navigate(PageId),
    OpenCommandPalette,
    StartBenchmark(BenchmarkProfile),
    StopBenchmark,
    ValidateConfig(PathBuf),
    SaveConfig,
    LoadAdapter(AdapterId),
    UnloadAdapter(AdapterId),
    SendChatMessage(String),
    RefreshMetrics,
    Quit,
}
```

Recommended loop:

```text
Crossterm input stream + tick stream + backend event stream
  -> tokio::select!
  -> Action mapper
  -> AppState reducer
  -> Ratatui render(frame, &state)
```

Rendering should target 15–30 FPS maximum, and also allow event-driven redraws to avoid unnecessary CPU burn while the model is running.

## Required pages

### Dashboard

Purpose: give the operator an immediate picture of whether the runtime can safely run on a 16GB Mac.

Required panels:

```text
runtime status
model target/drafter state
active adapter
max context/config profile
memory gauges: process RSS, MLX/native bytes when available, KV bytes, adapter bytes
throughput: TTFT, prefill tok/s, decode tok/s
cache: RAM/SSD hit rates and bytes
MTP: enabled, draft block size, acceptance rate
recent errors
```

### Config

Purpose: make `tiny16.toml` safe and discoverable.

Required behavior:

```text
load config
validate config
show derived memory budget
show warnings for risky options
save to explicit path only
show diff before write
```

The TUI must never silently overwrite config files.

### Benchmarks / Profiling

Purpose: run and inspect the benchmark matrix without leaving the terminal UI.

Required behavior:

```text
list benchmark profiles
run smoke/parity/tiny16 benchmarks
stream benchmark progress
show raw output path
show latest report summary
open/copy exact command
```

M05 may use fixture/mock benchmark results. Later milestones must connect this page to real `gemma4d-bench` commands.

### Chat

Purpose: exercise the runtime once greedy inference and the server/control API exist.

Required behavior:

```text
message composer
streaming output pane
adapter selector
MTP toggle/status, disabled unless verified
context/token count
stop generation
copy transcript
export prompt bundle for reproduction
```

The chat page can be present as a disabled placeholder in M05 and become live after M11.

### Cache

Purpose: inspect and operate KV/prefix cache state.

Required behavior:

```text
active KV bytes
RAM prefix cache blocks
SSD cache location/size
cache namespace, including adapter namespace
hits/misses/evictions
flush selected namespace with confirmation
restore failure log
compression mode summary
```

### Adapters

Purpose: manage dynamic LoRA/QLoRA specialists.

Required behavior:

```text
list adapters
validate manifests
load/unload/pin/unpin
show rank/target modules/hash/base compatibility
show memory residency: hot/warm/cold
show adapter-aware cache namespace status
show per-adapter MTP acceptance when available
```

### MTP

Purpose: make speculative decoding observable rather than magical.

Required behavior:

```text
show target/drafter compatibility
show exactness status
show draft block size
show acceptance rate
show rollbacks
show auto-disable reason
```

### Logs / Events

Purpose: expose traces and errors in a terminal-friendly way.

Required behavior:

```text
live log tail
filter by level/component
search current buffer
copy selected event details
```

### Help / Command Palette

Required behavior:

```text
? opens help
: opens command palette
q quits only from non-destructive contexts
Esc closes modal
Ctrl-C triggers graceful shutdown
```

## Keybinding baseline

```text
Tab / Shift-Tab      next/previous page
1..8                 direct page jump
?                    help
:                    command palette
/                    search/filter current page
r                    refresh
b                    run selected benchmark
s                    stop active benchmark/generation
c                    copy current command/report path
e                    export reproduction bundle
q                    quit or close page depending context
Esc                  close modal/cancel selection
Ctrl-C               graceful shutdown
```

All destructive operations need confirmation.

## Control/provider abstraction

Do not bind the TUI directly to HTTP from day one. Define a provider trait so pages can be tested with deterministic fixtures.

```rust
#[async_trait]
pub trait RuntimeProvider: Send + Sync {
    async fn health(&self) -> Result<RuntimeHealth>;
    async fn metrics_snapshot(&self) -> Result<MetricsSnapshot>;
    async fn list_adapters(&self) -> Result<Vec<AdapterSummary>>;
    async fn list_cache_namespaces(&self) -> Result<Vec<CacheNamespaceSummary>>;
    async fn validate_config(&self, path: &Path) -> Result<ConfigValidation>;
    async fn start_benchmark(&self, request: BenchmarkRequest) -> Result<BenchmarkRunId>;
    async fn stop_benchmark(&self, run_id: BenchmarkRunId) -> Result<()>;
    async fn stream_events(&self) -> Result<BoxStream<'static, BackendEvent>>;
}
```

Providers:

```text
MockProvider:
  deterministic fixture state for tests and screenshots.

FileProvider:
  reads configs, JSONL benchmark records, logs, and reports from disk.

HttpProvider:
  talks to gemma4d local daemon/server once M11 exists.
```

## Testing requirements

M05 must include:

```text
unit tests for action reducer
unit tests for keybinding map
snapshot tests for main pages at 80x24 and 120x40
config validation fixture test
mock benchmark lifecycle test
terminal restore test/panic hook review
```

Later milestones must add:

```text
live server attach test
streaming chat smoke test
cache/adapters page integration tests
benchmark-run-from-TUI acceptance gate
```

Use text snapshot testing where practical. Keep snapshot output deterministic by avoiding wall-clock timestamps in page renders unless injected.

## Safety and UX constraints

- The TUI must restore terminal state on panic or normal exit.
- Raw mode and alternate screen must be entered and exited through one lifecycle owner.
- The TUI must not run inside the MLX scheduler thread.
- The TUI must degrade gracefully when no daemon is running.
- Destructive actions require confirmation.
- Remote adapter loading remains disabled unless a trusted local config explicitly allows it.
- The default bind target remains localhost.

## Measurement requirements

For TUI milestones, record:

```text
render frame time p50/p95 for mock dashboard
idle CPU observation
memory overhead of TUI process
benchmark launch command fidelity
operator task timing for core workflows where manual measurement is feasible
```

M05 acceptance should not require model weights. The TUI must be useful before MLX inference is fully optimized, especially for configuration, benchmark orchestration, and evidence review.
