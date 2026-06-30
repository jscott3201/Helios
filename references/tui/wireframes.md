# Ratatui TUI Wireframes

These are layout contracts for implementation and snapshot tests, not pixel-perfect design mandates.

## 80x24 Dashboard

```text
┌Gemma4D tiny16─────────────────────────────┐
│ Server: connected  Model: 12B  MTP: off   │
│ RSS: 8.2GB / 12GB   Queue: 0   Active: no │
├Metrics────────────────────────────────────┤
│ TTFT p50: --   Decode: --   Prefill: --   │
│ KV active: --  RAM cache: -- SSD: --      │
├Adapters───────────────────────────────────┤
│ hot: none  default: base                  │
├Warnings───────────────────────────────────┤
│ No model loaded / or latest warning       │
└1 Dash 2 Chat 3 Adapters 4 Cache ? Help q Quit┘
```

## 120x40 Chat Workbench

```text
┌Transcript──────────────────────────────┬Run──────────────────────┐
│ user: ...                              │ model: gemma4-12b       │
│ assistant: streaming ...               │ adapter: rust-expert    │
│                                        │ mtp: off/on             │
│                                        │ context: 4,096 / 32,768 │
├Input───────────────────────────────────┤ tokens/sec: ...         │
│ >                                      │ Ctrl-C cancel           │
└────────────────────────────────────────┴──────────────────────────┘
```

## 120x40 Adapter Manager

```text
┌Adapters───────────────────────────────┬Manifest───────────────────┐
│ rust-expert     hot pinned            │ base hash: ok             │
│ python-expert   cold                  │ tokenizer: ok             │
│ sql-expert      warm                  │ rank: 16                  │
├Actions────────────────────────────────┤ modules: q/k/v/o/mlp      │
│ Enter load/unload  p pin  u unpin     │ MTP: unverified           │
└───────────────────────────────────────┴────────────────────────────┘
```

## Snapshot sizes

Every MVP screen must have snapshots at:

```text
80x24
120x40
160x50
```

Use monochrome/theme-independent snapshots when color creates unstable diffs.
