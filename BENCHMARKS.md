# Helios Benchmark Ledger

This file tracks benchmark runs and measurement changes that matter for Helios
performance claims. Raw benchmark artifacts stay under `benchmarks/out/` and are
intentionally ignored; this ledger records the stable index of what was run,
which code produced it, and what claims are allowed.

## Tracking Rules

- Record exact commands, output paths, git SHA, model path, and mode.
- Separate command/process overhead from model load, prefill, decode, and memory.
- Mark helper-backed, native-graph, fixture, and server paths explicitly.
- Do not claim native graph performance from helper-backed measurements.
- Keep `benchmarks/out/.../records.jsonl`, `summary.json`, and `report.md` as
  the authority for raw numbers.
- Add a new entry whenever benchmark harness behavior or performance evidence
  changes.

## Runs

| Date | Scope | Status | Git SHA | Mode | Artifacts | Notes |
|---|---|---|---|---|---|---|
| 2026-06-30 | M12 real tiny16 matrix | Passed | `940bdfb` | `target_greedy_mlx_lm_helper_via_c_abi` | `benchmarks/out/M12/real-matrix/{records.jsonl,summary.json,report.md}` | 1K/4K/8K/16K generated 128 tokens; 32K generated one token as a memory probe. |
| 2026-06-30 | P00 performance baseline | Passed | `d5de5db` plus local P00 changes | `target_greedy_mlx_lm_helper_via_c_abi` | `benchmarks/out/P00-performance-baseline/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p00-1782841624`; all 1K/4K/8K/16K cases generated 128 tokens. |
| 2026-06-30 | M12 compatibility rerun after P00 instrumentation | Passed | `d5de5db` plus local P00 changes | `target_greedy_mlx_lm_helper_via_c_abi` | `benchmarks/out/M12/real-matrix/{records.jsonl,summary.json,report.md}` | Existing matrix still passes after richer `generate --json`; 1K/4K/8K/16K generated 128 tokens and 32K generated one token. |
| 2026-06-30 | P01 persistent helper session | Passed | `d5de5db` plus local P00/P01 changes | `target_greedy_mlx_lm_helper_via_c_abi` | `benchmarks/out/P01-persistent-helper-session/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p01-1782843052`; one target load, two warm rounds across 1K/4K/8K/16K; all warm outputs matched M12 cold output. |

## P00 Baseline Snapshot

| Context | Generated | Load ms | Prefill ms | Decode ms | Total ms | Command wall ms | Command overhead ms | Decode tok/s | Decode p50 ms | Decode p95 ms | Peak MLX GB | Peak RSS MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 128/128 | 1912.631 | 2102.335 | 7984.413 | 11999.488 | 12517.384 | 517.896 | 15.906 | 62.706 | 63.725 | 8.065 | 3705.500 |
| 4K | 128/128 | 1832.444 | 9253.118 | 8827.035 | 19912.890 | 20370.634 | 457.744 | 14.388 | 64.212 | 65.247 | 9.480 | 4694.300 |
| 8K | 128/128 | 1779.308 | 18577.923 | 9322.173 | 29679.501 | 30087.687 | 408.186 | 13.623 | 64.186 | 67.041 | 9.833 | 5598.200 |
| 16K | 128/128 | 1825.656 | 40622.532 | 21363.721 | 63812.228 | 64448.723 | 636.495 | 5.945 | 65.744 | 68.958 | 10.512 | 5283.100 |

P00 model identity:

| Field | Value |
|---|---|
| Model path | `artifacts/models/gemma-4-12B-it-4bit` |
| Model revision | `unavailable:GEMMA4D_MODEL_REVISION not set` |
| Config SHA-256 | `fbc1c1cb48ed86ec98482b2d41f5a03d3991aba74b7c29a93d430761e6518a38` |
| Tokenizer SHA-256 | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| Tokenizer config SHA-256 | `fc1384a911d2c9860ac07bc3ceafff20bff26695991744b7dbe5e1e4522bfa57` |
| Safetensors inventory SHA-256 | `a8c71f9c30898c00e3e82d1dd6524882d3ec7c078d477a8004ea642bac561440` |

## M12 Compatibility Rerun Snapshot

| Context | Generated | TTFT ms | Prefill tok/s | Decode tok/s | Peak MLX GB | Peak RSS MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 128/128 | 2065.456 | 495.774 | 15.905 | 8.065 | 5089.600 |
| 4K | 128/128 | 9270.323 | 441.840 | 13.225 | 9.480 | 4907.700 |
| 8K | 128/128 | 18480.872 | 443.269 | 15.601 | 9.833 | 5843.500 |
| 16K | 128/128 | 40427.491 | 405.269 | 13.263 | 10.512 | 5698.100 |
| 32K | 1/1 | 96862.987 | 338.292 | 0.000 | 11.888 | 5632.200 |

## P01 Warm-Session Snapshot

P01 loads the helper-backed target once, reuses the same process for all cases,
and calls `KvCache::reset` before each case. The helper-backed prefill path also
recreates the Python prompt cache for the new prefix.

Load amortization:

| Warm cases | Warm load once ms | Equivalent cold load ms | Load ms saved | Saved % |
|---:|---:|---:|---:|---:|
| 8 | 2009.969 | 14169.072 | 12159.103 | 85.814 |

Cold vs warm comparison:

| Context | Output stable | Cold total ms | Warm case ms | Warm amortized total ms | Delta ms | Cold load ms | Warm amortized load ms | Cold prefill ms | Warm prefill ms | Cold decode ms | Warm decode ms | Warm peak GB | Warm RSS MB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | `true` | 11887.014 | 10378.486 | 10629.732 | -1257.282 | 1836.691 | 251.246 | 2065.456 | 2196.965 | 7984.709 | 8181.520 | 10.512 | 2502.531 |
| 4K | `true` | 20552.826 | 22558.533 | 22809.779 | 2256.953 | 1679.052 | 251.246 | 9270.323 | 9988.277 | 9603.036 | 12570.256 | 10.512 | 2502.531 |
| 8K | `true` | 28416.294 | 41772.115 | 42023.361 | 13607.067 | 1794.682 | 251.246 | 18480.872 | 21330.333 | 8140.638 | 20441.782 | 10.512 | 2502.531 |
| 16K | `true` | 51777.396 | 74837.233 | 75088.479 | 23311.083 | 1774.111 | 251.246 | 40427.491 | 42702.082 | 9575.674 | 32135.150 | 10.512 | 2502.531 |

Warm-session memory growth:

| Round | Context | Peak MLX GB | Growth From First GB | Helper RSS MB | RSS Growth MB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1K | 8.065 | 0.000 | 2502.531 | 0.000 |
| 1 | 4K | 9.480 | 1.416 | 2502.531 | 0.000 |
| 1 | 8K | 9.833 | 1.768 | 2502.531 | 0.000 |
| 1 | 16K | 10.512 | 2.447 | 2502.531 | 0.000 |
| 2 | 1K | 10.512 | 2.447 | 2502.531 | 0.000 |
| 2 | 4K | 10.512 | 2.447 | 2502.531 | 0.000 |
| 2 | 8K | 10.512 | 2.447 | 2502.531 | 0.000 |
| 2 | 16K | 10.512 | 2.447 | 2502.531 | 0.000 |

## Measurement Changes

| Date | Change | Files | Verification |
|---|---|---|---|
| 2026-06-30 | Added P00 measurement fields to `gemma4d-server generate --json`: `model_load_ms`, `prefill_ms`, `total_ms`, `decode_token_latencies_ms`, and explicit nullable MLX active/cache memory fields. Legacy `ttft_ms`, `decode_ms`, `decode_tps`, `peak_memory_gb`, and `peak_rss_mb` remain present. | `crates/gemma4d-server/src/lib.rs` | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- --out-dir benchmarks/out/M12/real-matrix --model-path artifacts/models/gemma-4-12B-it-4bit` |
| 2026-06-30 | Added P00 baseline harness producing JSONL, summary JSON, Markdown report, and blocker report for 1K/4K/8K/16K helper-backed generation. | `crates/gemma4d-bench/examples/p00_performance_baseline.rs` | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example p00_performance_baseline -- --out-dir benchmarks/out/P00-performance-baseline --model-path artifacts/models/gemma-4-12B-it-4bit` |
| 2026-06-30 | Added P01 persistent helper/session benchmark that loads one FFI `Target`, reuses a single process, calls `KvCache::reset` before each warm case, compares generated tokens against M12 cold CLI records, and reports load amortization plus memory growth. | `crates/gemma4d-bench/examples/p01_persistent_helper_session.rs` | `cargo test -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example p01_persistent_helper_session -- --out-dir benchmarks/out/P01-persistent-helper-session --model-path artifacts/models/gemma-4-12B-it-4bit --cold-records benchmarks/out/M12/real-matrix/records.jsonl` |

## Verification Gates

| Date | Command | Status | Notes |
|---|---|---|---|
| 2026-06-30 | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for changed server and benchmark code. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p00_performance_baseline -- --out-dir benchmarks/out/P00-performance-baseline --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Wrote P00 records, summary, report, and blocker report. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- --out-dir benchmarks/out/M12/real-matrix --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Confirms existing M12 matrix still runs after P00 JSON additions. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p01_persistent_helper_session -- --out-dir benchmarks/out/P01-persistent-helper-session --model-path artifacts/models/gemma-4-12B-it-4bit --cold-records benchmarks/out/M12/real-matrix/records.jsonl` | Passed | Wrote warm-session records, summary, report, and blocker report. |
| 2026-06-30 | `make verify` | Passed | Sandboxed attempt failed at localhost bind with `Operation not permitted`; escalated rerun passed. |

## Current Claim Boundaries

- Current real target benchmark claims are helper-backed through the Rust C ABI
  and MLX-LM helper.
- The hand-written native Gemma 4 graph remains opt-in and is not represented by
  M12 or P00 helper-backed throughput numbers.
- `mlx_active_memory_gb` and `mlx_cache_memory_gb` are tracked as nullable P00
  fields until the helper/native boundary exposes those measurements.
