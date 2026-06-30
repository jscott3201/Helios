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
| 2026-06-30 | P02 real server inference | Passed | `57f8d5f` plus local P02 benchmark changes | `server_openai_http_real_helper_generate_per_request` | `benchmarks/out/P02-real-server-inference/{records.jsonl,summary.json,report.md,blockers.md,curl-fixtures.md}` | Run ID `p02-1782844669`; localhost HTTP server route generated 128 tokens for 1K/4K/8K/16K and compared against P01 warm session. |
| 2026-06-30 | P03 native graph triage | Passed | `88788a5` | `native_graph_vs_helper_cli_triage` | `benchmarks/out/P03-native-graph-triage/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p03-1782845820`; helper/default and `GEMMA4D_USE_NATIVE_GRAPH=1` outputs/logits matched on two tokenizer-controlled prompts plus 1K/4K/8K one-token probes. |

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

## P02 Real-Server Snapshot

P02 uses the localhost OpenAI-compatible HTTP route with `--backend
real-helper`. The current implementation calls the helper-backed `generate`
path per request, so `model_load_ms` is paid on every server request. P01 warm
session remains the comparison point for future persistent-server work.

Server vs P01 warm-session comparison:

| Context | Actual Prompt Tokens | Generated | P02 Wall ms | P02 Load ms | P02 Prefill ms | P02 Decode ms | P02 Total ms | P02 Decode tok/s | P01 Warm Case ms | P01 Warm Amortized ms | Total Delta ms | Wall Delta ms | P02 Peak GB | P02 RSS MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1029 | 128 | 18050.316 | 2217.938 | 3041.773 | 9270.007 | 16131.910 | 13.808 | 10378.486 | 10629.732 | 5753.424 | 7420.584 | 8.079 | 2404.828 |
| 4K | 4101 | 128 | 23110.969 | 1618.500 | 9523.336 | 8507.166 | 21280.233 | 15.046 | 22558.533 | 22809.779 | -1278.300 | 301.190 | 8.623 | 4277.000 |
| 8K | 8197 | 128 | 32386.188 | 1576.719 | 18841.804 | 8483.078 | 30547.733 | 15.089 | 41772.115 | 42023.361 | -11224.382 | -9637.173 | 9.001 | 4902.391 |
| 16K | 16389 | 128 | 55308.857 | 1549.061 | 41558.039 | 8564.273 | 53413.779 | 14.946 | 74837.233 | 75088.479 | -21423.454 | -19779.622 | 9.695 | 4943.609 |

Prometheus snapshot after the P02 run:

| Context | Requests | Model Load s | Prefill Tokens | Decode Tokens | Prefill s | Decode s | Tok/s | Peak MLX Bytes | RSS Bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2.000 | 2.218 | 1029.000 | 128.000 | 3.042 | 9.270 | 13.808 | 8674852864.000 | 2521645056.000 |
| 4K | 4.000 | 3.836 | 5130.000 | 256.000 | 12.565 | 17.777 | 15.046 | 9259312128.000 | 4484759552.000 |
| 8K | 6.000 | 5.413 | 13327.000 | 384.000 | 31.407 | 26.260 | 15.089 | 9664636928.000 | 5140529152.000 |
| 16K | 8.000 | 6.962 | 29716.000 | 512.000 | 72.965 | 34.825 | 14.946 | 10410061824.000 | 5183750144.000 |

## P03 Native-Graph Triage Snapshot

P03 compares the default helper-backed `gemma4d generate` path against
`GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1` for tokenizer-controlled
prompts. It does not switch defaults or claim broad serving readiness.

Claim inventory from run `p03-1782845820`:

| Category | Result |
|---|---|
| Confirmed parity | `hello_smoke`, `hello_reference_prefix`, `repeat_9259_1k`, `repeat_9259_4k`, and `repeat_9259_8k` matched helper tokens and greedy logits within `0.5`. |
| Numerical drift | None recorded. Max logit deltas were `0.000`, `0.000`, `0.125`, `0.000`, and `0.250`. |
| Unsupported ops / runtime failures | None recorded. |
| Memory cliffs | None recorded at the 12 GB threshold; 8K native peak was `10.103 GB`. |
| Measured hotspot | Native prefill dominated every probe. |

Native vs helper probe results:

| Probe | Input Tokens | Generated | Status | Max Logit Delta | Helper Total ms | Native Total ms | Total Delta ms | Helper Prefill ms | Native Prefill ms | Helper Decode ms | Native Decode ms | Helper Peak GB | Native Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hello_smoke` | 1 | 8 | `parity_confirmed` | 0.000 | 3113.716 | 1705.165 | -1408.551 | 671.393 | 937.793 | 432.562 | 758.540 | 6.792 | 6.723 |
| `hello_reference_prefix` | 3 | 1 | `parity_confirmed` | 0.000 | 2094.945 | 798.059 | -1296.886 | 387.048 | 785.355 | 0.001 | 0.001 | 6.716 | 6.705 |
| `repeat_9259_1k` | 1024 | 1 | `parity_confirmed` | 0.125 | 4032.647 | 2227.819 | -1804.828 | 2496.317 | 2214.949 | 0.001 | 0.001 | 8.065 | 7.209 |
| `repeat_9259_4k` | 4096 | 1 | `parity_confirmed` | 0.000 | 10921.167 | 10312.717 | -608.450 | 9331.091 | 10304.003 | 0.001 | 0.001 | 9.480 | 7.947 |
| `repeat_9259_8k` | 8192 | 1 | `parity_confirmed` | 0.250 | 20694.542 | 26664.866 | 5970.324 | 19157.336 | 26651.685 | 0.001 | 0.001 | 9.833 | 10.103 |

## Measurement Changes

| Date | Change | Files | Verification |
|---|---|---|---|
| 2026-06-30 | Added P00 measurement fields to `gemma4d-server generate --json`: `model_load_ms`, `prefill_ms`, `total_ms`, `decode_token_latencies_ms`, and explicit nullable MLX active/cache memory fields. Legacy `ttft_ms`, `decode_ms`, `decode_tps`, `peak_memory_gb`, and `peak_rss_mb` remain present. | `crates/gemma4d-server/src/lib.rs` | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- --out-dir benchmarks/out/M12/real-matrix --model-path artifacts/models/gemma-4-12B-it-4bit` |
| 2026-06-30 | Added P00 baseline harness producing JSONL, summary JSON, Markdown report, and blocker report for 1K/4K/8K/16K helper-backed generation. | `crates/gemma4d-bench/examples/p00_performance_baseline.rs` | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example p00_performance_baseline -- --out-dir benchmarks/out/P00-performance-baseline --model-path artifacts/models/gemma-4-12B-it-4bit` |
| 2026-06-30 | Added P01 persistent helper/session benchmark that loads one FFI `Target`, reuses a single process, calls `KvCache::reset` before each warm case, compares generated tokens against M12 cold CLI records, and reports load amortization plus memory growth. | `crates/gemma4d-bench/examples/p01_persistent_helper_session.rs` | `cargo test -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example p01_persistent_helper_session -- --out-dir benchmarks/out/P01-persistent-helper-session --model-path artifacts/models/gemma-4-12B-it-4bit --cold-records benchmarks/out/M12/real-matrix/records.jsonl` |
| 2026-06-30 | Added opt-in real-helper server mode for `/v1/chat/completions`, CLI flags `--backend real-helper --model-path`, real response `gemma4d_metrics`, and Prometheus counters for helper load, prefill, decode, token, RSS, and peak MLX memory. Stub remains the default backend. | `crates/gemma4d-server/src/http.rs`, `crates/gemma4d-server/src/lib.rs` | `cargo test -p gemma4d-server --all-targets`; curl non-streaming, streaming, and metrics smoke against `gemma4d serve --backend real-helper`. |
| 2026-06-30 | Added P02 localhost server benchmark harness that runs an actual HTTP listener, records server response metrics and Prometheus snapshots, compares against P01 warm-session records, and writes curl fixture commands. | `crates/gemma4d-bench/examples/p02_real_server_inference.rs`, `codex/goals/P02-real-server-inference-path.goal.md` | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example p02_real_server_inference -- --out-dir benchmarks/out/P02-real-server-inference --model-path artifacts/models/gemma-4-12B-it-4bit --p01-summary benchmarks/out/P01-persistent-helper-session/summary.json` |
| 2026-06-30 | Added diagnostic `generated_logits` to `gemma4d-server generate --json` so native/helper triage can compare greedy logits alongside generated token IDs. | `crates/gemma4d-server/src/lib.rs` | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets`; P03 triage run. |
| 2026-06-30 | Added P03 native graph triage harness and goal contract. The harness runs paired helper/default and `GEMMA4D_USE_NATIVE_GRAPH=1` CLI probes, writes records/report/blockers, and inventories parity, drift, unsupported ops, memory cliffs, and hotspots. | `crates/gemma4d-bench/examples/p03_native_graph_triage.rs`, `codex/goals/P03-native-graph-triage.goal.md` | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example p03_native_graph_triage -- --out-dir benchmarks/out/P03-native-graph-triage --model-path artifacts/models/gemma-4-12B-it-4bit` |

## Verification Gates

| Date | Command | Status | Notes |
|---|---|---|---|
| 2026-06-30 | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for changed server and benchmark code. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p00_performance_baseline -- --out-dir benchmarks/out/P00-performance-baseline --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Wrote P00 records, summary, report, and blocker report. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- --out-dir benchmarks/out/M12/real-matrix --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Confirms existing M12 matrix still runs after P00 JSON additions. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p01_persistent_helper_session -- --out-dir benchmarks/out/P01-persistent-helper-session --model-path artifacts/models/gemma-4-12B-it-4bit --cold-records benchmarks/out/M12/real-matrix/records.jsonl` | Passed | Wrote warm-session records, summary, report, and blocker report. |
| 2026-06-30 | `make verify` | Passed | Sandboxed attempt failed at localhost bind with `Operation not permitted`; escalated rerun passed. |
| 2026-06-30 | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for P02 server and benchmark changes. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p02_real_server_inference -- --out-dir benchmarks/out/P02-real-server-inference --model-path artifacts/models/gemma-4-12B-it-4bit --p01-summary benchmarks/out/P01-persistent-helper-session/summary.json` | Passed | Wrote P02 records, summary, report, blocker report, and curl fixtures. |
| 2026-06-30 | `curl -sS -i -X POST http://127.0.0.1:18082/v1/chat/completions ... "max_tokens":8` | Passed | Non-streaming smoke returned HTTP 200, `object:"chat.completion"`, `gemma4d_metrics`, and usage `prompt_tokens=11`, `completion_tokens=8`. Required escalated local networking after sandboxed curl could not connect. |
| 2026-06-30 | `curl -sS -i -N -X POST http://127.0.0.1:18082/v1/chat/completions ... "stream":true` | Passed | Streaming smoke returned HTTP 200 `text/event-stream`, content chunk, stop chunk, and `data: [DONE]`. Required escalated local networking after sandboxed curl could not connect. |
| 2026-06-30 | `curl -sS http://127.0.0.1:18082/metrics` | Passed | Metrics after two real smoke generations showed `gemma4d_model_load_seconds 3.090923`, `gemma4d_prefill_tokens_total 22`, `gemma4d_decode_tokens_total 16`, and non-zero RSS/peak MLX memory counters. |
| 2026-06-30 | `make verify` | Passed | Sandboxed rerun reached tests but failed at localhost bind with `Operation not permitted`; escalated rerun passed. |
| 2026-06-30 | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for P03 diagnostic JSON and benchmark harness. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p03_native_graph_triage -- --out-dir benchmarks/out/P03-native-graph-triage --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Wrote P03 records, summary, report, and blocker report; no blockers recorded. |
| 2026-06-30 | `make verify` | Passed | Sandboxed run failed only at localhost bind with `Operation not permitted`; escalated rerun passed. |

## Current Claim Boundaries

- Current real target benchmark claims are helper-backed through the Rust C ABI
  and MLX-LM helper.
- The hand-written native Gemma 4 graph remains opt-in and is not represented by
  M12 or P00 helper-backed throughput numbers.
- `mlx_active_memory_gb` and `mlx_cache_memory_gb` are tracked as nullable P00
  fields until the helper/native boundary exposes those measurements.
- P02 real-helper server inference is opt-in. The default server backend remains
  the M11 stub, and P02 does not apply adapters or MTP on the real server path.
- P02 server benchmark measurements include HTTP route overhead but still pay
  model load per request. They should not be interpreted as persistent
  server-session latency until a later goal keeps a loaded target inside the
  server runtime.
- P03 confirms native graph parity only for the tokenizer-controlled probes in
  the P03 report. It does not justify switching defaults, server use, adapter
  use, MTP use, or unmeasured prompt/context shapes.
- P03 native RSS is not yet measured; native memory claims rely on MLX peak
  memory until native RSS reporting is added.
