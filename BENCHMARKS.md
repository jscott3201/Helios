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
| 2026-06-30 | P04 incremental native KV decode | Passed | `4f265cc` | `incremental_native_kv_vs_helper_cli` | `benchmarks/out/P04-incremental-native-kv/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p04-1782847670`; helper/default and native generated tokens matched on small prompts plus 1K/4K/8K probes; steady decode p50/p95 stayed flat across 8x context growth. |
| 2026-06-30 | P05 true native MTP verification | Passed | `57ac3a6` | `native_target_and_native_mtp_ffi` | `benchmarks/out/P05-native-mtp/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p05-1782849629`; real native target+assistant FFI loop matched non-MTP native output for block sizes 1 and 2, then auto-disabled because acceptance was 0.000. |
| 2026-06-30 | P06 real RAM prefix cache | Passed | `e5e61ad` | `native_ram_prefix_snapshot_ffi` | `benchmarks/out/P06-real-ram-prefix-cache/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p06-1782851001`; native RAM snapshot restore matched fresh-prefill logits and continued decode at 4K/8K/16K, with wrong model/adapter/cache-mode namespace rejection. |
| 2026-06-30 | P07 real SSD prefix cache | Passed | `9a4cd13` | `native_ssd_prefix_snapshot_payload` | `benchmarks/out/P07-real-ssd-prefix-cache/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p07-1782853459`; real SSD safetensors payload restore improved warm TTFT at 4K/8K/16K, rejected namespace/corruption/mid-decode fetches, and keeps SSD disabled by default pending broader variance data. |
| 2026-06-30 | P08 real KV compression gates | Passed | `5993b86` | `native_kv_prefix_payload_compression` | `benchmarks/out/P08-kv-compression/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p08-1782855932`; q8 full-attention payload compression passed continued-decode quality gates at 4K/8K/16K, q4 reduced payload bytes but failed greedy agreement, and compressed active decode remains disabled. |
| 2026-06-30 | P09 real LoRA adapter hot path | Passed | `8723d50` | `native_lora_adapter_hot_path` | `benchmarks/out/P09-real-lora-adapter/{records.jsonl,summary.json,report.md,blockers.md}` | Run ID `p09-1782857770747`; trusted local rank-16 q_proj/v_proj LoRA fixture loaded into real native inference, changed greedy-logit output, rejected wrong manifests, isolated adapter KV namespace, measured load/hotswap/residency, and disabled MTP while active. |
| 2026-06-30 | P10 TUI live optimization console | Passed | `4ee1ccd` plus local P10 harness changes | `localhost_http_server_tui_provider_stub_backend` | `benchmarks/out/P10-tui-live-console/{tui-report.md,metrics.json,snapshots/}` | Command `cargo run -p gemma4d-bench --example p10_tui_live_console -- --out-dir benchmarks/out/P10-tui-live-console`; 18 snapshots, render p95 `1731 us` below `20000 us`, server health `ok`, latest benchmark report surfaced from the provider. |
| 2026-06-30 | P11 model revision and manifest pinning | Passed | final SHA recorded in generated manifest | `manifest_capture_local_artifact_identity` | `benchmarks/out/P11-manifest-pinning/{manifest.json,report.md}` | Command `cargo run -p gemma4d-bench -- manifest --out-dir benchmarks/out/P11-manifest-pinning`; target and drafter revisions are explicitly pinned in `tiny16.toml` to local artifact SHA-256s because local revision metadata is unavailable. |
| 2026-06-30 | XR00 real-context workload corpus | Passed | final SHA recorded in generated summary | `real_context_corpus_tokenizer_count_only` | `benchmarks/workloads/real-contexts/{workloads.jsonl,prompts/*.txt}` and `benchmarks/out/XR00-real-workload-corpus/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Command `cargo run -p gemma4d-bench -- workload-corpus --model-path artifacts/models/gemma-4-12B-it-4bit --workload-dir benchmarks/workloads/real-contexts --out-dir benchmarks/out/XR00-real-workload-corpus --python /opt/homebrew/opt/mlx-lm/libexec/bin/python --seed 20260630`; no model execution or runtime optimization. |
| 2026-06-30 | XR01 real-context A/B harness | Passed | final SHA recorded in generated summary | `real_context_ab_harness_dry_run_plus_helper_smoke` | `benchmarks/out/XR01-real-context-ab-harness/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Command `cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode both --out-dir benchmarks/out/XR01-real-context-ab-harness --max-workloads 1 --max-new-tokens 2`; writes dry-run and real helper smoke records for the XR00 corpus schema, no runtime optimization. |
| 2026-06-30 | XR02 native vs helper real-context A/B | Blocked with evidence | `d60664b` plus local XR02 harness changes | `native_incremental_vs_helper_real_contexts` | `benchmarks/out/XR02-native-helper-real-context-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Command `cargo run -p gemma4d-bench --example xr02_native_helper_real_context_ab -- --trials 2 --max-new-tokens 8`; 5 real XR00 workloads, 2 variants, 2 trials, 20 records. Native is blocked by chat/tool token mismatches and a 16K tiny16 memory cliff; code-review is opt-in only. |
| 2026-06-30 | XR03 MTP real-context diagnosis | Blocked with evidence | `16efd5d` plus local XR03 trace changes | `native_mtp_real_context_trace` | `benchmarks/out/XR03-mtp-real-context-diagnosis/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 4`; 5 real XR00 workloads x block sizes 1/2, nonzero acceptance observed, but `benchmark_qa_4k_001` failed byte-identical exactness for both block sizes. |
| 2026-07-01 | XR04 MTP repair and A/B evidence | Accept candidate | `50fe4e2` plus local XR04 verifier repair | `native_mtp_incremental_verify_trace` | `benchmarks/out/XR04-mtp-repair-and-autotune/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` plus `xr03-repro/` and `exactness-smoke/` subruns | Reproduced the XR03 blocker first, then repaired live MTP verify to stage against cloned incremental KV. The 32-token root run stayed byte-identical for 10/10 records with acceptance `162/370 = 0.438`; MTP remains opt-in because generation speedups are workload/block dependent. |
| 2026-07-01 | XR05 prefill and MLX eval scheduling A/B | Reject candidate | `5b145fc` plus local candidate-wide decision-gate fix | `prefill_eval_scheduling_real_context_ab` | `benchmarks/out/XR05-prefill-and-eval-scheduling-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr05-1782873617-153379000`; command `GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR05-prefill-and-eval-scheduling-ab`; 72/72 records passed runtime with no blockers, but no candidate satisfied the candidate-wide no-correctness-regression gate. |
| 2026-07-01 | XR06 native decode tail-latency A/B | Accept candidate | `92b0757` | `native_decode_tail_latency_real_context_ab` | `benchmarks/out/XR06-native-decode-tail-latency-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr06-1782877235-943162000`; command `GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR06-native-decode-tail-latency-ab`; 60/60 records passed with no blockers. Native decode eval scheduling remains opt-in; accepted comparisons were workload-local and several tail hypotheses failed. |
| 2026-07-01 | XR07 prefix cache real reuse A/B | Blocked with evidence | `6e4280b` | `native_ram_prefix_cache_real_reuse_ab` | `benchmarks/out/XR07-prefix-cache-real-reuse-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr07-1782880867-63480000`; command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab -- --out-dir benchmarks/out/XR07-prefix-cache-real-reuse-ab --trials 2 --suffix-tokens 4 --suffix-edit-tokens 2 --continued-decode-tokens 4`; namespace isolation passed, but restored continuation/continued decode parity failed and tiny16 memory gates failed at 8K/16K. Default policy is `do_not_enable_ram_prefix_cache_by_default_for_tiny16`. |
| 2026-07-01 | XR08 SSD cache policy and variance A/B | Keep experimental | `0e4b0cd` | `native_ssd_cache_policy_variance` | `benchmarks/out/XR08-ssd-cache-policy-variance/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr08-1782883921-278286000`; command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr08_ssd_cache_policy_variance -- --out-dir benchmarks/out/XR08-ssd-cache-policy-variance`; 12/12 restore records passed correctness and rejection gates. 8K BF16/q8 profiles passed TTFT, variance, and memory gates; 16K BF16/q8 profiles were rejected for the 14 GB tiny16 memory gate. Policy remains opt-in, profile-gated, and experimental. |

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

## P04 Incremental Native-KV Snapshot

P04 keeps the native graph opt-in behind `GEMMA4D_USE_NATIVE_GRAPH=1` and
preserves the helper-backed path as the default fallback. The benchmark records
raw decode samples and computes the growth claim from steady-state samples after
discarding the first four decode calls for MLX/JIT/cache warmup.

Claim inventory from run `p04-1782847670`:

| Category | Result |
|---|---|
| Generated-token parity | `hello_smoke`, `hello_reference_prefix`, `repeat_9259_1k`, `repeat_9259_4k`, and `repeat_9259_8k` matched helper generated token IDs. |
| Decode growth | Native steady p50 ratio was `0.957` and steady p95 ratio was `0.959` from 1K to 8K context, versus `8.000x` context growth. |
| KV memory | Native active KV was `336.234 MiB` at 1K, `384.234 MiB` at 4K, and `448.234 MiB` at 8K. |
| Peak MLX memory | Native peak MLX memory was `7.321 GB` at 1K, `9.212 GB` at 4K, and `12.763 GB` at 8K, below the P04 14 GB tiny16 cliff. |
| Numerical drift | Long-context token parity held while max greedy-logit deltas were diagnostic: `2.375`, `1.125`, and `1.000` for 1K/4K/8K. |
| Runtime blockers | None recorded. |

Native context probe results:

| Probe | Input Tokens | Generated | Status | Max Logit Delta | Native Active KV MiB | Native Prefill ms | Native Decode ms | Native Steady p50 ms | Native Steady p95 ms | Native Raw p95 ms | Native Peak GB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `repeat_9259_1k` | 1024 | 16 | `parity_with_logit_drift` | 2.375 | 336.234 | 3433.292 | 2203.483 | 89.639 | 92.488 | 92.488 | 7.321 |
| `repeat_9259_4k` | 4096 | 16 | `parity_with_logit_drift` | 1.125 | 384.234 | 10929.037 | 2179.249 | 84.154 | 88.571 | 88.571 | 9.212 |
| `repeat_9259_8k` | 8192 | 16 | `parity_with_logit_drift` | 1.000 | 448.234 | 27663.036 | 12515.177 | 85.814 | 88.730 | 1202.597 | 12.763 |

## P05 Native MTP Snapshot

P05 drives the real native FFI path directly: native target load, native MTP
assistant load, `gemma4_mtp_draft_block`, and `gemma4_verify_tokens`. The
benchmark reconstructs emitted tokens from verifier committed-token metadata and
falls back to native `decode_one` when acceptance gates auto-disable MTP.

Claim inventory from the `57ac3a6` run:

| Category | Result |
|---|---|
| Exactness | `hello_smoke` and `hello_reference_prefix` matched the non-MTP native baseline for block sizes `1` and `2`. |
| Acceptance | All four cases had acceptance rate `0.000`; each run attempted one verify pass and rolled back once. |
| Auto-disable | All four cases auto-disabled because acceptance `0.000` fell below the `0.350` threshold. |
| Default recommendation | `keep_disabled_by_default`. |
| Peak MLX memory | MTP peak was `6.946 GB` to `6.957 GB`, below the 14 GB P05 threshold. |

Native MTP probe results:

| Probe | Block | Exact | Attempted | Accepted | Rate | Accepted/Verify | Verify Passes | Rollbacks | Auto Disabled | Baseline tok/s | MTP tok/s | MTP Peak GB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `hello_smoke` | 1 | `true` | 1 | 0 | 0.000 | 0.000 | 1 | 1 | `true` | 4.880 | 4.561 | 6.946 |
| `hello_smoke` | 2 | `true` | 2 | 0 | 0.000 | 0.000 | 1 | 1 | `true` | 4.880 | 4.627 | 6.950 |
| `hello_reference_prefix` | 1 | `true` | 1 | 0 | 0.000 | 0.000 | 1 | 1 | `true` | 4.978 | 4.306 | 6.952 |
| `hello_reference_prefix` | 2 | `true` | 2 | 0 | 0.000 | 0.000 | 1 | 1 | `true` | 4.978 | 4.235 | 6.957 |

## P06 Real RAM Prefix Cache Snapshot

P06 uses the real native FFI path to export/import in-memory KV snapshots. The
namespace gate is still handled by `gemma4d-kv`; the native snapshot is imported
only after RAM prefix restore succeeds for the expected namespace.

Claim inventory from the `e5e61ad` run:

| Category | Result |
|---|---|
| Exactness | 4K, 8K, and 16K restored-prefix last-step greedy token/logit matched fresh prefill; one continued `decode_one` after restore also matched the cold-cache continuation. |
| Warm TTFT | Warm restore plus cached last-step retrieval was `0.074 ms`, `0.077 ms`, and `0.080 ms` for 4K/8K/16K. |
| Namespace safety | Wrong model, wrong adapter, and wrong cache mode rejected before native snapshot import for every measured context. |
| Cache accounting | Each context recorded one hit, one same-namespace miss, three restore failures, and zero evictions. |
| Runtime blockers | None recorded. |

Native RAM prefix-cache probe results:

| Context | Cold TTFT ms | Warm TTFT ms | Speedup | Active KV MiB | Export ms | Hit/Miss/Fail/Evict |
|---:|---:|---:|---:|---:|---:|---|
| 4K | 10502.690 | 0.074 | 141450.37x | 384.000 | 0.020 | 1/1/3/0 |
| 8K | 26726.993 | 0.077 | 345609.15x | 448.000 | 0.011 | 1/1/3/0 |
| 16K | 95772.166 | 0.080 | 1203424.92x | 576.000 | 0.024 | 1/1/3/0 |

## P07 Real SSD Prefix Cache Snapshot

P07 persists the real native KV snapshot payload to SSD in safetensors format.
`gemma4d-kv` still owns namespace and cache-mode admission; the native payload is
checksummed and imported only after before-prefill SSD metadata restore succeeds.
Mid-decode SSD restore is rejected before payload read/import.

Claim inventory from the `9a4cd13` run:

| Category | Result |
|---|---|
| Exactness | 4K, 8K, and 16K restored-prefix last-step greedy token/logit matched fresh prefill; one continued `decode_one` after restore also matched the cold-cache continuation. |
| Warm TTFT | Warm SSD restore was faster than cold prefill at every measured context: `3.615x` at 4K, `7.835x` at 8K, and `18.174x` at 16K. |
| Payload format | Each run wrote SSD metadata plus a real safetensors payload with checksum, cache mode, namespace hash, KV layout, shape metadata, and per-layer attention metadata. |
| Rejection safety | Wrong model, wrong adapter, wrong cache mode, corrupted payload, and mid-decode restore were rejected for every measured context. |
| Cache accounting | Each context recorded metadata bytes, payload bytes, restore latency metrics, and zero mid-decode SSD fetches. |
| Default recommendation | `keep_ssd_disabled_by_default_until_more_variance_data`. |
| Runtime blockers | None recorded. |

Native SSD prefix-cache probe results:

| Context | Cold TTFT ms | Warm SSD TTFT ms | Speedup | Payload MiB | Metadata Read/Write bytes | Payload Read/Write bytes | Mid-Decode Fetches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 10567.721 | 2923.475 | 3.615x | 424.045 | 52735/52735 | 444643887/444643887 | 0 |
| 8K | 28582.644 | 3647.974 | 7.835x | 528.065 | 52735/52735 | 553716282/553716282 | 0 |
| 16K | 92350.582 | 5081.531 | 18.174x | 736.104 | 53070/53070 | 771861096/771861096 | 0 |

## P08 Real KV Compression Snapshot

P08 measures compression on real native KV prefix payloads rather than fixture
logits. The native compressed snapshot writer applies MLX affine q8/q4 only to
global/full-attention KV tensors; sliding-window KV tensors and hidden state stay
BF16. Payloads are decompressed to BF16 before import, so active compressed
decode remains disabled and active KV memory is unchanged.

Claim inventory from the `5993b86` run:

| Category | Result |
|---|---|
| BF16 exactness | BF16 safetensors payload restore and one continued `decode_one` matched the cold BF16 continuation at 4K, 8K, and 16K. |
| q8 quality | q8 passed continued-decode greedy agreement at all measured contexts with greedy-logit delta `0.250000`. |
| q4 quality | q4 reduced payload size at all measured contexts but failed continued-decode greedy agreement at 4K, 8K, and 16K; it must stay disabled pending better quality evidence. |
| Payload memory | q8 payload reduction was `7.541%`, `12.116%`, and `17.386%` at 4K/8K/16K. q4 payload reduction was `11.314%`, `18.175%`, and `26.080%`. |
| Active memory | Active KV reduction was `0.000%` for BF16/q8/q4 because compressed SSD payloads restore into BF16 active decode state. |
| Planar/Iso | Planar/Iso remains feature-disabled by default and has no reportable P08 evidence. |
| Default recommendation | `keep_compressed_active_decode_disabled`. |
| Runtime blockers | None recorded. |

Native KV compression probe results:

| Context | Mode | Gate | Greedy Agree | Logit Delta | Payload MiB | Payload Reduction | Warm Restore ms | Decode ms | Active KV Reduction |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 4K | `bf16` | `true` | `true` | 0.000000 | 424.045 | 0.000% | 5.156 | 234.680 | 0.000% |
| 4K | `mlx_affine_q8` | `true` | `true` | 0.250000 | 392.068 | 7.541% | 1.353 | 128.176 | 0.000% |
| 4K | `mlx_affine_q4` | `false` | `false` | 0.250000 | 376.067 | 11.314% | 1.439 | 122.283 | 0.000% |
| 8K | `bf16` | `true` | `true` | 0.000000 | 528.065 | 0.000% | 4.236 | 478.155 | 0.000% |
| 8K | `mlx_affine_q8` | `true` | `true` | 0.250000 | 464.087 | 12.116% | 2.135 | 162.893 | 0.000% |
| 8K | `mlx_affine_q4` | `false` | `false` | 1.500000 | 432.087 | 18.175% | 1.930 | 207.608 | 0.000% |
| 16K | `bf16` | `true` | `true` | 0.000000 | 736.104 | 0.000% | 3.543 | 8354.318 | 0.000% |
| 16K | `mlx_affine_q8` | `true` | `true` | 0.250000 | 608.126 | 17.386% | 3.270 | 360.565 | 0.000% |
| 16K | `mlx_affine_q4` | `false` | `false` | 1.937500 | 544.126 | 26.080% | 6.373 | 178.773 | 0.000% |

## P09 Real LoRA Adapter Snapshot

P09 moves adapters from registry/control-plane fixtures into the real native
inference path for one trusted local rank-16 PEFT LoRA adapter fixture. The
fixture uses real Gemma 4 layer-0 `q_proj` and `v_proj` shapes and is loaded
through the native C ABI after registry import/manifest validation.

Claim inventory from the `8723d50` run:

| Category | Result |
|---|---|
| Adapter output | Active adapter output differed from base by greedy-logit delta `0.250000` on the 128-token native prefill. Greedy token IDs stayed the same for this prompt. |
| Manifest rejection | Wrong base model, base weight hash, tokenizer hash, and chat-template hash were rejected before native load. |
| KV namespace | Adapter identity and adapter weight hash changed namespace hash and block ID; wrong-adapter RAM prefix restore was rejected. |
| Residency | Native adapter loaded `2` LoRA module pairs with `884736` resident bytes and `40566 us` native load latency. |
| Hotswap | Base-to-adapter and adapter-to-base activation calls were both measured at `1 us`; clearing restored base output for the deterministic prompt. |
| MTP default | Native MTP drafter load/verify are disabled while the standard adapter is active. |
| Runtime blockers | None recorded. |

Native adapter generation results:

| Run | Context | Decode | Prefill ms | Decode ms | Total ms | Prefill Token | Prefill Logit | Generated Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `base` | 128 | 2 | 1118.985 | 158.188 | 1277.173 | 236772 | 18.625000 | `236772,236772,236772` |
| `adapter` | 128 | 2 | 544.963 | 156.611 | 701.574 | 236772 | 18.375000 | `236772,236772,236772` |
| `base_after_clear` | 128 | 2 | 307.862 | 150.023 | 457.885 | 236772 | 18.625000 | `236772,236772,236772` |

## P10 TUI Live Console Snapshot

P10 drives the Ratatui console through the HTTP provider against a spawned
localhost `gemma4d-server` instance. The TUI remains provider-only; the
benchmark harness owns server startup and shutdown.

| Field | Value |
|---|---|
| Command | `cargo run -p gemma4d-bench --example p10_tui_live_console -- --out-dir benchmarks/out/P10-tui-live-console` |
| Report | `benchmarks/out/P10-tui-live-console/tui-report.md` |
| Metrics JSON | `benchmarks/out/P10-tui-live-console/metrics.json` |
| Snapshot count | `18` |
| Render p50 / p95 / threshold | `1373 us` / `1731 us` / `20000 us` |
| Server health | `ok`, `model_loaded=true` |
| Live timing | load `0.000 ms`, prefill `0.120 ms`, TTFT `3.000 ms`, decode `0.180 ms` |
| Throughput | `1000.000 tok/s` over prefill `12` and decode `18` tokens |
| Cache / MTP | cache `stub`, active KV `0`, MTP `disabled` with adapter gate shown |
| Adapter residency | `1` loaded adapter, `2551` resident bytes |

## P11 Manifest Pinning Snapshot

P11 records reproducible artifact identity for the local target and drafter
model directories. The downloaded local artifacts do not contain a pinned
upstream revision, so `references/configs/tiny16.toml` pins explicit local
artifact SHA-256 values instead.

| Field | Value |
|---|---|
| Command | `cargo run -p gemma4d-bench -- manifest --out-dir benchmarks/out/P11-manifest-pinning` |
| Manifest | `benchmarks/out/P11-manifest-pinning/manifest.json` |
| Report | `benchmarks/out/P11-manifest-pinning/report.md` |
| Target local artifact SHA-256 | `d8b821776d41a61dad4f23f9b85cc8c6b09df2be04e5e4583f73c48739d8535c` |
| Target safetensors inventory SHA-256 | `4af9af81c81dcba1edb5290573e58efc28f71c887ab25a871d3917f4240459af` |
| Drafter local artifact SHA-256 | `6b31aa79ef7fce128572671b3890b55477694b52e24c75f48168f34770f85f2b` |
| Drafter safetensors inventory SHA-256 | `7a5d3a9eabd8ec983c4ef5139badf2da187a455133446be21b3c3dc0006b70bd` |
| Versions | Rust `1.95.0`, MLX `0.31.2`, mlx-lm `0.31.3` |

## XR00 Real-Context Corpus Snapshot

XR00 creates deterministic repo-local prompt workloads for later A/B goals. It
only generates prompts and token metadata; it does not run model inference.

| Field | Value |
|---|---|
| Command | `cargo run -p gemma4d-bench -- workload-corpus --model-path artifacts/models/gemma-4-12B-it-4bit --workload-dir benchmarks/workloads/real-contexts --out-dir benchmarks/out/XR00-real-workload-corpus --python /opt/homebrew/opt/mlx-lm/libexec/bin/python --seed 20260630` |
| Workload manifest | `benchmarks/workloads/real-contexts/workloads.jsonl` |
| Evidence | `benchmarks/out/XR00-real-workload-corpus/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Workloads | `13` |
| Families | `adapter_expert`, `benchmark_qa`, `chat_short`, `code_review_rust`, `long_repo_pack`, `mtp_candidate`, `prefix_reuse_edit`, `tool_json` |
| Target context tokens | `1024`, `4096`, `8192`, `16384`, `24576` |
| Actual context tokens | tokenizer-measured with `mlx_lm.utils.load_tokenizer:TokenizerWrapper`; 12 workloads match target exactly and `benchmark_qa_4k_001` measures `4095` against the `4096` target |
| Seed base | `20260630`; per-record seeds `20260630` through `20260642` |
| Decision | `accept_candidate` |
| Blockers | none recorded |

## XR01 Real-Context A/B Harness Snapshot

XR01 adds the reusable harness for running explicit baseline/candidate variants
against the XR00 corpus. The final evidence accepts the harness schema and
smoke command paths only; it is not a performance win claim.

| Field | Value |
|---|---|
| Command | `cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode both --out-dir benchmarks/out/XR01-real-context-ab-harness --max-workloads 1 --max-new-tokens 2` |
| CI/offline dry-run command | `cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode dry-run --out-dir benchmarks/out/XR01-real-context-ab-harness-dry-run --max-workloads 1 --max-new-tokens 2` |
| Workload manifest | `benchmarks/workloads/real-contexts/workloads.jsonl` |
| Evidence | `benchmarks/out/XR01-real-context-ab-harness/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Selected smoke workload | `chat_short_1k_001` |
| Variants | `baseline` and `candidate`, both explicit `helper` backend configs with cache/MTP/adapter disabled |
| Records | `4`: two dry-run records and two real model smoke records |
| Required fields | p50/p95/p99 decode latency, prefill, total, peak memory, active KV bytes, output token IDs, and correctness gate status are present in every record |
| Correctness | candidate output token IDs match baseline output token IDs for dry-run and real smoke records |
| Model artifact | `artifacts/models/gemma-4-12B-it-4bit`, local artifact SHA-256 recorded in `summary.json` |
| Decision | `accept_candidate` |
| Blockers | none recorded |

## XR02 Native vs Helper Real-Context A/B Snapshot

XR02 reuses the XR01 harness shape against real XR00 prompt files and compares
the helper/default baseline with the opt-in native incremental path
(`GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1`). It does not optimize
runtime code or switch defaults.

| Field | Value |
|---|---|
| Command | `cargo run -p gemma4d-bench --example xr02_native_helper_real_context_ab -- --trials 2 --max-new-tokens 8` |
| Workload manifest | `benchmarks/workloads/real-contexts/workloads.jsonl` |
| Evidence | `benchmarks/out/XR02-native-helper-real-context-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Variants | `baseline=helper`, `candidate=native` with `GEMMA4D_REQUIRE_MLX=1,GEMMA4D_USE_NATIVE_GRAPH=1`; cache/MTP/adapter disabled |
| Records | `20` real records: 5 workloads x 2 variants x 2 trials |
| Requested max new tokens | `8` |
| Correctness | Native token IDs mismatched helper on `chat_short_1k_001` and `tool_json_1k_001`; token IDs matched on `code_review_rust_4k_001`, `code_review_rust_8k_001`, and `benchmark_qa_16k_001` |
| Decision | `blocked_with_evidence` |
| Blockers | 4 failed candidate records: both trials of `chat_short_1k_001` and `tool_json_1k_001` |

Workload selection:

| Workload | Family | Target tokens | Actual tokens | Workload max new tokens | Seed |
|---|---|---:|---:|---:|---:|
| `chat_short_1k_001` | `chat_short` | 1024 | 1024 | 128 | 20260630 |
| `code_review_rust_4k_001` | `code_review_rust` | 4096 | 4096 | 192 | 20260631 |
| `code_review_rust_8k_001` | `code_review_rust` | 8192 | 8192 | 256 | 20260632 |
| `benchmark_qa_16k_001` | `benchmark_qa` | 16384 | 16384 | 256 | 20260634 |
| `tool_json_1k_001` | `tool_json` | 1024 | 1024 | 160 | 20260635 |

Family recommendations:

| Family | Recommendation | Token match | Max logit delta | Helper p95 ms | Native p95 ms | Native p95 delta | Native peak GB | Active KV bytes | Reason |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `benchmark_qa` | `blocked` | `true` | 0.500 | 1498.076 | 25246.230 | 1585.244% | 21.868 | 604094464 | Native peak MLX memory exceeded the 14 GB tiny16 cliff. |
| `chat_short` | `blocked` | `false` | 1.375 | 346.156 | 340.265 | -1.702% | 7.321 | 352436224 | Candidate generated token IDs did not match helper baseline. |
| `code_review_rust` | `native_opt_in` | `true` | 1.750 | 379.987 | 4468.121 | 1075.861% | 12.763 | 469876736 | Token parity held and active KV bytes were observed, but p95 missed the default gate. |
| `tool_json` | `blocked` | `false` | 2.375 | 110.312 | 227.390 | 106.133% | 7.321 | 352436224 | Candidate generated token IDs did not match helper baseline. |

## XR03 MTP Real-Context Diagnosis Snapshot

XR03 runs the real native target plus native MTP assistant over selected XR00
workloads and records per-draft-token trace evidence. It does not optimize
runtime code, switch defaults, or enable MTP.

| Field | Value |
|---|---|
| Command | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 4` |
| Evidence | `benchmarks/out/XR03-mtp-real-context-diagnosis/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Run ID | `xr03-1782868109-458074000` |
| Mode | `native_mtp_real_context_trace` |
| Records | `10`: 5 workloads x block sizes `1` and `2` |
| Decision | `blocked_with_evidence` |
| Root cause classification | `verifier_exactness_failure` |
| Exactness | `8/10` records byte-identical to non-MTP native output |
| Acceptance | `16/46 = 0.348` accepted draft tokens; every record had nonzero acceptance |
| Target top-k hits | `28/46 = 0.609` draft tokens appeared in target top-k |
| Target artifact SHA-256 | `d8b821776d41a61dad4f23f9b85cc8c6b09df2be04e5e4583f73c48739d8535c` |
| Assistant artifact SHA-256 | `6b31aa79ef7fce128572671b3890b55477694b52e24c75f48168f34770f85f2b` |
| Revision compatibility | Local artifact hashes recorded; upstream target/assistant revision alignment remains unverified |

Workload selection:

| Workload | Family | Target tokens | Actual tokens | Workload max new tokens | Selected max new tokens | Seed |
|---|---|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `chat_short` | 1024 | 1024 | 128 | 4 | 20260630 |
| `code_review_rust_4k_001` | `code_review_rust` | 4096 | 4096 | 192 | 4 | 20260631 |
| `benchmark_qa_4k_001` | `benchmark_qa` | 4096 | 4095 | 192 | 4 | 20260633 |
| `mtp_candidate_1k_001` | `mtp_candidate` | 1024 | 1024 | 64 | 4 | 20260641 |
| `mtp_candidate_4k_001` | `mtp_candidate` | 4096 | 4096 | 128 | 4 | 20260642 |

Record outcomes:

| Workload | Block | Exact | Accepted/Attempted | Draft top-k rate | Mean margin | First mismatch |
|---|---:|---|---:|---:|---:|---|
| `chat_short_1k_001` | 1 | `true` | 1/4 | 0.500 | 5.375 | none |
| `chat_short_1k_001` | 2 | `true` | 1/6 | 0.500 | 4.229 | none |
| `code_review_rust_4k_001` | 1 | `true` | 2/4 | 0.750 | 4.156 | none |
| `code_review_rust_4k_001` | 2 | `true` | 1/5 | 0.600 | 5.341 | none |
| `benchmark_qa_4k_001` | 1 | `false` | 1/4 | 0.500 | 4.609 | generated index 1: baseline `107`, MTP `45518` |
| `benchmark_qa_4k_001` | 2 | `false` | 1/6 | 0.500 | 3.990 | generated index 1: baseline `107`, MTP `45518` |
| `mtp_candidate_1k_001` | 1 | `true` | 3/4 | 1.000 | 2.438 | none |
| `mtp_candidate_1k_001` | 2 | `true` | 2/4 | 0.500 | 4.969 | none |
| `mtp_candidate_4k_001` | 1 | `true` | 2/4 | 0.750 | 4.828 | none |
| `mtp_candidate_4k_001` | 2 | `true` | 2/5 | 0.600 | 4.663 | none |

Blockers:

- `benchmark_qa_4k_001` block size `1` differed from non-MTP native output at generated index `1`: baseline `107`, MTP `45518`.
- `benchmark_qa_4k_001` block size `2` differed from non-MTP native output at generated index `1`: baseline `107`, MTP `45518`.

Ranked fix hypotheses:

1. Add a focused parity trace comparing target incremental decode with full verifier logits at the first divergent generated token.
2. Audit native MTP verify position offsets and target KV state after fallback commits near the 4K context boundary.
3. Keep MTP disabled by default and reject any acceptance-rate optimization until byte-identical exactness is restored.
4. Do not enable block sizes 3/4 until block sizes 1/2 have exactness and non-trivial acceptance.

## XR04 MTP Repair And A/B Snapshot

XR04 repairs the XR03 target-verifier exactness failure by verifying MTP drafts
against a cloned incremental target KV state, then committing the staged KV,
hidden state, and token list only after accepted/fallback tokens are known. It
does not enable MTP by default and does not test block sizes above `2`.

Generated files:

- Pre-fix repro: `benchmarks/out/XR04-mtp-repair-and-autotune/xr03-repro/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Exactness smoke: `benchmarks/out/XR04-mtp-repair-and-autotune/exactness-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Root A/B evidence: `benchmarks/out/XR04-mtp-repair-and-autotune/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.

Commands:

- Pre-fix repro: `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --workload-id benchmark_qa_4k_001 --max-new-tokens 4 --out-dir benchmarks/out/XR04-mtp-repair-and-autotune/xr03-repro`.
- Exactness smoke: `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 4 --block-sizes 1,2 --out-dir benchmarks/out/XR04-mtp-repair-and-autotune/exactness-smoke`.
- Root A/B evidence: `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 32 --block-sizes 1,2 --out-dir benchmarks/out/XR04-mtp-repair-and-autotune`.

Workload selection:

| Workload | Family | Target tokens | Actual tokens | Workload max new tokens | XR04 root selected max new tokens | Seed |
|---|---|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `chat_short` | 1024 | 1024 | 128 | 32 | 20260630 |
| `code_review_rust_4k_001` | `code_review_rust` | 4096 | 4096 | 192 | 32 | 20260631 |
| `benchmark_qa_4k_001` | `benchmark_qa` | 4096 | 4095 | 192 | 32 | 20260633 |
| `mtp_candidate_1k_001` | `mtp_candidate` | 1024 | 1024 | 64 | 32 | 20260641 |
| `mtp_candidate_4k_001` | `mtp_candidate` | 4096 | 4096 | 128 | 32 | 20260642 |

Pre-fix repro outcome:

- Decision: `blocked_with_evidence`.
- Exactness: `8/10` records byte-identical.
- Acceptance: `16/46 = 0.348`.
- Blocker reproduced: `benchmark_qa_4k_001` block sizes `1` and `2` differed at generated index `1`, baseline token `107`, MTP token `45518`.

Post-fix exactness smoke:

- Decision: `accept_candidate`.
- Exactness: `10/10` records byte-identical.
- Acceptance: `20/45 = 0.444`.
- Blockers: none.

Root 32-token A/B outcome:

- Decision: `accept_candidate`.
- Exactness: `10/10` records byte-identical.
- Acceptance: `162/370 = 0.438`.
- Blockers: none.
- Trace boundary: the repaired live incremental verifier records target top-1
  rather than XR03's full-forward target top-5; use the raw exactness and
  acceptance counts as the authoritative XR04 claim.

| Workload | Block | Exact | Accepted/Attempted | Baseline decode ms | MTP draft+verify ms | Generation result |
|---|---:|---|---:|---:|---:|---|
| `chat_short_1k_001` | 1 | `true` | 19/32 | 3090.559 | 3128.779 | slower |
| `chat_short_1k_001` | 2 | `true` | 22/40 | 3090.559 | 3151.925 | slower |
| `code_review_rust_4k_001` | 1 | `true` | 10/32 | 3614.960 | 9034.778 | slower |
| `code_review_rust_4k_001` | 2 | `true` | 17/45 | 3614.960 | 5216.344 | slower |
| `benchmark_qa_4k_001` | 1 | `true` | 8/32 | 7544.450 | 5565.894 | faster |
| `benchmark_qa_4k_001` | 2 | `true` | 8/49 | 7544.450 | 12475.474 | slower |
| `mtp_candidate_1k_001` | 1 | `true` | 15/32 | 4293.659 | 3427.004 | faster |
| `mtp_candidate_1k_001` | 2 | `true` | 13/40 | 4293.659 | 3377.771 | faster |
| `mtp_candidate_4k_001` | 1 | `true` | 25/32 | 4108.397 | 17801.738 | slower |
| `mtp_candidate_4k_001` | 2 | `true` | 25/36 | 4108.397 | 7537.366 | slower |

XR04 interpretation:

- Exactness blocker is repaired for the selected XR00 real-context corpus and
  block sizes `1` and `2`.
- MTP is a candidate for opt-in, per-family policy only. It won on
  `benchmark_qa_4k_001` block `1` and `mtp_candidate_1k_001` blocks `1`/`2`,
  but lost on `chat_short`, `code_review_rust`, and `mtp_candidate_4k`.
- No default enablement claim is allowed without variance runs, a stable
  per-family policy gate, restored top-k trace depth, and block-size-specific
  memory/latency guardrails.

## XR06 Native Decode Tail-Latency A/B Snapshot

XR06 compares native decode KV eval scheduling policies against the current
per-layer eval behavior. The benchmark records per-token decode traces with
input/output token IDs, position before/after `decode_one`, wall latency,
active KV bytes, peak MLX memory, and eval-policy markers. It does not optimize
runtime defaults.

| Field | Value |
|---|---|
| Command | `GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR06-native-decode-tail-latency-ab` |
| Evidence | `benchmarks/out/XR06-native-decode-tail-latency-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Smoke evidence | `benchmarks/out/XR06-native-decode-tail-latency-ab-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Run ID | `xr06-1782877235-943162000` |
| Git SHA | `92b0757fac8e789c98d02201a918d8b253a889ed` |
| Mode | `native_decode_tail_latency_real_context_ab` |
| Variants | `native_decode_eval_per_layer`, `native_decode_eval_end_of_decode`, `native_decode_eval_selective_full_attention`, `native_decode_eval_defer_to_logits` |
| Records | `60`: 5 workloads x 4 variants x 3 trials |
| Generated tokens per record | `64` |
| Correctness | `60/60` records passed native-vs-native token/logit gates |
| Memory gate | All selected workloads stayed below the 14 GB tiny16 gate; max selected peak was `12.829 GB` on `code_review_rust_8k_001` |
| Decision | `accept_candidate` |
| Blockers | none recorded |
| Runtime observation | System memory pressure was observed in yellow during the run with roughly `5 GB` swap; the benchmark process completed and wrote all artifacts |

Workload selection:

| Workload | Family | Target tokens | Actual tokens | Workload max new tokens | Selected max new tokens | Seed |
|---|---|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `chat_short` | 1024 | 1024 | 128 | 64 | 20260630 |
| `code_review_rust_4k_001` | `code_review_rust` | 4096 | 4096 | 192 | 64 | 20260631 |
| `code_review_rust_8k_001` | `code_review_rust` | 8192 | 8192 | 256 | 64 | 20260632 |
| `benchmark_qa_4k_001` | `benchmark_qa` | 4096 | 4095 | 192 | 64 | 20260633 |
| `tool_json_1k_001` | `tool_json` | 1024 | 1024 | 160 | 64 | 20260635 |

Baseline aggregates:

| Workload | Baseline raw p50 ms | Baseline raw p95 ms | Baseline raw p99 ms | Baseline steady p50 ms | Peak MLX GB | Tail reproduced |
|---|---:|---:|---:|---:|---:|---|
| `chat_short_1k_001` | 82.400 | 84.598 | 510.888 | 82.418 | 7.322 | `true` |
| `tool_json_1k_001` | 82.799 | 84.840 | 116.613 | 82.799 | 7.322 | `true` |
| `code_review_rust_4k_001` | 84.226 | 107.560 | 259.325 | 84.238 | 9.279 | `true` |
| `benchmark_qa_4k_001` | 84.314 | 89.092 | 620.689 | 84.214 | 9.279 | `true` |
| `code_review_rust_8k_001` | 85.333 | 86.731 | 2161.658 | 85.327 | 12.829 | `true` |

Accepted comparisons:

| Workload | Candidate | Raw p50 regression % | Raw p95 improvement % | Raw p99 improvement % | Interpretation |
|---|---|---:|---:|---:|---|
| `tool_json_1k_001` | `native_decode_eval_end_of_decode` | -13.546 | 11.772 | 23.168 | Passed via p99 tail gate with p50 improvement. |
| `chat_short_1k_001` | `native_decode_eval_selective_full_attention` | -12.353 | 9.772 | 19.082 | Passed via p99 tail gate with p50 improvement. |
| `code_review_rust_4k_001` | `native_decode_eval_selective_full_attention` | -8.458 | 16.065 | -134.795 | Passed via p95 tail gate; p99 worsened and must be treated as a workload-local tradeoff. |

XR06 interpretation:

- The current per-layer native decode eval path reproduced raw tail spikes on
  every selected workload while steady p50 stayed around `82-85 ms`.
- All candidates preserved native-vs-native greedy token/logit correctness and
  stayed below the selected-workload memory gate.
- `native_decode_eval_selective_full_attention` is the strongest follow-up
  candidate because it met gates on two workloads and kept p50 improved, but it
  worsened p99 on `code_review_rust_4k_001`, `code_review_rust_8k_001`,
  `benchmark_qa_4k_001`, and `tool_json_1k_001`.
- `native_decode_eval_defer_to_logits` improved p50 but failed every XR06
  p95/p99 tail gate and produced large p99 regressions on several workloads.
- No default runtime policy should change from XR06 alone. The evidence supports
  keeping decode eval scheduling opt-in while pursuing a stricter per-family or
  per-position policy and adding progress logging to the long runner.

## XR07 Prefix Cache Real Reuse A/B Snapshot

XR07 measures realistic RAM prefix reuse where a long real-context prefix is
cached and a small edited suffix is replayed before continuing generation. The
candidate warm path includes namespace lookup, native snapshot import, and
edited suffix replay overhead. Runtime code was not optimized.

| Field | Value |
|---|---|
| Command | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab -- --out-dir benchmarks/out/XR07-prefix-cache-real-reuse-ab --trials 2 --suffix-tokens 4 --suffix-edit-tokens 2 --continued-decode-tokens 4` |
| Evidence | `benchmarks/out/XR07-prefix-cache-real-reuse-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Smoke evidence | `benchmarks/out/XR07-prefix-cache-real-reuse-ab-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Run ID | `xr07-1782880867-63480000` |
| Git SHA | `6e4280bcb31787847e1b9696018e51b9a6baa1ed` |
| Mode | `native_ram_prefix_cache_real_reuse_ab` |
| Records | `6`: 3 contexts x 2 trials |
| Suffix shape | 4-token suffix with 2-token deterministic edit; 4 continued decode tokens |
| Namespace safety | Passed for every trial: base/adapter namespaces and block IDs differed, base-to-adapter and adapter-to-base restores rejected, wrong cache mode rejected, same-namespace miss recorded |
| Decision | `blocked_with_evidence` |
| Default policy | `do_not_enable_ram_prefix_cache_by_default_for_tiny16`; candidate cap would be `634 MiB` only if correctness, speed, and memory blockers are resolved |

Workload cases:

| Case | Context | Source workload | Prefix tokens | Suffix tokens | Edit distance | Derived seed | Suffix source |
|---|---:|---|---:|---:|---:|---:|---|
| `xr07_4k_code_review_rust_4k_001` | 4096 | `code_review_rust_4k_001` | 4092 | 4 | 2 | 1231492896 | `deterministic_token_suffix_edit` |
| `xr07_8k_prefix_reuse_edit_8k_a_001` | 8192 | `prefix_reuse_edit_8k_a_001` | 8188 | 4 | 2 | 2036799275 | `deterministic_token_suffix_edit` |
| `xr07_16k_long_repo_pack_16k_001` | 16384 | `long_repo_pack_16k_001` | 16380 | 4 | 2 | 426186536 | `deterministic_token_suffix_edit` |

Aggregate results:

| Case | Trials | Fresh full ms | Warm TTFT ms | Speedup | Lookup ms | Import ms | Suffix replay ms | Active KV MiB | Resident MiB | Peak MLX GB | Correct | Namespace | Meaningful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `xr07_4k_code_review_rust_4k_001` | 2/2 | 10601.746 | 1434.190 | 2.746x | 0.080 | 0.007 | 1434.103 | 384.000 | 767.875 | 10.718 | `false` | `true` | `true` |
| `xr07_8k_prefix_reuse_edit_8k_a_001` | 2/2 | 39631.771 | 21908.417 | 0.812x | 0.333 | 0.016 | 21908.067 | 448.000 | 895.875 | 15.710 | `false` | `true` | `false` |
| `xr07_16k_long_repo_pack_16k_001` | 2/2 | 100412.709 | 45707.847 | 2.197x | 1.011 | 0.562 | 45706.274 | 576.000 | 1151.875 | 27.353 | `false` | `true` | `true` |

XR07 blockers:

- Restored full-context continuation did not match fresh full prefill for both
  4K trials and both 16K trials.
- Continued greedy decode after restored suffix replay did not match fresh
  continuation for all 6 records.
- 8K and 16K crossed the 14 GB tiny16 peak MLX memory gate: `15.710 GB` and
  `27.353 GB`.
- 8K did not meet the warm TTFT gate after suffix replay; median speedup was
  `0.812x`.
- Every aggregate is low-N evidence (`2/2` trials).

XR07 interpretation:

- Real edited-suffix prefix reuse is not safe to enable by default. P06 exact
  snapshot restore remains valid for exact restored prefixes, but XR07 shows the
  current restore-plus-suffix-replay path is not fresh-prefill exact on real
  edited contexts.
- Namespace isolation behaved correctly, including adapter-qualified and
  cache-mode rejection, but safety admission is insufficient without restored
  continuation parity.
- The candidate cap estimate (`634 MiB`) is only a sizing note for a future
  corrected implementation. It is not a default policy recommendation while the
  decision remains `blocked_with_evidence`.

## XR08 SSD Cache Policy and Variance A/B Snapshot

XR08 measures repeated SSD prefix restore over real-context 8K and 16K
prefix-reuse workloads. It compares BF16 payloads against q8-compressed payloads
already available from P08. Runtime code was not optimized, and mid-decode SSD
fetch was tested only as a rejection path.

| Field | Value |
|---|---|
| Command | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr08_ssd_cache_policy_variance -- --out-dir benchmarks/out/XR08-ssd-cache-policy-variance` |
| Evidence | `benchmarks/out/XR08-ssd-cache-policy-variance/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Smoke evidence | `benchmarks/out/XR08-ssd-cache-policy-variance-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Run ID | `xr08-1782883921-278286000` |
| Git SHA | `0e4b0cd599f10a60e916d4b17c1abef1e7e78d38` |
| Mode | `native_ssd_cache_policy_variance` |
| Records | `12`: 2 contexts x 3 trials x 2 storage formats |
| Generated files | `records.jsonl`, `summary.json`, `report.md`, `blockers.md`, `decision.md`; per-trial SSD metadata, payload manifests, and safetensors payload paths are recorded in each record under `ssd_write` |
| Admission policy | `min_prefix_tokens=8192`, `max_cache_size_bytes=2147483648`, `ssd_metadata_budget_bytes=67108864`; 4K minimum-prefix and synthetic max-cache rejection probes passed |
| Decision | `keep_experimental` |
| Profile policy | `ssd_prefix_cache_opt_in_only_for_accepted_profiles` |

Workload cases:

| Case | Context | Source workload | Token length | Source deterministic seed | Prefix hash |
|---|---:|---|---:|---:|---|
| `xr08_8k_prefix_reuse_edit_8k_a_001` | 8192 | `prefix_reuse_edit_8k_a_001` | 8192 | 20260636 | `c54fb9aa08c7a1e5758782cc3cc8d5bd0e965bda133a294248f6dd2b380a05c5` |
| `xr08_16k_long_repo_pack_16k_001` | 16384 | `long_repo_pack_16k_001` | 16384 | 20260639 | `24b46f9ada942263e9c8a6fe134800b1b6dfeab8a0e14e05e30c4b4d55736e3e` |

Aggregate results:

| Case | Variant | Trials | Fresh p50 ms | Warm p50 ms | Warm p95 ms | p50 improvement | p95 improvement | Warm CV | Payload MiB | Metadata bytes | Peak MLX GB | Correct | Rejects | Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `xr08_8k_prefix_reuse_edit_8k_a_001` | `bf16` | 3/3 | 31654.550 | 3702.107 | 3976.410 | 88.305% | 87.695% | 0.036 | 528.069 | 52648 | 12.829 | `true` | `true` | `true` |
| `xr08_8k_prefix_reuse_edit_8k_a_001` | `mlx_affine_q8` | 3/3 | 31654.550 | 3224.018 | 3231.880 | 89.815% | 89.999% | 0.002 | 464.091 | 53577 | 12.829 | `true` | `true` | `true` |
| `xr08_16k_long_repo_pack_16k_001` | `bf16` | 3/3 | 94422.032 | 5124.271 | 5203.594 | 94.573% | 94.551% | 0.007 | 736.123 | 52983 | 21.986 | `true` | `true` | `false` |
| `xr08_16k_long_repo_pack_16k_001` | `mlx_affine_q8` | 3/3 | 94422.032 | 4231.843 | 4236.473 | 95.518% | 95.563% | 0.003 | 608.145 | 53912 | 21.986 | `true` | `true` | `false` |

XR08 blockers and caveats:

- No hard blockers were recorded.
- 8K BF16 and q8 both passed correctness, namespace/corruption/cache-mode
  rejection, zero-mid-decode-fetch, p50/p95 TTFT, variance, and memory gates.
- 16K BF16 and q8 passed correctness, rejection, p50/p95 TTFT, and variance
  gates, but crossed the 14 GB tiny16 memory gate in all three trials with peak
  MLX memory `21.986 GB`.
- q8 reduced payload size versus BF16 at both contexts, but active decode is
  restored into BF16 state; compressed active decode remains disabled.
- Full-run system snapshots showed substantial compression/swap history and zero
  throttled pages during sampled intervals. Treat 16K results as memory-risk
  evidence, not an enablement signal.

XR08 interpretation:

- SSD prefix cache should stay disabled by default. The measured evidence only
  supports opt-in, profile-gated experimentation for exact 8K real-context
  prefix restores under the same model/artifact/profile.
- 16K should not be accepted for the 16 GB profile despite strong warm TTFT
  wins because it repeatedly exceeded the memory cliff.
- Mid-decode SSD fetch remains disallowed; all measured restores happen before
  prefill and payload import.

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
| 2026-06-30 | Added native incremental KV decode for the opt-in hand-written graph: prefill materializes per-layer KV state, decode_one consumes cached K/V, sliding-window layers retain the last 1024 positions, full-attention layers retain the full prefix, and `active_kv_bytes` is surfaced through FFI/server JSON/HTTP metrics. | `native/gemma4_mlx/src/native_model.cc`, `native/gemma4_mlx/src/native_model.h`, `native/gemma4_mlx/src/runtime.cc`, `native/gemma4_mlx/include/gemma4_mlx.h`, `crates/gemma4d-ffi/src/lib.rs`, `crates/gemma4d-server/src/lib.rs`, `crates/gemma4d-server/src/http.rs` | `cargo test -p gemma4d-ffi -p gemma4d-server --all-targets`; native short probe with `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-server -- generate --model-path artifacts/models/gemma-4-12B-it-4bit --token-ids 9259 --max-context-tokens 32768 --max-new-tokens 8 --json`. |
| 2026-06-30 | Added P04 incremental native-KV benchmark harness and goal contract. The harness runs paired helper/default and native CLI probes, records active KV bytes, peak MLX memory, generated-token parity, greedy-logit diagnostics, raw decode latencies, and steady-state p50/p95 decode growth. | `crates/gemma4d-bench/examples/p04_incremental_native_kv.rs`, `codex/goals/P04-incremental-native-kv.goal.md` | `cargo test -p gemma4d-ffi -p gemma4d-server -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example p04_incremental_native_kv -- --out-dir benchmarks/out/P04-incremental-native-kv --model-path artifacts/models/gemma-4-12B-it-4bit`; `make verify`. |
| 2026-06-30 | Added committed-token metadata to `Gemma4StepResult` so real MTP verify/rollback can emit the target fallback token without scripted fixture knowledge. | `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/native_model.cc`, `crates/gemma4d-ffi/src/lib.rs` | `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets`; P05 benchmark run. |
| 2026-06-30 | Added P05 native MTP benchmark harness and goal contract. The harness uses real native target and assistant FFI handles, compares MTP output against non-MTP native output, records acceptance/rollback/speed/memory, and exercises auto-disable fallback. | `crates/gemma4d-bench/examples/p05_native_mtp.rs`, `codex/goals/P05-native-mtp.goal.md` | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p05_native_mtp -- --out-dir benchmarks/out/P05-native-mtp --model-path artifacts/models/gemma-4-12B-it-4bit --assistant-model-path artifacts/models/gemma-4-12B-it-qat-assistant-4bit`. |
| 2026-06-30 | Added native RAM KV snapshot export/import through the narrow C ABI, including cache-owned last-step retrieval and safe Rust `KvSnapshot` wrappers. | `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/native_model.cc`, `native/gemma4_mlx/src/native_model.h`, `native/gemma4_mlx/src/runtime.cc`, `crates/gemma4d-ffi/src/lib.rs` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi --all-targets`; P06 benchmark run. |
| 2026-06-30 | Added P06 real RAM prefix-cache benchmark harness and goal contract. The harness validates namespace-gated restore, imports real native snapshots, compares restored last-step and continued decode parity, and records warm TTFT/cache accounting for 4K/8K/16K. | `crates/gemma4d-bench/examples/p06_real_ram_prefix_cache.rs`, `codex/goals/P06-real-ram-prefix-cache.goal.md`, `crates/gemma4d-bench/Cargo.toml` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p06_real_ram_prefix_cache -- --out-dir benchmarks/out/P06-real-ram-prefix-cache --model-path artifacts/models/gemma-4-12B-it-4bit`. |
| 2026-06-30 | Added native SSD KV snapshot payload save/load through the narrow C ABI using safetensors-compatible files and safe Rust `KvSnapshot` wrappers. The payload path is failure-closed for non-MLX builds. | `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/native_model.cc`, `native/gemma4_mlx/src/native_model.h`, `native/gemma4_mlx/src/runtime.cc`, `crates/gemma4d-ffi/src/lib.rs` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi --all-targets`; `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --all-targets --no-run`; P07 benchmark run. |
| 2026-06-30 | Added P07 real SSD prefix-cache benchmark harness and goal contract. The harness writes SSD metadata plus real native safetensors payloads, restores before prefill only, verifies restored last-step and continued decode parity, records IO/latency metrics, and exercises namespace, corruption, and mid-decode rejection paths. | `crates/gemma4d-bench/examples/p07_real_ssd_prefix_cache.rs`, `codex/goals/P07-real-ssd-prefix-cache.goal.md`, `crates/gemma4d-bench/Cargo.toml` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets`; `make verify`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p07_real_ssd_prefix_cache -- --out-dir benchmarks/out/P07-real-ssd-prefix-cache --cache-dir benchmarks/out/P07-real-ssd-prefix-cache/ssd-cache --model-path artifacts/models/gemma-4-12B-it-4bit`. |
| 2026-06-30 | Added native compressed KV snapshot payload save through the narrow C ABI. The writer applies MLX affine q8 or packed q4 to selected KV tensors, records per-tensor min/scale metadata, keeps hidden/sliding tensors BF16 for P08 full-attention-only mode, and transparently reconstructs BF16 tensors on snapshot load. | `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/native_model.cc`, `native/gemma4_mlx/src/native_model.h`, `native/gemma4_mlx/src/runtime.cc`, `crates/gemma4d-ffi/src/lib.rs` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi --all-targets`; `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --all-targets --no-run`; P08 benchmark run. |
| 2026-06-30 | Added P08 real KV compression benchmark harness and goal contract. The harness compares BF16/q8/q4 real native prefix payloads at 4K/8K/16K, records payload memory reduction, warm restore latency, continued-decode greedy agreement/logit delta, active KV memory, and Planar/Iso disabled status. | `crates/gemma4d-bench/examples/p08_kv_compression.rs`, `codex/goals/P08-kv-compression.goal.md` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p08_kv_compression -- --out-dir benchmarks/out/P08-kv-compression --model-path artifacts/models/gemma-4-12B-it-4bit`. |
| 2026-06-30 | Added native PEFT LoRA adapter load/activate/clear/free through the narrow C ABI and safe Rust wrappers. The native graph applies active LoRA deltas inside target `quantized_linear`, shape-validates adapter A/B tensors against loaded Gemma 4 weights, and fails MTP closed while an adapter is active. | `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/native_model.cc`, `native/gemma4_mlx/src/native_model.h`, `native/gemma4_mlx/src/runtime.cc`, `crates/gemma4d-ffi/src/lib.rs` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi --all-targets --no-run`; `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --all-targets --no-run`; P09 benchmark run. |
| 2026-06-30 | Added P09 real LoRA adapter benchmark harness and goal contract. The harness creates a trusted local deterministic rank-16 adapter fixture with real Gemma 4 q_proj/v_proj shapes, imports it through the adapter registry, runs base/adapter/post-clear native generation, records load/hotswap/residency latency, checks manifest rejection, KV namespace isolation, and MTP-disabled behavior. | `crates/gemma4d-bench/examples/p09_real_lora_adapter.rs`, `codex/goals/P09-real-lora-adapter-hot-path.goal.md`, `crates/gemma4d-bench/Cargo.toml` | `cargo fmt --all --check`; `cargo test -p gemma4d-bench --all-targets --no-run`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p09_real_lora_adapter -- --out-dir benchmarks/out/P09-real-lora-adapter --model-path artifacts/models/gemma-4-12B-it-4bit`. |
| 2026-06-30 | Added P10 TUI live optimization console metrics, report writer, and benchmark harness. The TUI parses provider-only HTTP metrics for load/prefill/decode timing, throughput, memory, cache, MTP, adapters, server health, and latest benchmark report; the harness starts a localhost server and writes `tui-report.md`, `metrics.json`, and snapshots. | `crates/gemma4d-tui/src/{app.rs,provider.rs,ui.rs,lib.rs}`, `crates/gemma4d-tui/tests/m05_acceptance.rs`, `crates/gemma4d-bench/examples/p10_tui_live_console.rs`, `crates/gemma4d-bench/Cargo.toml` | `cargo test -p gemma4d-tui --all-targets`; `cargo run -p gemma4d-bench --example p10_tui_live_console -- --out-dir benchmarks/out/P10-tui-live-console`. |
| 2026-06-30 | Added `gemma4d-bench manifest`, reusable manifest capture structs, SHA-256 model identity in generic benchmark reports, P00 local artifact identity fields, and config validation that accepts local-artifact pins while warning on `PIN_ME` or unavailable revisions. | `crates/gemma4d-bench/src/manifest.rs`, `crates/gemma4d-bench/src/lib.rs`, `crates/gemma4d-bench/examples/p00_performance_baseline.rs`, `crates/gemma4d-tui/src/config.rs`, `references/configs/tiny16.toml`, `references/templates/benchmark-report.md` | `cargo fmt --all --check`; `cargo test -p gemma4d-bench --lib`; `cargo test -p gemma4d-bench --all-targets --no-run`; `cargo test -p gemma4d-tui --all-targets`; `cargo run -p gemma4d-bench -- manifest --out-dir benchmarks/out/P11-manifest-pinning`; `make verify`. |
| 2026-06-30 | Added XR00 real-context workload corpus generation: copied XR methodology docs/goal into root paths, added `gemma4d-bench workload-corpus`, generated deterministic prompt files and `workloads.jsonl`, and wrote XR00 decision/evidence artifacts. | `docs/xr-*.md`, `codex/goals/XR00-real-workload-corpus.goal.md`, `crates/gemma4d-bench/src/workload_corpus.rs`, `crates/gemma4d-bench/src/lib.rs`, `benchmarks/workloads/real-contexts/` | `cargo fmt --all --check`; `cargo test -p gemma4d-bench --lib`; `cargo test -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench -- workload-corpus --model-path artifacts/models/gemma-4-12B-it-4bit --workload-dir benchmarks/workloads/real-contexts --out-dir benchmarks/out/XR00-real-workload-corpus --python /opt/homebrew/opt/mlx-lm/libexec/bin/python --seed 20260630`. |
| 2026-06-30 | Added XR01 real-context A/B harness: reusable `xr_ab` report/evidence module, example runner, explicit baseline/candidate variant config, dry-run mode, failure-closed real-run mode, real helper smoke records, and XR01 decision artifacts. | `codex/goals/XR01-real-context-ab-harness.goal.md`, `crates/gemma4d-bench/src/xr_ab.rs`, `crates/gemma4d-bench/src/lib.rs`, `crates/gemma4d-bench/examples/xr01_real_context_ab.rs` | `cargo fmt --all --check`; `cargo test -p gemma4d-bench --lib`; `cargo test -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode dry-run --out-dir benchmarks/out/XR01-real-context-ab-harness-dry-run --max-workloads 1 --max-new-tokens 2`; `cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode both --out-dir benchmarks/out/XR01-real-context-ab-harness --max-workloads 1 --max-new-tokens 2`. |
| 2026-06-30 | Added XR02 native/helper real-context A/B profile on the reusable XR harness: XR02 defaults, native candidate env, generated-logit comparison, first-token and steady-state decode fields, per-family recommendations, deterministic seed/token metadata in records and reports, and failure-closed decision artifacts. | `codex/goals/XR02-native-helper-real-context-ab.goal.md`, `crates/gemma4d-bench/src/xr_ab.rs`, `crates/gemma4d-bench/examples/xr02_native_helper_real_context_ab.rs`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo test -p gemma4d-bench --all-targets`; `cargo run -p gemma4d-bench --example xr02_native_helper_real_context_ab -- --trials 2 --max-new-tokens 8`. |
| 2026-06-30 | Added XR03 native MTP real-context diagnosis trace path: C ABI trace metadata on MTP verify, Rust FFI trace decoding, a real-context XR03 runner, and generated decision artifacts. The change records draft tokens, target greedy tokens, target top-k, margins, accepted counts, verify time, sequence length, shared KV shapes, position offsets, artifact hashes, token lengths, and deterministic seeds. | `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/native_model.cc`, `crates/gemma4d-ffi/src/lib.rs`, `crates/gemma4d-bench/examples/xr03_mtp_real_context_diagnosis.rs`, `codex/goals/XR03-mtp-real-context-diagnosis.goal.md`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi --lib --no-run`; `cargo test -p gemma4d-bench --all-targets --no-run`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo test -p gemma4d-ffi --all-targets --no-run`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 4`. |
| 2026-07-01 | Repaired native MTP verification to stage against cloned incremental target KV instead of full-prefix verifier recompute. The live verifier now compares drafts against the cache's last-step prediction, advances accepted/fallback tokens through `decode_incremental`, swaps staged KV/hidden/tokens only after success, and records top-1 incremental trace evidence. | `native/gemma4_mlx/src/runtime.cc`, `BENCHMARKS.md` | `cargo check -p gemma4d-ffi`; `cargo check -p gemma4d-bench --example xr03_mtp_real_context_diagnosis`; `cargo test -p gemma4d-ffi --lib`; `cargo test -p gemma4d-bench --lib`; `cargo check -p gemma4d-bench --example p05_native_mtp --example xr03_mtp_real_context_diagnosis`; XR04 pre-fix repro, exactness smoke, and root 32-token A/B runs. |
| 2026-07-01 | Added XR05 prefill/eval scheduling A/B harness and opt-in knobs: helper `GEMMA4D_MLX_LM_PREFILL_CHUNK_TOKENS`, helper `GEMMA4D_MLX_LM_PREFILL_CLEAR_CACHE`, and native `GEMMA4D_NATIVE_PREFILL_KV_EVAL`. The runner records command, seeds, token lengths, MLX peak memory, RSS, prefill tok/s, TTFT, correctness gates, low-N status, blockers, and decision artifacts. It also enforces candidate-wide no-correctness-regression before accepting any workload-local win. | `.codex/agents/tui-ux-engineer.toml`, `codex/goals/XR05-prefill-and-eval-scheduling-ab.goal.md`, `native/gemma4_mlx/scripts/gemma4d_mlx_lm_helper.py`, `native/gemma4_mlx/src/native_model.cc`, `crates/gemma4d-bench/examples/xr05_prefill_eval_scheduling_ab.rs`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo check -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab`; `cargo check -p gemma4d-ffi`; TOML parse for `.codex/agents/tui-ux-engineer.toml`; smoke and full XR05 runs with escalated Metal access. |
| 2026-07-01 | Added XR06 native decode tail-latency A/B harness and opt-in native decode KV eval scheduling modes. The runner records deterministic workload seeds/token lengths, per-token latency traces, position before/after decode, active KV bytes, peak MLX memory, eval-policy markers, correctness gates, blockers, failed hypotheses, and decision artifacts. | `codex/goals/XR06-native-decode-tail-latency-ab.goal.md`, `native/gemma4_mlx/src/native_model.cc`, `crates/gemma4d-bench/examples/xr06_native_decode_tail_latency_ab.rs`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo check -p gemma4d-ffi`; `cargo check -p gemma4d-bench --example xr06_native_decode_tail_latency_ab`; smoke and full XR06 runs with escalated Metal access. |
| 2026-07-01 | Added XR07 real-prefix RAM cache A/B harness and goal contract. The runner derives 4K/8K/16K real-context repeated-prefix cases from the XR00 corpus, applies deterministic small suffix edits, compares fresh full prefill against RAM restore plus native import and suffix replay, records hit rate, warm TTFT, restore/import/replay latency, continued decode parity, active KV bytes, cache residency, adapter namespace isolation, failed hypotheses, blockers, and default-policy decision artifacts. It does not optimize runtime code. | `codex/goals/XR07-prefix-cache-real-reuse-ab.goal.md`, `crates/gemma4d-bench/examples/xr07_prefix_cache_real_reuse_ab.rs`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo check -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab`; `cargo check -p gemma4d-ffi`; `cargo test -p gemma4d-kv --lib`; XR07 smoke and full runs with escalated Metal access. |
| 2026-07-01 | Added XR08 SSD cache policy and variance harness and goal contract. The runner measures real-context SSD prefix restore variance for BF16 and q8 payloads, records exact generated artifacts, deterministic seeds, token lengths, metadata/payload IO, warm TTFT, fresh prefill, payload checksum time, native import time, cache accounting, corruption/namespace/cache-mode rejection, mid-decode rejection, admission probes, failed hypotheses, blockers, and profile-gated decision artifacts. It does not optimize runtime code. | `codex/goals/XR08-ssd-cache-policy-variance.goal.md`, `crates/gemma4d-bench/examples/xr08_ssd_cache_policy_variance.rs`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo check -p gemma4d-bench --example xr08_ssd_cache_policy_variance`; `cargo check -p gemma4d-ffi`; `cargo check -p gemma4d-bench --examples`; `cargo test -p gemma4d-kv --lib`; XR08 smoke and full runs with escalated Metal access. |

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
| 2026-06-30 | `cargo fmt --all --check` | Passed | Formatting gate after P04 native KV and benchmark changes. |
| 2026-06-30 | `cargo test -p gemma4d-ffi -p gemma4d-server -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for P04 FFI/server metrics and benchmark harness. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p04_incremental_native_kv -- --out-dir benchmarks/out/P04-incremental-native-kv --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Required escalated Metal access; wrote P04 records, summary, report, and blocker report with no blockers. |
| 2026-06-30 | `make verify` | Passed | Sandboxed run failed only at localhost bind with `Operation not permitted`; escalated rerun passed. |
| 2026-06-30 | `cargo fmt --all --check` | Passed | Formatting gate after P05 FFI and benchmark changes. |
| 2026-06-30 | `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for P05 FFI committed-token metadata and benchmark harness. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p05_native_mtp -- --out-dir benchmarks/out/P05-native-mtp --model-path artifacts/models/gemma-4-12B-it-4bit --assistant-model-path artifacts/models/gemma-4-12B-it-qat-assistant-4bit` | Passed | Required escalated Metal access; wrote P05 records, summary, report, and blocker report with no blockers. |
| 2026-06-30 | `cargo fmt --all --check` | Passed | Formatting gate after P06 native snapshot ABI and benchmark changes. |
| 2026-06-30 | `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for P06 FFI wrappers and benchmark harness. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p06_real_ram_prefix_cache -- --out-dir benchmarks/out/P06-real-ram-prefix-cache --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Required escalated Metal access; wrote P06 records, summary, report, and blocker report with no blockers at clean SHA `e5e61ad`. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_FULL_MODEL_TESTS=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo test -p gemma4d-ffi native_graph_prefills_one_token_when_explicitly_enabled -- --nocapture` | Passed | Required escalated Metal access; covers real native target/assistant FFI path and committed-token metadata assertions. |
| 2026-06-30 | `make verify` | Passed | Sandboxed run failed only at localhost bind with `Operation not permitted`; escalated rerun passed. |
| 2026-06-30 | `cargo fmt --all --check` | Passed | Formatting gate after P08 compressed snapshot API and benchmark changes. |
| 2026-06-30 | `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for P08 FFI wrappers and benchmark harness. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --all-targets --no-run` | Passed | Required MLX build gate for compressed native snapshot payload API. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p08_kv_compression -- --out-dir benchmarks/out/P08-kv-compression --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Required escalated Metal access; wrote P08 records, summary, report, and blocker report with no blockers at clean SHA `5993b86`. |
| 2026-06-30 | `cargo fmt --all --check` | Passed | Formatting gate after P09 native adapter ABI and benchmark changes. |
| 2026-06-30 | `cargo test -p gemma4d-ffi --all-targets --no-run` | Passed | Focused FFI compile gate for native adapter load/activate/clear wrappers. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --all-targets --no-run` | Passed | Required MLX build gate for native LoRA adapter loading and delta application code. |
| 2026-06-30 | `cargo test -p gemma4d-bench --all-targets --no-run` | Passed | Focused compile coverage for the P09 benchmark harness and adapter-registry dependency. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p09_real_lora_adapter -- --out-dir benchmarks/out/P09-real-lora-adapter --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | Required escalated Metal access; wrote P09 records, summary, report, and blocker report with no blockers at clean SHA `8723d50`. |
| 2026-06-30 | `cargo test -p gemma4d-ffi -p gemma4d-bench --all-targets` | Passed | Focused post-P09 test coverage for FFI wrappers and benchmark harness after ledger update. |
| 2026-06-30 | `make verify` | Passed | Sandboxed run failed only at localhost bind with `Operation not permitted`; escalated rerun passed after P09 changes. |
| 2026-06-30 | `cargo test -p gemma4d-tui --all-targets` | Passed | Focused P10 TUI coverage for live HTTP metrics, required page snapshots, render p95 reporting, and terminal lifecycle tests. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example p10_tui_live_console -- --out-dir benchmarks/out/P10-tui-live-console` | Passed | Spawned localhost server, drove the TUI over `HttpProvider`, wrote `tui-report.md`, `metrics.json`, and 18 snapshots; render p95 `1731 us` under `20000 us`. |
| 2026-06-30 | `cargo test -p gemma4d-bench --lib` | Passed | Unit coverage for manifest CLI parsing and generic benchmark report manifest identity rendering. |
| 2026-06-30 | `cargo test -p gemma4d-bench --all-targets --no-run` | Passed | Compile coverage for benchmark examples after the manifest module and dependency changes. |
| 2026-06-30 | `cargo test -p gemma4d-tui --all-targets` | Passed | Config validation coverage for local-artifact pins and `PIN_ME` warning behavior. |
| 2026-06-30 | `cargo run -p gemma4d-bench -- manifest --out-dir benchmarks/out/P11-manifest-pinning` | Passed | Wrote manifest and report with target/drafter hashes, safetensor inventories, Rust/MLX/mlx-lm versions, git SHA, and machine summary. |
| 2026-06-30 | `cargo run -p gemma4d-bench -- workload-corpus --model-path artifacts/models/gemma-4-12B-it-4bit --workload-dir benchmarks/workloads/real-contexts --out-dir benchmarks/out/XR00-real-workload-corpus --python /opt/homebrew/opt/mlx-lm/libexec/bin/python --seed 20260630` | Passed | Wrote 13 workload records, prompt files, and XR00 evidence artifacts; local tokenizer measured exact 1K/4K/8K/16K/24K context lengths with no blockers. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode dry-run --out-dir benchmarks/out/XR01-real-context-ab-harness-dry-run --max-workloads 1 --max-new-tokens 2` | Passed | CI/offline smoke wrote dry-run records and decision artifacts without requiring the 12B model; decision is `needs_more_data` by design because no real model path is exercised. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode both --out-dir benchmarks/out/XR01-real-context-ab-harness --max-workloads 1 --max-new-tokens 2` | Passed | Wrote final XR01 records, summary, report, blocker report, and decision; includes dry-run and real helper smoke records with no blockers. |
| 2026-06-30 | `cargo fmt --all --check` | Passed | Formatting gate after XR02 harness/report metadata changes. |
| 2026-06-30 | `cargo test -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for XR02 harness defaults, report schema, and example runner. |
| 2026-06-30 | `cargo run -p gemma4d-bench --example xr02_native_helper_real_context_ab -- --trials 2 --max-new-tokens 8` | Blocked with evidence | Wrote 20 real records and XR02 decision artifacts; example exits nonzero by design when decision is `blocked_with_evidence`. |
| 2026-06-30 | `cargo fmt --all --check` | Passed | Formatting gate after XR03 trace ABI, FFI, and runner changes. |
| 2026-06-30 | `cargo test -p gemma4d-ffi --lib --no-run` | Passed | Focused Rust FFI compile gate for `MtpTraceInfo` decoding. |
| 2026-06-30 | `cargo test -p gemma4d-bench --all-targets --no-run` | Passed | Focused compile coverage for the XR03 benchmark example. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo test -p gemma4d-ffi --all-targets --no-run` | Passed | Required MLX build gate for native MTP trace ABI changes. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-workloads 1 --max-new-tokens 2 --out-dir benchmarks/out/XR03-mtp-real-context-diagnosis-smoke` | Passed | Required escalated Metal access; wrote smoke artifacts for one workload and confirmed trace records/top-k output. Sandboxed attempt failed before benchmark execution because MLX could not access a Metal device. |
| 2026-06-30 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 4` | Blocked with evidence | Required escalated Metal access; wrote 10 real records and XR03 decision artifacts. Example exits nonzero by design when decision is `blocked_with_evidence`; blocker is `benchmark_qa_4k_001` exactness failure for block sizes 1 and 2. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --workload-id benchmark_qa_4k_001 --max-new-tokens 4 --out-dir benchmarks/out/XR04-mtp-repair-and-autotune/xr03-repro` | Blocked with evidence | Required escalated Metal access; pre-fix reproduction wrote 10 records because default selected workload IDs remained active. Reproduced `benchmark_qa_4k_001` block sizes `1` and `2` mismatch at generated index `1`, baseline `107`, MTP `45518`. |
| 2026-07-01 | `cargo check -p gemma4d-ffi` | Passed | Focused compile gate for the native runtime verifier repair through the Rust FFI crate. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example xr03_mtp_real_context_diagnosis` | Passed | Focused compile gate for the XR03/XR04 evidence runner after the native verifier change. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 4 --block-sizes 1,2 --out-dir benchmarks/out/XR04-mtp-repair-and-autotune/exactness-smoke` | Passed | Required escalated Metal access; wrote 10 records, decision `accept_candidate`, exactness `10/10`, acceptance `20/45 = 0.444`, no blockers. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- --max-new-tokens 32 --block-sizes 1,2 --out-dir benchmarks/out/XR04-mtp-repair-and-autotune` | Passed | Required escalated Metal access; wrote root XR04 evidence artifacts, decision `accept_candidate`, exactness `10/10`, acceptance `162/370 = 0.438`, and no blockers. |
| 2026-07-01 | `cargo test -p gemma4d-ffi --lib` | Passed | 12 passed, 1 ignored; includes native graph full-model test path when local model env is available. |
| 2026-07-01 | `cargo test -p gemma4d-bench --lib` | Passed | 14 passed; covers benchmark report/schema helpers and workload corpus validation. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example p05_native_mtp --example xr03_mtp_real_context_diagnosis` | Passed | Ensures the older P05 MTP harness and XR03/XR04 trace runner still compile after the native runtime verifier change. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after the XR04 runtime and benchmark-ledger changes. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after XR05 harness and decision-gate changes. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab` | Passed | Focused compile gate for the XR05 benchmark runner. |
| 2026-07-01 | `cargo check -p gemma4d-ffi` | Passed | Focused native/FFI compile gate after helper/native scheduling knobs. |
| 2026-07-01 | `python3 -c 'import pathlib,tomllib; data=tomllib.loads(pathlib.Path(".codex/agents/tui-ux-engineer.toml").read_text()); assert data["developer_instructions"].strip(); assert data["sandbox_mode"] == "workspace-write"; print("tui agent toml ok")'` | Passed | Confirms the `tui_ux_engineer` agent role has a parseable `developer_instructions` field and normalized schema metadata. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR05-prefill-and-eval-scheduling-ab` | Passed | Required escalated Metal access; wrote 72 real-context prefill records for 4K/8K/16K across helper chunk/cache and native eval variants. Derived decision is `reject_candidate` after candidate-wide correctness gating. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after XR06 native decode eval scheduling and benchmark runner changes. |
| 2026-07-01 | `cargo check -p gemma4d-ffi` | Passed | Focused native/FFI compile gate for the XR06 decode KV eval scheduling knob. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example xr06_native_decode_tail_latency_ab` | Passed | Focused compile gate for the XR06 benchmark runner. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --trials 1 --max-new-tokens 8 --clear-workload-ids --workload-id chat_short_1k_001 --variants native_decode_eval_per_layer,native_decode_eval_defer_to_logits --out-dir benchmarks/out/XR06-native-decode-tail-latency-ab-smoke` | Passed | Required escalated Metal access; wrote smoke artifacts with 2/2 records passed and no blockers. Decision was `reject_candidate` because the smoke intentionally had fewer than three trials. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR06-native-decode-tail-latency-ab` | Passed | Required escalated Metal access; wrote 60 real-context records, 3 trials, 64 generated tokens, no blockers, and decision `accept_candidate`. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after the XR07 runner and benchmark-ledger changes. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab` | Passed | Focused compile gate for the XR07 prefix-cache real-reuse runner. |
| 2026-07-01 | `cargo check -p gemma4d-ffi` | Passed | Focused native/FFI compile gate before XR07 native MLX execution. |
| 2026-07-01 | `cargo test -p gemma4d-kv --lib` | Passed | 18 passed; covers namespace mismatch, adapter partitioning, RAM/SSD restore, cache accounting, and compression metadata tests. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab -- --out-dir benchmarks/out/XR07-prefix-cache-real-reuse-ab-smoke --clear-contexts --context 4096 --trials 1 --suffix-tokens 4 --suffix-edit-tokens 2 --continued-decode-tokens 1` | Blocked with evidence | Required escalated Metal access; wrote smoke artifacts and exposed restored-continuation plus continued-decode parity blockers at 4K. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab -- --out-dir benchmarks/out/XR07-prefix-cache-real-reuse-ab --trials 2 --suffix-tokens 4 --suffix-edit-tokens 2 --continued-decode-tokens 4` | Blocked with evidence | Required escalated Metal access; wrote 6 real-context records and final XR07 artifacts. Decision is `blocked_with_evidence`; default policy is `do_not_enable_ram_prefix_cache_by_default_for_tiny16`. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after the XR08 runner. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example xr08_ssd_cache_policy_variance` | Passed | Focused compile gate for the XR08 SSD policy and variance runner. |
| 2026-07-01 | `cargo check -p gemma4d-ffi` | Passed | Focused native/FFI compile gate before XR08 native MLX execution. |
| 2026-07-01 | `cargo check -p gemma4d-bench --examples` | Passed | Compile coverage for all benchmark examples after adding XR08. |
| 2026-07-01 | `cargo test -p gemma4d-kv --lib` | Passed | 18 passed; covers namespace mismatch, adapter partitioning, RAM/SSD restore, cache accounting, compression metadata, corruption rejection, and mid-decode SSD rejection. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr08_ssd_cache_policy_variance -- --out-dir benchmarks/out/XR08-ssd-cache-policy-variance-smoke --clear-contexts --context 8192 --trials 1 --modes bf16,q8` | Passed | Required escalated Metal access after sandboxed MLX failed with no Metal device; wrote 2 smoke records and final smoke artifacts. Decision was `reject_candidate` because low-N evidence cannot establish variance. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr08_ssd_cache_policy_variance -- --out-dir benchmarks/out/XR08-ssd-cache-policy-variance` | Passed | Required escalated Metal access; wrote 12 real-context records and final XR08 artifacts. Decision is `keep_experimental`; 8K BF16/q8 accepted for opt-in experimentation, 16K BF16/q8 rejected for tiny16 memory. |

## Current Claim Boundaries

- M12 and P00 broad throughput claims are helper-backed through the Rust C ABI
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
- P04 confirms incremental native KV decode only for text-only greedy probes in
  the P04 report. The native graph remains opt-in; helper/default fallback
  remains available.
- P04 steady-state decode growth excludes the first four native decode_one
  samples to separate MLX/JIT/cache warmup from sustained decode latency. Raw
  samples remain in `records.jsonl` and `summary.json`.
- P04 long-context greedy-logit deltas are diagnostic because generated token
  IDs matched helper outputs. They should not be used as proof of broad
  numerical parity outside the measured probes.
- P05 proves real native MTP correctness only for the measured text-only greedy
  probes and block sizes `1` and `2`.
- P05 does not justify enabling MTP by default: the measured assistant acceptance
  rate was `0.000`, and the benchmark recommends `keep_disabled_by_default`.
- P05 excludes adapter-active MTP, compressed active KV, and sampling MTP.
- P06 proves RAM-only native snapshot restore for measured 4K/8K/16K
  text-only greedy prefixes. It does not prove SSD payload persistence,
  adapter-active snapshot reuse, compressed active KV, server integration, or
  sampling behavior.
- P06 warm TTFT measures namespace restore plus native snapshot import and
  cached last-step retrieval. Snapshot export cost is reported separately and is
  paid when the prefix is first cached, not on the warm restore path.
- P07 proves SSD-backed native snapshot payload restore only before prefill for
  the measured 4K/8K/16K text-only greedy prefixes. SSD remains disabled by
  default pending broader variance data.
- P08 proves q8/q4 prefix-payload compression only for full-attention KV tensors
  restored back into BF16 active decode state. It does not enable compressed
  active decode; q4 failed greedy agreement in the measured run.
- P09 proves one trusted local deterministic rank-16 PEFT LoRA adapter fixture
  on the opt-in native graph. It does not enable remote adapter loading, aLoRA,
  adapter fusion, default server adapter routing, or adapter-active MTP.
- P09 adapter output evidence is a greedy-logit delta on the measured 128-token
  deterministic prompt; generated token IDs did not differ in the final default
  run, though the shorter smoke run changed the prefill greedy token.
- P10 validates the TUI live optimization console against a localhost HTTP
  server with the stub backend. It is a provider/API and render-latency claim,
  not a native model throughput claim.
- P10 render latency is for deterministic 120x40 snapshot rendering in the
  artifact run. It does not measure interactive terminal overhead or long-running
  operator sessions.
- P11 records local artifact identity because the local downloaded model
  directories do not include upstream revision metadata. The target and drafter
  are pinned by `local-artifact-sha256:*` values in `tiny16.toml`; this is
  reproducible for the local artifact set but is not a claim about an upstream
  Hugging Face commit.
- XR00 is a corpus and token-metadata claim only. It does not execute Gemma 4
  inference, measure latency, compare backends, enable MTP, enable cache
  policies, or optimize runtime code.
- XR01 accepts the A/B harness shape and evidence schema only. The final smoke
  run uses one 1K workload and helper-backed baseline/candidate configs, so it
  does not claim a runtime speedup, native backend superiority, server
  readiness, cache benefit, MTP benefit, or adapter behavior.
- XR02 does not justify making native incremental the default on measured
  real-context workloads. `chat_short` and `tool_json` failed token parity, and
  `benchmark_qa` hit a 21.868 GB native peak MLX memory cliff against the
  14 GB tiny16 gate.
- XR02's `code_review_rust` family is `native_opt_in` only: generated token
  parity held and native active KV bytes were observed, but native p95 decode
  missed the default gate by 1075.861%.
- XR03 does not justify enabling MTP by default. Nonzero acceptance was observed
  on real-context workloads, but byte-identical exactness failed on
  `benchmark_qa_4k_001` for block sizes `1` and `2`.
- XR03 is diagnosis evidence only. The top-k, margin, shared-KV-shape, and
  position-offset traces are valid for the measured selected XR00 workloads and
  artifact hashes, not a fix or a runtime performance claim.
- XR03 keeps block sizes above `2` disabled; block sizes `3` and `4` remain
  design-only until exactness gates pass.
- XR04 repairs native MTP exactness for the selected XR00 real-context
  workloads at block sizes `1` and `2`. It does not enable MTP by default,
  does not enable block sizes above `2`, and does not change adapter-active,
  compressed-active-KV, server-default, or sampling behavior.
- XR04 performance evidence is mixed. The 32-token root run shows generation
  wins only for `benchmark_qa_4k_001` block `1` and `mtp_candidate_1k_001`
  blocks `1`/`2`; other measured workload/block pairs were slower.
- XR04 incremental verifier trace records target top-1, not XR03's full-forward
  target top-5. Use XR04 exactness, acceptance, and timing artifacts for repair
  claims; restore deeper incremental score tracing before making rank/top-k
  drafter-quality claims.
- XR05 rejects all prefill/eval scheduling candidates for default adoption.
  Helper chunk `512` and `1024` produced workload-local memory wins, but each
  variant had correctness failures on another selected workload. Helper
  no-clear-cache and chunk `4096` regressed p95 or memory. Native
  `end_of_prefill` and `selective_full_attention` stayed correctness-clean but
  did not meet the 10% p50 or 5% memory gate.
- XR05 records the native 16K memory cliff as still present: native 16K peak MLX
  stayed around `22 GB`, above the tiny16 comfort envelope. No default runtime
  code path or policy should change from XR05 alone.
- XR06 accepts native decode eval scheduling as an opt-in experimental candidate
  only. It does not change the default per-layer decode eval policy.
- XR06 tail improvements are workload-local. `end_of_decode` met the p99 gate
  on `tool_json_1k_001`; `selective_full_attention` met the p99 gate on
  `chat_short_1k_001` and the p95 gate on `code_review_rust_4k_001`, but it
  worsened p99 on several other selected workloads.
- XR06 excludes 16K/24K memory-sentinel workloads. The selected 1K/4K/8K matrix
  stayed below 14 GB peak MLX, but system memory pressure reached yellow with
  roughly 5 GB swap during the run. Treat tiny16 adoption as unresolved until a
  smaller policy matrix and sentinel run pass.
- XR07 does not justify enabling RAM prefix cache by default. The real edited
  suffix restore path failed restored-continuation or continued-decode parity on
  every selected context, even though namespace safety checks passed.
- XR07 warm TTFT claims include namespace lookup, native snapshot import, and
  edited suffix replay. The 8K case was slower than fresh full prefill after
  suffix replay, and 8K/16K crossed the 14 GB tiny16 memory gate.
- XR07's `634 MiB` cap is only a candidate sizing note after blockers are
  resolved. It is not an adoption recommendation while the decision remains
  `blocked_with_evidence`.
- XR08 keeps SSD prefix cache experimental and profile-gated. It does not enable
  SSD prefix cache by default, does not permit mid-decode SSD fetch, and does not
  make production serving readiness claims.
- XR08 supports only exact real-context prefix restore claims for the measured
  8K profiles under BF16/q8 payload storage. The 16K profiles are rejected for
  the 16 GB profile because peak MLX memory crossed `21.986 GB`.
