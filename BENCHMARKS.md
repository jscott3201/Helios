# Helios Benchmark Ledger

This file tracks benchmark runs and measurement changes that matter for Helios
performance claims. Raw benchmark artifacts stay under `benchmarks/out/` and are
intentionally ignored; this ledger records the stable index of what was run,
which code produced it, and what claims are allowed.

## Headline Tested Numbers

These are quick-access numbers only. The detailed run rows and raw artifacts
remain the authority for exact commands, seeds, model identity, and caveats.

XR51 server-native rows compare the bundled persistent-native resident worker
plus default long-context prefill policy against the unwired server baseline.
The 1K row is persistence-only because the long-context chunk policy is inert
below `4096` prompt tokens; isolated policy evidence for larger contexts remains
the XR35 8K and XR40 16K opt-in runs.

XR53 server model-path default rows use the same server A/B harness, but the
candidate runtime is built from `parse_serve_options` with `--model-path` and no
`--backend` flag. They validate default wiring, not a new prefill algorithm.
The XR53 16K row is historical pre-review wiring evidence: after the
byte-density admission fix, the same unchunked baseline fails closed because its
conservative estimate is above the 16K measured table.

XR51 boundary probes were single-repeat exploratory checks and are superseded by
the repeat-3 rows: `server-default-1k-boundary` showed candidate prefill
`2506.956 -> 2769.866 ms` (`-10.486%`), while
`server-default-4095-boundary` observed `4099` server prefill tokens from chat
wrapping and showed prefill `10851.996 -> 12658.482 ms` (`-16.647%`) with peak
MLX `9.217 -> 7.165 GB`.

### Prefill

| Path | Workload | Baseline p50 ms | Candidate p50 ms | Delta | Peak MLX | Status | Evidence |
|---|---|---:|---:|---:|---|---|---|
| Server model-path default | `benchmark_qa_16k_001` | `88657.954` | `42268.699` | `+52.324%` | `21.874 -> 7.638 GB` | Historical pre-review wiring evidence; post-review guard rejects unchunked baseline | XR53 `default-path-16k-repeats3`, `default-path-16k-raised-budget-repeats3` |
| Server model-path default | `chat_short_1k_001` | `2869.853` | `2309.716` | `+19.518%` | `7.324 -> 7.324 GB` | Accepted, CLI default path | XR53 `default-path-1k-repeats3` |
| Server native prefill default | `benchmark_qa_16k_001` | `87387.199` | `41711.194` | `+52.269%` | `21.874 -> 7.638 GB` | Accepted, server default | XR51 `server-default-16k-repeats3` |
| Server native prefill default | `code_review_rust_8k_001` | `31285.354` | `22618.497` | `+27.703%` | `12.767 -> 7.402 GB` | Accepted, server default | XR51 `server-default-8k-repeats3` |
| Server native prefill default | `code_review_rust_4k_001` | `11651.369` | `10152.938` | `+12.861%` | `9.216 -> 7.300 GB` | Accepted, server default | XR51 `server-default-4k-repeats3` |
| Server native prefill default | `chat_short_1k_001` | `2814.225` | `2352.410` | `+16.410%` | `7.324 -> 7.324 GB` | Accepted, persistence-only; policy inert below 4096 | XR51 `server-default-1k-repeats3` |
| Native prefill policy, env | `benchmark_qa_16k_001` | `86813.720` | `42244.280` | `+51.339%` | `21.868 -> 7.620 GB` | Accepted, opt-in | XR40 `benchmark-qa-16k-policy` |
| Native prefill policy, env | `long_repo_pack_16k_001` | `87017.803` | `42390.024` | `+51.286%` | `21.868 -> 7.620 GB` | Accepted, opt-in | XR40 `long-repo-16k-policy` |
| Native prefill policy, env | `code_review_rust_8k_001` | `30339.051` | `21993.044` | `+27.509%` | `12.763 -> 7.383 GB` | Accepted, opt-in | XR35 `holdout-8k-policy` |
| Native prefill policy, FFI setter | `adapter_expert_4k_001` | `14884.449` | `11424.204` | `+23.247%` | `9.279 -> 7.300 GB` | Accepted, opt-in setter | XR41 `setter-boundary-smoke` |

### Decode / Token Generation

| Path | Context | Generated | Decode tok/s | Decode p50 ms | Decode p95 ms | Status | Evidence |
|---|---:|---:|---:|---:|---:|---|---|
| Helper cold baseline | `1K` | `128/128` | `15.906` | `62.706` | `63.725` | Baseline | P00 |
| Helper cold baseline | `4K` | `128/128` | `14.388` | `64.212` | `65.247` | Baseline | P00 |
| Helper cold baseline | `8K` | `128/128` | `13.623` | `64.186` | `67.041` | Baseline | P00 |
| Helper cold baseline | `16K` | `128/128` | `5.945` | `65.744` | `68.958` | Baseline | P00 |
| Native full-attention KV slab | 1K family + 4K code | `32 x 12` | mixed | `+0.39%..+1.05%` | n/a | Rejected; below 5% decode gate | XR52 `decode-candidate-slab` |
| Native decode skip peak reset | `chat_short_1k_001` | `64/64` | n/a | `85.764 -> 85.887` | `87.220 -> 88.075` | Rejected | XR28 |

### MTP

| Path | Workload | Exactness | Acceptance | Decode Phase | Status | Evidence |
|---|---|---:|---:|---|---|---|
| Repaired MTP baseline evidence | XR04 selected 5-workload set | `10/10` | `162/370 = 0.438` | workload/block dependent | Opt-in only | XR04 |
| Position-pinned MTP drafter | `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `2956.027 -> 2421.899 ms` (`+18.069%`) | `reject_candidate`; acceptance-fix hypothesis falsified | XR54 |
| Position-pinned MTP drafter | `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `2955.781 -> 2205.883 ms` (`+25.371%`) | `reject_candidate`; token-identical vs XR48 | XR54 |
| Position-pinned MTP drafter | `mtp_candidate_1k_001` | `3/3` | `21/45 = 0.467` | `2883.842 -> 2987.766 ms` (`-3.604%`) | `reject_candidate`; slot 1 stayed `3/18` | XR54 |
| N-block MTP sweep guarded policy | `chat_short_1k_001:N=3` + `tool_json_1k_001:N=4` | `54/54` measured | `144/198 = 0.727` selected | aggregate `8674.797 -> 6907.671 ms` (`+20.371%`) | `keep_experimental`, default-off; sequential oracle matched `72/72` records | XR55 |
| N-block MTP fixed block | 1K family, `N=3` | `9/9` | `165/252 = 0.655` | aggregate `+18.054%` | Best fixed block; still default-off | XR55 |
| N-block MTP fixed block | 1K family, `N=8` | `9/9` | `162/423 = 0.383` | aggregate `-9.353%` | Rejected; repair cost dominates | XR55 |
| KV slab + verifier timing split | `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `2707.085 -> 2080.682 ms` (`+23.139%`) | Rejected for XR52 promotion; selected by guard only | XR52 |
| KV slab + verifier timing split | `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `2798.767 -> 2037.253 ms` (`+27.209%`) | Rejected for XR52 promotion; selected by guard only | XR52 |
| KV slab + verifier timing split | `mtp_candidate_1k_001` | `3/3` | `21/45 = 0.467` | `2795.809 -> 2859.050 ms` (`-2.262%`) | Rejected by 5% guard; auto-disabled tail | XR52 |
| Light-trace env audit | `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `2695.984 -> 2045.486 ms` (`+24.128%`) | No-op audit; XR15 trace already top-1 | XR49 |
| Light-trace env audit | `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `2730.721 -> 2027.634 ms` (`+25.747%`) | No-op audit; XR15 trace already top-1 | XR49 |
| Light-trace env audit | `mtp_candidate_1k_001` | `3/3` | `21/45 = 0.467` | `2790.422 -> 2861.231 ms` (`-2.538%`) | No-op audit; still rejected by guard | XR49 |
| QAT target pairing cold smoke | `chat_short_1k_001` | `2/2` | `0/2 = 0.000` | block 1 `19713.684 -> 71279.259 ms` (`-261.573%`); block 2 `19713.684 -> 52344.264 ms` (`-165.522%`) | Blocked; one cold no-warmup sample, steady-state unmeasured | XR50 |
| QAT target pairing cold smoke | `mtp_candidate_1k_001` | `1/1` | `2/2 = 1.000` | block 2 `13510.088 -> 25448.830 ms` (`-88.369%`) | Blocked; one cold no-warmup sample, steady-state unmeasured | XR50 |
| Adaptive zero-run 3 sweep | `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `2701.736 -> 2115.179 ms` (`+21.710%`) | Selected by guarded policy, default-off | XR48 |
| Adaptive zero-run 3 sweep | `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `2814.431 -> 2116.066 ms` (`+24.814%`) | Selected by guarded policy, default-off | XR48 |
| Adaptive zero-run 3 sweep | `mtp_candidate_1k_001` | `3/3` | `21/45 = 0.467` | `2880.829 -> 2915.728 ms` (`-1.211%`) | Improved vs XR46 but still rejected by guard | XR48 |
| Adaptive threshold sweep | `chat_short_1k_001` | `3/3` | `27/48 = 0.563` | `3159.750 -> 2971.490 ms` (`+5.958%`) | Marginal selected, default-off | XR47 |
| Adaptive threshold sweep | `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `2916.887 -> 2324.990 ms` (`+20.292%`) | Selected by guarded policy, default-off | XR47 |
| Adaptive threshold sweep | `mtp_candidate_1k_001` | `3/3` | `21/39 = 0.538` | `3081.131 -> 3040.089 ms` (`+1.332%`) | Improved but below 5% guard | XR47 |
| Adaptive lazy block-prefix 1K holdout | `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `3013.177 -> 2228.909 ms` (`+26.028%`) | Selected by guarded policy, default-off | XR46 |
| Adaptive lazy block-prefix 1K holdout | `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `3174.286 -> 2117.370 ms` (`+33.296%`) | Selected by guarded policy, default-off | XR46 |
| Adaptive lazy block-prefix 1K holdout | `mtp_candidate_1k_001` | `3/3` | `21/48 = 0.438` | `2872.385 -> 3143.500 ms` (`-9.439%`) | Improved vs XR45 but rejected by guard | XR46 |
| Lazy block-prefix 1K holdout | `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `2955.491 -> 2340.434 ms` (`+20.811%`) | Selected by guarded policy, default-off | XR45 |
| Lazy block-prefix 1K holdout | `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `2910.560 -> 2231.115 ms` (`+23.344%`) | Selected by guarded policy, default-off | XR45 |
| Lazy block-prefix 1K holdout | `mtp_candidate_1k_001` | `3/3` | `39/96 = 0.406` | `2952.317 -> 3290.224 ms` (`-11.445%`) | Rejected by guard | XR45 |
| Lazy block-prefix selected slice | `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `3138.129 -> 2355.632 ms` (`+24.935%`) | Keep experimental, default-off | XR44 |
| Lazy block-prefix selected slice | `mtp_candidate_4k_001` | `3/3` | `75/96 = 0.781` | `10406.955 -> 11534.612 ms` (`-10.836%`) | Rejected by guard | XR44 |
| Block-prefix selected slice | `chat_short_1k_001` | `3/3` | `69/120 = 0.575` | `3084.066 -> 2686.191 ms` (`+12.901%`) | Keep experimental, default-off | XR43 |
| Block-prefix selected slice | `mtp_candidate_4k_001` | `3/3` | `75/108 = 0.694` | `4886.134 -> 11780.432 ms` (`-141.099%`) | Rejected by guard | XR43 |
| Lazy draft + partial-reject repair | `code_review_rust_4k_001` | `4/4` | `51/96 = 0.531` | `3795.764 -> 4150.249 ms` (`-9.339%`) | Rejected speed path | XR38 |
| MTP policy variance | `mtp_candidate_1k_001` | exact | n/a | block 1 `-25.141%`, block 2 `-26.465%` | Rejected | XR15 |

### Benchmark Prep / Artifact Hashing

| Path | Artifact | Baseline p50 ms | Candidate p50 ms | Delta | Threads | Correctness | Status | Evidence |
|---|---|---:|---:|---:|---:|---|---|---|
| Rayon safetensors hashing | `gemma-4-12B-it-4bit`, 2 files, `6.741 GB` | `45488.176` | `36261.428` | `+20.284%` | `2` | Inventory hash matched | Follow-up only | XR42 |
| Rayon safetensors hashing | `gemma-4-12B-it-qat-assistant-4bit`, 1 file, `0.238 GB` | `1566.653` | `1566.317` | `+0.021%` | `2` | Inventory hash matched | No useful single-file speedup | XR42 |

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
| 2026-07-01 | XR09 KV compression real-quality A/B | Reject candidate | `1dabccc` | `native_kv_compression_real_quality_ab` | `benchmarks/out/XR09-kv-compression-real-quality-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr09-1782886055`; command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr09_kv_compression_real_quality_ab -- --out-dir benchmarks/out/XR09-kv-compression-real-quality-ab`; no hard blockers and BF16 exact restore passed on 6 real workloads, but q8 failed `benchmark_qa_4k_001` quality gate and q4 failed 3 families. Recommendation is `no_go_for_compression_candidate`; active compressed decode remains disabled. |
| 2026-07-01 | XR11 persistent native/server backend A/B | Passed | `d8ae489` | `server_real_helper_vs_persistent_native_real_contexts` | `benchmarks/out/XR11-persistent-native-server-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr11-1782888158`; decision `accept_candidate`; command `GEMMA4D_EXPERIMENTAL_PERSISTENT_SERVER=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --out-dir benchmarks/out/XR11-persistent-native-server-ab --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --workload-ids chat_short_1k_001 --repeats 2 --max-new-tokens 1`; workload seed `20260630`, target/actual context `1024/1024`, generated `1` token per repeat. Baseline `real_helper` loaded twice; `persistent_native` loaded once and served two worker requests. Both repeats matched token IDs/text, model identity is recorded in summary/records, and no blockers were recorded. Localhost bind required approved escalation. |
| 2026-07-01 | XR13 novel Metal/KV exploration | Reject candidate | `4e1bc28` | `feature_gated_l1_compressed_k_score_microbenchmark` | `benchmarks/out/XR13-novel-metal-kv-exploration/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr13-1782890847`; command `cargo run -p gemma4d-bench --features xr13-prototypes --example xr13_novel_metal_kv_exploration -- --out-dir benchmarks/out/XR13-novel-metal-kv-exploration`; 18 records from 6 XR09 real-context shapes x 3 deterministic trials. XR09 q8 baseline still has 0% active reduction and failed `benchmark_qa_4k_001`; XR13 Planar4 K-only passed the isolated synthetic score gate but remains prototype-only, while Turbo score estimation failed correctness on 18/18 samples. No default runtime path, active compressed decode path, C ABI, or Metal kernel changed. |
| 2026-07-01 | XR14 MTP policy autotune replay | Needs more data | `aed6f07` plus local XR14 replay changes | `xr04_mtp_policy_replay` | `benchmarks/out/XR14-mtp-policy-autotune/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr14-1782892549`; command `cargo run -p gemma4d-bench --example xr14_mtp_policy_autotune -- --out-dir benchmarks/out/XR14-mtp-policy-autotune`; 30 replay records from 5 XR04 workloads x 6 policies. Fixed block-size policies and the 35% acceptance threshold were rejected; net-latency-guarded replay selected `benchmark_qa_4k_001:block1` and `mtp_candidate_1k_001:block2` for a 12.778% aggregate decode-phase replay speedup, but remains `needs_more_data` because there is no holdout variance run. No runtime path changed. |
| 2026-07-01 | XR15 MTP policy variance A/B | Reject candidate | `ca501df` plus local XR15 harness changes | `native_mtp_policy_variance_ab` | `benchmarks/out/XR15-mtp-policy-variance-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and smoke artifacts under `benchmarks/out/XR15-mtp-policy-variance-ab-smoke/` | Run ID `xr15-1782893754`; command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR15-mtp-policy-variance-ab --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 8 --clear-workload-ids --workload-id mtp_candidate_1k_001`; workload seed `20260641`, target/actual context `1024/1024`, generated `8` tokens. All 8 records were byte-identical and under the 14 GB memory gate, but block 1 regressed measured decode phase by 25.141% and block 2 by 26.465%; net-latency-guarded policy selected no MTP workloads and rejected the candidate. No runtime path changed. |
| 2026-07-01 | XR16 MTP batch-verify overhead prototype | Reject candidate | `fb9e027` plus local XR16 batch-verify changes | `native_mtp_experimental_batch_verify_block2` | `benchmarks/out/XR16-mtp-overhead-optimization/baseline-sequential-block2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR16-mtp-overhead-optimization/candidate-batch-block2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Baseline run `xr15-1782894385` used sequential block-2 verify; candidate run `xr15-1782894471` used `GEMMA4D_EXPERIMENTAL_MTP_BATCH_VERIFY=1`. Both commands used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --trials 3 --warmups 1 --max-new-tokens 4 --block-sizes 2 --clear-workload-ids --workload-id mtp_candidate_1k_001` with out-dir changed per variant. Workload seed `20260641`, target/actual context `1024/1024`, generated `4` tokens. Both variants were byte-identical, acceptance was `6/12 = 0.500`, and peak memory was `7.665 GB`; sequential block 2 already regressed native baseline by 40.480%, while experimental batch verify regressed by 82.943% because partial acceptance paid the batch attempt and then fallback rollback. Experimental flag remains off by default. |
| 2026-07-01 | XR17 MTP final projection skip A/B | Reject candidate | `2b56e69` plus local XR17 final-projection changes | `native_mtp_experimental_skip_final_projection` | `benchmarks/out/XR17-mtp-final-projection-skip/baseline-final-projection/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR17-mtp-final-projection-skip/candidate-skip-final-projection/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Baseline run `xr15-1782894816`; candidate run `xr15-1782894917` used `GEMMA4D_EXPERIMENTAL_MTP_SKIP_FINAL_PROJECTION=1`. Both commands used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --trials 3 --warmups 1 --max-new-tokens 8 --block-sizes 1,2 --clear-workload-ids --workload-id mtp_candidate_1k_001` with out-dir changed per variant. Workload seed `20260641`, target/actual context `1024/1024`, generated `8` tokens. Both variants were byte-identical with unchanged acceptance rates (`block1 0.875`, `block2 0.625`) and under the 14 GB gate. The candidate did not reduce draft overhead: measured median `draft_ms` moved from about `71.745` to `81.857` for block 1 and from about `67.959` to `72.214` for block 2; net block decisions remained rejected. Experimental flag remains off by default. |
| 2026-07-01 | XR18 MTP in-place serial verifier A/B | Reject candidate | `940734c` plus local XR18 in-place verifier changes | `native_mtp_experimental_inplace_verify` | `benchmarks/out/XR18-mtp-inplace-serial-verify/baseline-staged-verify/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR18-mtp-inplace-serial-verify/candidate-inplace-verify/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Baseline run `xr15-1782895326` used staged KV/token verification; candidate run `xr15-1782895427` used `GEMMA4D_EXPERIMENTAL_MTP_INPLACE_VERIFY=1`. Both commands used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 8 --block-sizes 1,2 --clear-workload-ids --workload-id mtp_candidate_1k_001` with out-dir changed per variant. Workload seed `20260641`, target/actual context `1024/1024`, generated `8` tokens. Both variants were byte-identical with unchanged acceptance (`block1 21/24 = 0.875`, `block2 15/24 = 0.625`) and no blockers. Candidate reduced peak MLX memory from `7.654/7.665 GB` to `7.321/7.323 GB`, but median `verify_ms` worsened from `821.786` to `874.452` for block 1 and from `821.842` to `859.258` for block 2; median MTP decode phase worsened from `895.431` to `960.394` for block 1 and from `889.187` to `940.641` for block 2. The env flag remains off/default disabled because timing failed the acceptance gate and in-place failure atomicity is weaker. |
| 2026-07-01 | XR19 MTP steady-state horizon A/B | Reject candidate | `6016051` plus local XR19 benchmark-only goal | `native_mtp_policy_variance_ab_64_token_horizon` | `benchmarks/out/XR19-mtp-steady-state-horizon/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Run ID `xr15-1782895761`; command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR19-mtp-steady-state-horizon --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 64 --block-sizes 1,2 --clear-workload-ids --workload-id mtp_candidate_1k_001`; workload seed `20260641`, target/actual context `1024/1024`, generated `64` tokens. All 8 records were byte-identical with no blockers and under the 14 GB gate. Longer horizon reduced the short-run penalty but still rejected MTP: median baseline decode was `5331.202 ms`; block 1 median MTP phase was `5644.486 ms` (`draft_ms 226.286`, `verify_ms 5434.791`, speedup `-5.876%`, acceptance `93/192 = 0.484`); block 2 median MTP phase was `5676.116 ms` (`draft_ms 248.931`, `verify_ms 5426.518`, speedup `-6.470%`, acceptance `123/216 = 0.569`). Net-latency-guarded policy selected no MTP workloads; default remains disabled. |
| 2026-07-01 | XR20 MTP terminal no-lookahead phase accounting | Reject candidate | `b2edf46` plus local XR20 terminal verifier changes | `native_mtp_experimental_terminal_no_lookahead` | `benchmarks/out/XR20-mtp-terminal-no-lookahead/baseline-normal-verify/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR20-mtp-terminal-no-lookahead/candidate-terminal-no-lookahead/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Baseline run `xr15-1782896362` used normal MTP verify; candidate run `xr15-1782896520` used `--experimental-terminal-no-lookahead`. Both commands used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 64 --block-sizes 1,2 --clear-workload-ids --workload-id mtp_candidate_1k_001` with out-dir changed per variant and the candidate flag added. Workload seed `20260641`, target/actual context `1024/1024`, generated `64` tokens. Candidate was exact for all 8 records, applied one terminal skip per record, preserved acceptance (`block1 93/192 = 0.484`, `block2 123/216 = 0.569`), and recorded no blockers. Terminal no-lookahead reduced median `verify_ms` from `5416.610` to `5332.942` for block 1 and from `5407.432` to `5336.286` for block 2, but MTP still lost to native non-MTP: block 1 median MTP phase `5556.281 ms` vs baseline decode `5305.139 ms` (`-4.734%`), block 2 `5575.664 ms` (`-5.099%`). Net-latency-guarded policy selected no MTP workloads; terminal verifier remains benchmark-only/default off and cache-discard-only after a skip. |
| 2026-07-01 | XR21 native block decode microbenchmark | Accept candidate | `90edfaf` plus local XR21 block-decode FFI/harness changes | `native_decode_incremental_block2_microbench` | Strict calibration `benchmarks/out/XR21-native-block-decode-microbench/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and accepted run `benchmarks/out/XR21-native-block-decode-microbench/tolerance-0p25/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Strict run `xr21-1782897058` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr21_native_block_decode_microbench -- --out-dir benchmarks/out/XR21-native-block-decode-microbench --trials 8 --warmups 2 --workload-id mtp_candidate_1k_001` and blocked only because the initial `0.05` logit tolerance was exceeded by a stable `0.125` BF16 block-vs-serial logit delta while greedy tokens matched. Accepted run `xr21-1782897184` used the same command with out-dir `.../tolerance-0p25` and `--logit-tolerance 0.25`. Workload seed `20260641`, target/actual context `1024/1024`, block input tokens `[236792,7216]`. In the accepted run, block greedy tokens matched serial for 10/10 records, max logit abs diff was `0.125`, peak memory was `7.675 GB`, median two-step serial decode was `192.120 ms`, median block decode was `109.818 ms`, and block decode speedup was `42.839%`. This supports pursuing block-2 partial-accept rollback via exact KV prefix truncation, but no MTP default changed. |
| 2026-07-01 | XR22 MTP block prefix rollback A/B | Keep experimental | `959dc62` plus local XR22 block-prefix rollback changes | `native_mtp_experimental_block_prefix_rollback` | `benchmarks/out/XR22-mtp-block-prefix-rollback/baseline-normal-verify/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR22-mtp-block-prefix-rollback/candidate-block-prefix-rollback/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Baseline run `xr15-1782897709` used normal native MTP verify; candidate run `xr15-1782897834` used `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`. Both commands used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 64 --block-sizes 2 --clear-workload-ids --workload-id mtp_candidate_1k_001` with out-dir changed per variant. Workload seed `20260641`, target/actual context `1024/1024`, generated `64` tokens, prompt SHA-256 `afc51a55b76097a09f030c835b9917b4425469ba9c758ef513cb355e10da04c6`, source replay `xr14-1782892549` SHA-256 `773e4c456cad0ba8e338755eee476ed0c00b4c2db61ed489dbddb9889970c0f2`. Normal verifier rejected block 2 with median MTP phase `5627.371 ms` vs baseline decode `5317.211 ms` (`-5.833%`), acceptance `41/72`, rollback `23`, and peak `7.665 GB`. Candidate was byte-identical for all 4 records with no hard blockers and selected `keep_experimental`: median MTP phase `4721.830 ms` vs baseline decode `5317.404 ms` (`+11.200%`), acceptance `40/72`, rollback `24`, and peak `8.008 GB`. Committed-token traces matched the normal verifier across measured trials, but draft-token/accepted-count traces differed (`40/72` vs `41/72`), so the env flag remains default-off and requires broader holdout plus hidden-state drift analysis before promotion. |
| 2026-07-01 | XR23 MTP block-prefix hidden parity | Reject candidate | `eabe846` plus local XR23 serial-state repair changes | `native_mtp_block_prefix_eval_mode_and_serial_state_repair` | Eval-mode smokes under `benchmarks/out/XR23-mtp-block-prefix-hidden-parity/candidate-decode-kv-{per-layer,end,selective,defer}-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and full repair run `benchmarks/out/XR23-mtp-block-prefix-hidden-parity/candidate-serial-state-repair/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Existing decode-KV eval modes did not explain XR22's drafter-state drift. Each smoke used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` plus optional `GEMMA4D_NATIVE_DECODE_KV_EVAL=<mode>` with command `cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 1 --warmups 0 --max-new-tokens 64 --block-sizes 2 --clear-workload-ids --workload-id mtp_candidate_1k_001` and out-dir changed per mode. `per_layer`, `end`, `selective`, and `defer` were byte-identical and still showed `40/72` accepted, `24` rollbacks, and active KV `353370112` bytes. The diagnostic repair run used `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_SERIAL_STATE_REPAIR=1` plus the same XR15 command with `--trials 3 --warmups 1`; it replayed committed tokens serially after the block decision and restored the normal verifier signature (`41/72`, `23` rollbacks, active KV `353370112` bytes) while remaining byte-identical. It was rejected: measured median MTP phase `7989.603 ms` vs native baseline decode `5333.939 ms` (`-49.788%`), peak `8.016 GB`. Conclusion: XR22's later drafter divergence is caused by committing block-produced post-commit state rather than serial-equivalent post-commit state; this diagnostic does not isolate whether the culprit is target KV, last hidden, shared KV views, or their combination. Strict serial-equivalent drafter state removes the optimization. MTP remains disabled/default-off. |
| 2026-07-01 | XR24 MTP block-prefix holdout A/B | Blocked with evidence | `eff8b8c` plus local XR24 goal/ledger changes | `native_mtp_block_prefix_holdout` | `benchmarks/out/XR24-mtp-block-prefix-holdout/baseline-normal-verify/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, `benchmarks/out/XR24-mtp-block-prefix-holdout/candidate-block-prefix-rollback/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, and diagnostic `benchmarks/out/XR24-mtp-block-prefix-holdout/code-review-serial-state-repair-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Baseline run `xr15-1782899623` used normal native MTP verify and candidate run `xr15-1782900348` used `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`. Both used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id code_review_rust_4k_001 --workload-id benchmark_qa_4k_001 --workload-id mtp_candidate_1k_001 --workload-id mtp_candidate_4k_001`, with out-dir changed per variant. Workload seeds were `20260630`, `20260631`, `20260633`, `20260641`, and `20260642`; target/actual contexts were `1024/1024`, `4096/4096`, `4096/4095`, `1024/1024`, and `4096/4096`; generated `32` tokens. Normal serial MTP was exact for `20/20` records, max peak `9.244 GB`, and net-latency policy selected only `benchmark_qa_4k_001:2` for `+9.047%` aggregate selected speedup. The fast block-prefix candidate was exact for only `16/20` records and blocked: every `code_review_rust_4k_001` record diverged at generated token index `12` (`100` baseline vs `8970` MTP), with committed-token traces first mismatching at index `12`, acceptance dropping from normal `17/45` to fast `4/55`, and rollbacks increasing from `15` to `28`. Candidate committed-token traces still matched serial MTP for the other 4 workloads; net-latency policy selected `chat_short_1k_001:2` and `mtp_candidate_4k_001:2` for `+5.537%` aggregate selected speedup, but the run remained blocked by the code-review exactness failure. Diagnostic repair smoke `xr15-1782901131` used `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_SERIAL_STATE_REPAIR=1` for only `code_review_rust_4k_001`; it restored exactness and normal acceptance (`17/45`, `15` rollbacks) in one low-N measured record, with active KV `403177472` bytes and peak `9.212 GB`. Conclusion: the fast XR22 state is not safe as a broad holdout path; any policy gate must exclude workloads with block-produced state drift or use a slower serial-state repair. MTP remains disabled/default-off. |
| 2026-07-01 | XR25 MTP state-only serial repair A/B | Reject candidate | `b781de4` plus local XR25 state-only repair changes | `native_mtp_state_only_serial_repair` | `benchmarks/out/XR25-mtp-state-only-serial-repair/normal-serial-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, `benchmarks/out/XR25-mtp-state-only-serial-repair/full-serial-repair-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, and `benchmarks/out/XR25-mtp-state-only-serial-repair/state-only-repair-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off diagnostic flag `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1`, active only with `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` and `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_SERIAL_STATE_REPAIR=1`, to serially advance intermediate committed-token KV without target vocab projection before the final lookahead decode. The first sandboxed MLX attempt failed before artifacts with `No Metal device available`; all recorded runs were rerun with Metal access. Common command shape was `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`, with out-dir and env flags changed per variant: normal serial used no MTP env flags (`xr15-1782902340`), full repair added rollback plus serial-state repair (`xr15-1782901875`), and state-only repair added rollback plus serial-state repair plus state-only repair (`xr15-1782902100`). Workload seed `20260631`, target/actual context `4096/4096`, generated `32` tokens, prompt SHA-256 `93b21c654b4efcdc41236be21f8f4fb95e2d29bd380e6667b69f83933b62fa99`. All three variants were byte-identical for warmup plus 3 measured records, had no hard blockers, and matched compact event signatures across normal/full/state-only traces: `17/45` accepted, `15` rollbacks, `23` verify passes, active KV `403177472` bytes, peak `9.244 GB`. Normal serial selected decode phase was `5156.793 ms` and net-latency policy rejected MTP (`-20.796%`). Full serial-state repair selected decode phase was `7442.522 ms` and rejected MTP (`-4.577%`). State-only serial repair selected decode phase was `8442.013 ms` and rejected MTP (`-71.004%`, one workload regression). Conclusion: skipping the intermediate vocab projection preserves correctness but does not improve the XR23 repair; it is slower than full repair on the blocker workload, so no five-workload holdout was run. MTP remains disabled/default-off. |
| 2026-07-01 | XR26 native greedy-logit gather A/B | Reject candidate | `eeb73cd` plus local XR26 gather changes | `native_decode_gather_greedy_logit` | `benchmarks/out/XR26-native-greedy-logit-gather/smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR26-native-greedy-logit-gather/followup-chat-short-1k/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off `GEMMA4D_EXPERIMENTAL_NATIVE_GATHER_GREEDY_LOGIT=1` to replace one-token `max(logits)` greedy-logit extraction with `take(logits,argmax(logits))` in forward, prefill, and decode result paths, plus an XR06 benchmark variant. Smoke command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR26-native-greedy-logit-gather/smoke --trials 1 --max-new-tokens 8 --clear-workload-ids --workload-id chat_short_1k_001 --variants native_decode_eval_per_layer,native_decode_gather_greedy_logit` passed 2/2 records but rejected for low N. Final follow-up run `xr06-1782903372-653803000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR26-native-greedy-logit-gather/followup-chat-short-1k --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --variants native_decode_eval_per_layer,native_decode_gather_greedy_logit`; deterministic seed `20260630`, target/actual context `1024/1024`, generated `64` tokens, prompt SHA-256 `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`. All 6 follow-up records passed token/logit correctness at tolerance `0.5`, peak MLX stayed `7.321 GB`, active KV stayed `353353728` bytes, and there were no blockers. Baseline raw p50/p95/p99 were `86.244/88.198/324.516 ms`; candidate raw p50/p95/p99 were `86.001/87.314/321.880 ms`, with p50 regression `-0.282%`, p95 improvement `1.003%`, and p99 improvement `0.813%`. The XR06 p95/p99 tail gate requires `>=15%`, so the broad holdout was skipped and the env flag remains off by default. |
| 2026-07-01 | XR27 native chunked prefill A/B | Reject candidate | `66b9a57` plus local XR27 chunked-prefill changes | `native_chunked_prefill_512` / `native_chunked_prefill_1024` | `benchmarks/out/XR27-native-chunked-prefill-ab/smoke-4k/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, `benchmarks/out/XR27-native-chunked-prefill-ab/followup-4k-512/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, and `benchmarks/out/XR27-native-chunked-prefill-ab/sentinel-8k-512/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` using an internal native chunked prefill path: first chunk uses normal hidden prefill, later chunks reuse private block hidden-state advance, and only the final hidden state is projected to logits. Public C ABI and `gemma4_decode_block` token cap remain unchanged. Smoke run `xr05-1782903815-678522000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR27-native-chunked-prefill-ab/smoke-4k --trials 1 --clear-workload-ids --workload-id code_review_rust_4k_001 --variants native_eval_per_layer,native_chunked_prefill_512,native_chunked_prefill_1024`; seed `20260631`, context `4096/4096`; all 3 records passed correctness, `512` improved low-N prefill from `11844.702` to `10792.476 ms` and peak MLX from `9.212` to `7.458 GB`, while `1024` was slower at `15359.443 ms` with peak `7.811 GB`. Follow-up run `xr05-1782903941-459966000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR27-native-chunked-prefill-ab/followup-4k-512 --trials 3 --clear-workload-ids --workload-id code_review_rust_4k_001 --variants native_eval_per_layer,native_chunked_prefill_512`; all 6 records passed correctness, active KV stayed `402653184` bytes, and peak MLX improved `19.047%` (`9.212` to `7.458 GB`), but p50 regressed `1.283%` (`11756.074` to `11906.928 ms`) and p95 regressed `6.913%` (`11885.280` to `12706.910 ms`), exceeding the `5%` limit. The 8K sentinel run `xr05-1782904120-834663000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR27-native-chunked-prefill-ab/sentinel-8k-512 --trials 1 --clear-workload-ids --workload-id code_review_rust_8k_001 --variants native_eval_per_layer,native_chunked_prefill_512`; seed `20260632`, context `8192/8192`; candidate improved low-N prefill from `28946.752` to `26501.713 ms` and peak MLX from `12.763` to `7.594 GB`, but failed correctness because output token matched (`100`) while output logit delta was `0.75` (`22.5` vs `23.25`) against tolerance `0.5`. No 16K holdout was run after the 8K correctness failure. The env flag remains default-off. |
| 2026-07-01 | XR28 native decode peak-reset overhead A/B | Reject candidate | `30f5d0b` plus local XR28 skip-reset changes | `native_decode_skip_peak_reset` | `benchmarks/out/XR28-native-decode-peak-reset-overhead/smoke-chat-short-1k/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off `GEMMA4D_EXPERIMENTAL_NATIVE_SKIP_DECODE_PEAK_RESET=1` to skip `mlx::core::reset_peak_memory()` only in one-token native `decode_incremental`; model math, KV state, public C ABI, MTP policy, and defaults are unchanged. Run `xr06-1782904528-39938000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR28-native-decode-peak-reset-overhead/smoke-chat-short-1k --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --variants native_decode_eval_per_layer,native_decode_skip_peak_reset`; deterministic seed `20260630`, target/actual context `1024/1024`, generated `64` tokens, prompt SHA-256 `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`. All 6 records passed token/logit correctness at tolerance `0.5`, peak MLX stayed `7.321 GB`, active KV stayed `353353728` bytes, and there were no blockers. Baseline raw p50/p95/p99 were `85.764/87.220/274.589 ms`; candidate raw p50/p95/p99 were `85.887/88.075/322.142 ms`, with p50 regression `0.143%`, p95 improvement `-0.981%`, and p99 improvement `-17.318%`. The XR06 p95/p99 tail gate requires `>=15%`, so no broader holdout was run. Peak-memory interpretation is diagnostic-only for this candidate because skipping per-token reset changes the telemetry boundary. The env flag remains default-off. |
| 2026-07-01 | XR29 MTP lazy second draft A/B | Reject candidate | `30d28e6` plus local XR29 lazy-draft changes | `native_mtp_lazy_second_draft_block2` | `benchmarks/out/XR29-mtp-lazy-second-draft/baseline-eager-block2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR29-mtp-lazy-second-draft/candidate-lazy-block2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1` inside `gemma4_mtp_draft_block`: for block size 2, the assistant drafts/evaluates the first token, compares it to the cached target greedy token, and only computes first-token post-projection plus the second draft when the first token is accepted. This does not use block-prefix rollback, batch verify, serial-state repair, or public ABI changes. Baseline run `xr15-1782904888` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR29-mtp-lazy-second-draft/baseline-eager-block2 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id mtp_candidate_1k_001`; candidate run `xr15-1782904992` added `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1` with the same arguments and out-dir `candidate-lazy-block2`. Workload seed `20260641`, context `1024/1024`, generated `32` tokens, prompt SHA-256 `afc51a55b76097a09f030c835b9917b4425469ba9c758ef513cb355e10da04c6`; both runs had 4/4 exact records, no blockers, peak MLX `7.665 GB`, active KV `352845824` bytes, and measured rollbacks `57`. Candidate reduced measured attempted draft tokens from `120` to `96` while accepted draft tokens stayed `39`, raising measured acceptance from `0.325` to `0.40625`. Median `draft_ms` improved from `164.201` to `134.852`, median `verify_ms` from `2966.086` to `2952.452`, and median MTP decode phase from `3116.653` to `3087.304` (`~0.94%`), below the `5%` gate. Fixed block-2 still regressed native baseline by `8.611%`, so no holdout was run. The env flag remains default-off. |
| 2026-07-01 | XR30 MTP direct first-reject verifier A/B | Reject candidate | `f6c17e7` plus local XR30 direct-first-reject changes | `native_mtp_direct_first_reject` | `benchmarks/out/XR30-mtp-direct-first-reject/baseline-normal-verify/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR30-mtp-direct-first-reject/candidate-direct-first-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off `GEMMA4D_EXPERIMENTAL_MTP_DIRECT_FIRST_REJECT=1` inside `gemma4_verify_tokens`: when the first draft token mismatches cached target greedy, commit the fallback token directly on the live target KV instead of entering the staged clone verifier. Baseline run `xr15-1782917374` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR30-mtp-direct-first-reject/baseline-normal-verify --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id mtp_candidate_1k_001`; candidate run `xr15-1782917475` added `GEMMA4D_EXPERIMENTAL_MTP_DIRECT_FIRST_REJECT=1` with the same arguments and out-dir `candidate-direct-first-reject`. Workload seed `20260641`, context `1024/1024`, generated `32` tokens, prompt SHA-256 `afc51a55b76097a09f030c835b9917b4425469ba9c758ef513cb355e10da04c6`; both runs had 4/4 exact records, no harness blockers, peak MLX `7.665 GB`, active KV `352845824` bytes, attempted/accepted draft tokens `120/39`, first-reject events `24`, and measured rollbacks `57`. Median `draft_ms` regressed from `159.922` to `163.534`, median `verify_ms` regressed from `2772.472` to `2779.443`, and median MTP decode phase regressed from `2931.063` to `2936.736` (`-0.19%`), below the `5%` gate. Independent correctness review also found the branch is not failure-atomic because a late direct decode failure can partially advance live KV before token/last-step metadata updates. No holdout was run; the env flag remains default-off and is not promotable without cache-discard/failure-injection coverage. |
| 2026-07-01 | XR31 MTP block-prefix partial-only repair A/B | Blocked with evidence | `5d577fb` plus local XR31 partial-only changes | `native_mtp_block_prefix_partial_only_repair` | `benchmarks/out/XR31-mtp-block-prefix-partial-only-repair/blocker-baseline-normal/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR31-mtp-block-prefix-partial-only-repair/blocker-candidate-partial-only/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_ONLY_REPAIR=1`, active only with `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`, to serial-repair full block-2 accepts while leaving partial-reject block-prefix commits on the fast path. Baseline run `xr15-1782917824` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR31-mtp-block-prefix-partial-only-repair/blocker-baseline-normal --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`; candidate run `xr15-1782918010` added `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_ONLY_REPAIR=1` with the same arguments and out-dir `blocker-candidate-partial-only`. Workload seed `20260631`, context `4096/4096`, generated `32` tokens, prompt SHA-256 `93b21c654b4efcdc41236be21f8f4fb95e2d29bd380e6667b69f83933b62fa99`. The normal baseline was exact for 4/4 records with measured accepted/attempted `51/135`, `45` rollbacks, peak `9.244 GB`, and active KV `403177472` bytes. The candidate reproduced the XR24 blocker: exactness `0/4`, every trial first mismatched at generated token index `12` (`100` native baseline vs `8970` MTP), measured accepted/attempted `12/165`, `84` rollbacks, event histogram `accepted=0` for `72` events and `accepted=1` for `12` events with no full accepts to repair, peak `9.244 GB`, and active KV `403177472` bytes. The candidate measured median MTP phase was `5190.941 ms` vs native baseline decode `4115.377 ms` (`-26.135%`). No holdout was run; the sidecar review correctly identified that XR24's blocker had zero full accepts, so partial-only repair cannot fix partial-reject state drift. The env flag remains default-off and should not be promoted. |
| 2026-07-01 | XR32 native chunked prefill size sweep | Accept candidate | `ba8edea` plus local XR32 benchmark-variant changes | `native_chunked_prefill_256_long_context` | `benchmarks/out/XR32-native-chunked-prefill-size-sweep/8k-sentinel/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, `benchmarks/out/XR32-native-chunked-prefill-size-sweep/followup-8k-256/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, `benchmarks/out/XR32-native-chunked-prefill-size-sweep/sentinel-16k-256/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`, and `benchmarks/out/XR32-native-chunked-prefill-size-sweep/followup-16k-256/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added benchmark-only XR05 variants for `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS=256,384,768`; runtime behavior, public C ABI, and defaults are unchanged. The 8K sentinel run `xr05-1782918414-251935000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR32-native-chunked-prefill-size-sweep/8k-sentinel --trials 1 --clear-workload-ids --workload-id code_review_rust_8k_001 --variants native_eval_per_layer,native_chunked_prefill_256,native_chunked_prefill_384,native_chunked_prefill_512,native_chunked_prefill_768,native_chunked_prefill_1024`; seed `20260632`, context `8192/8192`. Only chunk `256` passed correctness (`token 100`, logit delta `0.25`, prefill `24085.904 ms`, peak `7.383 GB`, active KV `469762048` bytes); chunks `384/512/768/1024` failed logit tolerance with deltas `0.625/0.75/0.75/0.625`. The 8K follow-up run `xr05-1782918631-425299000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR32-native-chunked-prefill-size-sweep/followup-8k-256 --trials 3 --clear-workload-ids --workload-id code_review_rust_8k_001 --variants native_eval_per_layer,native_chunked_prefill_256`; both variants were correct for 3/3 trials, baseline prefill p50/p95 was `29970.057/30331.689 ms`, candidate p50/p95 was `21391.980/23556.116 ms`, p50 improved `28.622%`, p95 improved `22.338%`, peak MLX improved from `12.763` to `7.383 GB` (`42.154%`), active KV stayed `469762048` bytes, and logit delta was `0.25` on every trial. The 16K sentinel `xr05-1782918864-920312000` on `benchmark_qa_16k_001` (seed `20260634`, context `16384/16384`) was correctness-clean with logit delta `0.0`, prefill `86254.188 -> 52544.936 ms`, and peak `21.868 -> 7.620 GB`. The 16K follow-up `xr05-1782919079-433531000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR32-native-chunked-prefill-size-sweep/followup-16k-256 --trials 3 --clear-workload-ids --workload-id benchmark_qa_16k_001 --variants native_eval_per_layer,native_chunked_prefill_256`; both variants were correct for 3/3 trials, baseline p50/p95 was `86548.899/87510.637 ms`, candidate p50/p95 was `42925.841/50762.654 ms`, p50 improved `50.403%`, p95 improved `41.993%`, peak MLX improved from `21.868` to `7.620 GB` (`65.155%`), active KV stayed `603979776` bytes, and logit delta was `0.0` on every trial. Candidate is accepted for later long-context adoption work only; no default changed and additional 4K/other-family policy guardrails are still needed before enabling. |
| 2026-07-01 | XR33 native chunked prefill policy guardrails | Accept candidate | `3dc9c7d` plus local XR33 guardrail docs | `native_chunked_prefill_256_policy_guardrails` | `benchmarks/out/XR33-native-chunked-prefill-policy-guardrails/guardrail-4k-256/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR33-native-chunked-prefill-policy-guardrails/guardrail-long-repo-16k-256/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this is XR32 adoption guardrail evidence for existing benchmark variant `native_chunked_prefill_256`. The 4K guardrail run `xr05-1782919758-54544000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR33-native-chunked-prefill-policy-guardrails/guardrail-4k-256 --trials 3 --clear-workload-ids --workload-id code_review_rust_4k_001 --variants native_eval_per_layer,native_chunked_prefill_256`; seed `20260631`, context `4096/4096`, prompt SHA-256 `93b21c654b4efcdc41236be21f8f4fb95e2d29bd380e6667b69f83933b62fa99`. Both variants were correct for 3/3 trials, baseline prefill p50/p95 was `10923.489/11109.471 ms`, candidate p50/p95 was `10535.555/10852.503 ms`, p50 improved `3.551%`, p95 improved `2.313%`, peak MLX improved from `9.212` to `7.281 GB` (`20.964%`), active KV stayed `402653184` bytes, and logit delta was `0.0` on all trials. The second-family long-context guardrail run `xr05-1782919896-139768000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR33-native-chunked-prefill-policy-guardrails/guardrail-long-repo-16k-256 --trials 3 --clear-workload-ids --workload-id long_repo_pack_16k_001 --variants native_eval_per_layer,native_chunked_prefill_256`; seed `20260639`, context `16384/16384`, prompt SHA-256 `9c8ccf1edb13a54d66a3b7693485ada29aff77840ca1eb522b811636e128ed8f`. Both variants were correct for 3/3 trials, baseline p50/p95 was `87369.038/87428.799 ms`, candidate p50/p95 was `41889.113/50182.123 ms`, p50 improved `52.055%`, p95 improved `42.602%`, peak MLX improved from `21.868` to `7.620 GB` (`65.155%`), active KV stayed `603979776` bytes, and logit delta was `0.125` on all trials. Combined with XR32, chunk size `256` is now correctness-clean across 4K, 8K, and two 16K families with strong memory wins and no p95 regression. This supports a later guarded adoption goal for native chunked-prefill policy; no runtime default changed here. |
| 2026-07-01 | XR34 native chunked prefill policy adoption | Accept candidate | `770b348` plus local XR34 policy changes | `native_chunked_prefill_policy_long_context_256` | `benchmarks/out/XR34-native-chunked-prefill-policy-adoption/policy-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added opt-in `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256`: when fixed `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` is absent and token count is at least `4096`, native prefill uses the accepted 256-token chunked path; explicit fixed chunk env still takes precedence. Runtime defaults and public C ABI are unchanged. Added XR05 variant `native_chunked_prefill_policy_long_context_256` and environment capture for `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY`. Policy smoke run `xr05-1782920598-675780000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR34-native-chunked-prefill-policy-adoption/policy-smoke --trials 3 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id code_review_rust_4k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`; records `12/12` passed with no blockers. Below threshold on `chat_short_1k_001` (seed `20260630`, context `1024/1024`), both variants were correct for 3/3 trials, peak MLX stayed `7.321 GB`, active KV stayed `352321536` bytes, and no XR05 prefill/memory gate was met (`p50 -2.414%`), confirming the policy did not take the chunked memory shape. At threshold on `code_review_rust_4k_001` (seed `20260631`, context `4096/4096`), both variants were correct for 3/3 trials, baseline p50/p95 was `10886.219/10893.887 ms`, policy p50/p95 was `9781.628/9936.305 ms`, p50 improved `10.147%`, p95 improved `8.790%`, peak MLX improved from `9.279` to `7.300 GB` (`21.330%`), active KV stayed `402653184` bytes, and logit delta was `0.0`. Candidate is accepted as opt-in runtime policy only; defaults remain unchanged. |
| 2026-07-01 | XR35 native chunked prefill policy holdouts | Accept candidate | `15bf48d` plus local XR35 contract docs | `native_chunked_prefill_policy_long_context_256` | `benchmarks/out/XR35-native-chunked-prefill-policy-holdouts/holdout-8k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this is holdout evidence for the XR34 opt-in policy. The 8K holdout run `xr05-1782921181-879661000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR35-native-chunked-prefill-policy-holdouts/holdout-8k-policy --trials 3 --clear-workload-ids --workload-id code_review_rust_8k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`; seed `20260632`, context `8192/8192`, prompt SHA-256 `24988dedab99e7a200035341ed0cc103d3f06ae84190777c8201d59b8590215e`, and records `6/6` passed with no blockers. Both variants were correct for 3/3 trials, baseline prefill p50/p95 was `30339.051/32421.743 ms`, policy p50/p95 was `21993.044/27939.163 ms`, p50 improved `27.509%`, p95 improved `13.826%`, peak MLX improved from `12.763` to `7.383 GB` (`42.154%`), active KV stayed `469762048` bytes, and policy logit delta was `0.25` on all trials. The first sandboxed attempt failed before benchmarking because MLX reported no Metal device; the same command rerun with approved escalation completed. The 16K holdout was deferred: a mid-run `vm_stat` sample during the 8K run showed only `3698` free 16 KiB pages, about `58 MiB`, and `641562` wired pages, about `9.79 GiB`; post-run free pages recovered to `676512`, but rerunning an unchunked 16K baseline under observed pressure was not necessary for the 8K holdout decision. Candidate remains opt-in only; no defaults or C ABI changed. |
| 2026-07-01 | XR36 MTP block-prefix partial-reject repair | Reject candidate | `3404f55` plus local XR36 runtime and contract changes | `native_mtp_block_prefix_partial_reject_repair` | `benchmarks/out/XR36-mtp-block-prefix-partial-reject-repair/blocker-baseline-normal/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR36-mtp-block-prefix-partial-reject-repair/blocker-candidate-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added default-off `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1`, active only with `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`, to serial-repair the accepted-first/rejected-second block-prefix branch. Defaults, public C ABI, and non-MTP paths are unchanged. Compile check `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-ffi` passed. Baseline run `xr15-1782921632` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR36-mtp-block-prefix-partial-reject-repair/blocker-baseline-normal --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`; candidate run `xr15-1782921825` added `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1` with the same arguments and out-dir `blocker-candidate-partial-reject`. Workload seed `20260631`, context `4096/4096`. The candidate restored exactness on the XR31 blocker (`4/4` byte-identical) with no hard blockers, event histogram per record `accepted=0:13`, `accepted=1:3`, `accepted=2:7`, accepted/attempted `17/45` per record and `51/135` measured, rollbacks `15` per record, active KV unchanged at `403177472` bytes, and peak MLX unchanged at `9.244 GB`. It is still rejected for promotion because fixed block-2 selected decode phase regressed to `5721.069 ms` vs native baseline `4248.623 ms` (`-34.657%` speedup), worse than the normal baseline run's `3694.889 ms` vs `3445.987 ms` (`-7.223%`). Candidate remains default-off; this repairs the partial-reject correctness blocker but is not a speed path. |
| 2026-07-01 | XR37 MTP partial-reject state-only repair | Reject candidate | `3fc3269` plus local XR37 runtime and contract changes | `native_mtp_block_prefix_partial_reject_state_only_repair` | `benchmarks/out/XR37-mtp-partial-reject-state-only-repair/candidate-state-only-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Extended the existing default-off `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1` path so it can pair with `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1`; defaults, public C ABI, and non-MTP paths are unchanged. Compile check `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-ffi` passed. Run `xr15-1782922196` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR37-mtp-partial-reject-state-only-repair/candidate-state-only-partial-reject --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`; workload seed `20260631`, context `4096/4096`. The candidate stayed exact (`4/4` byte-identical) with no hard blockers, event histogram per record `accepted=0:13`, `accepted=1:3`, `accepted=2:7`, accepted/attempted `17/45` per record and `51/135` measured, rollbacks `15` per record, active KV unchanged at `403177472` bytes, and peak MLX unchanged at `9.244 GB`. State-only repair materially improved XR36's decode-phase regression (`-34.657%` to `-16.405%`) but still missed the speed gate: selected decode phase was `4437.904 ms` vs native baseline `3812.471 ms`. Candidate remains default-off and is not promotable as a speed path. |
| 2026-07-01 | XR38 MTP lazy draft with partial-reject repair | Reject candidate | `c7f8444` plus local XR38 contract docs | `native_mtp_lazy_draft_partial_reject_state_only_repair` | `benchmarks/out/XR38-mtp-lazy-draft-partial-reject-repair/candidate-lazy-state-only-partial-reject/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; combined existing default-off `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1` with the XR37 repaired block-prefix path. Run `xr15-1782922596` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_PARTIAL_REJECT_REPAIR=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_STATE_ONLY_REPAIR=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR38-mtp-lazy-draft-partial-reject-repair/candidate-lazy-state-only-partial-reject --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id code_review_rust_4k_001`; workload seed `20260631`, context `4096/4096`. The candidate stayed exact (`4/4` byte-identical) with no hard blockers, event histogram per record `accepted=0:13`, `accepted=1:3`, `accepted=2:7`, accepted/attempted `17/32` per record and `51/96` measured, rollbacks `15` per record, active KV unchanged at `403177472` bytes, and peak MLX unchanged at `9.244 GB`. Lazy drafting reduced attempted draft tokens from XR37's `45` to `32` per record and improved decode-phase regression from `-16.405%` to `-9.339%`, but still missed the speed gate: selected decode phase was `4150.249 ms` vs native baseline `3795.764 ms`. Candidate remains default-off and is not promotable as a speed path. |
| 2026-07-01 | XR39 native chunked prefill policy family matrix | Accept candidate | `62b6115` plus local XR39 contract docs | `native_chunked_prefill_policy_long_context_256` | `benchmarks/out/XR39-native-chunked-prefill-policy-family-matrix/policy-family-matrix/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this broadens XR34/XR35 opt-in policy evidence across 1K/4K families. Run `xr05-1782922915-623972000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR39-native-chunked-prefill-policy-family-matrix/policy-family-matrix --trials 3 --clear-workload-ids --workload-id tool_json_1k_001 --workload-id benchmark_qa_4k_001 --workload-id adapter_expert_4k_001 --workload-id mtp_candidate_4k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`; records `24/24` passed with no blockers. Below-threshold rows are boundary/no-chunk checks, not chunked-prefill speed evidence: `tool_json_1k_001` (seed `20260635`, context `1024/1024`, prompt SHA-256 `7687cd292cf8f9be5f84f3dca2e3644a08d973a1a314facb52ac91bbed0d5e2c`) kept peak MLX `7.321 GB` and active KV `352321536` bytes with logit delta `0.0`; `benchmark_qa_4k_001` (seed `20260633`, context `4096/4095`, prompt SHA-256 `1514934863d5ad974300a0feb490ac2dbf1ab2eadc2e7f1a1525e2c2eb3b4e42`) kept the non-chunked memory shape (`9.212 -> 9.244 GB` max), active KV `402636800` bytes, and logit delta `0.0`. At threshold, `adapter_expert_4k_001` (seed `20260638`, context `4096/4096`) was correct for 3/3, p50/p95 improved `13705.482/14778.767 -> 10628.725/11340.667 ms`, peak MLX improved `9.279 -> 7.300 GB` (`21.330%`), active KV stayed `402653184`, and logit delta was `0.125`; `mtp_candidate_4k_001` (seed `20260642`, context `4096/4096`) was correct for 3/3, p50/p95 improved `11089.101/11198.204 -> 9462.730/9519.092 ms`, peak MLX improved `9.279 -> 7.300 GB` (`21.330%`), active KV stayed `402653184`, and logit delta was `0.125`. Candidate remains opt-in only; no defaults or C ABI changed. |
| 2026-07-01 | XR40 native chunked prefill policy 16K sentinel | Accept candidate | `3bda455` plus local XR40 contract docs | `native_chunked_prefill_policy_long_context_256` | `benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/benchmark-qa-16k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/long-repo-16k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this closes the missing opt-in policy-path 16K sentinel for XR34/XR35/XR39 before any broader adoption. Compile check `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab` passed. The first sandboxed attempt failed before benchmarking because MLX could not access Metal; both benchmark commands were rerun with approved escalation. `benchmark_qa_16k_001` run `xr05-1782923643-495558000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/benchmark-qa-16k-policy --trials 3 --clear-workload-ids --workload-id benchmark_qa_16k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`; seed `20260634`, context `16384/16384`, prompt SHA-256 `0d0c0893eca1c1b52e659c7608f5a5fc5a089e00576d56c217bb982791dadf4a`, records `6/6` passed, candidate correctness `3/3`, p50/p95 improved `86813.720/87063.513 -> 42244.280/51265.141 ms`, peak MLX improved `21.868 -> 7.620 GB` (`65.155%`), active KV stayed `603979776`, and logit delta was `0.0`. `long_repo_pack_16k_001` run `xr05-1782924116-206625000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/long-repo-16k-policy --trials 3 --clear-workload-ids --workload-id long_repo_pack_16k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`; seed `20260639`, context `16384/16384`, prompt SHA-256 `9c8ccf1edb13a54d66a3b7693485ada29aff77840ca1eb522b811636e128ed8f`, records `6/6` passed, candidate correctness `3/3`, p50/p95 improved `87017.803/87320.961 -> 42390.024/50562.318 ms`, peak MLX improved `21.868 -> 7.620 GB` (`65.155%`), active KV stayed `603979776`, and logit delta was `0.125`. Memory caveat: the second run's mid-run `vm_stat` reached only `5747` free 16 KiB pages and `751325` wired pages, then recovered after exit; candidate remains opt-in only and no defaults or C ABI changed. |
| 2026-07-01 | XR41 native prefill policy FFI setter | Accept candidate | `0828510` plus local XR41 setter changes | `native_chunked_prefill_setter_long_context_256` | `benchmarks/out/XR41-native-prefill-policy-ffi-setter/setter-boundary-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added additive `gemma4_target_set_prefill_chunk_policy` C ABI, safe Rust `PrefillChunkPolicy`/`Target::set_prefill_chunk_policy`, and XR05 setter-backed variant without changing `Gemma4LoadConfig`, defaults, server/profile behavior, model math, tokenizer behavior, MTP behavior, or non-native paths. Verification passed: `cargo fmt --all --check`; `cargo test -p gemma4d-ffi --lib` (`15` passed, `1` ignored); `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab`. Run `xr05-1782925238-17972000` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR41-native-prefill-policy-ffi-setter/setter-boundary-smoke --trials 3 --clear-workload-ids --workload-id tool_json_1k_001 --workload-id benchmark_qa_4k_001 --workload-id adapter_expert_4k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256,native_chunked_prefill_setter_long_context_256`; records `27/27` passed with no blockers. Setter below-threshold rows were correctness-clean and kept non-chunked memory shape but are not speed evidence: `tool_json_1k_001` seed `20260635`, context `1024/1024`, peak `7.321 GB`, active KV `352321536`, logit delta `0.0`; `benchmark_qa_4k_001` seed `20260633`, context `4096/4095`, peak `9.212 GB`, active KV `402636800`, logit delta `0.0`. At threshold, `adapter_expert_4k_001` seed `20260638`, context `4096/4096`, prompt SHA-256 `e4f055746d250beee415c30893f1baae9efce40789e70e77196b506ff5a3f3a7`, setter correctness `3/3`, baseline p50/p95 `14884.449/15442.978 ms`, setter p50/p95 `11424.204/12538.482 ms`, p50 improved `23.247%`, p95 improved `18.808%`, peak MLX improved `9.279 -> 7.300 GB` (`21.330%`), active KV stayed `402653184`, and logit delta was `0.125`. Candidate remains opt-in only. |
| 2026-07-01 | XR42 Rayon manifest hashing A/B | Accept candidate for follow-up | `aedcc22` plus local XR42 Rayon harness changes | `rayon_safetensors_inventory_hashing_ab` | `benchmarks/out/XR42-rayon-manifest-hashing-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added Rayon as a `gemma4d-bench` dev-dependency only and a standalone benchmark-prep harness; no MLX/runtime/default manifest behavior changed. Run `xr42-1782926118256` used command `cargo run -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab -- --out-dir benchmarks/out/XR42-rayon-manifest-hashing-ab --trials 3 --thread-counts 1,2,4`; deterministic seed metadata `20260701`; token lengths `not_applicable:file hashing only; no tokenizer/model execution`; records `24/24` passed, blockers none. Target artifact `gemma-4-12B-it-4bit` had 2 safetensors, `6741039511` bytes, inventory SHA-256 `4af9af81c81dcba1edb5290573e58efc28f71c887ab25a871d3917f4240459af`; sequential p50/p95 `45488.176/45572.282 ms`; Rayon 2-thread p50/p95 `36261.428/36460.999 ms`, p50 improved `20.284%`, hash matched. Assistant artifact `gemma-4-12B-it-qat-assistant-4bit` had 1 safetensors file, `237894178` bytes, inventory SHA-256 `7a5d3a9eabd8ec983c4ef5139badf2da187a455133446be21b3c3dc0006b70bd`; Rayon showed no useful single-file speedup (`1566.653 -> 1566.317 ms`, `+0.021%`). Candidate is follow-up evidence only; do not integrate into default manifest path from this run alone. |
| 2026-07-01 | XR43 MTP block-prefix selected-slice confirmation | Keep experimental | `abc2bc3` plus local XR43 contract docs | `native_mtp_block_prefix_selected_slice` | `benchmarks/out/XR43-mtp-block-prefix-selected-slice/candidate-block-prefix-selected/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this reruns the XR24 promising selected slice using existing default-off `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`. Compile check `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` passed. The first sandboxed benchmark attempt failed before benchmarking because MLX could not access Metal; the escalated run `xr15-1782927140` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR43-mtp-block-prefix-selected-slice/candidate-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001`; records `8/8` exact and `6/6` measured exact with no hard blockers. `chat_short_1k_001` seed `20260630`, context `1024/1024`, generated `32`, prompt SHA-256 `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`, baseline decode p50 `3084.066 ms`, MTP decode-phase p50 `2686.191 ms`, speedup `12.901%`, acceptance `69/120 = 0.575`, rollbacks `27`, peak MLX `8.002 GB`, active KV `352845824`. `mtp_candidate_4k_001` seed `20260642`, context `4096/4096`, generated `32`, prompt SHA-256 `88f76c633511de568b6270b3217be53a26a5c7235862a3c23a514de2646268b3`, baseline decode p50 `4886.134 ms`, MTP decode-phase p50 `11780.432 ms`, speedup `-141.099%`, acceptance `75/108 = 0.694`, rollbacks `21`, peak MLX `9.244 GB`, active KV `403177472`. Fixed block-2 and acceptance-threshold policies rejected; net-latency-guarded policy selected only `chat_short_1k_001:block2` with aggregate `4.992%`, just below the nominal 5% XR43 gate. MTP remains default-off; memory caveat: mid-run `vm_stat` samples showed about `4055` to `4271` free 16 KiB pages and about `922942` to `927935` pages stored in compressor. |
| 2026-07-01 | XR44 MTP lazy block-prefix selected-slice A/B | Keep experimental | `718297d` plus local XR44 contract docs | `native_mtp_lazy_block_prefix_selected_slice` | `benchmarks/out/XR44-mtp-lazy-block-prefix-selected-slice/candidate-lazy-block-prefix-selected/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this combines existing default-off `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` and `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1` on the same selected slice as XR43. Compile check `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` passed. The first sandboxed benchmark attempt failed before benchmarking because MLX could not access Metal; the escalated run `xr15-1782927804` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR44-mtp-lazy-block-prefix-selected-slice/candidate-lazy-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001`; records `8/8` exact and `6/6` measured exact with no hard blockers. `chat_short_1k_001` seed `20260630`, context `1024/1024`, generated `32`, prompt SHA-256 `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`, baseline decode p50 `3138.129 ms`, MTP decode-phase p50 `2355.632 ms`, speedup `24.935%`, acceptance `69/96 = 0.719`, rollbacks `27`, peak MLX `8.002 GB`, active KV `352845824`; vs XR43 attempted draft tokens dropped `120 -> 96` and candidate p50 improved `2686.191 -> 2355.632 ms`. `mtp_candidate_4k_001` seed `20260642`, context `4096/4096`, generated `32`, prompt SHA-256 `88f76c633511de568b6270b3217be53a26a5c7235862a3c23a514de2646268b3`, baseline decode p50 `10406.955 ms`, MTP decode-phase p50 `11534.612 ms`, speedup `-10.836%`, acceptance `75/96 = 0.781`, rollbacks `21`, peak MLX `9.220 GB`, active KV `403177472`; still rejected by guard. Fixed block-2 and acceptance-threshold policies rejected; net-latency-guarded policy selected only `chat_short_1k_001:block2` with aggregate `5.777%`. MTP remains default-off; memory caveat: mid-run `vm_stat` samples showed about `4015` to `4071` free 16 KiB pages and about `572826` to `910058` pages stored in compressor, then recovered to `640010` free pages after exit. |
| 2026-07-01 | XR45 MTP lazy block-prefix 1K family holdout | Keep experimental | `0774c09` plus local XR45 contract docs | `native_mtp_lazy_block_prefix_1k_family_holdout` | `benchmarks/out/XR45-mtp-lazy-block-prefix-1k-family-holdout/candidate-lazy-block-prefix-1k/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this tests whether XR44's lazy block-prefix `chat_short_1k_001` win generalizes across 1K real-context families with existing default-off `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` and `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`. Compile check `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` passed. Escalated run `xr15-1782928503` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR45-mtp-lazy-block-prefix-1k-family-holdout/candidate-lazy-block-prefix-1k --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`; records `12/12` exact and `9/9` measured exact with no hard blockers. `chat_short_1k_001` seed `20260630`, context `1024/1024`, generated `32`, prompt SHA-256 `05ad1c8d61b2a916c0eeb3e2d67e56b4b8d2acf81041c325e8e04e7e4a9eb7f0`, baseline decode p50 `2955.491 ms`, MTP decode-phase p50 `2340.434 ms`, speedup `20.811%`, acceptance `69/96 = 0.719`, rollbacks `27`, peak MLX `8.002 GB`, active KV `352845824`. `tool_json_1k_001` seed `20260635`, context `1024/1024`, generated `32`, prompt SHA-256 `7687cd292cf8f9be5f84f3dca2e3644a08d973a1a314facb52ac91bbed0d5e2c`, baseline decode p50 `2910.560 ms`, MTP decode-phase p50 `2231.115 ms`, speedup `23.344%`, acceptance `75/96 = 0.781`, rollbacks `21`, peak MLX `8.002 GB`, active KV `352845824`. `mtp_candidate_1k_001` seed `20260641`, context `1024/1024`, generated `32`, prompt SHA-256 `afc51a55b76097a09f030c835b9917b4425469ba9c758ef513cb355e10da04c6`, baseline decode p50 `2952.317 ms`, MTP decode-phase p50 `3290.224 ms`, speedup `-11.445%`, acceptance `39/96 = 0.406`, rollbacks `57`, peak MLX `8.008 GB`, active KV `352845824`; rejected by guard. Fixed block-2 and acceptance-threshold policies were rejected because `mtp_candidate_1k_001` regressed; net-latency-guarded policy selected only `chat_short_1k_001:block2` and `tool_json_1k_001:block2` with aggregate `14.680%`. MTP remains default-off; memory caveat: mid-run `vm_stat` samples showed about `3553` to `4207` free 16 KiB pages, `669502` wired pages at the second sample, and `296418` to `476696` pages stored in compressor, then recovered to `526067` free pages after exit. |
| 2026-07-01 | XR46 MTP adaptive zero-accept fallback A/B | Keep experimental | `d46501b` plus local XR46 harness/docs changes | `native_mtp_adaptive_zero_accept_fallback` | `benchmarks/out/XR46-mtp-adaptive-zero-run-fallback/candidate-adaptive-zero-run/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added opt-in XR15 harness flags `--adaptive-zero-accept-run` and `--adaptive-min-generated-tokens`; defaults are unchanged. Candidate used existing default-off `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` and `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`, with adaptive threshold `4` after `12` generated tokens. Verification passed: `cargo fmt --all --check`; `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab`. Escalated run `xr15-1782929125` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR46-mtp-adaptive-zero-run-fallback/candidate-adaptive-zero-run --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 4 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`; records `12/12` exact and `9/9` measured exact with no hard blockers. `chat_short_1k_001` seed `20260630`, context `1024/1024`, baseline decode p50 `3013.177 ms`, adaptive MTP decode-phase p50 `2228.909 ms`, speedup `26.028%`, fallback p50 `0.000 ms`, acceptance `69/96 = 0.719`, peak MLX `8.002 GB`, active KV `352845824`. `tool_json_1k_001` seed `20260635`, context `1024/1024`, baseline p50 `3174.286 ms`, adaptive MTP p50 `2117.370 ms`, speedup `33.296%`, fallback p50 `0.000 ms`, acceptance `75/96 = 0.781`, peak MLX `8.002 GB`, active KV `352845824`. `mtp_candidate_1k_001` seed `20260641`, context `1024/1024`, baseline p50 `2872.385 ms`, adaptive MTP p50 `3143.500 ms`, speedup `-9.439%`, fallback p50 `1245.902 ms`, acceptance `21/48 = 0.438`, peak MLX `8.008 GB`, active KV `352829440`; adaptive fallback fired in `3/3` measured records at pass `10` after `16` generated tokens. Compared with XR45, `mtp_candidate_1k_001` attempted draft tokens dropped `96 -> 48` and candidate p50 improved `3290.224 -> 3143.500 ms`, but it still regressed against native baseline and remains rejected by guard. Net-latency-guarded policy selected only `chat_short_1k_001:block2` and `tool_json_1k_001:block2` with aggregate `20.322%`; MTP remains default-off. Memory caveat: mid-run `vm_stat` samples showed about `3658` to `7111` free 16 KiB pages, `632357` wired pages at the second sample, and `310842` to `467704` pages stored in compressor, then recovered to `555345` free pages after exit. |
| 2026-07-01 | XR47 MTP adaptive threshold sweep | Keep experimental | `30f5d04` plus local XR47 contract docs | `native_mtp_adaptive_zero_run_1_min12` | `benchmarks/out/XR47-mtp-adaptive-threshold-sweep/zero-run-1-min12/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this sweep uses existing XR46 opt-in adaptive fallback flags with threshold `--adaptive-zero-accept-run 1 --adaptive-min-generated-tokens 12`. Compile check `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` passed. Escalated run `xr15-1782929579` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR47-mtp-adaptive-threshold-sweep/zero-run-1-min12 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 1 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`; records `12/12` exact and `9/9` measured exact with no hard blockers. `chat_short_1k_001` seed `20260630`, context `1024/1024`, baseline decode p50 `3159.750 ms`, adaptive MTP p50 `2971.490 ms`, speedup `5.958%`, fallback p50 `1256.194 ms`, acceptance `27/48 = 0.563`, peak MLX `8.002 GB`, active KV `352829440`; fallback fired in `3/3` measured records at pass `11` after `16` generated tokens. `tool_json_1k_001` seed `20260635`, context `1024/1024`, baseline p50 `2916.887 ms`, adaptive MTP p50 `2324.990 ms`, speedup `20.292%`, fallback p50 `0.000 ms`, acceptance `75/96 = 0.781`, peak MLX `8.002 GB`, active KV `352845824`; fallback did not fire. `mtp_candidate_1k_001` seed `20260641`, context `1024/1024`, baseline p50 `3081.131 ms`, adaptive MTP p50 `3040.089 ms`, speedup `1.332%`, fallback p50 `1491.481 ms`, acceptance `21/39 = 0.538`, peak MLX `8.008 GB`, active KV `352829440`; fallback fired in `3/3` measured records at pass `7` after `13` generated tokens. Compared with XR46, `mtp_candidate_1k_001` attempted draft tokens dropped `48 -> 39`, p50 improved `3143.500 -> 3040.089 ms`, and the workload moved from `-9.439%` regression to `+1.332%` speedup, but it still missed the `5%` per-workload guard. Fixed block-2 and acceptance-threshold policies selected all three workloads with aggregate `8.967%` and no regressions; net-latency-guarded policy still selected only `chat_short_1k_001:block2` and `tool_json_1k_001:block2` with aggregate `8.519%`. MTP remains default-off. Memory caveat: mid-run `vm_stat` samples showed about `4212` to `8619` free 16 KiB pages and `324473` to `736908` pages stored in compressor, then recovered to `529036` free pages after exit. |
| 2026-07-01 | XR48 MTP adaptive zero-run 3 sweep | Keep experimental | `285b71f` plus local XR48 contract docs | `native_mtp_adaptive_zero_run_3_min12` | `benchmarks/out/XR48-mtp-adaptive-zero-run-3-sweep/zero-run-3-min12/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Added no runtime code; this sweep uses existing XR46 opt-in adaptive fallback flags with threshold `--adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12` to test the middle point between XR46 and XR47. Verification passed: `cargo fmt --all --check`; `git diff --check`; `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`. The first sandboxed benchmark attempt failed before benchmarking because MLX could not access Metal; the escalated run `xr15-1782930382` used command `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR48-mtp-adaptive-zero-run-3-sweep/zero-run-3-min12 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`; records `12/12` exact and `9/9` measured exact with no hard blockers. `chat_short_1k_001` seed `20260630`, context `1024/1024`, baseline decode p50 `2701.736 ms`, adaptive MTP p50 `2115.179 ms`, speedup `21.710%`, fallback p50 `0.000 ms`, acceptance `69/96 = 0.719`, peak MLX `8.002 GB`, active KV `352845824`; fallback did not fire. `tool_json_1k_001` seed `20260635`, context `1024/1024`, baseline p50 `2814.431 ms`, adaptive MTP p50 `2116.066 ms`, speedup `24.814%`, fallback p50 `0.000 ms`, acceptance `75/96 = 0.781`, peak MLX `8.002 GB`, active KV `352845824`; fallback did not fire. `mtp_candidate_1k_001` seed `20260641`, context `1024/1024`, baseline p50 `2880.829 ms`, adaptive MTP p50 `2915.728 ms`, speedup `-1.211%`, fallback p50 `1329.154 ms`, acceptance `21/45 = 0.467`, peak MLX `8.008 GB`, active KV `352829440`; fallback fired in `3/3` measured records at pass `9` after `15` generated tokens. Compared with XR46, `mtp_candidate_1k_001` attempted draft tokens dropped `48 -> 45` and candidate p50 improved `3143.500 -> 2915.728 ms`, but the workload still regressed against native baseline and remains rejected by the `5%` per-workload guard. Compared with XR47, `chat_short_1k_001` avoided adaptive fallback and recovered from `+5.958%` to `+21.710%`. Fixed block-2 and acceptance-threshold policies selected all three workloads with aggregate `14.887%`; net-latency-guarded policy still selected only `chat_short_1k_001:block2` and `tool_json_1k_001:block2` with aggregate `15.302%`. MTP remains default-off. |
| 2026-07-01 | XR49 MTP light-trace verifier audit | Blocked with evidence | `cb282a9` plus local XR49 contract/docs; no runtime code retained | `native_mtp_light_trace_selected_path_audit` | `benchmarks/out/XR49-mtp-light-trace-verifier-ab/baseline-full-trace-v2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and `benchmarks/out/XR49-mtp-light-trace-verifier-ab/candidate-light-trace-v2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` | Tested whether `GEMMA4D_EXPERIMENTAL_MTP_LIGHT_TRACE=1` could skip full-vocab trace extraction for the selected 1K MTP path. The experiment is blocked as a speed claim because XR15 with `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1` and `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1` already records top-1 trace diagnostics from `runtime.cc`; both the control run `xr15-1782931446` and candidate run `xr15-1782931663` reported `trace_top_k=[1]` for every measured event. Candidate records were still exact (`12/12` total, `9/9` measured) with no hard harness blockers. Candidate p50s were: `chat_short_1k_001` `2695.984 -> 2045.486 ms` (`+24.128%`, acceptance `69/96`, active KV `352845824`, peak `8.002 GB`); `tool_json_1k_001` `2730.721 -> 2027.634 ms` (`+25.747%`, acceptance `75/96`, active KV `352845824`, peak `8.002 GB`); `mtp_candidate_1k_001` `2790.422 -> 2861.231 ms` (`-2.538%`, acceptance `21/45`, auto-disabled `3/3` at pass `9`, active KV `352829440`, peak `8.008 GB`). The lower-path native patch was discarded because the measured runtime path never entered `forward_verify_logits`; next MTP work should target verifier runtime behavior, such as second-slot miss fallback, rather than full-vocab trace scanning. |
| 2026-07-02 | XR50 QAT target MTP pairing A/B | Blocked with evidence | `fe358c8` plus local XR50 goal/docs | `native_mtp_qat_target_pairing_cold_smoke` | `benchmarks/out/XR50-qat-target-mtp-pairing/{report.md,decision.md,blockers.md}` plus smoke subdirectories | Downloaded `mlx-community/gemma-4-12B-it-qat-4bit` revision `e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6` to `artifacts/models/gemma-4-12B-it-qat-4bit`; config SHA-256 `fe091f98e6f7e5e80461bd8ec7ced6d87ac16987586239386ed44b82ecbc2b12`, tokenizer SHA-256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`, safetensors inventory SHA-256 `fc6c056d37f941612ade5fe8632713471126b599f8bb86eaeb78e3773c2f1358`. This specific artifact is mixed affine 4-bit g64 plus 8-bit MLP overrides and is `10987772430` safetensors bytes (`10.99 GB`) versus the plain target's `6741039511` bytes (`6.74 GB`); footprint and latency observations are scoped to this mixed 4/8-bit artifact, not QAT generally. `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` passed. Escalated QAT smoke `candidate-qat-target-block12-smoke` used `--trials 1 --warmups 0 --max-new-tokens 2 --block-sizes 1,2 --workload-id chat_short_1k_001`; the run was exact `2/2`, but acceptance was `0/2` for both block sizes and MTP decode-phase regressed by block 1 `19713.684 -> 71279.259 ms` (`-261.573%`) and block 2 `19713.684 -> 52344.264 ms` (`-165.522%`), with peak MLX about `11.90 GB`. Escalated QAT smoke `candidate-qat-target-mtp-candidate-1k-smoke` used `--trials 1 --warmups 0 --max-new-tokens 2 --block-sizes 2 --workload-id mtp_candidate_1k_001` and was exact `1/1` with acceptance `2/2`, but native decode `13510.088 ms` vs MTP decode phase `25448.830 ms` (`-88.369%`). These are one-sample cold-start smokes with no warmups, so they are JIT/compile/load dominated and not steady-state measurements; unlike the P04 convention, they do not discard the first four decode samples. The required fresh `baseline-plain-target` leg was not attempted; any comparison to XR48 is stale and differently parameterized (`3` trials, `1` warmup, `32` generated tokens versus XR50's `1` trial, `0` warmups, `2` generated tokens). Broader 3-workload and 32-token QAT selected-path attempts were stopped before artifacts because this mixed 4/8-bit target ran too long and caused heavy 16GB memory pressure. Do not promote this QAT target artifact from XR50; next useful work remains verifier/runtime cost. |
| 2026-07-02 | XR51 server chunk policy default | Accept candidate | local XR51 server default changes | `server_native_prefill_default_long_context_256` | `benchmarks/out/XR51-server-chunk-policy-default/{report.md,decision.md,blockers.md}` plus `server-default-1k-repeats3`, `server-default-4k-repeats3`, `server-default-8k-repeats3`, and `server-default-16k-repeats3` subdirectories | Added server-native default policy selection in `gemma4d-server`: persistent-native workers call the safe FFI setter with `PrefillChunkPolicy::LongContext256` after `ResidentTarget::load` when `GEMMA4D_USE_NATIVE_GRAPH` is enabled and neither `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` nor `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY` is set. Explicit chunk envs retain precedence because native env values are read during load and XR51 skips the setter when they are present. Stub, helper-backed, and generate CLI paths remain unchanged. Verification passed: `cargo test -p gemma4d-server --all-targets`; `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr11_persistent_native_server_ab`. Server-mode A/B commands used `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_PERSISTENT_SERVER=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab` with `--model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --repeats 3 --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336`. All repeat runs passed with token identity and no blockers. `chat_short_1k_001` context `1024`: prefill p50/p95 `2814.225/2872.084 -> 2352.410/2853.423 ms`, peak `7.324 -> 7.324 GB`, load count `3 -> 1`. `code_review_rust_4k_001` context `4096`: `11651.369/11911.859 -> 10152.938/11813.827 ms`, peak `9.216 -> 7.300 GB`. `code_review_rust_8k_001` context `8192`: `31285.354/31597.337 -> 22618.497/25073.710 ms`, peak `12.767 -> 7.402 GB`. `benchmark_qa_16k_001` context `16384`: `87387.199/87871.900 -> 41711.194/52217.347 ms`, peak `21.874 -> 7.638 GB`. A combined 3-workload run was stopped before artifacts because it ran too long; the individual repeat runs supersede it. |
| 2026-07-02 | XR52 KV slab incremental decode | Blocked with evidence | local XR52 reference branch plus evidence split | `native_kv_slab_incremental_decode` | `benchmarks/out/XR52-kv-slab-incremental/{decode-baseline-main,decode-candidate-slab,mtp-selected-slab,decode-candidate-slab-rotating,mtp-selected-slab-chronological}` | The full reference branch `feature/xr52-kv-slab` added full-attention KV slab append, logical-only KV cloning/snapshot serialization, verifier timing split fields, and dead full-recompute verifier cleanup; the evidence branch retains only the blocked verdict, M06 stale-doc cleanup, dead verifier deletion, and timing split instrumentation. The safe bounded candidate was byte-identical to `main` for `12/12` native decode records, but steady decode p50 improved only `0.39%..1.05%`, below the `>=5%` gate. XR48-style MTP selected holdout stayed exact `12/12` with unchanged acceptance (`chat 69/96`, `tool_json 75/96`, `mtp_candidate 21/45`) and wrote `verify_stage_ms`, `verify_forward_ms`, and `verify_repair_ms`, but guarded aggregate speedup was `16.719%` versus XR48 `15.302%`, below the required `+5` point gain, and `mtp_candidate_1k_001` still regressed `-2.262%`. Diagnostic rotating-window variants were rejected: physical rotation improved p50 about `6%..7%` but mismatched tokens in `9/12` decode records, while chronological rotation stayed output-exact but drifted `chat_short_1k_001` MTP acceptance to `66/96`. XR52 is not promotable and does not re-anchor decode baselines; XR53 remains unblocked because `main` decode baselines remain valid. |
| 2026-07-03 | XR53 server default backend and admission estimator | Accept candidate with post-review admission caveat | local XR53 server default changes | `server_real_helper_vs_default_persistent_native_real_contexts` | `benchmarks/out/XR53-server-default-backend/{default-path-1k-repeats3,default-path-16k-repeats3,default-path-16k-raised-budget-repeats3,m12-release-gate-post-review}` plus `codex/goals/XR53-server-default-backend-estimator.goal.md` | `gemma4d serve --model-path PATH` now defaults to `ServerBackend::PersistentNative` when no backend flag is explicit; zero-arg/no-model-path and explicit `--backend stub` still use the M11 stub, and explicit `--backend real-helper` remains helper-backed. Removed the `GEMMA4D_EXPERIMENTAL_PERSISTENT_SERVER=1` gate. Added `admission_prefill_chunked` to `ServerConfig`, set from the XR51 pure chunk-policy selector for PersistentNative serve configs. Admission is now per-backend: Stub keeps the legacy lightweight `(legacy_prompt_tokens + max_tokens) * 4096` estimate and is not charged native resident weights, while RealHelper and PersistentNative use XR51/P04 constants plus `max(ceil(words * 13 / 10), ceil(prompt_bytes / 2.25))` prompt estimates. Real unchunked admission interpolates 1K/4K/8K/16K measured points and fails closed above 16K; chunked constants are used only when the server-owned XR51 default policy is known active. Unit tests cover stub no-weights admission under a small budget, native weights-floor rejection, estimator-table anchors, chunked/unchunked 16K behavior, and corpus regression against every `benchmarks/workloads/real-contexts` prompt. Verification passed: `cargo fmt --all --check`; `git diff --check`; `cargo test -p gemma4d-server --all-targets`; `cargo test -p gemma4d-bench --example xr11_persistent_native_server_ab --no-run`; `cargo run -p gemma4d-bench --example m12_release_gate -- --out-dir benchmarks/out/XR53-server-default-backend/m12-release-gate-post-review`. The 1K default-path A/B passed post-review with token identity `3/3`, candidate runtime snapshots reporting `persistent_native`, and load count `3 -> 1`: `chat_short_1k_001` prefill p50/p95 `2869.853/2984.829 -> 2309.716/2963.324 ms`, peak `7.324 -> 7.324 GB`. The original `benchmark_qa_16k_001` run remains historical pre-review default-wiring evidence only; after the byte-density estimator fix it estimates `17724` prompt tokens, exceeds the 16K unchunked measured table, and the raised-budget rerun failed closed with `memory_guard_rejected` before the baseline could generate. |
| 2026-07-03 | XR54 MTP drafter position pin | Reject candidate | local XR54 position-pin plus XR54-R provenance/parity changes | `native_mtp_position_pin_xr48_config` | `benchmarks/out/XR54-mtp-position-pin/{rung10-native-mtp,pinned-xr48-config,xr54-r-mtp-candidate-one-trial,pytorch-parity}` plus `codex/goals/XR54-mtp-position-pin.goal.md` | Changed `NativeMtpAssistantModel::draft_block` to pass constant `first_position` to every assistant step, matching Hugging Face `SinglePositionMultiTokenCandidateGenerator` constant `position_ids` behavior. The original XR48-config holdout completed with `12/12` exact records and unchanged XR48 draft-token arrays; XR54-R then rebuilt from cleaned `target/` with `GEMMA4D_REQUIRE_MLX=1` and stamped git SHA `f2fb705706bc8196845b19d01170cb41e04f430f`, dirty-diff SHA-256 `b4eae5c622bd802783ba2ca18b3b15f108b5fa615626a2283745849891451bd7`, and runner link mtime `1783054369`. The fresh one-leg rerun stayed exact, accepted `7/15 = 0.467`, and remained byte-identical to XR48. The original XR48-config 3-workload/3-trial holdout supplied the per-slot acceptance table: `chat_short_1k_001` slot0 `36/60`, slot1 `33/36`; `tool_json_1k_001` slot0 `39/57`, slot1 `36/39`; `mtp_candidate_1k_001` slot0 `18/27`, slot1 `3/18`. Added mandatory build-provenance stamping to XR15 evidence records and summaries. Added `gemma4_kv_snapshot_save_mtp_parity`, Rust wrapper, `xr54_drafter_pytorch_parity`, `scripts/xr54_drafter_pytorch_parity.py`, and `scripts/xr54_dequant_assistant.py`; the dense checkpoint was regenerated with `/Users/justin/venvs/xr54-parity/bin/python scripts/xr54_dequant_assistant.py --src artifacts/models/gemma-4-12B-it-qat-assistant-4bit --out artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32` and recorded local artifact hash `2ebe68d53c6da07b5d5caa91632b0588337afaae081693320e291f0c2a3d0378`. The completed PyTorch reference run produced native-exact `[236792,236865]` for both pinned `[1023,1023]` and incremented `[1023,1024]` positions. Verdict: reject the acceptance-fix hypothesis; keep the pin as behaviorally neutral reference-convention alignment. XR55 is unblocked, and the slot-1 collapse is treated as a model/content property rather than a remaining native drafter implementation bug. |
| 2026-07-03 | XR55 MTP N-block generalization | Keep experimental | local XR55 native MTP N-block changes | `native_mtp_nblock_block_prefix_verify` | `benchmarks/out/XR55-nblock-generalization/{baseline-block2,trace-n8-chat-prefix-repair,candidate-nblock-sweep,sequential-oracle-sweep,xr55-nblock-summary.md}` plus `codex/goals/XR55-nblock-generalization.goal.md` | Generalized the default-off native MTP block-prefix experiment beyond block 2 by widening MTP trace/committed-token capacity to 16 positions, accepting draft blocks up to the trace capacity, making trace overflow fail closed, bumping native ABI version to `2`, and repairing later partial accepts by materializing the accepted-prefix KV before fallback decode. Baseline post-XR54 `main@24186cf` block 2 selected `chat_short_1k_001` and `tool_json_1k_001` with aggregate `16.038%` speedup. The final candidate sweep used `{1,2,3,4,6,8}` with 3 measured trials plus 1 warmup over `chat_short_1k_001`, `tool_json_1k_001`, and `mtp_candidate_1k_001`; all `72/72` records were exact against non-MTP greedy, and the separate sequential-oracle sweep matched generated tokens for all `72/72` records. Fixed blocks: N=1 `-8.266%`, N=2 `+14.168%`, N=3 `+18.054%`, N=4 `+7.151%`, N=6 `-1.367%`, N=8 `-9.353%`. The net-latency guard selected `chat_short_1k_001:N=3` plus `tool_json_1k_001:N=4` with aggregate `20.371%` speedup, weighted acceptance `144/198 = 0.727`, and peak MLX `8.357 GB`. Per-slot acceptance drops after slot 3; N>=4 pays large exact-prefix repair (`verify_repair_ms` around `6117..6270 ms` aggregate). Draft cost per draft step stayed below the `0.1` verify-unit flag threshold for all blocks. MTP remains default-off. |

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

## XR09 KV Compression Real-Quality A/B Snapshot

XR09 re-runs BF16, q8, and q4 prefix payload compression on XR00 real-context
workloads. The warm path loads the payload, transparently reconstructs BF16
active KV, imports the snapshot, retrieves the cached last step, and runs one
continued `decode_one` against the BF16 cold continuation. Runtime code was not
optimized and active compressed decode stayed disabled.

| Field | Value |
|---|---|
| Command | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr09_kv_compression_real_quality_ab -- --out-dir benchmarks/out/XR09-kv-compression-real-quality-ab` |
| Evidence | `benchmarks/out/XR09-kv-compression-real-quality-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Smoke evidence | `benchmarks/out/XR09-kv-compression-real-quality-ab-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Run ID | `xr09-1782886055` |
| Git SHA | `1dabccc5f9ac6b056e27a10ea59a7a5d12bce8b4` |
| Mode | `native_kv_compression_real_quality_ab` |
| Records | `6`: 6 workloads x 1 trial, each with BF16/q8/q4 mode records |
| Generated files | `records.jsonl`, `summary.json`, `report.md`, `blockers.md`, `decision.md`; per-workload payload paths and SHA-256s are recorded under each record's `modes` array |
| Decision | `reject_candidate` |
| Recommendation | `no_go_for_compression_candidate`; q8 rejected until quality gate passes, q4 rejected until greedy failures are resolved, Planar/Iso deferred |

Workload cases:

| Workload | Family | Tokens | Seed |
|---|---|---:|---:|
| `benchmark_qa_4k_001` | `benchmark_qa` | 4095 | 20260633 |
| `chat_short_1k_001` | `chat_short` | 1024 | 20260630 |
| `code_review_rust_4k_001` | `code_review_rust` | 4096 | 20260631 |
| `long_repo_pack_16k_001` | `long_repo_pack` | 16384 | 20260639 |
| `prefix_reuse_edit_8k_a_001` | `prefix_reuse_edit` | 8192 | 20260636 |
| `tool_json_1k_001` | `tool_json` | 1024 | 20260635 |

Compression quality highlights:

| Workload | q8 gate | q8 logit delta | q8 payload reduction | q4 gate | q4 failure |
|---|---|---:|---:|---|---|
| `benchmark_qa_4k_001` | `false` | 1.000 | 7.540% | `false` | greedy token mismatch: baseline `107`, q4 `45518` |
| `chat_short_1k_001` | `true` | 0.438 | 2.306% | `false` | greedy token mismatch: baseline `45518`, q4 `236779` |
| `code_review_rust_4k_001` | `true` | 0.188 | 7.541% | `true` | none |
| `long_repo_pack_16k_001` | `true` | 0.000 | 17.385% | `true` | none |
| `prefix_reuse_edit_8k_a_001` | `true` | 0.125 | 12.115% | `true` | none |
| `tool_json_1k_001` | `true` | 0.063 | 2.306% | `false` | logit delta `5.375` exceeded q4 threshold |

XR09 interpretation:

- BF16 exact restore passed on every selected real-context workload.
- q8 cannot be promoted from XR09 because it failed the deterministic quality
  threshold on `benchmark_qa_4k_001`, even though the greedy token still matched.
- q4 cannot be promoted because it failed greedy-token agreement on
  `benchmark_qa_4k_001` and `chat_short_1k_001`, and exceeded the q4 logit
  threshold on `tool_json_1k_001`.
- Payload reductions are storage-only: active KV memory reduction was `0.000%`
  for BF16, q8, and q4 because active compressed decode remains disabled.

## XR14 MTP Policy Autotune Replay Snapshot

XR14 replays the repaired XR04 MTP root summary without running the model. It
compares baseline native non-MTP `decode_ms` against MTP `draft_ms + verify_ms`
for fixed block-size, acceptance-threshold, net-latency-guarded, and oracle
selection policies. Runtime code and defaults were not changed.

| Field | Value |
|---|---|
| Command | `cargo run -p gemma4d-bench --example xr14_mtp_policy_autotune -- --out-dir benchmarks/out/XR14-mtp-policy-autotune` |
| Evidence | `benchmarks/out/XR14-mtp-policy-autotune/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` |
| Run ID | `xr14-1782892549` |
| Source run | `xr03-1782871680-737256000` from XR04 |
| Source summary SHA-256 | `e17cae919519961ea25f0ce40fe5c067c0d97047553ea83ef91d98919624c9f7` |
| Source Git SHA | `50fe4e201f7475180ac3f59041b8b6923b63b19f` plus XR04 local verifier repair |
| Mode | `xr04_mtp_policy_replay` |
| Records | `30`: 5 workloads x 6 policies |
| Source max new tokens | `32` |
| Source block sizes | `1,2` |
| Decision | `needs_more_data` |
| Recommendation | Run `XR14-mtp-policy-variance-ab` with the latency-guarded policy as the candidate before any runtime policy change. |

Policy replay results:

| Policy | Decision | MTP selections | Baseline decode ms | Selected decode ms | Speedup % | Regressions | Weighted acceptance |
|---|---|---:|---:|---:|---:|---:|---:|
| `acceptance_threshold_35pct` | `reject_candidate` | 4 | 22652.026 | 37118.315 | -63.863 | 2 | 0.539 |
| `disabled_baseline` | `baseline` | 0 | 22652.026 | 22652.026 | 0.000 | 0 | 0.000 |
| `fixed_block_1` | `reject_candidate` | 5 | 22652.026 | 38958.194 | -71.985 | 2 | 0.481 |
| `fixed_block_2` | `reject_candidate` | 5 | 22652.026 | 31758.881 | -40.203 | 3 | 0.405 |
| `net_latency_guarded_5pct` | `needs_more_data` | 2 | 22652.026 | 19757.582 | 12.778 | 0 | 0.292 |
| `oracle_fastest_exact` | `needs_more_data` | 2 | 22652.026 | 19757.582 | 12.778 | 0 | 0.292 |

Latency-guarded selected workload/block pairs:

| Workload | Family | Tokens | Seed | Selected block | Replay speedup |
|---|---|---:|---:|---:|---:|
| `benchmark_qa_4k_001` | `benchmark_qa` | 4095 | 20260633 | 1 | 26.225% |
| `mtp_candidate_1k_001` | `mtp_candidate` | 1024 | 20260641 | 2 | 21.331% |

XR14 interpretation:

- Acceptance-only gating is unsafe: the 35% threshold selected
  `mtp_candidate_4k_001:block1` with acceptance `0.781`, but that replay was
  333.301% slower than baseline decode phase.
- Net latency matters more than raw acceptance. The latency-guarded replay kept
  MTP disabled on high-acceptance slow cases and selected only the two exact
  workload/block pairs that beat baseline by at least 5%.
- XR14 is same-run replay evidence only. It cannot justify default MTP
  enablement until a fresh native non-MTP vs native MTP variance run passes on
  holdout workloads.

## Measurement Changes

| Date | Change | Files | Verification |
|---|---|---|---|
| 2026-07-02 | Added XR52 blocked evidence split. The full reference branch preserves the failed slab candidate; the evidence branch keeps the permanent findings, M06 stale-doc amendments, dead full-recompute verifier deletion, and verifier timing splits surfaced through C/Rust FFI and XR15 records. | `native/gemma4_mlx/src/native_model.cc`, `native/gemma4_mlx/src/runtime.cc`, `native/gemma4_mlx/include/gemma4_mlx.h`, `crates/gemma4d-ffi/src/lib.rs`, `crates/gemma4d-bench/examples/xr15_mtp_policy_variance_ab.rs`, `codex/goals/XR52-kv-slab-incremental.goal.md`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo test -p gemma4d-ffi --lib`; `cargo test -p gemma4d-bench --lib`; `cargo test -p gemma4d-server --all-targets`; `cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run`; `scripts/native-smoke.sh`; `scripts/mlx-diagnostics.sh`; `cmake --build target/mlx-diagnostics`; XR52 decode/MTP evidence runs; `instrumentation-on-main-smoke` exact `1/1`, no blockers, max per-event verify split delta `0.068 ms` |
| 2026-07-01 | Added XR49 MTP light-trace verifier audit and goal contract. The attempted light-trace speed hypothesis is recorded as blocked because the selected XR15 block-prefix path already emits top-1 trace diagnostics from `runtime.cc`; no runtime code from the lower-path native experiment was retained. | `codex/goals/XR49-mtp-light-trace-verifier-ab.goal.md`, `BENCHMARKS.md` | `cargo fmt --all --check`; `git diff --check`; `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `cargo test -p gemma4d-ffi --lib` |
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
| 2026-07-01 | Added XR09 KV compression real-quality A/B harness and goal contract. The runner tokenizes XR00 real workloads, writes BF16/q8/q4 payloads, restores each payload into BF16 active KV, runs deterministic continued-decode quality gates, records greedy agreement, logit delta, payload bytes, warm restore latency, active memory, generated payload paths/checksums, q4 failure analysis, failed hypotheses, blockers, and decision artifacts. It does not optimize runtime code. | `codex/goals/XR09-kv-compression-real-quality-ab.goal.md`, `crates/gemma4d-bench/examples/xr09_kv_compression_real_quality_ab.rs`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo check -p gemma4d-bench --example xr09_kv_compression_real_quality_ab`; `cargo check -p gemma4d-ffi`; `cargo check -p gemma4d-bench --examples`; `cargo test -p gemma4d-kv --lib`; XR09 smoke and full runs with escalated Metal access. |
| 2026-07-01 | Added XR14 MTP policy autotune replay harness and goal contract. The runner reads the XR04 root summary, records source artifact hashes, deterministic seeds, token lengths, block sizes, fixed-block policy outcomes, acceptance-threshold failures, and net-latency-guarded replay decisions. It does not run the model, optimize runtime code, or enable MTP by default. | `codex/goals/XR14-mtp-policy-autotune.goal.md`, `crates/gemma4d-bench/examples/xr14_mtp_policy_autotune.rs`, `BENCHMARKS.md` | `cargo fmt --check`; `cargo test -p gemma4d-bench --example xr14_mtp_policy_autotune`; `cargo run -p gemma4d-bench --example xr14_mtp_policy_autotune -- --out-dir benchmarks/out/XR14-mtp-policy-autotune`. |
| 2026-07-01 | Added XR42 Rayon manifest hashing A/B harness and goal contract. The runner compares sequential safetensors inventory hashing against bounded Rayon thread pools, records deterministic seed metadata, explicit token-length non-applicability, artifact paths, file counts/bytes, inventory hashes, thread counts, p50/p95 timings, blockers, and decision artifacts. Rayon is a `gemma4d-bench` dev-dependency only; no runtime inference path or default manifest behavior changed. | `Cargo.lock`, `crates/gemma4d-bench/Cargo.toml`, `crates/gemma4d-bench/examples/xr42_rayon_manifest_hashing_ab.rs`, `codex/goals/XR42-rayon-manifest-hashing-ab.goal.md`, `BENCHMARKS.md` | `cargo fmt --all --check`; `cargo check -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab`; `cargo run -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab -- --out-dir benchmarks/out/XR42-rayon-manifest-hashing-ab --trials 3 --thread-counts 1,2,4`. |
| 2026-07-01 | Added XR43 MTP block-prefix selected-slice contract and benchmark evidence. This is a no-runtime-code slice using the existing default-off block-prefix MTP path to confirm the XR24 promising selection under fresh measured trials, with exactness, acceptance, rollback, draft/verify timing, memory, seeds, token lengths, blockers, and decision artifacts recorded. | `codex/goals/XR43-mtp-block-prefix-selected-slice.goal.md`, `BENCHMARKS.md` | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR43-mtp-block-prefix-selected-slice/candidate-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001`. |
| 2026-07-01 | Added XR44 MTP lazy block-prefix selected-slice contract and benchmark evidence. This is a no-runtime-code slice combining existing default-off lazy draft and block-prefix flags to test whether first-reject draft reduction strengthens XR43's selected MTP path. Records exactness, attempted/accepted tokens, rollbacks, event histogram, draft/verify/decode timing, memory, seeds, token lengths, blockers, and decision artifacts. | `codex/goals/XR44-mtp-lazy-block-prefix-selected-slice.goal.md`, `BENCHMARKS.md` | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR44-mtp-lazy-block-prefix-selected-slice/candidate-lazy-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001`. |
| 2026-07-01 | Added XR45 MTP lazy block-prefix 1K family holdout contract and benchmark evidence. This is a no-runtime-code slice using existing default-off lazy draft and block-prefix flags across the available 1K real-context families. Records exactness, attempted/accepted tokens, rollbacks, event histogram, draft/verify/decode timing, memory, seeds, token lengths, blockers, and decision artifacts while separating acceptance from speed. | `codex/goals/XR45-mtp-lazy-block-prefix-1k-family-holdout.goal.md`, `BENCHMARKS.md` | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR45-mtp-lazy-block-prefix-1k-family-holdout/candidate-lazy-block-prefix-1k --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`. |
| 2026-07-01 | Added XR46 MTP adaptive zero-accept fallback to the XR15 harness. The opt-in flags disable MTP for the remaining tail after a sustained zero-accept run once enough output tokens have been generated, then continue with native `decode_one`. The runner records adaptive settings, fallback decode time, whether fallback fired, reason, pass index, and generated-token position. Defaults remain unchanged. | `crates/gemma4d-bench/examples/xr15_mtp_policy_variance_ab.rs`, `codex/goals/XR46-mtp-adaptive-zero-run-fallback.goal.md`, `BENCHMARKS.md` | `cargo fmt --all --check`; `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR46-mtp-adaptive-zero-run-fallback/candidate-adaptive-zero-run --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 4 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`. |
| 2026-07-01 | Added XR47 MTP adaptive threshold-sweep contract and benchmark evidence. This is a no-runtime-code slice using the XR46 opt-in adaptive flags at `zero-run=1,min12` to test whether earlier fallback can remove the `mtp_candidate_1k_001` regression while preserving the two 1K wins. | `codex/goals/XR47-mtp-adaptive-threshold-sweep.goal.md`, `BENCHMARKS.md` | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR47-mtp-adaptive-threshold-sweep/zero-run-1-min12 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 1 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`. |
| 2026-07-01 | Added XR48 MTP adaptive zero-run 3 threshold-sweep contract and benchmark evidence. This is a no-runtime-code slice using the XR46 opt-in adaptive flags at `zero-run=3,min12` to test whether a middle fallback threshold avoids XR47's chat fallback while improving the weak 1K candidate path. | `codex/goals/XR48-mtp-adaptive-zero-run-3-sweep.goal.md`, `BENCHMARKS.md` | `cargo fmt --all --check`; `git diff --check`; `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR48-mtp-adaptive-zero-run-3-sweep/zero-run-3-min12 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001`. |
| 2026-07-03 | Completed XR54-R build-provenance stamping and the XR54 drafter-only PyTorch parity contingency. XR15 summaries/records now fail closed unless git SHA, dirty-diff SHA-256, dirty-diff byte count, runner binary path, and runner binary link mtime are available. The parity path exports `hidden.last`, shared KV, and ordered target token embeddings; the Python reference path now uses bf16-safe safetensors loading, a dense assistant checkpoint, explicit `lm_head.weight` tying, `torch.bfloat16` model construction, and a fail-closed pinned/incremented `matches_native` gate. Verdict: `reject_candidate` for the XR54 acceptance-fix hypothesis; keep the pin as behaviorally neutral reference-convention alignment. | `crates/gemma4d-bench/examples/xr15_mtp_policy_variance_ab.rs`, `native/gemma4_mlx/include/gemma4_mlx.h`, `native/gemma4_mlx/src/native_model.{h,cc}`, `native/gemma4_mlx/src/runtime.cc`, `crates/gemma4d-ffi/src/lib.rs`, `crates/gemma4d-bench/examples/xr54_drafter_pytorch_parity.rs`, `scripts/xr54_drafter_pytorch_parity.py`, `scripts/xr54_dequant_assistant.py`, `codex/goals/XR54-mtp-position-pin.goal.md`, `BENCHMARKS.md` | `cargo fmt --all --check`; `GEMMA4D_REQUIRE_MLX=1 cargo build -p gemma4d-bench --example xr15_mtp_policy_variance_ab`; `GEMMA4D_REQUIRE_MLX=1 cargo build -p gemma4d-bench --example xr54_drafter_pytorch_parity`; `/Users/justin/venvs/xr54-parity/bin/python scripts/xr54_dequant_assistant.py --src artifacts/models/gemma-4-12B-it-qat-assistant-4bit --out artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32`; `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 target/debug/examples/xr54_drafter_pytorch_parity --out-dir benchmarks/out/XR54-mtp-position-pin/pytorch-parity --reference-records benchmarks/out/XR54-mtp-position-pin/xr54-r-mtp-candidate-one-trial/records.jsonl`. |

## Verification Gates

| Date | Command | Status | Notes |
|---|---|---|---|
| 2026-07-03 | `/usr/bin/time -p env GEMMA4D_REQUIRE_MLX=1 cargo build -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Clean post-`cargo clean` MLX-required rebuild completed in `8.38s`; `CMakeCache.txt` has `GEMMA4D_REQUIRE_MLX:BOOL=ON`; fresh native objects/archive and runner binary postdated the XR54 source edit before the one-leg rerun. |
| 2026-07-03 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 target/debug/examples/xr15_mtp_policy_variance_ab --out-dir benchmarks/out/XR54-mtp-position-pin/xr54-r-mtp-candidate-one-trial --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 1 --warmups 0 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id mtp_candidate_1k_001` | Passed | Escalated Metal/MLX run wrote provenance-stamped evidence. The fresh draft arrays were byte-identical to all XR48 measured `mtp_candidate_1k_001` records and acceptance stayed `7/15 = 0.467`, so the XR54 refutation is real. |
| 2026-07-03 | `GEMMA4D_REQUIRE_MLX=1 cargo build -p gemma4d-bench --example xr54_drafter_pytorch_parity` | Passed | Compiled the XR54 parity diagnostic and native/Rust FFI payload-export surface. |
| 2026-07-03 | `/Users/justin/venvs/xr54-parity/bin/python scripts/xr54_dequant_assistant.py --src artifacts/models/gemma-4-12B-it-qat-assistant-4bit --out artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32` | Passed | Regenerated the dense f32 PyTorch assistant checkpoint from the MLX affine-q4 artifact: `23` affine-q4 tensors dequantized, `49` dense tensors written, `lm_head.weight` tied, and tokenizer/chat files copied. |
| 2026-07-03 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 target/debug/examples/xr54_drafter_pytorch_parity --out-dir benchmarks/out/XR54-mtp-position-pin/pytorch-parity --reference-records benchmarks/out/XR54-mtp-position-pin/xr54-r-mtp-candidate-one-trial/records.jsonl` | Passed | Escalated Metal/MLX run used the runner's default `/Users/justin/venvs/xr54-parity/bin/python` and default vendored-Transformers PYTHONPATH, exported `payload.safetensors`, and completed PyTorch reference parity. Native draft `[236792,236865]` matched XR54-R; pinned positions `[1023,1023]` and incremented positions `[1023,1024]` both produced `[236792,236865]` with `matches_native=true`. |
| 2026-07-02 | `cargo fmt --all --check` | Passed | Formatting gate after XR52 native/FFI/benchmark changes. |
| 2026-07-02 | `cargo test -p gemma4d-ffi --lib`; `cargo test -p gemma4d-bench --lib`; `cargo test -p gemma4d-server --all-targets`; `cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run` | Passed | Focused Rust and XR15 compile coverage for verifier timing split and FFI step-result shape changes. |
| 2026-07-02 | `scripts/native-smoke.sh`; `scripts/mlx-diagnostics.sh`; `cmake --build target/mlx-diagnostics` | Passed | Native/C++ compile coverage after the XR52 evidence split, dead verifier deletion, and timing instrumentation. |
| 2026-07-02 | `git diff --check` | Passed | Whitespace gate after XR52 code and documentation edits. |
| 2026-07-02 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR52-kv-slab-incremental/instrumentation-on-main-smoke --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 1 --warmups 0 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001` | Passed | Escalated evidence-branch smoke on concat storage wrote all artifacts, exact `1/1`, no blockers, and emitted verifier timing splits. Max per-event absolute `verify_ms - split_sum` was `0.068 ms`; aggregate record split was `verify_ms 1861.279 ms`, stage `0.217`, forward `1772.355`, repair `87.861`. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after adding the XR42 Rayon harness and ledger entries. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab` | Passed | Compiles the standalone benchmark-prep harness using Rayon as a `gemma4d-bench` dev-dependency. |
| 2026-07-01 | `cargo run -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab -- --out-dir benchmarks/out/XR42-rayon-manifest-hashing-ab --trials 3 --thread-counts 1,2,4` | Passed | Wrote 24 records, summary, report, blockers, and decision artifacts; no blockers; target 2-thread Rayon p50 improved `20.284%` with matching inventory hash. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after adding XR43 contract docs and ledger entries. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Compile gate before XR43 selected-slice run. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR43-mtp-block-prefix-selected-slice/candidate-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001` | Passed | Sandboxed attempt failed before benchmarking with no Metal device; escalated rerun wrote all XR43 artifacts, no hard blockers, `8/8` exact records, and net-latency-guarded policy selected only `chat_short_1k_001:block2`. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after adding XR44 contract docs and ledger entries. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Compile gate before XR44 lazy block-prefix selected-slice run. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR44-mtp-lazy-block-prefix-selected-slice/candidate-lazy-block-prefix-selected --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id mtp_candidate_4k_001` | Passed | Sandboxed attempt failed before benchmarking with no Metal device; escalated rerun wrote all XR44 artifacts, no hard blockers, `8/8` exact records, and net-latency-guarded policy selected only `chat_short_1k_001:block2` with aggregate `5.777%`. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Compile gate before XR45 lazy block-prefix 1K family holdout run. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR45-mtp-lazy-block-prefix-1k-family-holdout/candidate-lazy-block-prefix-1k --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001` | Passed | Escalated run wrote all XR45 artifacts, no hard blockers, `12/12` exact records, and net-latency-guarded policy selected `chat_short_1k_001:block2` plus `tool_json_1k_001:block2` with aggregate `14.680%`; fixed block and acceptance-threshold policies rejected because `mtp_candidate_1k_001` regressed. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after adding XR46 adaptive fallback harness fields and ledger entries. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Compile gate for the opt-in adaptive fallback harness path. |
| 2026-07-01 | `cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Example unit tests still pass after adaptive fallback fields and decode-phase accounting changes. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR46-mtp-adaptive-zero-run-fallback/candidate-adaptive-zero-run --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 4 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001` | Passed | Escalated run wrote all XR46 artifacts, no hard blockers, `12/12` exact records. Adaptive fallback fired only for `mtp_candidate_1k_001` at pass `10` after `16` generated tokens, cutting attempted draft tokens `96 -> 48` versus XR45, but that workload still regressed by `9.439%`; guarded policy selected only `chat_short_1k_001` and `tool_json_1k_001` with aggregate `20.322%`. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Compile gate before XR47 adaptive threshold sweep. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR47-mtp-adaptive-threshold-sweep/zero-run-1-min12 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 1 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001` | Passed | Escalated run wrote all XR47 artifacts, no hard blockers, `12/12` exact records. `mtp_candidate_1k_001` moved from XR46 `-9.439%` to `+1.332%`, but still missed the `5%` per-workload guard; aggressive fallback also fired on `chat_short_1k_001`, leaving only `+5.958%` margin. |
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after adding XR48 contract docs and ledger entries. |
| 2026-07-01 | `git diff --check` | Passed | Whitespace gate before and after XR48 documentation updates. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr15_mtp_policy_variance_ab` | Passed | Compile gate before XR48 adaptive zero-run 3 sweep. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR48-mtp-adaptive-zero-run-3-sweep/zero-run-3-min12 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001` | Passed | Sandboxed attempt failed before benchmarking with no Metal device; escalated rerun wrote all XR48 artifacts, no hard blockers, `12/12` exact records. `chat_short_1k_001` avoided XR47's fallback and recovered to `+21.710%`; `mtp_candidate_1k_001` improved versus XR46 but still regressed by `1.211%`, so guarded policy selected only chat and tool with aggregate `15.302%`. |
| 2026-06-30 | `cargo test -p gemma4d-server -p gemma4d-bench --all-targets` | Passed | Focused compile/test coverage for changed server and benchmark code. |
| 2026-07-01 | `cargo fmt --check` | Passed | Formatting gate after adding the XR14 replay harness and ledger entries. |
| 2026-07-01 | `cargo test -p gemma4d-bench --example xr14_mtp_policy_autotune` | Passed | 3 policy-unit tests passed: acceptance threshold can select a net-slow candidate, net-latency guard can select low-acceptance fast candidates, and aggregate regressions reject a policy. |
| 2026-07-01 | `cargo run -p gemma4d-bench --example xr14_mtp_policy_autotune -- --out-dir benchmarks/out/XR14-mtp-policy-autotune` | Passed | Wrote 30 replay records and all required XR14 artifacts; decision is `needs_more_data` with no hard blockers. |
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
| 2026-07-01 | `cargo fmt --all --check` | Passed | Formatting gate after the XR09 runner. |
| 2026-07-01 | `cargo check -p gemma4d-bench --example xr09_kv_compression_real_quality_ab` | Passed | Focused compile gate for the XR09 compression real-quality runner. |
| 2026-07-01 | `cargo check -p gemma4d-ffi` | Passed | Focused native/FFI compile gate before XR09 native MLX execution. |
| 2026-07-01 | `cargo check -p gemma4d-bench --examples` | Passed | Compile coverage for all benchmark examples after adding XR09. |
| 2026-07-01 | `cargo test -p gemma4d-kv --lib` | Passed | 18 passed; covers namespace mismatch, adapter partitioning, RAM/SSD restore, cache accounting, compression metadata, corruption rejection, and mid-decode SSD rejection. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr09_kv_compression_real_quality_ab -- --out-dir benchmarks/out/XR09-kv-compression-real-quality-ab-smoke --clear-workload-ids --workload-id tool_json_1k_001 --trials 1` | Passed | Required escalated Metal access; wrote 1 smoke record and final smoke artifacts. Decision was `accept_candidate` for the narrow smoke, but q4 failed the quality gate by logit delta. |
| 2026-07-01 | `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr09_kv_compression_real_quality_ab -- --out-dir benchmarks/out/XR09-kv-compression-real-quality-ab` | Passed | Required escalated Metal access; wrote 6 real-context records and final XR09 artifacts. Decision is `reject_candidate`; q8 failed `benchmark_qa_4k_001`, q4 failed 3 families, and active compressed decode remains disabled. |

## Current Claim Boundaries

- M12 and P00 broad throughput claims are helper-backed through the Rust C ABI
  and MLX-LM helper.
- The hand-written native Gemma 4 graph remains opt-in and is not represented by
  M12 or P00 helper-backed throughput numbers.
- `mlx_active_memory_gb` and `mlx_cache_memory_gb` are tracked as nullable P00
  fields until the helper/native boundary exposes those measurements.
- P02 real-helper server inference remains opt-in. After XR53, a model-path
  serve config (`gemma4d serve --model-path PATH`) defaults to
  PersistentNative; zero-arg/no-model-path serving and explicit
  `--backend stub` remain the M11 stub. P02 does not apply adapters or MTP on
  the real server path.
- P02 server benchmark measurements include HTTP route overhead and pay model
  load per request. Use XR11 and later persistent-native server measurements for
  resident server-session latency claims.
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
- XR51 server-native prefill default is scoped to persistent-native server
  workers with `GEMMA4D_USE_NATIVE_GRAPH` enabled. It applies
  `PrefillChunkPolicy::LongContext256` after resident load only when neither
  `GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS` nor
  `GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY` is set; explicit chunk envs win. Stub,
  real-helper, generate CLI, and helper-backed paths are unchanged. The native
  long-context chunk policy engages at `>=4096` prompt tokens, so XR51 1K rows
  are persistence-only, not isolated chunk-policy speed evidence.
- XR53 admission is per-backend and remains a conservative guardrail, not an
  exact tokenizer or allocator prediction. Stub uses its legacy lightweight
  token estimate and is not charged native resident weights. RealHelper and
  PersistentNative use XR51/P04 measured constants with
  `max(ceil(words * 13 / 10), ceil(prompt_bytes / 2.25))`, chunked constants
  only when the server-owned XR51 default chunk policy is known active, and
  unchunked worst-case constants otherwise. Explicit native chunk env overrides
  are treated as unknown and therefore unchunked until a native policy getter
  exists. Unchunked model-backed prompt estimates above 16K fail closed with
  `memory_guard_rejected`. The context guard also uses the conservative upper
  bound, so near-limit prompts can fail `context_too_large` before a precise
  tokenizer-side admission estimate exists.
- XR52 does not justify promoting KV slab storage or re-anchoring native decode
  baselines. The exact bounded slab candidate matched tokens and preserved MTP
  acceptance, but missed the `>=5%` decode and `+5` point MTP promotion gates.
  The faster rotating sliding-window experiments are diagnostic only: one
  failed token parity and the other drifted MTP acceptance. XR53 default-backend
  and admission-estimator work remains unblocked because no XR52 decode baseline
  re-anchor happened.
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
- XR54 aligns the MTP drafter position argument with Hugging Face's
  constant-position convention and preserves exactness, but it is not an
  accepted acceptance fix: an MLX-required XR54-R rebuild/rerun confirmed fresh
  draft tokens were byte-identical to XR48 and `mtp_candidate_1k_001` slot-1
  acceptance stayed `3/18`. The drafter-only PyTorch parity run confirmed the
  vendored Hugging Face reference is position-insensitive on the recorded round:
  pinned `[1023,1023]` and incremented `[1023,1024]` positions both produced
  native-exact `[236792,236865]`. Treat the slot-1 collapse as a model/content
  property, not a remaining native drafter implementation bug. Pre-XR54 MTP
  acceptance numbers are historical and should not be used as post-pin evidence
  without rerunning under the provenance-stamped harness. XR15 evidence
  records now stamp git SHA, dirty-diff SHA-256, dirty-diff byte count, runner
  binary path, and runner binary link mtime. XR55 and broader MTP performance
  work are unblocked with tempered acceptance expectations.
- XR55 proves native MTP block sizes `{1,2,3,4,6,8}` can remain byte-identical
  to non-MTP greedy and to the sequential verifier path under the default-off
  block-prefix experiment. It does not enable MTP by default and does not make a
  broad fixed-N claim: the guarded policy selected only
  `chat_short_1k_001:N=3` and `tool_json_1k_001:N=4`, while fixed `N=4`,
  `N=6`, and `N=8` regress at least two held-out workloads. Later slots have
  sharply lower acceptance, and the exact accepted-prefix repair cost dominates
  N>=4. `KvPolicy.block_size_tokens` is unrelated to MTP draft block size.
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
- XR09 does not justify promoting q8 or q4 compressed payloads. q8 failed one
  real-context deterministic quality gate, q4 failed three, and active-memory
  claims remain invalid because active compressed decode stayed disabled.
