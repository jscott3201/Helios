# XR49 - MTP light trace verifier A/B

## Outcome

Test whether skipping full-vocab MTP diagnostic trace extraction reduces verifier
latency enough to improve the selected 1K MTP path without changing greedy
accept/reject behavior.

## Scope

- Baseline evidence: XR48 adaptive zero-run 3 sweep using full MTP trace
  diagnostics.
- Candidate: default-off native verifier flag
  `GEMMA4D_EXPERIMENTAL_MTP_LIGHT_TRACE=1`, combined with the current selected
  1K MTP flags:
  - `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  - `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`
  - `--adaptive-zero-accept-run 3`
  - `--adaptive-min-generated-tokens 12`
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`.
- Workloads:
  - `chat_short_1k_001`
  - `tool_json_1k_001`
  - `mtp_candidate_1k_001`
- Horizon: `32` generated tokens.
- Block size: `2`.
- Trials: `3` measured plus `1` warmup.

## Rationale

The native verifier computes greedy target tokens for accept/reject, but it also
materializes selected-position logits as float32 and scans the full vocabulary to
populate top-k diagnostics. For policy runs, full top-k trace detail is useful
for analysis but not required to decide whether a draft token matches the target
greedy token.

`GEMMA4D_EXPERIMENTAL_MTP_LIGHT_TRACE=1` should:

- Preserve target greedy token and greedy logit extraction.
- Preserve draft accept/reject and committed token behavior.
- Set MTP trace `top_k` to `1` and report only the target greedy token/logit as
  top-1.
- Avoid full-vocab top-k scanning and exact draft-logit lookup.
- Keep defaults and public ABI layout unchanged.

## Required Work

1. Implement the default-off light-trace flag in the native verifier.
2. Run compile/correctness gates.
3. Run the XR15 real-context MTP A/B harness with the selected 1K workload set.
4. Record exact commands, generated files, git SHA, deterministic workload
   seeds, token lengths, exactness, trace `top_k`, attempted/accepted tokens,
   rollback count, `draft_ms`, `verify_ms`, `fallback_decode_ms`, decode phase,
   peak MLX, active KV bytes, and blockers.
5. Update `BENCHMARKS.md`, including headline MTP rows only for stable top-line
   numbers.
6. Keep MTP and light trace disabled by default.

## Acceptance Gates

- Candidate output is byte-identical to native non-MTP baseline for every
  measured record.
- Trace `top_k` is `1` only when the light-trace flag is set.
- `chat_short_1k_001` and `tool_json_1k_001` still clear the `5%` guarded
  speedup threshold.
- `mtp_candidate_1k_001` improves over XR48's `2915.728 ms` MTP decode-phase
  p50 and ideally clears the native baseline.
- Peak MLX memory stays under the configured tiny16 gate.
- Active KV bytes stay in the expected 1K shape.
- Acceptance rate and speed are reported separately.
- No default-on runtime, server, adapter, tokenizer, or public ABI behavior
  changes in this goal.

## Non-goals

- Do not enable MTP or light trace by default.
- Do not change target greedy math, drafter math, verifier accept/reject
  semantics, sampling behavior, server defaults, active KV compression, prefix
  cache policy, adapter policy, or prefill policy.
- Do not remove the full trace path.
- Do not change the C ABI struct layout.

## Required Artifacts

```text
benchmarks/out/XR49-mtp-light-trace-verifier-ab/baseline-full-trace-v2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR49-mtp-light-trace-verifier-ab/candidate-light-trace-v2/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `blocked_with_evidence`.

The light-trace speed hypothesis is not a valid optimization claim for the
selected XR15 path. The selected runtime path already records MTP diagnostics as
top-1 trace data from `native/gemma4_mlx/src/runtime.cc` via
`initialize_mtp_trace`, `record_mtp_target_step`, and
`record_mtp_draft_score`. With `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
and `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`, XR15 did not exercise the
lower native `forward_verify_logits` path where a full top-k scan can occur.
The lower-path light-trace patch was discarded and no runtime code is retained
from this pass.

Authoritative v2 runs:

- Control run ID: `xr15-1782931446`
- Candidate run ID: `xr15-1782931663`
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`
- Trials: `3` measured plus `1` warmup
- Horizon: `32` generated tokens
- Block size: `2`
- Adaptive fallback: `--adaptive-zero-accept-run 3
  --adaptive-min-generated-tokens 12`
- Workloads: `chat_short_1k_001`, `tool_json_1k_001`,
  `mtp_candidate_1k_001`

Control command:

```bash
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR49-mtp-light-trace-verifier-ab/baseline-full-trace-v2 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001
```

Candidate command:

```bash
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 GEMMA4D_EXPERIMENTAL_MTP_LIGHT_TRACE=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR49-mtp-light-trace-verifier-ab/candidate-light-trace-v2 --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001
```

Both v2 runs reported `trace_top_k=[1]` for every measured event. Therefore the
candidate env var did not prove that full-vocab top-k scanning was skipped in
the measured path; the measured path was already light-trace/top-1.

Candidate v2 per-workload evidence:

| Workload | Exact | Acceptance | Native p50 ms | MTP p50 ms | Speedup | Fallback p50 ms | Active KV | Peak MLX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `3/3` | `69/96 = 0.719` | `2695.984` | `2045.486` | `+24.128%` | `0.000` | `352845824` | `8.002 GB` |
| `tool_json_1k_001` | `3/3` | `75/96 = 0.781` | `2730.721` | `2027.634` | `+25.747%` | `0.000` | `352845824` | `8.002 GB` |
| `mtp_candidate_1k_001` | `3/3` | `21/45 = 0.467` | `2790.422` | `2861.231` | `-2.538%` | `1307.586` | `352829440` | `8.008 GB` |

`mtp_candidate_1k_001` still failed the `5%` per-workload speed guard and
auto-disabled in `3/3` measured records at pass `9`. The useful next candidate
is a runtime-path policy change that avoids repeated second-slot misses, not a
full-vocab diagnostic trace shortcut.

Non-authoritative intermediate directories also exist under
`benchmarks/out/XR49-mtp-light-trace-verifier-ab/{baseline-full-trace,candidate-light-trace}`.
They were generated before the final path audit and should not be used for
decisions.

## Completion Rule

Stop when the light-trace candidate has fresh measured evidence against native
baseline and XR48-style full-trace behavior, or when blockers explain why it
cannot be judged.
