# XR63 - MTP terminal block-prefix no-lookahead

## Outcome

Evaluate the P3 MTP verifier/fallback overhead lane that skips final
lookahead state when the generation budget ends, while preserving the
block-prefix verifier path instead of falling back to the slower serial
terminal verifier.

Decision: `reject_candidate`.

The candidate is correct and remains benchmark-only/default-off, but it does
not clear the P3 `+5%` patch gate. On the selected XR61 1K lanes it improved
candidate selected decode phase by only `+2.043%` versus the fresh P3 baseline.

## Scope

- Add `--allow-missing-source-replay` to the XR15 harness so fresh P3 live
  verifier runs can disclose and proceed when ignored XR14 replay artifacts are
  absent locally. The summary still records the missing replay path,
  `source_replay_run_id=unavailable`, and `allow_missing_source_replay=true`.
- In the native experimental block-prefix verifier, allow
  `terminal_commit_count == draft_count` and verify a terminal final block with
  `N-1` target forwards.
- Return no continuation state only when the committed tail fills the terminal
  generation budget. The path is reachable only through
  `--experimental-terminal-no-lookahead` plus the existing experimental
  block-prefix rollback env.

## Non-goals

- Do not enable MTP by default.
- Do not change default `gemma4_verify_tokens` behavior when
  `terminal_commit_count == 0`.
- Do not change verifier commit semantics outside the terminal
  cache-discard-only case.
- Do not claim a default-on MTP result from this experiment.

## Commands

Baseline selected-lane run:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR61-adaptive-n-mtp/p3-terminal-block-prefix-baseline --allow-missing-source-replay --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-policy xr61-real-margin-v1 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001
```

Candidate selected-lane run:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR61-adaptive-n-mtp/p3-terminal-block-prefix-candidate --allow-missing-source-replay --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-policy xr61-real-margin-v1 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --experimental-terminal-no-lookahead --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001
```

Sequential-oracle differential:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 GEMMA4D_EXPERIMENTAL_MTP_ADAPTIVE_N=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR61-adaptive-n-mtp/p3-terminal-block-prefix-sequential-oracle --allow-missing-source-replay --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 1,2,3,4,6,8 --adaptive-policy xr61-real-margin-v1 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --experimental-terminal-no-lookahead --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001
```

Static gates run before the MLX benchmarks:

```text
cargo fmt --all --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run
```

The first sandboxed baseline attempt failed before measurement because MLX
could not access Metal. The baseline, candidate, and oracle runs above were
rerun with approved unsandboxed Metal access.

## Evidence

- Baseline: `benchmarks/out/XR61-adaptive-n-mtp/p3-terminal-block-prefix-baseline/`
- Candidate: `benchmarks/out/XR61-adaptive-n-mtp/p3-terminal-block-prefix-candidate/`
- Sequential oracle:
  `benchmarks/out/XR61-adaptive-n-mtp/p3-terminal-block-prefix-sequential-oracle/`
- Summary:
  `benchmarks/out/XR61-adaptive-n-mtp/p3-terminal-block-prefix-summary.md`
  and `.json`

## Result

Correctness:

- Candidate record exactness: `8/8`; measured exactness: `6/6`.
- Candidate-vs-baseline MTP generated-token differential: `6/6` compared,
  no missing, extra, or mismatched measured records.
- Candidate-vs-sequential-oracle differential: `6/6` compared, no missing,
  extra, or mismatched measured records.

Selected policy comparison:

| Metric | P3 baseline | Terminal candidate | Delta |
|---|---:|---:|---:|
| Selected decode phase | `3822.879 ms` | `3744.794 ms` | `+2.043%` |
| Aggregate speedup vs native in-run | `+31.475%` | `+32.317%` | `+0.842 pp` |
| Weighted acceptance | `144/204 = 0.706` | `144/204 = 0.706` | unchanged |
| Terminal skips | `0` | `6` | `+6` |
| Peak MLX | `8.008 GB` | `8.008 GB` | unchanged |

Measured per-workload totals:

| Workload | Decode phase | Verify ms | Verify forward ms | Repair fallback ms | Draft ms | Terminal skips |
|---|---:|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `5482.894 -> 5313.268 ms` (`+3.094%`) | `5007.504 -> 4825.516` (`+3.634%`) | `4758.202 -> 4580.433` (`+3.736%`) | `247.653 -> 243.235` (`+1.784%`) | `475.390 -> 487.752` (`-2.600%`) | `0 -> 3` |
| `tool_json_1k_001` | `6037.414 -> 5996.257 ms` (`+0.682%`) | `5492.032 -> 5434.436` (`+1.049%`) | `5245.750 -> 5183.698` (`+1.183%`) | `244.798 -> 249.297` (`-1.838%`) | `545.382 -> 561.821` (`-3.014%`) | `0 -> 3` |

Interpretation:

- The native change proves the terminal block-prefix path can skip final
  lookahead while preserving exact generated tokens and oracle agreement.
- The measured gain is too small for P3 promotion: `+2.043%` selected decode
  improvement is below the required `+5%` lane gate.
- Remaining MTP limiter is still verifier/fallback cost and accepted tokens per
  verifier pass, not terminal lookahead alone.

## Completion Rule

XR63 is complete when the terminal no-lookahead candidate is either accepted by
the P3 patch gate or rejected with exactness, oracle, memory, provenance, and
ledger evidence. This run rejects it for promotion and keeps it experimental /
default-off.
