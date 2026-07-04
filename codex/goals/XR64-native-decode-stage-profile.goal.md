# XR64 - Native decode stage profile

## Outcome

Produce a stage-level native decode profile before attempting any new native
kernel, KV layout, or cache-mutation rewrite.

Decision: `needs_more_data` for XR06 A/B promotion, and
`no_valid_decode_patch_lane` for small cleanup work.

The profile found one dominant lane: native forward graph construction/execution
plus decode KV append/cache mutation. On the two selected XR61 1K lanes,
`forward_graph_ms` averaged `76.012..79.996 ms` of `82.283..86.331 ms` native
decode total. FFI overhead, greedy selection, scalar output reads, peak memory
reset/read, and hidden-view handoff were all sub-millisecond means. The next
high-value work is therefore native graph/KV internals, not Rust/FFI or greedy
selection cleanup.

## Scope

- Add `GEMMA4D_NATIVE_DECODE_PROFILE=1`, cached at native target load time.
- Keep disabled overhead to a cached flag branch inside `decode_incremental`.
- Extend the narrow C/Rust ABI with `Gemma4DecodeProfileInfo` and ABI version
  `5`.
- Extend XR06 records with optional per-token decode stage timing and emit
  `profile.json` plus `profile.md`.
- Do not change native decode behavior or defaults.

## Commands

```text
cargo fmt
cargo test -p gemma4d-ffi
cargo check -p gemma4d-bench --example xr06_native_decode_tail_latency_ab
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR61-adaptive-n-mtp/native-decode-profile --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --variant native_decode_eval_per_layer
```

The first sandboxed profile attempt failed before measurement with
`No Metal device available`; the recorded run above was rerun with approved
unsandboxed Metal access.

## Evidence

- `benchmarks/out/XR61-adaptive-n-mtp/native-decode-profile/records.jsonl`
- `benchmarks/out/XR61-adaptive-n-mtp/native-decode-profile/summary.json`
- `benchmarks/out/XR61-adaptive-n-mtp/native-decode-profile/report.md`
- `benchmarks/out/XR61-adaptive-n-mtp/native-decode-profile/profile.json`
- `benchmarks/out/XR61-adaptive-n-mtp/native-decode-profile/profile.md`

## Result

The profile run wrote `6` records with `6/6` passed and `378/378` profiled
decode samples. Peak MLX stayed `7.321 GB`; both selected 1K lanes reproduced
tail spikes and had no blockers.

| Workload | Samples | Host mean | Native total mean | Forward graph mean | Eval sync mean | Rust/FFI overhead mean | Largest stage |
|---|---:|---:|---:|---:|---:|---:|---|
| `chat_short_1k_001` | `189/189` | `86.341 ms` | `86.331 ms` | `79.996 ms` | `6.330 ms` | `0.010 ms` | `forward_graph_ms` |
| `tool_json_1k_001` | `189/189` | `82.292 ms` | `82.283 ms` | `76.012 ms` | `6.264 ms` | `0.009 ms` | `forward_graph_ms` |

Interpretation:

- The measurable decode cost is inside `decode_last_logits`: layer graph work
  plus decode KV append/cache mutation. P4 cannot isolate those internal pieces
  further without deeper layer-level instrumentation or a scoped native graph/KV
  experiment.
- `eval_sync_ms` is a secondary lane at about `6.2 ms`, but XR06 already found
  eval scheduling to be workload-local and not default-promotable.
- Peak reset/read, greedy selection/logit gather, hidden-view allocation/copy,
  scalar reads, and Rust/FFI crossing are not worthwhile near-term targets.
- Native decode remains unchanged; no new kernel or KV rewrite was implemented.

## Completion Rule

XR64 is complete when stage timings are captured under the env gate, artifacts
are written, and the next patch lane is either accepted for a scoped follow-up
or rejected as not specific enough for a correctness-preserving `>5%` patch.
This run completes P4 by pointing follow-up work at native graph/KV internals
and stopping short of a broad rewrite.
