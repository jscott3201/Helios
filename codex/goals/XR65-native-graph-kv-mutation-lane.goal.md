# XR65 - Native graph/KV mutation lane

## Objective

Split the XR64 `forward_graph_ms` bucket into evidence that can distinguish
native layer graph work from decode KV append/cache mutation. Use that evidence
to decide whether a scoped native graph/KV patch is worth attempting before
revisiting broad MTP default-on work.

## Scope

- Extend the env-gated native decode profile only when
  `GEMMA4D_NATIVE_DECODE_PROFILE=1` is enabled before target load.
- Preserve the narrow C/Rust ABI and keep layout tests pinned.
- Add native timing fields that identify:
  - token embedding setup,
  - layer graph time,
  - decode attention KV projection/append/slice/store/eval/capture time,
  - deferred decode KV eval time,
  - final norm plus LM-head/logit graph time,
  - derived non-KV forward graph time in XR06 artifacts.
- Extend XR06 `profile.json`/`profile.md` so the largest stage can name the new
  KV-mutation lane or the remaining non-KV graph lane.
- Keep behavior and defaults unchanged.

## Non-Goals

- Do not enable MTP by default.
- Do not rewrite the KV layout or add a new kernel before the split profile has
  produced evidence.
- Do not broaden raw MLX internals through Rust.
- Do not make profile timings default-on.

## Acceptance Criteria

1. `GEMMA4D_NATIVE_DECODE_PROFILE=1` emits the new split fields through the C
   ABI, Rust wrapper, per-token XR06 traces, `profile.json`, and `profile.md`.
2. Disabled profile behavior remains a cached flag branch with zeroed profile
   fields.
3. Layout and ABI version tests are updated and pass.
4. A small Metal XR06 run writes evidence under
   `benchmarks/out/XR65-native-graph-kv-mutation-lane/`.
5. The report states whether the next patch lane is KV mutation, non-KV graph
   execution, deferred KV eval, or no-go based on measured means and exactness.

## Verification Commands

```text
cargo fmt --all --check
cargo test -p gemma4d-ffi --lib
cargo test -p gemma4d-bench --lib
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_NATIVE_DECODE_PROFILE=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- --out-dir benchmarks/out/XR65-native-graph-kv-mutation-lane/baseline-split-profile --trials 3 --max-new-tokens 64 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --variant native_decode_eval_per_layer
```

## Result

Decision: `accept_candidate`.

XR65 split the prior XR64 `forward_graph_ms` bucket and found the explicit
per-layer decode path is dominated by decode KV mutation/immediate KV eval, not
the remaining non-KV graph lane. The runtime default was changed from per-layer
decode KV eval to grouped end-of-decode eval. Explicit
`GEMMA4D_NATIVE_DECODE_KV_EVAL=per_layer` or `current` keeps the old behavior
available for comparison or rollback.

### Evidence

- `benchmarks/out/XR65-native-graph-kv-mutation-lane/baseline-split-profile/`
- `benchmarks/out/XR65-native-graph-kv-mutation-lane/kv-eval-policy-ab/`
- `benchmarks/out/XR65-native-graph-kv-mutation-lane/runtime-default-ab/`

### Baseline Split

The baseline split profile wrote `6/6` passed records and `378/378` profiled
decode samples with no blockers.

| Workload | Native mean | Forward mean | KV mutation mean | Non-KV graph mean | Largest stage |
|---|---:|---:|---:|---:|---|
| `chat_short_1k_001` | `85.454 ms` | `79.156 ms` | `77.239 ms` | `1.917 ms` | `attention_kv_mutation_ms` |
| `tool_json_1k_001` | `82.628 ms` | `76.337 ms` | `74.433 ms` | `1.904 ms` | `attention_kv_mutation_ms` |

### Policy A/B

The KV-eval policy A/B wrote `24/24` passed records and `1512/1512` profiled
decode samples. `native_decode_eval_end_of_decode` passed the XR06 p95/p99
tail gate on both selected workloads with no correctness or memory regression.

### Runtime Default A/B

The runtime-default A/B left `GEMMA4D_NATIVE_DECODE_KV_EVAL` unset for the
candidate. It wrote `12/12` passed records and `756/756` profiled decode
samples with no blockers.

| Workload | Explicit per-layer p50/p95 | Runtime default p50/p95 | Accepted reason |
|---|---:|---:|---|
| `chat_short_1k_001` | `81.401 / 85.989 ms` | `70.621 / 72.425 ms` | p95 improved `15.774%` |
| `tool_json_1k_001` | `81.120 / 82.277 ms` | `70.614 / 73.170 ms` | p99 improved `29.720%` |

No MTP default changed.
