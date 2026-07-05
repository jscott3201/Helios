# XR84 - Native prefix warm policy probe

## Objective

Test whether a cheaper out-of-request/load-time warm shape can reproduce the
chat first-token tail benefit without paying the full exact-context warmup
cost. The candidate warms the loaded target with only the first `128` workload
tokens, then measures the full workload request on a fresh cache.

## Current Evidence

- XR76 showed exact same-context warmup reduced the non-profile chat
  first-token/raw tail, but the warmup cost was not charged to measured request
  latency.
- XR77/XR78 showed exact-shape warmup cost is too large for request-path use
  and remains viable only as out-of-request/load-time or amortized work.
- XR82 showed the same warm-state behavior affects the MTP verifier path, but
  request-path preverify warmup is net rejected once costed.
- XR83 rejected the existing `native_decode_full_attention_kv_update_256`
  materialization candidate on the real non-profile chat tail.

## Scope

- Add a default-off XR06 variant:
  `native_decode_runtime_default_warmup_prefix_128`.
- The variant must use only the first 128 workload tokens for discarded warmup.
- The measured request must still run the full workload prompt on a fresh cache.
- Compare against `native_decode_eval_per_layer`,
  `native_decode_runtime_default`, and the exact-context
  `native_decode_runtime_default_warmup_probe`.

## Non-Goals

- Do not change runtime/server defaults.
- Do not promote a server warm policy from this focused harness run alone.
- Do not rerun MTP protected aggregate until native tail behavior is cleaner.
- Do not resume DSpark.

## Acceptance Criteria

1. Prefix warmup records `warmup_context_tokens=128` and leaves measured
   `input_tokens=1024` on `chat_short_1k_001`.
2. Candidate output tokens match the reference path.
3. Decode profile samples remain disabled for the measured run.
4. Peak MLX remains under the tiny16 gate.
5. Prefix warmup must improve first-token/raw tail similarly to exact warmup
   before it is worth broader load-time/server policy work.

## Verification Commands

```text
cargo fmt --all --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128 \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_runtime_default_warmup_probe,native_decode_runtime_default_warmup_prefix_128
```

## Result

Decision: `accept_candidate` as cheaper warm-shape evidence only.

Artifacts:

- `benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128/records.jsonl`
- `benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128/summary.json`
- `benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128/report.md`
- `benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128/blockers.md`
- `benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128/decision.md`
- `benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128/profile.json`
- `benchmarks/out/XR84-native-prefix-warm-policy-probe/chat-1k-prefix128/profile.md`

Evidence:

- `12/12` records passed with no blockers.
- Prefix warmup records show `warmup_context_tokens=128`; measured requests
  stayed at `input_tokens=1024`.
- Decode profile samples stayed disabled: `0/756`.
- Peak MLX stayed under tiny16 at `7.321 GB`.
- Versus `native_decode_runtime_default`, prefix warmup kept p50/p95 neutral
  (`69.702 -> 69.811 ms`, `70.189 -> 70.199 ms`) while improving raw p99/max
  and first-token p50 `150.450 -> 102.426 ms` (`+31.920%`).
- Exact-context warmup was still stronger at `92.789 ms`, but its warmup total
  p50 was `2837.263 ms`; prefix warmup total p50 was `1238.137 ms`.

Next step: build a guarded server/load-time prefix warm policy candidate before
rerunning MTP protected aggregate gates. Do not change runtime defaults from
this focused harness result alone.
