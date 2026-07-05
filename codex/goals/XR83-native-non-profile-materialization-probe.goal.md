# XR83 - Native non-profile materialization probe

## Objective

Measure the real, non-profile runtime behavior of the current native
full-attention materialization candidate on the chat first-token tail lane.
Decide whether `native_decode_full_attention_kv_update_256` is worth a broader
non-profile matrix or whether the materialization path remains below the next
high-value threshold.

## Current Evidence

- XR72 attributed chat first-token outliers to full-attention deferred eval, not
  host collection, capacity growth, or final sync.
- XR75 rejected serial full-attention group eval.
- XR76 showed profile mode was not the main tail source and a same-shape warmup
  probe reduced the non-profile chat first-token tail, but without cost
  accounting.
- XR77/XR78 showed request-path warmup cost is too large and only load-time or
  out-of-request warm shapes remain viable.
- XR82 showed the same warm-state shape exists on the MTP first verifier path,
  but request-path preverify warmup regressed net decode phase once costed.

## Scope

- Run the XR06 native decode harness without `GEMMA4D_NATIVE_DECODE_PROFILE`.
- Compare `native_decode_runtime_default` against
  `native_decode_full_attention_kv_update_256`.
- Start with `chat_short_1k_001`, 3 trials, 64 generated tokens.
- Record first-token, raw p50/p95/p99, correctness, memory, and profile-sample
  absence.

## Non-Goals

- Do not change runtime defaults.
- Do not enable broad MTP default-on.
- Do not rerun DSpark.
- Do not use profile-mode group splitting as promotion evidence for this goal.

## Acceptance Criteria

1. Produce focused XR83 artifacts under
   `benchmarks/out/XR83-native-non-profile-materialization-probe/`.
2. Candidate output tokens must match runtime-default baseline.
3. Peak MLX must remain under the tiny16 gate.
4. Candidate must improve the chat first-token/raw tail without more than a 5%
   p50 regression before any broader non-profile matrix is warranted.
5. If accepted for follow-up, the next run must broaden to the prior XR70/XR71
   matrix before MTP protected aggregate work resumes.

## Verification Commands

```text
cargo fmt --all --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR83-native-non-profile-materialization-probe/chat-1k-materialization-256 \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_runtime_default,native_decode_full_attention_kv_update_256
```

## Result - 2026-07-05

Status: `reject_candidate`.

Artifacts:

- Initial focused run without the per-layer reference baseline:
  `benchmarks/out/XR83-native-non-profile-materialization-probe/chat-1k-materialization-256/`
- Corrected run with the required reference baseline:
  `benchmarks/out/XR83-native-non-profile-materialization-probe/chat-1k-materialization-256-with-reference/`

Evidence from the corrected run:

- Records: `9/9` passed, `0` blockers.
- Correctness: `9/9` correct against the reference path.
- Decode profile samples: `0/567`, proving this is non-profile runtime-path
  evidence.
- Runtime default versus per-layer stayed correct and faster at p50, but did
  not clear the XR06 p95/p99 tail gate:
  raw p50 `81.529 -> 70.750 ms`, p95 `83.445 -> 72.603 ms`, p99
  `220.612 -> 201.004 ms`.
- `native_decode_full_attention_kv_update_256` versus runtime default stayed
  correct and under the memory gate, but missed the tail gate and worsened
  p99/first-token:
  raw p50 `70.750 -> 70.631 ms` (`+0.169%`), p95
  `72.603 -> 71.988 ms` (`+0.847%`), p99 `201.004 -> 234.602 ms`
  (`-16.715%`), peak MLX `7.321 -> 7.327 GB`.
- First-token p50 regressed `201.004 -> 234.602 ms`; first-token p95/p99/max
  also regressed `210.701 -> 247.910 ms`.

Recommendation: do not broaden the existing `full_attention_kv_update_256`
candidate to a full non-profile matrix. The next native theoretical-max work
should test a true out-of-request/load-time warm policy or a new lower-level
materialization strategy that is materially different from the already-rejected
256 slice-update candidate.
