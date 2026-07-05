# XR77 - Native first-token warmup policy candidate

## Objective

Turn the XR76 same-context warmup hypothesis into a cost-accounted,
default-off native warmup candidate, then decide whether it is worth broadening
beyond the `chat_short_1k_001` first-token tail lane.

## Current Evidence

- XR72 showed the remaining chat first-token tail is dominated by
  full-attention eval, not host collection, capacity growth, or final sync.
- XR75 rejected simple serial full-attention group scheduling against the
  current runtime default.
- XR76 showed profile mode is not the main source of the non-profile runtime
  tail. The warmup probe improved chat raw p99/max
  `177.571 -> 86.680 ms` and first-token p50 `177.571 -> 86.680 ms`, but the
  discarded warmup cost was excluded from measured request latency.
- XR73 accepted scoped chat/tool MTP opt-in only. Broad MTP default-on stays
  parked until the protected aggregate clears the release gate.
- DSpark stays parked.

## Scope

- Add XR06 record/report support for discarded same-context warmup cost:
  warmup context tokens, warmup prefill ms, warmup decode ms, and warmup total
  ms.
- Add a default-off benchmark variant:
  `native_decode_runtime_default_warmup_costed`.
- Preserve runtime, server, native ABI, MTP, adapter, and DSpark defaults when
  the costed variant is not selected.
- Compare against the non-profile `native_decode_runtime_default` baseline.
- Keep artifacts under
  `benchmarks/out/XR77-native-first-token-warmup-policy-candidate/`.

## Non-goals

- Do not enable a runtime or server warmup policy by default.
- Do not claim per-request speedup without including or explicitly accounting
  for discarded warmup cost.
- Do not promote broad MTP default-on from XR77.
- Do not restart DSpark.
- Do not change native C/Rust ABI for this slice.

## Acceptance Criteria

1. XR06 records warmup context tokens, warmup prefill ms, warmup decode ms, and
   warmup total ms for warmup variants.
2. XR06 report and summary expose warmup cost aggregates alongside first-token
   latency.
3. The costed warmup variant is selected only by name and leaves runtime
   defaults unchanged when not selected.
4. Static gates pass:
   `cargo fmt --all --check`, `git diff --check`, and
   `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.
5. A focused Metal run on `chat_short_1k_001` compares
   `native_decode_eval_per_layer`, `native_decode_runtime_default`, and
   `native_decode_runtime_default_warmup_costed`, recording exactness,
   raw p50/p95/p99/max, first-token p50/p95/p99/max, warmup costs, peak MLX,
   active KV, and profile sample counts.
6. Any broader policy remains blocked until warmup cost and shape scope are
   proven beyond `chat_short_1k_001`.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR77-native-first-token-warmup-policy-candidate/costed-chat-1k \
  --trials 3 \
  --max-new-tokens 64 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_runtime_default_warmup_costed
```

## Completion Rule

Complete XR77 only when the cost-accounted artifacts support a decision:
`accept_candidate`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence`. The result must state whether warmup is viable only as
out-of-request/load-time amortized work, viable as a request-path policy, or not
worth pursuing.

## Result - 2026-07-05

Status: `accept_candidate` for cost-accounted hypothesis evidence only.
Warmup is viable only as out-of-request/load-time or amortized exact-shape work
unless broader matrix evidence changes the cost model.

XR77 added XR06 warmup-cost accounting and the default-off
`native_decode_runtime_default_warmup_costed` variant. It did not change native
ABI, runtime defaults, server defaults, MTP defaults, adapters, or DSpark.

Evidence:

- Focused Metal artifacts:
  `benchmarks/out/XR77-native-first-token-warmup-policy-candidate/costed-chat-1k/`
- Static gates passed:
  `cargo fmt --all --check`, `git diff --check`, and
  `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.
- The focused run wrote `9/9` correct records, `0/567` profile samples, no
  blockers, and peak MLX `7.321 GB`.
- Costed warmup versus non-profile runtime default repeated the first-token tail
  win: raw p50 regressed only `69.499 -> 69.643 ms` (`+0.208%`), raw p95 moved
  `70.740 -> 70.977 ms`, raw p99/max improved `188.836 -> 92.922 ms`
  (`+50.792%`), and first-token p50 improved `188.836 -> 92.922 ms`.
- Discarded warmup cost was large: total p50 `3203.529 ms`, prefill p50
  `2737.664 ms`, and decode p50 `360.509 ms` for the same 1024-token context.

Claim boundary:

- The accepted XR06 tail comparison excludes discarded warmup cost from measured
  request latency, while recording that cost separately.
- A naive per-request warmup policy is not supported.
- Next input: test a default-off load-time or amortized exact-shape warmup
  matrix across more shapes/workload families, then rerun protected MTP
  aggregate only after the native tail baseline is cleaner.
