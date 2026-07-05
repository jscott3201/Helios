# XR78 - Native amortized warmup matrix

## Objective

Test whether the XR77 warm/JIT/cache effect can be paid once per exact shape
and amortized across repeated same-shape native requests, without changing
runtime or server defaults.

## Current Evidence

- XR77 repeated the same-context first-token tail win with cost accounting:
  `chat_short_1k_001` first-token p50 moved `188.836 -> 92.922 ms`, and raw
  p99/max improved `188.836 -> 92.922 ms` (`+50.792%`).
- XR77 also showed naive per-request warmup is too expensive: discarded warmup
  total p50 was `3203.529 ms` for the 1024-token chat context.
- The next useful question is whether a warm state survives across repeated
  fresh-cache requests on the same loaded target and whether the cost can be
  amortized across same-shape requests.
- MTP protected aggregate work remains parked until this native tail baseline is
  cleaner. DSpark remains parked.

## Scope

- Add an XR06 benchmark variant that performs one discarded same-context warmup
  per loaded target/workload, then measures repeated fresh-cache requests
  against the same loaded target.
- Record both full warmup event cost and amortized warmup cost per measured
  request.
- Compare against the non-profile `native_decode_runtime_default` baseline.
- Run a focused matrix over selected 1K and 4K workloads.
- Keep artifacts under
  `benchmarks/out/XR78-native-amortized-warmup-matrix/`.

## Non-goals

- Do not change runtime/server/native ABI defaults.
- Do not promote warmup into a production request path.
- Do not run or promote MTP from XR78.
- Do not restart DSpark.

## Acceptance Criteria

1. XR06 exposes a default-off amortized warmup variant with config markers for
   warmup reuse scope, measured request count, and same-loaded-target lifetime
   probing.
2. Records include request repeat index, warmup event cost, amortization
   denominator, and amortized warmup cost.
3. Reports distinguish warmup event count from measured request count.
4. Static gates pass:
   `cargo fmt --all --check`, `git diff --check`, and
   `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.
5. A focused Metal run records exactness, raw p50/p95/p99/max,
   first-token p50/p95/p99/max, warmup event cost, amortized cost, peak MLX,
   active KV, and profile sample counts for at least one 1K and one 4K
   workload.
6. The result states whether native tail behavior is clean enough to rerun the
   protected MTP aggregate, or whether native warmup needs another matrix.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256 \
cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- \
  --out-dir benchmarks/out/XR78-native-amortized-warmup-matrix/amortized-1k-4k \
  --trials 2 \
  --max-new-tokens 32 \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id code_review_rust_4k_001 \
  --variants native_decode_eval_per_layer,native_decode_runtime_default,native_decode_runtime_default_warmup_amortized_4
```

## Completion Rule

Complete XR78 only when the amortized matrix supports a decision:
`accept_candidate`, `reject_candidate`, `needs_more_data`, or
`blocked_with_evidence`, with exact commands, artifacts, claim boundaries, and
the next native/MTP sequencing decision recorded.

## Result - 2026-07-05

Status: `accept_candidate` for the chat first-token tail lane only. Native
warmup remains default-off and viable only as out-of-request/load-time shape
work. The next high-value step is a warmup-aware MTP protected aggregate rerun.

XR78 added `native_decode_runtime_default_warmup_amortized_4`, which performs
one discarded same-context warmup per loaded target/workload, then measures
four fresh-cache requests on the same loaded target. XR06 records now include
`request_repeat_index`, warmup event cost, amortization denominator, and
amortized warmup cost. Runtime defaults, server defaults, native ABI, MTP, and
DSpark behavior were unchanged.

Evidence:

- Gate-valid artifacts:
  `benchmarks/out/XR78-native-amortized-warmup-matrix/amortized-1k-4k-trials3/`
- Exploratory low-N artifacts:
  `benchmarks/out/XR78-native-amortized-warmup-matrix/amortized-1k-4k/`
- Static gates passed:
  `cargo fmt --all --check` and
  `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.
- Gate-valid run wrote `36/36` correct records, `0/1116` profile samples, no
  blockers, and peak MLX `7.639 GB`.
- On `chat_short_1k_001`, amortized warmup versus runtime default passed the
  XR06 tail gate: raw p50 moved `70.524 -> 70.300 ms`, raw p95 moved
  `72.888 -> 71.508 ms`, raw p99/max improved `387.059 -> 92.292 ms`
  (`+76.155%`), and first-token p50 improved `387.059 -> 92.292 ms`.
- Chat warmup event p50 was `3843.020 ms`, amortized over four measured
  requests to `960.755 ms`.
- On `code_review_rust_4k_001`, the baseline did not reproduce a tail; warmup
  changed raw p50 `72.319 -> 72.424 ms` and raw p99 `75.071 -> 74.852 ms`, so
  no 4K warmup promotion claim is supported.

Claim boundary:

- Warmup costs are recorded separately and are not included in measured request
  `prefill_ms`, `decode_ms`, or `total_ms`.
- XR78 proves same-loaded-target, same-shape fresh-cache warm-state lifetime for
  the chat tail lane. It does not prove a production server warmup policy.
- Broad MTP default-on remains parked until the protected aggregate clears the
  release gate. DSpark remains parked.
