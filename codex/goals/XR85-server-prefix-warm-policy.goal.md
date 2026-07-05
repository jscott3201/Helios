# XR85 - Server prefix warm policy candidate

## Objective

Turn the XR84 prefix-warm evidence into a real server-path candidate. The
candidate uses the persistent-native resident worker, runs an explicit local
prefix warmup before measured requests, records warmup telemetry separately,
and leaves normal request handling and runtime defaults unchanged.

## Current Evidence

- XR76/XR77 showed same-context warmup removes most of the non-profile chat
  first-token tail but naive request-path warmup is too expensive.
- XR78 showed the warm state can survive fresh-cache requests on the same
  loaded target for the chat tail lane.
- XR82 showed the same warm/JIT/cache shape affects the MTP verifier path, but
  request-path preverify warmup is net rejected.
- XR83 rejected the existing `native_decode_full_attention_kv_update_256`
  materialization path for the real non-profile chat tail.
- XR84 showed a cheaper `128`-token prefix warmup preserves most of the
  first-token/raw-tail improvement while measuring the full `1024`-token
  request on a fresh cache.

## Scope

- Add an explicit local server control endpoint:
  `POST /v1/runtime/warmup/prefix`.
- The endpoint must work only for `persistent-native`, call the resident native
  target, warm a prompt prefix off the request path, discard the cache, and
  return warmup telemetry.
- Expose warmup state through `/v1/runtime/snapshot` and Prometheus metrics.
- Extend the server A/B harness so candidate servers can run prefix warmup
  before measured requests with `--candidate-prefix-warmup-tokens`.
- Run a focused persistent-native baseline vs default persistent-native
  candidate with prefix warmup over `chat_short_1k_001` and `tool_json_1k_001`.

## Non-Goals

- Do not make prefix warmup automatic or default-on.
- Do not change request generation semantics.
- Do not change MTP defaults or run broad MTP default-on.
- Do not resume DSpark.

## Acceptance Criteria

1. Server endpoint rejects unsupported backends and invalid prefix sizes before
   model work.
2. Candidate warmups record requested prefix tokens, full prompt token count,
   warm context tokens, tokenize/prefill/decode/total time, peak MLX, and active
   KV bytes.
3. Measured request records remain token/text-identical to the explicit
   persistent-native baseline.
4. Candidate warmup cost is recorded separately and not included in request
   `gemma4d_metrics`.
5. Candidate stays under the tiny16 memory gate on the focused 1K chat/tool
   slice.
6. Runtime defaults remain unchanged when the endpoint is not called.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-server --lib
cargo test -p gemma4d-bench --example xr11_persistent_native_server_ab --no-run
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr11_persistent_native_server_ab --no-run

GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- \
  --out-dir benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128 \
  --baseline-backend persistent-native \
  --clear-workload-ids \
  --workload-id chat_short_1k_001 \
  --workload-id tool_json_1k_001 \
  --repeats 3 \
  --max-new-tokens 64 \
  --candidate-prefix-warmup-tokens 128
```

## Result

Decision: `accept_candidate`

Evidence:

- `benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/records.jsonl`
- `benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/summary.json`
- `benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/report.md`
- `benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/blockers.md`
- `benchmarks/out/XR85-server-prefix-warm-policy/chat-tool-1k-prefix128/decision.md`

Verification completed:

- `cargo fmt --all --check`
- `git diff --check`
- `cargo test -p gemma4d-server --lib`
- `cargo test -p gemma4d-bench --example xr11_persistent_native_server_ab --no-run`
- `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr11_persistent_native_server_ab --no-run`
- Focused Metal/MLX XR11 run above.

The focused XR11 run passed with no blockers. Candidate warmups used `128` of
`1028` prompt tokens for both `chat_short_1k_001` and `tool_json_1k_001`, took
`2771.085 ms` and `1983.183 ms`, peaked at `6.776 GB`, and recorded
`prefix_warmups_total=2`, `prefix_warmup_tokens_total=256`, and
`prefix_warmup_seconds=4.754268`. Measured requests remained token/text
identical to the explicit persistent-native baseline for `6/6` records, and
request-path peak MLX stayed under the tiny16 gate at `7.324 GB`.

XR85 validates the server/load-time control surface and observability. It does
not justify automatic warmup or broad defaults by itself: the first cold
measured chat token improved `392.688 -> 73.314 ms`, but steady-state 1K
chat/tool request totals were mostly neutral. The next high-value step is to
rerun/productize scoped MTP against this server surface and protected aggregate
gate.
