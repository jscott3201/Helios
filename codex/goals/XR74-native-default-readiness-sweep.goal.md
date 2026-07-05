# XR74 - Native default-readiness sweep

## Objective

After XR72/XR73, decide whether the native graph/runtime path is ready for
broader defaulting from an operational and guardrail perspective. XR74 is not a
kernel optimization goal; it is a readiness gate.

## Current Evidence

- Server native prefill default evidence is strong for 4K/8K/16K and accepted
  sentinel rows exist for 8K/16K/24K.
- `serve --model-path PATH` selects `persistent-native` when `--backend` is
  omitted.
- XR65 made native grouped end-of-decode KV eval the runtime default.
- XR72 accepted runtime default against explicit per-layer on the five-workload
  native decode matrix and isolated remaining chat first-token tails to
  full-attention group eval, not collection or final sync.
- XR73 accepted scoped chat/tool MTP opt-in only, with exactness, oracle,
  default-overhead, and holdout gates clean.
- XR70/XR71 full-attention update candidates remain default-off because tail
  gates are not clean enough.
- MTP remains default-off broadly; XR73 protected aggregate speedup was
  `+19.235%`, below the `25%` broad default-on gate.

## Scope

- Server/default backend selection and rollback behavior.
- Admission and tokenizer guardrails for tiny16 memory protection.
- 8K, 16K, and 24K tiny16 sentinels on persistent-native server paths.
- Operator observability and rollback flags for native graph, decode policy,
  MTP, and experimental candidates.
- Benchmark ledger cleanup so default, opt-in, experimental, and historical
  helper-backed claims are clearly separated.
- Documentation updates for any changed default/readiness decision.

## Non-Goals

- Do not promote XR70/XR71 candidates from XR74.
- Do not enable broad MTP default-on.
- Do not add multimodal support.
- Do not claim production internet-facing serving readiness.
- Do not restart DSpark.

## Acceptance Criteria

1. Server default selection is verified by tests and documented behavior.
2. Admission/tokenizer guardrails fail closed for over-budget or unsupported
   prompt shapes.
3. 8K/16K/24K sentinels pass token identity, memory, and policy-application
   checks under the tiny16 envelope.
4. Operator surfaces expose enough state to distinguish native graph,
   helper-backed, MTP, cache, and experimental policy states.
5. Rollback flags are documented and verified for native graph and decode
   policy behavior.
6. `BENCHMARKS.md`, `README.md`, and any evidence docs distinguish accepted
   defaults from default-off candidates.
7. The final readiness decision is recorded as ready, not ready, or
   blocked-with-evidence with exact commands and artifacts.

## Verification Commands

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-server --lib
cargo test -p gemma4d-tui --all-targets
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
```

Accepted 8K/16K/24K sentinel artifacts come from XR62. XR74 reused those
artifacts because the XR74 source diff only widens `/health` and TUI operator
visibility; it does not change server generation, default backend selection,
native prefill policy application, or MTP behavior.

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --out-dir benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-8k --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --clear-workload-ids --workload-id code_review_rust_8k_001 --repeats 3 --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336 --baseline-backend persistent-native
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --out-dir benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-16k --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --clear-workload-ids --workload-id benchmark_qa_16k_001 --repeats 3 --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336 --baseline-backend persistent-native
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --out-dir benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-24k-low-n --model-path artifacts/models/gemma-4-12B-it-4bit --workloads benchmarks/workloads/real-contexts/workloads.jsonl --clear-workload-ids --workload-id long_repo_pack_24k_001 --repeats 2 --max-new-tokens 1 --max-context-tokens 32768 --memory-budget-mb 14336 --baseline-backend persistent-native
```

## Completion Rule

Complete XR74 only when the readiness decision is supported by source diffs,
tests, tiny16 sentinel artifacts, docs, rollback evidence, and benchmark-ledger
updates. If any default-readiness gate cannot be proven, record the blocker and
keep the native path scoped to the proven default surface.

## Result

Decision: `ready` for the current local persistent-native default surface.

XR74 added backend and native-prefill policy visibility to `/health` and the TUI
dashboard, preserving existing server default behavior and keeping all
experimental native/MTP candidates default-off. The readiness decision applies
to local operator/server use on the proven persistent-native path; it does not
claim production internet-facing serving readiness, broad MTP default-on, or
promotion of XR70/XR71 full-attention update candidates.

Evidence:

- Server default selection: `serve --model-path PATH` defaults to
  `persistent-native`; explicit `--backend stub`, `--backend real-helper`, and
  `--backend persistent-native` remain tested rollback/selection paths.
- Native prefill rollback: `GEMMA4D_USE_NATIVE_GRAPH=0`, unset native graph, and
  explicit native prefill env overrides skip the server-owned default policy.
- Admission/tokenizer guardrails: over-context, over-memory, native weight-floor,
  16K real workload, byte-density, and real workload corpus coverage tests pass
  in `cargo test -p gemma4d-server --lib`.
- Sentinels: XR62 artifacts under
  `benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-{8k,16k,24k-low-n}`
  passed token identity, tiny16 memory, and policy-application checks for
  explicit persistent-native vs default model-path persistent-native.
- Operator visibility: `/health` now exposes `backend`, `max_context_tokens`,
  `admission_prefill_chunked`, and `native_prefill`; the TUI dashboard renders
  backend and native prefill policy and has live-provider assertions.
- MTP/default-off state: XR73 accepted explicit scoped chat/tool MTP opt-in only;
  broad default-on remains below the protected aggregate gate and disabled by
  default.

Fresh XR74 verification passed:

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-server --lib
cargo test -p gemma4d-tui --all-targets
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
```
