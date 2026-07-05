# XR74 - Native default-readiness sweep

## Objective

After XR72, decide whether the native graph/runtime path is ready for broader
defaulting from an operational and guardrail perspective. XR74 is not a kernel
optimization goal; it is a readiness gate.

## Current Evidence

- Server native prefill default evidence is strong for 4K/8K/16K and accepted
  sentinel rows exist for 8K/16K/24K.
- `serve --model-path PATH` selects `persistent-native` when `--backend` is
  omitted.
- XR65 made native grouped end-of-decode KV eval the runtime default.
- XR70/XR71 full-attention update candidates remain default-off because tail
  gates are not clean enough.
- MTP remains default-off; broad default-on is not supported by protected
  aggregate evidence.

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

- Do not promote XR70/XR71 candidates unless XR72 produces cleaner evidence.
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

Add the exact 8K/16K/24K sentinel commands after XR72 determines which native
decode policy is under readiness review.

## Completion Rule

Complete XR74 only when the readiness decision is supported by source diffs,
tests, tiny16 sentinel artifacts, docs, rollback evidence, and benchmark-ledger
updates. If any default-readiness gate cannot be proven, record the blocker and
keep the native path scoped to the proven default surface.
