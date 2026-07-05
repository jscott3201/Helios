# XR74 Native Default Readiness

Date: 2026-07-05

Decision: `ready` for the current local persistent-native default surface.

## Scope

XR74 is a readiness gate, not a speed promotion. The decision covers local
operator/server use of the current persistent-native default path with native
prefill policy and runtime-default decode already accepted by earlier evidence.

Out of scope: production internet-facing serving, broad MTP default-on,
multimodal, DSpark, and promotion of XR70/XR71 full-attention update candidates.

## Evidence

| Gate | Result | Evidence |
|---|---|---|
| Server default selection | Pass | `serve --model-path PATH` defaults to `persistent-native`; explicit `--backend stub`, `--backend real-helper`, and `--backend persistent-native` remain tested selection/rollback paths in `crates/gemma4d-server/src/lib.rs`. |
| Admission/tokenizer guardrails | Pass | `cargo test -p gemma4d-server --lib` covers context-too-large, memory guard, native weight-floor charging, real 16K fail-closed behavior, chunked admission estimates, and workload-corpus token estimate coverage. |
| Tiny16 sentinels | Pass | XR62 sentinel artifacts: 8K `3/3` token identity, 16K `3/3`, 24K low-N `2/2`; policy `applied -> applied`; peak MLX stayed `7.402..7.859 GB`. |
| Operator visibility | Pass | XR74 exposes `backend`, `max_context_tokens`, `admission_prefill_chunked`, and `native_prefill` through `/health`; the TUI dashboard renders backend and native prefill policy. |
| Rollback | Pass | Native graph rollback remains available through `GEMMA4D_USE_NATIVE_GRAPH=0`; native prefill policy override is respected by explicit native prefill env vars; server backend rollback remains CLI-selectable. |
| MTP/default-off boundaries | Pass | XR73 accepted explicit scoped chat/tool MTP opt-in only; broad default-on remains disabled because protected aggregate speed was below the `25%` gate. |

## Fresh XR74 Verification

```text
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-server --lib
cargo test -p gemma4d-tui --all-targets
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run
```

All commands passed on 2026-07-05.

## Sentinel Artifacts

XR74 reuses XR62 sentinel artifacts because XR74 changed only health/TUI
operator visibility, not generation, server backend selection, native prefill
policy application, or MTP behavior.

```text
benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-8k
benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-16k
benchmarks/out/XR61-adaptive-n-mtp/server-default-sentinel-24k-low-n
```

Exact sentinel commands are recorded in
`codex/goals/XR74-native-default-readiness-sweep.goal.md` and `BENCHMARKS.md`.

## Follow-Up

The next high-value work is native full-attention group-eval follow-up based on
XR72's profile evidence. Broader MTP promotion remains parked until protected
aggregate speed clears the release gate.
