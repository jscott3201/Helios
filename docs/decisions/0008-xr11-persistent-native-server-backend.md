# Decision Record: XR11 Persistent Native Server Backend

- Status: accepted
- Date: 2026-07-01
- Goal: XR11

## Context

P02 `real-helper` server mode exercises the OpenAI-compatible HTTP route, but it
calls `generate` for every request. That means tokenizer work, target load, KV
creation, prefill/decode, and detokenization all happen in the request path. XR11
needs an A/B target that keeps model state resident across repeated server
requests without widening the Rust/native boundary or changing default serving
behavior.

## Decision

Add an explicit experimental backend named `persistent-native`:

- It is opt-in through `--backend persistent-native --model-path PATH` and the
  `GEMMA4D_EXPERIMENTAL_PERSISTENT_SERVER=1` gate.
- It keeps the existing localhost-only bind guard and the default backend remains
  `stub`.
- A dedicated worker thread owns the resident `ResidentTarget`; HTTP connection
  threads send generation requests over a channel and wait for the reply.
- The active HTTP admission guard still allows only one active generation when
  queue capacity is zero. The worker also serializes native access, so future
  queueing cannot mutate the resident target concurrently.
- Each request creates a fresh KV cache and uses the existing tokenizer and
  detokenizer helpers. XR11 does not optimize runtime kernels, tokenizer cost,
  or KV reuse.
- `/metrics` and `/v1/runtime/snapshot` expose resident load count, load
  seconds, model-loaded status, and worker request count for TUI/provider use.

## Consequences

- Repeated persistent-server requests can avoid repeated target loads while
  preserving deterministic greedy output checks against `real-helper`.
- Startup/load failures are reported through the backend snapshot and request
  errors; the server does not silently fall back to a different backend.
- No broad MLX internals are exposed to Rust callers. The persistent target
  remains behind the existing opaque FFI wrapper and is not shared across HTTP
  threads.
- Performance claims must come from XR11 artifacts under
  `benchmarks/out/XR11-persistent-native-server-ab/`.

## Amendment: XR53 default model-path serving

Date: 2026-07-03

XR53 promotes the accepted PersistentNative path out of the experimental CLI
gate for model-backed serving:

- `gemma4d serve --model-path PATH` now defaults to `persistent-native` when no
  backend flag is explicit.
- Zero-arg/no-model-path serving and explicit `--backend stub` remain the M11
  stub behavior.
- Explicit `--backend real-helper --model-path PATH` remains available as a
  helper-backed opt-out.
- Admission memory estimates now use measured XR51/P04 native memory constants
  rather than the original stub-era `(prompt + max_tokens) * 4096` estimate.

## Evidence

- `crates/gemma4d-server/src/lib.rs`
- `crates/gemma4d-server/src/http.rs`
- `crates/gemma4d-bench/examples/xr11_persistent_native_server_ab.rs`
- `codex/goals/XR11-persistent-native-server-ab.goal.md`
- `codex/goals/XR53-server-default-backend-estimator.goal.md`
