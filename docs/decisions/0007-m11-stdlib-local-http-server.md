# Decision Record: M11 Stdlib Local HTTP Server

- Status: accepted
- Date: 2026-06-30
- Milestone: M11

## Context

M11 requires an OpenAI-compatible local server with streaming chat, model/adapters endpoints, health, metrics, request guards, and a live TUI attach provider. The milestone allows `axum`/`tokio` or a selected stack. `axum` and `hyper` were not already present in the lockfile, and M11 must remain offline-verifiable in the current workspace.

The compatibility target is the Chat Completions-style shape: `POST /v1/chat/completions` accepts `model`, `messages`, optional `stream`, and returns chat completion objects or streaming chat completion chunks. Official API reference consulted: <https://platform.openai.com/docs/api-reference/chat/create>.

## Decision

Implement M11 with a small Rust stdlib HTTP/1.1 stack:

- Bind to `127.0.0.1:8080` by default.
- Reject non-local bind addresses in the M11 CLI path.
- Return OpenAI-compatible JSON for non-streaming chat completions.
- Return `text/event-stream` SSE frames for streaming chat completions, ending with `data: [DONE]`.
- Expose Prometheus-style `/metrics`.
- Expose stable read-only/stub control JSON for the TUI.
- Keep adapter loading constrained to known local registry IDs; do not expose remote/path adapter load fields.
- Use deterministic stub generation for tests and evidence; real native model serving remains a later integration layer.

## Consequences

- M11 can be built, tested, and smoked without network dependency downloads or a full model artifact.
- The server stack is intentionally minimal and should be revisited if production HTTP needs grow beyond the localhost developer surface.
- The TUI can attach to live localhost JSON/SSE control endpoints while preserving mock and file providers.

## Evidence

- `crates/gemma4d-server/src/http.rs`
- `crates/gemma4d-server/src/lib.rs`
- `crates/gemma4d-server/examples/m11_server_smoke.rs`
- `crates/gemma4d-tui/src/provider.rs`
- `crates/gemma4d-tui/src/app.rs`
- `crates/gemma4d-tui/src/ui.rs`
- `docs/evidence/M11.md`
- `docs/evidence/M11-compliance.md`
- `benchmarks/out/M11/server-smoke.json` (generated and ignored)
