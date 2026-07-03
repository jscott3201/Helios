# M11 Compliance Matrix

## Scope

- Milestone: `milestones/M11-openai-server.md`
- Spec: `spec/08-serving-api-runtime.md`
- References: `spec/09-observability-profiling.md`, `spec/10-correctness-evals-benchmarks.md`, `spec/11-security-licensing-safety.md`, `spec/13-tui-operator-ux.md`, `docs/decisions/0007-m11-stdlib-local-http-server.md`

XR53 amendment (2026-07-03): M11 compliance remains scoped to the zero-arg
stub server and explicit `--backend stub` behavior. Model-backed serve configs
now default to PersistentNative when `--model-path` is present and no backend is
explicit, and the admission memory guard uses measured XR51/P04 constants
instead of the original stub-era token-count estimate.

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M11-T01 | Add `gemma4d-server` using axum/tokio or selected stack. | `gemma4d-server::http`; `gemma4d serve`; decision `0007`. | Complete | Selected stack is stdlib HTTP/1.1 for localhost M11, not axum. |
| M11-T02 | Implement `/v1/chat/completions` non-streaming then streaming. | `ServerRuntime::chat_completions_response`; tests `chat_completion_non_streaming_matches_openai_shape`, `streaming_chat_completion_uses_sse_done`; smoke report. | Complete | Stub generation only. |
| M11-T03 | Implement `/v1/models`, `/v1/adapters`, `/health`, `/metrics`. | `handle_request` route table; tests and smoke report. | Complete | Adapter endpoints mutate only known local IDs. |
| M11-T04 | Add request admission and memory guard errors. | `admit_chat_request`; tests `admission_and_memory_guard_return_stable_error_codes`, `admission_estimator_matches_f8_memory_table`, `admission_guard_rejects_16k_unchunked_and_admits_chunked`; smoke report context/memory guard gates. | Complete | XR53 keeps deterministic prompt estimation but applies measured real-server memory constants and a `13/10` BPE correction for admission. |
| M11-T05 | Add integration tests with small/stub backend and optional full model tests. | Server listener smoke; TUI live attach smoke; `m11_server_smoke` evidence runner. | Complete | Optional full-model tests are not required for M11 stub scope. |
| M11-T06 | Implement TUI `HttpProvider` attach path for health, metrics, adapters, cache summaries, benchmark status, and streaming chat status. | `HttpProvider`; `ChatSnapshot`; live attach test; TUI snapshots. | Complete | Runtime events are SSE stub; no benchmark execution is spawned. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| Simple streaming chat works locally. | `/v1/chat/completions` SSE response; server listener smoke; TUI live attach streaming smoke. | Complete. |
| Adapter selection field is parsed and routed. | `ChatCompletionRequest.adapter`; server adapter routing test; smoke report `adapter_selection_routed=true`. | Complete. |
| Metrics endpoint exposes core counters. | `/metrics`; metrics test; smoke report `metrics_endpoint_core_counters=true`. | Complete. |
| Server binds localhost by default. | `ServerConfig::default`; `parse_bind_addr`; tests; smoke report `bind_localhost_default=true`. | Complete. |
| Unsafe remote adapter loading not exposed. | `adapter_mutation_response` rejects `source`, `path`, and `url`; tests; smoke report. | Complete. |
| TUI can attach to localhost server/control provider and display live health/metrics with stub backend. | `HttpProvider`; test `http_provider_attaches_to_live_server_and_streams_chat`; Chat/Dashboard/Adapters/Cache render paths. | Complete. |

## Spec Compliance Summary

- Compliant: localhost OpenAI-compatible chat completions subset, streaming SSE, model/adapters/health/metrics endpoints, stable JSON errors, request and memory guards, read-only/stub TUI control endpoints, TUI live HTTP provider, remote adapter loading disabled, and deterministic evidence. XR53 extends model-path serving outside M11 stub scope by defaulting accepted PersistentNative backend selection when `--model-path` is present.
- Intentionally deferred: production HTTP framework, full native model-backed chat server, authenticated non-localhost serving, real benchmark execution through HTTP, runtime event stream beyond stub snapshots, and native scheduler queueing.

## Risk

The main residual risk is server-stack maturity. The stdlib HTTP stack is sufficient for M11 localhost tests and evidence, but a production-quality server should revisit axum/hyper or another maintained stack once network dependency management is acceptable.
