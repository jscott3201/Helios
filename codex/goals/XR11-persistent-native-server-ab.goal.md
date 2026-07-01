# XR11 — Persistent Native/Server Backend A/B

## Outcome

Design and benchmark a persistent server backend that holds model state across
requests and optionally uses the native graph, then compare it with current
real-helper-per-request behavior.

## Required Work

1. Map current server call path and confirm where model load/tokenizer/detokenizer costs are paid.
2. Add a feature-gated persistent backend design before implementation.
3. Implement the smallest safe persistent backend: single active generation, localhost-only, one resident target, clear error/fallback path.
4. A/B server requests over real workloads against current `real-helper` server.
5. Surface real metrics through `/metrics` and `/v1/runtime/snapshot` for TUI.
6. Preserve queue/admission guards.

## Acceptance

- Server response text/tokens match selected baseline for deterministic greedy.
- Repeated requests avoid model reload cost.
- No unsafe concurrent native handle mutation.
- TUI can display real backend metrics without perturbing runtime.

## Required Artifacts

Produce:

- `benchmarks/out/XR11-persistent-native-server-ab/records.jsonl`
- `benchmarks/out/XR11-persistent-native-server-ab/summary.json`
- `benchmarks/out/XR11-persistent-native-server-ab/report.md`
- `benchmarks/out/XR11-persistent-native-server-ab/blockers.md`
- `benchmarks/out/XR11-persistent-native-server-ab/decision.md`

Stop only when `decision.md` exists with raw evidence, or `blockers.md`
explains why the goal is blocked.
