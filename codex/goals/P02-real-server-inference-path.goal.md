# P02 — Real Helper-Backed Server Inference Path

```text
goal Replace or augment the localhost /v1/chat/completions stub with an opt-in real helper-backed generation path for temperature=0, single active generation, and text-only Gemma 4 12B. Preserve the current stub fallback behind config. Implement non-streaming and streaming SSE responses with accurate prefill/decode/token/memory metrics. Verify with curl fixtures, server smoke tests, and a 1K/4K/8K/16K server benchmark report under benchmarks/out/P02-real-server-inference/. Keep localhost-only safety behavior and make verify green.
```

## Outcome

The OpenAI-compatible server has a meaningful opt-in real inference surface rather than only the M11 control-plane stub.

## Verification surface

- Real non-streaming `/v1/chat/completions` smoke.
- Real streaming SSE `/v1/chat/completions` smoke.
- `/metrics` shows real prompt/decode token counts plus helper load, prefill, decode, and memory values.
- `benchmarks/out/P02-real-server-inference/{records.jsonl,summary.json,report.md,blockers.md}` exists.
- Benchmark report compares the real server path with the P01 warm helper/session path.
- `make verify`.

## Boundaries

- Temperature 0 only.
- Localhost serving only.
- Text-only Gemma 4 12B.
- No remote serving.
- No real adapter math.
- No MTP-on-server unless separately proven.

## Completion rule

Mark this goal complete only when evidence artifacts exist and verification commands have been run, or when the goal is blocked with a blocker report that lists exact commands attempted, observed output, and the next required input.

## Suggested subagents

- `performance-analyst` for benchmark interpretation.
- `test-verifier` for final build/test/lint verification.
- `security-reliability-reviewer` for localhost, adapter, and terminal/server risk review.
