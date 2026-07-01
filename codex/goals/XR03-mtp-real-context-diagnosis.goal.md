# XR03 — MTP real-context acceptance diagnosis

## Outcome

Diagnose why real native MTP currently accepts no draft tokens on P05 short probes, using realistic prompts and trace-level evidence. Do not make random MTP fixes before the trace exists.

## Required work

1. Extend or add an MTP trace harness over XR00 `mtp_candidate`, `chat_short`, `code_review_rust`, and `benchmark_qa` workloads.
2. Capture per-draft-token data: draft token, target greedy token, target top-k, logit margin, accepted count, verify time, sequence length, shared KV shapes, and position offsets.
3. Compare block sizes 1 and 2; optionally design support for 3/4 but do not enable without exactness gates.
4. Check target/drafter model identity and revision/hash compatibility.
5. Identify whether low acceptance is due to workload, model/artifact mismatch, implementation mismatch, or verifier inefficiency.
6. Write fix hypotheses ranked by expected payoff and risk.

## Verification surface

- MTP output remains byte-identical to non-MTP native output.
- Trace artifacts can reproduce the zero/low acceptance finding.
- If acceptance is nonzero on any workload, quantify where and why.

## Decision

Valid decisions include `needs_more_data`, `blocked_with_evidence`, or `accept_candidate` for a specific fix plan. Do not enable MTP by default in this goal.


## Non-goals

- Do not make broad model support changes.
- Do not claim production serving readiness.
- Do not remove existing P00-P10 benchmark harnesses.
- Do not hide failed hypotheses; write them to `blockers.md`.

## Required artifacts

```text
benchmarks/out/XR03-mtp-real-context-diagnosis/records.jsonl
benchmarks/out/XR03-mtp-real-context-diagnosis/summary.json
benchmarks/out/XR03-mtp-real-context-diagnosis/report.md
benchmarks/out/XR03-mtp-real-context-diagnosis/blockers.md
benchmarks/out/XR03-mtp-real-context-diagnosis/decision.md
```

## Completion rule

Stop only when the decision file exists and is backed by raw evidence, or when `blockers.md` explains why the goal cannot proceed without external input.
