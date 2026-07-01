# XR15 - MTP policy variance A/B

## Outcome

Run fresh native non-MTP versus repaired native MTP evidence on real-context
workloads, with repeated measured trials, so the XR14 replay policy can be
validated or rejected with variance data.

## Scope

- Baseline: native non-MTP greedy decode.
- Candidates: repaired native MTP block sizes `1` and `2`.
- Policy candidate: net-latency guard that enables MTP only when
  `draft_ms + verify_ms` beats baseline `decode_ms` by the configured threshold.
- Workloads should include at least the XR14 selected winners and guardrail
  workloads when memory pressure permits:
  - `benchmark_qa_4k_001`
  - `mtp_candidate_1k_001`
  - `chat_short_1k_001`
  - `code_review_rust_4k_001`
  - `mtp_candidate_4k_001`
  - `tool_json_1k_001`

## Required work

1. Record warmup and measured trials separately.
2. Record exact source files, generated files, commands, git SHA, deterministic
   workload seeds, token lengths, model/drafter artifact identities, and block
   sizes.
3. Preserve the MTP exactness gate: every selected MTP candidate must match
   native non-MTP output at temperature 0.
4. Separate acceptance rate from net speedup. Acceptance-only policies must not
   be accepted when they regress net decode phase.
5. Keep MTP disabled by default unless a later goal explicitly accepts a
   runtime policy change.

## Acceptance gates

- Required files exist under
  `benchmarks/out/XR15-mtp-policy-variance-ab/`:
  `records.jsonl`, `summary.json`, `report.md`, `blockers.md`, and
  `decision.md`.
- At least 3 measured trials for default policy claims; lower-N runs must be
  labeled `needs_more_data`.
- Every MTP record used for a speed claim is byte-identical to baseline.
- Peak MLX memory must stay under the configured tiny16 gate.
- A policy may not regress any non-selected workload by more than the configured
  regression gate.
- Decisions use the standard labels: `accept_candidate`, `keep_experimental`,
  `reject_candidate`, `needs_more_data`, or `blocked_with_evidence`.

## Non-goals

- Do not change runtime/native/FFI behavior in this goal.
- Do not enable MTP by default.
- Do not test sampling, adapters, compressed active KV, server default paths, or
  block sizes above `2`.

## Required artifacts

```text
benchmarks/out/XR15-mtp-policy-variance-ab/records.jsonl
benchmarks/out/XR15-mtp-policy-variance-ab/summary.json
benchmarks/out/XR15-mtp-policy-variance-ab/report.md
benchmarks/out/XR15-mtp-policy-variance-ab/blockers.md
benchmarks/out/XR15-mtp-policy-variance-ab/decision.md
```

## Completion rule

Stop when the decision file exists with measured-trial evidence or blockers
explain why the fresh variance run could not complete.
