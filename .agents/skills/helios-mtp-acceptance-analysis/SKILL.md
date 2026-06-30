---
name: helios-mtp-acceptance-analysis
description: Diagnose Gemma 4 MTP acceptance and speculative decoding behavior in Helios using real-context prompts, trace records, and exactness gates.
---

# Helios MTP acceptance analysis skill

Use this skill for MTP, speculative decoding, drafter, assistant model, verify/rollback, accepted tokens, or block-size tuning work.

## Required discipline

- Do not make random MTP fixes before trace evidence exists.
- MTP output must match non-MTP native output at temperature 0.
- Record draft token, target token, top-k when feasible, accepted count, rollback, verify latency, shared KV shapes, and position offsets.
- Compare short prompts and real-context prompts separately.
- Treat acceptance improvement and net speedup as separate metrics.
- Keep MTP disabled by default unless a Goal explicitly accepts a gated default change.

## Evidence to return

- Acceptance by workload family.
- Accepted tokens per verify pass.
- Exactness result.
- Speedup/slowdown after load and prefill are separated.
- Top likely root cause if acceptance is poor.
