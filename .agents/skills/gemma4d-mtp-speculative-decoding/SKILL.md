---
name: gemma4d-mtp-speculative-decoding
description: Use for Gemma 4 MTP speculative decoding: drafter loading, draft/verify/accept/rollback, greedy exactness, acceptance metrics, and MTP auto-disable.
---
# Gemma4D MTP Speculative Decoding

## Trigger

Use for Gemma 4 MTP assistant, speculative decoding, draft verification, rollback, and MTP profiling.

## Invariants

- Start with `temperature=0`.
- MTP greedy must match non-MTP greedy for the same target/KV/adapter mode.
- Rollback must be tested.
- Adapters are disabled until per-adapter exactness is proven.
- If exactness fails, auto-disable and report fixture.

## Metrics

Report attempted drafts, accepted drafts, acceptance rate, accepted tokens per verify, rollbacks, decode tok/s, and peak memory.
