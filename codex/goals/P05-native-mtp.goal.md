# P05 - True Native MTP Block Verification

```text
goal Connect Gemma 4 MTP to the real native FFI path by using native drafter loading, gemma4_mtp_draft_block, and gemma4_verify_tokens rather than scripted trait fixtures. At temperature=0, MTP output must be byte-identical to the same native target without MTP. Measure acceptance rate, accepted tokens per verify pass, rollback count, peak memory, and decode speed for block_size=1 and block_size=2. Produce benchmarks/out/P05-native-mtp/{records.jsonl,summary.json,report.md}. Keep MTP disabled by default unless the evidence exceeds the configured acceptance and correctness gates.
```

## Outcome

MTP becomes a measured native runtime feature instead of only an algorithm
fixture.

## Verification Surface

- Native non-MTP greedy output vs real native MTP output.
- Native drafter load through the FFI.
- Native `gemma4_mtp_draft_block` and `gemma4_verify_tokens`.
- Acceptance rate, accepted tokens per verify pass, verify passes, rollbacks,
  decode speed, and peak memory.
- Auto-disable/fallback behavior when acceptance or memory gates fail.
- `make verify`.

## Boundaries

- Text-only.
- Greedy / temperature 0 only.
- Block size 1 and 2 only.
- No adapter-active MTP.
- No compressed active KV.
- No sampling MTP.

## Completion Rule

Mark this goal complete only when the evidence artifacts exist and the
verification commands have been run, or when the goal is blocked with a blocker
report that lists exact commands attempted, observed output, and the next
required input.

## Suggested Subagents

- `performance-analyst` for metric interpretation.
- `gemma4_correctness_reviewer` for native target/MTP exactness.
- `test-verifier` for final build/test/lint verification.
