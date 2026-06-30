# 05 — Gemma 4 MTP Speculative Decoding

## Goal

Implement Gemma 4 MTP as a native target+assistant draft/verify loop, not as generic speculative decoding first.

## Initial mode

```text
temperature = 0
sampling = greedy only
draft_block_size = 1, then 2, then 4 after exactness
single active generation
active KV = BF16
adapters disabled
```

## Loop

```text
1. Prefill target on prompt.
2. Store target KV cache.
3. Keep last target hidden state and last accepted token.
4. Drafter proposes N tokens.
5. Target verifies drafted tokens in one pass.
6. Accept longest valid prefix.
7. Commit accepted target KV states.
8. Roll back rejected speculative states.
9. Emit accepted tokens.
10. Repeat until stop.
```

## Correctness invariant

For the same target execution mode:

```text
non-MTP greedy token sequence == MTP greedy token sequence
```

If this invariant fails, MTP must auto-disable for that config and report the failing fixture.

## Metrics

Record per run:

```text
draft_block_size
attempted_draft_tokens
accepted_draft_tokens
acceptance_rate
accepted_tokens_per_verify
target_verify_passes
decode_tokens_per_second
peak memory
rollback_count
```

## Interaction with adapters

MVP disables MTP when `adapter != none`. Later milestones enable MTP per adapter only after exactness tests pass for that adapter.

## Interaction with compressed KV

MTP exactness must be tested per KV mode. Compressed target + MTP must match compressed target without MTP. It is not required to match BF16 if compression changes the target behavior.
