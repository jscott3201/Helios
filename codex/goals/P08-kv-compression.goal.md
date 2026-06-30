# P08 - Real KV Compression Quality and Memory Gates

```text
Move KV compression evaluation from synthetic fixtures to real-model prefix-cache
payloads. Implement and measure MLX affine q8/q4 compression for global/full-
attention prefix blocks first, with BF16 as the baseline. Compare restored
logits, greedy agreement, memory reduction, TTFT, and decode impact at
4K/8K/16K. Keep Planar/Iso experiments behind a feature flag and only add
reportable candidates if real evidence exists. Produce
benchmarks/out/P08-kv-compression/{records.jsonl,summary.json,report.md}. Do
not enable compressed active decode by default.
```

## Outcome

Compression decisions become evidence-backed on real Gemma 4 12B tensors.

## Verification Surface

- BF16 vs q8/q4 real-model logits.
- Memory reduction and latency tables.
- Quality gate pass/fail.
- Feature-gated Planar/Iso report if implemented.
- `make verify`.

## Boundaries

- Prefix cache first.
- Global/full layers first.
- No default active compressed decode.

## Completion Rule

Mark this goal complete only when the evidence artifacts exist and the
verification commands have been run, or when the goal is blocked with a blocker
report that lists exact commands attempted, observed output, and the next
required input.
