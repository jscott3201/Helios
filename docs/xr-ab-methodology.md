# A/B benchmark methodology

## Definitions

- **Baseline**: current `main` behavior for the selected workload and mode.
- **Candidate**: one isolated change or one explicit config variant.
- **Run unit**: a single workload under a single backend/config/mode.
- **Trial**: repeated run unit used to estimate variance.
- **Decision**: accepted/rejected/deferred based on measured evidence.

## Minimum A/B record fields

```json
{
  "schema_version": 1,
  "goal": "XRxx",
  "run_id": "xrxx-...",
  "git_sha": "...",
  "git_status_short": "...",
  "model_identity": {
    "model_path": "...",
    "model_revision": "...",
    "config_sha256": "...",
    "tokenizer_sha256": "...",
    "safetensors_inventory_sha256": "..."
  },
  "workload_id": "...",
  "family": "...",
  "variant": "baseline|candidate",
  "backend": "helper|native|server_real_helper|server_native",
  "config": {},
  "trial_index": 0,
  "input_tokens": 0,
  "generated_tokens": 0,
  "model_load_ms": 0.0,
  "prefill_ms": 0.0,
  "decode_ms": 0.0,
  "total_ms": 0.0,
  "decode_token_latencies_ms": [],
  "decode_p50_ms": 0.0,
  "decode_p95_ms": 0.0,
  "decode_p99_ms": 0.0,
  "prefill_tps": 0.0,
  "decode_tps": 0.0,
  "peak_mlx_gb": 0.0,
  "active_kv_bytes": 0,
  "rss_mb": 0.0,
  "correctness": {},
  "notes": []
}
```

## Trial policy

Default for low-cost cases:

- 1 warmup trial ignored for latency summary.
- 3 measured trials minimum.
- Report median, min, max, p95, and coefficient of variation where possible.

For expensive 16K/32K cases:

- 1 warmup optional if memory pressure allows.
- 2 measured trials minimum.
- Mark reports as `low_n` if fewer than 3 measured trials.

## Acceptance policy

A candidate can be accepted only if:

- correctness gate passes;
- memory stays under tiny16 budget or has an explicit opt-in profile;
- p50 or p95 improves by at least the goal-specific threshold;
- no worse than 5% regression on non-target workloads unless the goal explicitly allows tradeoff;
- variance is recorded.

## Report format

Every goal writes:

- `records.jsonl`: raw records only.
- `summary.json`: aggregate machine-readable summary.
- `report.md`: human-readable table and charts/text summaries.
- `blockers.md`: blockers, failed hypotheses, and how to reproduce.
- `decision.md`: explicit final decision.
