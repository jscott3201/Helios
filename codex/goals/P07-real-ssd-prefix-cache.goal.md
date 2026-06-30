# P07 - Real SSD Prefix Cache Payloads

```text
goal Persist real prefix KV payloads to SSD and restore them before prefill only. Use a manifest plus binary or safetensors-compatible payload format with checksums, shape metadata, per-layer attention metadata, cache mode, and namespace hash. Verify warm SSD TTFT improvement over cold prefill, corruption rejection, namespace rejection, and zero mid-decode SSD fetches. Produce benchmarks/out/P07-real-ssd-prefix-cache/{records.jsonl,summary.json,report.md}. Keep SSD cache disabled by default in tiny16 until the report supports enabling it.
```

## Outcome

SSD prefix cache becomes a real warm-start mechanism, not just metadata or
fixture coverage.

## Verification Surface

- Native snapshot payload save/load through the narrow C ABI.
- Safetensors-compatible SSD payload plus manifest/checksum metadata.
- Existing SSD namespace/manifest gate before native payload import.
- Fresh prefill greedy token/logit vs restored SSD last-step greedy token/logit.
- One continued `decode_one` after restore vs the cold-cache continuation.
- Warm SSD TTFT improvement over cold prefill.
- Wrong namespace, adapter, and cache-mode rejection before payload import.
- Payload corruption rejection.
- Explicit mid-decode SSD restore rejection with zero mid-decode fetches.
- Bytes read/written and restore latency metrics.
- `make verify`.

## Boundaries

- Text-only greedy inference.
- Restore before prefill only.
- No mid-decode SSD paging.
- No weight offload.
- Keep SSD disabled by default for tiny16 unless later evidence justifies
  enabling it.

## Completion Rule

Mark this goal complete only when the evidence artifacts exist and the
verification commands have been run, or when the goal is blocked with a blocker
report that lists exact commands attempted, observed output, and the next
required input.

## Suggested Subagents

- `performance-analyst` for TTFT and I/O variance interpretation.
- `gemma4_correctness_reviewer` for restored-logit and continued-decode parity.
- `security-reliability-reviewer` for namespace, corruption, and mid-decode
  rejection behavior.
- `test-verifier` for final build/test/lint verification.
