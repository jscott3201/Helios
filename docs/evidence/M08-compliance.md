# M08 Compliance Matrix

## Scope

- Milestone: `milestones/M08-ssd-prefix-cache.md`
- Spec: `spec/06-kv-cache-offload-compression.md`
- References: `spec/10-correctness-evals-benchmarks.md`, `docs/decisions/0004-m08-ssd-cold-cache-json-envelope.md`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M08-T01 | Define persisted KV manifest and versioning. | `PersistedKvManifest`; `PersistedLayerManifest`; `SSD_MANIFEST_VERSION`; `SSD_BLOCK_FILE_VERSION`; schema update. | Complete | None. |
| M08-T02 | Implement block writer/reader and checksums. | `SsdPrefixCache::write_block`; `restore_before_prefill`; `checksum_block`; `validate_persisted_file`; corrupt-block test. | Complete | JSON fixture format, not final tensor format. |
| M08-T03 | Add SSD index and eviction policy. | `SsdIndexEntry`; persisted `index.json`; `SsdPrefixCache::evict_to_fit`; `ssd_lru_evicts_to_disk_budget`. | Complete | None. |
| M08-T04 | Add restore-before-prefill path. | `SsdPrefixCache::restore_before_prefill`; `SsdRestorePhase`; no-mid-decode test. | Complete | Native tensor import remains future work. |
| M08-T05 | Benchmark cold vs warm SSD TTFT. | `m08_ssd_benchmark`; `benchmarks/out/M08/ssd-benchmark.json`; `docs/evidence/M08.md`. | Complete for fixture path | Real-model TTFT remains later integration work. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| SSD-restored logits/tokens match fresh prefill for same mode. | `ssd_restore_before_prefill_matches_fresh_for_m08_context_lengths`; raw benchmark exact=true for 1K/4K/8K/16K. | Complete for fixture observations. |
| Corrupt block rejection test passes. | `ssd_wrong_namespace_and_corrupt_block_are_rejected`; benchmark `rejected_corrupt_block=true`. | Complete. |
| Wrong namespace rejection test passes. | `ssd_wrong_namespace_and_corrupt_block_are_rejected`; benchmark `rejected_wrong_namespace=true`. | Complete. |
| Benchmark shows raw read/write bytes and TTFT comparison. | `ssd-benchmark.json` records cold/warm TTFT ms and per-case write/read bytes; accounting records total bytes read/written. | Complete. |
| No mid-decode SSD fetch. | `SsdRestorePhase::MidDecode` rejects before file read; test asserts reads/bytes_read stay zero; benchmark records `no_mid_decode_ssd_fetch=true`. | Complete. |

## Spec Compliance Summary

- Compliant: SSD cold prefix persistence, restore-before-prefill only, wrong namespace rejection, corruption rejection, SSD byte accounting, disk LRU eviction, and cold-vs-warm fixture benchmark evidence.
- Intentionally deferred: real native tensor export/import, safetensors-compatible payloads, compression, and real-model TTFT performance claims.

## Risk

The main residual risk is integration depth: M08 validates the SSD cache contract and fixture restore semantics, but native tensor payload persistence and real TTFT gains depend on later native KV export/import work.
