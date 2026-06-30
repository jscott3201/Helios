# M07 Compliance Matrix

## Scope

- Milestone: `milestones/M07-kv-cache-core.md`
- Spec: `spec/06-kv-cache-offload-compression.md`
- References: `spec/10-correctness-evals-benchmarks.md`, `docs/decisions/0003-m07-ram-prefix-native-logical-handles.md`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M07-T01 | Create `gemma4d-kv` block/key types. | `KvNamespace`, `KvBlockKey`, `BlockId`, `LayerBlockMetadata`, `RamPrefixBlock`, `RestoredPrefix`. | Complete | None. |
| M07-T02 | Add native export/import for RAM prefix blocks or native-managed logical handles. | `NativeLogicalHandle`; `RamPrefixBlock::with_native_handle`; restore returns the handle. | Complete for logical-handle path | Native tensor serialization is deferred. |
| M07-T03 | Implement cache namespace hashing. | `KvNamespace::namespace_hash`; `namespace_hash_changes_for_required_fields`; evidence runner wrong-namespace checks. | Complete | None. |
| M07-T04 | Implement RAM LRU with memory budget. | `RamPrefixCache::new`, `insert`, `restore`, `accounting`; `ram_lru_evicts_to_budget_and_reports_accounting`. | Complete | None. |
| M07-T05 | Add restore-vs-fresh tests for 1K/4K/8K/16K. | `restore_matches_fresh_prefill_for_m07_context_lengths`; `m07_restore_matrix` report. | Complete for fixture observations | Real-model replay remains a later native integration step. |
| M07-T06 | Expose cache byte/accounting summaries through provider DTO used by TUI cache page. | `CacheSnapshot`; `RuntimeProvider::cache_snapshot`; `render_cache`; `cache_page_renders_m07_accounting_summary`; ignored TUI snapshots under `benchmarks/out/M07/tui-snapshots`. | Complete | File provider remains offline until runtime reports exist. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| Fresh prefill and RAM-restored logits/tokens match for same mode. | `restore_matches_fresh_prefill_for_m07_context_lengths`; `restore-matrix.json` exact=true for 1K/4K/8K/16K. | Complete for fixture observations. |
| Wrong model/template/hash blocks are rejected. | Namespace hash tests and evidence runner wrong model/template/prompt hash rejection. | Complete. |
| Memory accounting is visible. | `CacheAccountingSnapshot`; TUI cache page test; generated accounting report. | Complete. |
| No SSD dependency yet. | `CacheAccountingSnapshot.ssd_enabled=false`; evidence report `no_ssd_dependency=true`. | Complete. |

## Spec Compliance Summary

- Compliant: namespace hash coverage, RAM-only prefix cache, LRU byte budget, copy-on-write fork metadata, wrong-namespace rejection, corruption rejection, and TUI-visible accounting.
- Intentionally deferred: SSD cold tier, compressed KV quality gates, and real native tensor export/import.

## Risk

The main residual risk is integration depth: M07 validates the cache contract and fixture restore semantics, but real-model KV tensor restore will require native KV ABI work and parity gates in later milestones.
