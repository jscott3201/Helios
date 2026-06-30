# M09 Compliance Matrix

## Scope

- Milestone: `milestones/M09-kv-compression-research.md`
- Spec: `spec/06-kv-cache-offload-compression.md`
- References: `spec/10-correctness-evals-benchmarks.md`, `docs/decisions/0005-m09-fixture-affine-compression-research.md`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M09-T01 | Implement MLX affine q8/q4 prefix cache modes. | `CacheMode::MlxAffineQ8`; `CacheMode::MlxAffineQ4`; `fixture_block_with_mode`; q8/q4 byte estimates; tests. | Complete for fixture cache modes | Native MLX tensor compression remains future work. |
| M09-T02 | Add compression metadata to manifest. | `CompressionManifestMetadata`; `PersistedKvManifest.compression`; schema update. | Complete | None for fixture format. |
| M09-T03 | Create Planar/Iso experiment interface behind feature flag. | Cargo feature `planar-iso-experiments`; `ExperimentalCompressionMode`; `ExperimentalCompressionPlan`; feature-gated test. | Complete | No production enablement. |
| M09-T04 | Run quality comparisons: logit cosine, greedy agreement, JSON/tool fixtures. | `evaluate_compression_fixture`; `m09_compression_eval`; raw report includes JSON/tool workload. | Complete for fixture logits | Real-model logits remain future work. |
| M09-T05 | Run memory/speed comparisons at 16K/32K and 64K if possible. | `compression-eval.json` includes 16K, 32K, 64K q8/q4 memory deltas and `eval_us`. | Complete for fixture path | Real TTFT/native speed remains future work. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| BF16 fallback remains default. | `KvNamespace::fixture` defaults to `CacheMode::Bf16`; test `bf16_fallback_remains_default_cache_mode`; raw report `bf16_fallback_default=true`. | Complete. |
| q8/q4 results include quality and memory deltas. | `compression-eval.json`; docs/evidence/M09 quality table. | Complete. |
| Planar/Iso remains experimental unless all gates pass. | Feature-gated API; raw report `feature_enabled=false`, `accepted_by_default=false`; feature test. | Complete. |
| Compression never silently changes cache namespace semantics. | Namespace includes `cache_mode`; q8/q4 namespace/block IDs differ from BF16; manifest metadata declares namespace hash includes mode. | Complete. |

## Spec Compliance Summary

- Compliant: q8/q4 fixture modes, compression metadata, namespace isolation, quality/memory/speed evidence, Planar/Iso feature-gating, BF16 default fallback.
- Intentionally deferred: real MLX affine tensor compression, Planar/Iso quality gates, fused Metal decode, and real native KV restore parity.

## Risk

The main residual risk is fidelity: fixture logits and byte estimates are useful for plumbing and gate shape, but real Gemma 4 KV tensor compression must be validated later with native export/import and reference comparisons.
