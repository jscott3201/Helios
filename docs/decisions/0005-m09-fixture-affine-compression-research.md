# Decision Record: M09 Fixture Affine Compression Research

- Status: accepted
- Date: 2026-06-30
- Milestone: M09

## Context

M09 requires evaluation of MLX affine q8/q4 prefix-cache modes and Planar/Iso-style experiments. The native KV tensor export/import ABI is still not stable, so real MLX tensor compression cannot yet be validated end to end without prematurely committing to a storage layout.

## Decision

Implement M09 as a fixture-based compression research layer in `gemma4d-kv`:

- `CacheMode::MlxAffineQ8` and `CacheMode::MlxAffineQ4` are first-class namespace modes.
- Persisted manifests include `CompressionManifestMetadata` with mode, algorithm, bit width, affine scale format, experimental flag, and namespace-mode declaration.
- q8/q4 quality is evaluated with deterministic fixture logits for simple chat, JSON/tool, and code-review workloads at 16K, 32K, and 64K.
- BF16 remains the default fallback mode.
- Planar/Iso interfaces exist only behind the `planar-iso-experiments` feature and are not accepted by default.

## Consequences

- M09 can prove namespace isolation, metadata propagation, quality/memory reporting, and feature-gated experimental status now.
- The reported q8/q4 results are fixture research metrics, not real tensor quality guarantees.
- Real MLX tensor compression and native restore parity must be revisited after native KV export/import stabilizes.

## Evidence

- `crates/gemma4d-kv/src/lib.rs`
- `crates/gemma4d-kv/examples/m09_compression_eval.rs`
- `references/schemas/kv-cache-manifest.schema.json`
- `crates/gemma4d-tui/src/app.rs`
- `crates/gemma4d-tui/src/ui.rs`
- `docs/evidence/M09.md`
- `docs/evidence/M09-compliance.md`
- `benchmarks/out/M09/compression-eval.json` (generated and ignored)
