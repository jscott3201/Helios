# Decision Record: M08 SSD Cold Cache JSON Envelope

- Status: accepted
- Date: 2026-06-30
- Milestone: M08

## Context

M08 requires inactive prefix KV blocks to be persisted to SSD and restored before prefill. The KV spec allows either a safetensors-compatible format or a simple packed internal format with checksums. The native KV ABI still uses logical handles and fixture observations for milestone-level restore validation.

## Decision

Implement the first SSD cold-cache format as a versioned JSON envelope containing:

- `PersistedKvManifest` with manifest and block-file versions.
- Full cache namespace identity, including quantization, prompt hashes, adapter fields, KV layout, cache mode, MLX version, and engine version.
- Gemma 4 layer layout metadata and per-layer checksums.
- A serialized `RamPrefixBlock` plus block checksum.
- A persistent `index.json` with block paths, logical bytes, stored bytes, and LRU ordering.

SSD restore is exposed only through `restore_before_prefill`; `MidDecode` restore attempts fail before any file read.

## Consequences

- M08 can validate SSD restore, namespace rejection, corruption rejection, byte accounting, and no mid-decode fetches without committing to a native tensor ABI.
- The JSON format is inspectable and easy to corrupt in tests.
- It is not intended as the final high-performance tensor storage format. M09/M12 can replace or extend it with packed/safetensors-compatible tensor payloads once real KV export/import stabilizes.

## Evidence

- `crates/gemma4d-kv/src/lib.rs`
- `crates/gemma4d-kv/examples/m08_ssd_benchmark.rs`
- `references/schemas/kv-cache-manifest.schema.json`
- `docs/evidence/M08.md`
- `docs/evidence/M08-compliance.md`
- `benchmarks/out/M08/ssd-benchmark.json` (generated and ignored)
