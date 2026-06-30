# Decision Record: M07 RAM Prefix Blocks With Native Logical Handles

- Status: accepted
- Date: 2026-06-30
- Milestone: M07

## Context

M07 requires an in-memory KV cache core with logical block metadata, namespace hashing, RAM LRU residency, copy-on-write conversation forks, exact restore tests, and TUI-visible accounting. The milestone permits either native export/import of RAM prefix blocks or native-managed logical handles.

The native Gemma 4 graph is still expanding its incremental KV execution surface. Serializing real MLX tensors through Rust at this stage would force premature ABI and layout commitments before M08/M09 cover SSD cold tiers and KV compression modes.

## Decision

Implement the M07 Rust cache core around stable metadata, namespace hashes, block IDs, RAM LRU policy, and native-managed logical handle metadata. `RamPrefixBlock` carries a `NativeLogicalHandle` when the block is owned by the native runtime; Rust validates namespace/checksum/accounting and returns the handle on restore without claiming tensor serialization semantics.

The namespace hash includes model identity, revision, weight hash, quantization hash, tokenizer hash, chat-template hash, prompt token-prefix hash, raw prompt hash, adapter identity/hash, KV layout version, cache mode, MLX version, and engine version.

## Consequences

- M07 can verify exact restore-vs-fresh behavior and wrong-namespace rejection without SSD or tensor serialization.
- The TUI can display RAM prefix accounting through a provider DTO now.
- Future native export/import can be added behind the same block/key namespace contract when the native KV ABI stabilizes.
- SSD cold cache and compressed KV remain out of scope for M07.

## Evidence

- `crates/gemma4d-kv/src/lib.rs`
- `crates/gemma4d-kv/examples/m07_restore_matrix.rs`
- `crates/gemma4d-tui/src/app.rs`
- `crates/gemma4d-tui/src/provider.rs`
- `crates/gemma4d-tui/src/ui.rs`
- `docs/evidence/M07.md`
- `docs/evidence/M07-compliance.md`
- `benchmarks/out/M07/restore-matrix.json` (generated and ignored)
