# XR42 - Rayon manifest hashing A/B

## Outcome

Explore Rayon for CPU-side benchmark artifact preparation only, starting with
safetensors manifest hashing. Do not change MLX/native inference, decode,
prefill, MTP, server behavior, or default manifest behavior.

## Source Context

- Rayon upstream describes data-parallel iterators, fork-join primitives, and
  custom thread pools for Rust data parallelism:
  https://raw.githubusercontent.com/rayon-rs/rayon/main/README.md
- This pass intentionally does not use `wide` or `wgpu`; those remain future
  candidates after profiling proves a scalar numeric loop or an isolated
  non-MLX GPU experiment is worth testing.

## Scope

- Add Rayon as a `gemma4d-bench` dev-dependency only.
- Add a standalone benchmark example for sequential vs bounded Rayon hashing of
  safetensors inventories.
- Preserve deterministic entry ordering and inventory hashes.
- Record exact commands, generated files, thread counts, artifact paths, file
  counts/bytes, inventory hashes, timings, blockers, and decision.
- Record deterministic seed metadata even though the workload uses no
  randomness.
- Record token lengths as not applicable because no tokenizer or model run is
  involved.

## Required Work

1. Add a standalone XR42 harness under `crates/gemma4d-bench/examples/`.
2. Default artifacts:
   - `artifacts/models/gemma-4-12B-it-4bit`
   - `artifacts/models/gemma-4-12B-it-qat-assistant-4bit`
3. Compare variants:
   - `sequential`
   - `rayon_threads_1`
   - `rayon_threads_2`
   - `rayon_threads_4`
4. Use sorted relative safetensors paths for deterministic inventory hashing.
5. Reject any Rayon variant whose inventory hash differs from sequential.
6. Do not integrate Rayon into `manifest::capture_artifact_identity` in this
   goal; this pass only establishes evidence.
7. Update `BENCHMARKS.md` with headline and run ledger data.

## Acceptance Gates

- `cargo fmt --all --check` passes.
- `cargo check -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab`
  passes.
- XR42 writes:

```text
benchmarks/out/XR42-rayon-manifest-hashing-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

- Every Rayon candidate has the same inventory hash as sequential for each
  artifact.
- Any performance claim is based on per-artifact p50 timing from the generated
  summary, not a single trial.

## Completion Rule

Stop when the harness, one local run, `BENCHMARKS.md` update, and commit/push
are complete, or when dependency acquisition/compile evidence blocks the
experiment.

## Result

Decision: `accept_candidate_for_followup`.

Added Rayon as a `gemma4d-bench` dev-dependency only and added the standalone
`xr42_rayon_manifest_hashing_ab` example. No runtime inference path, MLX code,
default manifest behavior, server behavior, tokenizer behavior, or MTP behavior
changed.

### Verification

- `cargo fmt --all --check`: passed.
- `cargo check -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab`:
  passed.

### Benchmark

- Run: `xr42-1782926118256`.
- Command:
  `cargo run -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab -- --out-dir benchmarks/out/XR42-rayon-manifest-hashing-ab --trials 3 --thread-counts 1,2,4`.
- Artifacts:
  `benchmarks/out/XR42-rayon-manifest-hashing-ab/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Records: `24`; passed `24`; blockers: none.
- Deterministic seed metadata: `20260701`.
- Token lengths: `not_applicable:file hashing only; no tokenizer/model execution`.

### Artifact Results

- `gemma-4-12B-it-4bit`: 2 safetensors files, `6741039511` bytes,
  inventory SHA-256
  `4af9af81c81dcba1edb5290573e58efc28f71c887ab25a871d3917f4240459af`.
  Sequential p50/p95 was `45488.176/45572.282 ms`. Rayon 2-thread p50/p95
  was `36261.428/36460.999 ms`; p50 improved `20.284%`; inventory hash
  matched.
- `gemma-4-12B-it-qat-assistant-4bit`: 1 safetensors file, `237894178`
  bytes, inventory SHA-256
  `7a5d3a9eabd8ec983c4ef5139badf2da187a455133446be21b3c3dc0006b70bd`.
  Sequential p50/p95 was `1566.653/1587.529 ms`. Rayon 2-thread p50/p95 was
  `1566.317/1567.353 ms`; p50 improved only `0.021%`; inventory hash matched.

Do not integrate Rayon into the default manifest path from this run alone.
Follow-up should either add a guarded manifest-path option or broaden to larger
multi-shard artifacts before adoption.
