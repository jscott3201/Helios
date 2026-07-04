# XR60 DSpark Tooling

This directory contains fail-closed helper scripts for the XR60 DSpark native
MLX path. They intentionally separate reproducible manifests from heavyweight
model artifacts.

The upstream DSpark Gemma draft artifact is:

```text
deepseek-ai/dspark_gemma4_12b_block7
```

Use revision-pinned local artifacts under:

```text
artifacts/drafts/dspark-gemma4-12b-block7/
```

Do not commit downloaded model weights. The small `config.json` can be used for
local validation, but `artifacts/` is ignored by repository policy.

## Reference Fixture

```bash
python3 tools/dspark/export_reference_fixture.py \
  --draft-path artifacts/drafts/dspark-gemma4-12b-block7 \
  --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 \
  --native-tap-manifest benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures \
  --allow-blocked
```

The current reference path uses native tap snapshots as the
`target_hidden_states` input to pinned DeepSpec/PyTorch `Gemma4DSparkModel`.
This avoids loading the full 12B target in PyTorch for the first parity check:
native Helios emits `dspark_context.tap_*.hidden`, and DeepSpec runs only the
released DSpark drafter over those taps.

When `torch`, `safetensors`, `transformers`, or an importable pinned DeepSpec
checkout are missing, the script still validates the local DSpark config,
checkpoint hash, native tap manifest, and safetensors header, then writes
`manifest.json` plus `blockers.md`. It does not claim fixture parity until the
reference stack is available and `reference_fixture.json` is emitted.

## Native Trace Parity

After the native-tap fixture is emitted, compare it against Helios native trace
records:

```bash
python3 tools/dspark/compare_native_trace.py \
  --reference benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/reference_fixture.json \
  --records benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/records.jsonl \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-trace
```

The comparison matches only records with a corresponding fixture workload. It
checks DeepSpec greedy draft token prefixes, selected Markov logits, confidence,
and top-k margin against the first native verify trace for that context. The
default margin tolerance is wider than the selected-logit tolerance because
margin is a derived top1-minus-top2 diagnostic. Target token mismatch is
reported separately so zero acceptance can be distinguished from native DSpark
decoder mismatch.

## Target Distribution Diagnosis

Run the fixed-prefix benchmark with target top-k tracing enabled, then summarize
whether DSpark draft tokens appear in the verifier target distribution:

```bash
GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 \
cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk \
  --model-path artifacts/models/gemma-4-12B-it-4bit \
  --draft-path artifacts/drafts/dspark-gemma4-12b-block7 \
  --workloads hello_smoke,hello_reference_prefix \
  --block-sizes 1,2,4,7 \
  --max-new-tokens 3

python3 tools/dspark/analyze_target_distribution.py \
  --records benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/records.jsonl \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis
```

This writes `target_distribution_report.json`, `report.md`, and `blockers.md`.
When draft tokens are outside target top-k, the report records a conservative
lower bound on the target top-1-to-draft logit gap using the lowest observed
top-k logit.

## MLX Conversion Manifest

```bash
python3 tools/dspark/convert_to_mlx.py \
  --draft-path artifacts/drafts/dspark-gemma4-12b-block7 \
  --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity \
  --allow-blocked
```

This records the exact conversion prerequisites and blocks if the local weights
or MLX Python packages are unavailable.

## Parity Compare

```bash
python3 tools/dspark/compare_mlx_parity.py \
  --reference benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/reference_fixture.json \
  --mlx benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/mlx_fixture.json \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity
```

The comparator checks top-1 draft token exactness first, then configured numeric
tolerances for logits and confidence values.

## Native Hidden Tap Snapshot

The fixed-prefix benchmark can export native target hidden-tap snapshots for
Phase 2 hidden-tap parity:

```bash
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 \
cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/tap-snapshot-smoke \
  --native-tap-snapshot-dir benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke \
  --workloads hello_smoke \
  --block-sizes 1 \
  --max-new-tokens 1
```

This writes `native_tap_snapshot_manifest.json` plus one small safetensors
payload per workload. The payload is generated by the existing native KV
snapshot path and includes `dspark_context.tap_*.hidden` arrays for the selected
target layers. These artifacts are native-side parity inputs; they do not prove
DeepSpec/PyTorch equality until compared with a reference fixture.
