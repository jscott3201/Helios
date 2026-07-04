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
  --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures \
  --allow-blocked
```

This writes a manifest and blockers when the DeepSpec/PyTorch reference stack or
`model.safetensors` is missing. It does not claim fixture parity until the
reference stack is available.

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

