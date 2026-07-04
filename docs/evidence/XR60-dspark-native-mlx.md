# XR60 DSpark Native MLX Evidence

- Goal: `codex/goals/XR60-dspark-native-mlx.goal.md`
- Branch: `xr60-dspark-native-mlx`
- Upstream DSpark artifact: `deepseek-ai/dspark_gemma4_12b_block7`
- Pinned revision: `2fa72e765eec2965fc4d86a8663ce6769eba6218`
- DeepSpec source: `deepseek-ai/DeepSpec`
- DeepSpec commit: `afdfa7c9382a3341a3e6f17756dd816da79f132c`
- Target model: `google/gemma-4-12B-it`
- Target revision: `5926caa4ec0cac5cbfadaf4077420520de1d5205`

## Initial State

Local target artifacts exist:

- `artifacts/models/gemma-4-12B-it-4bit`
- `artifacts/models/gemma-4-12B-it-qat-assistant-4bit`
- `artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32`

The DSpark draft directory was created locally at:

```text
artifacts/drafts/dspark-gemma4-12b-block7/
```

Only `config.json` has been downloaded in this slice. The 6.86 GB
`model.safetensors` file is intentionally not committed.

## Upstream Config

The downloaded config validates the XR60 goal assumptions:

- architecture: `Gemma4DSparkModel`
- block size: `7`
- draft layers: `5`
- target layer taps: `[5, 17, 29, 41, 46]`
- Markov rank: `256`
- anchors: `512`
- mask token id: `4`
- dtype: `bfloat16`
- confidence head enabled with Markov input

DeepSpec semantics from the pinned source:

- `target_layer_ids` are 0-based decoder layer ids.
- Transformers fixture extraction uses `hidden_states[layer_id + 1]`, because
  hidden state index 0 is the embedding output.
- The layer taps are raw post-layer outputs, not final-normalized hidden states.
- The selected taps concatenate in `[5, 17, 29, 41, 46]` order, giving
  `5 * 3840 = 19200` input features for `fc.weight`.
- The released checkpoint has no `v_proj` tensors because
  `attention_k_eq_v = true`.
- The vanilla Markov head is sequential and adds a rank-256 previous-token bias
  to base draft logits before greedy selection.

## Added Tooling

- `tools/dspark/export_reference_fixture.py`
- `tools/dspark/convert_to_mlx.py`
- `tools/dspark/compare_mlx_parity.py`
- `tools/dspark/README.md`
- `crates/gemma4d-bench/examples/dspark_fixed_block_matrix.rs`

The scripts and benchmark scaffold are fail-closed: they write manifests and
blocker files when weights, DeepSpec/PyTorch, MLX, or native DSpark integration
are unavailable.

## Native ABI Scaffold

Added a default-off DSpark ABI and safe Rust wrapper slice:

- `Gemma4DSparkDrafter` opaque native handle.
- `Gemma4DSparkTapConfig` and `gemma4_target_set_dspark_taps`.
- `Gemma4DSparkTapInfo` and `gemma4_kv_dspark_tap_info`.
- `Gemma4DSparkDraftResult` and `gemma4_dspark_draft_block`.
- Rust `DSparkTapConfig`, `DSparkTapInfo`, `DSparkDrafter`, and
  `DSparkDraftBlock` wrappers in `crates/gemma4d-ffi`.

Current behavior is intentionally fail-closed:

- Smoke DSpark drafter handles can be created and freed for lifecycle coverage.
- Enabling taps only accepts the released XR60 layer set
  `[5, 17, 29, 41, 46]`.
- When enabled on the native graph, target prefill/decode captures selected
  post-layer hidden taps as last-token views and keeps them in
  `NativeHiddenState`.
- `gemma4_kv_dspark_tap_info` reports tap ids, shapes, and resident bytes
  without exposing raw MLX pointers over the C ABI.
- Strict DSpark loads return unsupported until the native tensor loader exists.
- Draft calls return unsupported until DSpark tensor loading and draft execution
  are implemented.
- Adapter-active and compressed-active-KV DSpark paths are rejected.

The reference header at `references/ffi/gemma4_mlx.h` was synced to the live
native header after adding the DSpark ABI.

## Verification

Commands run:

```text
cargo fmt --all
cargo test -p gemma4d-ffi --lib
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run
cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 1,2,4,7 --max-new-tokens 32
```

Observed result:

- `cargo test -p gemma4d-ffi --lib`: 19 passed, 1 ignored.
- `cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run`:
  compiled successfully.
- The fixed-prefix harness wrote
  `benchmarks/out/XR60-dspark-native-mlx/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`
  with decision `blocked`.
- The ignored-by-default full-model FFI test now enables XR60 DSpark taps before
  native prefill and asserts tap ids `[5, 17, 29, 41, 46]`, shapes
  `[1, 1, 3840]`, and nonzero tap bytes when `GEMMA4D_FULL_MODEL_TESTS` and
  `GEMMA4D_USE_NATIVE_GRAPH` are set.

## Current Blockers

- DSpark weights are not present locally:
  `artifacts/drafts/dspark-gemma4-12b-block7/model.safetensors`.
- DeepSpec/PyTorch fixture code is not yet integrated.
- Native DSpark tensor loading and draft execution are not yet implemented.

## Next Slice

Load and validate the released DSpark tensors behind the new ABI, then execute
fixed-prefix DSpark drafts against the captured native tap arrays and existing
`gemma4_verify_tokens` verifier semantics.
