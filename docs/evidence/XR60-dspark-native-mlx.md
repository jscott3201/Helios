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

`config.json` and the released `model.safetensors` are present locally. The
checkpoint is intentionally not committed.

- `model.safetensors` size: `6860897028` bytes
- `model.safetensors` SHA-256:
  `864d974efd2e4d636b946c88769a94fc5cb32b4a8ba5dec287ba6b0e4969685e`
- draft artifact inventory SHA-256:
  `da89117833a8ee34317fcdafa0d41c1c7228d7c3d8bace2f9fde88c0bd255aa5`
- draft local artifact SHA-256:
  `79bc9a537d13978b40adef51408a1431b312b427c9208e4952dfd142360dbcea`

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
- `tools/dspark/compare_native_trace.py`
- `tools/dspark/analyze_target_distribution.py`
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
  post-layer hidden taps as cache-owned context sequences in `NativeKvState`
  and last-token metadata views in `NativeHiddenState`.
- `gemma4_kv_dspark_tap_info` reports tap ids, shapes, and resident bytes
  without exposing raw MLX pointers over the C ABI.
- Strict DSpark loads validate the released block-7 config and tensor inventory.
- On a native-graph target build, strict DSpark loads materialize matching
  DSpark safetensors into an opaque `NativeDSparkModel`.
- Draft calls validate the loaded drafter, cached target tap context, last-token
  tap metadata, and native token alignment, then route through the native
  fixed-prefix DSpark decoder path.
- Adapter-active and compressed-active-KV DSpark paths are rejected.

The reference header at `references/ffi/gemma4_mlx.h` was synced to the live
native header after adding the DSpark ABI.

## Native DSpark Loader Slice

The DSpark manifest path now accepts only the current
`deepseek-ai/dspark_gemma4_12b_block7` shape:

- `Gemma4DSparkModel`
- `model_type = gemma4_text`
- `target_model_type = gemma4_unified`
- `dtype = bfloat16`
- `block_size = 7`
- `num_hidden_layers = 5`
- `hidden_size = 3840`
- `intermediate_size = 15360`
- `attention_k_eq_v = true`
- `tie_word_embeddings = false`
- `markov_head_type = vanilla`
- `markov_rank = 256`
- `enable_confidence_head = true`
- `confidence_head_with_markov = true`
- `target_layer_ids = [5, 17, 29, 41, 46]`

The tensor inventory is strict: 74 DSpark tensors, 0 quantized groups, and no
ignored extras. Required top-level tensors include `embed_tokens.weight`,
`fc.weight`, `hidden_norm.weight`, `norm.weight`, `lm_head.weight`,
`markov_head.markov_w1.weight`, `markov_head.markov_w2.weight`,
`confidence_head.proj.weight`, and `confidence_head.proj.bias`. Each of the
five draft layers requires q/k/o attention projections, q/k norms, four
layernorm/scalar tensors, and gate/up/down MLP projections. No `v_proj` is
accepted because the released config uses `attention_k_eq_v = true`.

In smoke/no-MLX builds, the strict loader still validates the manifest and
safetensors header, then defers tensor materialization unless the target was
loaded with the native graph. This mirrors the existing MTP assistant loader
behavior and keeps tests runnable without downloading the 6.86 GB checkpoint.

## Native Tap Context Slice

DeepSpec DSpark does not consume only the final target hidden state. Its
`extract_context_feature` path concatenates selected target hidden-state
sequences for layer ids `[5, 17, 29, 41, 46]`, producing a context tensor with
`5 * 3840 = 19200` features per target position before `fc.weight`.

The native target path now preserves that prerequisite state:

- `NativeKvState` owns selected DSpark tap context arrays beside target KV.
- Prefill replaces the DSpark context with full selected tap sequences.
- Incremental decode and block decode append selected tap deltas as target KV
  advances.
- Retroactive-prefix materialization preserves the corresponding DSpark tap
  prefix for rollback/verify semantics.
- KV snapshot save/load persists restored DSpark context taps, and hidden-state
  load reconstructs last-token tap views from the restored context.
- `NativeDSparkModel::draft_block` is now the single native admission point for
  DSpark drafting. It validates tensor load state, block size, context tokens,
  cached tap ids, cached tap shapes `[1, S, 3840]`, and last-token tap shapes
  `[1, 1, 3840]` before returning the current decoder-math blocker.

This slice deliberately avoids a last-token-only DSpark implementation, because
that would not match the released DeepSpec architecture.

## Native DSpark Decoder Slice

`NativeDSparkModel::draft_block` now contains the first native MLX
fixed-prefix block-7 decoder path for the released DSpark checkpoint:

- dense BF16 `embed_tokens`, `fc`, layer projection, `lm_head`, Markov, and
  confidence-head helpers;
- selected target tap context concatenation in `[5, 17, 29, 41, 46]` order,
  `fc.weight` projection, and `hidden_norm`;
- five Gemma-style DSpark layers with full-attention q/k/o projections,
  q/k RMS norms, RoPE, `attention_k_eq_v` value handling, RMS-normed residual
  blocks, GEGLU MLP, and layer scalar;
- full block-7 masked draft input construction using the current context token
  at slot 0 and `mask_token_id = 4` for remaining slots;
- softcapped `lm_head` logits followed by sequential vanilla Markov bias
  (`markov_w1` previous-token embedding and `markov_w2` vocab projection);
- per-token top-2 greedy token/logit/margin extraction and confidence sigmoid
  output for scheduled fixed prefixes 1, 2, 4, and 7.

The C ABI path `gemma4_dspark_draft_block` already routes into this native
method and records native draft latency in `Gemma4DSparkDraftResult`. This code
is build-verified against local MLX headers and has run through the released
checkpoint in a bounded native smoke. The current runtime evidence preserves
exact target output only because `gemma4_verify_tokens` rejects every draft and
commits the target fallback token.

## Native Anchor-Context Alignment Slice

Pinned DeepSpec source shows the DSpark attention mask excludes the anchor
token from target context (`kv_idx < anchor_pos`) and supplies that anchor
through the draft/noise stream. The native DSpark decoder now mirrors that
contract for the current Helios verifier shape:

- `NativeDSparkModel::draft_block` still anchors on the current context token,
  because `gemma4_verify_tokens` expects draft token 0 to be the next target
  token for the existing cache state.
- `dspark_project_context` slices cached target taps to the prefix before that
  anchor instead of projecting the full context including the anchor.
- The prompt-length-1 case is allowed to draft with zero target-context keys and
  the block-7 noise keys only.
- DSpark q/noise RoPE positions still start at the anchor position, while target
  context keys cover prefix positions `0..anchor_position-1`.

This fixes a native/reference contract mismatch. It does not by itself make the
released checkpoint a speedup candidate; acceptance remains zero in the updated
bounded matrix below.

## Runtime Checkpoint Slice

The released checkpoint was downloaded with:

```text
hf download deepseek-ai/dspark_gemma4_12b_block7 model.safetensors --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --local-dir artifacts/drafts/dspark-gemma4-12b-block7 --max-workers 1
```

A sandboxed Metal run failed as expected because the sandbox could not see a
GPU device:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 1 --max-new-tokens 1
```

Observed error:

```text
native Gemma 4 incremental prefill failed: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.
```

The same command succeeded unsandboxed and wrote
`benchmarks/out/XR60-dspark-native-mlx/smoke/`. The one-token smoke passed
exactness for `hello_smoke` and `hello_reference_prefix`, but both records had
`accepted_draft_tokens = 0`, `acceptance_rate = 0.0`, and rollback count `1`.

The bounded fixed-prefix matrix was then run with shared target/drafter handles:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/matrix-smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1,2,4,7 --max-new-tokens 2
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/matrix-smoke/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/matrix-smoke/summary.json
benchmarks/out/XR60-dspark-native-mlx/matrix-smoke/report.md
benchmarks/out/XR60-dspark-native-mlx/matrix-smoke/blockers.md
benchmarks/out/XR60-dspark-native-mlx/matrix-smoke/decision.md
```

Result:

- decision: `keep_disabled_pending_broader_evidence`
- status: `passed` for exactness on this bounded workload
- workload: `hello_smoke`
- block sizes: `1, 2, 4, 7`
- max new tokens: `2`
- baseline/DSpark token sequence SHA-256:
  `1070d9af5afdfd5c8555f50212ea73aace42e743e4261fa5463c6eb9ada04ea0`
- accepted draft tokens: `0` for every block size
- acceptance rate: `0.0` for every block size
- rollback count: `2` for every block size
- decode throughput range: `0.019` to `0.029` tok/s
- draft time range: `22236.446` to `52406.151` ms
- verify forward time range: `45249.832` to `52471.542` ms
- peak memory: `13.565` GB
- hidden tap bytes: `38400`

This is not a speedup candidate. It is useful evidence that the native DSpark
checkpoint path can execute and that the verifier preserves exact output, but
the drafter output or decoder math still needs reference parity diagnosis before
any promotion or broader benchmark claim.

## Native Verify Trace Slice

The PyTorch/DeepSpec fixture path was rerun with the checkpoint present:

```text
python3 tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures --prompt-token-ids 9259 --allow-blocked
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/manifest.json
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/blockers.md
```

The manifest confirms the DSpark checkpoint and config are present and valid,
but the local Python environment is missing `torch`, `safetensors`,
`transformers`, and an importable `deepspec` package. The DeepSpec/PyTorch
reference fixture remains blocked on those dependencies.

To make progress while that external fixture is gated, the native benchmark
record schema now includes `verify_trace` entries with DSpark draft
tokens/logits/margins/confidence and the verifier target tokens/top-k/committed
tokens. A one-token trace smoke was run:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/trace-smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1 --max-new-tokens 1
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/trace-smoke/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/trace-smoke/summary.json
benchmarks/out/XR60-dspark-native-mlx/trace-smoke/report.md
benchmarks/out/XR60-dspark-native-mlx/trace-smoke/blockers.md
benchmarks/out/XR60-dspark-native-mlx/trace-smoke/decision.md
```

Result:

- decision: `keep_disabled_pending_broader_evidence`
- exactness: passed through verifier fallback
- DSpark draft token: `236764`
- DSpark draft logit/margin/confidence: `14.375`, `1.1875`, `0.3828125`
- target greedy token: `236772`
- committed token: `236772`
- accepted draft count: `0`
- draft in target top-k trace: `false`
- draft time: `6401.496` ms
- verify forward time: `18763.262` ms
- peak memory: `13.564` GB

This local trace narrows the zero-acceptance symptom to a first-token draft
mismatch on the smoke prompt. It does not replace the required DeepSpec/PyTorch
fixture, but it gives the next parity pass concrete token/logit values to
compare against reference DSpark output.

After aligning native target context to exclude the anchor, the bounded matrix
was rerun:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1,2,4,7 --max-new-tokens 2
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask/summary.json
benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask/report.md
benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask/blockers.md
benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask/decision.md
```

Result:

- decision: `keep_disabled_pending_broader_evidence`
- exactness: passed for `hello_smoke` and `hello_reference_prefix` at block
  sizes `1, 2, 4, 7`
- accepted draft tokens: `0` for every record
- `hello_smoke` first verify trace after the fix: DSpark draft `[9259]`,
  target greedy `[236772, 236772]`, committed `[236772]`
- `hello_reference_prefix` first verify trace after the fix: DSpark draft
  `[236766]` or `[236766, 18252]` depending on scheduled length, target greedy
  `[236761, 236779]`, committed `[236761]`
- peak memory range: `13.565` to `13.567` GB
- decode throughput range: `0.021` to `0.032` tok/s

The local target artifact for these runs is
`artifacts/models/gemma-4-12B-it-4bit`, whose config declares 4-bit affine
quantization. The DSpark checkpoint config is BF16 and identifies the target as
`gemma4_unified`. Hidden-tap parity must therefore check both native math and
whether the 4-bit target tap distribution is compatible with the released
DeepSpec drafter.

## Native Hidden Tap Snapshot Slice

The fixed-prefix benchmark can now emit native target hidden-tap snapshots for
Phase 2 parity without adding a new C ABI surface. The implementation uses the
existing native KV snapshot exporter, which already persists
`dspark_context.tap_*.hidden` arrays when XR60 taps are enabled.

Smoke command:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/tap-snapshot-smoke --native-tap-snapshot-dir benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1 --max-new-tokens 1
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json
benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/xr60-1783154137-hello_smoke.safetensors
benchmarks/out/XR60-dspark-native-mlx/tap-snapshot-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

Result:

- manifest status: `ready_for_reference_compare`
- workload: `hello_smoke`
- prompt tokens: `[9259]`
- prefill greedy token/logit: `236772`, `17.75`
- tap layer ids: `[5, 17, 29, 41, 46]`
- tap shapes: five `[1, 1, 3840]` tensors
- tap bytes: `38400`
- snapshot payload size: about `410K`
- DSpark exactness still passed only through verifier fallback; accepted draft
  tokens remained `0`

This gives the native side of hidden-tap parity a concrete, small artifact.
DeepSpec/PyTorch fixture comparison remains blocked on local Python
dependencies.

## DeepSpec Warm-Start Alignment

DeepSpec's DSpark evaluator prefills the prompt, commits the first target token,
then proposes DSpark tokens from that committed current-token anchor. The XR60
fixed-prefix harness now mirrors that flow explicitly with
`warmup_target_tokens = 1` before the DSpark draft/verify loop. This keeps
`gemma4_verify_tokens` semantics unchanged while preventing the benchmark from
drafting directly from the prompt's last token.

Matrix command:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1,2,4,7 --max-new-tokens 3
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/summary.json
benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/report.md
benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/blockers.md
benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/decision.md
```

Result:

- decision: `keep_disabled_pending_broader_evidence`
- exactness: passed for both workloads at block sizes `1, 2, 4, 7`
- warmup target tokens: `1` for every measured record
- accepted draft tokens: `0` for every record
- attempted draft tokens: `2` per record for `max_new_tokens = 3`
- `hello_smoke` first post-warmup draft: `[236745]` or `[236745, 735]`;
  target greedy trace: `[236772, 236761]`; committed fallback: `[236772]`
- `hello_reference_prefix` first post-warmup draft: `[107]` or
  `[107, 236829]`; target greedy trace: `[236779, 236772]`; committed
  fallback: `[236779]`
- peak memory range: `13.566` to `13.569` GB
- decode throughput range: `0.028` to `0.063` tok/s

The alignment removes a benchmark-semantics mismatch but does not improve
acceptance. The next required evidence is reference parity: compare the native
DSpark draft outputs against DeepSpec/PyTorch using the native tap snapshot as
the target-hidden input, then determine whether the remaining zero acceptance is
native decoder math, 4-bit-vs-BF16 target distribution, or prompt selection.

## Native-Tap DeepSpec Reference Fixture Path

The reference fixture exporter now has a concrete native-tap mode for G1/G2:
it reads `native_tap_snapshot_manifest.json`, validates the small native
safetensors payload header without requiring the Python `safetensors` package,
and, when the PyTorch stack is available, runs pinned DeepSpec
`Gemma4DSparkModel` with those native taps as `target_hidden_states`.

This avoids loading the full 12B target in PyTorch for the first parity check.
The target side is Helios-owned native MLX; DeepSpec is only the released
DSpark drafter reference over the exported target hidden taps.

Command:

```text
python3 tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --native-tap-manifest benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap --prompt-token-ids 9259 --allow-blocked
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/manifest.json
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/blockers.md
```

Current result:

- status: `passed`
- native tap manifest: `ready`
- snapshot payload: present, `419662` bytes
- snapshot SHA-256:
  `4aff723da113f6afe941a29c030603fceb60c22a66a24089050005d7be2b3bd9`
- prompt tokens: `[9259]`
- anchor token from native prefill: `236772`
- tap tensors: `dspark_context.tap_0.hidden` through
  `dspark_context.tap_4.hidden`
- tap layer ids: `[5, 17, 29, 41, 46]`
- tap tensor dtype/shape: five BF16 `[1, 1, 3840]` tensors
- reference output:
  `benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/reference_fixture.json`
- reference greedy draft tokens:
  `[236745, 735, 496, 59398, 236761, 107, 236909]`
- blockers: none for this native-tap DeepSpec fixture

The local reference stack lives in ignored artifact paths:

```text
artifacts/envs/dspark-reference/
artifacts/reference/DeepSpec/
.uv-cache/
```

Reference environment setup:

```text
UV_CACHE_DIR=.uv-cache uv venv --python /Users/justin/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 artifacts/envs/dspark-reference
git clone https://github.com/deepseek-ai/DeepSpec artifacts/reference/DeepSpec
git -C artifacts/reference/DeepSpec checkout afdfa7c9382a3341a3e6f17756dd816da79f132c
UV_CACHE_DIR=.uv-cache uv pip install --python artifacts/envs/dspark-reference/bin/python torch==2.9.1 transformers==5.10.2 safetensors==0.7.0 numpy==2.4.4 PyYAML==6.0.3 typing_extensions==4.15.0 sentencepiece==0.2.1
```

Fixture command:

```text
PYTHONPATH=artifacts/reference/DeepSpec artifacts/envs/dspark-reference/bin/python tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --native-tap-manifest benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap --prompt-token-ids 9259
```

The exporter emits compact top-k base logits, Markov logits, greedy draft
tokens, confidence logits/probabilities, native target tap values, and
`hidden.last` for the tiny smoke fixture. Full logits can be requested with
`--include-full-logits`.

## Native Trace Parity Against DeepSpec

The native DSpark warm-start trace now has a direct DeepSpec comparison for the
`hello_smoke` native-tap fixture.

Command:

```text
python3 tools/dspark/compare_native_trace.py --reference benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/reference_fixture.json --records benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-trace
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-trace/parity_report.json
benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-trace/blockers.md
```

Result:

- status: `passed`
- compared records: `4` `hello_smoke` records from the warm-start matrix
- skipped records: `4` `hello_reference_prefix` records, because no matching
  DeepSpec native-tap fixture has been exported for that workload yet
- native draft token prefixes matched DeepSpec exactly:
  `[236745]` for scheduled length `1` and `[236745, 735]` for scheduled
  length `2`
- selected Markov logit max absolute error: `0.125`
- confidence max absolute error: `0.0002943401370620602`
- Markov margin max absolute error: `0.125`
- verifier target prefix still differed from the DeepSpec/native DSpark draft:
  `[236772]` or `[236772, 236761]` for the compared `hello_smoke` records

Interpretation: for this smoke context, zero acceptance is not caused by a
native DSpark decoder math mismatch against DeepSpec. The released DeepSpec
drafter and Helios native 4-bit target distribution disagree on the next target
tokens for the measured prompt/tap context.

## Broader Native-Tap Parity Corpus

The native tap snapshot path was rerun for both bounded warm-start workloads so
the `hello_reference_prefix` native trace could be compared against DeepSpec as
well.

Snapshot command:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/tap-snapshot-warm-corpus --native-tap-snapshot-dir benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-warm-corpus --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1 --max-new-tokens 1
```

Fixture command:

```text
PYTHONPATH=artifacts/reference/DeepSpec artifacts/envs/dspark-reference/bin/python tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --native-tap-manifest benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-warm-corpus/native_tap_snapshot_manifest.json --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-warm-corpus --prompt-token-ids 9259
```

Parity command:

```text
python3 tools/dspark/compare_native_trace.py --reference benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-warm-corpus/reference_fixture.json --records benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-warm-corpus --margin-tolerance 0.5
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-warm-corpus/native_tap_snapshot_manifest.json
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-warm-corpus/reference_fixture.json
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-warm-corpus/manifest.json
benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-warm-corpus/parity_report.json
benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-warm-corpus/blockers.md
```

Result:

- native snapshot manifest status: `ready_for_reference_compare`
- snapshots: `2`
  - `hello_smoke`, prompt `[9259]`, anchor `236772`, context tokens `1`
  - `hello_reference_prefix`, prompt `[9259, 236772, 236772]`, anchor
    `236761`, context tokens `3`
- DeepSpec fixture status: `passed`
- DeepSpec fixture count: `2`
- `hello_smoke` DeepSpec draft prefix:
  `[236745, 735, 496, 59398]`
- `hello_reference_prefix` DeepSpec draft prefix:
  `[107, 236829, 139, 1018]`
- native trace parity status: `passed`
- compared records: `8`
- skipped records: `0`
- native draft token prefixes matched DeepSpec exactly for both workloads
- selected Markov logit max absolute error: `0.1875`
- confidence max absolute error: `0.005936026573181152`
- Markov margin max absolute error: `0.28125` with margin tolerance `0.5`
- verifier target prefixes still differed:
  - `hello_smoke`: `[236772]` or `[236772, 236761]`
  - `hello_reference_prefix`: `[236779]` or `[236779, 236772]`

Interpretation: native DSpark decoder parity now holds for the first warm-start
trace of both bounded workloads. The zero-acceptance symptom remains because
the released DSpark drafter predicts a different continuation than the local
Helios 4-bit target verifies.

## Target Distribution Diagnosis

The warm-start matrix was rerun with target top-k tracing enabled to quantify
how far DSpark drafts are from the verifier target distribution.

Benchmark command:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1,2,4,7 --max-new-tokens 3
```

Analysis command:

```text
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/summary.json
benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/report.md
benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis/target_distribution_report.json
benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis/report.md
benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis/blockers.md
```

Result:

- benchmark status: `passed`
- benchmark decision: `keep_disabled_pending_broader_evidence`
- diagnosis status: `passed`
- diagnosis:
  `released_dspark_drafts_outside_target_top_k_on_measured_corpus`
- measured records: `8`
- target top-k width: `5`
- observations: `22`
- accepted observations: `0`
- draft-in-target-top-k count: `0`
- draft-in-target-top-k rate: `0.0`
- outside-top-k lower-bound gap min/median/max:
  `2.625` / `2.6875` / `4.3125`
- draft confidence min/median/max:
  `0.14453125` / `0.21875` / `0.64453125`
- `hello_smoke` unique draft tokens: `[735, 236745, 236766]`
- `hello_smoke` unique target tokens: `[236761, 236772]`
- `hello_reference_prefix` unique draft tokens: `[107, 602, 236829]`
- `hello_reference_prefix` unique target tokens: `[236772, 236779]`

The lower-bound gap uses the target top-1 logit minus the lowest observed
target top-5 logit when the DSpark draft token is not in target top-5. This is a
conservative lower bound: the draft token's actual target-rank logit is lower
than the top-5 floor.

Interpretation: for the measured bounded corpus, zero acceptance is explained
by DSpark drafts landing outside the local 4-bit target's top-5 distribution,
not by native DSpark decoder mismatch against DeepSpec. This supports keeping
DSpark default-off and shifts the next experiment toward broader prompt/context
selection or BF16-vs-4-bit target distribution comparison.

## Real-Context Target Distribution Diagnosis

The target-distribution diagnosis now has a real-context token workload path.
`tools/dspark/export_token_workloads.py` validates prompt SHA-256 and local
Gemma tokenizer counts against `benchmarks/workloads/real-contexts/workloads.jsonl`,
then writes JSONL token records consumed directly by the Rust fixed-prefix
harness through `--token-workloads`.

Token export command:

```text
artifacts/envs/dspark-reference/bin/python tools/dspark/export_token_workloads.py --workloads chat_short_1k_001,mtp_candidate_1k_001 --out benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.jsonl
```

Benchmark command:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-topk --token-workloads benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.jsonl --workloads chat_short_1k_001,mtp_candidate_1k_001 --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 1,2 --max-new-tokens 3
```

Analysis command:

```text
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/real-context-topk/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-target-distribution
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.jsonl
benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.manifest.json
benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.blockers.md
benchmarks/out/XR60-dspark-native-mlx/real-context-topk/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/real-context-topk/summary.json
benchmarks/out/XR60-dspark-native-mlx/real-context-topk/report.md
benchmarks/out/XR60-dspark-native-mlx/real-context-target-distribution/target_distribution_report.json
benchmarks/out/XR60-dspark-native-mlx/real-context-target-distribution/report.md
benchmarks/out/XR60-dspark-native-mlx/real-context-target-distribution/blockers.md
```

Result:

- token export status: `passed`
- token export workloads: `chat_short_1k_001`, `mtp_candidate_1k_001`
- benchmark status: `passed`
- benchmark decision: `keep_disabled_pending_broader_evidence`
- measured records: `4`
- exact records: `4`
- scheduled lengths: `[1, 2]`
- diagnosis status: `passed`
- diagnosis: `some_drafts_align_with_target_distribution`
- observations: `9`
- accepted observations: `4`
- accepted observation rate: `0.4444444444444444`
- draft-in-target-top-k count: `8`
- draft-in-target-top-k rate: `0.8888888888888888`
- outside-top-k lower-bound gap min/median/max:
  `7.375` / `7.375` / `7.375`
- `chat_short_1k_001`: `5` observations, `0` accepted, `4/5` draft
  positions in target top-5
- `mtp_candidate_1k_001`: `4` observations, `4` accepted, `4/4` draft
  positions in target top-5

Interpretation: the real-context pair does not support the earlier blanket
"outside target top-k" finding from the bounded toy prefixes. The released
DSpark checkpoint aligns with the local 4-bit target on the MTP-shaped 1K
prompt, while the chat-shaped 1K prompt still has zero accepted drafts. DSpark
remains default-off because this is a narrow two-workload slice, latency is
still severe, and code/4K/later-decode coverage is not yet measured.

### 4K code/MTP follow-up

The token-workload diagnosis was extended to one code-review workload and one
4K MTP-shaped workload with `max_new_tokens=5`, fixed block sizes `1,2,4`, and
target top-k tracing enabled. This gives later decode positions and a
multi-position block-4 verify trace.

Token export command:

```text
artifacts/envs/dspark-reference/bin/python tools/dspark/export_token_workloads.py --workloads code_review_rust_4k_001,mtp_candidate_4k_001 --out benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.jsonl
```

Benchmark command:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk --token-workloads benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.jsonl --workloads code_review_rust_4k_001,mtp_candidate_4k_001 --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 1,2,4 --max-new-tokens 5
```

Analysis command:

```text
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-target-distribution
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.jsonl
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.manifest.json
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.blockers.md
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk/summary.json
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk/report.md
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-target-distribution/target_distribution_report.json
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-target-distribution/report.md
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-target-distribution/blockers.md
```

Result:

- token export status: `passed`
- token export workloads: `code_review_rust_4k_001`, `mtp_candidate_4k_001`
- benchmark status: `passed`
- benchmark decision: `keep_disabled_pending_broader_evidence`
- measured records: `6`
- exact records: `6`
- scheduled lengths: `[1, 2, 4]`
- diagnosis status: `passed`
- diagnosis: `some_drafts_align_with_target_distribution`
- observations: `26`
- accepted observations: `18`
- accepted observation rate: `0.6923076923076923`
- draft-in-target-top-k count: `18`
- draft-in-target-top-k rate: `0.6923076923076923`
- outside-top-k lower-bound gap min/median/max:
  `0.0` / `5.25` / `5.875`
- `code_review_rust_4k_001`: `14` observations, `6` accepted, `6/14`
  draft positions in target top-5
- `mtp_candidate_4k_001`: `12` observations, `12` accepted, `12/12`
  draft positions in target top-5
- decode throughput stayed far below useful speed:
  - `code_review_rust_4k_001`: `0.0278` to `0.0628` tok/s
  - `mtp_candidate_4k_001`: `0.0253` to `0.0852` tok/s
- peak memory reached `16.26557159423828` GB on measured 4K records
- active KV bytes were `402735104`; hidden tap bytes were `38400`

The analyzer also now tolerates missing target top-k rows on longer block-size
traces. One block-size 4 code trace had four draft tokens but only three target
top-k rows; missing rows are treated as unknown rank/gap instead of crashing.

Interpretation: the released DSpark checkpoint has strong alignment on
MTP-shaped 1K and 4K prompts, partial alignment on a 4K code-review prompt, and
poor alignment on the chat-shaped prompt. This supports further scheduling and
calibration investigation only as a default-off experimental path. It does not
support promotion because throughput is orders of magnitude below the native
target baseline and peak memory is already at the tiny16 edge.

### Block-size 7 scheduler/value slice

The fixed-prefix test was extended to block size `7`, which is the released
DSpark checkpoint's configured maximum block size. The run uses
`max_new_tokens=8` so the scheduler can exercise a full seven-token proposal
after the warm-start target token.

Benchmark command:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7 --token-workloads benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.jsonl --workloads code_review_rust_4k_001,mtp_candidate_4k_001 --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 7 --max-new-tokens 8
```

Analysis command:

```text
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7-target-distribution
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7/summary.json
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7/report.md
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7/decision.md
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7-target-distribution/target_distribution_report.json
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7-target-distribution/report.md
benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7-target-distribution/blockers.md
```

Result:

- benchmark status: `passed`
- benchmark decision: `keep_disabled_pending_broader_evidence`
- measured records: `2`
- exact records: `2`
- scheduled lengths: `[7]`
- diagnosis status: `passed`
- diagnosis: `some_drafts_align_with_target_distribution`
- observations: `23`
- accepted observations: `10`
- accepted observation rate: `0.43478260869565216`
- draft-in-target-top-k count: `10`
- draft-in-target-top-k rate: `0.43478260869565216`
- `code_review_rust_4k_001`: `4/16` observed draft positions accepted and
  in target top-5; throughput `0.05228086951637692` tok/s; `4` verify passes;
  `3` rollbacks
- `mtp_candidate_4k_001`: `6/7` observed draft positions accepted and in
  target top-5; throughput `0.15441365086323453` tok/s; `1` verify pass;
  `1` rollback
- peak memory reached `16.26557159423828` GB

Interpretation: block size `7` improves the high-acceptance MTP-shaped workload
relative to block size `4` (`0.1544` tok/s vs `0.0852` tok/s), but it remains
orders of magnitude slower than the native target baseline range. The code
workload degrades in acceptance at block size `7`. These results do not justify
confidence scheduling or custom kernel work in the current implementation.

## Final Decision Rollup

A goal-level finalizer now writes the required root XR60 artifact set from the
measured source artifacts and uses the goal's decision vocabulary.

Finalizer command:

```text
python3 tools/dspark/summarize_xr60_decision.py --decision reject_for_now
```

Artifacts:

```text
benchmarks/out/XR60-dspark-native-mlx/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/summary.json
benchmarks/out/XR60-dspark-native-mlx/report.md
benchmarks/out/XR60-dspark-native-mlx/blockers.md
benchmarks/out/XR60-dspark-native-mlx/decision.md
```

Result:

- final decision: `reject_for_now`
- final status: `passed`
- measured records in final rollup: `20`
- exact records in final rollup: `20`
- best measured DSpark decode throughput: `0.15441365086323453` tok/s
- peak measured memory: `16.26557159423828` GB
- final blockers: none

Decision rationale:

- Measured fixed-prefix DSpark output is exact on the rollup corpus, but no
  measured scheduler is remotely speed-profitable.
- Best measured DSpark decode throughput is far below the `12-16` tok/s native
  baseline range cited by the XR60 goal.
- Peak measured memory is at or beyond the tiny16 budget edge.
- Target-distribution evidence is domain-shaped: MTP-shaped prompts align well,
  code is partial, toy/chat prompts remain poor.
- Confidence and custom-kernel work are not justified until native DSpark
  draft/verify overhead is reduced.

## Verification

Commands run:

```text
cargo fmt --all
cargo test -p gemma4d-ffi --lib
cargo fmt --all --check
git diff --check
cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib --no-run
cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 1,2,4,7 --max-new-tokens 32
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/matrix-smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1,2,4,7 --max-new-tokens 2
python3 tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures --prompt-token-ids 9259 --allow-blocked
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/trace-smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1 --max-new-tokens 1
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/trace-anchor-mask --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1 --max-new-tokens 1
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/trace-anchor-mask-2tok --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1 --max-new-tokens 2
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1,2,4,7 --max-new-tokens 2
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/tap-snapshot-smoke --native-tap-snapshot-dir benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1 --max-new-tokens 1
GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/warm-anchor-smoke --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke --block-sizes 1 --max-new-tokens 2
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1,2,4,7 --max-new-tokens 3
python3 -m py_compile tools/dspark/dspark_common.py tools/dspark/export_reference_fixture.py tools/dspark/convert_to_mlx.py tools/dspark/compare_mlx_parity.py
python3 tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --native-tap-manifest benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap --prompt-token-ids 9259 --allow-blocked
UV_CACHE_DIR=.uv-cache uv venv --python /Users/justin/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12 artifacts/envs/dspark-reference
git clone https://github.com/deepseek-ai/DeepSpec artifacts/reference/DeepSpec
git -C artifacts/reference/DeepSpec checkout afdfa7c9382a3341a3e6f17756dd816da79f132c
UV_CACHE_DIR=.uv-cache uv pip install --python artifacts/envs/dspark-reference/bin/python torch==2.9.1 transformers==5.10.2 safetensors==0.7.0 numpy==2.4.4 PyYAML==6.0.3 typing_extensions==4.15.0 sentencepiece==0.2.1
PYTHONPATH=artifacts/reference/DeepSpec artifacts/envs/dspark-reference/bin/python tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --native-tap-manifest benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap --prompt-token-ids 9259
python3 -m py_compile tools/dspark/dspark_common.py tools/dspark/export_reference_fixture.py tools/dspark/convert_to_mlx.py tools/dspark/compare_mlx_parity.py tools/dspark/compare_native_trace.py
python3 tools/dspark/compare_native_trace.py --reference benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/reference_fixture.json --records benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-trace
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/tap-snapshot-warm-corpus --native-tap-snapshot-dir benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-warm-corpus --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1 --max-new-tokens 1
PYTHONPATH=artifacts/reference/DeepSpec artifacts/envs/dspark-reference/bin/python tools/dspark/export_reference_fixture.py --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --native-tap-manifest benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-warm-corpus/native_tap_snapshot_manifest.json --out-dir benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-warm-corpus --prompt-token-ids 9259
python3 tools/dspark/compare_native_trace.py --reference benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-warm-corpus/reference_fixture.json --records benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-warm-corpus --margin-tolerance 0.5
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --workloads hello_smoke,hello_reference_prefix --block-sizes 1,2,4,7 --max-new-tokens 3
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis
python3 -m py_compile tools/dspark/export_token_workloads.py
cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run
artifacts/envs/dspark-reference/bin/python tools/dspark/export_token_workloads.py --workloads chat_short_1k_001,mtp_candidate_1k_001 --out benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.jsonl
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-topk --token-workloads benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.jsonl --workloads chat_short_1k_001,mtp_candidate_1k_001 --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 1,2 --max-new-tokens 3
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/real-context-topk/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-target-distribution
artifacts/envs/dspark-reference/bin/python tools/dspark/export_token_workloads.py --workloads code_review_rust_4k_001,mtp_candidate_4k_001 --out benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.jsonl
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk --token-workloads benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.jsonl --workloads code_review_rust_4k_001,mtp_candidate_4k_001 --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 1,2,4 --max-new-tokens 5
python3 -m py_compile tools/dspark/analyze_target_distribution.py
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-target-distribution
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_REAL_MARGINS=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7 --token-workloads benchmarks/out/XR60-dspark-native-mlx/real-context-4k-token-workloads.jsonl --workloads code_review_rust_4k_001,mtp_candidate_4k_001 --model-path artifacts/models/gemma-4-12B-it-4bit --draft-path artifacts/drafts/dspark-gemma4-12b-block7 --block-sizes 7 --max-new-tokens 8
python3 tools/dspark/analyze_target_distribution.py --records benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7/records.jsonl --out-dir benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7-target-distribution
python3 -m py_compile tools/dspark/summarize_xr60_decision.py
python3 tools/dspark/summarize_xr60_decision.py --decision reject_for_now
```

Observed result:

- `cargo test -p gemma4d-ffi --lib`: 21 passed, 1 ignored after adding
  DSpark strict loader, tap-context admission changes, and native decoder math.
- `cargo test -p gemma4d-bench --example dspark_fixed_block_matrix --no-run`:
  compiled successfully.
- `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-ffi --lib --no-run`:
  compiled successfully, covering the MLX-only native DSpark helper code.
- The fixed-prefix harness wrote
  `benchmarks/out/XR60-dspark-native-mlx/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`
  with decision `blocked`.
- The bounded fixed-prefix harness wrote
  `benchmarks/out/XR60-dspark-native-mlx/matrix-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`
  with decision `keep_disabled_pending_broader_evidence`. Exactness passed on
  `hello_smoke` for fixed-prefix block sizes `1,2,4,7`, but acceptance was `0.0`
  and throughput was `0.019` to `0.029` tok/s.
- The reference fixture command wrote blocked artifacts under
  `benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/` with valid
  checkpoint/config identity and blockers for missing `torch`, `safetensors`,
  `transformers`, and `deepspec`.
- The trace smoke wrote
  `benchmarks/out/XR60-dspark-native-mlx/trace-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
  Its first DSpark token was `236764`, target greedy was `236772`, accepted
  draft count was `0`, and the draft token was not in the target top-k trace.
- The anchor-context rerun wrote
  `benchmarks/out/XR60-dspark-native-mlx/matrix-anchor-mask/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
  Exactness passed for both bounded workloads and all fixed block sizes, but
  every record still had `accepted_draft_tokens = 0`.
- The native hidden-tap snapshot smoke wrote
  `benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json`
  and a small safetensors payload containing the selected native DSpark context
  taps for `hello_smoke`.
- The DeepSpec warm-start matrix wrote
  `benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
  Exactness passed for both bounded workloads and all fixed block sizes with
  `warmup_target_tokens = 1`, but every record still had
  `accepted_draft_tokens = 0`.
- The native-tap reference fixture command wrote blocked artifacts under
  `benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/`
  before the reference environment was available.
- After installing the pinned local reference environment, the same native-tap
  reference fixture passed and wrote `reference_fixture.json` for `hello_smoke`.
- The native trace parity report wrote
  `benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-trace/{parity_report.json,blockers.md}`.
  It matched native DSpark draft tokens, selected Markov logits, confidence, and
  margins to DeepSpec for the compared `hello_smoke` records. The verifier
  target prefix still differed, explaining zero acceptance for that context.
- The broader native-tap parity corpus wrote
  `benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-warm-corpus/`,
  `benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-warm-corpus/`,
  and
  `benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-warm-corpus/`.
  It matched native DSpark first-trace draft tokens exactly to DeepSpec across
  all 8 warm-start matrix records for `hello_smoke` and
  `hello_reference_prefix`; selected Markov logits, confidence, and margins
  were within configured tolerances, but verifier target prefixes still differed.
- The target-distribution diagnosis wrote
  `benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/` and
  `benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis/`.
  With target top-5 tracing enabled, all 22 observed DSpark draft positions were
  outside the verifier target top-5 and accepted draft count stayed at zero.
- The real-context token workload export wrote
  `benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.jsonl`
  plus a passed manifest/blockers pair for `chat_short_1k_001` and
  `mtp_candidate_1k_001`.
- The real-context target-distribution diagnosis wrote
  `benchmarks/out/XR60-dspark-native-mlx/real-context-topk/` and
  `benchmarks/out/XR60-dspark-native-mlx/real-context-target-distribution/`.
  All 4 measured records were exact. `mtp_candidate_1k_001` accepted all 4
  observed draft positions, while `chat_short_1k_001` accepted 0/5 observed
  positions.
- The 4K real-context target-distribution diagnosis wrote
  `benchmarks/out/XR60-dspark-native-mlx/real-context-4k-topk/` and
  `benchmarks/out/XR60-dspark-native-mlx/real-context-4k-target-distribution/`.
  All 6 measured records were exact. `mtp_candidate_4k_001` accepted all 12
  observed draft positions, while `code_review_rust_4k_001` accepted 6/14
  observed positions. The analyzer records missing target top-k rows as unknown
  instead of failing on long block-size traces.
- The block-size 7 scheduler/value slice wrote
  `benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7/` and
  `benchmarks/out/XR60-dspark-native-mlx/real-context-4k-block7-target-distribution/`.
  Both measured records were exact. `mtp_candidate_4k_001` accepted 6/7 draft
  positions and reached `0.1544` tok/s, while `code_review_rust_4k_001`
  accepted 4/16 draft positions and reached `0.0523` tok/s.
- The final decision rollup wrote
  `benchmarks/out/XR60-dspark-native-mlx/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
  It selected `reject_for_now`: 20 measured records, 20 exact records, best
  DSpark throughput `0.1544` tok/s, peak memory `16.27` GB, no final blockers.
- The ignored-by-default full-model FFI test now enables XR60 DSpark taps before
  native prefill and asserts tap ids `[5, 17, 29, 41, 46]`, shapes
  `[1, 1, 3840]`, and nonzero tap bytes when `GEMMA4D_FULL_MODEL_TESTS` and
  `GEMMA4D_USE_NATIVE_GRAPH` are set.

## Current Blockers / Gated Work

- Full target-hidden parity against a PyTorch BF16 target is not measured; the
  current fixture uses Helios native 4-bit target taps as DeepSpec input.
- Native DSpark decoder math is parity-verified only for the first warm-start
  traces of `hello_smoke` and `hello_reference_prefix`, not for a broad
  real-context fixture corpus or later decode positions.
- The available native benchmark target is a 4-bit affine MLX artifact, while
  the released DSpark checkpoint is BF16 and target-compatible only after hidden
  tap parity is proven.
- Final XR60 decision is `reject_for_now` for this implementation: exactness is
  preserved on measured records, but speed is non-viable and memory is at the
  tiny16 edge.
- The toy-prefix corpus still has zero draft acceptance, while real-context
  acceptance is domain-shaped: strong on MTP-shaped prompts, partial on code,
  and poor on chat.
- 8K/16K memory and sustained decode evidence are not pursued for this
  implementation because 4K peak memory already reaches the tiny16 edge and
  4K speed is far below the native baseline.

## Next Slice

Keep DSpark default-off and do not promote confidence scheduling on the current
path. Revisit only if native DSpark draft/verify overhead can be reduced through
graph-level or kernel work, or if a BF16 target comparison materially changes
the target-distribution conclusion.
