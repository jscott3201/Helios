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

- status: `blocked`
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
- expected reference output:
  `benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/reference_fixture.json`
- blockers: missing `torch`, `safetensors`, `transformers`, and importable
  pinned DeepSpec package in the local Python environment

When unblocked, the exporter will emit compact top-k base logits, Markov logits,
greedy draft tokens, confidence logits/probabilities, native target tap values,
and `hidden.last` for the tiny smoke fixture. Full logits can be requested with
`--include-full-logits`.

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
  `benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/`.
  It validated the native tap safetensors header and checkpoint identity, but
  could not run DeepSpec/PyTorch because the local Python environment lacks
  `torch`, `safetensors`, `transformers`, and importable `deepspec`.
- The ignored-by-default full-model FFI test now enables XR60 DSpark taps before
  native prefill and asserts tap ids `[5, 17, 29, 41, 46]`, shapes
  `[1, 1, 3840]`, and nonzero tap bytes when `GEMMA4D_FULL_MODEL_TESTS` and
  `GEMMA4D_USE_NATIVE_GRAPH` are set.

## Current Blockers / Gated Work

- DeepSpec/PyTorch native-tap fixture code is integrated, but fixture execution
  is blocked by missing local Python packages/importable pinned DeepSpec.
- Hidden-tap parity against a revision-pinned DeepSpec fixture is not measured.
- Native DSpark decoder math is not parity-verified against the released
  checkpoint.
- The available native benchmark target is a 4-bit affine MLX artifact, while
  the released DSpark checkpoint is BF16 and target-compatible only after hidden
  tap parity is proven.
- The first runtime evidence has zero draft acceptance and severe latency, so
  DSpark must remain default-off.
- Broader real-context workload evidence is still missing.

## Next Slice

Install/provide the pinned DeepSpec/PyTorch reference environment, rerun the
native-tap reference fixture command to emit `reference_fixture.json`, then
compare native DSpark draft logits/tokens against it to diagnose whether zero
acceptance comes from decoder math, checkpoint/target mismatch, prompt
selection, or verifier overhead.
