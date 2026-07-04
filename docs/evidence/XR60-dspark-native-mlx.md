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
- The ignored-by-default full-model FFI test now enables XR60 DSpark taps before
  native prefill and asserts tap ids `[5, 17, 29, 41, 46]`, shapes
  `[1, 1, 3840]`, and nonzero tap bytes when `GEMMA4D_FULL_MODEL_TESTS` and
  `GEMMA4D_USE_NATIVE_GRAPH` are set.

## Current Blockers / Gated Work

- DeepSpec/PyTorch fixture code is not yet integrated.
- Hidden-tap parity against a revision-pinned DeepSpec fixture is not measured.
- Native DSpark decoder math is not parity-verified against the released
  checkpoint.
- The first runtime evidence has zero draft acceptance and severe latency, so
  DSpark must remain default-off.
- Broader real-context workload evidence is still missing.

## Next Slice

Generate a revision-pinned DeepSpec/PyTorch fixture for the released checkpoint,
compare native hidden taps and draft logits/tokens against it, then diagnose
whether zero acceptance comes from decoder math, checkpoint/target mismatch,
prompt selection, or verifier overhead.
