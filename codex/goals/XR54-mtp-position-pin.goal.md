# XR54 - MTP drafter position pin

## Outcome

Align the Gemma 4 MTP assistant drafter with the Hugging Face constant-position
reference convention by pinning the drafter position for every step in a draft
round, then test whether that alignment repairs low slot-1 acceptance.

Decision: `reject_candidate` for the acceptance-fix hypothesis. Keep the pin as
behaviorally neutral reference-convention alignment.

## Scope

- Runtime behavior change:
  - `NativeMtpAssistantModel::draft_block` passes `first_position` to every
    assistant draft step.
- Reference boundary:
  - Hugging Face `SinglePositionMultiTokenCandidateGenerator` computes
    `position_ids = input_ids.shape[1] - 1` once before its drafter loop and
    passes that same value on every assistant call.
  - Its docstring states Gemma 4 shared-KV MTP effectively locks the assistant
    to a constant `position_ids` value.
- Non-goals:
  - Do not change target verification, rollback, block-prefix policy, lazy
    draft policy, trace layout, public ABI, server defaults, adapters, sampling,
    or MTP default enablement.
  - Do not cite `logit_margins`; current trace margins are diagnostic debt.

## Prediction

XR48 showed block-2 per-slot acceptance with slot 0 already position-correct and
slot 1 shifted by one RoPE position:

| Workload | XR48 slot 0 | XR48 slot 1 | Overall |
|---|---:|---:|---:|
| `chat_short_1k_001` | `60%` | `92%` | `0.719` |
| `tool_json_1k_001` | `68%` | `92%` | `0.781` |
| `mtp_candidate_1k_001` | `67%` | `17%` | `0.467` |

XR54 tests whether the pin raises `mtp_candidate_1k_001` slot-1 acceptance
toward the high-acceptance band without regressing `chat_short_1k_001` or
`tool_json_1k_001`.

## Gates

- Greedy exactness must hold for MTP block sizes 1 and 2 against non-MTP native
  greedy.
- Verification commands must pass:
  - `cargo fmt --all --check`
  - `git diff --check`
  - `cargo test -p gemma4d-ffi --lib`
  - `cargo test -p gemma4d-server --all-targets`
  - `cargo test -p gemma4d-bench --example xr15_mtp_policy_variance_ab --no-run`
  - native MTP smoke using `p05_native_mtp`
- A/B evidence must use the XR48 selected configuration:
  - `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  - `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`
  - `--adaptive-zero-accept-run 3`
  - `--adaptive-min-generated-tokens 12`
  - `3` measured trials plus `1` warmup
  - `32` generated tokens
  - workloads `chat_short_1k_001`, `tool_json_1k_001`,
    `mtp_candidate_1k_001`

## Commands

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p05_native_mtp -- --out-dir benchmarks/out/XR54-mtp-position-pin/rung10-native-mtp --model-path artifacts/models/gemma-4-12B-it-4bit --assistant-model-path artifacts/models/gemma-4-12B-it-qat-assistant-4bit --max-new-tokens 32 --block-sizes 1,2

GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- --out-dir benchmarks/out/XR54-mtp-position-pin/pinned-xr48-config --source-replay benchmarks/out/XR14-mtp-policy-autotune/summary.json --trials 3 --warmups 1 --max-new-tokens 32 --block-sizes 2 --adaptive-zero-accept-run 3 --adaptive-min-generated-tokens 12 --clear-workload-ids --workload-id chat_short_1k_001 --workload-id tool_json_1k_001 --workload-id mtp_candidate_1k_001
```

## Required Evidence

- `benchmarks/out/XR54-mtp-position-pin/rung10-native-mtp/`
- `benchmarks/out/XR54-mtp-position-pin/pinned-xr48-config/`
- `benchmarks/out/XR54-mtp-position-pin/xr54-r-mtp-candidate-one-trial/`
- `benchmarks/out/XR54-mtp-position-pin/pytorch-parity/`
- Per-slot acceptance table derived from `records.jsonl`.
- `BENCHMARKS.md` ledger row and claim-boundary update stating that pre-XR54
  MTP acceptance numbers are historical.

## Evidence

The native MTP smoke passed for block sizes 1 and 2:

| Probe | Block | Exact | Attempted | Accepted | Recommendation |
|---|---:|---|---:|---:|---|
| `hello_smoke` | 1 | `true` | `1` | `0` | `keep_disabled_auto_disable_gate` |
| `hello_smoke` | 2 | `true` | `2` | `0` | `keep_disabled_auto_disable_gate` |
| `hello_reference_prefix` | 1 | `true` | `1` | `0` | `keep_disabled_auto_disable_gate` |
| `hello_reference_prefix` | 2 | `true` | `2` | `0` | `keep_disabled_auto_disable_gate` |

The XR48-config A/B completed with `decision: keep_experimental`, `12/12`
exact records, and `9/9` measured exact records. The net-latency guarded policy
still selected only `chat_short_1k_001:block2` and `tool_json_1k_001:block2`
with aggregate `14.598%` speedup.

Measured per-slot acceptance did not move versus XR48:

| Workload | XR48 slot 0 | XR48 slot 1 | XR54 slot 0 | XR54 slot 1 | XR54 overall |
|---|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `36/60 = 0.600` | `33/36 = 0.917` | `36/60 = 0.600` | `33/36 = 0.917` | `69/96 = 0.719` |
| `tool_json_1k_001` | `39/57 = 0.684` | `36/39 = 0.923` | `39/57 = 0.684` | `36/39 = 0.923` | `75/96 = 0.781` |
| `mtp_candidate_1k_001` | `18/27 = 0.667` | `3/18 = 0.167` | `18/27 = 0.667` | `3/18 = 0.167` | `21/45 = 0.467` |

Measured latency and memory:

| Workload | Exact | Baseline p50 ms | MTP p50 ms | Speedup | Fallback p50 ms | Auto-disabled | Peak MLX | Active KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `chat_short_1k_001` | `3/3` | `2956.027` | `2421.899` | `+18.069%` | `0.000` | `0/3` | `8.002 GB` | `352845824` |
| `tool_json_1k_001` | `3/3` | `2955.781` | `2205.883` | `+25.371%` | `0.000` | `0/3` | `8.002 GB` | `352845824` |
| `mtp_candidate_1k_001` | `3/3` | `2883.842` | `2987.766` | `-3.604%` | `1307.574` | `3/3` | `8.008 GB` | `352829440` |

Sanity check: all measured `draft_tokens` arrays in XR54 are byte-identical to
XR48 for matching workload/trial records. This means the reference-aligned
position pin did not affect native drafter outputs in this runtime path.

## XR54-R Review Rerun

Claude flagged the original XR54 A/B as potentially stale-binary evidence. The
review rerun started from a cleaned `target/` and rebuilt with
`GEMMA4D_REQUIRE_MLX=1`. Provenance before measurement:

- Source pin edit mtime: `native_model.cc` `2026-07-03 00:26:39`.
- Fresh MLX build: `CMakeCache.txt` contains `GEMMA4D_REQUIRE_MLX:BOOL=ON`.
- Native objects: `model_manifest.cc.o` `00:52:47`,
  `native_model.cc.o` `00:52:48`, `runtime.cc.o` `00:52:48`.
- Native archive: `libgemma4_mlx.a` `2026-07-03 00:52:48`.
- Runner binary: `target/debug/examples/xr15_mtp_policy_variance_ab`
  `2026-07-03 00:52:49`.

The one-leg rerun wrote
`benchmarks/out/XR54-mtp-position-pin/xr54-r-mtp-candidate-one-trial/` with
build provenance in summary and records: git SHA
`f2fb705706bc8196845b19d01170cb41e04f430f`, dirty-diff SHA-256
`b4eae5c622bd802783ba2ca18b3b15f108b5fa615626a2283745849891451bd7`,
dirty diff bytes `9177`, and runner link mtime `1783054369`.

The fresh one-leg record for `mtp_candidate_1k_001` was exact and still drafted
`[[236792, 236865], [2426, 236779], [236787, 107], [236825, 107],
[236792, 7216], [107, 236792], [107], [2861], [107]]`, byte-identical to all
three XR48 measured records. Acceptance stayed `7/15 = 0.467`. Therefore the
XR54 refutation is real, not stale-binary fallout.

The XR15 runner now stamps each evidence summary and record with git SHA,
dirty-diff SHA-256, dirty-diff byte count, runner binary path, and runner binary
link mtime; missing provenance aborts before measurement.

## PyTorch Parity Contingency

The contingency implementation added a dedicated native parity payload export:
`gemma4_kv_snapshot_save_mtp_parity` saves the existing native snapshot tensors
plus ordered standalone target token embeddings. The diagnostic harness
`xr54_drafter_pytorch_parity` exported
`benchmarks/out/XR54-mtp-position-pin/pytorch-parity/payload.safetensors` for
the first `mtp_candidate_1k_001` draft round. Payload metadata includes
`hidden.last.shape = 1x1x3840`, shared full/sliding KV, and
`target.token_embeddings.shape = 1x2x3840` for token IDs `107,236792`.

The completed PyTorch reference run used
`/Users/justin/venvs/xr54-parity/bin/python` with
`PYTHONPATH=/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/lib/python3.14/site-packages`
and the dequantized dense assistant checkpoint at
`artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32/`. The productionized
dequant path is `scripts/xr54_dequant_assistant.py`; the parity script now uses
bf16-safe safetensors loading (`framework="pt"`), constructs the reference model
under `torch.bfloat16`, casts the payload to the model dtype, and explicitly
ties `lm_head.weight` to `model.embed_tokens.weight`.

The dense checkpoint was regenerated with:

```text
/Users/justin/venvs/xr54-parity/bin/python scripts/xr54_dequant_assistant.py --src artifacts/models/gemma-4-12B-it-qat-assistant-4bit --out artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32
```

The final parity command used the runner defaults for the same Python and
PYTHONPATH:

```text
GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1 target/debug/examples/xr54_drafter_pytorch_parity --out-dir benchmarks/out/XR54-mtp-position-pin/pytorch-parity --reference-records benchmarks/out/XR54-mtp-position-pin/xr54-r-mtp-candidate-one-trial/records.jsonl
```

Run ID `xr54-parity-1783057018` completed with no blockers:

| Variant | Positions | Draft tokens | Matches native |
|---|---:|---:|---:|
| Native | n/a | `[236792, 236865]` | n/a |
| PyTorch pinned | `[1023, 1023]` | `[236792, 236865]` | `true` |
| PyTorch incremented | `[1023, 1024]` | `[236792, 236865]` | `true` |

This falsifies F9: the vendored Hugging Face reference itself is
position-insensitive for the recorded round. The byte-identical A/B evidence is
therefore faithful behavior, and the `mtp_candidate_1k_001` slot-1 collapse is
a model/content property rather than a native drafter implementation bug.

## Result

| Item | Verdict | Evidence |
|---|---|---|
| Acceptance-fix hypothesis | `reject_candidate` | XR54 A/B and XR54-R rerun were byte-identical to XR48; `mtp_candidate_1k_001` slot 1 stayed `3/18`. |
| Position pin | Keep | Reference-convention alignment; behaviorally neutral and exactness-proven. |
| PyTorch parity contingency | Complete | Pinned and incremented HF reference drafts both matched native `[236792, 236865]`. |

XR55 is unblocked. It should proceed as broader MTP performance work with the
revised expectation that acceptance is model/content-driven, not a remaining
drafter position implementation defect.

## Completion Rule

The acceptance-fix candidate is rejected because exactness remained intact but
`mtp_candidate_1k_001` acceptance did not improve. The parity contingency is
complete, so no additional drafter-implementation XR is required before XR55.
