# XR50 - QAT target MTP pairing A/B

## Outcome

Test whether pairing the QAT target artifact with the QAT MTP assistant improves
Gemma 4 MTP acceptance and net decode speed versus the current plain-target plus
QAT-assistant pairing.

## Scope

- Baseline pairing:
  - Target: `artifacts/models/gemma-4-12B-it-4bit`
  - Assistant: `artifacts/models/gemma-4-12B-it-qat-assistant-4bit`
- Candidate pairing:
  - Target: `artifacts/models/gemma-4-12B-it-qat-4bit`
  - Assistant: `artifacts/models/gemma-4-12B-it-qat-assistant-4bit`
- QAT target source: `mlx-community/gemma-4-12B-it-qat-4bit`
- QAT target revision: `e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6`
- Source replay: `benchmarks/out/XR14-mtp-policy-autotune/summary.json`
- Workloads:
  - `chat_short_1k_001`
  - `tool_json_1k_001`
  - `mtp_candidate_1k_001`
- Horizon: `32` generated tokens.
- Selected policy path:
  - `GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK=1`
  - `GEMMA4D_EXPERIMENTAL_MTP_LAZY_SECOND_DRAFT=1`
  - `--adaptive-zero-accept-run 3`
  - `--adaptive-min-generated-tokens 12`
- Trials: `3` measured plus `1` warmup.

## Artifact Validation

Downloaded target path:

```text
artifacts/models/gemma-4-12B-it-qat-4bit
```

Validated `config.json` fields:

- `model_type = gemma4_unified`
- `text_config.num_hidden_layers = 48`
- `text_config.hidden_size = 3840`
- `text_config.head_dim = 256`
- `text_config.global_head_dim = 512`
- `text_config.num_key_value_heads = 8`
- `text_config.num_global_key_value_heads = 1`
- `text_config.final_logit_softcapping = 30.0`
- root quantization is affine `bits=4`, `group_size=64`, with per-layer MLP
  8-bit overrides.
- The QAT target safetensors total `10987772430` bytes (`10.99 GB`) versus the
  plain target's `6741039511` bytes (`6.74 GB`); XR50 footprint and latency
  caveats are scoped to this mixed 4-bit plus 8-bit-MLP artifact, not QAT
  generally.

Artifact hashes:

| Field | SHA-256 |
|---|---|
| QAT target config | `fe091f98e6f7e5e80461bd8ec7ced6d87ac16987586239386ed44b82ecbc2b12` |
| QAT target tokenizer | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| QAT target tokenizer config | `a4260621db48fa22f2b09ce3ba5ad0ec0cc0e032aa702e3ab743a0bc9d6e1d06` |
| QAT target safetensors index | `b87c93774de5d13ca9d0e21b045793e42e5df032fb5e7622212524f56f9695f2` |

## Required Work

1. Verify the QAT target artifact identity and loadability.
2. Run a QAT-target exactness smoke with block sizes `1` and `2`.
3. Run a fresh baseline plain-target pairing on the XR48 selected path.
4. Run a fresh candidate QAT-target pairing on the same path.
5. Compare acceptance, rollbacks, decode-phase p50/p95, net decode speedup,
   peak MLX memory, and active KV bytes by workload.
6. Update `BENCHMARKS.md`, including headline MTP rows only for stable top-line
   numbers.
7. Keep MTP exactness as the hard veto.

## Acceptance Gates

- Candidate MTP output is byte-identical to native non-MTP output for the same
  QAT target at block sizes `1` and `2`.
- QAT target loads and runs native non-MTP generation for the selected 1K
  workloads.
- Candidate peak MLX memory stays under the configured tiny16 gate.
- Active KV bytes remain in the expected 1K shape.
- Acceptance and speed are reported separately.
- If `mtp_candidate_1k_001` clears the `5%` net-latency guard, or acceptance
  improves by at least 10 percentage points on any workload with exactness
  intact, accept the pairing candidate and propose the QAT target as the new
  default target artifact.
- If acceptance is effectively unchanged, reject the pairing hypothesis and
  redirect effort to verifier cost.

## Non-goals

- Do not start KV-backed incremental verifier work.
- Do not change MTP math, verifier semantics, sampler behavior, server
  behavior, adapter policy, active KV compression, or prefill policy in this
  goal.
- Do not treat plain-target output differences from QAT-target output as a
  failure. Exactness is judged within each target mode.
- Do not merge the feature branch directly to `main`.

## Required Artifacts

```text
benchmarks/out/XR50-qat-target-mtp-pairing/baseline-plain-target/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR50-qat-target-mtp-pairing/candidate-qat-target-block12-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR50-qat-target-mtp-pairing/candidate-qat-target-mtp-candidate-1k-smoke/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR50-qat-target-mtp-pairing/{report.md,decision.md}
```

## Result

Decision: `blocked_with_evidence`.

The QAT target downloaded and loaded successfully, and QAT-target MTP exactness
passed in completed low-N smoke probes, but XR50 did not produce defensible
evidence for default adoption.

All completed benchmark rows were cold-start smokes using `--trials 1`,
`--warmups 0`, and `--max-new-tokens 2`. Treat their latency as
JIT/compile/load dominated and not steady-state; unlike the P04 convention, these
smokes do not discard the first four decode samples.

Completed QAT runs:

- `candidate-qat-target-block12-smoke`: `chat_short_1k_001`, block sizes `1`
  and `2`, `2` generated tokens, `1` measured trial, `0` warmups. Exactness was
  `2/2`, but acceptance was `0/2` for both block sizes. Native baseline decode
  was `19713.684 ms`; MTP decode phase was `71279.259 ms` for block `1`
  (`-261.573%`) and `52344.264 ms` for block `2` (`-165.522%`). Peak MLX was
  about `11.90 GB`.
- `candidate-qat-target-mtp-candidate-1k-smoke`: `mtp_candidate_1k_001`, block
  size `2`, `2` generated tokens, `1` measured trial, `0` warmups. Exactness was
  `1/1`; acceptance was `2/2`; native baseline decode was `13510.088 ms`; MTP
  decode phase was `25448.830 ms` (`-88.369%`). Peak MLX was `11.902 GB`.

Attempted but stopped paths:

- A sandboxed smoke failed before benchmarking because MLX could not access the
  Mac Metal device.
- The required fresh `baseline-plain-target` leg was not attempted. XR48 remains
  only a stale comparator with different parameters: `3` trials, `1` warmup, and
  `32` generated tokens versus XR50's `1` trial, `0` warmups, and `2` generated
  tokens.
- A broader 3-workload block-1/2 smoke and two 32-token selected-path QAT runs
  were stopped before artifacts because this mixed 4-bit plus 8-bit-MLP target
  reload/decode runtime was too long and pushed heavy 16GB memory pressure.

XR50 does not promote this QAT target artifact as the default target artifact.
The evidence points back to verifier/runtime cost as the next MTP bottleneck. A
separate reduced QAT-target baseline goal would be required before making any
target-default decision.

Artifact cleanup: after the blocked decision, the unused local QAT target model
directory `artifacts/models/gemma-4-12B-it-qat-4bit` was removed. The retained
model artifacts are the plain target and QAT assistant used by current
benchmarks.

## Completion Rule

Stop when the QAT target pairing has fresh measured evidence against the current
plain-target pairing, or when blockers explain why it cannot be judged.
