# XR60 - DSpark native MLX speculative decoding for Helios Gemma 4 12B

> Long-running Codex goal file for autonomous implementation, benchmark, and optimization work.
>
> Intended repo location: `codex/goals/XR60-dspark-native-mlx.goal.md`
>
> Generated from the MLX-only DSpark + Helios research report and current Helios MTP/native-graph conventions.

```text
goal Implement and benchmark an MLX/Helios-only DSpark speculative decoding path for Gemma 4 12B using Helios's existing native graph, Rust engine, C ABI, KV-backed verifier, exact greedy parity gates, and benchmark harness. Do not add GGUF, llama.cpp, subprocess inference runtimes, or external speculative decoding runtimes. Start from the released DeepSpec Gemma DSpark architecture/checkpoint as the reference, create revision-pinned PyTorch fixtures, port the drafter to Python/MLX for parity, then move the validated path behind native Helios C ABI/Rust integration. Expose selected target hidden-state taps for layers [5, 17, 29, 41, 46], implement fixed-prefix DSpark scheduling for lengths 1/2/4/7 first, reuse `gemma4_verify_tokens` semantics for exact commit/rollback, then add confidence scheduling and measured custom MLX kernel optimizations only after correctness is proven. Keep the feature default-off unless every correctness, memory, and benchmark gate is satisfied. Produce `benchmarks/out/XR60-dspark-native-mlx/{records.jsonl,summary.json,report.md,blockers.md,decision.md}` and update repo docs/evidence with exact commands, git SHA, benchmark records, speedups/regressions, blockers, and next steps.
```

## Operator intent

Run Codex autonomously for a long overnight implementation and optimization pass. The objective is not a paper reproduction and not a new runtime integration. The objective is to turn the existing Helios MTP/native graph foundation into a DSpark-capable, MLX-native speculative decode path for the local Gemma 4 12B 4-bit target.

The work should focus on the pieces Helios already owns:

- Native Gemma/MLX graph and selected hidden-state taps.
- Narrow C ABI and Rust FFI wrappers.
- Rust speculative decoding state machine and metrics.
- KV-backed verification, exact greedy parity, commit/rollback, and auto-disable behavior.
- Existing benchmark harnesses and real-context workloads.
- MLX-native drafter implementation and eventually custom MLX/Metal kernels.

Do not pull in a separate local inference dependency to validate DSpark. Do not add a GGUF path. Do not add llama.cpp. Do not add a subprocess server baseline. Treat Helios's native graph and helper-backed target path as the only acceptable integration base.

## Why this is the right goal shape

This is a performance/research implementation goal with a clear finish line but an uncertain path. It needs repeated inspect -> patch -> test -> benchmark -> decide loops. It must not stop when a partial implementation compiles; it should stop only when evidence proves success or a blocker is documented.

The completion standard must be evidence-based:

- Exact greedy output parity against the same target without speculative decoding.
- Target hidden tap parity against the reference fixture.
- PyTorch reference fixture parity before MLX implementation claims.
- Benchmark evidence with decode tok/s, accepted tokens per verify, acceptance rate, stage timings, memory, and rollback count.
- Clear default-off safety behavior if speed is not profitable or correctness is not proven.

## Source evidence to keep in working memory

### DSpark/DeepSpec facts

- DeepSpec is the reference codebase for DSpark data prep, draft model implementation, training, and evaluation.
- The Gemma DSpark configuration targets `google/gemma-4-12B-it`.
- Key Gemma DSpark config values:
  - `block_size = 7`
  - `num_draft_layers = 5`
  - `target_layer_ids = [5, 17, 29, 41, 46]`
  - `mask_token_id = 4`
  - `num_anchors = 512`
  - `markov_rank = 256`
  - `markov_head_type = "vanilla"`
  - `confidence_head_alpha = 1.0`
  - `confidence_head_with_markov = true`
  - training precision `bf16`
  - training `max_length = 4096`
- DSpark consumes selected target hidden states. It does not only need final logits.
- The DSpark Gemma implementation projects concatenated selected target hidden states through `fc`, normalizes them, then runs a compact Gemma-style draft stack over masked/noise draft positions.
- The Markov head adds low-rank sequential token-conditioned bias. For greedy MLX this is a likely custom-kernel opportunity: rank-256 previous-token embedding -> vocab bias -> fused argmax/top-k.
- Confidence scheduling should not be first. Fixed-prefix 1/2/4/7 must pass exactness and show measurable speed before confidence thresholds are trusted.
- Full target-hidden cache training can be extremely large. Use the released draft first; reserve Modal for calibration or small finetuning if Helios prompt acceptance is poor.

### Helios facts

- Helios currently targets local Gemma 4 12B 4-bit inference on Apple Silicon through the Rust CLI, C ABI / MLX-LM helper, localhost API, and TUI.
- Current M12 helper-backed baseline is roughly 12-16 decode tok/s depending context length; M12 table shows 1K around 15.6 tok/s, 4K around 12.2 tok/s, 8K around 15.3 tok/s, and 16K around 12.9 tok/s.
- Helios already defines the speculative loop: prefill target, store KV, draft N tokens, verify one target pass, accept longest valid prefix, commit accepted states, roll back rejected states, emit accepted tokens, repeat.
- Helios invariant: at temperature 0, non-MTP greedy token sequence must equal speculative/MTP greedy token sequence for the same target mode. If invariant fails, the mode must auto-disable and report the failing fixture.
- Helios already has native MTP-related C ABI entrypoints and Rust wrappers, including `gemma4_mtp_draft_block`, `gemma4_verify_tokens`, `draft_block_with_scores`, `verify_tokens`, `StepResult`, and `MtpTraceInfo`.
- Existing MTP admission boundaries should remain: text-only, temperature 0 first, adapters disabled, no unverified compressed active KV, no sampling speculative decoding.
- Existing result style should match prior goal files under `codex/goals/`, especially exact commands, artifacts, decision, and completion rule.

## Desired end state

A DSpark path exists beside the current MTP assistant path. It is not a replacement for the verifier or the native target. It is a new drafter family plus scheduler metadata.

At the end of a successful pass:

1. Helios can load or reference an MLX-converted DSpark Gemma 12B block-7 draft artifact.
2. The native target graph can expose selected hidden taps `[5, 17, 29, 41, 46]` and last hidden state without CPU copies in the hot path whenever feasible.
3. A DSpark drafter can propose tokens at temperature 0 through a Helios-owned MLX path.
4. Helios verifies the scheduled DSpark prefix through the existing KV-backed target verifier semantics.
5. Generated output is byte-identical to non-spec native target greedy output on the measured fixture corpus.
6. Benchmarks show whether DSpark improves speed, regresses speed, or is blocked by draft/verify/hidden-copy overhead.
7. The feature remains default-off unless all gates pass.
8. Evidence artifacts explain exactly what was implemented, what was measured, what remains blocked, and which next experiment should run.

## Performance target and interpretation

Use these as optimization targets, not correctness substitutes:

- Early useful win: at least +15% decode tok/s at 1K and 4K fixed-prefix tests with exactness passing.
- Practical target: mixed interactive Gemma 4 12B decode in the 23-32 tok/s range when DSpark is enabled and exactness passes.
- Code/repo target: 30+ tok/s on high-acceptance code/repo continuation workloads.
- Stretch target: 35-45 tok/s only for high-acceptance workloads after native DSpark, hidden tap caching, and Markov/argmax kernel work.

Never trade exactness for speed in this goal. A fast run with mismatch is a failed run. A fast run that requires GGUF/llama.cpp/external runtime is out of scope. A fast run that only works by changing target semantics is out of scope.

## Scope

### In scope

- Native Gemma/MLX target graph work required for selected hidden taps.
- DSpark reference fixture generation using DeepSpec/PyTorch.
- Python/MLX DSpark parity implementation, if useful as a stepping stone.
- MLX weight conversion for the released DSpark Gemma checkpoint.
- New DSpark-specific manifest and loader code.
- New C ABI functions or extensions that preserve existing ABI behavior.
- Rust FFI wrappers and engine integration.
- Fixed-prefix DSpark scheduler for lengths 1, 2, 4, and 7.
- Confidence outputs and confidence scheduler after fixed-prefix exactness.
- Benchmark harnesses and real-context workload integration.
- Custom MLX/Metal kernels only after profiling identifies a hot path.
- Modal skeleton/config only for calibration or small finetune planning; do not launch full retraining unless explicitly configured and justified by benchmark evidence.

### Out of scope

- GGUF, llama.cpp, or any external local inference runtime.
- Subprocess server integration for DSpark.
- Sampling speculative decoding.
- Adapter-active DSpark.
- Unverified compressed active KV DSpark.
- Default-on behavior.
- Broad model-family generalization beyond Gemma 4 12B.
- Multimodal Gemma support.
- Full DeepSpec retraining as the first step.
- Any public performance claim not backed by committed benchmark artifacts.

## Branch and artifact naming

Use a branch name similar to:

```text
xr60-dspark-native-mlx
```

Use this artifact root:

```text
benchmarks/out/XR60-dspark-native-mlx/
```

Required final artifact set:

```text
benchmarks/out/XR60-dspark-native-mlx/records.jsonl
benchmarks/out/XR60-dspark-native-mlx/summary.json
benchmarks/out/XR60-dspark-native-mlx/report.md
benchmarks/out/XR60-dspark-native-mlx/blockers.md
benchmarks/out/XR60-dspark-native-mlx/decision.md
```

Optional subdirectories:

```text
benchmarks/out/XR60-dspark-native-mlx/00-baseline/
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/
benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/
benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/
benchmarks/out/XR60-dspark-native-mlx/04-fixed-prefix/
benchmarks/out/XR60-dspark-native-mlx/05-confidence-scheduler/
benchmarks/out/XR60-dspark-native-mlx/06-kernel-ab/
```

Do not commit model weights, downloaded checkpoints, or giant fixture tensors unless repo policy already permits a small fixture. Prefer manifests, checksums, scripts, and small deterministic samples.

## Suggested repo paths

Likely touched paths:

```text
codex/goals/XR60-dspark-native-mlx.goal.md
native/gemma4_mlx/include/gemma4_mlx.h
references/ffi/gemma4_mlx.h
native/gemma4_mlx/src/runtime.cc
native/gemma4_mlx/src/model_manifest.cc
crates/gemma4d-ffi/src/lib.rs
crates/gemma4d-engine/src/lib.rs
crates/gemma4d-bench/examples/dspark_fixed_block_matrix.rs
crates/gemma4d-bench/src/*
crates/gemma4d-server/src/*
references/configs/tiny16.toml
BENCHMARKS.md
docs/evidence/XR60-dspark-native-mlx.md
tools/dspark/export_reference_fixture.py
tools/dspark/convert_to_mlx.py
tools/dspark/compare_mlx_parity.py
tools/dspark/README.md
```

Avoid changing unrelated server/API/TUI defaults unless the exact change is required to surface DSpark metrics and is backward compatible.

## Proposed implementation architecture

### Rust engine shape

Add a generic speculative-drafter interface that can support DSpark without breaking existing MTP assistant code.

Sketch:

```rust
pub struct DraftBlock {
    pub tokens: Vec<i32>,
    pub token_logits: Vec<f32>,
    pub token_logprobs: Option<Vec<f32>>,
    pub confidence: Option<Vec<f32>>,
    pub max_len: usize,
    pub scheduled_len: usize,
    pub draft_ms: f64,
    pub scheduler_us: u64,
}

pub enum SpecMode {
    Off,
    MtpAssistant,
    DSpark,
}

pub enum SpecScheduler {
    Fixed { len: usize },
    ConfidenceThreshold { threshold: f32, max_len: usize },
    HardwareAware { profile_path: std::path::PathBuf, max_len: usize },
}

pub trait SpecDrafter {
    fn draft_block(
        &mut self,
        cache: &mut KvCache,
        max_len: std::num::NonZeroU32,
    ) -> Result<DraftBlock>;
}
```

This is a shape, not a required exact API. Prefer integrating cleanly with existing `MtpConfig`, `Drafter`, and FFI wrappers if that is less invasive.

### C ABI shape

Prefer DSpark-specific entrypoints rather than overloading MTP assistant semantics:

```c
typedef struct Gemma4DSparkDrafter Gemma4DSparkDrafter;

typedef struct Gemma4DSparkDraftResult {
    uint32_t token_count;
    uint32_t scheduled_count;
    int32_t tokens[GEMMA4_MTP_MAX_DRAFT_TOKENS];
    float token_logits[GEMMA4_MTP_MAX_DRAFT_TOKENS];
    float token_margins[GEMMA4_MTP_MAX_DRAFT_TOKENS];
    float confidence[GEMMA4_MTP_MAX_DRAFT_TOKENS];
    double draft_ms;
    double scheduler_us;
} Gemma4DSparkDraftResult;

Gemma4Status gemma4_load_dspark_drafter(
    const Gemma4LoadConfig* config,
    Gemma4Target* target,
    Gemma4DSparkDrafter** out);

Gemma4Status gemma4_free_dspark_drafter(Gemma4DSparkDrafter* drafter);

Gemma4Status gemma4_dspark_draft_block(
    Gemma4DSparkDrafter* drafter,
    Gemma4KvCache* cache,
    uint32_t max_block_size,
    Gemma4DSparkDraftResult* out);
```

If adding new opaque types is too heavy for the overnight pass, add an internal DSpark backend behind the existing drafter handle only if the manifest/config can distinguish `gemma4_unified_assistant` from `Gemma4DSparkModel` cleanly and all current tests still pass.

### Native target hidden taps

Implement a narrow hidden tap registry:

```text
required target taps: [5, 17, 29, 41, 46]
required final hidden: last target hidden for aligned logits / diagnostics
hot-path rule: selected taps only, cache-owned MLX arrays/views, no all-layer capture
```

Requirements:

- Taps are opt-in and only active when DSpark mode requires them.
- Taps preserve dtype and shape expected by DeepSpec reference fixtures.
- Taps are cache-owned or target-owned with clear lifetime rules.
- Advancing/resetting/freeing cache invalidates views safely.
- No CPU round trips in the decode hot path unless explicitly marked as temporary and measured.

### Verifier semantics

Reuse `gemma4_verify_tokens` behavior:

- Verify scheduled prefix only.
- Accept the longest matching draft prefix.
- Commit matching draft tokens.
- Commit target fallback token on first mismatch.
- Do not append rejected draft tokens to KV cache.
- Preserve terminal no-lookahead behavior where applicable.
- Preserve exact greedy parity vs non-spec target.

If DSpark drafts 7 tokens but scheduler selects 4, verify only 4. Record dropped draft tokens separately.

## Suggested config shape

Add or prototype a config similar to:

```toml
[speculative]
mode = "dspark"
temperature = 0.0
max_draft_tokens = 7
scheduler = "fixed"
fixed_tokens = 1
require_native_hidden_taps = true
auto_disable_min_acceptance_rate = 0.35

[speculative.dspark]
draft_path = "artifacts/drafts/dspark-gemma4-12b-block7-mlx"
target_layer_ids = [5, 17, 29, 41, 46]
confidence_enabled = false
markov_rank = 256
```

Config must keep DSpark default-off. Do not alter existing MTP defaults unless a test proves old behavior remains identical.

## Work plan

### Phase 0 - Prepare and map current state

1. Record current git SHA and working tree status.
2. Read existing `AGENTS.md`, `README.md`, MTP specs, M06 decision record, latest benchmark docs, and recent XR goals/results.
3. Run or inspect baseline commands that are cheap and available.
4. Determine whether local model artifacts exist:
   - `artifacts/models/gemma-4-12B-it-4bit`
   - existing assistant artifacts, if any
   - DSpark checkpoint/artifact path, if any
5. Create `benchmarks/out/XR60-dspark-native-mlx/blockers.md` immediately and append blockers as they occur rather than waiting until the end.
6. Do not begin large refactors until the current test and benchmark surface is understood.

Suggested starting commands:

```bash
git status --short
git rev-parse HEAD
make verify
cargo test -p gemma4d-engine --all-targets
cargo test -p gemma4d-ffi --all-targets
```

If `make verify` is too slow or blocked by environment, record the blocker and run narrower tests first.

### Phase 1 - Reference fixture path

Goal: establish a deterministic DeepSpec/PyTorch source of truth.

Work:

1. Add `tools/dspark/README.md` explaining fixture and conversion workflow.
2. Add `tools/dspark/export_reference_fixture.py` or equivalent.
3. Pin references in a manifest:
   - DeepSpec commit.
   - Target model revision.
   - Tokenizer revision.
   - DSpark checkpoint revision.
   - Python package versions.
   - Prompt fixture checksums.
4. Generate a tiny deterministic fixture set if artifacts are available.
5. Fixture should include at minimum:
   - input token ids
   - target selected hidden taps for layers `[5, 17, 29, 41, 46]`
   - target last hidden state where applicable
   - DSpark base logits
   - Markov-corrected logits
   - confidence values
   - greedy draft tokens at temperature 0
6. If checkpoint or Python dependencies are unavailable, write the scripts and manifest stubs anyway, and mark fixture generation blocked with exact error output.

Exit criterion:

```text
benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/manifest.json exists, or blockers.md explains exactly why it cannot be produced.
```

### Phase 2 - Native hidden tap registry

Goal: expose selected target hidden states from the native Gemma/MLX graph.

Work:

1. Locate the native graph layer loop and existing last-hidden/shared KV materialization.
2. Add opt-in tap selection for `[5, 17, 29, 41, 46]`.
3. Keep taps disabled unless DSpark mode requires them.
4. Add tests for tap selection parsing, shape metadata, and lifetime safety.
5. Add a debug/fixture path that can export small tap summaries without dumping huge tensors.
6. Avoid hot-path CPU copies; if a CPU copy is unavoidable in the first pass, mark it temporary and add timing/bytes metrics.

Exit criterion:

```text
Native graph can report selected tap shapes and last hidden shape for at least a smoke prompt, or blockers.md explains exact native graph limitation.
```

### Phase 3 - Python/MLX DSpark parity

Goal: prove the DSpark model can run in MLX against saved target hidden taps.

Work:

1. Add `tools/dspark/convert_to_mlx.py` for safetensors/checkpoint -> MLX weight bundle.
2. Add `tools/dspark/compare_mlx_parity.py` for PyTorch fixture vs MLX output.
3. Implement or stub an MLX Gemma4DSparkModel equivalent:
   - embeddings
   - `fc` over selected hidden taps
   - hidden norm
   - compact draft decoder layers
   - lm head
   - Markov rank-256 head
   - confidence head
4. Validate tensor mapping with strict missing/extra tensor reports.
5. Compare top-1 tokens first. Then compare logits/confidence within tolerance.

Exit criterion:

```text
Top-1 draft token parity passes on fixture prompts, or blockers.md captures missing MLX/API/checkpoint details.
```

### Phase 4 - Helios C ABI and Rust integration

Goal: wire DSpark into Helios as a native drafter using the existing verifier.

Work:

1. Add DSpark-specific loader/manifest validation.
2. Add C ABI result struct for tokens, scores, confidence, and timing.
3. Add Rust FFI wrappers.
4. Add a fixed scheduler for lengths 1/2/4/7.
5. Feed only scheduled prefix into `verify_tokens`.
6. Extend metrics with DSpark-specific fields without breaking MTP fields.
7. Add server/TUI/API reporting only if already structured and low risk; otherwise add benchmark-only metrics first.
8. Preserve current MTP assistant behavior exactly.

Exit criterion:

```text
A Helios benchmark/example can attempt DSpark fixed-prefix drafting and verification through the native/Rust path, even if performance is not yet profitable.
```

### Phase 5 - Exactness and fixed-prefix benchmark matrix

Goal: prove exact greedy parity and get first speed numbers.

Suggested benchmark dimensions:

```text
context lengths: 1K, 4K, 8K, 16K when available; 32K one-token memory probe optional
workloads: chat_short_1k_001, mtp_candidate_4k_001, repo/code workloads, benchmark QA prompts
max_new_tokens: 32 and 128 where practical
block/scheduled lengths: off baseline, 1, 2, 4, 7
trials: 1 warmup + 3 measured when runtime permits
mode: temperature=0 only
```

Suggested command shape, adjust to real harness names:

```bash
GEMMA4D_REQUIRE_MLX=1 \
GEMMA4D_USE_NATIVE_GRAPH=1 \
GEMMA4D_EXPERIMENTAL_DSPARK=1 \
cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- \
  --out-dir benchmarks/out/XR60-dspark-native-mlx/04-fixed-prefix \
  --model-path artifacts/models/gemma-4-12B-it-4bit \
  --draft-path artifacts/drafts/dspark-gemma4-12b-block7-mlx \
  --block-sizes 1,2,4,7 \
  --max-new-tokens 32 \
  --trials 3 \
  --warmups 1
```

If `dspark_fixed_block_matrix` does not exist, create it or adapt the closest existing XR/MTP benchmark harness.

Required per-record metrics:

- git SHA
- command and env flags
- workload id
- prompt checksum
- context length
- generated token count
- baseline token sequence checksum
- DSpark token sequence checksum
- exactness boolean
- mismatch position and token ids if any
- scheduled length
- attempted draft tokens
- scheduled draft tokens
- accepted draft tokens
- accepted tokens per verify
- acceptance rate
- target verify passes
- rollback count
- draft_ms
- scheduler_us
- verify_stage_ms
- verify_forward_ms
- repair_ms
- decode tok/s
- decode phase ms
- peak native GB / peak MLX GB
- peak RSS MB
- active KV bytes
- hidden tap bytes
- draft resident bytes
- auto-disable reason if any

Exit criterion:

```text
records.jsonl and summary.json exist for fixed-prefix runs, with exactness status for every measured record.
```

### Phase 6 - Confidence scheduler

Start only after fixed-prefix exactness passes.

Work:

1. Export DSpark confidence per position.
2. Add calibration metrics:
   - Brier/log loss when available
   - expected tau vs realized tau
   - confidence bias
   - calibration buckets
3. Implement a conservative confidence threshold scheduler.
4. Compare confidence scheduler to fixed 1/2/4/7.
5. Auto-disable below acceptance/speed thresholds.

Exit criterion:

```text
Confidence scheduling either improves net latency with exactness preserved or is rejected with evidence.
```

### Phase 7 - Custom MLX kernel optimization

Start only after profiler/benchmarks identify real bottlenecks.

Priority order:

1. Selected hidden tap extraction and view/copy elimination.
2. KV-safe multi-token verification staging/rollback overhead.
3. Markov rank-256 bias + argmax/top-k fusion for greedy path.
4. Confidence head and threshold scheduling overhead.
5. Fused DSpark attention over target context + draft positions.
6. Ragged verification packing for multiple requests.

Rules:

- Every kernel change needs an A/B correctness fixture.
- Every kernel change needs a before/after timing table.
- Do not optimize all hidden layers; only selected DSpark taps matter.
- Do not quantize draft weights before BF16/FP16 parity and correctness pass.
- Keep kernel fallback path available.

Exit criterion:

```text
Kernel A/B report shows correctness preserved and net measured improvement, or rejects the kernel as not worth carrying.
```

### Phase 8 - Modal calibration or finetune planning

This is optional. Do not make it the first path.

Use Modal only if:

- Released DSpark checkpoint is correctly integrated but acceptance is too low on Helios workloads.
- Confidence is poorly calibrated for Helios prompt families.
- There is evidence that small domain finetuning could improve net decode speed.

Prefer calibration and small subset finetuning first.

Required Modal manifest fields:

- target revision
- tokenizer revision
- DeepSpec commit
- base DSpark checkpoint
- data hash
- layer tap ids
- sequence length
- training/eval sampling settings
- Modal image hash or package versions
- checkpoint output path
- eval summary path

Do not launch a full target-hidden cache build without explicit storage budget and an operator decision.

## Acceptance gates

### G0 - Baseline state captured

Pass when:

- Git SHA and working tree status recorded.
- Existing tests or blockers recorded.
- Baseline MTP/native graph benchmark state is understood.

### G1 - Reference fixture determinism

Pass when:

- DeepSpec/PyTorch DSpark reference fixture is deterministic at temperature 0, or unavailable dependencies/checkpoint are documented as blockers.
- Manifest pins all revisions and prompt checksums.

### G2 - MLX DSpark parity

Pass when:

- MLX DSpark top-1 draft tokens match PyTorch reference on fixture prompts.
- Logits/confidence tolerance is defined and measured.
- Tensor mapping report has no unexplained missing/extra tensors.

### G3 - Native hidden tap parity

Pass when:

- Native graph selected tap shapes match expected Gemma DSpark shapes.
- Sampled values match reference within tolerance where comparison is feasible.
- Tap lifetime and cache advancement safety are tested.

### G4 - Greedy exactness

Pass when:

- DSpark speculative output is byte-identical to native non-spec target output across measured fixture corpus.
- Any mismatch auto-disables DSpark and records failing prompt, position, draft token, target token, and top-k trace.

### G5 - Memory safety

Pass when:

- Target + DSpark + KV fits tiny16 budget for 1K/4K.
- 8K/16K memory is measured and does not create unacceptable pressure.
- 32K is treated only as optional one-token memory probe unless sustained decode is explicitly measured.

### G6 - Net speed

Pass when:

- At least one fixed or confidence scheduler shows >=15% decode tok/s improvement on 1K/4K with exactness passing.
- No promoted scheduler regresses key workloads without being rejected by the decision logic.
- Stage timings identify draft, scheduler, verify, repair, and hidden tap overhead.

### G7 - Default-off safety

Pass when:

- DSpark remains default-off unless all correctness, speed, and memory gates pass.
- Existing MTP assistant and non-MTP target behavior remain unchanged.
- Adapter and compressed-KV admission disables remain intact.

## Decision logic

At the end, write `decision.md` with exactly one of:

```text
promote_experimental
keep_experimental
reject_for_now
blocked
```

Use this logic:

- `promote_experimental`: Exactness passes, memory passes, at least one stable scheduler provides meaningful net speedup, and the feature remains opt-in/default-off.
- `keep_experimental`: Correctness passes and some evidence is promising, but speed/memory/calibration is not broad enough for a stronger recommendation.
- `reject_for_now`: Correctness may pass but speed regresses or architecture overhead appears too high under current implementation.
- `blocked`: Required checkpoint/model/Metal/native graph fixture/MLX API is unavailable or broken, and the blocker is documented with commands and logs.

Never write `promote_experimental` if exactness fails.

## Completion rule

Mark this goal complete only when one of these is true:

1. DSpark native MLX path has evidence artifacts proving exactness, memory status, and benchmark result/decision for fixed-prefix and any attempted confidence/kernel work.
2. The goal is blocked with exact attempted commands, observed output, missing dependency/artifact, and next input needed.
3. The implementation is rejected for now with benchmark evidence showing why DSpark is not currently profitable.

Partial code without artifacts is not complete. Passing tests without benchmarks is not complete. Benchmarks without exactness are not complete. A report without commands and artifact paths is not complete.

## Iteration policy for Codex

After each attempt:

1. Record what changed.
2. Run the narrowest relevant test.
3. Run or update the smallest benchmark that can confirm/deny the hypothesis.
4. Compare against the previous artifact.
5. Decide the next highest-leverage action based on evidence, not intuition.
6. Append blockers immediately.
7. Keep unrelated files untouched.

If a step fails:

- Prefer a minimal reproduction over broad refactoring.
- Add a fixture for the failure before fixing if feasible.
- Do not suppress assertions that protect exactness or memory safety.
- Do not widen scope to sampling/adapters/compressed KV to avoid a blocker.

## Suggested subagent plan

Codex subagents must be requested explicitly. At the start of the Codex run, use a prompt like this:

```text
Use the XR60 goal file as the operating contract. Spawn these subagents in parallel, wait for all results, and synthesize a concrete implementation plan before editing code:

1. dspark_archaeologist: map DeepSpec DSpark Gemma architecture, checkpoint tensor names, required hidden taps, Markov/confidence outputs, and fixture requirements.
2. helios_native_mapper: map Helios native Gemma/MLX graph, KV cache, last-hidden materialization, MTP verifier, and where selected hidden taps should be exposed.
3. ffi_engine_mapper: map C ABI, Rust FFI, gemma4d-engine speculative loop, metrics, and benchmark harness integration points.
4. benchmark_optimizer: map existing MTP/XR benchmark harnesses, workloads, output schema, and how to add DSpark fixed-prefix and confidence benchmarks with exactness gates.
5. correctness_reviewer: read the planned changes and list exactness, rollback, cache lifetime, adapter/KV admission, and default-off risks before implementation starts.

After all agents report, implement the smallest path that can reach G1-G4 first. Then benchmark and optimize toward G5-G6.
```

### Subagent responsibilities

#### `dspark_archaeologist`

Use when reading DeepSpec, checkpoint manifests, and MLX conversion code.

Return:

- Tensor map.
- Required modules.
- Shape expectations.
- Hidden tap requirements.
- Fixture schema.
- Known blockers.

Do not edit Helios runtime code.

#### `helios_native_mapper`

Use when tracing native C++/MLX target graph and KV cache.

Return:

- Layer loop entry points.
- Last-hidden and KV ownership/lifetime notes.
- Best hidden tap insertion point.
- Potential CPU-copy risks.
- Test points.

Avoid implementation until the parent agent assigns a specific change.

#### `ffi_engine_mapper`

Use when tracing ABI, Rust wrappers, and engine state machine.

Return:

- Minimal API changes.
- Compatibility risks.
- Metrics schema.
- Verification and rollback flow.
- Unit/integration tests to add.

#### `benchmark_optimizer`

Use when finding or creating benchmarks.

Return:

- Existing harnesses to reuse.
- Exact command templates.
- Artifact schema.
- Baseline/comparator plan.
- Runtime/memory caveats.

#### `correctness_reviewer`

Use after implementation plan and before final decision.

Return concrete findings only:

- Token exactness risk.
- KV cache lifetime risk.
- Hidden tap shape/dtype risk.
- Rejected-token commit risk.
- Default-on/admission risk.
- Missing tests.

## Optional project-scoped subagent config snippets

If the repo uses project-scoped Codex agents, add these files under `.codex/agents/`. Keep `max_depth = 1` so subagents do not recursively fan out.

### `.codex/config.toml`

```toml
[agents]
max_threads = 6
max_depth = 1
job_max_runtime_seconds = 7200
```

### `.codex/agents/dspark-archaeologist.toml`

```toml
name = "dspark_archaeologist"
description = "Read-only DSpark/DeepSpec researcher for architecture, checkpoint, tensor map, and fixture requirements."
model_reasoning_effort = "high"
sandbox_mode = "read-only"
developer_instructions = """
Stay in research mode. Map DeepSpec DSpark Gemma implementation details, target layer taps, checkpoint tensor names, Markov/confidence heads, and fixture requirements. Return concise evidence with paths and symbols. Do not edit Helios code. Do not suggest GGUF, llama.cpp, or external inference runtimes.
"""
nickname_candidates = ["Kepler", "Noether", "Curie"]
```

### `.codex/agents/helios-native-mapper.toml`

```toml
name = "helios_native_mapper"
description = "Read-only native graph explorer for Helios Gemma/MLX hidden taps, KV cache, verifier, and memory behavior."
model_reasoning_effort = "high"
sandbox_mode = "read-only"
developer_instructions = """
Trace Helios native C++/MLX execution paths. Identify where selected hidden taps [5, 17, 29, 41, 46] can be captured with minimal copies and safe cache-owned lifetimes. Map verify_tokens semantics, rollback, terminal no-lookahead, and memory guard behavior. Do not edit code unless the parent agent later asks for a targeted patch.
"""
nickname_candidates = ["Atlas", "Vector", "Faraday"]
```

### `.codex/agents/ffi-engine-mapper.toml`

```toml
name = "ffi_engine_mapper"
description = "Read-only mapper for Helios C ABI, Rust FFI, engine traits, metrics, and benchmark wiring."
model_reasoning_effort = "high"
sandbox_mode = "read-only"
developer_instructions = """
Map the narrowest DSpark integration across native headers, references/ffi mirrors, crates/gemma4d-ffi, crates/gemma4d-engine, server/config surfaces, and benchmark harnesses. Preserve existing MTP assistant behavior and default-off safety. Return exact files, structs, functions, and tests to update.
"""
nickname_candidates = ["Bridge", "Turing", "Ada"]
```

### `.codex/agents/benchmark-optimizer.toml`

```toml
name = "benchmark_optimizer"
description = "Benchmark-focused agent for XR/MTP harness reuse, exact command templates, output schema, and speed/memory interpretation."
model_reasoning_effort = "medium"
sandbox_mode = "workspace-write"
developer_instructions = """
Own benchmark harness work and evidence quality. Reuse existing real-context workloads and MTP/XR harnesses where possible. Ensure records include exactness, acceptance, latency stages, memory, active KV bytes, hidden tap bytes, and command/env metadata. Do not change runtime behavior except benchmark/example code unless the parent agent asks.
"""
nickname_candidates = ["Gauge", "Laplace", "Fermi"]
```

### `.codex/agents/correctness-reviewer.toml`

```toml
name = "correctness_reviewer"
description = "Read-only reviewer for exactness, rollback, KV lifetime, default-off behavior, and missing tests."
model_reasoning_effort = "high"
sandbox_mode = "read-only"
developer_instructions = """
Review like a runtime owner. Prioritize exact greedy parity, cache correctness, rejected token handling, target hidden tap shape/dtype/lifetime, adapter/KV admission boundaries, and default-off safety. Lead with concrete findings and reproduction steps. Avoid style-only comments.
"""
nickname_candidates = ["Verifier", "Gauss", "Sentinel"]
```

## Parent agent orchestration prompt

After adding this goal file, a useful first Codex prompt is:

```text
/goal Implement XR60 from codex/goals/XR60-dspark-native-mlx.goal.md. Use the goal file as the source of truth. Do not add GGUF, llama.cpp, subprocess inference runtimes, or external speculative runtimes. Start by spawning dspark_archaeologist, helios_native_mapper, ffi_engine_mapper, benchmark_optimizer, and correctness_reviewer to map the implementation and risk surface. Wait for all subagents, synthesize a plan, then implement the smallest path to G1-G4. Continue through benchmarks and optimization until the completion rule is satisfied or a blocker is documented.
```

If custom agents are not configured, use built-in agents explicitly:

```text
/goal Implement XR60 from codex/goals/XR60-dspark-native-mlx.goal.md. Spawn explorer agents for DSpark, native graph, FFI/engine, and benchmarks, plus a read-only reviewer for correctness risks. Then use worker agents only for targeted implementation tasks. Keep the feature default-off and verify exactness before speed work.
```

## Final report template

Write `benchmarks/out/XR60-dspark-native-mlx/report.md` with this structure:

```markdown
# XR60 DSpark native MLX report

## Decision
promote_experimental | keep_experimental | reject_for_now | blocked

## Git and environment
- Git SHA:
- Branch:
- Date:
- Machine:
- macOS:
- Rust:
- MLX / mlx_lm:
- Model path:
- Draft path:

## What changed

## Exact commands run

## Correctness results

## Benchmark summary

| workload | context | scheduler | block/max | exact | decode tok/s | speedup | acceptance | accepted/verify | draft ms | verify ms | peak GB | active KV bytes |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Hidden tap parity

## MLX parity

## Confidence calibration, if attempted

## Kernel A/B results, if attempted

## Memory and tiny16 notes

## Regressions or rejected variants

## Blockers

## Next recommended goal
```

## Blocker template

Append to `blockers.md` whenever blocked:

```markdown
## Blocker: <short name>

- Time:
- Git SHA:
- Phase/Gate:
- Command:
- Expected:
- Observed:
- Relevant logs:
- Files inspected:
- Attempted fixes:
- Why not safe to continue guessing:
- Next input needed:
```

## Safety and repo hygiene

- Do not commit secrets, tokens, model weights, large checkpoints, or private source material.
- Keep generated benchmark outputs under ignored benchmark output directories unless repo policy says otherwise.
- Keep default behavior unchanged unless a gate explicitly authorizes a default-off experimental flag.
- Prefer additive APIs and metrics to breaking changes.
- Mirror C ABI header changes into any reference headers and Rust raw bindings.
- Maintain `make verify` as the final broad check when environment allows.
- If Metal/MLX is unavailable in the sandbox, record it and separate compile/test work from local hardware benchmark work.
