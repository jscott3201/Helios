# Current state review for the XR optimization phase

Date: 2026-07-05

This review reflects the current `main` branch, `BENCHMARKS.md`, and the
post-XR71 native graph evidence. `BENCHMARKS.md` remains the authority for exact
commands, run IDs, artifacts, and caveats.

## Decision

The next high-value goal is XR72: full-attention deferred-eval tail jitter.

Native graph work should stay ahead of MTP. XR71 showed that capacity growth,
`slice_update`, and visible-slice overhead are not the remaining bottleneck; the
dominant unresolved lane is the grouped full-attention deferred eval barrier and
its tail behavior. MTP still has strong selected chat/tool lanes, but the broad
protected aggregate remains below the default-on gate.

Recommended order:

1. XR72: isolate full-attention deferred-eval p95/p99 jitter.
2. XR73: add scoped MTP chat/tool opt-in or workload-gated behavior.
3. XR74: run a native default-readiness sweep after XR72.
4. Keep DSpark parked until native tail behavior and scoped MTP are cleaner.

## Evidence summary

- Server native prefill default is already a large accepted win on long
  contexts: 16K moved `87387.199 -> 41711.194 ms` with peak MLX
  `21.874 -> 7.638 GB`; 8K moved `31285.354 -> 22618.497 ms`.
- Server default sentinels passed at 8K, 16K, and 24K. The 24K low-N sentinel
  was neutral, `60939.960 -> 61254.060 ms`, with peak `7.859 GB`.
- XR65 made grouped end-of-decode KV eval the native runtime default for decode;
  chat/tool p50 rows moved from roughly `81 ms` to roughly `70 ms`.
- XR69 showed runtime-default deferred KV eval is dominated by full-attention
  materialization: full-attention eval means were `63.118..77.984 ms`, while
  sliding eval was only `0.006..0.009 ms`.
- XR70 proved a default-off full-attention update candidate is real but uneven:
  total decode improved `78020.982 -> 73634.458 ms` (`+5.622%`), but
  `chat_short_1k_001` regressed and only part of the XR06 tail gate cleared.
- XR71 narrowed the candidate and improved total decode
  `73947.065 -> 68583.341 ms` (`+7.253%`) with 16K peak `7.929 GB`, but
  `chat_short_1k_001` raw p95/p99 regressed by `5.168%`/`26.805%`; only
  `code_review_rust_8k_001` cleared the candidate tail gate.
- XR71 profile fields show full-attention update overhead is about
  `0.010 ms/token`. Capacity growth, slice update, and visible-slice creation
  should not drive the next optimization.
- Post-XR70 MTP kept exactness and oracle checks, but protected aggregate speedup
  was `+19.845%`, below the `25%` broad default-on gate. Selected chat/tool lanes
  remain attractive at about `+30.784%`.

## System map

| Area | Files / symbols | Responsibility | Notes |
|---|---|---|---|
| Native decode benchmark | `crates/gemma4d-bench/examples/xr06_native_decode_tail_latency_ab.rs` | Runs XR06-style real-context decode A/B matrix, variants, profile reports, correctness and tail gates | Existing variants include runtime default and full-attention KV update capacity candidates |
| Native profile ABI | `native/gemma4_mlx/include/gemma4_mlx.h`, `crates/gemma4d-ffi/src/lib.rs` | Carries per-token decode profile fields across C ABI and Rust | Current fields split broad forward, deferred KV eval, full-attention/sliding eval, update/capacity/slice/visible-slice, and eval sync |
| Full-attention deferred eval | `native/gemma4_mlx/src/native_model.cc::eval_deferred_decode_kv` | Collects full-attention and sliding KV arrays, then calls `mlx::core::eval` | XR72 should add finer attribution here before kernel changes |
| Full-attention update candidate | `native/gemma4_mlx/src/native_model.cc::decode_layer`, capacity helpers | Maintains default-off slice-update-backed full-attention active KV storage | XR71 says this overhead is small and not the main blocker |
| Runtime sync point | `native/gemma4_mlx/src/native_model.cc::decode_one` | Runs logits, greedy selection, and final `mlx::core::eval({greedy, max_logit})` | XR72 must distinguish deferred KV eval tails from final eval sync tails |
| MTP policy harness | `crates/gemma4d-bench/examples/xr15_mtp_policy_variance_ab.rs`, `scripts/xr61_adaptive_n_report.py` | Measures MTP exactness, acceptance, holdouts, oracle, and aggregate gates | Use for XR73 after XR72 clarifies native baseline behavior |

## Findings

### high: Full-attention deferred eval is now the limiting native lane

Evidence: XR69 split the deferred barrier and found sliding eval at
`0.006..0.009 ms`, while full-attention eval accounts for almost all
`63..78 ms/token`. XR70 and XR71 both improved aggregate decode but left
`chat_short_1k_001` tail regressions.

Impact: More capacity or visible-slice tuning is unlikely to close the tail
gate. The next change needs attribution around the MLX eval barrier itself:
layer/group contribution, eval scheduling, sync, shape stability, and warm/JIT
effects.

Recommendation: Do XR72 as profiling-first work. Do not promote the XR70/XR71
candidate or add kernels until the p95/p99 source is explained.

### high: Broad MTP default-on is still unsupported

Evidence: XR66 selected chat/tool lanes were `+31.033%`, but the protected
aggregate was only `+20.334%`. XR70 selected lanes stayed strong at `+30.784%`,
but protected aggregate was `+19.845%`, still below the `25%` broad gate.

Impact: MTP is useful, but only for scoped workloads. Turning it on broadly
would promote a narrower result than the evidence supports.

Recommendation: XR73 should implement explicit opt-in or workload-gated
chat/tool behavior, with no default-path overhead and the existing
`mtp_candidate_1k_001` holdout protection.

### medium: Native default-readiness is a separate gate from speed

Evidence: Server default sentinels passed, but XR70/XR71 candidates remain
default-off and the next native work is still a tail investigation. Operator
observability, rollback flags, admission/tokenizer guardrails, and benchmark
ledger cleanup are readiness work, not kernel work.

Impact: A faster native path can still be unsafe to broaden if guardrails and
rollback surfaces are incomplete.

Recommendation: Keep XR74 after XR72, and treat it as a readiness sweep rather
than an optimization patch.

### info: CI workflow removal is already true in this checkout

Evidence: the current `main` tree has no tracked `.github` directory or
workflow YAML. The only remaining CI mention found in the repo is historical
M00 evidence noting that a workflow skeleton existed outside the local
acceptance gate.

Impact: There is no CI workflow job to delete from this branch.

Recommendation: Leave historical evidence alone unless the project wants to
rewrite old milestone reports.

## Next work items

### XR72: full-attention deferred-eval tail jitter

Scope the first patch to profile attribution. Extend the profile surface around
`eval_deferred_decode_kv` so profile artifacts can show whether p95/p99 tails
come from specific full-attention layers/groups, array count/shape churn,
`mlx::core::eval` scheduling, final sync, or warm/JIT/cache effects.

Required matrix:

- `chat_short_1k_001`
- `tool_json_1k_001`
- `code_review_rust_4k_001`
- `code_review_rust_8k_001`
- `benchmark_qa_16k_001`

Required gates:

- token/logit exactness on every row;
- no default runtime/server/API behavior change;
- 16K peak MLX below `14 GB` with `long_context_256`;
- p95/p99 explanation before kernel changes;
- candidate promotion only if no row regresses over `5%` and at least three of
  five rows clear the XR06 tail gate.

### XR73: scoped MTP chat/tool opt-in

Use the existing XR66/XR70 evidence to ship a narrow opt-in path only after the
native baseline is stable enough. Preserve exactness, sequential oracle,
holdout, memory, and no-default-overhead gates. Do not chase broad default-on
unless the protected aggregate clears `25%`.

### XR74: native default-readiness sweep

After XR72, audit server/default wiring, admission and tokenizer guardrails,
tiny16 8K/16K/24K sentinels, operator observability, rollback flags, and
benchmark ledger cleanup. The output should be a readiness decision, not just a
speed table.

## Gaps and unknowns

- XR72 does not yet have committed profile artifacts because the goal contract
  is newly defined.
- Fine-grained full-attention layer/group timing may require a C ABI profile
  extension and Rust report-field updates.
- The existing XR06 harness can run the right workload matrix, but it may need
  new variant/profile labels to make XR72 artifacts self-describing.
- MTP selected-lane value is clear, but the correct operator/server opt-in
  surface still needs a product decision during XR73.
