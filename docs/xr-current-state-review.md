# Current state review for the XR optimization phase

Date: 2026-07-05

This review reflects the current `main` branch, `BENCHMARKS.md`, and the
post-XR72 native graph evidence. `BENCHMARKS.md` remains the authority for exact
commands, run IDs, artifacts, and caveats.

## Decision

The next high-value goal is XR73: scoped MTP chat/tool opt-in.

XR72 closed the immediate native diagnostic question: the remaining
full-attention tail is dominated by MLX full-attention group eval, especially
chat first-token outliers, not host collection, capacity growth, or final sync.
The broad native runtime default is already accepted against explicit
per-layer. The best near-term path to the theoretical max is therefore to
productize the repeatedly strong chat/tool MTP lane behind an explicit opt-in
or workload gate, then run a native default-readiness sweep.

Recommended order:

1. XR73: add scoped MTP chat/tool opt-in or workload-gated behavior.
2. XR74: run a native default-readiness sweep.
3. Native full-attention group-eval follow-up if kernel/JIT/scheduling work is
   still needed after the readiness pass.
4. Keep DSpark parked until native readiness and scoped MTP are cleaner.

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
- XR72 added profile-only full-attention group attribution and accepted the
  runtime default against explicit per-layer on all five rows. The full matrix
  wrote `45/45` correct records, `2835/2835` profiled samples, no blockers, and
  peak MLX `7.321..7.929 GB`. Collection time was `0.006..0.009 ms` mean on
  runtime default. Chat first-token outliers were dominated by full-attention
  group eval: runtime-default first-token host latency was
  `406.584..511.937 ms`, full-attention eval was `397.454..503.863 ms`,
  collection was `0.010..0.308 ms`, and final eval sync was about
  `6.337..7.000 ms`.
- Post-XR70 MTP kept exactness and oracle checks, but protected aggregate speedup
  was `+19.845%`, below the `25%` broad default-on gate. Selected chat/tool lanes
  remain attractive at about `+30.784%`.

## System map

| Area | Files / symbols | Responsibility | Notes |
|---|---|---|---|
| Native decode benchmark | `crates/gemma4d-bench/examples/xr06_native_decode_tail_latency_ab.rs` | Runs XR06-style real-context decode A/B matrix, variants, profile reports, correctness and tail gates | Existing variants include runtime default and full-attention KV update capacity candidates |
| Native profile ABI | `native/gemma4_mlx/include/gemma4_mlx.h`, `crates/gemma4d-ffi/src/lib.rs` | Carries per-token decode profile fields across C ABI and Rust | Current fields split broad forward, deferred KV eval, full-attention/sliding eval, update/capacity/slice/visible-slice, group eval attribution, and eval sync |
| Full-attention deferred eval | `native/gemma4_mlx/src/native_model.cc::eval_deferred_decode_kv` | Collects full-attention and sliding KV arrays, then calls `mlx::core::eval` | XR72 attributed tails to full-attention group eval; future kernel/JIT/scheduling work should stay scoped |
| Full-attention update candidate | `native/gemma4_mlx/src/native_model.cc::decode_layer`, capacity helpers | Maintains default-off slice-update-backed full-attention active KV storage | XR71 says this overhead is small and not the main blocker |
| Runtime sync point | `native/gemma4_mlx/src/native_model.cc::decode_one` | Runs logits, greedy selection, and final `mlx::core::eval({greedy, max_logit})` | XR72 showed chat outliers are not primarily final eval sync tails |
| MTP policy harness | `crates/gemma4d-bench/examples/xr15_mtp_policy_variance_ab.rs`, `scripts/xr61_adaptive_n_report.py` | Measures MTP exactness, acceptance, holdouts, oracle, and aggregate gates | Use next for XR73 scoped opt-in/default-overhead evidence |

## Findings

### high: XR73 is the next speed/value lane

Evidence: XR72 accepted runtime default against explicit per-layer on all five
rows and showed the remaining chat tail is in full-attention group eval, not
collection, capacity growth, visible-slice work, or final sync. The
`native_decode_full_attention_kv_update_256` candidate did not beat runtime
default.

Impact: More capacity tuning is low value. A deeper native kernel/JIT/scheduler
change may still be useful, but it is a narrower research lane than converting
the already-proven MTP chat/tool speedup into a controlled opt-in.

Recommendation: Do XR73 next. Make MTP explicit and scoped, prove default-path
overhead is zero or indistinguishable, and keep broad default-on off unless the
protected aggregate clears the release gate.

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

Evidence: Server default sentinels passed, runtime default decode is accepted
against explicit per-layer, and XR70/XR71 candidates remain default-off.
Operator
observability, rollback flags, admission/tokenizer guardrails, and benchmark
ledger cleanup are readiness work, not kernel work.

Impact: A faster native path can still be unsafe to broaden if guardrails and
rollback surfaces are incomplete.

Recommendation: Keep XR74 after XR73 unless the team wants to freeze MTP for a
release. Treat XR74 as a readiness sweep rather than an optimization patch.

### info: CI workflow removal is already true in this checkout

Evidence: the current `main` tree has no tracked `.github` directory or
workflow YAML. The only remaining CI mention found in the repo is historical
M00 evidence noting that a workflow skeleton existed outside the local
acceptance gate.

Impact: There is no CI workflow job to delete from this branch.

Recommendation: Leave historical evidence alone unless the project wants to
rewrite old milestone reports.

## Next work items

### XR73: scoped MTP chat/tool opt-in

Use the existing XR66/XR70 evidence to ship a narrow opt-in path. Preserve
exactness, sequential oracle, holdout, memory, and no-default-overhead gates.
Do not chase broad default-on unless the protected aggregate clears `25%`.

### XR74: native default-readiness sweep

After XR73, audit server/default wiring, admission and tokenizer guardrails,
tiny16 8K/16K/24K sentinels, operator observability, rollback flags, MTP/native
experimental-state reporting, and benchmark ledger cleanup. The output should
be a readiness decision, not just a speed table.

### Native full-attention group-eval follow-up

XR72 isolated chat first-token tails to the full-attention group eval path. If
more native optimization is needed after XR73/XR74, scope it to MLX group eval
scheduling, warm/JIT/cache behavior, or a lower-level full-attention materialized
KV path. Do not spend more time on capacity growth or visible-slice overhead
without new evidence.

## Gaps and unknowns

- MTP selected-lane value is clear, but the correct operator/server opt-in
  surface still needs a product decision during XR73.
- XR74 still needs exact sentinel commands and the final readiness decision.
- Native full-attention group-eval kernel work remains unscoped beyond XR72's
  diagnostic profile artifacts.
