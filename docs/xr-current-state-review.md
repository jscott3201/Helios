# Current state review for the XR optimization phase

Date: 2026-07-05

This review reflects the current `main` branch, `BENCHMARKS.md`, and the
post-XR75 native/MTP evidence. `BENCHMARKS.md` remains the authority for exact
commands, run IDs, artifacts, and caveats.

## Decision

The next high-value goal is native non-profile first-token/full-attention tail
isolation.

XR72 closed the immediate native diagnostic question: the remaining
full-attention tail is dominated by MLX full-attention eval, especially chat
first-token outliers, not host collection, capacity growth, or final sync. XR75
then falsified the simplest follow-up: serializing deferred full-attention KV
eval by stable layer group did not beat the current runtime default on the
3-trial chat follow-up. XR73 accepted the repeatedly strong chat/tool MTP lane
as an explicit scoped opt-in, while rejecting broad default-on because the
protected aggregate remains below the release gate. XR74 closed the readiness
sweep for the current local persistent-native default surface. The next useful
step is therefore not another serial group-eval pass, but a narrower native
measurement/candidate pass that separates true non-profile runtime behavior
from profile-mode scheduling and tests warm/JIT/cache behavior around the
first-token full-attention tail.

Recommended order:

1. Native non-profile first-token/full-attention tail isolation, with any
   candidate kept default-off until it beats runtime default without profiling
   perturbation.
2. Keep broader MTP promotion parked until protected aggregate speed clears the
   release gate.
3. Keep DSpark parked until native and MTP gates are cleaner.

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
- XR75 added default-off
  `GEMMA4D_EXPERIMENTAL_NATIVE_FULL_ATTENTION_GROUP_EVAL=1` and rejected serial
  full-attention group scheduling as a promotion lane. The decisive
  `chat_short_1k_001` follow-up wrote `9/9` correct records and `567/567`
  profiled samples with peak `7.321 GB`, but the candidate regressed p50
  `69.655 -> 71.448 ms` (`+2.574%`), regressed p95
  `70.191 -> 72.057 ms`, and improved p99 only
  `163.705 -> 162.719 ms` (`+0.602%`) versus runtime default.
- Post-XR70 MTP kept exactness and oracle checks, but protected aggregate speedup
  was `+19.845%`, below the `25%` broad default-on gate. Selected chat/tool lanes
  remain attractive at about `+30.784%`.
- XR73 accepted explicit scoped chat/tool MTP opt-in only: candidate `12/12`
  exact, oracle `9/9`, default-disabled overhead `12/12` exact with zero MTP
  side effects, and 4K holdouts `12/12` exact with zero attempted drafts.
  Protected aggregate improved `7523.808 -> 6076.627 ms` (`+19.235%`), below
  the `25%` broad default-on gate; selected chat/tool lanes alone improved
  `+28.820%`.
- XR74 added backend/native-prefill policy visibility to `/health` and the TUI
  dashboard, with tests covering `backend`, `max_context_tokens`, and native
  prefill policy state. Static gates passed: `cargo fmt --all --check`,
  `git diff --check`, `cargo test -p gemma4d-server --lib`,
  `cargo test -p gemma4d-tui --all-targets`, and
  `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.

## System map

| Area | Files / symbols | Responsibility | Notes |
|---|---|---|---|
| Native decode benchmark | `crates/gemma4d-bench/examples/xr06_native_decode_tail_latency_ab.rs` | Runs XR06-style real-context decode A/B matrix, variants, profile reports, correctness and tail gates | Existing variants include runtime default, full-attention KV update capacity, and XR75 group-eval candidates |
| Native profile ABI | `native/gemma4_mlx/include/gemma4_mlx.h`, `crates/gemma4d-ffi/src/lib.rs` | Carries per-token decode profile fields across C ABI and Rust | Current fields split broad forward, deferred KV eval, full-attention/sliding eval, update/capacity/slice/visible-slice, group eval attribution, and eval sync |
| Full-attention deferred eval | `native/gemma4_mlx/src/native_model.cc::eval_deferred_decode_kv` | Collects full-attention and sliding KV arrays, then calls `mlx::core::eval` | XR72 attributed tails to full-attention eval; XR75 rejected simple serial group scheduling |
| Full-attention update candidate | `native/gemma4_mlx/src/native_model.cc::decode_layer`, capacity helpers | Maintains default-off slice-update-backed full-attention active KV storage | XR71 says this overhead is small and not the main blocker |
| Runtime sync point | `native/gemma4_mlx/src/native_model.cc::decode_one` | Runs logits, greedy selection, and final `mlx::core::eval({greedy, max_logit})` | XR72 showed chat outliers are not primarily final eval sync tails |
| MTP policy harness | `crates/gemma4d-bench/examples/xr15_mtp_policy_variance_ab.rs`, `scripts/xr61_adaptive_n_report.py`, `scripts/xr73_scoped_mtp_report.py` | Measures MTP exactness, acceptance, holdouts, oracle, default-overhead, and aggregate gates | XR73 accepts only explicit scoped chat/tool opt-in; broad default remains unsupported |
| Server default/readiness | `crates/gemma4d-server/src/lib.rs`, `crates/gemma4d-server/src/http.rs`, `crates/gemma4d-bench/examples/xr11_persistent_native_server_ab.rs` | Selects persistent-native for `serve --model-path`, applies long-context native prefill default, exposes admission/default state, and runs sentinels | XR74 ready for local persistent-native default surface; explicit rollback flags remain available |
| Operator visibility | `crates/gemma4d-tui/src/{app,provider,ui}.rs`, `crates/gemma4d-tui/tests/m05_acceptance.rs` | Surfaces backend, context, native prefill policy, MTP, cache, adapter, and live metrics state through provider-backed pages | XR74 added dashboard backend/native-prefill visibility |

## Findings

### high: Non-profile first-token tail isolation is the next value lane

Evidence: XR72 accepted runtime default against explicit per-layer on all five
rows, and XR73 accepted scoped chat/tool MTP opt-in with exactness, oracle,
holdout, memory, and default-overhead gates. XR74 closed the readiness sweep for
the current local persistent-native default surface. XR75 rejected the simple
serial group-eval candidate against runtime default on the 3-trial chat
follow-up. Broad MTP default-on remains below the protected aggregate gate, and
XR70/XR71 full-attention update candidates remain default-off.

Impact: The fastest remaining route toward the theoretical max is no longer
another readiness/doc pass or a serial group-eval full matrix. The remaining
native speed evidence points at first-token/full-attention runtime behavior that
profile-mode scheduling may perturb.

Recommendation: Scope the next native patch around no-profile timing evidence,
warm/JIT/cache behavior, or lower-level full-attention materialization. Treat
serial group eval as rejected unless a new kernel-level implementation changes
the cost model.

### high: Broad MTP default-on is still unsupported

Evidence: XR66 selected chat/tool lanes were `+31.033%`, XR70 selected lanes
were `+30.784%`, and XR73 selected lanes were `+28.820%`; however, protected
aggregate speed stayed below the `25%` broad gate, with XR73 at `+19.235%`.

Impact: MTP is useful, but only for scoped workloads. Turning it on broadly
would promote a narrower result than the evidence supports.

Recommendation: Keep MTP explicit/scoped/default-off. Broader promotion should
wait for protected aggregate evidence above the release gate.

### medium: Native default-readiness is complete for the local default surface

Evidence: Server default sentinels passed, runtime default decode is accepted
against explicit per-layer, XR70/XR71 candidates remain default-off, and XR74
added health/dashboard visibility for backend and native prefill policy state.

Impact: The current local persistent-native default can be treated as ready
within the documented scope. This is not production internet-facing serving
readiness and does not promote MTP or default-off experimental native candidates.

Recommendation: Keep rollback flags and accepted/default-off boundaries
explicit in docs while moving speed work back to the native first-token tail
lane.

### info: CI workflow removal is already true in this checkout

Evidence: the current `main` tree has no tracked `.github` directory or
workflow YAML. The only remaining CI mention found in the repo is historical
M00 evidence noting that a workflow skeleton existed outside the local
acceptance gate.

Impact: There is no CI workflow job to delete from this branch.

Recommendation: Leave historical evidence alone unless the project wants to
rewrite old milestone reports.

## Next work items

### Native non-profile first-token/full-attention tail isolation

XR72 isolated chat first-token tails to the full-attention eval path, and XR75
showed serial group scheduling is not enough. The next task should measure the
actual non-profile runtime path and test warm/JIT/cache hypotheses or a
lower-level full-attention materialized KV path. Do not spend more time on
capacity growth, visible-slice overhead, or serial group scheduling without new
evidence.

## Gaps and unknowns

- Native non-profile first-token timing remains less directly attributed than
  profile-mode timing because the current profile path splits deferred full
  attention and sliding eval for attribution.
