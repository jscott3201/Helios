# Current state review for the XR optimization phase

Date: 2026-07-05

This review reflects the current `main` branch, `BENCHMARKS.md`, and the
post-XR79 native/MTP evidence. `BENCHMARKS.md` remains the authority for exact
commands, run IDs, artifacts, and caveats.

## Decision

XR79 confirms the next high-value work is MTP protected-aggregate gap closure
or productizing the accepted scoped chat/tool MTP opt-in, depending on whether
the priority is broad theoretical max or shippable scoped value.

XR72 closed the immediate native diagnostic question: the remaining
full-attention tail is dominated by MLX full-attention eval, especially chat
first-token outliers, not host collection, capacity growth, or final sync. XR75
then falsified the simplest follow-up: serializing deferred full-attention KV
eval by stable layer group did not beat the current runtime default on the
3-trial chat follow-up. XR76 separated non-profile runtime behavior from
profile-mode scheduling and showed profiling is not the main source of the
tail, while a harness-only same-shape warmup probe sharply reduced the chat
first-token/raw p99 tail. XR77 then made that warmup evidence cost-accounted:
the first-token/raw p99 tail improvement repeated, but the discarded warmup
cost was roughly `3.2 s` at p50, which rules out naive per-request warmup.
XR78 then showed the warm state survives repeated same-loaded-target,
fresh-cache requests for the chat tail lane, while the 4K code workload did not
reproduce the tail and gained no meaningful warmup benefit. XR73 accepted the
repeatedly strong chat/tool MTP lane as an explicit scoped opt-in, while
rejecting broad default-on because the protected aggregate remained below the
release gate. XR79 reran that protected aggregate with the XR78 native warmup
claim boundaries attached: scoped gates still pass, but protected speed is only
`+19.482%`, so broad default-on is still parked behind the `25%` protected
aggregate gate. XR74 closed the readiness sweep for the current local
persistent-native default surface.

Recommended order:

1. Productize the accepted scoped chat/tool MTP opt-in if the priority is
   shippable value.
2. If the priority is broad theoretical max, attack the MTP protected aggregate
   gap directly by reducing draft/verify overhead while preserving exactness,
   oracle, holdout, memory, and default-overhead gates.
3. Keep native warmup as out-of-request/load-time shape work unless a later
   server policy proves amortization without user-visible cost.
4. Keep DSpark parked until native and MTP gates are cleaner.

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
- XR76 added first-token latency aggregates and two harness variants:
  `native_decode_runtime_default_profiled` and
  `native_decode_runtime_default_warmup_probe`. The profile-perturbation run
  wrote `9/9` correct records and showed profile mode is not the main tail
  source: first-token p50 moved `216.004 -> 218.580 ms`, and p95/max moved
  `222.905 -> 229.669 ms` versus non-profile runtime default. The warmup probe
  wrote `9/9` correct records and accepted as hypothesis evidence only:
  raw p50 regressed `69.778 -> 70.460 ms` (`+0.977%`), raw p99/max improved
  `177.571 -> 86.680 ms` (`+51.186%`), and first-token p50 improved
  `177.571 -> 86.680 ms`. The warmup probe excludes discarded warmup cost from
  the measured record, so it does not by itself justify a default or production
  speed claim.
- XR77 added cost accounting for discarded same-context warmup work and a
  default-off `native_decode_runtime_default_warmup_costed` variant. The
  focused chat run wrote `9/9` correct records with no blockers, peak MLX
  `7.321 GB`, and `0/567` profile samples. Costed warmup versus runtime default
  repeated the tail win: raw p50 regressed only `69.499 -> 69.643 ms`
  (`+0.208%`), raw p99/max improved `188.836 -> 92.922 ms` (`+50.792%`), and
  first-token p50 improved `188.836 -> 92.922 ms`. The discarded warmup cost
  was large: total p50 `3203.529 ms`, with prefill p50 `2737.664 ms` and decode
  p50 `360.509 ms`, so the viable direction is out-of-request/load-time or
  amortized exact-shape warmup, not per-request warmup.
- XR78 added a default-off
  `native_decode_runtime_default_warmup_amortized_4` variant that warms once
  per loaded target/workload and then measures four repeated fresh-cache
  requests on the same loaded target. The gate-valid 1K/4K run wrote `36/36`
  correct records with no blockers, peak MLX `7.639 GB`, and `0/1116` profile
  samples. On `chat_short_1k_001`, warmup-amortized versus runtime default
  passed the XR06 tail gate: raw p50 moved `70.524 -> 70.300 ms`, raw p99/max
  improved `387.059 -> 92.292 ms` (`+76.155%`), and first-token p50 improved
  `387.059 -> 92.292 ms`. Warmup event p50 was still large at `3843.020 ms`,
  amortized over four requests to `960.755 ms`, so it remains out-of-request
  or load-time shape work. On `code_review_rust_4k_001`, the baseline did not
  reproduce a tail and warmup was not selected.
- Post-XR70 MTP kept exactness and oracle checks, but protected aggregate speedup
  was `+19.845%`, below the `25%` broad default-on gate. Selected chat/tool lanes
  remain attractive at about `+30.784%`.
- XR73 accepted explicit scoped chat/tool MTP opt-in only: candidate `12/12`
  exact, oracle `9/9`, default-disabled overhead `12/12` exact with zero MTP
  side effects, and 4K holdouts `12/12` exact with zero attempted drafts.
  Protected aggregate improved `7523.808 -> 6076.627 ms` (`+19.235%`), below
  the `25%` broad default-on gate; selected chat/tool lanes alone improved
  `+28.820%`.
- XR79 reran the protected MTP aggregate with XR78 native warmup context and
  preserved the same safety posture: candidate `12/12` exact, oracle `9/9`,
  default-disabled overhead `12/12`, 4K holdout `12/12`, default overhead
  `-0.000%`, peak MLX `8.008 GB`, and all scoped gates passed. Protected
  aggregate improved `7479.958 -> 6022.716 ms` (`+19.482%`), still below the
  `25%` broad default-on gate; selected chat/tool lanes alone improved
  `+29.237%`.
- XR74 added backend/native-prefill policy visibility to `/health` and the TUI
  dashboard, with tests covering `backend`, `max_context_tokens`, and native
  prefill policy state. Static gates passed: `cargo fmt --all --check`,
  `git diff --check`, `cargo test -p gemma4d-server --lib`,
  `cargo test -p gemma4d-tui --all-targets`, and
  `GEMMA4D_REQUIRE_MLX=1 cargo test -p gemma4d-bench --example xr06_native_decode_tail_latency_ab --no-run`.

## System map

| Area | Files / symbols | Responsibility | Notes |
|---|---|---|---|
| Native decode benchmark | `crates/gemma4d-bench/examples/xr06_native_decode_tail_latency_ab.rs` | Runs XR06-style real-context decode A/B matrix, variants, profile reports, correctness and tail gates | Existing variants include runtime default, profile-perturbation, warmup-probe, warmup-costed, amortized warmup, full-attention KV update capacity, and XR75 group-eval candidates |
| Native profile ABI | `native/gemma4_mlx/include/gemma4_mlx.h`, `crates/gemma4d-ffi/src/lib.rs` | Carries per-token decode profile fields across C ABI and Rust | Current fields split broad forward, deferred KV eval, full-attention/sliding eval, update/capacity/slice/visible-slice, group eval attribution, and eval sync |
| Full-attention deferred eval | `native/gemma4_mlx/src/native_model.cc::eval_deferred_decode_kv` | Collects full-attention and sliding KV arrays, then calls `mlx::core::eval` | XR72 attributed tails to full-attention eval; XR75 rejected simple serial group scheduling |
| Full-attention update candidate | `native/gemma4_mlx/src/native_model.cc::decode_layer`, capacity helpers | Maintains default-off slice-update-backed full-attention active KV storage | XR71 says this overhead is small and not the main blocker |
| Runtime sync point | `native/gemma4_mlx/src/native_model.cc::decode_one` | Runs logits, greedy selection, and final `mlx::core::eval({greedy, max_logit})` | XR72 showed chat outliers are not primarily final eval sync tails |
| MTP policy harness | `crates/gemma4d-bench/examples/xr15_mtp_policy_variance_ab.rs`, `scripts/xr61_adaptive_n_report.py`, `scripts/xr73_scoped_mtp_report.py` | Measures MTP exactness, acceptance, holdouts, oracle, default-overhead, aggregate gates, and optional native warmup context | XR79 accepts scoped chat/tool evidence again; broad default remains unsupported |
| Server default/readiness | `crates/gemma4d-server/src/lib.rs`, `crates/gemma4d-server/src/http.rs`, `crates/gemma4d-bench/examples/xr11_persistent_native_server_ab.rs` | Selects persistent-native for `serve --model-path`, applies long-context native prefill default, exposes admission/default state, and runs sentinels | XR74 ready for local persistent-native default surface; explicit rollback flags remain available |
| Operator visibility | `crates/gemma4d-tui/src/{app,provider,ui}.rs`, `crates/gemma4d-tui/tests/m05_acceptance.rs` | Surfaces backend, context, native prefill policy, MTP, cache, adapter, and live metrics state through provider-backed pages | XR74 added dashboard backend/native-prefill visibility |

## Findings

### high: XR79 leaves broad MTP default-on gated by protected speed

Evidence: XR72 accepted runtime default against explicit per-layer on all five
rows, and XR73 accepted scoped chat/tool MTP opt-in with exactness, oracle,
holdout, memory, and default-overhead gates. XR74 closed the readiness sweep for
the current local persistent-native default surface. XR75 rejected the simple
serial group-eval candidate against runtime default on the 3-trial chat
follow-up. XR76 showed profile-mode perturbation is not the main cause and a
same-shape warmup probe cut chat first-token p50 `177.571 -> 86.680 ms` while
raw p99 improved `51.186%`. XR77 repeated the win with cost accounting:
first-token p50 moved `188.836 -> 92.922 ms` and raw p99 improved `50.792%`,
but discarded warmup total p50 was `3203.529 ms`. XR78 showed repeated
same-loaded-target fresh-cache chat requests keep that first-token tail benefit:
raw p99 improved `76.155%`, first-token p50 moved `387.059 -> 92.292 ms`, and
the 4K code workload did not reproduce a tail. XR79 then reran the protected
MTP aggregate with those native warmup boundaries attached: scoped gates passed,
default overhead was clean, oracle/holdouts passed, selected chat/tool lanes
improved `+29.237%`, but protected aggregate speed was only `+19.482%`.

Impact: The fastest remaining route toward the theoretical max is no longer
another readiness/doc pass or a serial group-eval full matrix. The remaining
native speed evidence points at first-token warm/JIT/cache behavior, but the
cost model rules out naive request-path warmup. The next broad-default blocker
is now MTP protected aggregate speed, not native-tail evidence.

Recommendation: Keep native warmup as default-off out-of-request/load-time
shape work. For theoretical max, directly reduce MTP draft/verify overhead while
preserving exactness, oracle, holdout, memory, and default-overhead gates.
Treat profile mode and serial group eval as rejected promotion lanes unless new
evidence changes their cost model.

### high: Broad MTP default-on is still unsupported

Evidence: XR66 selected chat/tool lanes were `+31.033%`, XR70 selected lanes
were `+30.784%`, XR73 selected lanes were `+28.820%`, and XR79 selected lanes
were `+29.237%`; however, protected aggregate speed stayed below the `25%`
broad gate, with XR79 at `+19.482%`.

Impact: MTP is useful, but only for scoped workloads. Turning it on broadly
would promote a narrower result than the evidence supports.

Recommendation: Keep MTP explicit/scoped/default-off. Productize the scoped
chat/tool opt-in if near-term value matters; broader promotion should wait for
protected aggregate evidence above the release gate.

### medium: Native default-readiness is complete for the local default surface

Evidence: Server default sentinels passed, runtime default decode is accepted
against explicit per-layer, XR70/XR71 candidates remain default-off, and XR74
added health/dashboard visibility for backend and native prefill policy state.

Impact: The current local persistent-native default can be treated as ready
within the documented scope. This is not production internet-facing serving
readiness and does not promote MTP or default-off experimental native candidates.

Recommendation: Keep rollback flags and accepted/default-off boundaries
explicit in docs while moving speed work to scoped MTP productization or the
MTP protected aggregate gap.

### info: CI workflow removal is already true in this checkout

Evidence: the current `main` tree has no tracked `.github` directory or
workflow YAML. The only remaining CI mention found in the repo is historical
M00 evidence noting that a workflow skeleton existed outside the local
acceptance gate.

Impact: There is no CI workflow job to delete from this branch.

Recommendation: Leave historical evidence alone unless the project wants to
rewrite old milestone reports.

## Next work items

### Scoped MTP opt-in productization

The accepted XR73/XR79 chat/tool lane is repeatedly exact, oracle-clean,
holdout-protected, default-overhead-clean, and roughly `+29%` on selected lanes.
The next shippable-value task is to expose it as an explicit local opt-in with
clear workload/request gating, observability, and rollback, while keeping broad
MTP default-off.

### MTP protected aggregate gap

For broad theoretical max, the next research task is to reduce draft/verify
overhead enough that the protected aggregate clears `25%` without weakening the
`mtp_candidate_1k_001` and 4K holdout protections. Selected-lane evidence alone
is not sufficient for broad default-on.

## Gaps and unknowns

- XR78 proves warm-state lifetime only within the same loaded target and
  same-shape fresh-cache benchmark path; a production server warmup policy
  would still need separate admission, scheduling, and observability work.
- XR79 proves current scoped MTP gates, but it does not identify which specific
  draft/verify cost component must move to clear the protected aggregate gate.
