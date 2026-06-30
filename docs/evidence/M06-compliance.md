# M06 Compliance Matrix

## Scope

- Milestone: `milestones/M06-mtp-speculative-decoding.md`
- Goal: `codex/goals/M06-mtp-speculative-decoding.goal.md`
- Spec: `spec/05-speculative-decoding-mtp.md`
- References: `references/acceptance-checklists.md`, `docs/decisions/0002-m06-mtp-fail-closed-until-native-assistant-execution.md`

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M06-T01 | Add drafter load FFI function. | `native/gemma4_mlx/include/gemma4_mlx.h`; `native/gemma4_mlx/src/runtime.cc`; `crates/gemma4d-ffi/src/lib.rs`; `cargo test -p gemma4d-ffi`. | Complete | None. |
| M06-T02 | Expose last target hidden state/shared views needed by drafter. | `Gemma4StepResult.native_last_hidden`; Rust `NativeLastHiddenView`; `NativeHiddenState`; final full/sliding shared KV capture; gated native graph test. | Complete for opt-in native graph | Helper-backed path does not expose native views. |
| M06-T03 | Implement draft block size 1, then 2. | `MtpConfig::draft_block_size`; `speculative_greedy`; engine tests; fixture report; `NativeMtpAssistantModel::draft_block`; gated native graph test with real assistant artifact. | Complete for engine fixtures; partial for native graph | Native assistant drafts run for block size 1/2, but real-model acceptance quality is not yet proven. |
| M06-T04 | Implement verify/accept/rollback. | `speculative_greedy`; `MtpMetrics.rollback_count`; `mtp_block_size_2_matches_non_mtp_with_rollback`; fixture `block_size_2_rollback_exact`. | Complete for engine fixtures | Native C ABI verify advances over supplied draft tokens and still lacks exact accept/rollback semantics. |
| M06-T05 | Add MTP exactness tests against non-MTP greedy. | `cargo test -p gemma4d-engine --all-targets`; `benchmarks/out/M06/mtp-fixture-report.json`. | Complete | Fixture/scripted target scope only. |
| M06-T06 | Add MTP metrics and auto-disable. | `MtpMetrics`; auto-disable tests; fixture cases for low acceptance, adapter, and compressed active KV. | Complete | None for M06 fixture scope. |
| M06-T07 | Update TUI MTP placeholder/provider payload. | `MtpSnapshot`; `MockProvider::mtp_snapshot`; `render_mtp`; TUI test and snapshots. | Complete | File provider remains offline. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| MTP block size 1 exactness passes. | `block_size_1_exact` fixture: baseline `[236772,236772,236772,236772]` equals MTP output; acceptance rate 1.00. | Complete for fixture set. |
| MTP block size 2 exactness passes on fixture set or auto-disables with evidence. | `block_size_2_rollback_exact` passes with rollback count 1; `block_size_2_auto_disable` falls back exactly after low acceptance. | Complete for fixture set. |
| Acceptance metrics are recorded. | Raw JSON records attempted/accepted draft tokens, acceptance rate, accepted tokens per verify, verify passes, decode tokens/sec, peak memory, rollback count, and auto-disable reason. | Complete. |
| Adapters remain disabled for MTP in this milestone. | `adapter_active_auto_disable` records attempted 0, accepted 0, auto-disabled true. | Complete. |
| Compressed active KV remains disabled for MTP in this milestone. | `compressed_active_kv_auto_disable` records attempted 0, accepted 0, auto-disabled true. | Complete. |

## Spec Compliance Summary

- Compliant: greedy-only fixture loop, block size 1/2 exactness, rollback behavior, acceptance metrics, adapter disable, compressed active KV disable, and TUI operator visibility.
- Partial: native FFI interface, strict assistant artifact loading, native target hidden/shared-view materialization, and native assistant block drafting are present. Full real-model speculative decoding remains partial because native accept/rollback is not implemented.
- Follow-up: implement native exact verify/accept/rollback without committing rejected draft tokens, then rerun exactness against real Gemma 4 target and assistant artifacts.

## Risk

The main behavioral risk is treating fixture exactness or native draft-token production as proof of real-model MTP. The code avoids that by keeping exactness claims scoped to the engine fixture gates and by documenting the remaining native accept/rollback gap in the decision record.
