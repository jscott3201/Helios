# M12 Compliance Matrix

## Scope

- Milestone: `milestones/M12-tiny16-profiling-release.md`
- Goal: `codex/goals/M12-tiny16-profiling-release.goal.md`
- Specs: `spec/09-observability-profiling.md`, `spec/10-correctness-evals-benchmarks.md`, `spec/12-risk-register.md`, `spec/13-tui-operator-ux.md`
- Included: real 4-bit target generation through the current C ABI / MLX-LM helper path, local server/control/TUI release gate, deterministic fixture coverage, memory/admission guard evidence, feature fallback paths, raw artifact retention.
- Excluded: hand-written native graph performance and model-revision-pinned distributable release claims.

## Task Matrix

| ID | Requirement | Evidence | Status | Gap |
|---|---|---|---|---|
| M12-T01 | Run standard benchmark matrix at 1K/4K/8K/16K/32K. | `m12_real_tiny16_matrix`; `real-matrix/records.jsonl`; `real-matrix/report.md`; `docs/evidence/M12-release-report.md`. | Complete | 32K uses one generated token as a memory probe to protect tiny16 headroom. |
| M12-T02 | Run MTP exactness and acceptance benchmarks if enabled. | `mtp-fixture.json`; MTP default disabled in `tiny16.toml`; M06 fixture still passes. | Complete | MTP remains disabled by default for tiny16. |
| M12-T03 | Run RAM and SSD prefix cache warm TTFT tests. | `ram-restore-matrix.json`; `ssd-benchmark.json`. | Complete | Fixture TTFT/restore timing only. |
| M12-T04 | Run one Rust expert adapter load/route/unload test. | `adapter-fixture.json`; `release-gate.json` adapter gate. | Complete | Native adapter tensor math remains future work. |
| M12-T05 | Run TUI-assisted validation of dashboard, chat, cache, adapter, benchmark, and config screens. | TUI release walkthrough report and 18 snapshots under `benchmarks/out/M12/tui-walkthrough/`; HTTP walkthrough under `tui-http-walkthrough/`. | Complete | Mock walkthrough is deterministic; HTTP walkthrough verifies live attach. |
| M12-T06 | Run release-readiness review and risk audit. | `docs/evidence/M12-release-readiness.md`; this compliance matrix. | Complete | Decision is ready with known limitations, not production native serving. |
| M12-T07 | Produce release report and known limitations. | `docs/evidence/M12-release-report.md`; raw `benchmarks/out/M12/release-report.md`. | Complete | Limitations are explicitly non-blocking for this local release-gate slice. |
| M12-T08 | Run TUI-driven release walkthrough covering config validation, benchmark launch, metrics review, adapter status, cache status, and report export. | `tui-release-walkthrough.md` and JSON summaries under `benchmarks/out/M12/tui-walkthrough/` and `tui-http-walkthrough/`. | Complete | TUI records/launches benchmark surfaces; the real long-running matrix is captured separately by `m12_real_tiny16_matrix`. |

## Acceptance Matrix

| Criterion | Evidence | Status |
|---|---|---|
| Release report exists. | `docs/evidence/M12-release-report.md`; raw `benchmarks/out/M12/release-report.md`. | Complete. |
| 16K passes end-to-end. | `real-matrix/records.jsonl` context `16384` generated 128/128 tokens with status `passed`. | Complete. |
| 32K passes or fails gracefully with memory evidence. | `real-matrix/records.jsonl` context `32768` generated 1/1 token with status `passed` and peak native memory `11.888` GB. | Complete. |
| TUI can be disabled or crash without killing/corrupting the server. | Live server health before/after controlled TUI failure plus metrics artifact under `benchmarks/out/M12/`. | Complete. |
| All enabled features have fallback/disable paths. | `release-gate.json` fallback gates; `tiny16.toml` disables MTP/SSD by default; adapter/remote/cache guards. | Complete. |
| No blocker findings remain open. | `release-gate.json` `blocker_findings_open = 0`; release-readiness review blocker table empty. | Complete. |
| TUI release walkthrough report exists under `benchmarks/out/M12/` or `artifacts/tui/`. | `benchmarks/out/M12/tui-walkthrough/tui-release-walkthrough.md`; HTTP variant also exists. | Complete. |

## Coverage Summary

- Implemented and tested: real target matrix through C ABI / MLX-LM helper, release-gate runner, tiny16 config validation, local server context/memory guard matrix, metrics coverage, adapter route/load/unload, MTP/cache/adapter fixture evidence, TUI walkthrough/report/snapshots, TUI failure isolation evidence.
- Implemented but limited: 32K uses a one-token decode memory probe; TUI benchmark launch evidence is control-plane evidence, not the long-running matrix itself.
- Not implemented in M12: hand-written native graph default serving, production HTTP stack, non-localhost security model, revision-pinned model release.
- Ambiguous / owner decision: whether the future native graph follow-up should become a release blocker for a later non-stub release.

## Next Work Items

1. Complete the hand-written native Gemma 4 graph serving follow-up before making native-graph performance claims.
2. Pin target/drafter revisions in `tiny16.toml` before distributable release evidence.
3. Replace or wrap the stdlib localhost HTTP stack before any non-localhost serving mode.
