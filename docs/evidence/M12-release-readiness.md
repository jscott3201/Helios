# M12 Release Readiness Review

## Release Unit

Helios M12 local tiny16 release gate for the current deterministic local server/control/TUI runtime slice on `main`.

## Decision

`ready_with_known_limitations`

The M12 gate is ready for the current milestone boundary: all required local artifacts exist, the real 4-bit target path passes 16K with 128 generated tokens, 32K passes as a one-token memory probe, TUI walkthrough and failure-isolation evidence exist, and no blocker findings remain open. This is not a hand-written-native-graph performance release.

## Verification Matrix

| Gate | Command/source | Result | Evidence | Notes |
|---|---|---|---|---|
| Format | `cargo fmt --all --check` | Passed | Command output | Final formatting gate passed after docs/code edits. |
| Focused tests | `cargo test -p gemma4d-bench -p gemma4d-tui --all-targets` | Passed | Command output | 7 bench tests, 2 TUI unit tests, 13 TUI acceptance tests. |
| Real tiny16 matrix | `cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- --out-dir benchmarks/out/M12/real-matrix --model-path artifacts/models/gemma-4-12B-it-4bit` | Passed | `real-matrix/records.jsonl`, `summary.json`, `report.md` | 1K/4K/8K/16K generated 128 tokens; 32K generated 1 token. |
| M12 release gate | `cargo run -p gemma4d-bench --example m12_release_gate -- --out-dir benchmarks/out/M12` | Passed | `release-gate.json` | 1K/4K/8K/16K pass; 32K graceful memory guard. |
| MTP fixture | `cargo run -p gemma4d-engine --example mtp_fixture -- --out benchmarks/out/M12/mtp-fixture.json` | Passed | `mtp-fixture.json` | 5 cases passed. |
| RAM cache fixture | `cargo run -p gemma4d-kv --example m07_restore_matrix -- --out benchmarks/out/M12/ram-restore-matrix.json` | Passed | `ram-restore-matrix.json` | 4 cases passed. |
| SSD cache fixture | `cargo run -p gemma4d-kv --example m08_ssd_benchmark -- --out benchmarks/out/M12/ssd-benchmark.json --cache-dir benchmarks/out/M12/ssd-cache` | Passed | `ssd-benchmark.json` | 4 cases passed. |
| Adapter fixture | `cargo run -p gemma4d-adapters --example m10_adapter_fixture -- --out benchmarks/out/M12/adapter-fixture.json` | Passed | `adapter-fixture.json` | Rust adapter import/route/load/unload passed. |
| Server smoke | `cargo run -p gemma4d-server --example m11_server_smoke -- --out benchmarks/out/M12/server-smoke.json` | Passed | `server-smoke.json` | Streaming, metrics, guards, remote adapter rejection. |
| TUI walkthrough | `cargo run -p gemma4d-tui -- --provider mock --config references/configs/tiny16.toml release-walkthrough --out-dir benchmarks/out/M12/tui-walkthrough` | Passed | `tui-release-walkthrough.md` | Config, benchmark launch, metrics, adapter, cache, chat, report export. |
| TUI/server isolation | Live server + TUI HTTP walkthrough + controlled TUI failure + health/metrics curls | Passed | `server-health-after-live-tui-failure.json`, `server-metrics-after-live-tui-failure.prom`, failure log | Server remained healthy after TUI failure. |
| Full workspace verification | `scripts/verify.sh` | Passed | Command output | Run outside the restricted sandbox so localhost bind tests could bind. |

## Blockers

| Severity | Issue | Evidence | Required action |
|---|---|---|---|
| none | No blocker findings remain open for the M12 local release-gate slice. | `release-gate.json` records `blocker_findings_open = 0`. | None for M12. |

## Operational Readiness

- Rollback: revert the M12 commit; raw artifacts are ignored and do not affect runtime state.
- Migration: no data migration; `references/configs/tiny16.toml` duplicate `[tui]` table was fixed in-place.
- Observability: `/metrics`, release-gate metrics coverage, cache/MTP/adapter fixture reports, and TUI walkthrough reports exist.
- Docs/release notes: `docs/evidence/M12-release-report.md`, this readiness review, and `M12-compliance.md`.
- Compatibility: localhost-only server posture remains; remote adapter loading remains disabled.

## Final Checklist

- [x] Raw M12 artifacts generated under `benchmarks/out/M12/`.
- [x] Release report exists.
- [x] 16K real target generation passes with 128 generated tokens.
- [x] 32K real target memory probe passes with 1 generated token and memory evidence.
- [x] TUI walkthrough report exists.
- [x] TUI failure/server survival evidence exists.
- [x] No blocker findings remain open.
- [x] Final format/clippy/test/workspace verification rerun after docs land.
