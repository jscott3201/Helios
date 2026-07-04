# XR61 Wave 5 completion audit

## Decision

Overall decision: `keep_experimental`.

Wave 5 did not justify default-on MTP. It did close the requested frontier pass:
Adaptive-N MTP is proven keep-experimental, native server/default closure is
accepted, P3 verifier/fallback cleanup was rejected below gate, and P4 native
decode profiling identified the remaining limiter.

## Requirement audit

| Requirement | Current evidence | Status |
|---|---|---|
| Adaptive-N MTP decision | `benchmarks/out/XR61-adaptive-n-mtp/xr61-adaptive-n-summary.{md,json}` and `codex/goals/XR61-adaptive-n-mtp.goal.md`; primary v2 exact `9/9`, oracle compared `9`, aggregate `+21.303%` below `25%` | `keep_experimental` |
| Holdout protection | `candidate-adaptive-n-v2-safe-bypass-holdouts/`; 4K/protected workloads baseline-bypassed, exact `9/9`, selected no MTP | passed |
| Sequential oracle | `sequential-oracle-adaptive-n-v2-safe-bypass/`; v2 candidate generated-token differential matched `9/9` | passed |
| Native server/default closure | `codex/goals/XR62-server-default-sentinel-closure.goal.md`; sentinel artifacts for 8K, 16K, and 24K-low-N | `accept_candidate` |
| P3 verifier/fallback overhead lane | `codex/goals/XR63-mtp-terminal-block-prefix-no-lookahead.goal.md`; exact/oracle clean but selected decode only `+2.043%` | `reject_candidate` |
| P4 native decode stage profile | `codex/goals/XR64-native-decode-stage-profile.goal.md`; `native-decode-profile/` has `378/378` profiled decode samples | complete |
| BENCHMARKS claim boundaries | `BENCHMARKS.md` rows for XR61, XR62, XR63, XR64 and claim-boundary bullets | passed |

## Artifact reconciliation

The canonical goal listed early expected paths for v1 evidence. The final
accepted evidence uses v2 safe-bypass paths where the v1 candidate was
superseded by the safer policy.

| Original expected path | Final evidence |
|---|---|
| `benchmarks/out/XR61-adaptive-n-mtp/baseline-xr56-policy/` | Not produced as a fresh directory. XR61 summary records `baseline_summary: not provided` and recomputes the XR56 guarded comparator from `benchmarks/out/XR56-repair-cost/`; the final decision does not claim default-on promotion. |
| `benchmarks/out/XR61-adaptive-n-mtp/sequential-oracle-adaptive-n/` | Superseded by `benchmarks/out/XR61-adaptive-n-mtp/sequential-oracle-adaptive-n-v2-safe-bypass/`, which is the oracle path used by the final XR61 decision. |

## Final limiter

The measured limiter is native forward graph plus decode KV append/cache
mutation. XR64 shows `forward_graph_ms` mean `76.012..79.996 ms` of
`82.283..86.331 ms` native decode total, while FFI, greedy selection, scalar
reads, hidden-view handoff, and peak-memory telemetry are sub-millisecond.

Next single action: design XR65 as a scoped native graph/KV mutation lane that
first separates layer graph cost from decode KV append/cache mutation before
any kernel or cache-layout rewrite.
