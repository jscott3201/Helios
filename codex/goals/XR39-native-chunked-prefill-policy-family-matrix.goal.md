# XR39 - Native chunked prefill policy family matrix

## Outcome

Broaden evidence for the opt-in
`GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256` policy across additional
1K and 4K real-context families, including a 4095-token boundary workload.

## Scope

- Baseline: `native_eval_per_layer`.
- Candidate: `native_chunked_prefill_policy_long_context_256`.
- Workload source: `benchmarks/workloads/real-contexts/workloads.jsonl`.
- Workloads:
  - `tool_json_1k_001` (`1024/1024`), below threshold.
  - `benchmark_qa_4k_001` (`4096/4095`), boundary case below threshold.
  - `adapter_expert_4k_001` (`4096/4096`), at threshold.
  - `mtp_candidate_4k_001` (`4096/4096`), at threshold.
- Do not run 8K, 16K, 24K, or 32K in this goal.
- Do not change runtime code, defaults, public C ABI, model math, tokenizer
  behavior, or non-native paths.

## Required Work

1. Use the existing XR05 prefill/eval scheduling harness and XR34 policy
   variant.
2. Verify that below-threshold workloads do not show chunked-prefill memory
   shape.
3. Verify that at-threshold workloads reproduce the accepted chunked-prefill
   correctness and memory/timing shape.
4. Record exact commands, generated files, git SHA, deterministic workload
   seeds, context lengths, prefill p50/p95, correctness/logit deltas, peak MLX
   memory, active KV bytes, system memory observations, and blockers.
5. Update `BENCHMARKS.md` with the decision and generated artifact paths.

## Acceptance Gates

- Candidate output greedy token and greedy logit match `native_eval_per_layer`
  within the existing XR05 tolerance for every measured record.
- Below-threshold workloads are not used as speed/memory acceptance evidence and
  should not show the chunked memory shape.
- At-threshold workloads must either improve peak MLX by at least `5%` or p50
  prefill by at least `10%`.
- Candidate prefill p95 must not regress by more than `5%` on accepted
  at-threshold workloads.
- Active KV bytes do not increase.
- No default-on runtime change is made in this goal.

## Required Artifacts

```text
benchmarks/out/XR39-native-chunked-prefill-policy-family-matrix/policy-family-matrix/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

## Result

Decision: `accept_candidate` for the selected family matrix.

The opt-in policy stayed correct across all selected workloads. The
below-threshold rows behaved as boundary/no-chunk checks, while the 4096-token
rows reproduced the accepted chunked memory/timing shape.

- Run: `xr05-1782922915-623972000`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR39-native-chunked-prefill-policy-family-matrix/policy-family-matrix --trials 3 --clear-workload-ids --workload-id tool_json_1k_001 --workload-id benchmark_qa_4k_001 --workload-id adapter_expert_4k_001 --workload-id mtp_candidate_4k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`.
- Artifacts:
  `benchmarks/out/XR39-native-chunked-prefill-policy-family-matrix/policy-family-matrix/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Records: `24`; passed `24`; blockers: none.

### Below Threshold

- `tool_json_1k_001`, seed `20260635`, context `1024/1024`, prompt SHA-256
  `7687cd292cf8f9be5f84f3dca2e3644a08d973a1a314facb52ac91bbed0d5e2c`.
  Correctness was `3/3`; baseline p50/p95 was `3151.692/3322.586 ms`;
  policy p50/p95 was `2707.186/3019.340 ms`; peak MLX stayed
  `7.321 GB`; active KV stayed `352321536` bytes; logit delta was `0.0`.
- `benchmark_qa_4k_001`, seed `20260633`, context `4096/4095`, prompt SHA-256
  `1514934863d5ad974300a0feb490ac2dbf1ab2eadc2e7f1a1525e2c2eb3b4e42`.
  Correctness was `3/3`; baseline p50/p95 was `16288.026/16366.339 ms`;
  policy p50/p95 was `13619.109/14908.799 ms`; peak MLX was effectively the
  non-chunked shape (`9.212 -> 9.244 GB` max); active KV stayed `402636800`
  bytes; logit delta was `0.0`.

These below-threshold timings are recorded as same-path variance checks, not as
chunked-prefill speed evidence.

### At Threshold

- `adapter_expert_4k_001`, seed `20260638`, context `4096/4096`, prompt SHA-256
  `e4f055746d250beee415c30893f1baae9efce40789e70e77196b506ff5a3f3a7`.
  Correctness was `3/3`; baseline p50/p95 was `13705.482/14778.767 ms`;
  policy p50/p95 was `10628.725/11340.667 ms`; p50 improved `22.449%`; p95
  improved `23.264%`; peak MLX improved from `9.279` to `7.300 GB`
  (`21.330%`); active KV stayed `402653184` bytes; logit delta was `0.125`.
- `mtp_candidate_4k_001`, seed `20260642`, context `4096/4096`, prompt SHA-256
  `88f76c633511de568b6270b3217be53a26a5c7235862a3c23a514de2646268b3`.
  Correctness was `3/3`; baseline p50/p95 was `11089.101/11198.204 ms`;
  policy p50/p95 was `9462.730/9519.092 ms`; p50 improved `14.666%`; p95
  improved `14.994%`; peak MLX improved from `9.279` to `7.300 GB`
  (`21.330%`); active KV stayed `402653184` bytes; logit delta was `0.125`.

System memory pressure remained a benchmark caveat. A mid-run `vm_stat` sample
showed `4664` free 16 KiB pages, about `73 MiB`, and `603650` wired pages,
about `9.21 GiB`, before recovering after the process exited.

## Completion Rule

Stop when the policy matrix either supports the opt-in policy boundary across
the selected families or identifies a correctness, memory, or p95 regression
that blocks broader policy exposure.
