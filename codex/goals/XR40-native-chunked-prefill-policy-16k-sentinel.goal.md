# XR40 - Native chunked prefill policy 16K sentinel

## Outcome

Close the remaining 16K evidence gap for the opt-in
`GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY=long_context_256` path before any
profile, server, or default adoption work.

## Scope

- Baseline: `native_eval_per_layer`.
- Candidate: `native_chunked_prefill_policy_long_context_256`.
- Workload source: `benchmarks/workloads/real-contexts/workloads.jsonl`.
- Workloads:
  - `benchmark_qa_16k_001` (`16384/16384`), seed `20260634`.
  - `long_repo_pack_16k_001` (`16384/16384`), seed `20260639`.
- Run one 16K workload at a time and record host memory state before, between,
  and after runs.
- Do not change runtime code, defaults, public C ABI, model math, tokenizer
  behavior, or non-native paths.
- If host memory pressure is already unhealthy or gets contaminated by the
  unchunked baseline, stop and record `needs_more_data` instead of forcing the
  second workload.

## Required Work

1. Use the existing XR05 prefill/eval scheduling harness and XR34 policy
   variant.
2. Capture environment and host-memory provenance:
   - `git rev-parse HEAD`
   - `git status --short`
   - `rustc -Vv`
   - `sw_vers`
   - `sysctl -n hw.memsize`
   - `vm_stat`
3. Run `cargo check` for the benchmark harness.
4. Run the `benchmark_qa_16k_001` policy sentinel first.
5. Re-sample `vm_stat`; only run `long_repo_pack_16k_001` if memory pressure
   has recovered enough for another unchunked 16K baseline.
6. Record exact commands, generated files, git SHA, deterministic workload
   seeds, context lengths, prefill p50/p95, correctness/logit deltas, peak MLX
   memory, active KV bytes, system memory observations, and blockers.
7. Update `BENCHMARKS.md` with the decision and generated artifact paths.

## Acceptance Gates

- Candidate output greedy token and greedy logit match `native_eval_per_layer`
  within the existing XR05 tolerance for every measured record.
- Candidate peak MLX memory improves by at least `5%`.
- Candidate prefill p95 does not regress by more than `5%`.
- Active KV bytes do not increase.
- Host memory observations do not show a contaminated run that invalidates the
  comparison.
- No default-on runtime change is made in this goal.

## Required Artifacts

```text
benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/benchmark-qa-16k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/long-repo-16k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}
```

If the second workload is skipped because memory pressure does not recover,
record that as an explicit blocker in this file and in `BENCHMARKS.md`.

## Result

Decision: `accept_candidate` for both 16K sentinel workloads.

No runtime code, default behavior, public C ABI, model math, tokenizer behavior,
or non-native path changed. The policy remains opt-in only.

### Provenance

- Git SHA before benchmark runs:
  `3bda45532906aa72ff322f975fc5d07e39a2af72`.
- `git status --short` before benchmark runs:
  `?? codex/goals/XR40-native-chunked-prefill-policy-16k-sentinel.goal.md`.
- `rustc -Vv`:
  `rustc 1.95.0 (59807616e 2026-04-14)`, host
  `aarch64-apple-darwin`, LLVM `22.1.2`.
- `sw_vers`: macOS `26.6`, build `25G5043d`.
- `sysctl -n hw.memsize`: `17179869184`.
- Compile check passed:
  `GEMMA4D_REQUIRE_MLX=1 cargo check -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab`.

The first sandboxed benchmark attempt failed before any usable benchmark result
because MLX could not access Metal:
`RuntimeError: [metal::load_device] No Metal device available`. The same
benchmark commands were rerun outside the sandbox with approval so native MLX
could use the GPU.

### System Memory Samples

- Initial `vm_stat`: `427297` free 16 KiB pages (about `6.52 GiB`),
  `137248` wired pages (about `2.09 GiB`).
- Mid `benchmark_qa_16k_001` run: `348068` free pages (about `5.31 GiB`),
  `189718` wired pages (about `2.90 GiB`).
- After `benchmark_qa_16k_001`: `716013` free pages (about `10.93 GiB`),
  `133593` wired pages (about `2.04 GiB`).
- Mid `long_repo_pack_16k_001` run: `5747` free pages (about `89.8 MiB`),
  `751325` wired pages (about `11.46 GiB`).
- Final sample after `long_repo_pack_16k_001`: `711929` free pages
  (about `10.86 GiB`), `133333` wired pages (about `2.03 GiB`).

The mid-run memory cliff is recorded as a tiny16 benchmarking caveat for
unchunked 16K baselines, but both runs completed and recovered without recorded
XR05 blockers.

### `benchmark_qa_16k_001`

- Run: `xr05-1782923643-495558000`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/benchmark-qa-16k-policy --trials 3 --clear-workload-ids --workload-id benchmark_qa_16k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`.
- Artifacts:
  `benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/benchmark-qa-16k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Records: `6`; passed `6`; blockers: none.
- Seed: `20260634`.
- Context: `16384/16384`.
- Prompt SHA-256:
  `0d0c0893eca1c1b52e659c7608f5a5fc5a089e00576d56c217bb982791dadf4a`.
- Correctness: candidate correct for `3/3` trials.
- Baseline prefill p50/p95: `86813.720/87063.513 ms`.
- Policy prefill p50/p95: `42244.280/51265.141 ms`.
- p50 improvement: `51.339%`.
- p95 regression value: `-41.118%` (policy p95 improved).
- Peak MLX: baseline `21.868 GB`, policy `7.620 GB`
  (`65.155%` improvement).
- Active KV: `603979776` bytes for both.
- Candidate logit delta: `0.0`.

### `long_repo_pack_16k_001`

- Run: `xr05-1782924116-206625000`.
- Command:
  `GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- --out-dir benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/long-repo-16k-policy --trials 3 --clear-workload-ids --workload-id long_repo_pack_16k_001 --variants native_eval_per_layer,native_chunked_prefill_policy_long_context_256`.
- Artifacts:
  `benchmarks/out/XR40-native-chunked-prefill-policy-16k-sentinel/long-repo-16k-policy/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Records: `6`; passed `6`; blockers: none.
- Seed: `20260639`.
- Context: `16384/16384`.
- Prompt SHA-256:
  `9c8ccf1edb13a54d66a3b7693485ada29aff77840ca1eb522b811636e128ed8f`.
- Correctness: candidate correct for `3/3` trials.
- Baseline prefill p50/p95: `87017.803/87320.961 ms`.
- Policy prefill p50/p95: `42390.024/50562.318 ms`.
- p50 improvement: `51.286%`.
- p95 regression value: `-42.096%` (policy p95 improved).
- Peak MLX: baseline `21.868 GB`, policy `7.620 GB`
  (`65.155%` improvement).
- Active KV: `603979776` bytes for both.
- Candidate logit delta: `0.125`.

## Completion Rule

Stop when the 16K policy path either passes the sentinel gates on the selected
workloads or memory/correctness/runtime evidence shows the policy needs more
data before broader profile/server adoption.
