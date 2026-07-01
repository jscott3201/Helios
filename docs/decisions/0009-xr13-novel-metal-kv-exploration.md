# Decision Record: XR13 Novel Metal/KV Exploration

- Status: accepted
- Date: 2026-07-01
- Goal: XR13

## Context

XR09 rejected the current q8/q4 KV compression candidates for default promotion:
q8 failed the `benchmark_qa_4k_001` real-context quality gate, q4 failed three
families, and both restored compressed payloads into BF16 active KV memory. That
means XR13 should not change runtime decode behavior until a smaller experiment
proves that active compressed scoring can preserve attention ordering while
reducing active K memory.

## Hypothesis

An isolated decode L=1 score path can identify whether K-only compression is
worth native/Metal integration before touching the runtime:

- `bf16_reference` is the exact score-ordering baseline.
- `mlx_affine_q8_reference` is the XR09 comparison baseline.
- `planar4_k_only_candidate` stores K in a 4-bit K-only representation and
  dequantizes while scoring.
- `turbo_score_estimation_candidate` stores a compact projected score sketch and
  estimates attention ordering without reconstructing full BF16 K.

The candidate is worth further Metal work only if it preserves reference top-1 or
top-8 ordering on XR09 real-context-shaped cases, reduces active K bytes relative
to q8, and does not show an obvious single-token score latency regression in the
microbenchmark.

## Decision

Create a feature-gated benchmark example,
`xr13_novel_metal_kv_exploration`, behind Cargo feature `xr13-prototypes`.
It reads XR09 summary/records, reuses XR09 workload IDs, deterministic seeds,
context token lengths, and prefix hashes, then generates deterministic synthetic
KV-like vectors for an isolated L=1 attention-score microbenchmark.

This is intentionally a prototype slice:

- No default runtime path changes.
- No native MLX or custom Metal kernel is merged.
- No broad model abstraction is added.
- Speed claims are limited to the isolated score loop and are reported only with
  correctness and estimated active-memory evidence.
- Any failed candidate or kernel/API blocker is written to
  `benchmarks/out/XR13-novel-metal-kv-exploration/blockers.md`.

## Consequences

- If Planar4 or Turbo fails ordering/correctness gates, XR13 can reject or keep
  the idea experimental without spending time on a custom Metal boundary.
- If a candidate passes, a later goal can design the narrow C ABI and MLX/Metal
  implementation with XR13 evidence as input.
- XR09 remains the real-model BF16/q8/q4 authority; XR13 is only a synthetic
  score-path exploration tied to XR09 workload shapes and seeds.

## Evidence

- Clean run `xr13-1782890847` at code commit `4e1bc28` generated
  `benchmarks/out/XR13-novel-metal-kv-exploration/{records.jsonl,summary.json,report.md,blockers.md,decision.md}`.
- Decision is `reject_candidate`: Turbo score estimation failed correctness on
  18/18 samples.
- Planar4 K-only passed the isolated synthetic score gate but remains
  prototype-only because XR13 did not add a real-model active decode path, C ABI,
  or Metal kernel.
- Active compressed KV decode remains disabled by default.
