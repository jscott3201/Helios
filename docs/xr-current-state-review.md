# Current state review for the XR optimization phase

This review is based on the current `main` branch as inspected from the GitHub repo and `BENCHMARKS.md`.

## What is already strong

- The repo has a Rust 1.95 workspace, narrow Rust-to-MLX FFI boundary, helper-backed Gemma 4 generation, local OpenAI-compatible server, and Ratatui TUI surface.
- `BENCHMARKS.md` now records P00-P10 runs with a clear rule that helper-backed, native-graph, fixture, and server modes must not be conflated.
- P00 added rich timing fields: model load, prefill, decode, total, per-token latency, memory, and model identity hashes.
- P04 added opt-in native incremental KV decode; P04 evidence says steady p50/p95 stayed flat from 1K to 8K despite 8x context growth.
- P06/P07 moved RAM and SSD prefix cache from pure fixtures to native snapshot payload import/export.
- P08 measured real native prefix payload compression, with q8 passing continued-decode agreement and q4 failing greedy agreement.
- P09 moved LoRA from control-plane fixture into native inference for one deterministic rank-16 q_proj/v_proj adapter fixture.

## Main gaps this pack targets

### 1. Repeated-token workloads are overrepresented

Many pivotal numbers still come from `repeat_9259_*` probes or two short `hello_*` prompts. That is useful for deterministic smoke tests but weak for claims about code, tool, adapter, long-prefix, or MTP behavior.

### 2. MTP has real plumbing but poor current acceptance evidence

P05 uses real native target + assistant FFI, but the only probes are `hello_smoke` and `hello_reference_prefix`. Current acceptance was 0.000 for block sizes 1 and 2. The next step is not blind optimization; it is trace-level diagnosis on realistic prompts and token distributions.

### 3. Native decode p50/p95 looks promising, but raw p95 outliers remain

P04 shows a strong steady-state claim, but the 8K raw p95 was much higher than steady p95 after warmup discard. XR06 isolates tail latency and MLX lazy-eval synchronization boundaries.

### 4. Server path still needs persistent-native evolution

P02 real-helper server mode calls the `generate` path per request, so model load and tokenizer/detokenizer helper costs are still paid in the request path. XR11 targets persistent model state and real server A/B evidence.

### 5. Prefix cache and SSD wins need real reuse patterns

P06/P07 demonstrate exactness and impressive warm restore on measured contexts. The next question is whether this pays off on realistic repeated prefixes: repo context + small user edits, tool loops, adapter routes, and long chat history.

### 6. KV compression needs quality realism and active-decode research

P08 q8 passed, q4 failed, and active decode still restores BF16. XR09 re-tests on real workloads; XR13 explores fused compressed-domain attention only behind feature flags.

## Measurement principle

Every XR goal must end with a decision:

```text
accept_candidate
reject_candidate
keep_experimental
needs_more_data
blocked_with_evidence
```

A performance improvement without correctness, memory, and variance evidence is not an accepted improvement.
