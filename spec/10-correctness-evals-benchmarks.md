# 10 — Correctness, Evals, and Benchmarks

## Correctness ladder

1. Config parsing tests.
2. Tokenizer/chat-template fixture tests.
3. Native FFI smoke tests without model.
4. Target model load smoke test.
5. Greedy short-prompt generation stability.
6. Reference parity token sequence comparisons where practical.
7. Chunked prefill equals unchunked prefill.
8. RAM prefix restore equals fresh prefill.
9. SSD prefix restore equals fresh prefill.
10. MTP greedy equals non-MTP greedy for same target mode.
11. Adapter target output matches reference adapter path where practical.
12. Adapter cache namespace isolation.
13. TUI reducer/action tests.
14. TUI snapshot tests at required terminal sizes.
15. TUI mock/file-provider and live-server attach smoke tests.

## Benchmark matrix

Standard context lengths:

```text
1K, 4K, 8K, 16K, 32K, 64K if memory allows
```

Standard generation length:

```text
128 output tokens for most tests
512 output tokens for decode stability tests
```

Standard workload categories:

```text
simple chat
Rust code review
Python debugging
long shared repo context with small suffix change
tool-call formatting prompt
adapter-routed Rust expert prompt
adapter-routed Python expert prompt
TUI streaming chat smoke
TUI dashboard/config/cache snapshot layouts
```

## Release acceptance for tiny16

The `tiny16` profile is accepted only when:

- base 4-bit target loads without system memory collapse,
- 16K context generation works,
- 32K context generation either works or fails gracefully with memory guard evidence,
- MTP exactness passes at block size 2 if enabled,
- server returns streaming responses for a simple request,
- TUI can run a simple streaming request through the local server,
- cache namespace errors are tested,
- adapter manifest rejection tests pass,
- benchmark report includes exact environment and raw data.
