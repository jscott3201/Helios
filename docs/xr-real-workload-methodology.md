# Real-context workload methodology

The XR phase must stop relying on repeated-token probes as the primary evidence for performance claims. Repeated-token probes remain useful for deterministic smoke tests and parity, but optimization decisions need realistic prompt shapes.

## Workload sources

Use only local/repo-contained content by default so the benchmark can run offline:

- `native/gemma4_mlx/src/native_model.cc` slices for C++/MLX code reasoning.
- `crates/gemma4d-server/src/http.rs` for server/API reasoning.
- `crates/gemma4d-bench/examples/*.rs` for benchmark-analysis prompts.
- `BENCHMARKS.md` for performance-report QA prompts.
- `docs/evidence/*.md` for release/evidence summarization.
- optional user-provided adapter docs or local files under `artifacts/workloads/`, never committed if private.

## Required workload families

| ID | Family | Purpose |
|---|---|---|
| `chat_short` | 1-2K natural chat | sanity, decode speed, MTP acceptance baseline |
| `code_review_rust` | Rust/C++ code review 4K-16K | realistic coding assistant behavior |
| `benchmark_qa` | BENCHMARKS/doc QA 4K-16K | long document retrieval/analysis |
| `tool_json` | structured JSON/tool-style output | correctness under formatting constraints |
| `prefix_reuse_edit` | same long prefix, small suffix changes | RAM/SSD prefix-cache value |
| `adapter_expert` | Rust/Python expert prompt variants | adapter hot path and namespace isolation |
| `long_repo_pack` | 16K/24K/32K repo-context pack | tiny16 edge behavior |
| `mtp_candidate` | prompts likely to have predictable next tokens | MTP acceptance exploration |

## Corpus artifact format

Write JSONL records under:

```text
benchmarks/workloads/real-contexts/workloads.jsonl
```

Each record:

```json
{
  "schema_version": 1,
  "workload_id": "code_review_rust_4k_001",
  "family": "code_review_rust",
  "source_files": ["native/gemma4_mlx/src/native_model.cc"],
  "prompt_path": "benchmarks/workloads/real-contexts/prompts/code_review_rust_4k_001.txt",
  "expected_output_style": "concise_code_review",
  "max_new_tokens": 128,
  "target_context_tokens": 4096,
  "actual_context_tokens": 0,
  "deterministic_seed": 12345,
  "notes": "token count populated by tokenizer pass"
}
```

## Token length policy

Every workload must include a tokenizer-measured `actual_context_tokens` field for the target local model. Do not use character counts as a proxy in final reports.

## Correctness checks

At minimum:

- no panic / no malformed server response;
- greedy output byte/token equality when comparing backend variants that should be equivalent;
- for MTP, MTP output equals non-MTP target output for the same backend/mode;
- for compression/cache, restored or compressed output agrees with the selected gate;
- optional lightweight semantic checks for tool JSON and answer format.

## Privacy

Do not commit proprietary user files. If a private workload is needed, reference it through an ignored manifest under `artifacts/workloads/` and write a placeholder row in committed docs.
