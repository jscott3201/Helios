use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use gemma4d_tokenizer::sha256_hex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{CliError, manifest};

pub const DEFAULT_MODEL_PATH: &str = "artifacts/models/gemma-4-12B-it-4bit";
pub const DEFAULT_WORKLOAD_DIR: &str = "benchmarks/workloads/real-contexts";
pub const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR00-real-workload-corpus";
pub const DEFAULT_SEED: u64 = 20_260_630;

const REQUIRED_FAMILIES: &[&str] = &[
    "chat_short",
    "code_review_rust",
    "benchmark_qa",
    "tool_json",
    "prefix_reuse_edit",
    "adapter_expert",
    "long_repo_pack",
    "mtp_candidate",
];

const REQUIRED_TARGETS: &[usize] = &[1024, 4096, 8192, 16_384];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadCorpusOptions {
    pub model_path: PathBuf,
    pub workload_dir: PathBuf,
    pub out_dir: PathBuf,
    pub python: PathBuf,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadSpec {
    pub workload_id: &'static str,
    pub family: &'static str,
    pub source_files: Vec<&'static str>,
    pub expected_output_style: &'static str,
    pub max_new_tokens: usize,
    pub target_context_tokens: usize,
    pub instruction: &'static str,
    pub final_task: &'static str,
    pub notes: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadRecord {
    pub schema_version: u32,
    pub workload_id: String,
    pub family: String,
    pub source_files: Vec<String>,
    pub prompt_path: String,
    pub expected_output_style: String,
    pub max_new_tokens: usize,
    pub target_context_tokens: usize,
    pub actual_context_tokens: usize,
    pub deterministic_seed: u64,
    pub prompt_sha256: String,
    pub tokenizer_backend: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceRecord {
    schema_version: u32,
    goal: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    model_identity: manifest::ArtifactIdentity,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    prompt_bytes: usize,
    source_files: Vec<String>,
    expected_output_style: String,
    max_new_tokens: usize,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    deterministic_seed: u64,
    tokenizer_backend: String,
    validation_status: String,
    notes: String,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: String,
    decision: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    model_identity: manifest::ArtifactIdentity,
    tokenizer_backend: String,
    deterministic_seed_base: u64,
    workload_count: usize,
    families: Vec<String>,
    target_context_tokens: Vec<usize>,
    actual_context_tokens_min: usize,
    actual_context_tokens_max: usize,
    generated_files: Vec<String>,
    blockers: Vec<String>,
}

pub fn write_workload_corpus_artifacts(
    options: &WorkloadCorpusOptions,
) -> Result<String, CliError> {
    let prompt_dir = options.workload_dir.join("prompts");
    fs::create_dir_all(&prompt_dir)
        .map_err(|error| CliError::Runtime(format!("failed to create prompt dir: {error}")))?;
    fs::create_dir_all(&options.out_dir)
        .map_err(|error| CliError::Runtime(format!("failed to create out dir: {error}")))?;

    let run_id = run_id();
    let git_sha =
        command_stdout("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let git_status_short =
        command_stdout("git", &["status", "--short"]).unwrap_or_else(|| "unknown".to_owned());
    let model_identity =
        manifest::capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION");
    let command = command_display(options);
    let mut counter = TokenizerCounter::start(&options.python, &options.model_path)?;

    let specs = workload_specs();
    let mut records = Vec::with_capacity(specs.len());
    let mut evidence = Vec::with_capacity(specs.len());
    let mut generated_files = vec![
        options.workload_dir.join("README.md").display().to_string(),
        options
            .workload_dir
            .join("workloads.jsonl")
            .display()
            .to_string(),
    ];

    for (index, spec) in specs.iter().enumerate() {
        let seed = options.seed + index as u64;
        let (prompt, actual_context_tokens) = build_prompt_for_target(spec, seed, &mut counter)?;
        let prompt_path = prompt_dir.join(format!("{}.txt", spec.workload_id));
        fs::write(&prompt_path, &prompt).map_err(|error| {
            CliError::Runtime(format!(
                "failed to write prompt {}: {error}",
                prompt_path.display()
            ))
        })?;
        let prompt_sha256 = sha256_hex(prompt.as_bytes());
        let relative_prompt_path = prompt_path.display().to_string();
        generated_files.push(relative_prompt_path.clone());
        let source_files = spec
            .source_files
            .iter()
            .map(|path| (*path).to_owned())
            .collect::<Vec<_>>();
        let record = WorkloadRecord {
            schema_version: 1,
            workload_id: spec.workload_id.to_owned(),
            family: spec.family.to_owned(),
            source_files: source_files.clone(),
            prompt_path: relative_prompt_path.clone(),
            expected_output_style: spec.expected_output_style.to_owned(),
            max_new_tokens: spec.max_new_tokens,
            target_context_tokens: spec.target_context_tokens,
            actual_context_tokens,
            deterministic_seed: seed,
            prompt_sha256: prompt_sha256.clone(),
            tokenizer_backend: counter.backend().to_owned(),
            notes: spec.notes.to_owned(),
        };
        evidence.push(EvidenceRecord {
            schema_version: 1,
            goal: "XR00-real-workload-corpus".to_owned(),
            run_id: run_id.clone(),
            git_sha: git_sha.clone(),
            git_status_short: git_status_short.clone(),
            model_identity: model_identity.clone(),
            workload_id: record.workload_id.clone(),
            family: record.family.clone(),
            prompt_path: record.prompt_path.clone(),
            prompt_sha256,
            prompt_bytes: prompt.len(),
            source_files,
            expected_output_style: record.expected_output_style.clone(),
            max_new_tokens: record.max_new_tokens,
            target_context_tokens: record.target_context_tokens,
            actual_context_tokens: record.actual_context_tokens,
            deterministic_seed: record.deterministic_seed,
            tokenizer_backend: record.tokenizer_backend.clone(),
            validation_status: "passed".to_owned(),
            notes: record.notes.clone(),
        });
        records.push(record);
    }
    let tokenizer_backend = counter.backend().to_owned();
    counter.shutdown();

    let blockers = validate_records(&records);
    let decision = if blockers.is_empty() {
        "accept_candidate"
    } else {
        "blocked_with_evidence"
    };

    write_readme(options, &records, &command)?;
    write_workload_jsonl(&options.workload_dir.join("workloads.jsonl"), &records)?;

    let records_path = options.out_dir.join("records.jsonl");
    write_evidence_jsonl(&records_path, &evidence)?;
    let mut evidence_files = vec![
        records_path.display().to_string(),
        options.out_dir.join("summary.json").display().to_string(),
        options.out_dir.join("report.md").display().to_string(),
        options.out_dir.join("blockers.md").display().to_string(),
        options.out_dir.join("decision.md").display().to_string(),
    ];
    generated_files.append(&mut evidence_files);
    generated_files.sort();

    let summary = build_summary(
        options,
        &records,
        &generated_files,
        &blockers,
        decision,
        &run_id,
        &git_sha,
        &git_status_short,
        &model_identity,
        &tokenizer_backend,
        &command,
    );
    write_json_pretty(&options.out_dir.join("summary.json"), &summary)?;
    fs::write(
        options.out_dir.join("report.md"),
        render_report(&records, &summary),
    )
    .map_err(|error| CliError::Runtime(format!("failed to write report.md: {error}")))?;
    fs::write(
        options.out_dir.join("blockers.md"),
        render_blockers(&blockers, &command),
    )
    .map_err(|error| CliError::Runtime(format!("failed to write blockers.md: {error}")))?;
    fs::write(
        options.out_dir.join("decision.md"),
        render_decision(decision, &records, &blockers),
    )
    .map_err(|error| CliError::Runtime(format!("failed to write decision.md: {error}")))?;

    Ok(format!(
        "wrote corpus {} and evidence {}",
        options.workload_dir.join("workloads.jsonl").display(),
        options.out_dir.display()
    ))
}

pub fn workload_specs() -> Vec<WorkloadSpec> {
    vec![
        WorkloadSpec {
            workload_id: "chat_short_1k_001",
            family: "chat_short",
            source_files: vec!["AGENTS.md", "docs/evidence/M12.md"],
            expected_output_style: "concise_operator_answer",
            max_new_tokens: 128,
            target_context_tokens: 1024,
            instruction: "Simulate a short operator chat about Helios release evidence. Preserve concrete file paths, status words, and constraints from the context.",
            final_task: "Answer as the local operator console assistant. Give the next safe action and cite the relevant evidence file names.",
            notes: "1K natural chat sanity workload from repo instructions and release evidence.",
        },
        WorkloadSpec {
            workload_id: "code_review_rust_4k_001",
            family: "code_review_rust",
            source_files: vec![
                "native/gemma4_mlx/src/runtime.cc",
                "crates/gemma4d-ffi/src/lib.rs",
            ],
            expected_output_style: "concise_code_review",
            max_new_tokens: 192,
            target_context_tokens: 4096,
            instruction: "Review the Rust/C++ FFI boundary for correctness risks. Focus on ownership, failure-closed behavior, and whether broad raw MLX internals leak into Rust.",
            final_task: "Return prioritized findings with file/function anchors and one verification command.",
            notes: "4K code review workload for native boundary reasoning.",
        },
        WorkloadSpec {
            workload_id: "code_review_rust_8k_001",
            family: "code_review_rust",
            source_files: vec![
                "native/gemma4_mlx/src/native_model.cc",
                "native/gemma4_mlx/src/runtime.cc",
            ],
            expected_output_style: "deep_code_review",
            max_new_tokens: 256,
            target_context_tokens: 8192,
            instruction: "Review the native MLX text graph and KV path for correctness and measurement risks. Separate real bugs from missing future optimizations.",
            final_task: "Return only high-signal findings, grouped by correctness, memory, and benchmark validity.",
            notes: "8K native code review workload using real C++/MLX source.",
        },
        WorkloadSpec {
            workload_id: "benchmark_qa_4k_001",
            family: "benchmark_qa",
            source_files: vec!["BENCHMARKS.md", "docs/xr-current-state-review.md"],
            expected_output_style: "benchmark_claim_audit",
            max_new_tokens: 192,
            target_context_tokens: 4096,
            instruction: "Audit benchmark claims for mode confusion. Treat helper-backed, native, server, fixture, cache, adapter, and TUI measurements as separate evidence classes.",
            final_task: "List which claims are supported, which are out of scope, and what artifact paths prove each claim.",
            notes: "4K benchmark QA workload from the ledger and XR state review.",
        },
        WorkloadSpec {
            workload_id: "benchmark_qa_16k_001",
            family: "benchmark_qa",
            source_files: vec![
                "BENCHMARKS.md",
                "docs/evidence/M12-release-readiness.md",
                "docs/evidence/M12-release-report.md",
            ],
            expected_output_style: "long_evidence_synthesis",
            max_new_tokens: 256,
            target_context_tokens: 16_384,
            instruction: "Synthesize release evidence without overstating native performance. Pay attention to claim boundaries, blocker reports, and verification commands.",
            final_task: "Write a release-evidence summary with accepted claims, deferred claims, and exact evidence paths.",
            notes: "16K benchmark/document QA workload for long evidence retrieval.",
        },
        WorkloadSpec {
            workload_id: "tool_json_1k_001",
            family: "tool_json",
            source_files: vec!["crates/gemma4d-server/src/http.rs"],
            expected_output_style: "strict_json",
            max_new_tokens: 160,
            target_context_tokens: 1024,
            instruction: "Use the local HTTP server code context to infer stable endpoint names and response fields. The answer must be JSON only.",
            final_task: "Return a JSON object with keys endpoints, metrics, risks, and verification_commands. Do not include Markdown.",
            notes: "1K structured-output workload for formatting correctness.",
        },
        WorkloadSpec {
            workload_id: "prefix_reuse_edit_8k_a_001",
            family: "prefix_reuse_edit",
            source_files: vec!["crates/gemma4d-server/src/http.rs", "BENCHMARKS.md"],
            expected_output_style: "prefix_reuse_answer",
            max_new_tokens: 128,
            target_context_tokens: 8192,
            instruction: "This workload is half of a repeated-prefix pair. Most context is intentionally shared with the sibling workload to measure prefix cache value.",
            final_task: "User edit A: summarize whether the server path evidence supports a persistent-native latency claim.",
            notes: "Shares a long prefix with prefix_reuse_edit_8k_b_001; suffix asks server-latency question A.",
        },
        WorkloadSpec {
            workload_id: "prefix_reuse_edit_8k_b_001",
            family: "prefix_reuse_edit",
            source_files: vec!["crates/gemma4d-server/src/http.rs", "BENCHMARKS.md"],
            expected_output_style: "prefix_reuse_answer",
            max_new_tokens: 128,
            target_context_tokens: 8192,
            instruction: "This workload is half of a repeated-prefix pair. Most context is intentionally shared with the sibling workload to measure prefix cache value.",
            final_task: "User edit B: summarize whether the server path evidence supports enabling SSD prefix cache by default.",
            notes: "Shares a long prefix with prefix_reuse_edit_8k_a_001; suffix asks cache-default question B.",
        },
        WorkloadSpec {
            workload_id: "adapter_expert_4k_001",
            family: "adapter_expert",
            source_files: vec![
                "crates/gemma4d-adapters/src/lib.rs",
                "crates/gemma4d-bench/examples/p09_real_lora_adapter.rs",
            ],
            expected_output_style: "adapter_expert_review",
            max_new_tokens: 192,
            target_context_tokens: 4096,
            instruction: "Review adapter import and native hot-path evidence. Emphasize trusted roots, tokenizer/template compatibility, namespace isolation, and MTP adapter gates.",
            final_task: "Return a concise adapter-runtime risk assessment and the exact tests or benchmark records to inspect next.",
            notes: "4K adapter-specialist workload for LoRA routing and namespace isolation.",
        },
        WorkloadSpec {
            workload_id: "long_repo_pack_16k_001",
            family: "long_repo_pack",
            source_files: vec![
                "AGENTS.md",
                "BENCHMARKS.md",
                "native/gemma4_mlx/src/native_model.cc",
                "crates/gemma4d-server/src/http.rs",
                "crates/gemma4d-bench/examples/p10_tui_live_console.rs",
            ],
            expected_output_style: "long_context_project_synthesis",
            max_new_tokens: 256,
            target_context_tokens: 16_384,
            instruction: "Use the mixed repo context to identify the safest next benchmark goal. Separate runtime, server, TUI, and evidence-ledger concerns.",
            final_task: "Return a short implementation plan with non-goals and verification gates.",
            notes: "16K long repo-context pack for tiny16 realistic prompt shape.",
        },
        WorkloadSpec {
            workload_id: "long_repo_pack_24k_001",
            family: "long_repo_pack",
            source_files: vec![
                "AGENTS.md",
                "BENCHMARKS.md",
                "docs/evidence/M12-release-readiness.md",
                "native/gemma4_mlx/src/native_model.cc",
                "crates/gemma4d-server/src/http.rs",
                "crates/gemma4d-bench/examples/p04_incremental_native_kv.rs",
                "crates/gemma4d-bench/examples/p07_real_ssd_prefix_cache.rs",
            ],
            expected_output_style: "edge_context_triage",
            max_new_tokens: 128,
            target_context_tokens: 24_576,
            instruction: "Treat this as a tiny16 edge-case context pack. Do not assume the model run will be cheap; focus on whether the corpus metadata is reproducible.",
            final_task: "Return only the top three risks for running this edge context in an A/B benchmark.",
            notes: "Optional 24K edge workload; XR00 records tokenizer length only and performs no model execution.",
        },
        WorkloadSpec {
            workload_id: "mtp_candidate_1k_001",
            family: "mtp_candidate",
            source_files: vec!["crates/gemma4d-bench/examples/p05_native_mtp.rs"],
            expected_output_style: "predictable_continuation",
            max_new_tokens: 64,
            target_context_tokens: 1024,
            instruction: "This prompt is designed for speculative decoding diagnosis. It asks for a deterministic continuation with repeated structure and explicit numbering.",
            final_task: "Continue the checklist with items 6 through 12, preserving the exact prefix 'MTP check N:' for each line.",
            notes: "1K MTP candidate with predictable numbered continuation.",
        },
        WorkloadSpec {
            workload_id: "mtp_candidate_4k_001",
            family: "mtp_candidate",
            source_files: vec![
                "crates/gemma4d-bench/examples/p05_native_mtp.rs",
                "docs/xr-current-state-review.md",
            ],
            expected_output_style: "mtp_trace_plan",
            max_new_tokens: 128,
            target_context_tokens: 4096,
            instruction: "Use the MTP benchmark code and XR state review to design trace evidence before proposing fixes.",
            final_task: "Return a trace plan with fields draft_token, target_token, accepted_count, rollback, verify_latency, and suspected_root_cause.",
            notes: "4K MTP candidate for realistic acceptance diagnosis.",
        },
    ]
}

fn build_prompt_for_target(
    spec: &WorkloadSpec,
    seed: u64,
    counter: &mut TokenizerCounter,
) -> Result<(String, usize), CliError> {
    let source_unit = source_unit(spec)?;
    let mut body = String::new();
    let mut round = 0usize;
    loop {
        round += 1;
        body.push_str(&format!(
            "\n\n## Source Round {round} for {}\n\n",
            spec.workload_id
        ));
        body.push_str(&source_unit);
        let candidate = render_prompt(spec, seed, &body);
        if counter.count_tokens(&candidate)? >= spec.target_context_tokens || round >= 16 {
            break;
        }
    }

    let positions = char_positions(&body);
    let mut low = 0usize;
    let mut high = positions.len().saturating_sub(1);
    let mut best_body = String::new();
    let mut best_prompt = render_prompt(spec, seed, "");
    let mut best_count = counter.count_tokens(&best_prompt)?;

    while low <= high {
        let mid = low + (high - low) / 2;
        let candidate_body = &body[..positions[mid]];
        let candidate = render_prompt(spec, seed, candidate_body);
        let count = counter.count_tokens(&candidate)?;
        if count <= spec.target_context_tokens {
            best_body = candidate_body.to_owned();
            best_prompt = candidate;
            best_count = count;
            low = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }

    if best_count < spec.target_context_tokens.saturating_mul(9) / 10 {
        return Err(CliError::Runtime(format!(
            "{} could not reach 90% of target tokens: actual={} target={}",
            spec.workload_id, best_count, spec.target_context_tokens
        )));
    }

    if best_count < spec.target_context_tokens {
        (best_prompt, best_count) =
            pad_prompt_to_target(spec, seed, best_body, best_count, counter)?;
    }

    Ok((best_prompt, best_count))
}

fn pad_prompt_to_target(
    spec: &WorkloadSpec,
    seed: u64,
    mut body: String,
    mut best_count: usize,
    counter: &mut TokenizerCounter,
) -> Result<(String, usize), CliError> {
    let mut best_prompt = render_prompt(spec, seed, &body);
    let fragments = [" a", " b", " c", " d", " 0", " 1", ".", "\n"];
    while best_count < spec.target_context_tokens {
        let mut progressed = false;
        for fragment in fragments {
            let candidate_body = format!("{body}{fragment}");
            let candidate = render_prompt(spec, seed, &candidate_body);
            let count = counter.count_tokens(&candidate)?;
            if count > best_count && count <= spec.target_context_tokens {
                body = candidate_body;
                best_prompt = candidate;
                best_count = count;
                progressed = true;
                break;
            }
        }
        if !progressed {
            break;
        }
    }
    Ok((best_prompt, best_count))
}

fn source_unit(spec: &WorkloadSpec) -> Result<String, CliError> {
    let mut out = String::new();
    for source in &spec.source_files {
        let path = Path::new(source);
        let text = fs::read_to_string(path).map_err(|error| {
            CliError::Runtime(format!(
                "failed to read source file {}: {error}",
                path.display()
            ))
        })?;
        out.push_str(&format!("### BEGIN {}\n", path.display()));
        out.push_str(&text);
        if !text.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("### END {}\n\n", path.display()));
    }
    Ok(out)
}

fn render_prompt(spec: &WorkloadSpec, seed: u64, source_body: &str) -> String {
    let sources = spec.source_files.join(", ");
    format!(
        "# Helios XR00 Real-Context Workload\n\n\
Workload ID: `{}`\n\
Family: `{}`\n\
Target context tokens: `{}`\n\
Deterministic seed: `{}`\n\
Source files: `{}`\n\n\
## Corpus Instruction\n\n{}\n\n\
## Repo Context\n{}\
\n\n## Final User Task\n\n{}\n\n\
## Expected Output Style\n\n{}\n",
        spec.workload_id,
        spec.family,
        spec.target_context_tokens,
        seed,
        sources,
        spec.instruction,
        source_body,
        spec.final_task,
        spec.expected_output_style
    )
}

fn char_positions(value: &str) -> Vec<usize> {
    let mut positions = value
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    positions.push(value.len());
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn validate_records(records: &[WorkloadRecord]) -> Vec<String> {
    let mut blockers = Vec::new();
    let families = records
        .iter()
        .map(|record| record.family.as_str())
        .collect::<BTreeSet<_>>();
    for family in REQUIRED_FAMILIES {
        if !families.contains(family) {
            blockers.push(format!("missing required workload family {family}"));
        }
    }

    let targets = records
        .iter()
        .map(|record| record.target_context_tokens)
        .collect::<BTreeSet<_>>();
    for target in REQUIRED_TARGETS {
        if !targets.contains(target) {
            blockers.push(format!("missing required target context length {target}"));
        }
    }
    if !targets.iter().any(|target| *target >= 24_576) {
        blockers.push("missing optional 24K/32K edge context".to_owned());
    }

    for record in records {
        if record.actual_context_tokens == 0 {
            blockers.push(format!(
                "{} has zero actual_context_tokens",
                record.workload_id
            ));
        }
        if record.actual_context_tokens < record.target_context_tokens.saturating_mul(9) / 10 {
            blockers.push(format!(
                "{} actual_context_tokens {} is below 90% of target {}",
                record.workload_id, record.actual_context_tokens, record.target_context_tokens
            ));
        }
        if !Path::new(&record.prompt_path).exists() {
            blockers.push(format!("{} prompt_path does not exist", record.workload_id));
        }
    }

    blockers
}

fn write_readme(
    options: &WorkloadCorpusOptions,
    records: &[WorkloadRecord],
    command: &str,
) -> Result<(), CliError> {
    let mut body = String::new();
    body.push_str("# Helios XR00 Real-Context Workload Corpus\n\n");
    body.push_str(
        "This corpus contains deterministic, repo-local prompt contexts for Helios XR-phase A/B benchmarks. It replaces repeated-token-only probes with realistic prompt shapes while performing no model execution.\n\n",
    );
    body.push_str("## Regeneration\n\n");
    body.push_str("```text\n");
    body.push_str(command);
    body.push_str("\n```\n\n");
    body.push_str(&format!(
        "- Model tokenizer path: `{}`\n- Deterministic seed base: `{}`\n- Workload manifest: `{}`\n- Evidence directory: `{}`\n\n",
        options.model_path.display(),
        options.seed,
        options.workload_dir.join("workloads.jsonl").display(),
        options.out_dir.display()
    ));
    body.push_str("## Families\n\n");
    body.push_str("| Family | Workloads |\n|---|---:|\n");
    for family in REQUIRED_FAMILIES {
        let count = records
            .iter()
            .filter(|record| record.family == *family)
            .count();
        body.push_str(&format!("| `{family}` | {count} |\n"));
    }
    body.push_str("\n## Token Length Policy\n\n");
    body.push_str(
        "`actual_context_tokens` is measured with the local Gemma 4 tokenizer through `mlx_lm.utils.load_tokenizer`; character counts are not used as a proxy.\n\n",
    );
    body.push_str("## Privacy\n\n");
    body.push_str(
        "All committed prompts are generated from repo-local files. Private user artifacts must stay under ignored `artifacts/workloads/` paths and are not part of XR00.\n",
    );
    fs::write(options.workload_dir.join("README.md"), body)
        .map_err(|error| CliError::Runtime(format!("failed to write corpus README: {error}")))
}

fn write_workload_jsonl(path: &Path, records: &[WorkloadRecord]) -> Result<(), CliError> {
    let mut body = String::new();
    for record in records {
        body.push_str(
            &serde_json::to_string(record).map_err(|error| {
                CliError::Runtime(format!("failed to serialize record: {error}"))
            })?,
        );
        body.push('\n');
    }
    fs::write(path, body)
        .map_err(|error| CliError::Runtime(format!("failed to write workloads.jsonl: {error}")))
}

fn write_evidence_jsonl(path: &Path, records: &[EvidenceRecord]) -> Result<(), CliError> {
    let mut body = String::new();
    for record in records {
        body.push_str(&serde_json::to_string(record).map_err(|error| {
            CliError::Runtime(format!("failed to serialize evidence record: {error}"))
        })?);
        body.push('\n');
    }
    fs::write(path, body)
        .map_err(|error| CliError::Runtime(format!("failed to write records.jsonl: {error}")))
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), CliError> {
    let body = serde_json::to_vec_pretty(value)
        .map_err(|error| CliError::Runtime(format!("failed to serialize JSON: {error}")))?;
    fs::write(path, body)
        .map_err(|error| CliError::Runtime(format!("failed to write {}: {error}", path.display())))
}

#[allow(clippy::too_many_arguments)]
fn build_summary(
    options: &WorkloadCorpusOptions,
    records: &[WorkloadRecord],
    generated_files: &[String],
    blockers: &[String],
    decision: &str,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    model_identity: &manifest::ArtifactIdentity,
    tokenizer_backend: &str,
    command: &str,
) -> Summary {
    let families = records
        .iter()
        .map(|record| record.family.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let target_context_tokens = records
        .iter()
        .map(|record| record.target_context_tokens)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let actual_context_tokens_min = records
        .iter()
        .map(|record| record.actual_context_tokens)
        .min()
        .unwrap_or(0);
    let actual_context_tokens_max = records
        .iter()
        .map(|record| record.actual_context_tokens)
        .max()
        .unwrap_or(0);
    Summary {
        schema_version: 1,
        goal: "XR00-real-workload-corpus".to_owned(),
        decision: decision.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        command: command.to_owned(),
        model_identity: model_identity.clone(),
        tokenizer_backend: tokenizer_backend.to_owned(),
        deterministic_seed_base: options.seed,
        workload_count: records.len(),
        families,
        target_context_tokens,
        actual_context_tokens_min,
        actual_context_tokens_max,
        generated_files: generated_files.to_vec(),
        blockers: blockers.to_vec(),
    }
}

fn render_report(records: &[WorkloadRecord], summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR00 Real-Context Workload Corpus Report\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Decision | `{}` |\n", summary.decision));
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!(
        "| Git status | `{}` |\n",
        markdown_escape(&summary.git_status_short)
    ));
    out.push_str(&format!(
        "| Tokenizer backend | `{}` |\n",
        markdown_escape(&summary.tokenizer_backend)
    ));
    out.push_str(&format!(
        "| Model local artifact SHA-256 | `{}` |\n",
        summary.model_identity.local_artifact_sha256
    ));
    out.push_str(&format!("| Workloads | `{}` |\n\n", records.len()));

    out.push_str("## Workloads\n\n");
    out.push_str("| Workload | Family | Target tokens | Actual tokens | Seed | Prompt |\n");
    out.push_str("|---|---|---:|---:|---:|---|\n");
    for record in records {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | {} | {} | `{}` |\n",
            markdown_escape(&record.workload_id),
            markdown_escape(&record.family),
            record.target_context_tokens,
            record.actual_context_tokens,
            record.deterministic_seed,
            markdown_escape(&record.prompt_path)
        ));
    }

    out.push_str("\n## Generated Files\n\n");
    for path in &summary.generated_files {
        out.push_str(&format!("- `{}`\n", markdown_escape(path)));
    }

    out.push_str("\n## Command\n\n```text\n");
    out.push_str(&summary.command);
    out.push_str("\n```\n");
    out
}

fn render_blockers(blockers: &[String], command: &str) -> String {
    let mut out = String::new();
    out.push_str("# XR00 Blockers\n\n");
    if blockers.is_empty() {
        out.push_str("No blockers recorded.\n\n");
    } else {
        for blocker in blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
        out.push('\n');
    }
    out.push_str("## Reproduce\n\n```text\n");
    out.push_str(command);
    out.push_str("\n```\n");
    out
}

fn render_decision(decision: &str, records: &[WorkloadRecord], blockers: &[String]) -> String {
    let mut out = String::new();
    out.push_str("# XR00 Decision\n\n");
    out.push_str(&format!("Decision: `{decision}`\n\n"));
    if blockers.is_empty() {
        out.push_str("The corpus is reproducible, repo-local, and covers all required workload families plus 1K, 4K, 8K, 16K, and 24K target contexts. Token lengths were measured with the local Gemma 4 tokenizer. No model execution or runtime optimization was performed.\n\n");
    } else {
        out.push_str(
            "The corpus generation produced blockers that must be resolved before acceptance.\n\n",
        );
    }
    out.push_str("## Coverage\n\n");
    out.push_str(&format!("- Workloads: {}\n", records.len()));
    out.push_str(&format!(
        "- Families: {}\n",
        records
            .iter()
            .map(|record| record.family.as_str())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push('\n');
    out
}

fn command_display(options: &WorkloadCorpusOptions) -> String {
    format!(
        "cargo run -p gemma4d-bench -- workload-corpus --model-path {} --workload-dir {} --out-dir {} --python {} --seed {}",
        shell_quote(&options.model_path.display().to_string()),
        shell_quote(&options.workload_dir.display().to_string()),
        shell_quote(&options.out_dir.display().to_string()),
        shell_quote(&options.python.display().to_string()),
        options.seed
    )
}

struct TokenizerCounter {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    backend: String,
}

impl TokenizerCounter {
    fn start(python: &Path, model_path: &Path) -> Result<Self, CliError> {
        let script = r#"
import json
import sys
from pathlib import Path
from mlx_lm.utils import load_tokenizer

try:
    tokenizer = load_tokenizer(Path(sys.argv[1]))
    print(json.dumps({"ok": True, "backend": "mlx_lm.utils.load_tokenizer", "tokenizer_class": type(tokenizer).__name__}, separators=(",", ":")), flush=True)
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")), flush=True)
    raise SystemExit(1)

for line in sys.stdin:
    request = json.loads(line)
    if request.get("cmd") == "shutdown":
        print(json.dumps({"ok": True}, separators=(",", ":")), flush=True)
        break
    text = request["text"]
    ids = tokenizer.encode(text)
    print(json.dumps({"ok": True, "count": len(ids)}, separators=(",", ":")), flush=True)
"#;
        let mut child = Command::new(python)
            .arg("-c")
            .arg(script)
            .arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                CliError::Runtime(format!(
                    "failed to spawn tokenizer helper {}: {error}",
                    python.display()
                ))
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| CliError::Runtime("tokenizer helper stdin unavailable".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CliError::Runtime("tokenizer helper stdout unavailable".to_owned()))?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|error| {
            CliError::Runtime(format!("failed to read tokenizer startup: {error}"))
        })?;
        let value = serde_json::from_str::<serde_json::Value>(line.trim()).map_err(|error| {
            CliError::Runtime(format!(
                "tokenizer startup emitted invalid JSON: {error}: {line}"
            ))
        })?;
        if !value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Err(CliError::Runtime(format!(
                "tokenizer helper failed to start: {}",
                value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown error")
            )));
        }
        let backend = format!(
            "{}:{}",
            value
                .get("backend")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown"),
            value
                .get("tokenizer_class")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
        );
        Ok(Self {
            child,
            stdin,
            stdout: reader,
            backend,
        })
    }

    fn backend(&self) -> &str {
        &self.backend
    }

    fn count_tokens(&mut self, text: &str) -> Result<usize, CliError> {
        let request = json!({"text": text});
        writeln!(self.stdin, "{request}").map_err(|error| {
            CliError::Runtime(format!("failed to write tokenizer request: {error}"))
        })?;
        self.stdin.flush().map_err(|error| {
            CliError::Runtime(format!("failed to flush tokenizer request: {error}"))
        })?;
        let mut line = String::new();
        self.stdout.read_line(&mut line).map_err(|error| {
            CliError::Runtime(format!("failed to read tokenizer response: {error}"))
        })?;
        let value = serde_json::from_str::<serde_json::Value>(line.trim()).map_err(|error| {
            CliError::Runtime(format!(
                "tokenizer response emitted invalid JSON: {error}: {line}"
            ))
        })?;
        if !value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Err(CliError::Runtime(format!(
                "tokenizer count failed: {}",
                value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown error")
            )));
        }
        value
            .get("count")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize)
            .ok_or_else(|| CliError::Runtime("tokenizer response missing count".to_owned()))
    }

    fn shutdown(mut self) {
        let _ = writeln!(self.stdin, "{}", json!({"cmd": "shutdown"}));
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("xr00-{}-{}", now.as_secs(), now.subsec_nanos())
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value.chars().all(|ch| {
            ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | '=' | ':' | ',')
        })
    {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_cover_required_families_and_targets() {
        let specs = workload_specs();
        let families = specs
            .iter()
            .map(|spec| spec.family)
            .collect::<BTreeSet<_>>();
        for family in REQUIRED_FAMILIES {
            assert!(families.contains(family), "missing family {family}");
        }
        let targets = specs
            .iter()
            .map(|spec| spec.target_context_tokens)
            .collect::<BTreeSet<_>>();
        for target in REQUIRED_TARGETS {
            assert!(targets.contains(target), "missing target {target}");
        }
        assert!(targets.iter().any(|target| *target >= 24_576));
    }

    #[test]
    fn validation_rejects_missing_family_and_zero_tokens() {
        let blockers = validate_records(&[WorkloadRecord {
            schema_version: 1,
            workload_id: "bad".to_owned(),
            family: "chat_short".to_owned(),
            source_files: vec!["AGENTS.md".to_owned()],
            prompt_path: "missing.txt".to_owned(),
            expected_output_style: "answer".to_owned(),
            max_new_tokens: 1,
            target_context_tokens: 1024,
            actual_context_tokens: 0,
            deterministic_seed: DEFAULT_SEED,
            prompt_sha256: "sha".to_owned(),
            tokenizer_backend: "test".to_owned(),
            notes: "test".to_owned(),
        }]);
        assert!(blockers.iter().any(|blocker| blocker.contains("zero")));
        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.contains("code_review_rust"))
        );
    }
}
