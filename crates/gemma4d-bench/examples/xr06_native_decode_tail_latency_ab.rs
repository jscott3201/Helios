use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{CliError, manifest, workload_corpus::WorkloadRecord};
use gemma4d_ffi::{DecodeProfileInfo, KvCache, KvPolicy, LoadConfig, Target, decode_one, prefill};
use gemma4d_tokenizer::sha256_hex;
use serde::Serialize;

const GOAL: &str = "XR06-native-decode-tail-latency-ab";
const MODE: &str = "native_decode_tail_latency_real_context_ab";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR06-native-decode-tail-latency-ab";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const DEFAULT_TRIALS: usize = 3;
const DEFAULT_MAX_NEW_TOKENS: usize = 64;
const STEADY_WARMUP_SAMPLES: usize = 4;
const LOGIT_TOLERANCE: f64 = 0.5;
const TAIL_IMPROVEMENT_GATE_PERCENT: f64 = 15.0;
const MAX_P50_REGRESSION_PERCENT: f64 = 5.0;
const MEMORY_CLIFF_GB: f64 = 14.0;
const TAIL_P95_TO_P50_RATIO: f64 = 1.25;
const TAIL_P99_TO_P50_RATIO: f64 = 1.25;
const TAIL_MAX_TO_P50_RATIO: f64 = 2.0;
const DEFAULT_WORKLOAD_IDS: &[&str] = &[
    "chat_short_1k_001",
    "tool_json_1k_001",
    "code_review_rust_4k_001",
    "benchmark_qa_4k_001",
    "code_review_rust_8k_001",
];
const ENV_KEYS: &[&str] = &[
    "GEMMA4D_REQUIRE_MLX",
    "GEMMA4D_USE_NATIVE_GRAPH",
    "GEMMA4D_NATIVE_DECODE_KV_EVAL",
    "GEMMA4D_NATIVE_DECODE_PROFILE",
    "GEMMA4D_EXPERIMENTAL_NATIVE_GATHER_GREEDY_LOGIT",
    "GEMMA4D_EXPERIMENTAL_NATIVE_SKIP_DECODE_PEAK_RESET",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse(env::args().skip(1))?;
    fs::create_dir_all(&options.out_dir)?;

    let records_path = options.out_dir.join("records.jsonl");
    let summary_path = options.out_dir.join("summary.json");
    let report_path = options.out_dir.join("report.md");
    let blockers_path = options.out_dir.join("blockers.md");
    let decision_path = options.out_dir.join("decision.md");
    let profile_json_path = options.out_dir.join("profile.json");
    let profile_report_path = options.out_dir.join("profile.md");

    let run_id = run_id();
    let git_sha =
        command_stdout("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let git_status_short =
        command_stdout("git", &["status", "--short"]).unwrap_or_else(|| "unknown".to_owned());
    let command = command_line();
    let environment = capture_environment();
    let model_identity =
        manifest::capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION");
    let variants = selected_variants(&options)?;
    let mut blockers = startup_blockers(&options, &variants);
    let workloads = select_workloads(load_workloads(&options.workloads_path)?, &options)?;
    let selected_workloads = workloads
        .iter()
        .map(SelectedWorkload::from)
        .collect::<Vec<_>>();
    let mut records = Vec::new();
    let mut tokenizer_backend = "not_started".to_owned();

    if blockers.is_empty() {
        let mut tokenizer = TokenizerHelper::start(&options.python, &options.model_path)?;
        tokenizer_backend = tokenizer.backend().to_owned();
        let workload_inputs = prepare_workload_inputs(&mut tokenizer, &workloads)?;
        for variant in &variants {
            for trial_index in 0..options.trials {
                run_variant_trial(
                    &options,
                    &run_id,
                    &git_sha,
                    &git_status_short,
                    &command,
                    &model_identity,
                    variant,
                    trial_index,
                    &workload_inputs,
                    &mut records,
                )?;
            }
        }
    }

    apply_correctness_gates(&mut records);
    let aggregates = build_aggregates(&records);
    let decode_profile = build_decode_profile(&records);
    let comparisons = build_comparisons(&aggregates, &variants);
    let failed_hypotheses = failed_hypotheses(&comparisons, &records);
    blockers.extend(blockers_for_records(&records, &variants));
    blockers.sort();
    blockers.dedup();
    let decision = decision_for(&blockers, &comparisons, &records);

    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        generated_at_unix_seconds: unix_now(),
        decision,
        status: if blockers.is_empty() {
            "completed".to_owned()
        } else {
            "blocked".to_owned()
        },
        run_id,
        git_sha,
        git_status_short,
        command,
        mode: MODE.to_owned(),
        environment,
        model_identity,
        tokenizer_backend,
        workloads_path: options.workloads_path.display().to_string(),
        out_dir: options.out_dir.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        decision_path: decision_path.display().to_string(),
        profile_json_path: profile_json_path.display().to_string(),
        profile_report_path: profile_report_path.display().to_string(),
        requested_trials: options.trials,
        max_new_tokens: options.max_new_tokens,
        steady_warmup_samples: STEADY_WARMUP_SAMPLES,
        logit_tolerance: LOGIT_TOLERANCE,
        tail_improvement_gate_percent: TAIL_IMPROVEMENT_GATE_PERCENT,
        max_p50_regression_percent: MAX_P50_REGRESSION_PERCENT,
        memory_cliff_gb: MEMORY_CLIFF_GB,
        tail_p95_to_p50_ratio: TAIL_P95_TO_P50_RATIO,
        tail_p99_to_p50_ratio: TAIL_P99_TO_P50_RATIO,
        tail_max_to_p50_ratio: TAIL_MAX_TO_P50_RATIO,
        variants,
        selected_workloads,
        record_count: records.len(),
        passed_records: records
            .iter()
            .filter(|record| record.status == "passed")
            .count(),
        failed_records: records
            .iter()
            .filter(|record| record.status != "passed")
            .count(),
        aggregates,
        decode_profile,
        comparisons,
        failed_hypotheses,
        blockers,
        generated_files: vec![
            records_path.display().to_string(),
            summary_path.display().to_string(),
            report_path.display().to_string(),
            blockers_path.display().to_string(),
            decision_path.display().to_string(),
            profile_json_path.display().to_string(),
            profile_report_path.display().to_string(),
        ],
        measurement_notes: vec![
            "all variants use the native graph with GEMMA4D_REQUIRE_MLX=1 and GEMMA4D_USE_NATIVE_GRAPH=1 before target load".to_owned(),
            "decode_token_traces records the committed input token, output greedy token, position before/after decode_one, latency, active KV bytes, peak MLX memory, eval-policy markers, and optional GEMMA4D_NATIVE_DECODE_PROFILE stage timings".to_owned(),
            "baseline is explicit native_decode_eval_per_layer; native_decode_runtime_default leaves GEMMA4D_NATIVE_DECODE_KV_EVAL unset".to_owned(),
            "steady decode statistics discard the first four decode_one samples when available, while raw p50/p95/p99 keep every decode sample".to_owned(),
            "acceptance requires candidate-wide correctness, memory below the cliff, reproduced baseline tail latency on the workload, p95 or p99 improvement, and no p50 regression over the goal gate".to_owned(),
            "decode profile forward_graph_ms is split into attention_kv_mutation_ms, deferred_kv_eval_ms, and derived non_kv_forward_graph_ms; rust_ffi_overhead_ms is host decode_one latency minus native total, clamped at zero".to_owned(),
        ],
    };

    write_jsonl(&records_path, &summary.records_for_jsonl())?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;
    fs::write(
        &profile_json_path,
        serde_json::to_vec_pretty(&summary.decode_profile)?,
    )?;
    fs::write(&profile_report_path, render_profile_report(&summary))?;

    println!("XR06 native decode tail-latency A/B: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());
    println!("profile: {}", profile_report_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR06 benchmark blocked; see blockers.md".into())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Options {
    out_dir: PathBuf,
    workloads_path: PathBuf,
    model_path: PathBuf,
    python: PathBuf,
    trials: usize,
    max_new_tokens: usize,
    workload_ids: Vec<String>,
    max_workloads: Option<usize>,
    variant_names: Vec<String>,
}

impl Options {
    fn parse<I, S>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut options = Self {
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
            workloads_path: PathBuf::from(DEFAULT_WORKLOADS),
            model_path: PathBuf::from(DEFAULT_MODEL),
            python: PathBuf::from(DEFAULT_PYTHON),
            trials: DEFAULT_TRIALS,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            workload_ids: DEFAULT_WORKLOAD_IDS
                .iter()
                .map(|workload_id| (*workload_id).to_owned())
                .collect(),
            max_workloads: None,
            variant_names: Vec::new(),
        };
        let mut args = args.into_iter().map(Into::into).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    options.out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?)
                }
                "--workloads" | "--workloads-path" => {
                    options.workloads_path =
                        PathBuf::from(required_value(&mut args, "--workloads")?)
                }
                "--model-path" => {
                    options.model_path = PathBuf::from(required_value(&mut args, "--model-path")?)
                }
                "--python" => {
                    options.python = PathBuf::from(required_value(&mut args, "--python")?)
                }
                "--trials" => {
                    options.trials =
                        parse_positive_usize(&required_value(&mut args, "--trials")?, "--trials")?
                }
                "--max-new-tokens" => {
                    options.max_new_tokens = parse_positive_usize(
                        &required_value(&mut args, "--max-new-tokens")?,
                        "--max-new-tokens",
                    )?
                }
                "--workload-id" => {
                    options
                        .workload_ids
                        .push(required_value(&mut args, "--workload-id")?);
                }
                "--clear-workload-ids" => options.workload_ids.clear(),
                "--max-workloads" => {
                    options.max_workloads = Some(parse_positive_usize(
                        &required_value(&mut args, "--max-workloads")?,
                        "--max-workloads",
                    )?);
                }
                "--variant" => options
                    .variant_names
                    .push(required_value(&mut args, "--variant")?),
                "--variants" => {
                    options
                        .variant_names
                        .extend(parse_csv(&required_value(&mut args, "--variants")?));
                }
                "-h" | "--help" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                other => return Err(CliError::Usage(format!("unknown option '{other}'"))),
            }
        }
        Ok(options)
    }
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: String,
    generated_at_unix_seconds: u64,
    decision: String,
    status: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    mode: String,
    environment: Environment,
    model_identity: manifest::ArtifactIdentity,
    tokenizer_backend: String,
    workloads_path: String,
    out_dir: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    profile_json_path: String,
    profile_report_path: String,
    requested_trials: usize,
    max_new_tokens: usize,
    steady_warmup_samples: usize,
    logit_tolerance: f64,
    tail_improvement_gate_percent: f64,
    max_p50_regression_percent: f64,
    memory_cliff_gb: f64,
    tail_p95_to_p50_ratio: f64,
    tail_p99_to_p50_ratio: f64,
    tail_max_to_p50_ratio: f64,
    variants: Vec<Variant>,
    selected_workloads: Vec<SelectedWorkload>,
    record_count: usize,
    passed_records: usize,
    failed_records: usize,
    aggregates: Vec<Aggregate>,
    decode_profile: DecodeProfileSummary,
    comparisons: Vec<Comparison>,
    failed_hypotheses: Vec<String>,
    blockers: Vec<String>,
    generated_files: Vec<String>,
    measurement_notes: Vec<String>,
}

impl Summary {
    fn records_for_jsonl(&self) -> Vec<&Record> {
        self.aggregates
            .iter()
            .flat_map(|aggregate| aggregate.records.iter())
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
struct Environment {
    machine: String,
    macos: String,
    rustc: String,
    cargo: String,
    mlx_version: String,
    hw_memsize_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct SelectedWorkload {
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    workload_max_new_tokens: usize,
    deterministic_seed: u64,
}

impl From<&WorkloadRecord> for SelectedWorkload {
    fn from(record: &WorkloadRecord) -> Self {
        Self {
            workload_id: record.workload_id.clone(),
            family: record.family.clone(),
            prompt_path: record.prompt_path.clone(),
            prompt_sha256: record.prompt_sha256.clone(),
            target_context_tokens: record.target_context_tokens,
            actual_context_tokens: record.actual_context_tokens,
            workload_max_new_tokens: record.max_new_tokens,
            deterministic_seed: record.deterministic_seed,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Variant {
    name: String,
    config: BTreeMap<String, String>,
    env: BTreeMap<String, String>,
    baseline_variant: Option<String>,
    immediate_layer_kv_eval_count: usize,
    grouped_end_kv_eval_count: usize,
    logits_sync_after_decode: bool,
}

#[derive(Debug, Clone)]
struct WorkloadInput {
    record: WorkloadRecord,
    prompt_sha256: String,
    token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    schema_version: u32,
    goal: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    deterministic_seed: u64,
    variant: String,
    backend: String,
    config: BTreeMap<String, String>,
    env: BTreeMap<String, String>,
    trial_index: usize,
    input_tokens: usize,
    generated_tokens: usize,
    model_load_ms: f64,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    prefill_tps: f64,
    decode_tps: f64,
    output_token_ids: Vec<i32>,
    output_logits: Vec<f64>,
    decode_token_latencies_ms: Vec<f64>,
    decode_token_traces: Vec<TokenTrace>,
    raw_decode_latency_stats: Option<LatencyStats>,
    steady_decode_latency_stats: Option<LatencyStats>,
    peak_mlx_gb: f64,
    rss_mb: f64,
    active_kv_bytes: u64,
    status: String,
    blocker: Option<String>,
    correctness: Correctness,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TokenTrace {
    decode_index: usize,
    input_token_id: i32,
    output_token_id: i32,
    position_before: u64,
    position_after: u64,
    latency_ms: f64,
    active_kv_bytes: u64,
    peak_mlx_gb: f64,
    rss_mb: f64,
    immediate_layer_kv_eval_count: usize,
    grouped_end_kv_eval_count: usize,
    logits_sync_after_decode: bool,
    mlx_synchronization_marker: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_profile: Option<DecodeProfileTrace>,
}

#[derive(Debug, Clone, Serialize)]
struct DecodeProfileTrace {
    reset_peak_memory_ms: f64,
    forward_graph_ms: f64,
    decode_embedding_ms: f64,
    layer_graph_ms: f64,
    attention_kv_mutation_ms: f64,
    deferred_kv_eval_ms: f64,
    lm_head_ms: f64,
    non_kv_forward_graph_ms: f64,
    greedy_select_ms: f64,
    target_top_k_ms: f64,
    eval_sync_ms: f64,
    hidden_view_ms: f64,
    output_read_ms: f64,
    peak_memory_read_ms: f64,
    total_native_decode_ms: f64,
    rust_ffi_overhead_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DecodeProfileSummary {
    schema_version: u32,
    profile_env_key: String,
    enabled_samples: usize,
    total_decode_samples: usize,
    aggregates: Vec<DecodeProfileAggregate>,
    stage_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DecodeProfileAggregate {
    variant: String,
    workload_id: String,
    sample_count: usize,
    enabled_sample_count: usize,
    latency_ms: Option<LatencyStats>,
    reset_peak_memory_ms: Option<LatencyStats>,
    forward_graph_ms: Option<LatencyStats>,
    decode_embedding_ms: Option<LatencyStats>,
    layer_graph_ms: Option<LatencyStats>,
    attention_kv_mutation_ms: Option<LatencyStats>,
    deferred_kv_eval_ms: Option<LatencyStats>,
    lm_head_ms: Option<LatencyStats>,
    non_kv_forward_graph_ms: Option<LatencyStats>,
    greedy_select_ms: Option<LatencyStats>,
    target_top_k_ms: Option<LatencyStats>,
    eval_sync_ms: Option<LatencyStats>,
    hidden_view_ms: Option<LatencyStats>,
    output_read_ms: Option<LatencyStats>,
    peak_memory_read_ms: Option<LatencyStats>,
    total_native_decode_ms: Option<LatencyStats>,
    rust_ffi_overhead_ms: Option<LatencyStats>,
    largest_stage_by_mean: Option<String>,
    largest_stage_mean_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Correctness {
    status: String,
    reference_variant: Option<String>,
    token_match: Option<bool>,
    logit_match: Option<bool>,
    max_logit_abs_delta: Option<f64>,
    logit_tolerance: Option<f64>,
    notes: Vec<String>,
}

impl Correctness {
    fn pending() -> Self {
        Self {
            status: "pending".to_owned(),
            reference_variant: None,
            token_match: None,
            logit_match: None,
            max_logit_abs_delta: None,
            logit_tolerance: Some(LOGIT_TOLERANCE),
            notes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct LatencyStats {
    count: usize,
    min_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    cv: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Aggregate {
    variant: String,
    backend: String,
    workload_id: String,
    family: String,
    trial_count: usize,
    passed_trials: usize,
    correctness_passed_trials: usize,
    raw_decode_p50_ms: Option<f64>,
    raw_decode_p95_ms: Option<f64>,
    raw_decode_p99_ms: Option<f64>,
    raw_decode_max_ms: Option<f64>,
    steady_decode_p50_ms: Option<f64>,
    steady_decode_p95_ms: Option<f64>,
    steady_decode_p99_ms: Option<f64>,
    decode_tps_p50: Option<f64>,
    peak_mlx_max_gb: Option<f64>,
    rss_max_mb: Option<f64>,
    active_kv_max_bytes: Option<u64>,
    baseline_tail_reproduced: bool,
    memory_gate_passed: bool,
    low_n: bool,
    records: Vec<Record>,
}

#[derive(Debug, Clone, Serialize)]
struct Comparison {
    candidate_variant: String,
    baseline_variant: String,
    backend: String,
    workload_id: String,
    family: String,
    baseline_tail_reproduced: bool,
    correctness_passed: bool,
    candidate_trials: usize,
    baseline_trials: usize,
    raw_p50_regression_percent: Option<f64>,
    raw_p95_improvement_percent: Option<f64>,
    raw_p99_improvement_percent: Option<f64>,
    steady_p50_regression_percent: Option<f64>,
    peak_mlx_delta_percent: Option<f64>,
    memory_gate_passed: bool,
    accepted: bool,
    reason: String,
}

fn selected_variants(options: &Options) -> Result<Vec<Variant>, CliError> {
    let mut variants = vec![
        native_variant("native_decode_eval_per_layer", "per_layer", None, 48, 0),
        native_variant(
            "native_decode_eval_end_of_decode",
            "end_of_decode",
            Some("native_decode_eval_per_layer"),
            0,
            48,
        ),
        native_default_variant(
            "native_decode_runtime_default",
            Some("native_decode_eval_per_layer"),
            0,
            48,
        ),
        native_variant(
            "native_decode_eval_selective_full_attention",
            "selective_full_attention",
            Some("native_decode_eval_per_layer"),
            8,
            40,
        ),
        native_variant(
            "native_decode_eval_defer_to_logits",
            "defer_to_logits",
            Some("native_decode_eval_per_layer"),
            0,
            0,
        ),
        native_variant_with_extra_env(
            "native_decode_gather_greedy_logit",
            "per_layer",
            Some("native_decode_eval_per_layer"),
            48,
            0,
            [("GEMMA4D_EXPERIMENTAL_NATIVE_GATHER_GREEDY_LOGIT", "1")],
        ),
        native_variant_with_extra_env(
            "native_decode_skip_peak_reset",
            "per_layer",
            Some("native_decode_eval_per_layer"),
            48,
            0,
            [("GEMMA4D_EXPERIMENTAL_NATIVE_SKIP_DECODE_PEAK_RESET", "1")],
        ),
    ];
    if !options.variant_names.is_empty() {
        let requested = options
            .variant_names
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        variants.retain(|variant| requested.contains(&variant.name));
        let found = variants
            .iter()
            .map(|variant| variant.name.clone())
            .collect::<BTreeSet<_>>();
        for requested_name in requested {
            if !found.contains(&requested_name) {
                return Err(CliError::Usage(format!(
                    "unknown --variant '{requested_name}'"
                )));
            }
        }
    }
    Ok(variants)
}

fn native_variant(
    name: &str,
    eval_policy: &str,
    baseline_variant: Option<&str>,
    immediate_layer_kv_eval_count: usize,
    grouped_end_kv_eval_count: usize,
) -> Variant {
    native_variant_with_extra_env(
        name,
        eval_policy,
        baseline_variant,
        immediate_layer_kv_eval_count,
        grouped_end_kv_eval_count,
        [],
    )
}

fn native_default_variant(
    name: &str,
    baseline_variant: Option<&str>,
    immediate_layer_kv_eval_count: usize,
    grouped_end_kv_eval_count: usize,
) -> Variant {
    let mut config = BTreeMap::new();
    config.insert("decode_kv_eval".to_owned(), "runtime_default".to_owned());
    config.insert(
        "immediate_layer_kv_eval_count".to_owned(),
        immediate_layer_kv_eval_count.to_string(),
    );
    config.insert(
        "grouped_end_kv_eval_count".to_owned(),
        grouped_end_kv_eval_count.to_string(),
    );
    config.insert("logits_sync_after_decode".to_owned(), "true".to_owned());
    let mut env = BTreeMap::new();
    env.insert("GEMMA4D_REQUIRE_MLX".to_owned(), "1".to_owned());
    env.insert("GEMMA4D_USE_NATIVE_GRAPH".to_owned(), "1".to_owned());
    if let Ok(value) = std::env::var("GEMMA4D_NATIVE_DECODE_PROFILE")
        && !value.is_empty()
    {
        config.insert("GEMMA4D_NATIVE_DECODE_PROFILE".to_owned(), value.clone());
        env.insert("GEMMA4D_NATIVE_DECODE_PROFILE".to_owned(), value);
    }
    Variant {
        name: name.to_owned(),
        config,
        env,
        baseline_variant: baseline_variant.map(str::to_owned),
        immediate_layer_kv_eval_count,
        grouped_end_kv_eval_count,
        logits_sync_after_decode: true,
    }
}

fn native_variant_with_extra_env<const N: usize>(
    name: &str,
    eval_policy: &str,
    baseline_variant: Option<&str>,
    immediate_layer_kv_eval_count: usize,
    grouped_end_kv_eval_count: usize,
    extra_env: [(&str, &str); N],
) -> Variant {
    let mut config = BTreeMap::new();
    config.insert("decode_kv_eval".to_owned(), eval_policy.to_owned());
    config.insert(
        "immediate_layer_kv_eval_count".to_owned(),
        immediate_layer_kv_eval_count.to_string(),
    );
    config.insert(
        "grouped_end_kv_eval_count".to_owned(),
        grouped_end_kv_eval_count.to_string(),
    );
    config.insert("logits_sync_after_decode".to_owned(), "true".to_owned());
    let mut env = BTreeMap::new();
    env.insert("GEMMA4D_REQUIRE_MLX".to_owned(), "1".to_owned());
    env.insert("GEMMA4D_USE_NATIVE_GRAPH".to_owned(), "1".to_owned());
    env.insert(
        "GEMMA4D_NATIVE_DECODE_KV_EVAL".to_owned(),
        eval_policy.to_owned(),
    );
    if let Ok(value) = std::env::var("GEMMA4D_NATIVE_DECODE_PROFILE")
        && !value.is_empty()
    {
        config.insert("GEMMA4D_NATIVE_DECODE_PROFILE".to_owned(), value.clone());
        env.insert("GEMMA4D_NATIVE_DECODE_PROFILE".to_owned(), value);
    }
    for (key, value) in extra_env {
        config.insert(key.to_owned(), value.to_owned());
        env.insert(key.to_owned(), value.to_owned());
    }
    Variant {
        name: name.to_owned(),
        config,
        env,
        baseline_variant: baseline_variant.map(str::to_owned),
        immediate_layer_kv_eval_count,
        grouped_end_kv_eval_count,
        logits_sync_after_decode: true,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_variant_trial(
    options: &Options,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    model_identity: &manifest::ArtifactIdentity,
    variant: &Variant,
    trial_index: usize,
    workload_inputs: &[WorkloadInput],
    records: &mut Vec<Record>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _env = EnvGuard::apply(&variant.env);
    let load_config = LoadConfig {
        model_path: options.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: std::num::NonZeroU32::new(32_768).expect("non-zero"),
        allow_unsupported_config: false,
    };

    let load_started = Instant::now();
    let target = Target::load(&load_config);
    let model_load = load_started.elapsed();
    let target = match target {
        Ok(target) => target,
        Err(error) => {
            for workload in workload_inputs {
                records.push(failed_record(
                    run_id,
                    git_sha,
                    git_status_short,
                    command,
                    model_identity,
                    variant,
                    trial_index,
                    workload,
                    model_load,
                    format!("target load failed: {error}"),
                ));
            }
            return Ok(());
        }
    };

    for workload in workload_inputs {
        records.push(run_decode_record(
            options,
            run_id,
            git_sha,
            git_status_short,
            command,
            model_identity,
            variant,
            trial_index,
            workload,
            model_load,
            &target,
        )?);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_decode_record(
    options: &Options,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    _model_identity: &manifest::ArtifactIdentity,
    variant: &Variant,
    trial_index: usize,
    workload: &WorkloadInput,
    model_load: Duration,
    target: &Target,
) -> Result<Record, Box<dyn std::error::Error>> {
    let total_started = Instant::now();
    let mut cache = KvCache::create(&KvPolicy::default())?;
    let prefill_started = Instant::now();
    let prefill_step = prefill(target, &mut cache, &workload.token_ids);
    let prefill_duration = prefill_started.elapsed();

    let mut step = match prefill_step {
        Ok(step) => step,
        Err(error) => {
            return Ok(failed_record(
                run_id,
                git_sha,
                git_status_short,
                command,
                _model_identity,
                variant,
                trial_index,
                workload,
                model_load,
                format!("prefill failed: {error}"),
            ));
        }
    };
    let mut peak_mlx_gb = f64::from(step.peak_memory_gb);
    let mut rss_mb = f64::from(step.peak_rss_mb);
    let mut active_kv_bytes = step.active_kv_bytes;
    let mut notes = Vec::new();

    let mut output_token_ids = Vec::with_capacity(options.max_new_tokens);
    let mut output_logits = Vec::with_capacity(options.max_new_tokens);
    let mut decode_token_latencies_ms =
        Vec::with_capacity(options.max_new_tokens.saturating_sub(1));
    let mut decode_token_traces = Vec::with_capacity(options.max_new_tokens.saturating_sub(1));
    let decode_started = Instant::now();
    for decode_index in 0..options.max_new_tokens {
        output_token_ids.push(step.greedy_token);
        output_logits.push(f64::from(step.greedy_logit));
        if decode_index + 1 >= options.max_new_tokens {
            break;
        }
        let input_token_id = step.greedy_token;
        let position_before = step.sequence_len;
        let token_started = Instant::now();
        match decode_one(target, &mut cache, input_token_id) {
            Ok(next_step) => {
                let latency_ms = duration_ms(token_started.elapsed());
                step = next_step;
                let decode_profile = decode_profile_trace(&step.decode_profile, latency_ms);
                peak_mlx_gb = peak_mlx_gb.max(f64::from(step.peak_memory_gb));
                rss_mb = rss_mb.max(f64::from(step.peak_rss_mb));
                active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
                decode_token_latencies_ms.push(latency_ms);
                decode_token_traces.push(TokenTrace {
                    decode_index,
                    input_token_id,
                    output_token_id: step.greedy_token,
                    position_before,
                    position_after: step.sequence_len,
                    latency_ms,
                    active_kv_bytes: step.active_kv_bytes,
                    peak_mlx_gb: f64::from(step.peak_memory_gb),
                    rss_mb: f64::from(step.peak_rss_mb),
                    immediate_layer_kv_eval_count: variant.immediate_layer_kv_eval_count,
                    grouped_end_kv_eval_count: variant.grouped_end_kv_eval_count,
                    logits_sync_after_decode: variant.logits_sync_after_decode,
                    mlx_synchronization_marker: "decode_one_step_result_eval".to_owned(),
                    decode_profile,
                });
            }
            Err(error) => {
                notes.push(format!(
                    "decode failed at generated index {decode_index}: {error}"
                ));
                break;
            }
        }
    }
    let decode_duration = decode_started.elapsed();
    let total_duration = total_started.elapsed();
    let generated_tokens = output_token_ids.len();
    let decode_ms = duration_ms(decode_duration);
    let prefill_ms = duration_ms(prefill_duration);
    let total_ms = duration_ms(total_duration);
    let status = if generated_tokens == options.max_new_tokens {
        "passed".to_owned()
    } else {
        "failed".to_owned()
    };
    let blocker = if status == "passed" {
        None
    } else {
        Some(format!(
            "generated {generated_tokens} of requested {} tokens",
            options.max_new_tokens
        ))
    };

    Ok(Record {
        schema_version: 1,
        goal: GOAL.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        command: command.to_owned(),
        workload_id: workload.record.workload_id.clone(),
        family: workload.record.family.clone(),
        prompt_path: workload.record.prompt_path.clone(),
        prompt_sha256: workload.prompt_sha256.clone(),
        target_context_tokens: workload.record.target_context_tokens,
        actual_context_tokens: workload.record.actual_context_tokens,
        deterministic_seed: workload.record.deterministic_seed,
        variant: variant.name.clone(),
        backend: "native".to_owned(),
        config: variant.config.clone(),
        env: variant.env.clone(),
        trial_index,
        input_tokens: workload.token_ids.len(),
        generated_tokens,
        model_load_ms: duration_ms(model_load),
        prefill_ms,
        ttft_ms: prefill_ms,
        decode_ms,
        total_ms,
        prefill_tps: if prefill_ms > 0.0 {
            workload.token_ids.len() as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        },
        decode_tps: if decode_ms > 0.0 {
            decode_token_traces.len() as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        },
        output_token_ids,
        output_logits,
        raw_decode_latency_stats: latency_stats(&decode_token_latencies_ms),
        steady_decode_latency_stats: steady_latency_stats(&decode_token_latencies_ms),
        decode_token_latencies_ms,
        decode_token_traces,
        peak_mlx_gb,
        rss_mb,
        active_kv_bytes,
        status,
        blocker,
        correctness: Correctness::pending(),
        notes,
    })
}

#[allow(clippy::too_many_arguments)]
fn failed_record(
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    _model_identity: &manifest::ArtifactIdentity,
    variant: &Variant,
    trial_index: usize,
    workload: &WorkloadInput,
    model_load: Duration,
    blocker: String,
) -> Record {
    Record {
        schema_version: 1,
        goal: GOAL.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        command: command.to_owned(),
        workload_id: workload.record.workload_id.clone(),
        family: workload.record.family.clone(),
        prompt_path: workload.record.prompt_path.clone(),
        prompt_sha256: workload.prompt_sha256.clone(),
        target_context_tokens: workload.record.target_context_tokens,
        actual_context_tokens: workload.record.actual_context_tokens,
        deterministic_seed: workload.record.deterministic_seed,
        variant: variant.name.clone(),
        backend: "native".to_owned(),
        config: variant.config.clone(),
        env: variant.env.clone(),
        trial_index,
        input_tokens: workload.token_ids.len(),
        generated_tokens: 0,
        model_load_ms: duration_ms(model_load),
        prefill_ms: 0.0,
        ttft_ms: 0.0,
        decode_ms: 0.0,
        total_ms: 0.0,
        prefill_tps: 0.0,
        decode_tps: 0.0,
        output_token_ids: Vec::new(),
        output_logits: Vec::new(),
        decode_token_latencies_ms: Vec::new(),
        decode_token_traces: Vec::new(),
        raw_decode_latency_stats: None,
        steady_decode_latency_stats: None,
        peak_mlx_gb: 0.0,
        rss_mb: 0.0,
        active_kv_bytes: 0,
        status: "failed".to_owned(),
        blocker: Some(blocker),
        correctness: Correctness {
            status: "failed".to_owned(),
            reference_variant: None,
            token_match: None,
            logit_match: None,
            max_logit_abs_delta: None,
            logit_tolerance: Some(LOGIT_TOLERANCE),
            notes: vec!["run failed before correctness comparison".to_owned()],
        },
        notes: Vec::new(),
    }
}

fn apply_correctness_gates(records: &mut [Record]) {
    let references = records
        .iter()
        .filter(|record| record.status == "passed")
        .filter(|record| record.variant == "native_decode_eval_per_layer")
        .map(|record| {
            (
                (record.workload_id.clone(), record.trial_index),
                (
                    record.variant.clone(),
                    record.output_token_ids.clone(),
                    record.output_logits.clone(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for record in records {
        if record.status != "passed" {
            continue;
        }
        let key = (record.workload_id.clone(), record.trial_index);
        let Some((reference_variant, reference_tokens, reference_logits)) = references.get(&key)
        else {
            record.correctness = Correctness {
                status: "missing_reference".to_owned(),
                reference_variant: Some("native_decode_eval_per_layer".to_owned()),
                token_match: None,
                logit_match: None,
                max_logit_abs_delta: None,
                logit_tolerance: Some(LOGIT_TOLERANCE),
                notes: vec!["same-workload/trial baseline record was unavailable".to_owned()],
            };
            continue;
        };
        let token_match = record.output_token_ids == *reference_tokens;
        let max_logit_abs_delta = max_logit_delta(&record.output_logits, reference_logits);
        let logit_match = max_logit_abs_delta
            .map(|delta| delta <= LOGIT_TOLERANCE)
            .unwrap_or(false);
        record.correctness = Correctness {
            status: if token_match && logit_match {
                "passed".to_owned()
            } else {
                "failed".to_owned()
            },
            reference_variant: Some(reference_variant.clone()),
            token_match: Some(token_match),
            logit_match: Some(logit_match),
            max_logit_abs_delta,
            logit_tolerance: Some(LOGIT_TOLERANCE),
            notes: Vec::new(),
        };
    }
}

fn build_aggregates(records: &[Record]) -> Vec<Aggregate> {
    let keys = records
        .iter()
        .map(|record| {
            (
                record.variant.clone(),
                record.backend.clone(),
                record.workload_id.clone(),
                record.family.clone(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut out = Vec::new();
    for (variant, backend, workload_id, family) in keys {
        let group = records
            .iter()
            .filter(|record| {
                record.variant == variant
                    && record.backend == backend
                    && record.workload_id == workload_id
            })
            .cloned()
            .collect::<Vec<_>>();
        let passed = group
            .iter()
            .filter(|record| record.status == "passed")
            .collect::<Vec<_>>();
        let raw_p50 = passed
            .iter()
            .filter_map(|record| {
                record
                    .raw_decode_latency_stats
                    .as_ref()
                    .map(|stats| stats.p50_ms)
            })
            .collect::<Vec<_>>();
        let raw_p95 = passed
            .iter()
            .filter_map(|record| {
                record
                    .raw_decode_latency_stats
                    .as_ref()
                    .map(|stats| stats.p95_ms)
            })
            .collect::<Vec<_>>();
        let raw_p99 = passed
            .iter()
            .filter_map(|record| {
                record
                    .raw_decode_latency_stats
                    .as_ref()
                    .map(|stats| stats.p99_ms)
            })
            .collect::<Vec<_>>();
        let raw_max = passed
            .iter()
            .filter_map(|record| {
                record
                    .raw_decode_latency_stats
                    .as_ref()
                    .map(|stats| stats.max_ms)
            })
            .collect::<Vec<_>>();
        let steady_p50 = passed
            .iter()
            .filter_map(|record| {
                record
                    .steady_decode_latency_stats
                    .as_ref()
                    .map(|stats| stats.p50_ms)
            })
            .collect::<Vec<_>>();
        let steady_p95 = passed
            .iter()
            .filter_map(|record| {
                record
                    .steady_decode_latency_stats
                    .as_ref()
                    .map(|stats| stats.p95_ms)
            })
            .collect::<Vec<_>>();
        let steady_p99 = passed
            .iter()
            .filter_map(|record| {
                record
                    .steady_decode_latency_stats
                    .as_ref()
                    .map(|stats| stats.p99_ms)
            })
            .collect::<Vec<_>>();
        let decode_tps = passed
            .iter()
            .map(|record| record.decode_tps)
            .collect::<Vec<_>>();
        let peak = passed
            .iter()
            .map(|record| record.peak_mlx_gb)
            .collect::<Vec<_>>();
        let rss = passed
            .iter()
            .map(|record| record.rss_mb)
            .collect::<Vec<_>>();
        let active_kv = passed
            .iter()
            .map(|record| record.active_kv_bytes)
            .collect::<Vec<_>>();
        let raw_decode_p50_ms = percentile(raw_p50, 0.50);
        let raw_decode_p95_ms = percentile(raw_p95, 0.50);
        let raw_decode_p99_ms = percentile(raw_p99, 0.50);
        let raw_decode_max_ms = percentile(raw_max, 0.50);
        let baseline_tail_reproduced = tail_reproduced(
            raw_decode_p50_ms,
            raw_decode_p95_ms,
            raw_decode_p99_ms,
            raw_decode_max_ms,
        );
        let peak_mlx_max_gb = max_value(&peak);
        out.push(Aggregate {
            variant,
            backend,
            workload_id,
            family,
            trial_count: group.len(),
            passed_trials: passed.len(),
            correctness_passed_trials: group
                .iter()
                .filter(|record| record.correctness.status == "passed")
                .count(),
            raw_decode_p50_ms,
            raw_decode_p95_ms,
            raw_decode_p99_ms,
            raw_decode_max_ms,
            steady_decode_p50_ms: percentile(steady_p50, 0.50),
            steady_decode_p95_ms: percentile(steady_p95, 0.50),
            steady_decode_p99_ms: percentile(steady_p99, 0.50),
            decode_tps_p50: percentile(decode_tps, 0.50),
            peak_mlx_max_gb,
            rss_max_mb: max_value(&rss),
            active_kv_max_bytes: active_kv.into_iter().max(),
            baseline_tail_reproduced,
            memory_gate_passed: peak_mlx_max_gb
                .map(|peak| peak < MEMORY_CLIFF_GB)
                .unwrap_or(false),
            low_n: passed.len() < 3,
            records: group,
        });
    }
    out
}

fn decode_profile_trace(
    profile: &DecodeProfileInfo,
    latency_ms: f64,
) -> Option<DecodeProfileTrace> {
    if !profile.enabled {
        return None;
    }
    let non_kv_forward_graph_ms =
        (profile.forward_graph_ms - profile.attention_kv_mutation_ms - profile.deferred_kv_eval_ms)
            .max(0.0);
    Some(DecodeProfileTrace {
        reset_peak_memory_ms: profile.reset_peak_memory_ms,
        forward_graph_ms: profile.forward_graph_ms,
        decode_embedding_ms: profile.decode_embedding_ms,
        layer_graph_ms: profile.layer_graph_ms,
        attention_kv_mutation_ms: profile.attention_kv_mutation_ms,
        deferred_kv_eval_ms: profile.deferred_kv_eval_ms,
        lm_head_ms: profile.lm_head_ms,
        non_kv_forward_graph_ms,
        greedy_select_ms: profile.greedy_select_ms,
        target_top_k_ms: profile.target_top_k_ms,
        eval_sync_ms: profile.eval_sync_ms,
        hidden_view_ms: profile.hidden_view_ms,
        output_read_ms: profile.output_read_ms,
        peak_memory_read_ms: profile.peak_memory_read_ms,
        total_native_decode_ms: profile.total_native_decode_ms,
        rust_ffi_overhead_ms: (latency_ms - profile.total_native_decode_ms).max(0.0),
    })
}

fn build_decode_profile(records: &[Record]) -> DecodeProfileSummary {
    let total_decode_samples = records
        .iter()
        .map(|record| record.decode_token_traces.len())
        .sum::<usize>();
    let enabled_samples = records
        .iter()
        .flat_map(|record| &record.decode_token_traces)
        .filter(|trace| trace.decode_profile.is_some())
        .count();
    let keys = records
        .iter()
        .map(|record| (record.variant.clone(), record.workload_id.clone()))
        .collect::<BTreeSet<_>>();
    let mut aggregates = Vec::new();
    for (variant, workload_id) in keys {
        let traces = records
            .iter()
            .filter(|record| record.variant == variant && record.workload_id == workload_id)
            .flat_map(|record| record.decode_token_traces.iter())
            .collect::<Vec<_>>();
        let profiles = traces
            .iter()
            .filter_map(|trace| trace.decode_profile.as_ref())
            .collect::<Vec<_>>();

        let reset_peak_memory_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.reset_peak_memory_ms
        }));
        let forward_graph_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.forward_graph_ms
        }));
        let decode_embedding_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.decode_embedding_ms
        }));
        let layer_graph_ms =
            latency_stats(&profile_values(&profiles, |profile| profile.layer_graph_ms));
        let attention_kv_mutation_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.attention_kv_mutation_ms
        }));
        let deferred_kv_eval_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.deferred_kv_eval_ms
        }));
        let lm_head_ms = latency_stats(&profile_values(&profiles, |profile| profile.lm_head_ms));
        let non_kv_forward_graph_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.non_kv_forward_graph_ms
        }));
        let greedy_select_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.greedy_select_ms
        }));
        let target_top_k_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.target_top_k_ms
        }));
        let eval_sync_ms =
            latency_stats(&profile_values(&profiles, |profile| profile.eval_sync_ms));
        let hidden_view_ms =
            latency_stats(&profile_values(&profiles, |profile| profile.hidden_view_ms));
        let output_read_ms =
            latency_stats(&profile_values(&profiles, |profile| profile.output_read_ms));
        let peak_memory_read_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.peak_memory_read_ms
        }));
        let total_native_decode_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.total_native_decode_ms
        }));
        let rust_ffi_overhead_ms = latency_stats(&profile_values(&profiles, |profile| {
            profile.rust_ffi_overhead_ms
        }));
        let (largest_stage_by_mean, largest_stage_mean_ms) = largest_profile_stage_by_mean([
            ("reset_peak_memory_ms", &reset_peak_memory_ms),
            ("non_kv_forward_graph_ms", &non_kv_forward_graph_ms),
            ("attention_kv_mutation_ms", &attention_kv_mutation_ms),
            ("deferred_kv_eval_ms", &deferred_kv_eval_ms),
            ("lm_head_ms", &lm_head_ms),
            ("decode_embedding_ms", &decode_embedding_ms),
            ("greedy_select_ms", &greedy_select_ms),
            ("target_top_k_ms", &target_top_k_ms),
            ("eval_sync_ms", &eval_sync_ms),
            ("hidden_view_ms", &hidden_view_ms),
            ("output_read_ms", &output_read_ms),
            ("peak_memory_read_ms", &peak_memory_read_ms),
            ("rust_ffi_overhead_ms", &rust_ffi_overhead_ms),
        ]);

        aggregates.push(DecodeProfileAggregate {
            variant,
            workload_id,
            sample_count: traces.len(),
            enabled_sample_count: profiles.len(),
            latency_ms: latency_stats(
                &traces
                    .iter()
                    .map(|trace| trace.latency_ms)
                    .collect::<Vec<_>>(),
            ),
            reset_peak_memory_ms,
            forward_graph_ms,
            decode_embedding_ms,
            layer_graph_ms,
            attention_kv_mutation_ms,
            deferred_kv_eval_ms,
            lm_head_ms,
            non_kv_forward_graph_ms,
            greedy_select_ms,
            target_top_k_ms,
            eval_sync_ms,
            hidden_view_ms,
            output_read_ms,
            peak_memory_read_ms,
            total_native_decode_ms,
            rust_ffi_overhead_ms,
            largest_stage_by_mean,
            largest_stage_mean_ms,
        });
    }

    DecodeProfileSummary {
        schema_version: 1,
        profile_env_key: "GEMMA4D_NATIVE_DECODE_PROFILE".to_owned(),
        enabled_samples,
        total_decode_samples,
        aggregates,
        stage_notes: vec![
            "forward_graph_ms is the parent native decode_last_logits bucket and should not be treated as a patch lane by itself".to_owned(),
            "attention_kv_mutation_ms accumulates per-layer decode K/V projection, append/concat, sliding-window slice, target KV store, optional immediate KV eval, and shared-KV capture before attention".to_owned(),
            "non_kv_forward_graph_ms is derived as forward_graph_ms - attention_kv_mutation_ms - deferred_kv_eval_ms and represents the remaining graph lane: embedding, non-KV layer work, final norm/LM head, and small native book-keeping".to_owned(),
            "layer_graph_ms includes the full per-layer decode loop, including attention KV mutation and non-KV attention/MLP work".to_owned(),
            "deferred_kv_eval_ms is the grouped end-of-decode KV eval path when the selected KV eval mode defers layer KV synchronization".to_owned(),
            "eval_sync_ms is the explicit MLX eval call on greedy token and max-logit arrays".to_owned(),
            "output_read_ms covers scalar item reads for greedy token and greedy logit".to_owned(),
            "rust_ffi_overhead_ms is host decode_one latency minus total_native_decode_ms, clamped at zero".to_owned(),
            "profile timings are emitted only when GEMMA4D_NATIVE_DECODE_PROFILE is enabled before native target load".to_owned(),
        ],
    }
}

fn profile_values<F>(profiles: &[&DecodeProfileTrace], field: F) -> Vec<f64>
where
    F: Fn(&DecodeProfileTrace) -> f64,
{
    profiles.iter().map(|profile| field(profile)).collect()
}

fn largest_profile_stage_by_mean<const N: usize>(
    stages: [(&str, &Option<LatencyStats>); N],
) -> (Option<String>, Option<f64>) {
    stages
        .into_iter()
        .filter_map(|(name, stats)| stats.as_ref().map(|stats| (name, stats.mean_ms)))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(name, mean)| (Some(name.to_owned()), Some(mean)))
        .unwrap_or((None, None))
}

fn build_comparisons(aggregates: &[Aggregate], variants: &[Variant]) -> Vec<Comparison> {
    let mut by_key = BTreeMap::new();
    for aggregate in aggregates {
        by_key.insert(
            (aggregate.variant.clone(), aggregate.workload_id.clone()),
            aggregate,
        );
    }
    let mut variant_correctness_clean = BTreeMap::new();
    for variant in variants {
        let variant_aggregates = aggregates
            .iter()
            .filter(|aggregate| aggregate.variant == variant.name)
            .collect::<Vec<_>>();
        let clean = !variant_aggregates.is_empty()
            && variant_aggregates.iter().all(|aggregate| {
                aggregate.trial_count > 0
                    && aggregate.passed_trials == aggregate.trial_count
                    && aggregate.correctness_passed_trials == aggregate.trial_count
                    && aggregate.memory_gate_passed
            });
        variant_correctness_clean.insert(variant.name.clone(), clean);
    }

    let mut out = Vec::new();
    for variant in variants {
        let Some(baseline_variant) = &variant.baseline_variant else {
            continue;
        };
        for candidate in aggregates
            .iter()
            .filter(|aggregate| aggregate.variant == variant.name)
        {
            let Some(baseline) =
                by_key.get(&(baseline_variant.clone(), candidate.workload_id.clone()))
            else {
                out.push(Comparison {
                    candidate_variant: candidate.variant.clone(),
                    baseline_variant: baseline_variant.clone(),
                    backend: candidate.backend.clone(),
                    workload_id: candidate.workload_id.clone(),
                    family: candidate.family.clone(),
                    baseline_tail_reproduced: false,
                    correctness_passed: false,
                    candidate_trials: candidate.passed_trials,
                    baseline_trials: 0,
                    raw_p50_regression_percent: None,
                    raw_p95_improvement_percent: None,
                    raw_p99_improvement_percent: None,
                    steady_p50_regression_percent: None,
                    peak_mlx_delta_percent: None,
                    memory_gate_passed: false,
                    accepted: false,
                    reason: "missing baseline aggregate".to_owned(),
                });
                continue;
            };

            let raw_p50_regression =
                percent_regression(baseline.raw_decode_p50_ms, candidate.raw_decode_p50_ms);
            let raw_p95_improvement =
                percent_improvement(baseline.raw_decode_p95_ms, candidate.raw_decode_p95_ms);
            let raw_p99_improvement =
                percent_improvement(baseline.raw_decode_p99_ms, candidate.raw_decode_p99_ms);
            let steady_p50_regression = percent_regression(
                baseline.steady_decode_p50_ms,
                candidate.steady_decode_p50_ms,
            );
            let peak_delta =
                percent_improvement(baseline.peak_mlx_max_gb, candidate.peak_mlx_max_gb);
            let correctness_passed = candidate.passed_trials > 0
                && baseline.passed_trials > 0
                && candidate.correctness_passed_trials == candidate.trial_count;
            let candidate_variant_clean = *variant_correctness_clean
                .get(&candidate.variant)
                .unwrap_or(&false);
            let enough_trials = candidate.passed_trials >= 3 && baseline.passed_trials >= 3;
            let p50_ok = raw_p50_regression
                .map(|regression| regression <= MAX_P50_REGRESSION_PERCENT)
                .unwrap_or(false)
                && steady_p50_regression
                    .map(|regression| regression <= MAX_P50_REGRESSION_PERCENT)
                    .unwrap_or(false);
            let tail_gate = raw_p95_improvement
                .map(|delta| delta >= TAIL_IMPROVEMENT_GATE_PERCENT)
                .unwrap_or(false)
                || raw_p99_improvement
                    .map(|delta| delta >= TAIL_IMPROVEMENT_GATE_PERCENT)
                    .unwrap_or(false);
            let memory_gate_passed = candidate.memory_gate_passed;
            let accepted = correctness_passed
                && candidate_variant_clean
                && enough_trials
                && baseline.baseline_tail_reproduced
                && memory_gate_passed
                && p50_ok
                && tail_gate;
            let reason = if !correctness_passed {
                "correctness gate failed or records missing".to_owned()
            } else if !candidate_variant_clean {
                "candidate variant had correctness or memory regression on another selected workload"
                    .to_owned()
            } else if !enough_trials {
                "fewer than three passed trials for candidate or baseline".to_owned()
            } else if !baseline.baseline_tail_reproduced {
                "baseline did not reproduce XR06 tail-latency spike on this workload".to_owned()
            } else if !memory_gate_passed {
                "candidate memory gate failed".to_owned()
            } else if !p50_ok {
                "candidate p50 regression exceeded gate".to_owned()
            } else if accepted {
                "accepted by XR06 p95/p99 tail-latency gate".to_owned()
            } else {
                "no XR06 p95/p99 tail-latency gate met".to_owned()
            };
            out.push(Comparison {
                candidate_variant: candidate.variant.clone(),
                baseline_variant: baseline.variant.clone(),
                backend: candidate.backend.clone(),
                workload_id: candidate.workload_id.clone(),
                family: candidate.family.clone(),
                baseline_tail_reproduced: baseline.baseline_tail_reproduced,
                correctness_passed,
                candidate_trials: candidate.passed_trials,
                baseline_trials: baseline.passed_trials,
                raw_p50_regression_percent: raw_p50_regression,
                raw_p95_improvement_percent: raw_p95_improvement,
                raw_p99_improvement_percent: raw_p99_improvement,
                steady_p50_regression_percent: steady_p50_regression,
                peak_mlx_delta_percent: peak_delta,
                memory_gate_passed,
                accepted,
                reason,
            });
        }
    }
    out
}

fn failed_hypotheses(comparisons: &[Comparison], records: &[Record]) -> Vec<String> {
    let mut out = Vec::new();
    for comparison in comparisons {
        if !comparison.accepted {
            out.push(format!(
                "{} vs {} on {}: {}",
                comparison.candidate_variant,
                comparison.baseline_variant,
                comparison.workload_id,
                comparison.reason
            ));
        }
    }
    for record in records {
        if record.status != "passed" {
            out.push(format!(
                "{} {} trial {} failed: {}",
                record.variant,
                record.workload_id,
                record.trial_index,
                record
                    .blocker
                    .as_deref()
                    .unwrap_or("no blocker detail recorded")
            ));
        } else if record.correctness.status != "passed" {
            out.push(format!(
                "{} {} trial {} correctness failed against {:?}",
                record.variant,
                record.workload_id,
                record.trial_index,
                record.correctness.reference_variant
            ));
        }
    }
    out.sort();
    out.dedup();
    out
}

fn blockers_for_records(records: &[Record], variants: &[Variant]) -> Vec<String> {
    let mut blockers = Vec::new();
    for baseline in variants
        .iter()
        .filter(|variant| variant.baseline_variant.is_none())
    {
        let baseline_records = records
            .iter()
            .filter(|record| record.variant == baseline.name)
            .collect::<Vec<_>>();
        if baseline_records.is_empty() {
            blockers.push(format!(
                "missing baseline variant records for {}",
                baseline.name
            ));
        }
        for record in baseline_records {
            if record.status != "passed" {
                blockers.push(format!(
                    "baseline {} {} trial {} failed: {}",
                    record.variant,
                    record.workload_id,
                    record.trial_index,
                    record
                        .blocker
                        .as_deref()
                        .unwrap_or("no blocker detail recorded")
                ));
            } else if record
                .peak_mlx_gb
                .partial_cmp(&MEMORY_CLIFF_GB)
                .is_some_and(|ordering| ordering.is_ge())
            {
                blockers.push(format!(
                    "baseline {} {} trial {} crossed memory cliff {:.3} GB >= {:.1} GB",
                    record.variant,
                    record.workload_id,
                    record.trial_index,
                    record.peak_mlx_gb,
                    MEMORY_CLIFF_GB
                ));
            }
        }
    }
    blockers
}

fn decision_for(blockers: &[String], comparisons: &[Comparison], records: &[Record]) -> String {
    if !blockers.is_empty() || records.is_empty() {
        "blocked_with_evidence".to_owned()
    } else if comparisons.iter().any(|comparison| comparison.accepted) {
        "accept_candidate".to_owned()
    } else if comparisons
        .iter()
        .all(|comparison| !comparison.baseline_tail_reproduced)
    {
        "needs_more_data".to_owned()
    } else {
        "reject_candidate".to_owned()
    }
}

struct EnvGuard {
    previous: Vec<(String, Option<String>)>,
}

impl EnvGuard {
    fn apply(env: &BTreeMap<String, String>) -> Self {
        let previous = ENV_KEYS
            .iter()
            .map(|key| ((*key).to_owned(), std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in ENV_KEYS {
            // SAFETY: the XR06 benchmark is single-threaded and mutates process
            // environment only around native target loads to isolate variants.
            unsafe {
                std::env::remove_var(key);
            }
        }
        for (key, value) in env {
            // SAFETY: see the single-threaded benchmark note above.
            unsafe {
                std::env::set_var(key, value);
            }
        }
        Self { previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.previous {
            // SAFETY: the benchmark remains single-threaded while restoring env.
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}

struct TokenizerHelper {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    backend: String,
}

impl TokenizerHelper {
    fn start(python: &Path, model_path: &Path) -> Result<Self, CliError> {
        let script = r#"
import json
import sys
from pathlib import Path
from mlx_lm.utils import load_tokenizer
tokenizer = load_tokenizer(Path(sys.argv[1]))
print(json.dumps({"ok": True, "backend": "mlx_lm_tokenizer", "tokenizer_class": tokenizer.__class__.__name__}, separators=(",", ":")), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    cmd = request.get("cmd")
    if cmd == "shutdown":
        break
    if cmd == "encode":
        print(json.dumps({"ok": True, "ids": tokenizer.encode(request["text"])}, separators=(",", ":")), flush=True)
    else:
        print(json.dumps({"ok": False, "error": "unknown command"}, separators=(",", ":")), flush=True)
"#;
        let mut child = Command::new(python)
            .arg("-c")
            .arg(script)
            .arg(model_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
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
            .ok_or_else(|| CliError::Runtime("tokenizer stdin unavailable".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CliError::Runtime("tokenizer stdout unavailable".to_owned()))?;
        let mut stdout = BufReader::new(stdout);
        let mut line = String::new();
        stdout.read_line(&mut line).map_err(|error| {
            CliError::Runtime(format!("tokenizer startup read failed: {error}"))
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
                "tokenizer helper failed: {}",
                value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
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
            stdout,
            backend,
        })
    }

    fn backend(&self) -> &str {
        &self.backend
    }

    fn encode(&mut self, text: &str) -> Result<Vec<i32>, CliError> {
        serde_json::to_writer(
            &mut self.stdin,
            &serde_json::json!({"cmd":"encode","text":text}),
        )
        .map_err(|error| {
            CliError::Runtime(format!("failed to write tokenizer request JSON: {error}"))
        })?;
        writeln!(self.stdin).map_err(|error| {
            CliError::Runtime(format!("failed to write tokenizer newline: {error}"))
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
                "tokenizer request failed: {}",
                value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
            )));
        }
        let ids = value
            .get("ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| CliError::Runtime("tokenizer encode response missing ids".to_owned()))?;
        ids.iter()
            .map(|id| {
                let raw = id.as_i64().ok_or_else(|| {
                    CliError::Runtime(
                        "tokenizer encode response contained non-integer id".to_owned(),
                    )
                })?;
                i32::try_from(raw).map_err(|error| {
                    CliError::Runtime(format!("tokenizer id did not fit i32: {error}"))
                })
            })
            .collect()
    }
}

impl Drop for TokenizerHelper {
    fn drop(&mut self) {
        let _ = serde_json::to_writer(&mut self.stdin, &serde_json::json!({"cmd":"shutdown"}));
        let _ = writeln!(self.stdin);
        let _ = self.stdin.flush();
        let _ = self.child.try_wait();
    }
}

fn prepare_workload_inputs(
    tokenizer: &mut TokenizerHelper,
    workloads: &[WorkloadRecord],
) -> Result<Vec<WorkloadInput>, CliError> {
    let mut out = Vec::with_capacity(workloads.len());
    for workload in workloads {
        let prompt = fs::read_to_string(&workload.prompt_path).map_err(|error| {
            CliError::Runtime(format!(
                "failed to read prompt {}: {error}",
                workload.prompt_path
            ))
        })?;
        let prompt_sha256 = sha256_hex(prompt.as_bytes());
        if prompt_sha256 != workload.prompt_sha256 {
            return Err(CliError::Runtime(format!(
                "{} prompt sha mismatch: manifest={} actual={}",
                workload.workload_id, workload.prompt_sha256, prompt_sha256
            )));
        }
        let token_ids = tokenizer.encode(&prompt)?;
        if token_ids.len() != workload.actual_context_tokens {
            return Err(CliError::Runtime(format!(
                "{} tokenizer length mismatch: manifest={} actual={}",
                workload.workload_id,
                workload.actual_context_tokens,
                token_ids.len()
            )));
        }
        out.push(WorkloadInput {
            record: workload.clone(),
            prompt_sha256,
            token_ids,
        });
    }
    Ok(out)
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR06 Native Decode Tail-Latency A/B Report\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Summary\n\n");
    out.push_str(&format!(
        "- Records: `{}` passed `{}` failed `{}`\n",
        summary.record_count, summary.passed_records, summary.failed_records
    ));
    out.push_str(&format!(
        "- Trials requested: `{}`; generated tokens per run: `{}`; steady warmup samples discarded: `{}`\n",
        summary.requested_trials, summary.max_new_tokens, summary.steady_warmup_samples
    ));
    out.push_str(&format!(
        "- Tail gate: p95 or p99 improvement `>= {:.1}%`, p50 regression `<= {:.1}%`, memory `< {:.1} GB`\n\n",
        summary.tail_improvement_gate_percent,
        summary.max_p50_regression_percent,
        summary.memory_cliff_gb
    ));
    out.push_str(&format!(
        "- Decode profile samples: `{}` enabled of `{}` decode samples\n\n",
        summary.decode_profile.enabled_samples, summary.decode_profile.total_decode_samples
    ));

    out.push_str("## Aggregates\n\n");
    out.push_str("| Workload | Variant | Trials | Correct | Raw p50 | Raw p95 | Raw p99 | Raw max | Steady p50 | Peak MLX GB | Tail Reproduced | Low N |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for aggregate in &summary.aggregates {
        out.push_str(&format!(
            "| `{}` | `{}` | {}/{} | {} | {} | {} | {} | {} | {} | {} | `{}` | `{}` |\n",
            aggregate.workload_id,
            aggregate.variant,
            aggregate.passed_trials,
            aggregate.trial_count,
            aggregate.correctness_passed_trials,
            fmt_opt(aggregate.raw_decode_p50_ms),
            fmt_opt(aggregate.raw_decode_p95_ms),
            fmt_opt(aggregate.raw_decode_p99_ms),
            fmt_opt(aggregate.raw_decode_max_ms),
            fmt_opt(aggregate.steady_decode_p50_ms),
            fmt_opt(aggregate.peak_mlx_max_gb),
            aggregate.baseline_tail_reproduced,
            aggregate.low_n
        ));
    }

    out.push_str("\n## Comparisons\n\n");
    out.push_str("| Workload | Candidate | Baseline | Tail Reproduced | Correct | p50 reg % | p95 imp % | p99 imp % | Peak MLX imp % | Accepted | Reason |\n");
    out.push_str("|---|---|---|---|---|---:|---:|---:|---:|---|---|\n");
    for comparison in &summary.comparisons {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | {} | {} | {} | `{}` | {} |\n",
            comparison.workload_id,
            comparison.candidate_variant,
            comparison.baseline_variant,
            comparison.baseline_tail_reproduced,
            comparison.correctness_passed,
            fmt_opt(comparison.raw_p50_regression_percent),
            fmt_opt(comparison.raw_p95_improvement_percent),
            fmt_opt(comparison.raw_p99_improvement_percent),
            fmt_opt(comparison.peak_mlx_delta_percent),
            comparison.accepted,
            comparison.reason
        ));
    }
    out
}

fn render_profile_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR06 Native Decode Stage Profile\n\n");
    out.push_str(&format!(
        "- Profile env: `{}`\n",
        summary.decode_profile.profile_env_key
    ));
    out.push_str(&format!(
        "- Enabled samples: `{}` / `{}`\n",
        summary.decode_profile.enabled_samples, summary.decode_profile.total_decode_samples
    ));
    out.push_str(&format!("- Records: `{}`\n\n", summary.record_count));

    out.push_str("## Stage Aggregates\n\n");
    out.push_str("| Workload | Variant | Samples | Host latency mean | Native total mean | Forward graph mean | Non-KV graph mean | KV mutation mean | Deferred KV eval mean | LM head mean | Eval sync mean | Greedy select mean | Rust/FFI overhead mean | Largest stage |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for aggregate in &summary.decode_profile.aggregates {
        out.push_str(&format!(
            "| `{}` | `{}` | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` |\n",
            aggregate.workload_id,
            aggregate.variant,
            aggregate.enabled_sample_count,
            aggregate.sample_count,
            fmt_stats_mean(&aggregate.latency_ms),
            fmt_stats_mean(&aggregate.total_native_decode_ms),
            fmt_stats_mean(&aggregate.forward_graph_ms),
            fmt_stats_mean(&aggregate.non_kv_forward_graph_ms),
            fmt_stats_mean(&aggregate.attention_kv_mutation_ms),
            fmt_stats_mean(&aggregate.deferred_kv_eval_ms),
            fmt_stats_mean(&aggregate.lm_head_ms),
            fmt_stats_mean(&aggregate.eval_sync_ms),
            fmt_stats_mean(&aggregate.greedy_select_ms),
            fmt_stats_mean(&aggregate.rust_ffi_overhead_ms),
            aggregate.largest_stage_by_mean.as_deref().unwrap_or("n/a")
        ));
    }

    out.push_str("\n## Notes\n\n");
    for note in &summary.decode_profile.stage_notes {
        out.push_str(&format!("- {note}\n"));
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR06 Native Decode Tail-Latency A/B Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No blockers recorded.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out.push_str("\n## Failed Hypotheses\n\n");
    if summary.failed_hypotheses.is_empty() {
        out.push_str("No failed hypotheses recorded.\n");
    } else {
        for hypothesis in &summary.failed_hypotheses {
            out.push_str(&format!("- {hypothesis}\n"));
        }
    }
    out
}

fn render_decision(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR06 Native Decode Tail-Latency A/B Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Evidence\n\n");
    for path in &summary.generated_files {
        out.push_str(&format!("- `{path}`\n"));
    }
    out.push_str("\n## Accepted Comparisons\n\n");
    let accepted = summary
        .comparisons
        .iter()
        .filter(|comparison| comparison.accepted)
        .collect::<Vec<_>>();
    if accepted.is_empty() {
        out.push_str("No comparisons met the XR06 acceptance gate.\n");
    } else {
        for comparison in accepted {
            out.push_str(&format!(
                "- `{}` vs `{}` on `{}`: p95 improvement {}, p99 improvement {}, p50 regression {}\n",
                comparison.candidate_variant,
                comparison.baseline_variant,
                comparison.workload_id,
                fmt_opt(comparison.raw_p95_improvement_percent),
                fmt_opt(comparison.raw_p99_improvement_percent),
                fmt_opt(comparison.raw_p50_regression_percent)
            ));
        }
    }
    out.push_str("\n## Claim Boundary\n\n");
    out.push_str("- XR06 is native decode tail-latency evidence only; it does not change defaults unless a later goal adopts a policy.\n");
    out.push_str("- Candidate acceptance requires no correctness or memory regression on any selected workload/trial for that variant.\n");
    out.push_str("- Layer/eval markers are policy markers, not per-layer timing probes.\n");
    out
}

fn load_workloads(path: &Path) -> Result<Vec<WorkloadRecord>, CliError> {
    let text = fs::read_to_string(path)
        .map_err(|error| CliError::Runtime(format!("failed to read workloads JSONL: {error}")))?;
    let mut workloads = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        workloads.push(
            serde_json::from_str::<WorkloadRecord>(line).map_err(|error| {
                CliError::Runtime(format!(
                    "failed to parse workload JSONL line {}: {error}",
                    line_index + 1
                ))
            })?,
        );
    }
    Ok(workloads)
}

fn select_workloads(
    workloads: Vec<WorkloadRecord>,
    options: &Options,
) -> Result<Vec<WorkloadRecord>, CliError> {
    let requested = options
        .workload_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut selected = workloads
        .into_iter()
        .filter(|workload| requested.is_empty() || requested.contains(&workload.workload_id))
        .collect::<Vec<_>>();
    if let Some(max_workloads) = options.max_workloads {
        selected.truncate(max_workloads);
    }
    if selected.is_empty() {
        return Err(CliError::Usage(
            "no workloads selected for XR06 benchmark".to_owned(),
        ));
    }
    Ok(selected)
}

fn startup_blockers(options: &Options, variants: &[Variant]) -> Vec<String> {
    let mut blockers = Vec::new();
    if !options.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            options.model_path.display()
        ));
    }
    if !options.workloads_path.exists() {
        blockers.push(format!(
            "workloads path does not exist: {}",
            options.workloads_path.display()
        ));
    }
    if variants.is_empty() {
        blockers.push("no variants selected".to_owned());
    }
    blockers
}

fn write_jsonl(path: &Path, records: &[&Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        writeln!(file)?;
    }
    Ok(())
}

fn latency_stats(values: &[f64]) -> Option<LatencyStats> {
    if values.is_empty() {
        return None;
    }
    let values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let min_ms = min_value(&values)?;
    let max_ms = max_value(&values)?;
    let mean_ms = values.iter().sum::<f64>() / values.len() as f64;
    Some(LatencyStats {
        count: values.len(),
        min_ms,
        p50_ms: percentile(values.clone(), 0.50)?,
        p95_ms: percentile(values.clone(), 0.95)?,
        p99_ms: percentile(values.clone(), 0.99)?,
        max_ms,
        mean_ms,
        cv: coefficient_of_variation(&values),
    })
}

fn steady_latency_stats(values: &[f64]) -> Option<LatencyStats> {
    if values.len() <= STEADY_WARMUP_SAMPLES {
        return None;
    }
    latency_stats(&values[STEADY_WARMUP_SAMPLES..])
}

fn tail_reproduced(p50: Option<f64>, p95: Option<f64>, p99: Option<f64>, max: Option<f64>) -> bool {
    match p50 {
        Some(p50) if p50 > 0.0 => {
            p95.map(|value| value >= p50 * TAIL_P95_TO_P50_RATIO)
                .unwrap_or(false)
                || p99
                    .map(|value| value >= p50 * TAIL_P99_TO_P50_RATIO)
                    .unwrap_or(false)
                || max
                    .map(|value| value >= p50 * TAIL_MAX_TO_P50_RATIO)
                    .unwrap_or(false)
        }
        _ => false,
    }
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() as f64 * percentile).ceil() as usize).saturating_sub(1);
    values.get(index.min(values.len() - 1)).copied()
}

fn min_value(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::min)
}

fn max_value(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::max)
}

fn coefficient_of_variation(values: &[f64]) -> Option<f64> {
    let values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return None;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt() / mean)
}

fn percent_improvement(baseline: Option<f64>, candidate: Option<f64>) -> Option<f64> {
    match (baseline, candidate) {
        (Some(baseline), Some(candidate)) if baseline > 0.0 => {
            Some(((baseline - candidate) / baseline) * 100.0)
        }
        _ => None,
    }
}

fn percent_regression(baseline: Option<f64>, candidate: Option<f64>) -> Option<f64> {
    match (baseline, candidate) {
        (Some(baseline), Some(candidate)) if baseline > 0.0 => {
            Some(((candidate - baseline) / baseline) * 100.0)
        }
        _ => None,
    }
}

fn max_logit_delta(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .reduce(f64::max)
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn fmt_stats_mean(stats: &Option<LatencyStats>) -> String {
    fmt_opt(stats.as_ref().map(|stats| stats.mean_ms))
}

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn parse_positive_usize(value: &str, option: &str) -> Result<usize, CliError> {
    let parsed = value.parse::<usize>().map_err(|error| {
        CliError::Usage(format!("{option} must be a positive integer: {error}"))
    })?;
    if parsed == 0 {
        return Err(CliError::Usage(format!(
            "{option} must be greater than zero"
        )));
    }
    Ok(parsed)
}

fn required_value<I>(args: &mut std::iter::Peekable<I>, option: &str) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| CliError::Usage(format!("{option} requires a value")))
}

fn run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("xr06-{}-{}", now.as_secs(), now.subsec_nanos())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn command_line() -> String {
    std::env::args().collect::<Vec<_>>().join(" ")
}

fn capture_environment() -> Environment {
    Environment {
        machine: command_stdout("uname", &["-a"]).unwrap_or_else(|| "unknown".to_owned()),
        macos: command_stdout("sw_vers", &["-productVersion"])
            .unwrap_or_else(|| "unknown".to_owned()),
        rustc: command_stdout("rustc", &["-Vv"]).unwrap_or_else(|| "unknown".to_owned()),
        cargo: command_stdout("cargo", &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
        mlx_version: command_stdout(
            "python3",
            &[
                "-c",
                "import mlx.core as mx; print(getattr(mx, '__version__', 'unknown'))",
            ],
        )
        .unwrap_or_else(|| "unknown".to_owned()),
        hw_memsize_bytes: command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse::<u64>().ok()),
    }
}

fn usage() -> String {
    format!(
        "usage: GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr06_native_decode_tail_latency_ab -- [--out-dir PATH] [--workloads PATH] [--model-path PATH] [--python PATH] [--trials N] [--max-new-tokens N] [--workload-id ID] [--clear-workload-ids] [--max-workloads N] [--variant NAME|--variants CSV]\n\ndefault out-dir: {DEFAULT_OUT_DIR}\ndefault workloads: {DEFAULT_WORKLOADS}"
    )
}
