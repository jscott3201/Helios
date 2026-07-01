use std::{
    collections::BTreeMap,
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{CliError, workload_corpus::WorkloadRecord};
use gemma4d_ffi::{
    KvCache, KvMode, KvPolicy, KvSnapshot, LoadConfig, Target, decode_one, prefill, runtime_version,
};
use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR09-kv-compression-real-quality-ab";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const GOAL: &str = "XR09-kv-compression-real-quality-ab";
const MODE: &str = "native_kv_compression_real_quality_ab";
const DEFAULT_TRIALS: usize = 1;
const DEFAULT_WORKLOAD_IDS: &[&str] = &[
    "chat_short_1k_001",
    "tool_json_1k_001",
    "code_review_rust_4k_001",
    "benchmark_qa_4k_001",
    "prefix_reuse_edit_8k_a_001",
    "long_repo_pack_16k_001",
];
const EXACT_LOGIT_TOLERANCE: f64 = 0.000_001;
const Q8_MAX_GREEDY_LOGIT_DELTA: f64 = 0.5;
const Q4_MAX_GREEDY_LOGIT_DELTA: f64 = 2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let decision_path = args.out_dir.join("decision.md");
    let command = command_line();
    let git_sha = command_output("git", &["rev-parse", "HEAD"]);
    let git_status_short = command_output_allow_empty("git", &["status", "--short"]);
    let environment = capture_environment();
    let model_identity = capture_model_identity(&args.model_path);
    let mut blockers = startup_blockers(&args);
    let mut failed_hypotheses = Vec::new();
    let mut tokenizer_backend = "not_started".to_owned();
    let mut selected_cases = Vec::new();
    let mut records = Vec::new();

    let model_load_ms = if blockers.is_empty() {
        let workloads = load_workloads(&args.workloads_path)?;
        let mut tokenizer = TokenizerHelper::start(&args.python, &args.model_path)?;
        tokenizer_backend = tokenizer.backend().to_owned();
        selected_cases = prepare_cases(&args, &workloads, &mut tokenizer)?;
        tokenizer.shutdown();

        let load_started = Instant::now();
        match Target::load(&target_config(&args)) {
            Ok(target) => {
                let model_load_ms = duration_ms(load_started.elapsed());
                for case in &selected_cases {
                    for trial_index in 0..args.trials {
                        eprintln!(
                            "XR09 running workload={} context={} trial={}",
                            case.workload_id(),
                            case.context_tokens(),
                            trial_index
                        );
                        match run_case(&args, &target, &run_id, &git_sha, case, trial_index) {
                            Ok(record) => records.push(record),
                            Err(error) => blockers.push(format!(
                                "{} trial {} failed before record write: {error}",
                                case.workload_id(),
                                trial_index
                            )),
                        }
                    }
                }
                Some(model_load_ms)
            }
            Err(error) => {
                blockers.push(format!("target load failed: {error}"));
                Some(duration_ms(load_started.elapsed()))
            }
        }
    } else {
        None
    };

    blockers.extend(blockers_for_records(&records, &selected_cases, args.trials));
    let aggregates = build_aggregates(&records);
    failed_hypotheses.extend(failed_hypotheses_for(&records, &aggregates));
    let q4_failure_analysis = q4_failure_analysis_for(&records);
    let policy_decision = policy_decision_for(&blockers, &aggregates, &q4_failure_analysis);
    let decision = policy_decision.decision_label;
    let status = if decision == "blocked_with_evidence" {
        "blocked"
    } else {
        "completed"
    };
    let generated_files = vec![
        records_path.display().to_string(),
        summary_path.display().to_string(),
        report_path.display().to_string(),
        blockers_path.display().to_string(),
        decision_path.display().to_string(),
    ];

    let summary = P08Summary {
        schema_version: 1,
        goal: GOAL,
        status,
        decision,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        command,
        git_sha,
        git_status_short,
        model_path: args.model_path.display().to_string(),
        workloads_path: args.workloads_path.display().to_string(),
        out_dir: args.out_dir.display().to_string(),
        model_load_ms,
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        decision_path: decision_path.display().to_string(),
        requested_trials: args.trials,
        workload_ids: args.workload_ids.clone(),
        selected_cases: selected_cases
            .iter()
            .map(|case| case.selected.clone())
            .collect(),
        max_context_tokens: args.max_context_tokens,
        exact_logit_tolerance: EXACT_LOGIT_TOLERANCE,
        q8_max_greedy_logit_delta: Q8_MAX_GREEDY_LOGIT_DELTA,
        q4_max_greedy_logit_delta: Q4_MAX_GREEDY_LOGIT_DELTA,
        environment,
        relevant_environment: capture_relevant_environment(),
        model_identity,
        tokenizer_backend,
        policy_decision,
        planar_iso: PlanarIsoReport::default_disabled(),
        claims: claim_inventory(&records),
        aggregates,
        q4_failure_analysis,
        records,
        blockers,
        failed_hypotheses,
        generated_files,
        measurement_notes: vec![
            "cold_ttft_ms measures native BF16 prefill plus KV materialization for each real-context prefix",
            "warm_restore_ms measures payload load, transparent decompression if needed, snapshot import, and cached last-step retrieval",
            "q8/q4 compression is applied only to global/full-attention KV tensors; sliding-window KV tensors and hidden state remain BF16",
            "continued_decode compares one decode_one call after restore against the cold BF16 continuation and is the deterministic quality gate that exercises restored KV tensors",
            "top-k agreement is not recorded in XR09 because the current FFI exposes greedy logits only",
            "q4 quality failures are reportable evidence and are summarized by family and context length",
            "payload_memory_reduction is measured from actual safetensors payload bytes on disk, not an estimate",
            "active_kv_memory_reduction is expected to be zero because compressed payloads are decompressed before active decode",
            "compressed active decode remains disabled by default",
            "Planar/Iso candidates remain feature-disabled by default and need a separate feature-flag prototype before active-memory claims",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR09 KV compression real-quality A/B: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR09 KV compression real-quality A/B blocked; see blockers.md".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    workloads_path: PathBuf,
    python: PathBuf,
    workload_ids: Vec<String>,
    trials: usize,
    max_context_tokens: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut workloads_path = PathBuf::from(DEFAULT_WORKLOADS);
        let mut python = PathBuf::from(DEFAULT_PYTHON);
        let mut workload_ids = DEFAULT_WORKLOAD_IDS
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>();
        let mut trials = DEFAULT_TRIALS;
        let mut max_context_tokens = 32_768;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    out_dir = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--out-dir requires a path")?;
                }
                "--model-path" => {
                    model_path = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--model-path requires a path")?;
                }
                "--workloads" => {
                    workloads_path = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--workloads requires a path")?;
                }
                "--python" => {
                    python = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--python requires a path")?;
                }
                "--clear-workload-ids" => workload_ids.clear(),
                "--workload-id" => {
                    workload_ids.push(args.next().ok_or("--workload-id requires a value")?);
                }
                "--workload-ids" => {
                    let value = args.next().ok_or("--workload-ids requires a comma list")?;
                    workload_ids.extend(
                        value
                            .split(',')
                            .map(str::trim)
                            .filter(|item| !item.is_empty())
                            .map(str::to_owned),
                    );
                }
                "--trials" => {
                    let value = args.next().ok_or("--trials requires a value")?;
                    trials = parse_positive_usize(&value, "--trials")?;
                }
                "--max-context-tokens" => {
                    let value = args.next().ok_or("--max-context-tokens requires a value")?;
                    max_context_tokens = parse_positive_usize(&value, "--max-context-tokens")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr09_kv_compression_real_quality_ab -- [--out-dir PATH] [--model-path PATH] [--workloads PATH] [--python PATH] [--trials N] [--clear-workload-ids] [--workload-id ID] [--workload-ids CSV] [--max-context-tokens N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        if workload_ids.is_empty() {
            return Err("at least one --workload-id is required".into());
        }
        workload_ids.sort();
        workload_ids.dedup();

        Ok(Self {
            out_dir,
            model_path,
            workloads_path,
            python,
            workload_ids,
            trials,
            max_context_tokens,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P08Summary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    decision: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    command: String,
    git_sha: String,
    git_status_short: String,
    model_path: String,
    workloads_path: String,
    out_dir: String,
    model_load_ms: Option<f64>,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    requested_trials: usize,
    workload_ids: Vec<String>,
    selected_cases: Vec<SelectedCase>,
    max_context_tokens: usize,
    exact_logit_tolerance: f64,
    q8_max_greedy_logit_delta: f64,
    q4_max_greedy_logit_delta: f64,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    model_identity: ModelIdentity,
    tokenizer_backend: String,
    policy_decision: PolicyDecision,
    planar_iso: PlanarIsoReport,
    claims: ClaimInventory,
    aggregates: Vec<Aggregate>,
    q4_failure_analysis: Vec<Q4FailureAnalysis>,
    records: Vec<P08Record>,
    blockers: Vec<String>,
    failed_hypotheses: Vec<String>,
    generated_files: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P08Record {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    git_sha: String,
    case_id: String,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    source_deterministic_seed: u64,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    context_tokens: usize,
    input_tokens: usize,
    prefix_token_hash: String,
    trial_index: usize,
    mode: &'static str,
    cold: ColdPrefill,
    baseline_decode: BaselineDecode,
    modes: Vec<ModeRecord>,
    gate: ContextGate,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdPrefill {
    ttft_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    peak_memory_gb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineDecode {
    decode_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModeRecord {
    cache_mode: &'static str,
    payload_path: String,
    payload_sha256: String,
    payload_bytes: u64,
    payload_save_ms: f64,
    payload_load_ms: f64,
    snapshot_import_last_step_ms: f64,
    warm_restore_ms: f64,
    compress_global_layers: bool,
    compress_sliding_layers: bool,
    active_compressed_decode_enabled: bool,
    compressed_full_attention_only: bool,
    restored_last_step: RestoredLastStep,
    continued_decode: ContinuedDecode,
    memory: MemoryComparison,
    quality_gate: QualityGate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RestoredLastStep {
    greedy_token: i32,
    greedy_logit: f32,
    token_agreement: bool,
    greedy_logit_delta: f64,
    sequence_len_parity: bool,
    active_kv_bytes_parity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContinuedDecode {
    decode_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    greedy_agreement: bool,
    greedy_logit_delta: f64,
    top_k_agreement: Option<bool>,
    top_k_note: String,
    sequence_len_parity: bool,
    active_kv_bytes_parity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SelectedCase {
    case_id: String,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    source_deterministic_seed: u64,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    context_tokens: usize,
    prefix_token_hash: String,
}

#[derive(Debug, Clone)]
struct WorkloadCase {
    selected: SelectedCase,
    token_ids: Vec<i32>,
}

impl WorkloadCase {
    fn workload_id(&self) -> &str {
        &self.selected.workload_id
    }

    fn context_tokens(&self) -> usize {
        self.selected.context_tokens
    }
}

#[derive(Debug, Clone)]
struct EncodedWorkload {
    record: WorkloadRecord,
    prompt_sha256: String,
    token_ids: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PolicyDecision {
    decision_label: &'static str,
    recommended_next_candidate: &'static str,
    q8_recommendation: &'static str,
    q4_recommendation: &'static str,
    planar_iso_recommendation: &'static str,
    rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Aggregate {
    workload_id: String,
    family: String,
    context_tokens: usize,
    variant: String,
    trial_count: usize,
    passed_quality_count: usize,
    low_n: bool,
    greedy_agreement_rate: f64,
    max_greedy_logit_delta: f64,
    payload_mib_median: f64,
    payload_reduction_percent_median: f64,
    warm_restore_ms_median: f64,
    warm_restore_ms_p95: f64,
    active_kv_mib_median: f64,
    active_kv_reduction_percent_median: f64,
    peak_memory_gb_max: f64,
    quality_passed: bool,
    active_decode_disabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Q4FailureAnalysis {
    workload_id: String,
    family: String,
    context_tokens: usize,
    trial_index: usize,
    greedy_agreement: bool,
    greedy_logit_delta: f64,
    threshold_greedy_logit_delta: f64,
    baseline_greedy_token: i32,
    restored_greedy_token: i32,
    payload_reduction_percent: f64,
    active_kv_reduction_percent: f64,
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryComparison {
    bf16_payload_bytes: u64,
    compressed_payload_bytes: u64,
    payload_delta_bytes: i64,
    payload_memory_reduction: f64,
    bf16_active_kv_bytes: u64,
    restored_active_kv_bytes: u64,
    active_kv_delta_bytes: i64,
    active_kv_memory_reduction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityGate {
    passed: bool,
    threshold_greedy_logit_delta: f64,
    greedy_agreement: bool,
    greedy_logit_delta_within_threshold: bool,
    payload_smaller_than_bf16: bool,
    active_decode_remains_bf16: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextGate {
    passed: bool,
    bf16_exact_restore: bool,
    q8_measured: bool,
    q4_measured: bool,
    q8_payload_smaller: bool,
    q4_payload_smaller: bool,
    compressed_active_decode_disabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaimInventory {
    exactness: Vec<String>,
    quality: Vec<String>,
    memory: Vec<String>,
    latency: Vec<String>,
    defaults: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlanarIsoReport {
    feature_enabled: bool,
    accepted_by_default: bool,
    status: &'static str,
    candidates: Vec<&'static str>,
}

impl PlanarIsoReport {
    fn default_disabled() -> Self {
        Self {
            feature_enabled: false,
            accepted_by_default: false,
            status: "feature_disabled_default",
            candidates: vec!["planar4", "planar3", "iso4", "iso3"],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Environment {
    machine: String,
    macos: String,
    rustc: String,
    cargo: String,
    runtime_backend: String,
    runtime_backend_version: String,
    git_commit: String,
    git_status_short: String,
    hw_memsize_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelIdentity {
    model_path: String,
    exists: bool,
    configured_revision: String,
    config_sha256: String,
    tokenizer_sha256: String,
    tokenizer_config_sha256: String,
    chat_template_sha256: String,
    safetensors_inventory_sha256: String,
    safetensors_file_count: usize,
    safetensors_total_bytes: u64,
}

struct SafetensorsInventory {
    inventory_sha256: String,
    file_count: usize,
    total_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
enum BenchMode {
    Bf16,
    Q8,
    Q4,
}

impl BenchMode {
    fn all() -> [Self; 3] {
        [Self::Bf16, Self::Q8, Self::Q4]
    }

    fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "mlx_affine_q8",
            Self::Q4 => "mlx_affine_q4",
        }
    }

    fn ffi_mode(self) -> KvMode {
        match self {
            Self::Bf16 => KvMode::Bf16,
            Self::Q8 => KvMode::MlxAffineQ8,
            Self::Q4 => KvMode::MlxAffineQ4,
        }
    }

    fn max_logit_delta(self) -> f64 {
        match self {
            Self::Bf16 => EXACT_LOGIT_TOLERANCE,
            Self::Q8 => Q8_MAX_GREEDY_LOGIT_DELTA,
            Self::Q4 => Q4_MAX_GREEDY_LOGIT_DELTA,
        }
    }

    fn compresses_full_attention(self) -> bool {
        !matches!(self, Self::Bf16)
    }
}

fn run_case(
    args: &Args,
    target: &Target,
    run_id: &str,
    git_sha: &str,
    case: &WorkloadCase,
    trial_index: usize,
) -> Result<P08Record, Box<dyn std::error::Error>> {
    let context_dir = args.out_dir.join(format!(
        "{}-trial-{trial_index}",
        sanitize_path_component(&case.selected.case_id)
    ));
    if context_dir.exists() {
        fs::remove_dir_all(&context_dir)?;
    }
    fs::create_dir_all(&context_dir)?;

    let mut cold_cache = KvCache::create(&KvPolicy::default())?;
    let cold_started = Instant::now();
    let cold_step = prefill(target, &mut cold_cache, &case.token_ids)?;
    let cold_ttft = duration_ms(cold_started.elapsed());
    let snapshot = cold_cache.export_snapshot()?;

    let mut mode_records = Vec::new();
    for mode in BenchMode::all() {
        mode_records.push(write_mode_payload(
            &context_dir,
            &snapshot,
            mode,
            cold_step.active_kv_bytes,
        )?);
    }
    let bf16_payload_bytes = mode_records
        .iter()
        .find(|record| record.cache_mode == "bf16")
        .map(|record| record.payload_bytes)
        .ok_or("BF16 mode record missing")?;

    let baseline_started = Instant::now();
    let baseline_next = decode_one(target, &mut cold_cache, cold_step.greedy_token)?;
    let baseline_decode = BaselineDecode {
        decode_ms: duration_ms(baseline_started.elapsed()),
        greedy_token: baseline_next.greedy_token,
        greedy_logit: baseline_next.greedy_logit,
        sequence_len: baseline_next.sequence_len,
        active_kv_bytes: baseline_next.active_kv_bytes,
    };

    for record in &mut mode_records {
        restore_and_score_mode(
            target,
            record,
            &cold_step,
            &baseline_decode,
            bf16_payload_bytes,
        )?;
    }

    let gate = context_gate(&mode_records);
    let blockers = blockers_for_context(case, &mode_records, &gate);
    Ok(P08Record {
        schema_version: 1,
        goal: GOAL,
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        git_sha: git_sha.to_owned(),
        case_id: case.selected.case_id.clone(),
        workload_id: case.selected.workload_id.clone(),
        family: case.selected.family.clone(),
        prompt_path: case.selected.prompt_path.clone(),
        prompt_sha256: case.selected.prompt_sha256.clone(),
        source_deterministic_seed: case.selected.source_deterministic_seed,
        target_context_tokens: case.selected.target_context_tokens,
        actual_context_tokens: case.selected.actual_context_tokens,
        context_tokens: case.selected.context_tokens,
        input_tokens: case.token_ids.len(),
        prefix_token_hash: case.selected.prefix_token_hash.clone(),
        trial_index,
        mode: MODE,
        cold: ColdPrefill {
            ttft_ms: cold_ttft,
            greedy_token: cold_step.greedy_token,
            greedy_logit: cold_step.greedy_logit,
            sequence_len: cold_step.sequence_len,
            active_kv_bytes: cold_step.active_kv_bytes,
            peak_memory_gb: cold_step.peak_memory_gb,
        },
        baseline_decode,
        modes: mode_records,
        gate,
        blockers,
    })
}

fn write_mode_payload(
    context_dir: &Path,
    snapshot: &KvSnapshot,
    mode: BenchMode,
    bf16_active_kv_bytes: u64,
) -> Result<ModeRecord, Box<dyn std::error::Error>> {
    let payload_path = context_dir.join(format!("{}.safetensors", mode.label()));
    let save_started = Instant::now();
    if matches!(mode, BenchMode::Bf16) {
        snapshot.save_to_path(&payload_path)?;
    } else {
        snapshot.save_compressed_to_path(&payload_path, mode.ffi_mode(), true, false)?;
    }
    let payload_save_ms = duration_ms(save_started.elapsed());
    let payload_bytes = fs::metadata(&payload_path)?.len();
    let payload_sha256 = file_sha256(&payload_path)?;

    Ok(ModeRecord {
        cache_mode: mode.label(),
        payload_path: payload_path.display().to_string(),
        payload_sha256,
        payload_bytes,
        payload_save_ms,
        payload_load_ms: 0.0,
        snapshot_import_last_step_ms: 0.0,
        warm_restore_ms: 0.0,
        compress_global_layers: mode.compresses_full_attention(),
        compress_sliding_layers: false,
        active_compressed_decode_enabled: false,
        compressed_full_attention_only: mode.compresses_full_attention(),
        restored_last_step: RestoredLastStep {
            greedy_token: 0,
            greedy_logit: 0.0,
            token_agreement: false,
            greedy_logit_delta: f64::INFINITY,
            sequence_len_parity: false,
            active_kv_bytes_parity: false,
        },
        continued_decode: ContinuedDecode {
            decode_ms: 0.0,
            greedy_token: 0,
            greedy_logit: 0.0,
            greedy_agreement: false,
            greedy_logit_delta: f64::INFINITY,
            top_k_agreement: None,
            top_k_note: "not_available_current_ffi_exposes_greedy_only".to_owned(),
            sequence_len_parity: false,
            active_kv_bytes_parity: false,
        },
        memory: MemoryComparison {
            bf16_payload_bytes: payload_bytes,
            compressed_payload_bytes: payload_bytes,
            payload_delta_bytes: 0,
            payload_memory_reduction: 0.0,
            bf16_active_kv_bytes,
            restored_active_kv_bytes: 0,
            active_kv_delta_bytes: 0,
            active_kv_memory_reduction: 0.0,
        },
        quality_gate: QualityGate {
            passed: false,
            threshold_greedy_logit_delta: mode.max_logit_delta(),
            greedy_agreement: false,
            greedy_logit_delta_within_threshold: false,
            payload_smaller_than_bf16: matches!(mode, BenchMode::Bf16),
            active_decode_remains_bf16: true,
        },
    })
}

fn restore_and_score_mode(
    target: &Target,
    record: &mut ModeRecord,
    cold_step: &gemma4d_ffi::StepResult,
    baseline_decode: &BaselineDecode,
    bf16_payload_bytes: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mode = match record.cache_mode {
        "bf16" => BenchMode::Bf16,
        "mlx_affine_q8" => BenchMode::Q8,
        "mlx_affine_q4" => BenchMode::Q4,
        other => return Err(format!("unsupported mode record {other}").into()),
    };

    let load_started = Instant::now();
    let loaded_snapshot = KvSnapshot::load_from_path(&record.payload_path)?;
    record.payload_load_ms = duration_ms(load_started.elapsed());

    let mut restored_cache = KvCache::create(&policy_for_mode(mode))?;
    let import_started = Instant::now();
    restored_cache.import_snapshot(&loaded_snapshot)?;
    let restored_last = restored_cache.last_step()?;
    record.snapshot_import_last_step_ms = duration_ms(import_started.elapsed());
    record.warm_restore_ms = record.payload_load_ms + record.snapshot_import_last_step_ms;

    let decode_started = Instant::now();
    let restored_next = decode_one(target, &mut restored_cache, cold_step.greedy_token)?;
    let restored_decode_ms = duration_ms(decode_started.elapsed());

    let last_logit_delta =
        (f64::from(cold_step.greedy_logit) - f64::from(restored_last.greedy_logit)).abs();
    let decode_logit_delta =
        (f64::from(baseline_decode.greedy_logit) - f64::from(restored_next.greedy_logit)).abs();
    let payload_delta = record.payload_bytes as i64 - bf16_payload_bytes as i64;
    let payload_memory_reduction = if bf16_payload_bytes == 0 {
        0.0
    } else {
        1.0 - (record.payload_bytes as f64 / bf16_payload_bytes as f64)
    };
    let active_delta = restored_last.active_kv_bytes as i64 - cold_step.active_kv_bytes as i64;
    let active_reduction = if cold_step.active_kv_bytes == 0 {
        0.0
    } else {
        1.0 - (restored_last.active_kv_bytes as f64 / cold_step.active_kv_bytes as f64)
    };

    record.restored_last_step = RestoredLastStep {
        greedy_token: restored_last.greedy_token,
        greedy_logit: restored_last.greedy_logit,
        token_agreement: cold_step.greedy_token == restored_last.greedy_token,
        greedy_logit_delta: last_logit_delta,
        sequence_len_parity: cold_step.sequence_len == restored_last.sequence_len,
        active_kv_bytes_parity: cold_step.active_kv_bytes == restored_last.active_kv_bytes,
    };
    record.continued_decode = ContinuedDecode {
        decode_ms: restored_decode_ms,
        greedy_token: restored_next.greedy_token,
        greedy_logit: restored_next.greedy_logit,
        greedy_agreement: baseline_decode.greedy_token == restored_next.greedy_token,
        greedy_logit_delta: decode_logit_delta,
        top_k_agreement: None,
        top_k_note: "not_available_current_ffi_exposes_greedy_only".to_owned(),
        sequence_len_parity: baseline_decode.sequence_len == restored_next.sequence_len,
        active_kv_bytes_parity: baseline_decode.active_kv_bytes == restored_next.active_kv_bytes,
    };
    record.memory = MemoryComparison {
        bf16_payload_bytes,
        compressed_payload_bytes: record.payload_bytes,
        payload_delta_bytes: payload_delta,
        payload_memory_reduction,
        bf16_active_kv_bytes: cold_step.active_kv_bytes,
        restored_active_kv_bytes: restored_last.active_kv_bytes,
        active_kv_delta_bytes: active_delta,
        active_kv_memory_reduction: active_reduction,
    };
    let payload_smaller_than_bf16 =
        matches!(mode, BenchMode::Bf16) || record.payload_bytes < bf16_payload_bytes;
    let greedy_logit_delta_within_threshold = decode_logit_delta <= mode.max_logit_delta();
    record.quality_gate = QualityGate {
        passed: record.continued_decode.greedy_agreement
            && greedy_logit_delta_within_threshold
            && payload_smaller_than_bf16
            && !record.active_compressed_decode_enabled,
        threshold_greedy_logit_delta: mode.max_logit_delta(),
        greedy_agreement: record.continued_decode.greedy_agreement,
        greedy_logit_delta_within_threshold,
        payload_smaller_than_bf16,
        active_decode_remains_bf16: !record.active_compressed_decode_enabled,
    };

    Ok(())
}

fn policy_for_mode(mode: BenchMode) -> KvPolicy {
    KvPolicy {
        ssd_prefix_mode: mode.ffi_mode(),
        compress_global_layers: mode.compresses_full_attention(),
        compress_sliding_layers: false,
        allow_active_compressed_decode: false,
        ..Default::default()
    }
}

fn context_gate(records: &[ModeRecord]) -> ContextGate {
    let bf16 = records.iter().find(|record| record.cache_mode == "bf16");
    let q8 = records
        .iter()
        .find(|record| record.cache_mode == "mlx_affine_q8");
    let q4 = records
        .iter()
        .find(|record| record.cache_mode == "mlx_affine_q4");
    let bf16_exact_restore = bf16.is_some_and(|record| {
        record.restored_last_step.token_agreement
            && record.restored_last_step.greedy_logit_delta <= EXACT_LOGIT_TOLERANCE
            && record.continued_decode.greedy_agreement
            && record.continued_decode.greedy_logit_delta <= EXACT_LOGIT_TOLERANCE
    });
    let q8_payload_smaller = q8.is_some_and(|record| record.memory.payload_delta_bytes < 0);
    let q4_payload_smaller = q4.is_some_and(|record| record.memory.payload_delta_bytes < 0);
    let compressed_active_decode_disabled = records
        .iter()
        .all(|record| !record.active_compressed_decode_enabled);
    ContextGate {
        passed: bf16_exact_restore
            && q8.is_some()
            && q4.is_some()
            && q8_payload_smaller
            && q4_payload_smaller
            && compressed_active_decode_disabled,
        bf16_exact_restore,
        q8_measured: q8.is_some(),
        q4_measured: q4.is_some(),
        q8_payload_smaller,
        q4_payload_smaller,
        compressed_active_decode_disabled,
    }
}

fn blockers_for_context(
    case: &WorkloadCase,
    _records: &[ModeRecord],
    gate: &ContextGate,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if !gate.bf16_exact_restore {
        blockers.push(format!(
            "{} {} tokens: BF16 snapshot restore was not exact",
            case.workload_id(),
            case.context_tokens()
        ));
    }
    if !gate.q8_measured {
        blockers.push(format!(
            "{}: q8 payload was not measured",
            case.workload_id()
        ));
    }
    if !gate.q4_measured {
        blockers.push(format!(
            "{}: q4 payload was not measured",
            case.workload_id()
        ));
    }
    if !gate.compressed_active_decode_disabled {
        blockers.push(format!(
            "{}: active compressed decode was unexpectedly enabled",
            case.workload_id()
        ));
    }
    blockers
}

fn blockers_for_records(
    records: &[P08Record],
    selected_cases: &[WorkloadCase],
    trials: usize,
) -> Vec<String> {
    let mut blockers = Vec::new();
    for case in selected_cases {
        for trial_index in 0..trials {
            if !records.iter().any(|record| {
                record.workload_id == case.selected.workload_id && record.trial_index == trial_index
            }) {
                blockers.push(format!(
                    "{} trial {trial_index}: missing benchmark record",
                    case.workload_id()
                ));
            }
        }
    }
    for record in records {
        blockers.extend(record.blockers.iter().cloned());
    }
    blockers
}

fn claim_inventory(records: &[P08Record]) -> ClaimInventory {
    let mut exactness = Vec::new();
    let mut quality = Vec::new();
    let mut memory = Vec::new();
    let mut latency = Vec::new();
    let defaults = vec![
        "compressed active decode remains disabled by default".to_owned(),
        "Planar/Iso candidates remain feature-disabled without real XR09 prototype evidence"
            .to_owned(),
    ];

    for record in records {
        exactness.push(format!(
            "{} {} tokens BF16 exact restore={}",
            record.workload_id, record.context_tokens, record.gate.bf16_exact_restore
        ));
        latency.push(format!(
            "{} {} tokens cold TTFT {:.3} ms, BF16 warm restore {:.3} ms, q8 {:.3} ms, q4 {:.3} ms",
            record.workload_id,
            record.context_tokens,
            record.cold.ttft_ms,
            mode_record(record, "bf16").map_or(0.0, |mode| mode.warm_restore_ms),
            mode_record(record, "mlx_affine_q8").map_or(0.0, |mode| mode.warm_restore_ms),
            mode_record(record, "mlx_affine_q4").map_or(0.0, |mode| mode.warm_restore_ms),
        ));
        for mode in &record.modes {
            quality.push(format!(
                "{} {} tokens {} greedy_agreement={} greedy_logit_delta={:.6} gate={}",
                record.workload_id,
                record.context_tokens,
                mode.cache_mode,
                mode.continued_decode.greedy_agreement,
                mode.continued_decode.greedy_logit_delta,
                mode.quality_gate.passed
            ));
            memory.push(format!(
                "{} {} tokens {} payload reduction {:.3}% active reduction {:.3}%",
                record.workload_id,
                record.context_tokens,
                mode.cache_mode,
                mode.memory.payload_memory_reduction * 100.0,
                mode.memory.active_kv_memory_reduction * 100.0
            ));
        }
    }

    ClaimInventory {
        exactness,
        quality,
        memory,
        latency,
        defaults,
    }
}

fn mode_record<'a>(record: &'a P08Record, cache_mode: &str) -> Option<&'a ModeRecord> {
    record
        .modes
        .iter()
        .find(|mode| mode.cache_mode == cache_mode)
}

fn build_aggregates(records: &[P08Record]) -> Vec<Aggregate> {
    let mut keys = records
        .iter()
        .flat_map(|record| {
            record.modes.iter().map(|mode| {
                (
                    record.workload_id.clone(),
                    record.family.clone(),
                    record.context_tokens,
                    mode.cache_mode.to_owned(),
                )
            })
        })
        .collect::<Vec<_>>();
    keys.sort();
    keys.dedup();

    let mut aggregates = Vec::new();
    for (workload_id, family, context_tokens, variant) in keys {
        let modes = records
            .iter()
            .filter(|record| record.workload_id == workload_id)
            .filter_map(|record| mode_record(record, &variant))
            .collect::<Vec<_>>();
        let trial_count = modes.len();
        let passed_quality_count = modes.iter().filter(|mode| mode.quality_gate.passed).count();
        let greedy_agreement_rate = if trial_count == 0 {
            0.0
        } else {
            modes
                .iter()
                .filter(|mode| mode.continued_decode.greedy_agreement)
                .count() as f64
                / trial_count as f64
        };
        let max_greedy_logit_delta = modes
            .iter()
            .map(|mode| mode.continued_decode.greedy_logit_delta)
            .fold(0.0, f64::max);
        aggregates.push(Aggregate {
            workload_id: workload_id.clone(),
            family,
            context_tokens,
            variant,
            trial_count,
            passed_quality_count,
            low_n: trial_count < 3,
            greedy_agreement_rate,
            max_greedy_logit_delta,
            payload_mib_median: median(
                modes
                    .iter()
                    .map(|mode| mode.payload_bytes as f64 / 1_048_576.0)
                    .collect(),
            )
            .unwrap_or(0.0),
            payload_reduction_percent_median: median(
                modes
                    .iter()
                    .map(|mode| mode.memory.payload_memory_reduction * 100.0)
                    .collect(),
            )
            .unwrap_or(0.0),
            warm_restore_ms_median: median(modes.iter().map(|mode| mode.warm_restore_ms).collect())
                .unwrap_or(0.0),
            warm_restore_ms_p95: percentile(
                modes.iter().map(|mode| mode.warm_restore_ms).collect(),
                0.95,
            )
            .unwrap_or(0.0),
            active_kv_mib_median: median(
                modes
                    .iter()
                    .map(|mode| mode.memory.restored_active_kv_bytes as f64 / 1_048_576.0)
                    .collect(),
            )
            .unwrap_or(0.0),
            active_kv_reduction_percent_median: median(
                modes
                    .iter()
                    .map(|mode| mode.memory.active_kv_memory_reduction * 100.0)
                    .collect(),
            )
            .unwrap_or(0.0),
            peak_memory_gb_max: records
                .iter()
                .filter(|record| record.workload_id == workload_id)
                .map(|record| f64::from(record.cold.peak_memory_gb))
                .fold(0.0, f64::max),
            quality_passed: trial_count > 0 && passed_quality_count == trial_count,
            active_decode_disabled: modes
                .iter()
                .all(|mode| !mode.active_compressed_decode_enabled),
        });
    }
    aggregates
}

fn q4_failure_analysis_for(records: &[P08Record]) -> Vec<Q4FailureAnalysis> {
    let mut out = Vec::new();
    for record in records {
        let Some(q4) = mode_record(record, "mlx_affine_q4") else {
            continue;
        };
        if q4.quality_gate.passed {
            continue;
        }
        let baseline = &record.baseline_decode;
        out.push(Q4FailureAnalysis {
            workload_id: record.workload_id.clone(),
            family: record.family.clone(),
            context_tokens: record.context_tokens,
            trial_index: record.trial_index,
            greedy_agreement: q4.continued_decode.greedy_agreement,
            greedy_logit_delta: q4.continued_decode.greedy_logit_delta,
            threshold_greedy_logit_delta: q4.quality_gate.threshold_greedy_logit_delta,
            baseline_greedy_token: baseline.greedy_token,
            restored_greedy_token: q4.continued_decode.greedy_token,
            payload_reduction_percent: q4.memory.payload_memory_reduction * 100.0,
            active_kv_reduction_percent: q4.memory.active_kv_memory_reduction * 100.0,
            reason: if !q4.continued_decode.greedy_agreement {
                "continued_decode_greedy_token_mismatch".to_owned()
            } else if !q4.quality_gate.greedy_logit_delta_within_threshold {
                "continued_decode_logit_delta_exceeded_threshold".to_owned()
            } else if !q4.quality_gate.payload_smaller_than_bf16 {
                "payload_not_smaller_than_bf16".to_owned()
            } else {
                "quality_gate_failed".to_owned()
            },
        });
    }
    out
}

fn failed_hypotheses_for(records: &[P08Record], aggregates: &[Aggregate]) -> Vec<String> {
    let mut out = Vec::new();
    for aggregate in aggregates {
        if aggregate.low_n {
            out.push(format!(
                "{} {} is low-N latency evidence: {} trial(s)",
                aggregate.workload_id, aggregate.variant, aggregate.trial_count
            ));
        }
        if aggregate.variant == "mlx_affine_q4" && !aggregate.quality_passed {
            out.push(format!(
                "{} q4 failed quality gate: agreement_rate={:.3}, max_logit_delta={:.6}",
                aggregate.workload_id,
                aggregate.greedy_agreement_rate,
                aggregate.max_greedy_logit_delta
            ));
        }
        if aggregate.variant == "mlx_affine_q8" && !aggregate.quality_passed {
            out.push(format!(
                "{} q8 failed quality gate: agreement_rate={:.3}, max_logit_delta={:.6}",
                aggregate.workload_id,
                aggregate.greedy_agreement_rate,
                aggregate.max_greedy_logit_delta
            ));
        }
        if matches!(
            aggregate.variant.as_str(),
            "mlx_affine_q8" | "mlx_affine_q4"
        ) && aggregate.payload_reduction_percent_median <= 0.0
        {
            out.push(format!(
                "{} {} did not reduce payload size versus BF16",
                aggregate.workload_id, aggregate.variant
            ));
        }
    }
    for record in records {
        for mode in &record.modes {
            if mode.active_compressed_decode_enabled {
                out.push(format!(
                    "{} {} unexpectedly enabled active compressed decode",
                    record.workload_id, mode.cache_mode
                ));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn policy_decision_for(
    blockers: &[String],
    aggregates: &[Aggregate],
    q4_failures: &[Q4FailureAnalysis],
) -> PolicyDecision {
    if !blockers.is_empty() {
        return PolicyDecision {
            decision_label: "blocked_with_evidence",
            recommended_next_candidate: "no_go_until_blockers_resolved",
            q8_recommendation: "unresolved",
            q4_recommendation: "unresolved",
            planar_iso_recommendation: "defer",
            rationale: "hard blockers prevent a compression policy decision".to_owned(),
        };
    }

    let q8 = aggregates
        .iter()
        .filter(|aggregate| aggregate.variant == "mlx_affine_q8")
        .collect::<Vec<_>>();
    let q4 = aggregates
        .iter()
        .filter(|aggregate| aggregate.variant == "mlx_affine_q4")
        .collect::<Vec<_>>();
    let q8_passed = !q8.is_empty()
        && q8.iter().all(|aggregate| {
            aggregate.quality_passed
                && aggregate.payload_reduction_percent_median > 0.0
                && aggregate.active_decode_disabled
        });
    let q4_passed = !q4.is_empty()
        && q4.iter().all(|aggregate| {
            aggregate.quality_passed
                && aggregate.payload_reduction_percent_median > 0.0
                && aggregate.active_decode_disabled
        })
        && q4_failures.is_empty();

    if q8_passed && !q4_passed {
        PolicyDecision {
            decision_label: "accept_candidate",
            recommended_next_candidate: "q8_default_for_ssd_payload_q4_rejected_planar_iso_research",
            q8_recommendation: "promote_q8_as_next_ssd_payload_candidate",
            q4_recommendation: "reject_q4_until_greedy_failures_are_resolved",
            planar_iso_recommendation: "research_only_behind_feature_flags",
            rationale: "q8 passed deterministic continued-decode quality on real contexts while q4 produced quality failures".to_owned(),
        }
    } else if q8_passed && q4_passed {
        PolicyDecision {
            decision_label: "keep_experimental",
            recommended_next_candidate: "q8_default_for_ssd_payload_compare_q4_with_more_trials",
            q8_recommendation: "promote_q8_as_next_ssd_payload_candidate",
            q4_recommendation: "needs_more_real_context_and_active_decode_research",
            planar_iso_recommendation: "research_only_behind_feature_flags",
            rationale:
                "q8 and q4 passed measured quality, but q4 needs broader evidence before promotion"
                    .to_owned(),
        }
    } else {
        PolicyDecision {
            decision_label: "reject_candidate",
            recommended_next_candidate: "no_go_for_compression_candidate",
            q8_recommendation: "reject_until_quality_gate_passes",
            q4_recommendation: "reject_q4_until_greedy_failures_are_resolved",
            planar_iso_recommendation: "defer",
            rationale: "no compressed payload candidate satisfied the real-context quality gates"
                .to_owned(),
        }
    }
}

fn render_report(summary: &P08Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR09 KV Compression Real-Quality A/B\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Run\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Mode | `{}` |\n", summary.mode));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!("| Model path | `{}` |\n", summary.model_path));
    out.push_str(&format!("| Workloads | `{}` |\n", summary.workloads_path));
    out.push_str(&format!("| Trials | `{}` |\n", summary.requested_trials));
    out.push_str(&format!(
        "| Model load ms | `{}` |\n",
        option_ms(summary.model_load_ms)
    ));
    out.push_str(&format!(
        "| Runtime | `{}` `{}` |\n",
        summary.environment.runtime_backend, summary.environment.runtime_backend_version
    ));
    out.push_str(&format!(
        "| Recommended next candidate | `{}` |\n\n",
        summary.policy_decision.recommended_next_candidate
    ));

    out.push_str("## Workload Cases\n\n");
    out.push_str("| Case | Workload | Family | Context | Seed | Prefix hash |\n");
    out.push_str("|---|---|---|---:|---:|---|\n");
    for case in &summary.selected_cases {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | {} | {} | `{}` |\n",
            case.case_id,
            case.workload_id,
            case.family,
            case.context_tokens,
            case.source_deterministic_seed,
            case.prefix_token_hash
        ));
    }

    out.push_str("\n## Aggregate Results\n\n");
    out.push_str("| Workload | Family | Context | Mode | Trials | Quality | Greedy agree | Max logit delta | Payload MiB | Payload reduction | Warm p50 ms | Warm p95 ms | Active KV MiB | Active reduction | Peak GB |\n");
    out.push_str("|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for aggregate in &summary.aggregates {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | `{}` | {}/{} | `{}` | {:.3} | {:.6} | {:.3} | {:.3}% | {:.3} | {:.3} | {:.3} | {:.3}% | {:.3} |\n",
            aggregate.workload_id,
            aggregate.family,
            aggregate.context_tokens,
            aggregate.variant,
            aggregate.passed_quality_count,
            aggregate.trial_count,
            aggregate.quality_passed,
            aggregate.greedy_agreement_rate,
            aggregate.max_greedy_logit_delta,
            aggregate.payload_mib_median,
            aggregate.payload_reduction_percent_median,
            aggregate.warm_restore_ms_median,
            aggregate.warm_restore_ms_p95,
            aggregate.active_kv_mib_median,
            aggregate.active_kv_reduction_percent_median,
            aggregate.peak_memory_gb_max,
        ));
    }

    out.push_str("\n## Per-Trial Results\n\n");
    out.push_str("| Workload | Trial | Mode | Gate | Greedy Agree | Logit Delta | Payload MiB | Payload Reduction | Warm Restore ms | Decode ms | Active KV Reduction |\n");
    out.push_str("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.records {
        for mode in &record.modes {
            out.push_str(&format!(
                "| `{}` | {} | `{}` | `{}` | `{}` | {:.6} | {:.3} | {:.3}% | {:.3} | {:.3} | {:.3}% |\n",
                record.workload_id,
                record.trial_index,
                mode.cache_mode,
                mode.quality_gate.passed,
                mode.continued_decode.greedy_agreement,
                mode.continued_decode.greedy_logit_delta,
                mode.payload_bytes as f64 / 1_048_576.0,
                mode.memory.payload_memory_reduction * 100.0,
                mode.warm_restore_ms,
                mode.continued_decode.decode_ms,
                mode.memory.active_kv_memory_reduction * 100.0,
            ));
        }
    }

    out.push_str("\n## Q4 Failure Analysis\n\n");
    if summary.q4_failure_analysis.is_empty() {
        out.push_str("No q4 failures recorded.\n");
    } else {
        out.push_str("| Workload | Family | Context | Trial | Reason | Baseline token | q4 token | Logit delta | Payload reduction |\n");
        out.push_str("|---|---|---:|---:|---|---:|---:|---:|---:|\n");
        for failure in &summary.q4_failure_analysis {
            out.push_str(&format!(
                "| `{}` | `{}` | {} | {} | `{}` | {} | {} | {:.6} | {:.3}% |\n",
                failure.workload_id,
                failure.family,
                failure.context_tokens,
                failure.trial_index,
                failure.reason,
                failure.baseline_greedy_token,
                failure.restored_greedy_token,
                failure.greedy_logit_delta,
                failure.payload_reduction_percent,
            ));
        }
    }

    out.push_str("\n## Workload Gates\n\n");
    out.push_str("| Workload | Trial | BF16 Exact | q8 Measured | q4 Measured | q8 Smaller | q4 Smaller | Active Decode Disabled |\n");
    out.push_str("|---|---:|---|---|---|---|---|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            record.workload_id,
            record.trial_index,
            record.gate.bf16_exact_restore,
            record.gate.q8_measured,
            record.gate.q4_measured,
            record.gate.q8_payload_smaller,
            record.gate.q4_payload_smaller,
            record.gate.compressed_active_decode_disabled,
        ));
    }

    out.push_str("\n## Verification Command\n\n```sh\n");
    out.push_str("GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr09_kv_compression_real_quality_ab -- --out-dir benchmarks/out/XR09-kv-compression-real-quality-ab\n");
    out.push_str("```\n\n## Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {note}.\n"));
    }
    out
}

fn render_blockers(summary: &P08Summary) -> String {
    let mut out = "# XR09 KV Compression Real-Quality A/B Blockers\n\n".to_owned();
    if summary.blockers.is_empty() {
        out.push_str("No hard blockers recorded.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out.push_str("\n## Failed Hypotheses And Caveats\n\n");
    if summary.failed_hypotheses.is_empty() {
        out.push_str("- None recorded.\n");
    } else {
        for item in &summary.failed_hypotheses {
            out.push_str(&format!("- {item}\n"));
        }
    }
    if !summary.q4_failure_analysis.is_empty() {
        out.push_str("\n## Q4 Failure Analysis\n\n");
        for failure in &summary.q4_failure_analysis {
            out.push_str(&format!(
                "- {} {} tokens trial {}: {} (baseline token {}, q4 token {}, logit delta {:.6}, payload reduction {:.3}%).\n",
                failure.workload_id,
                failure.context_tokens,
                failure.trial_index,
                failure.reason,
                failure.baseline_greedy_token,
                failure.restored_greedy_token,
                failure.greedy_logit_delta,
                failure.payload_reduction_percent
            ));
        }
    }
    out
}

fn render_decision(summary: &P08Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR09 KV Compression Real-Quality A/B Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Evidence\n\n");
    for file in &summary.generated_files {
        out.push_str(&format!("- `{file}`\n"));
    }
    out.push_str("\n## Recommendation\n\n");
    out.push_str(&format!(
        "- Next candidate: `{}`\n",
        summary.policy_decision.recommended_next_candidate
    ));
    out.push_str(&format!(
        "- q8: `{}`\n",
        summary.policy_decision.q8_recommendation
    ));
    out.push_str(&format!(
        "- q4: `{}`\n",
        summary.policy_decision.q4_recommendation
    ));
    out.push_str(&format!(
        "- Planar/Iso: `{}`\n",
        summary.policy_decision.planar_iso_recommendation
    ));
    out.push_str(&format!(
        "- Rationale: {}\n",
        summary.policy_decision.rationale
    ));
    out.push_str("\n## Claim Boundary\n\n");
    out.push_str("- XR09 measures compressed prefix payload storage and BF16 active restore on real-context workloads.\n");
    out.push_str(
        "- Active compressed decode remains disabled; active-memory reductions are not claimed.\n",
    );
    out.push_str(
        "- q4 cannot be promoted while any greedy agreement failure remains unresolved.\n",
    );
    out.push_str("- This is not a production serving readiness claim.\n");
    out
}

fn write_jsonl(path: &Path, records: &[P08Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
    }
    Ok(())
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

fn prepare_cases(
    args: &Args,
    workloads: &[WorkloadRecord],
    tokenizer: &mut TokenizerHelper,
) -> Result<Vec<WorkloadCase>, CliError> {
    let mut cases = Vec::new();
    for workload_id in &args.workload_ids {
        let record = workloads
            .iter()
            .find(|record| record.workload_id == *workload_id)
            .ok_or_else(|| {
                CliError::Runtime(format!(
                    "required workload {workload_id} missing from {}",
                    args.workloads_path.display()
                ))
            })?;
        let encoded = encode_workload(tokenizer, record)?;
        if encoded.token_ids.len() > args.max_context_tokens {
            return Err(CliError::Runtime(format!(
                "{} has {} tokens, above max_context_tokens {}",
                encoded.record.workload_id,
                encoded.token_ids.len(),
                args.max_context_tokens
            )));
        }
        let prefix_token_hash = token_hash("xr09-prefix-token-ids-v1", &encoded.token_ids);
        cases.push(WorkloadCase {
            selected: SelectedCase {
                case_id: format!(
                    "xr09_{}k_{}",
                    encoded.token_ids.len().div_ceil(1024),
                    encoded.record.workload_id
                ),
                workload_id: encoded.record.workload_id.clone(),
                family: encoded.record.family.clone(),
                prompt_path: encoded.record.prompt_path.clone(),
                prompt_sha256: encoded.prompt_sha256.clone(),
                source_deterministic_seed: encoded.record.deterministic_seed,
                target_context_tokens: encoded.record.target_context_tokens,
                actual_context_tokens: encoded.record.actual_context_tokens,
                context_tokens: encoded.token_ids.len(),
                prefix_token_hash,
            },
            token_ids: encoded.token_ids,
        });
    }
    Ok(cases)
}

fn encode_workload(
    tokenizer: &mut TokenizerHelper,
    record: &WorkloadRecord,
) -> Result<EncodedWorkload, CliError> {
    let prompt = fs::read_to_string(&record.prompt_path).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read prompt {}: {error}",
            record.prompt_path
        ))
    })?;
    let prompt_sha256 = sha256_hex(prompt.as_bytes());
    if prompt_sha256 != record.prompt_sha256 {
        return Err(CliError::Runtime(format!(
            "{} prompt sha mismatch: manifest={} actual={}",
            record.workload_id, record.prompt_sha256, prompt_sha256
        )));
    }
    let token_ids = tokenizer.encode(&prompt)?;
    if token_ids.len() != record.actual_context_tokens {
        return Err(CliError::Runtime(format!(
            "{} tokenizer length mismatch: manifest={} actual={}",
            record.workload_id,
            record.actual_context_tokens,
            token_ids.len()
        )));
    }
    Ok(EncodedWorkload {
        record: record.clone(),
        prompt_sha256,
        token_ids,
    })
}

fn startup_blockers(args: &Args) -> Vec<String> {
    let mut blockers = Vec::new();
    if !args.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if !args.workloads_path.exists() {
        blockers.push(format!(
            "workloads path does not exist: {}",
            args.workloads_path.display()
        ));
    }
    if !args.python.exists() {
        blockers.push(format!(
            "python path does not exist: {}",
            args.python.display()
        ));
    }
    if env::var("GEMMA4D_REQUIRE_MLX").ok().as_deref() != Some("1") {
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for real native XR09 evidence".to_owned());
    }
    if env::var("GEMMA4D_USE_NATIVE_GRAPH").ok().as_deref() != Some("1") {
        blockers.push(
            "GEMMA4D_USE_NATIVE_GRAPH=1 is required for real native XR09 evidence".to_owned(),
        );
    }
    blockers
}

fn target_config(args: &Args) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: env::var("GEMMA4D_MODEL_REVISION").ok(),
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(args.max_context_tokens as u32)
            .expect("max_context_tokens is non-zero"),
        allow_unsupported_config: false,
    }
}

fn capture_environment() -> Environment {
    let runtime = runtime_version().ok();
    Environment {
        machine: command_output("uname", &["-m"]),
        macos: command_output("sw_vers", &["-productVersion"]),
        rustc: command_output("rustc", &["--version"]),
        cargo: command_output("cargo", &["--version"]),
        runtime_backend: runtime
            .as_ref()
            .map(|version| version.backend_name.clone())
            .unwrap_or_else(|| "unavailable".to_owned()),
        runtime_backend_version: runtime
            .as_ref()
            .map(|version| version.backend_version.clone())
            .unwrap_or_else(|| "unavailable".to_owned()),
        git_commit: command_output("git", &["rev-parse", "--short", "HEAD"]),
        git_status_short: command_output_allow_empty("git", &["status", "--short"]),
        hw_memsize_bytes: sysctl_hw_memsize(),
    }
}

fn capture_relevant_environment() -> BTreeMap<String, Option<String>> {
    [
        "GEMMA4D_FULL_MODEL_TESTS",
        "GEMMA4D_MLX_LM_PYTHON",
        "GEMMA4D_MODEL_PATH",
        "GEMMA4D_MODEL_REVISION",
        "GEMMA4D_REQUIRE_MLX",
        "GEMMA4D_USE_NATIVE_GRAPH",
        "RUSTFLAGS",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), env::var(key).ok()))
    .collect()
}

fn capture_model_identity(model_path: &Path) -> ModelIdentity {
    let inventory = safetensors_inventory(model_path).unwrap_or(SafetensorsInventory {
        inventory_sha256: format!("unavailable:{}", model_path.display()),
        file_count: 0,
        total_bytes: 0,
    });
    ModelIdentity {
        model_path: model_path.display().to_string(),
        exists: model_path.exists(),
        configured_revision: env::var("GEMMA4D_MODEL_REVISION")
            .unwrap_or_else(|_| "unavailable:GEMMA4D_MODEL_REVISION not set".to_owned()),
        config_sha256: file_sha_or_unavailable(&model_path.join("config.json")),
        tokenizer_sha256: file_sha_or_unavailable(&model_path.join("tokenizer.json")),
        tokenizer_config_sha256: file_sha_or_unavailable(&model_path.join("tokenizer_config.json")),
        chat_template_sha256: file_sha_or_unavailable(&model_path.join("chat_template.json")),
        safetensors_inventory_sha256: inventory.inventory_sha256,
        safetensors_file_count: inventory.file_count,
        safetensors_total_bytes: inventory.total_bytes,
    }
}

fn safetensors_inventory(
    model_path: &Path,
) -> Result<SafetensorsInventory, Box<dyn std::error::Error>> {
    let mut files = fs::read_dir(model_path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("safetensors"))
        .collect::<Vec<_>>();
    files.sort();
    let mut input = Vec::new();
    let mut total_bytes = 0_u64;
    for path in &files {
        let metadata = fs::metadata(path)?;
        total_bytes += metadata.len();
        input.extend_from_slice(
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .as_bytes(),
        );
        input.push(0);
        input.extend_from_slice(file_sha256(path)?.as_bytes());
        input.push(0);
        input.extend_from_slice(metadata.len().to_string().as_bytes());
        input.push(0);
    }
    Ok(SafetensorsInventory {
        inventory_sha256: sha256_hex(&input),
        file_count: files.len(),
        total_bytes,
    })
}

fn file_sha_or_unavailable(path: &Path) -> String {
    file_sha256(path).unwrap_or_else(|error| format!("unavailable:{}: {error}", path.display()))
}

fn command_output(command: &str, args: &[&str]) -> String {
    let value = command_output_allow_empty(command, args);
    if value.is_empty() {
        "unavailable".to_owned()
    } else {
        value
    }
}

fn command_output_allow_empty(command: &str, args: &[&str]) -> String {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn sysctl_hw_memsize() -> Option<u64> {
    let output = command_output("sysctl", &["-n", "hw.memsize"]);
    output.parse::<u64>().ok()
}

fn run_id() -> String {
    format!("xr09-{}", unix_now())
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

fn option_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn command_line() -> String {
    env::args().collect::<Vec<_>>().join(" ")
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn token_hash(prefix: &str, token_ids: &[i32]) -> String {
    let mut input = Vec::with_capacity(prefix.len() + 1 + token_ids.len() * 5);
    input.extend_from_slice(prefix.as_bytes());
    input.push(0);
    for token in token_ids {
        input.extend_from_slice(&token.to_le_bytes());
        input.push(0);
    }
    sha256_hex(&input)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    percentile_sorted(&mut values, 0.5)
}

fn percentile(mut values: Vec<f64>, q: f64) -> Option<f64> {
    percentile_sorted(&mut values, q)
}

fn percentile_sorted(values: &mut Vec<f64>, q: f64) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let q = q.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * q).ceil() as usize;
    values.get(index).copied()
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

    fn shutdown(&mut self) {
        let _ = serde_json::to_writer(&mut self.stdin, &serde_json::json!({"cmd":"shutdown"}));
        let _ = writeln!(self.stdin);
        let _ = self.stdin.flush();
        let _ = self.child.try_wait();
    }
}

impl Drop for TokenizerHelper {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{name} must be > 0").into());
    }
    Ok(parsed)
}
