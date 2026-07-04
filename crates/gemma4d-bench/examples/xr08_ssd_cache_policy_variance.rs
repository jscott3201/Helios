use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Write},
    num::{NonZeroU32, NonZeroU64},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{CliError, manifest, workload_corpus::WorkloadRecord};
use gemma4d_ffi::{
    KvCache, KvMode, KvPolicy, KvSnapshot, LoadConfig, Target, decode_one, prefill, runtime_version,
};
use gemma4d_kv::{
    CacheMode, Error as KvError, KV_LAYOUT_VERSION, KvBlockKey, KvNamespace, PrefillObservation,
    RamPrefixBlock, SsdCacheAccountingSnapshot, SsdPrefixCache, SsdRestorePhase,
};
use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};

const GOAL: &str = "XR08-ssd-cache-policy-variance";
const MODE: &str = "native_ssd_cache_policy_variance";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR08-ssd-cache-policy-variance";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const DEFAULT_TRIALS: usize = 3;
const DEFAULT_MIN_PREFIX_TOKENS: usize = 8192;
const DEFAULT_MAX_CACHE_SIZE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const DEFAULT_SSD_METADATA_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 32_768;
const EXACT_LOGIT_TOLERANCE: f64 = 0.000_001;
const Q8_MAX_GREEDY_LOGIT_DELTA: f64 = 0.5;
const TTFT_IMPROVEMENT_GATE_PERCENT: f64 = 20.0;
const MAX_WARM_TTFT_CV: f64 = 0.35;
const MEMORY_CLIFF_GB: f64 = 14.0;
const ENV_KEYS: &[&str] = &["GEMMA4D_REQUIRE_MLX", "GEMMA4D_USE_NATIVE_GRAPH"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse(env::args().skip(1))?;
    fs::create_dir_all(&options.out_dir)?;
    fs::create_dir_all(&options.cache_dir)?;

    let records_path = options.out_dir.join("records.jsonl");
    let summary_path = options.out_dir.join("summary.json");
    let report_path = options.out_dir.join("report.md");
    let blockers_path = options.out_dir.join("blockers.md");
    let decision_path = options.out_dir.join("decision.md");

    let run_id = run_id();
    let git_sha =
        command_stdout("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let git_status_short =
        command_stdout("git", &["status", "--short"]).unwrap_or_else(|| "unknown".to_owned());
    let command = command_line();
    let environment = capture_environment(&options.python);
    let model_identity =
        manifest::capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION");
    let workloads = load_workloads(&options.workloads_path)?;
    let mut blockers = startup_blockers(&options);
    let mut tokenizer_backend = "not_started".to_owned();
    let mut selected_cases = Vec::new();
    let mut admission_probes = Vec::new();
    let mut records = Vec::new();

    if blockers.is_empty() {
        let mut tokenizer = TokenizerHelper::start(&options.python, &options.model_path)?;
        tokenizer_backend = tokenizer.backend().to_owned();
        selected_cases = prepare_cases(&options, &workloads, &mut tokenizer)?;
        admission_probes = prepare_static_admission_probes(&options, &workloads, &mut tokenizer)?;
        tokenizer.shutdown();

        let load_started = Instant::now();
        let target = Target::load(&target_config(&options));
        let model_load_ms = duration_ms(load_started.elapsed());
        match target {
            Ok(target) => {
                for case in &selected_cases {
                    for trial_index in 0..options.trials {
                        eprintln!(
                            "XR08 running context={} trial={} workload={} modes={:?}",
                            case.context_tokens(),
                            trial_index,
                            case.workload_id(),
                            options.modes
                        );
                        match run_case_trial(
                            &options,
                            &run_id,
                            &git_sha,
                            &git_status_short,
                            &command,
                            &model_identity,
                            &target,
                            case,
                            trial_index,
                            model_load_ms,
                        ) {
                            Ok(mut trial_records) => records.append(&mut trial_records),
                            Err(error) => records.push(failed_record(
                                &run_id,
                                &git_sha,
                                &git_status_short,
                                &command,
                                case,
                                options.modes.first().copied().unwrap_or(BenchMode::Bf16),
                                trial_index,
                                model_load_ms,
                                format!("case trial failed: {error}"),
                            )),
                        }
                    }
                }
            }
            Err(error) => {
                for case in &selected_cases {
                    for mode in &options.modes {
                        for trial_index in 0..options.trials {
                            records.push(failed_record(
                                &run_id,
                                &git_sha,
                                &git_status_short,
                                &command,
                                case,
                                *mode,
                                trial_index,
                                model_load_ms,
                                format!("target load failed: {error}"),
                            ));
                        }
                    }
                }
            }
        }
    }

    enrich_admission_probes_from_records(&options, &records, &mut admission_probes);
    let aggregates = build_aggregates(&records);
    let failed_hypotheses = failed_hypotheses(&records, &aggregates, &admission_probes);
    blockers.extend(blockers_for_records(&records, &selected_cases, &options));
    blockers.sort();
    blockers.dedup();
    let policy_decision = policy_decision_for(&blockers, &aggregates, &admission_probes);
    let decision = policy_decision.decision_label.clone();
    let status = if decision == "blocked_with_evidence" {
        "blocked"
    } else {
        "completed"
    };
    let selected_case_records = selected_cases
        .iter()
        .map(|case| case.selected.clone())
        .collect::<Vec<_>>();
    let generated_files = vec![
        records_path.display().to_string(),
        summary_path.display().to_string(),
        report_path.display().to_string(),
        blockers_path.display().to_string(),
        decision_path.display().to_string(),
    ];
    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        generated_at_unix_seconds: unix_now(),
        status: status.to_owned(),
        decision,
        run_id,
        git_sha,
        git_status_short,
        command,
        mode: MODE.to_owned(),
        environment,
        relevant_environment: relevant_environment(),
        model_identity,
        tokenizer_backend,
        workloads_path: options.workloads_path.display().to_string(),
        out_dir: options.out_dir.display().to_string(),
        cache_dir: options.cache_dir.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        decision_path: decision_path.display().to_string(),
        requested_trials: options.trials,
        contexts: options.contexts.clone(),
        modes: options.modes.iter().map(|mode| mode.label().to_owned()).collect(),
        min_prefix_tokens: options.min_prefix_tokens,
        max_cache_size_bytes: options.max_cache_size_bytes,
        ssd_metadata_budget_bytes: options.ssd_metadata_budget_bytes,
        exact_logit_tolerance: EXACT_LOGIT_TOLERANCE,
        q8_max_greedy_logit_delta: Q8_MAX_GREEDY_LOGIT_DELTA,
        ttft_improvement_gate_percent: TTFT_IMPROVEMENT_GATE_PERCENT,
        max_warm_ttft_cv: MAX_WARM_TTFT_CV,
        memory_cliff_gb: MEMORY_CLIFF_GB,
        selected_cases: selected_case_records,
        admission_probes,
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
        policy_decision,
        failed_hypotheses,
        blockers,
        generated_files,
        records,
        measurement_notes: vec![
            "fresh_prefill_ms is native prefill of the exact real-context prefix".to_owned(),
            "warm_ttft_ms includes SSD metadata restore, payload checksum, payload load, native snapshot import, and last-step retrieval".to_owned(),
            "BF16 payload records compression off; q8 payload records compression on for full-attention KV tensors with active decode restored as BF16".to_owned(),
            "payload population cost is reported separately and is not counted as warm TTFT".to_owned(),
            "mid-decode SSD restore is called only as an explicit rejection probe and must not read payload bytes".to_owned(),
            "admission probes are policy checks for minimum prefix tokens and max cache size; rejected probes do not execute model inference".to_owned(),
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR08 SSD cache policy variance: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR08 benchmark blocked; see blockers.md".into())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Options {
    out_dir: PathBuf,
    cache_dir: PathBuf,
    workloads_path: PathBuf,
    model_path: PathBuf,
    python: PathBuf,
    contexts: Vec<usize>,
    modes: Vec<BenchMode>,
    trials: usize,
    min_prefix_tokens: usize,
    max_cache_size_bytes: u64,
    ssd_metadata_budget_bytes: u64,
    max_context_tokens: usize,
}

impl Options {
    fn parse<I, S>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut cache_dir: Option<PathBuf> = None;
        let mut options = Self {
            out_dir: out_dir.clone(),
            cache_dir: out_dir.join("ssd-cache"),
            workloads_path: PathBuf::from(DEFAULT_WORKLOADS),
            model_path: PathBuf::from(DEFAULT_MODEL),
            python: PathBuf::from(DEFAULT_PYTHON),
            contexts: vec![8192, 16_384],
            modes: vec![BenchMode::Bf16, BenchMode::Q8],
            trials: DEFAULT_TRIALS,
            min_prefix_tokens: DEFAULT_MIN_PREFIX_TOKENS,
            max_cache_size_bytes: DEFAULT_MAX_CACHE_SIZE_BYTES,
            ssd_metadata_budget_bytes: DEFAULT_SSD_METADATA_BUDGET_BYTES,
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
        };
        let mut args = args.into_iter().map(Into::into).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?);
                    options.out_dir = out_dir.clone();
                    if cache_dir.is_none() {
                        options.cache_dir = out_dir.join("ssd-cache");
                    }
                }
                "--cache-dir" => {
                    let value = PathBuf::from(required_value(&mut args, "--cache-dir")?);
                    options.cache_dir = value.clone();
                    cache_dir = Some(value);
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
                "--clear-contexts" => options.contexts.clear(),
                "--context" => options.contexts.push(parse_positive_usize(
                    &required_value(&mut args, "--context")?,
                    "--context",
                )?),
                "--contexts" => {
                    options
                        .contexts
                        .extend(parse_csv_usize(&required_value(&mut args, "--contexts")?)?);
                }
                "--modes" => {
                    options.modes = parse_modes(&required_value(&mut args, "--modes")?)?;
                }
                "--trials" => {
                    options.trials =
                        parse_positive_usize(&required_value(&mut args, "--trials")?, "--trials")?
                }
                "--min-prefix-tokens" => {
                    options.min_prefix_tokens = parse_positive_usize(
                        &required_value(&mut args, "--min-prefix-tokens")?,
                        "--min-prefix-tokens",
                    )?
                }
                "--max-cache-size-bytes" => {
                    options.max_cache_size_bytes = parse_positive_u64(
                        &required_value(&mut args, "--max-cache-size-bytes")?,
                        "--max-cache-size-bytes",
                    )?
                }
                "--ssd-metadata-budget-bytes" => {
                    options.ssd_metadata_budget_bytes = parse_positive_u64(
                        &required_value(&mut args, "--ssd-metadata-budget-bytes")?,
                        "--ssd-metadata-budget-bytes",
                    )?
                }
                "--max-context-tokens" => {
                    options.max_context_tokens = parse_positive_usize(
                        &required_value(&mut args, "--max-context-tokens")?,
                        "--max-context-tokens",
                    )?
                }
                "-h" | "--help" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                other => return Err(CliError::Usage(format!("unknown option '{other}'"))),
            }
        }
        if options.contexts.is_empty() {
            return Err(CliError::Usage(
                "at least one context must be selected".to_owned(),
            ));
        }
        if options.modes.is_empty() {
            return Err(CliError::Usage(
                "at least one storage mode must be selected".to_owned(),
            ));
        }
        options.contexts.sort_unstable();
        options.contexts.dedup();
        if options
            .contexts
            .iter()
            .any(|context| *context > options.max_context_tokens)
        {
            return Err(CliError::Usage(
                "--context values cannot exceed --max-context-tokens".to_owned(),
            ));
        }
        Ok(options)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum BenchMode {
    Bf16,
    Q8,
}

impl BenchMode {
    fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "mlx_affine_q8",
        }
    }

    fn cache_mode(self) -> CacheMode {
        match self {
            Self::Bf16 => CacheMode::Bf16,
            Self::Q8 => CacheMode::MlxAffineQ8,
        }
    }

    fn ffi_mode(self) -> KvMode {
        match self {
            Self::Bf16 => KvMode::Bf16,
            Self::Q8 => KvMode::MlxAffineQ8,
        }
    }

    fn compression_enabled(self) -> bool {
        matches!(self, Self::Q8)
    }

    fn exact_required(self) -> bool {
        matches!(self, Self::Bf16)
    }

    fn logit_tolerance(self) -> f64 {
        match self {
            Self::Bf16 => EXACT_LOGIT_TOLERANCE,
            Self::Q8 => Q8_MAX_GREEDY_LOGIT_DELTA,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: String,
    generated_at_unix_seconds: u64,
    status: String,
    decision: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    mode: String,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    model_identity: manifest::ArtifactIdentity,
    tokenizer_backend: String,
    workloads_path: String,
    out_dir: String,
    cache_dir: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    requested_trials: usize,
    contexts: Vec<usize>,
    modes: Vec<String>,
    min_prefix_tokens: usize,
    max_cache_size_bytes: u64,
    ssd_metadata_budget_bytes: u64,
    exact_logit_tolerance: f64,
    q8_max_greedy_logit_delta: f64,
    ttft_improvement_gate_percent: f64,
    max_warm_ttft_cv: f64,
    memory_cliff_gb: f64,
    selected_cases: Vec<SelectedCase>,
    admission_probes: Vec<AdmissionProbe>,
    record_count: usize,
    passed_records: usize,
    failed_records: usize,
    aggregates: Vec<Aggregate>,
    policy_decision: PolicyDecision,
    failed_hypotheses: Vec<String>,
    blockers: Vec<String>,
    generated_files: Vec<String>,
    records: Vec<Record>,
    measurement_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Environment {
    machine: String,
    macos: String,
    rustc: String,
    cargo: String,
    runtime_backend: String,
    runtime_backend_version: String,
    tokenizer_mlx_version: String,
    hw_memsize_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
struct AdmissionProbe {
    probe: String,
    workload_id: Option<String>,
    context_tokens: Option<usize>,
    variant: Option<String>,
    threshold_bytes: Option<u64>,
    observed_bytes: Option<u64>,
    min_prefix_tokens: Option<usize>,
    admitted: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    schema_version: u32,
    goal: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    case_id: String,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    source_deterministic_seed: u64,
    trial_index: usize,
    variant: String,
    backend: String,
    cache_mode: String,
    compression_enabled: bool,
    config: BTreeMap<String, String>,
    context_tokens: usize,
    input_tokens: usize,
    prefix_token_hash: String,
    model_load_ms: f64,
    fresh_prefill: FreshPrefill,
    snapshot: SnapshotRecord,
    ssd_write: SsdWriteRecord,
    warm_restore: WarmRestore,
    continued_decode: ContinuedDecode,
    rejection: RejectionRecord,
    accounting: SsdCacheAccountingSnapshot,
    correctness: Correctness,
    gate: GateOutcome,
    status: String,
    blocker: Option<String>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FreshPrefill {
    ttft_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SnapshotRecord {
    export_ms: f64,
    sequence_len: u64,
    active_kv_bytes: u64,
    token_count: u64,
    has_last_step: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SsdWriteRecord {
    metadata_write_ms: f64,
    payload_save_ms: f64,
    payload_manifest_write_ms: f64,
    metadata_bytes_written: u64,
    payload_bytes_written: u64,
    total_entry_bytes: u64,
    payload_sha256: String,
    metadata_manifest_path: String,
    payload_manifest_path: String,
    payload_path: String,
    admitted_by_min_prefix_tokens: bool,
    admitted_by_max_cache_size: bool,
}

#[derive(Debug, Clone, Serialize)]
struct WarmRestore {
    ttft_ms: f64,
    metadata_restore_ms: f64,
    payload_checksum_ms: f64,
    payload_load_ms: f64,
    snapshot_import_last_step_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    payload_bytes_read: u64,
    metadata_bytes_read: u64,
    ttft_improvement_ms: f64,
    ttft_improvement_percent: f64,
    ttft_speedup: f64,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ContinuedDecode {
    fresh_decode_ms: f64,
    restored_decode_ms: f64,
    fresh_greedy_token: i32,
    restored_greedy_token: i32,
    fresh_greedy_logit: f32,
    restored_greedy_logit: f32,
    token_parity: bool,
    logit_delta: f64,
    sequence_len_parity: bool,
    active_kv_bytes_parity: bool,
}

#[derive(Debug, Clone, Serialize)]
struct RejectionRecord {
    wrong_namespace_rejected: bool,
    wrong_adapter_rejected: bool,
    wrong_cache_mode_rejected: bool,
    payload_corruption_rejected: bool,
    mid_decode_restore_rejected: bool,
    zero_mid_decode_fetches: bool,
}

#[derive(Debug, Clone, Serialize)]
struct Correctness {
    status: String,
    restored_last_token_parity: bool,
    restored_last_logit_delta: f64,
    restored_last_logit_tolerance: f64,
    restored_sequence_len_parity: bool,
    restored_active_kv_bytes_parity: bool,
    continued_decode_token_parity: bool,
    continued_decode_logit_delta: f64,
    continued_decode_logit_tolerance: f64,
    exact_required: bool,
}

#[derive(Debug, Clone, Serialize)]
struct GateOutcome {
    passed: bool,
    correctness_passed: bool,
    warm_ttft_improved: bool,
    p50_p95_evaluated_in_aggregate: bool,
    namespace_rejections: bool,
    corruption_rejection: bool,
    zero_mid_decode_fetches: bool,
    io_metrics_present: bool,
    admission_thresholds_passed: bool,
    memory_below_cliff: bool,
}

#[derive(Debug, Clone, Serialize)]
struct Aggregate {
    case_id: String,
    workload_id: String,
    family: String,
    context_tokens: usize,
    variant: String,
    trial_count: usize,
    passed_trials: usize,
    low_n: bool,
    fresh_prefill_p50_ms: Option<f64>,
    fresh_prefill_p95_ms: Option<f64>,
    warm_ttft_p50_ms: Option<f64>,
    warm_ttft_p95_ms: Option<f64>,
    warm_ttft_min_ms: Option<f64>,
    warm_ttft_max_ms: Option<f64>,
    warm_ttft_mean_ms: Option<f64>,
    warm_ttft_cv: Option<f64>,
    warm_p50_improvement_percent: Option<f64>,
    warm_p95_improvement_percent: Option<f64>,
    restore_latency_p50_ms: Option<f64>,
    metadata_restore_p50_ms: Option<f64>,
    payload_load_p50_ms: Option<f64>,
    payload_bytes_written_median: Option<f64>,
    metadata_bytes_written_median: Option<f64>,
    total_entry_bytes_median: Option<f64>,
    accounting_hit_rate_median: Option<f64>,
    peak_mlx_max_gb: Option<f64>,
    correctness_passed: bool,
    rejection_passed: bool,
    p50_gate_passed: bool,
    p95_gate_passed: bool,
    variance_documented: bool,
    variance_gate_passed: bool,
    memory_gate_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyDecision {
    decision_label: String,
    profile_policy: String,
    accepted_variants: Vec<String>,
    rejected_variants: Vec<String>,
    rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PayloadManifest {
    schema_version: u32,
    format: String,
    block_id: String,
    namespace_hash: String,
    cache_mode: String,
    compression_enabled: bool,
    kv_layout_version: u32,
    payload_relative_path: String,
    payload_sha256: String,
    payload_bytes: u64,
    active_kv_bytes: u64,
    token_count: u64,
    sequence_len: u64,
    has_safetensors_shape_metadata: bool,
}

#[allow(clippy::too_many_arguments)]
fn run_case_trial(
    options: &Options,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    model_identity: &manifest::ArtifactIdentity,
    target: &Target,
    case: &WorkloadCase,
    trial_index: usize,
    model_load_ms: f64,
) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let mut fresh_cache = KvCache::create(&KvPolicy::default())?;
    let fresh_started = Instant::now();
    let fresh_step = prefill(target, &mut fresh_cache, &case.token_ids)?;
    let fresh_prefill_ms = duration_ms(fresh_started.elapsed());
    let export_started = Instant::now();
    let snapshot = fresh_cache.export_snapshot()?;
    let snapshot_export_ms = duration_ms(export_started.elapsed());
    let snapshot_info = snapshot.info()?;
    let baseline_decode_started = Instant::now();
    let baseline_next = decode_one(target, &mut fresh_cache, fresh_step.greedy_token)?;
    let baseline_decode_ms = duration_ms(baseline_decode_started.elapsed());

    let mut out = Vec::new();
    for mode in &options.modes {
        out.push(run_variant_restore(
            options,
            run_id,
            git_sha,
            git_status_short,
            command,
            model_identity,
            target,
            case,
            trial_index,
            model_load_ms,
            *mode,
            &snapshot,
            snapshot_export_ms,
            &snapshot_info,
            &fresh_step,
            fresh_prefill_ms,
            &baseline_next,
            baseline_decode_ms,
        )?);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_variant_restore(
    options: &Options,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    model_identity: &manifest::ArtifactIdentity,
    target: &Target,
    case: &WorkloadCase,
    trial_index: usize,
    model_load_ms: f64,
    mode: BenchMode,
    snapshot: &KvSnapshot,
    snapshot_export_ms: f64,
    snapshot_info: &gemma4d_ffi::KvSnapshotInfo,
    fresh_step: &gemma4d_ffi::StepResult,
    fresh_prefill_ms: f64,
    baseline_next: &gemma4d_ffi::StepResult,
    baseline_decode_ms: f64,
) -> Result<Record, Box<dyn std::error::Error>> {
    let namespace = namespace_for(model_identity, &case.token_ids, mode)?;
    let namespace_hash = namespace.namespace_hash()?.0;
    let block_size =
        NonZeroU64::new(case.token_ids.len() as u64).expect("case token length is non-zero");
    let observation = PrefillObservation {
        sequence_len: fresh_step.sequence_len,
        greedy_token: fresh_step.greedy_token as u32,
        greedy_logit_bits: fresh_step.greedy_logit.to_bits(),
    };
    let block = RamPrefixBlock::from_observation(
        namespace.clone(),
        0,
        block_size,
        0,
        observation,
        snapshot_info.active_kv_bytes,
    )?
    .with_native_handle(native_handle_id(case.context_tokens(), trial_index, mode));
    let key = block.key.clone();
    let trial_dir = options.cache_dir.join(format!(
        "{}-trial-{trial_index}-{}",
        case.selected.case_id,
        mode.label()
    ));
    if trial_dir.exists() {
        fs::remove_dir_all(&trial_dir)?;
    }
    fs::create_dir_all(&trial_dir)?;
    let mut ssd_cache = SsdPrefixCache::open(
        &trial_dir,
        NonZeroU64::new(options.ssd_metadata_budget_bytes).expect("metadata budget is non-zero"),
    )?;

    let metadata_write_started = Instant::now();
    let entry = ssd_cache.write_block(&block)?;
    let metadata_write_ms = duration_ms(metadata_write_started.elapsed());
    let metadata_manifest_path = ssd_cache.entry_path(&entry);
    let metadata_bytes_written = fs::metadata(&metadata_manifest_path)?.len();

    let payload_dir = trial_dir.join("payloads");
    fs::create_dir_all(&payload_dir)?;
    let payload_path = payload_dir.join(format!("{}.safetensors", key.block_id.0));
    let payload_save_started = Instant::now();
    if mode.compression_enabled() {
        snapshot.save_compressed_to_path(&payload_path, mode.ffi_mode(), true, false)?;
    } else {
        snapshot.save_to_path(&payload_path)?;
    }
    let payload_save_ms = duration_ms(payload_save_started.elapsed());
    let payload_bytes = fs::metadata(&payload_path)?.len();
    let payload_sha256 = file_sha256(&payload_path)?;
    let total_entry_bytes = payload_bytes.saturating_add(metadata_bytes_written);
    let payload_manifest_path = payload_dir.join(format!("{}.manifest.json", key.block_id.0));
    let payload_manifest = PayloadManifest {
        schema_version: 1,
        format: "gemma4d_native_kv_snapshot_safetensors_v1".to_owned(),
        block_id: key.block_id.0.clone(),
        namespace_hash: namespace_hash.clone(),
        cache_mode: namespace.cache_mode.label().to_owned(),
        compression_enabled: mode.compression_enabled(),
        kv_layout_version: namespace.kv_layout_version,
        payload_relative_path: payload_path
            .strip_prefix(&trial_dir)
            .unwrap_or(&payload_path)
            .display()
            .to_string(),
        payload_sha256: payload_sha256.clone(),
        payload_bytes,
        active_kv_bytes: snapshot_info.active_kv_bytes,
        token_count: snapshot_info.token_count,
        sequence_len: snapshot_info.sequence_len,
        has_safetensors_shape_metadata: true,
    };
    let payload_manifest_started = Instant::now();
    fs::write(
        &payload_manifest_path,
        serde_json::to_vec_pretty(&payload_manifest)?,
    )?;
    let payload_manifest_write_ms = duration_ms(payload_manifest_started.elapsed());

    let metadata_restore_started = Instant::now();
    ssd_cache.restore_before_prefill(&key, &namespace)?;
    let metadata_restore_ms = duration_ms(metadata_restore_started.elapsed());
    let metadata_bytes_read = metadata_bytes_written;

    let payload_checksum_started = Instant::now();
    let verified_payload = load_payload_manifest(&payload_manifest_path, &trial_dir)?;
    let payload_checksum_ms = duration_ms(payload_checksum_started.elapsed());
    let payload_load_started = Instant::now();
    let loaded_snapshot = KvSnapshot::load_from_path(&verified_payload)?;
    let payload_load_ms = duration_ms(payload_load_started.elapsed());
    let mut restored_cache = KvCache::create(&policy_for_mode(mode))?;
    let import_started = Instant::now();
    restored_cache.import_snapshot(&loaded_snapshot)?;
    let restored_last = restored_cache.last_step()?;
    let import_last_step_ms = duration_ms(import_started.elapsed());
    let warm_ttft_ms =
        metadata_restore_ms + payload_checksum_ms + payload_load_ms + import_last_step_ms;

    let restored_decode_started = Instant::now();
    let restored_next = decode_one(target, &mut restored_cache, fresh_step.greedy_token)?;
    let restored_decode_ms = duration_ms(restored_decode_started.elapsed());

    let wrong_namespace_rejected =
        namespace_rejected(&mut ssd_cache, &key, wrong_model(&namespace));
    let wrong_adapter_rejected =
        namespace_rejected(&mut ssd_cache, &key, wrong_adapter(&namespace));
    let wrong_cache_mode_rejected = namespace_rejected(
        &mut ssd_cache,
        &key,
        namespace.clone().with_cache_mode(other_cache_mode(mode)),
    );
    let payload_corruption_rejected =
        payload_corruption_rejected(&payload_manifest_path, &payload_path, &trial_dir)?;
    let mid_decode_restore_rejected = matches!(
        ssd_cache.restore_for_phase(&key, &namespace, SsdRestorePhase::MidDecode),
        Err(KvError::InvalidBlock(_))
    );
    let accounting = ssd_cache.accounting();
    let last_logit_delta =
        (f64::from(fresh_step.greedy_logit) - f64::from(restored_last.greedy_logit)).abs();
    let decode_logit_delta =
        (f64::from(baseline_next.greedy_logit) - f64::from(restored_next.greedy_logit)).abs();
    let correctness_tolerance = mode.logit_tolerance();
    let restored_last_ok = fresh_step.greedy_token == restored_last.greedy_token
        && last_logit_delta <= correctness_tolerance
        && fresh_step.sequence_len == restored_last.sequence_len
        && fresh_step.active_kv_bytes == restored_last.active_kv_bytes;
    let continued_ok = baseline_next.greedy_token == restored_next.greedy_token
        && decode_logit_delta <= correctness_tolerance
        && baseline_next.sequence_len == restored_next.sequence_len
        && baseline_next.active_kv_bytes == restored_next.active_kv_bytes;
    let correctness_passed = restored_last_ok && continued_ok;
    let warm_improvement_ms = fresh_prefill_ms - warm_ttft_ms;
    let warm_improvement_percent = if fresh_prefill_ms > 0.0 {
        (warm_improvement_ms / fresh_prefill_ms) * 100.0
    } else {
        0.0
    };
    let peak_memory_gb = f64::from(fresh_step.peak_memory_gb)
        .max(f64::from(restored_last.peak_memory_gb))
        .max(f64::from(restored_next.peak_memory_gb));
    let memory_below_cliff = peak_memory_gb < MEMORY_CLIFF_GB;
    let rejection = RejectionRecord {
        wrong_namespace_rejected,
        wrong_adapter_rejected,
        wrong_cache_mode_rejected,
        payload_corruption_rejected,
        mid_decode_restore_rejected,
        zero_mid_decode_fetches: accounting.mid_decode_fetches == 0,
    };
    let gate = GateOutcome {
        passed: false,
        correctness_passed,
        warm_ttft_improved: warm_ttft_ms < fresh_prefill_ms,
        p50_p95_evaluated_in_aggregate: true,
        namespace_rejections: rejection.wrong_namespace_rejected
            && rejection.wrong_adapter_rejected
            && rejection.wrong_cache_mode_rejected,
        corruption_rejection: rejection.payload_corruption_rejected,
        zero_mid_decode_fetches: rejection.mid_decode_restore_rejected
            && rejection.zero_mid_decode_fetches,
        io_metrics_present: metadata_bytes_written > 0
            && payload_bytes > 0
            && accounting.bytes_written > 0
            && accounting.bytes_read > 0,
        admission_thresholds_passed: case.context_tokens() >= options.min_prefix_tokens
            && total_entry_bytes <= options.max_cache_size_bytes,
        memory_below_cliff,
    };
    let mut gate = gate;
    gate.passed = gate.correctness_passed
        && gate.warm_ttft_improved
        && gate.namespace_rejections
        && gate.corruption_rejection
        && gate.zero_mid_decode_fetches
        && gate.io_metrics_present
        && gate.admission_thresholds_passed
        && gate.memory_below_cliff;

    let mut notes = Vec::new();
    if !gate.correctness_passed {
        notes.push(format!(
            "{} restore correctness failed: last_delta={last_logit_delta:.6}, decode_delta={decode_logit_delta:.6}, tolerance={correctness_tolerance:.6}",
            mode.label()
        ));
    }
    if !gate.memory_below_cliff {
        notes.push(format!(
            "peak MLX memory {peak_memory_gb:.3} GB crossed {MEMORY_CLIFF_GB:.1} GB cliff"
        ));
    }
    let blocker = if gate.correctness_passed
        && gate.namespace_rejections
        && gate.corruption_rejection
        && gate.zero_mid_decode_fetches
        && gate.io_metrics_present
    {
        None
    } else {
        Some(format!(
            "{} {} trial {} failed a hard SSD restore gate",
            case.selected.case_id,
            mode.label(),
            trial_index
        ))
    };

    Ok(Record {
        schema_version: 1,
        goal: GOAL.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        command: command.to_owned(),
        case_id: case.selected.case_id.clone(),
        workload_id: case.selected.workload_id.clone(),
        family: case.selected.family.clone(),
        prompt_path: case.selected.prompt_path.clone(),
        prompt_sha256: case.selected.prompt_sha256.clone(),
        source_deterministic_seed: case.selected.source_deterministic_seed,
        trial_index,
        variant: mode.label().to_owned(),
        backend: "native".to_owned(),
        cache_mode: namespace.cache_mode.label().to_owned(),
        compression_enabled: mode.compression_enabled(),
        config: config_for_record(options, mode),
        context_tokens: case.selected.context_tokens,
        input_tokens: case.token_ids.len(),
        prefix_token_hash: case.selected.prefix_token_hash.clone(),
        model_load_ms,
        fresh_prefill: FreshPrefill {
            ttft_ms: fresh_prefill_ms,
            greedy_token: fresh_step.greedy_token,
            greedy_logit: fresh_step.greedy_logit,
            sequence_len: fresh_step.sequence_len,
            active_kv_bytes: fresh_step.active_kv_bytes,
            peak_memory_gb: f64::from(fresh_step.peak_memory_gb),
            peak_rss_mb: f64::from(fresh_step.peak_rss_mb),
        },
        snapshot: SnapshotRecord {
            export_ms: snapshot_export_ms,
            sequence_len: snapshot_info.sequence_len,
            active_kv_bytes: snapshot_info.active_kv_bytes,
            token_count: snapshot_info.token_count,
            has_last_step: snapshot_info.has_last_step,
        },
        ssd_write: SsdWriteRecord {
            metadata_write_ms,
            payload_save_ms,
            payload_manifest_write_ms,
            metadata_bytes_written,
            payload_bytes_written: payload_bytes,
            total_entry_bytes,
            payload_sha256,
            metadata_manifest_path: metadata_manifest_path.display().to_string(),
            payload_manifest_path: payload_manifest_path.display().to_string(),
            payload_path: payload_path.display().to_string(),
            admitted_by_min_prefix_tokens: case.context_tokens() >= options.min_prefix_tokens,
            admitted_by_max_cache_size: total_entry_bytes <= options.max_cache_size_bytes,
        },
        warm_restore: WarmRestore {
            ttft_ms: warm_ttft_ms,
            metadata_restore_ms,
            payload_checksum_ms,
            payload_load_ms,
            snapshot_import_last_step_ms: import_last_step_ms,
            greedy_token: restored_last.greedy_token,
            greedy_logit: restored_last.greedy_logit,
            sequence_len: restored_last.sequence_len,
            active_kv_bytes: restored_last.active_kv_bytes,
            payload_bytes_read: payload_bytes,
            metadata_bytes_read,
            ttft_improvement_ms: warm_improvement_ms,
            ttft_improvement_percent: warm_improvement_percent,
            ttft_speedup: if warm_ttft_ms > 0.0 {
                fresh_prefill_ms / warm_ttft_ms
            } else {
                0.0
            },
            peak_memory_gb: f64::from(restored_last.peak_memory_gb)
                .max(f64::from(restored_next.peak_memory_gb)),
            peak_rss_mb: f64::from(restored_last.peak_rss_mb)
                .max(f64::from(restored_next.peak_rss_mb)),
        },
        continued_decode: ContinuedDecode {
            fresh_decode_ms: baseline_decode_ms,
            restored_decode_ms,
            fresh_greedy_token: baseline_next.greedy_token,
            restored_greedy_token: restored_next.greedy_token,
            fresh_greedy_logit: baseline_next.greedy_logit,
            restored_greedy_logit: restored_next.greedy_logit,
            token_parity: baseline_next.greedy_token == restored_next.greedy_token,
            logit_delta: decode_logit_delta,
            sequence_len_parity: baseline_next.sequence_len == restored_next.sequence_len,
            active_kv_bytes_parity: baseline_next.active_kv_bytes == restored_next.active_kv_bytes,
        },
        rejection,
        accounting,
        correctness: Correctness {
            status: if correctness_passed {
                "passed".to_owned()
            } else {
                "failed".to_owned()
            },
            restored_last_token_parity: fresh_step.greedy_token == restored_last.greedy_token,
            restored_last_logit_delta: last_logit_delta,
            restored_last_logit_tolerance: correctness_tolerance,
            restored_sequence_len_parity: fresh_step.sequence_len == restored_last.sequence_len,
            restored_active_kv_bytes_parity: fresh_step.active_kv_bytes
                == restored_last.active_kv_bytes,
            continued_decode_token_parity: baseline_next.greedy_token == restored_next.greedy_token,
            continued_decode_logit_delta: decode_logit_delta,
            continued_decode_logit_tolerance: correctness_tolerance,
            exact_required: mode.exact_required(),
        },
        gate,
        status: "passed".to_owned(),
        blocker,
        notes,
    })
}

#[allow(clippy::too_many_arguments)]
fn failed_record(
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    case: &WorkloadCase,
    mode: BenchMode,
    trial_index: usize,
    model_load_ms: f64,
    blocker: String,
) -> Record {
    Record {
        schema_version: 1,
        goal: GOAL.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        command: command.to_owned(),
        case_id: case.selected.case_id.clone(),
        workload_id: case.selected.workload_id.clone(),
        family: case.selected.family.clone(),
        prompt_path: case.selected.prompt_path.clone(),
        prompt_sha256: case.selected.prompt_sha256.clone(),
        source_deterministic_seed: case.selected.source_deterministic_seed,
        trial_index,
        variant: mode.label().to_owned(),
        backend: "native".to_owned(),
        cache_mode: mode.cache_mode().label().to_owned(),
        compression_enabled: mode.compression_enabled(),
        config: BTreeMap::new(),
        context_tokens: case.selected.context_tokens,
        input_tokens: case.token_ids.len(),
        prefix_token_hash: case.selected.prefix_token_hash.clone(),
        model_load_ms,
        fresh_prefill: FreshPrefill::empty(),
        snapshot: SnapshotRecord::empty(),
        ssd_write: SsdWriteRecord::empty(),
        warm_restore: WarmRestore::empty(),
        continued_decode: ContinuedDecode::empty(),
        rejection: RejectionRecord::empty(),
        accounting: empty_accounting(),
        correctness: Correctness::failed(mode),
        gate: GateOutcome::failed(),
        status: "failed".to_owned(),
        blocker: Some(blocker),
        notes: Vec::new(),
    }
}

impl FreshPrefill {
    fn empty() -> Self {
        Self {
            ttft_ms: 0.0,
            greedy_token: 0,
            greedy_logit: 0.0,
            sequence_len: 0,
            active_kv_bytes: 0,
            peak_memory_gb: 0.0,
            peak_rss_mb: 0.0,
        }
    }
}

impl SnapshotRecord {
    fn empty() -> Self {
        Self {
            export_ms: 0.0,
            sequence_len: 0,
            active_kv_bytes: 0,
            token_count: 0,
            has_last_step: false,
        }
    }
}

impl SsdWriteRecord {
    fn empty() -> Self {
        Self {
            metadata_write_ms: 0.0,
            payload_save_ms: 0.0,
            payload_manifest_write_ms: 0.0,
            metadata_bytes_written: 0,
            payload_bytes_written: 0,
            total_entry_bytes: 0,
            payload_sha256: "unavailable".to_owned(),
            metadata_manifest_path: "unavailable".to_owned(),
            payload_manifest_path: "unavailable".to_owned(),
            payload_path: "unavailable".to_owned(),
            admitted_by_min_prefix_tokens: false,
            admitted_by_max_cache_size: false,
        }
    }
}

impl WarmRestore {
    fn empty() -> Self {
        Self {
            ttft_ms: 0.0,
            metadata_restore_ms: 0.0,
            payload_checksum_ms: 0.0,
            payload_load_ms: 0.0,
            snapshot_import_last_step_ms: 0.0,
            greedy_token: 0,
            greedy_logit: 0.0,
            sequence_len: 0,
            active_kv_bytes: 0,
            payload_bytes_read: 0,
            metadata_bytes_read: 0,
            ttft_improvement_ms: 0.0,
            ttft_improvement_percent: 0.0,
            ttft_speedup: 0.0,
            peak_memory_gb: 0.0,
            peak_rss_mb: 0.0,
        }
    }
}

impl ContinuedDecode {
    fn empty() -> Self {
        Self {
            fresh_decode_ms: 0.0,
            restored_decode_ms: 0.0,
            fresh_greedy_token: 0,
            restored_greedy_token: 0,
            fresh_greedy_logit: 0.0,
            restored_greedy_logit: 0.0,
            token_parity: false,
            logit_delta: f64::INFINITY,
            sequence_len_parity: false,
            active_kv_bytes_parity: false,
        }
    }
}

impl RejectionRecord {
    fn empty() -> Self {
        Self {
            wrong_namespace_rejected: false,
            wrong_adapter_rejected: false,
            wrong_cache_mode_rejected: false,
            payload_corruption_rejected: false,
            mid_decode_restore_rejected: false,
            zero_mid_decode_fetches: false,
        }
    }
}

impl Correctness {
    fn failed(mode: BenchMode) -> Self {
        Self {
            status: "failed".to_owned(),
            restored_last_token_parity: false,
            restored_last_logit_delta: f64::INFINITY,
            restored_last_logit_tolerance: mode.logit_tolerance(),
            restored_sequence_len_parity: false,
            restored_active_kv_bytes_parity: false,
            continued_decode_token_parity: false,
            continued_decode_logit_delta: f64::INFINITY,
            continued_decode_logit_tolerance: mode.logit_tolerance(),
            exact_required: mode.exact_required(),
        }
    }
}

impl GateOutcome {
    fn failed() -> Self {
        Self {
            passed: false,
            correctness_passed: false,
            warm_ttft_improved: false,
            p50_p95_evaluated_in_aggregate: false,
            namespace_rejections: false,
            corruption_rejection: false,
            zero_mid_decode_fetches: false,
            io_metrics_present: false,
            admission_thresholds_passed: false,
            memory_below_cliff: false,
        }
    }
}

fn empty_accounting() -> SsdCacheAccountingSnapshot {
    SsdCacheAccountingSnapshot {
        budget_bytes: 0,
        stored_bytes: 0,
        stored_blocks: 0,
        hits: 0,
        misses: 0,
        writes: 0,
        reads: 0,
        evictions: 0,
        restore_failures: 0,
        namespace_rejections: 0,
        corruptions: 0,
        bytes_written: 0,
        bytes_read: 0,
        hit_rate: 0.0,
        mid_decode_fetches: 0,
        ssd_enabled: false,
    }
}

fn prepare_cases(
    options: &Options,
    workloads: &[WorkloadRecord],
    tokenizer: &mut TokenizerHelper,
) -> Result<Vec<WorkloadCase>, CliError> {
    let mut encoded = BTreeMap::new();
    for workload_id in required_workload_ids(&options.contexts) {
        let record = workloads
            .iter()
            .find(|record| record.workload_id == workload_id)
            .ok_or_else(|| {
                CliError::Runtime(format!(
                    "required workload {workload_id} missing from {}",
                    options.workloads_path.display()
                ))
            })?;
        encoded.insert(workload_id.to_owned(), encode_workload(tokenizer, record)?);
    }
    let mut cases = Vec::with_capacity(options.contexts.len());
    for context_tokens in &options.contexts {
        let workload_id = source_workload_id(*context_tokens)?;
        let source = encoded.get(workload_id).ok_or_else(|| {
            CliError::Runtime(format!("required workload {workload_id} was not encoded"))
        })?;
        if source.token_ids.len() < *context_tokens {
            return Err(CliError::Runtime(format!(
                "{} has {} tokens, fewer than requested context {}",
                source.record.workload_id,
                source.token_ids.len(),
                context_tokens
            )));
        }
        let token_ids = source.token_ids[..*context_tokens].to_vec();
        let prefix_token_hash = token_hash("xr08-prefix-token-ids-v1", &token_ids);
        cases.push(WorkloadCase {
            selected: SelectedCase {
                case_id: format!(
                    "xr08_{}k_{}",
                    context_tokens / 1024,
                    source.record.workload_id
                ),
                workload_id: source.record.workload_id.clone(),
                family: source.record.family.clone(),
                prompt_path: source.record.prompt_path.clone(),
                prompt_sha256: source.prompt_sha256.clone(),
                source_deterministic_seed: source.record.deterministic_seed,
                target_context_tokens: source.record.target_context_tokens,
                actual_context_tokens: source.record.actual_context_tokens,
                context_tokens: *context_tokens,
                prefix_token_hash,
            },
            token_ids,
        });
    }
    Ok(cases)
}

fn prepare_static_admission_probes(
    options: &Options,
    workloads: &[WorkloadRecord],
    tokenizer: &mut TokenizerHelper,
) -> Result<Vec<AdmissionProbe>, CliError> {
    let mut probes = Vec::new();
    let Some(record) = workloads
        .iter()
        .find(|record| record.workload_id == "code_review_rust_4k_001")
    else {
        return Ok(probes);
    };
    let encoded = encode_workload(tokenizer, record)?;
    probes.push(AdmissionProbe {
        probe: "min_prefix_tokens".to_owned(),
        workload_id: Some(record.workload_id.clone()),
        context_tokens: Some(encoded.token_ids.len()),
        variant: None,
        threshold_bytes: None,
        observed_bytes: None,
        min_prefix_tokens: Some(options.min_prefix_tokens),
        admitted: encoded.token_ids.len() >= options.min_prefix_tokens,
        reason: if encoded.token_ids.len() >= options.min_prefix_tokens {
            format!(
                "{} tokens meet min_prefix_tokens {}",
                encoded.token_ids.len(),
                options.min_prefix_tokens
            )
        } else {
            format!(
                "{} tokens below min_prefix_tokens {}; SSD admission rejected",
                encoded.token_ids.len(),
                options.min_prefix_tokens
            )
        },
    });
    Ok(probes)
}

fn enrich_admission_probes_from_records(
    options: &Options,
    records: &[Record],
    admission_probes: &mut Vec<AdmissionProbe>,
) {
    let largest = records
        .iter()
        .filter(|record| record.status == "passed")
        .max_by_key(|record| record.ssd_write.total_entry_bytes);
    if let Some(record) = largest {
        let probe_limit = record.ssd_write.total_entry_bytes.saturating_sub(1);
        admission_probes.push(AdmissionProbe {
            probe: "max_cache_size_rejection".to_owned(),
            workload_id: Some(record.workload_id.clone()),
            context_tokens: Some(record.context_tokens),
            variant: Some(record.variant.clone()),
            threshold_bytes: Some(probe_limit),
            observed_bytes: Some(record.ssd_write.total_entry_bytes),
            min_prefix_tokens: None,
            admitted: record.ssd_write.total_entry_bytes <= probe_limit,
            reason: format!(
                "observed entry {} bytes exceeds probe cap {} bytes; SSD admission rejected",
                record.ssd_write.total_entry_bytes, probe_limit
            ),
        });
        admission_probes.push(AdmissionProbe {
            probe: "configured_max_cache_size".to_owned(),
            workload_id: Some(record.workload_id.clone()),
            context_tokens: Some(record.context_tokens),
            variant: Some(record.variant.clone()),
            threshold_bytes: Some(options.max_cache_size_bytes),
            observed_bytes: Some(record.ssd_write.total_entry_bytes),
            min_prefix_tokens: None,
            admitted: record.ssd_write.total_entry_bytes <= options.max_cache_size_bytes,
            reason: if record.ssd_write.total_entry_bytes <= options.max_cache_size_bytes {
                format!(
                    "largest observed entry {} bytes fits configured cap {} bytes",
                    record.ssd_write.total_entry_bytes, options.max_cache_size_bytes
                )
            } else {
                format!(
                    "largest observed entry {} bytes exceeds configured cap {} bytes",
                    record.ssd_write.total_entry_bytes, options.max_cache_size_bytes
                )
            },
        });
    }
}

fn source_workload_id(context_tokens: usize) -> Result<&'static str, CliError> {
    match context_tokens {
        8192 => Ok("prefix_reuse_edit_8k_a_001"),
        16_384 => Ok("long_repo_pack_16k_001"),
        other => Err(CliError::Usage(format!(
            "XR08 default real SSD policy matrix supports 8K and 16K contexts; unsupported context {other}"
        ))),
    }
}

fn required_workload_ids(contexts: &[usize]) -> BTreeSet<&'static str> {
    contexts
        .iter()
        .filter_map(|context| source_workload_id(*context).ok())
        .collect()
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

fn build_aggregates(records: &[Record]) -> Vec<Aggregate> {
    let keys = records
        .iter()
        .map(|record| {
            (
                record.case_id.clone(),
                record.workload_id.clone(),
                record.family.clone(),
                record.context_tokens,
                record.variant.clone(),
            )
        })
        .collect::<BTreeSet<_>>();
    let mut out = Vec::new();
    for (case_id, workload_id, family, context_tokens, variant) in keys {
        let group = records
            .iter()
            .filter(|record| record.case_id == case_id && record.variant == variant)
            .collect::<Vec<_>>();
        let passed = group
            .iter()
            .copied()
            .filter(|record| record.status == "passed")
            .collect::<Vec<_>>();
        let fresh = passed
            .iter()
            .map(|record| record.fresh_prefill.ttft_ms)
            .collect::<Vec<_>>();
        let warm = passed
            .iter()
            .map(|record| record.warm_restore.ttft_ms)
            .collect::<Vec<_>>();
        let restore = passed
            .iter()
            .map(|record| record.warm_restore.ttft_ms)
            .collect::<Vec<_>>();
        let metadata_restore = passed
            .iter()
            .map(|record| record.warm_restore.metadata_restore_ms)
            .collect::<Vec<_>>();
        let payload_load = passed
            .iter()
            .map(|record| record.warm_restore.payload_load_ms)
            .collect::<Vec<_>>();
        let payload_bytes = passed
            .iter()
            .map(|record| record.ssd_write.payload_bytes_written as f64)
            .collect::<Vec<_>>();
        let metadata_bytes = passed
            .iter()
            .map(|record| record.ssd_write.metadata_bytes_written as f64)
            .collect::<Vec<_>>();
        let total_bytes = passed
            .iter()
            .map(|record| record.ssd_write.total_entry_bytes as f64)
            .collect::<Vec<_>>();
        let hit_rate = passed
            .iter()
            .map(|record| record.accounting.hit_rate)
            .collect::<Vec<_>>();
        let peak = passed
            .iter()
            .map(|record| {
                record
                    .fresh_prefill
                    .peak_memory_gb
                    .max(record.warm_restore.peak_memory_gb)
            })
            .collect::<Vec<_>>();
        let fresh_p50 = percentile(fresh.clone(), 0.50);
        let fresh_p95 = percentile(fresh, 0.95);
        let warm_p50 = percentile(warm.clone(), 0.50);
        let warm_p95 = percentile(warm.clone(), 0.95);
        let warm_p50_improvement_percent = percent_improvement(fresh_p50, warm_p50);
        let warm_p95_improvement_percent = percent_improvement(fresh_p95, warm_p95);
        let warm_cv = coefficient_of_variation(&warm);
        let correctness_passed = !group.is_empty()
            && group
                .iter()
                .all(|record| record.status == "passed" && record.gate.correctness_passed);
        let rejection_passed = !group.is_empty()
            && group.iter().all(|record| {
                record.status == "passed"
                    && record.gate.namespace_rejections
                    && record.gate.corruption_rejection
                    && record.gate.zero_mid_decode_fetches
            });
        let p50_gate_passed = warm_p50_improvement_percent
            .map(|value| value >= TTFT_IMPROVEMENT_GATE_PERCENT)
            .unwrap_or(false);
        let p95_gate_passed = warm_p95_improvement_percent
            .map(|value| value >= TTFT_IMPROVEMENT_GATE_PERCENT)
            .unwrap_or(false);
        let variance_documented = passed.len() >= 2 && warm_cv.is_some();
        let variance_gate_passed = warm_cv.map(|cv| cv <= MAX_WARM_TTFT_CV).unwrap_or(false);
        let peak_mlx_max_gb = max_value(&peak);
        let memory_gate_passed = peak_mlx_max_gb
            .map(|value| value < MEMORY_CLIFF_GB)
            .unwrap_or(false);
        out.push(Aggregate {
            case_id,
            workload_id,
            family,
            context_tokens,
            variant,
            trial_count: group.len(),
            passed_trials: passed.len(),
            low_n: passed.len() < 3,
            fresh_prefill_p50_ms: fresh_p50,
            fresh_prefill_p95_ms: fresh_p95,
            warm_ttft_p50_ms: warm_p50,
            warm_ttft_p95_ms: warm_p95,
            warm_ttft_min_ms: min_value(&warm),
            warm_ttft_max_ms: max_value(&warm),
            warm_ttft_mean_ms: mean_value(&warm),
            warm_ttft_cv: warm_cv,
            warm_p50_improvement_percent,
            warm_p95_improvement_percent,
            restore_latency_p50_ms: percentile(restore, 0.50),
            metadata_restore_p50_ms: percentile(metadata_restore, 0.50),
            payload_load_p50_ms: percentile(payload_load, 0.50),
            payload_bytes_written_median: percentile(payload_bytes, 0.50),
            metadata_bytes_written_median: percentile(metadata_bytes, 0.50),
            total_entry_bytes_median: percentile(total_bytes, 0.50),
            accounting_hit_rate_median: percentile(hit_rate, 0.50),
            peak_mlx_max_gb,
            correctness_passed,
            rejection_passed,
            p50_gate_passed,
            p95_gate_passed,
            variance_documented,
            variance_gate_passed,
            memory_gate_passed,
        });
    }
    out
}

fn policy_decision_for(
    blockers: &[String],
    aggregates: &[Aggregate],
    admission_probes: &[AdmissionProbe],
) -> PolicyDecision {
    if !blockers.is_empty() || aggregates.is_empty() {
        return PolicyDecision {
            decision_label: "blocked_with_evidence".to_owned(),
            profile_policy: "keep_ssd_prefix_cache_disabled".to_owned(),
            accepted_variants: Vec::new(),
            rejected_variants: aggregates
                .iter()
                .map(|aggregate| format!("{}:{}", aggregate.case_id, aggregate.variant))
                .collect(),
            rationale:
                "hard restore, rejection, or artifact completeness blocker prevents policy decision"
                    .to_owned(),
        };
    }
    let admission_ok = admission_probes
        .iter()
        .all(|probe| match probe.probe.as_str() {
            "min_prefix_tokens" | "max_cache_size_rejection" => !probe.admitted,
            "configured_max_cache_size" => probe.admitted,
            _ => true,
        });
    let accepted = aggregates
        .iter()
        .filter(|aggregate| {
            aggregate.correctness_passed
                && aggregate.rejection_passed
                && aggregate.p50_gate_passed
                && aggregate.p95_gate_passed
                && aggregate.variance_documented
                && aggregate.variance_gate_passed
                && aggregate.memory_gate_passed
                && admission_ok
        })
        .map(|aggregate| format!("{}:{}", aggregate.case_id, aggregate.variant))
        .collect::<Vec<_>>();
    let rejected = aggregates
        .iter()
        .filter(|aggregate| {
            !(aggregate.correctness_passed
                && aggregate.rejection_passed
                && aggregate.p50_gate_passed
                && aggregate.p95_gate_passed
                && aggregate.variance_documented
                && aggregate.variance_gate_passed
                && aggregate.memory_gate_passed
                && admission_ok)
        })
        .map(|aggregate| format!("{}:{}", aggregate.case_id, aggregate.variant))
        .collect::<Vec<_>>();
    if accepted.is_empty() {
        PolicyDecision {
            decision_label: "reject_candidate".to_owned(),
            profile_policy: "keep_ssd_prefix_cache_disabled".to_owned(),
            accepted_variants: accepted,
            rejected_variants: rejected,
            rationale: "no measured SSD profile satisfied correctness, p50/p95 TTFT, variance, memory, and admission gates".to_owned(),
        }
    } else if rejected.is_empty() {
        PolicyDecision {
            decision_label: "accept_candidate".to_owned(),
            profile_policy: "ssd_prefix_cache_opt_in_for_tiny16_8k_16k_with_min_prefix_and_cache_cap".to_owned(),
            accepted_variants: accepted,
            rejected_variants: rejected,
            rationale: "all measured SSD profiles satisfied correctness, p50/p95 TTFT, variance, memory, and admission gates; policy remains opt-in/profile-gated rather than production-default".to_owned(),
        }
    } else {
        PolicyDecision {
            decision_label: "keep_experimental".to_owned(),
            profile_policy: "ssd_prefix_cache_opt_in_only_for_accepted_profiles".to_owned(),
            accepted_variants: accepted,
            rejected_variants: rejected,
            rationale: "some SSD profiles satisfied the gates while others did not; keep policy profile-gated and experimental".to_owned(),
        }
    }
}

fn failed_hypotheses(
    records: &[Record],
    aggregates: &[Aggregate],
    admission_probes: &[AdmissionProbe],
) -> Vec<String> {
    let mut out = Vec::new();
    for record in records {
        if record.status != "passed" {
            out.push(format!(
                "{} {} trial {} failed: {}",
                record.case_id,
                record.variant,
                record.trial_index,
                record.blocker.as_deref().unwrap_or("no blocker detail")
            ));
        }
        for note in &record.notes {
            out.push(format!(
                "{} {} trial {}: {}",
                record.case_id, record.variant, record.trial_index, note
            ));
        }
    }
    for aggregate in aggregates {
        if !aggregate.p50_gate_passed || !aggregate.p95_gate_passed {
            out.push(format!(
                "{} {} did not meet {:.1}% p50/p95 warm TTFT gate",
                aggregate.case_id, aggregate.variant, TTFT_IMPROVEMENT_GATE_PERCENT
            ));
        }
        if !aggregate.variance_gate_passed {
            out.push(format!(
                "{} {} warm TTFT variance exceeded CV gate or was unavailable",
                aggregate.case_id, aggregate.variant
            ));
        }
        if !aggregate.memory_gate_passed {
            out.push(format!(
                "{} {} exceeded {:.1} GB memory gate or memory was unavailable",
                aggregate.case_id, aggregate.variant, MEMORY_CLIFF_GB
            ));
        }
        if aggregate.low_n {
            out.push(format!(
                "{} {} is low-N evidence: {}/{} passed trials",
                aggregate.case_id,
                aggregate.variant,
                aggregate.passed_trials,
                aggregate.trial_count
            ));
        }
    }
    for probe in admission_probes {
        out.push(format!(
            "admission probe {} admitted={}: {}",
            probe.probe, probe.admitted, probe.reason
        ));
    }
    out.sort();
    out.dedup();
    out
}

fn blockers_for_records(
    records: &[Record],
    selected_cases: &[WorkloadCase],
    options: &Options,
) -> Vec<String> {
    let mut blockers = Vec::new();
    for case in selected_cases {
        for mode in &options.modes {
            let group = records
                .iter()
                .filter(|record| {
                    record.case_id == case.selected.case_id && record.variant == mode.label()
                })
                .collect::<Vec<_>>();
            if group.len() != options.trials {
                blockers.push(format!(
                    "{} {} has {} records; expected {}",
                    case.selected.case_id,
                    mode.label(),
                    group.len(),
                    options.trials
                ));
            }
            for record in group {
                if record.status != "passed" {
                    blockers.push(format!(
                        "{} {} trial {} failed: {}",
                        record.case_id,
                        record.variant,
                        record.trial_index,
                        record.blocker.as_deref().unwrap_or("no blocker detail")
                    ));
                } else if let Some(blocker) = &record.blocker {
                    blockers.push(blocker.clone());
                }
            }
        }
    }
    blockers
}

fn load_payload_manifest(
    manifest_path: &Path,
    cache_root: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest: PayloadManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    let payload_path = cache_root.join(&manifest.payload_relative_path);
    let actual_sha256 = file_sha256(&payload_path)?;
    if actual_sha256 != manifest.payload_sha256 {
        return Err(format!(
            "payload checksum mismatch for {}: expected {}, got {}",
            payload_path.display(),
            manifest.payload_sha256,
            actual_sha256
        )
        .into());
    }
    let actual_bytes = fs::metadata(&payload_path)?.len();
    if actual_bytes != manifest.payload_bytes {
        return Err(format!(
            "payload byte length mismatch for {}: expected {}, got {}",
            payload_path.display(),
            manifest.payload_bytes,
            actual_bytes
        )
        .into());
    }
    Ok(payload_path)
}

fn payload_corruption_rejected(
    manifest_path: &Path,
    payload_path: &Path,
    cache_root: &Path,
) -> Result<bool, Box<dyn std::error::Error>> {
    let corrupt_path = payload_path.with_extension("corrupt.safetensors");
    let mut bytes = fs::read(payload_path)?;
    if bytes.is_empty() {
        return Ok(false);
    }
    bytes[0] ^= 0xff;
    fs::write(&corrupt_path, bytes)?;
    let mut manifest: PayloadManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    manifest.payload_relative_path = corrupt_path
        .strip_prefix(cache_root)
        .unwrap_or(&corrupt_path)
        .display()
        .to_string();
    let corrupt_manifest_path = manifest_path.with_extension("corrupt.json");
    fs::write(
        &corrupt_manifest_path,
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(load_payload_manifest(&corrupt_manifest_path, cache_root).is_err())
}

fn namespace_for(
    model_identity: &manifest::ArtifactIdentity,
    prefix_tokens: &[i32],
    mode: BenchMode,
) -> Result<KvNamespace, Box<dyn std::error::Error>> {
    let version = runtime_version()?;
    Ok(KvNamespace {
        model_id: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
        model_revision: model_identity
            .revision
            .clone()
            .unwrap_or_else(|| model_identity.revision_source.clone()),
        weights_sha256: model_identity.safetensors_inventory_sha256.clone(),
        quantization_sha256: quantization_hash(model_identity),
        tokenizer_sha256: model_identity.tokenizer_sha256.clone(),
        chat_template_sha256: model_identity.chat_template_sha256.clone(),
        prompt_token_hash: token_hash("xr08-prefix-token-ids-v1", prefix_tokens),
        raw_prompt_hash: token_hash("xr08-prefix-raw-token-ids-v1", prefix_tokens),
        adapter_id: None,
        adapter_weight_hash: None,
        kv_layout_version: KV_LAYOUT_VERSION,
        cache_mode: mode.cache_mode(),
        mlx_version: version.backend_version,
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

fn namespace_rejected(
    cache: &mut SsdPrefixCache,
    key: &KvBlockKey,
    namespace: KvNamespace,
) -> bool {
    matches!(
        cache.restore_before_prefill(key, &namespace),
        Err(KvError::NamespaceMismatch { .. })
    )
}

fn wrong_model(namespace: &KvNamespace) -> KvNamespace {
    let mut wrong = namespace.clone();
    wrong.model_id = "wrong-model".to_owned();
    wrong
}

fn wrong_adapter(namespace: &KvNamespace) -> KvNamespace {
    let mut wrong = namespace.clone();
    wrong.adapter_id = Some("wrong-adapter".to_owned());
    wrong.adapter_weight_hash = Some("wrong-adapter-weight-hash".to_owned());
    wrong
}

fn other_cache_mode(mode: BenchMode) -> CacheMode {
    match mode {
        BenchMode::Bf16 => CacheMode::MlxAffineQ8,
        BenchMode::Q8 => CacheMode::Bf16,
    }
}

fn policy_for_mode(mode: BenchMode) -> KvPolicy {
    KvPolicy {
        ssd_prefix_mode: mode.ffi_mode(),
        compress_global_layers: mode.compression_enabled(),
        compress_sliding_layers: false,
        allow_active_compressed_decode: false,
        ..Default::default()
    }
}

fn config_for_record(options: &Options, mode: BenchMode) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert("storage_format".to_owned(), mode.label().to_owned());
    out.insert(
        "compression_enabled".to_owned(),
        mode.compression_enabled().to_string(),
    );
    out.insert(
        "min_prefix_tokens".to_owned(),
        options.min_prefix_tokens.to_string(),
    );
    out.insert(
        "max_cache_size_bytes".to_owned(),
        options.max_cache_size_bytes.to_string(),
    );
    out.insert(
        "mid_decode_ssd_fetch_allowed".to_owned(),
        "false".to_owned(),
    );
    out
}

fn target_config(options: &Options) -> LoadConfig {
    LoadConfig {
        model_path: options.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: env::var("GEMMA4D_MODEL_REVISION").ok(),
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(options.max_context_tokens as u32)
            .expect("max context is non-zero"),
        allow_unsupported_config: false,
    }
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR08 SSD Cache Policy and Variance A/B Report\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Run\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!("| Model | `{}` |\n", summary.model_identity.path));
    out.push_str(&format!("| Modes | `{}` |\n", summary.modes.join(", ")));
    out.push_str(&format!("| Trials | `{}` |\n", summary.requested_trials));
    out.push_str(&format!(
        "| Policy | `{}` |\n",
        escape_md(&summary.policy_decision.profile_policy)
    ));
    out.push_str(&format!(
        "| Rationale | `{}` |\n",
        escape_md(&summary.policy_decision.rationale)
    ));
    out.push('\n');

    out.push_str("## Workload Cases\n\n");
    out.push_str("| Case | Context | Workload | Seed | Prefix Hash |\n");
    out.push_str("|---|---:|---|---:|---|\n");
    for case in &summary.selected_cases {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | {} | `{}` |\n",
            case.case_id,
            case.context_tokens,
            case.workload_id,
            case.source_deterministic_seed,
            case.prefix_token_hash
        ));
    }
    out.push('\n');

    out.push_str("## Aggregates\n\n");
    out.push_str("| Case | Variant | Trials | Fresh p50 | Fresh p95 | Warm p50 | Warm p95 | p50 Imp % | p95 Imp % | Warm CV | Payload MiB | Metadata bytes | Peak GB | Correct | Rejects | Variance | Memory |\n");
    out.push_str(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n",
    );
    for aggregate in &summary.aggregates {
        out.push_str(&format!(
            "| `{}` | `{}` | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` | `{}` | `{}` | `{}` |\n",
            aggregate.case_id,
            aggregate.variant,
            aggregate.passed_trials,
            aggregate.trial_count,
            fmt_opt(aggregate.fresh_prefill_p50_ms),
            fmt_opt(aggregate.fresh_prefill_p95_ms),
            fmt_opt(aggregate.warm_ttft_p50_ms),
            fmt_opt(aggregate.warm_ttft_p95_ms),
            fmt_opt(aggregate.warm_p50_improvement_percent),
            fmt_opt(aggregate.warm_p95_improvement_percent),
            fmt_opt(aggregate.warm_ttft_cv),
            fmt_bytes_mib_f64(aggregate.payload_bytes_written_median),
            fmt_opt(aggregate.metadata_bytes_written_median),
            fmt_opt(aggregate.peak_mlx_max_gb),
            aggregate.correctness_passed,
            aggregate.rejection_passed,
            aggregate.variance_gate_passed,
            aggregate.memory_gate_passed
        ));
    }
    out.push('\n');

    out.push_str("## Admission Probes\n\n");
    out.push_str("| Probe | Workload | Context | Variant | Admitted | Reason |\n");
    out.push_str("|---|---|---:|---|---|---|\n");
    for probe in &summary.admission_probes {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | `{}` | `{}` | {} |\n",
            probe.probe,
            probe.workload_id.as_deref().unwrap_or("n/a"),
            probe
                .context_tokens
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_owned()),
            probe.variant.as_deref().unwrap_or("n/a"),
            probe.admitted,
            escape_md(&probe.reason)
        ));
    }
    out.push('\n');

    out.push_str("## Verification Command\n\n");
    out.push_str("```sh\n");
    out.push_str(&summary.command);
    out.push('\n');
    out.push_str("```\n\n");

    out.push_str("## Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {note}.\n"));
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR08 SSD Cache Policy and Variance A/B Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No hard blockers recorded.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out.push_str("\n## Failed Hypotheses And Caveats\n\n");
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
    out.push_str("# XR08 SSD Cache Policy and Variance A/B Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Evidence\n\n");
    for path in &summary.generated_files {
        out.push_str(&format!("- `{path}`\n"));
    }
    out.push_str("\n## Profile Policy\n\n");
    out.push_str(&format!(
        "- Policy: `{}`\n",
        summary.policy_decision.profile_policy
    ));
    out.push_str(&format!(
        "- Accepted profiles: `{}`\n",
        if summary.policy_decision.accepted_variants.is_empty() {
            "none".to_owned()
        } else {
            summary.policy_decision.accepted_variants.join(", ")
        }
    ));
    out.push_str(&format!(
        "- Rejected profiles: `{}`\n",
        if summary.policy_decision.rejected_variants.is_empty() {
            "none".to_owned()
        } else {
            summary.policy_decision.rejected_variants.join(", ")
        }
    ));
    out.push_str(&format!(
        "- Rationale: {}\n",
        summary.policy_decision.rationale
    ));
    out.push_str("\n## Claim Boundary\n\n");
    out.push_str("- XR08 measures SSD prefix-cache policy and variance on exact real-context prefix restores only.\n");
    out.push_str("- Warm TTFT includes metadata restore, payload checksum, payload load, native import, and last-step retrieval.\n");
    out.push_str("- q8 payloads are compressed on disk but restored into BF16 active decode; compressed active decode remains disabled.\n");
    out.push_str(
        "- Mid-decode SSD fetch remains disallowed and is tested only as a rejection path.\n",
    );
    out.push_str("- Any accepted policy is profile-gated and opt-in; this is not production serving readiness.\n");
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

fn startup_blockers(options: &Options) -> Vec<String> {
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
    if !options.python.exists() {
        blockers.push(format!(
            "python path does not exist: {}",
            options.python.display()
        ));
    }
    if env::var("GEMMA4D_REQUIRE_MLX").ok().as_deref() != Some("1") {
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for XR08 native MLX evidence".to_owned());
    }
    if env::var("GEMMA4D_USE_NATIVE_GRAPH").ok().as_deref() != Some("1") {
        blockers.push(
            "GEMMA4D_USE_NATIVE_GRAPH=1 is required for XR08 native graph evidence".to_owned(),
        );
    }
    blockers
}

fn write_jsonl(path: &Path, records: &[Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        writeln!(file)?;
    }
    Ok(())
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

fn quantization_hash(model_identity: &manifest::ArtifactIdentity) -> String {
    sha256_hex(
        format!(
            "config={}\nsafetensors={}\n",
            model_identity.config_sha256, model_identity.safetensors_inventory_sha256
        )
        .as_bytes(),
    )
}

fn token_hash(domain: &str, tokens: &[i32]) -> String {
    let mut bytes = Vec::with_capacity(domain.len() + 1 + tokens.len() * 4);
    bytes.extend_from_slice(domain.as_bytes());
    bytes.push(0);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn native_handle_id(context_tokens: usize, trial_index: usize, mode: BenchMode) -> u64 {
    let mode_id = match mode {
        BenchMode::Bf16 => 0_u64,
        BenchMode::Q8 => 1_u64,
    };
    ((context_tokens as u64) << 32) | ((trial_index as u64) << 8) | mode_id
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
        .filter(|value| value.is_finite())
        .reduce(f64::min)
}

fn max_value(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

fn mean_value(values: &[f64]) -> Option<f64> {
    let values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
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

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, CliError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| parse_positive_usize(part, "--contexts"))
        .collect()
}

fn parse_modes(value: &str) -> Result<Vec<BenchMode>, CliError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| match part {
            "bf16" => Ok(BenchMode::Bf16),
            "q8" | "mlx_affine_q8" => Ok(BenchMode::Q8),
            other => Err(CliError::Usage(format!(
                "unsupported --modes value '{other}', expected bf16,q8"
            ))),
        })
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

fn parse_positive_u64(value: &str, option: &str) -> Result<u64, CliError> {
    let parsed = value.parse::<u64>().map_err(|error| {
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
    format!("xr08-{}-{}", now.as_secs(), now.subsec_nanos())
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
    env::args().collect::<Vec<_>>().join(" ")
}

fn capture_environment(python: &Path) -> Environment {
    let version = runtime_version().ok();
    Environment {
        machine: command_stdout("uname", &["-a"]).unwrap_or_else(|| "unknown".to_owned()),
        macos: command_stdout("sw_vers", &["-productVersion"])
            .unwrap_or_else(|| "unknown".to_owned()),
        rustc: command_stdout("rustc", &["-Vv"]).unwrap_or_else(|| "unknown".to_owned()),
        cargo: command_stdout("cargo", &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
        runtime_backend: version
            .as_ref()
            .map(|value| value.backend_name.clone())
            .unwrap_or_else(|| "unknown".to_owned()),
        runtime_backend_version: version
            .as_ref()
            .map(|value| value.backend_version.clone())
            .unwrap_or_else(|| "unknown".to_owned()),
        tokenizer_mlx_version: command_stdout(
            &python.display().to_string(),
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

fn relevant_environment() -> BTreeMap<String, Option<String>> {
    ENV_KEYS
        .iter()
        .chain(
            [
                "GEMMA4D_MODEL_REVISION",
                "GEMMA4D_FULL_MODEL_TESTS",
                "RUSTFLAGS",
            ]
            .iter(),
        )
        .map(|key| ((*key).to_owned(), env::var(key).ok()))
        .collect()
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn fmt_bytes_mib_f64(value: Option<f64>) -> String {
    value
        .map(|bytes| format!("{:.3}", bytes / (1024.0 * 1024.0)))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn usage() -> String {
    format!(
        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr08_ssd_cache_policy_variance -- [--out-dir PATH] [--cache-dir PATH] [--workloads PATH] [--model-path PATH] [--python PATH] [--trials N] [--clear-contexts] [--context N] [--contexts CSV] [--modes bf16,q8] [--min-prefix-tokens N] [--max-cache-size-bytes N]\n\ndefault out-dir: {DEFAULT_OUT_DIR}\ndefault workloads: {DEFAULT_WORKLOADS}"
    )
}
