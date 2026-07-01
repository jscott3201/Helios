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
use gemma4d_ffi::{KvCache, KvPolicy, LoadConfig, PrefillChunkPolicy, Target, prefill};
use gemma4d_tokenizer::sha256_hex;
use serde::Serialize;

const GOAL: &str = "XR05-prefill-and-eval-scheduling-ab";
const MODE: &str = "prefill_eval_scheduling_real_context_ab";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR05-prefill-and-eval-scheduling-ab";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const DEFAULT_TRIALS: usize = 3;
const LOGIT_TOLERANCE: f64 = 0.5;
const PREFILL_IMPROVEMENT_GATE_PERCENT: f64 = 10.0;
const MEMORY_IMPROVEMENT_GATE_PERCENT: f64 = 5.0;
const MAX_P95_REGRESSION_PERCENT: f64 = 5.0;
const DEFAULT_WORKLOAD_IDS: &[&str] = &[
    "code_review_rust_4k_001",
    "code_review_rust_8k_001",
    "benchmark_qa_16k_001",
];
const ENV_KEYS: &[&str] = &[
    "GEMMA4D_REQUIRE_MLX",
    "GEMMA4D_USE_NATIVE_GRAPH",
    "GEMMA4D_MLX_LM_PREFILL_CHUNK_TOKENS",
    "GEMMA4D_MLX_LM_PREFILL_CLEAR_CACHE",
    "GEMMA4D_NATIVE_PREFILL_KV_EVAL",
    "GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS",
    "GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse(env::args().skip(1))?;
    fs::create_dir_all(&options.out_dir)?;

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
    let environment = capture_environment();
    let model_identity =
        manifest::capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION");
    let variants = selected_variants(&options)?;
    let mut blockers = startup_blockers(&options, &variants);
    let workloads = select_workloads(load_workloads(&options.workloads_path)?, &options)?;
    let selected_workload_details = workloads
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
        requested_trials: options.trials,
        logit_tolerance: LOGIT_TOLERANCE,
        prefill_improvement_gate_percent: PREFILL_IMPROVEMENT_GATE_PERCENT,
        memory_improvement_gate_percent: MEMORY_IMPROVEMENT_GATE_PERCENT,
        max_p95_regression_percent: MAX_P95_REGRESSION_PERCENT,
        variants,
        selected_workloads: selected_workload_details,
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
        comparisons,
        failed_hypotheses,
        blockers,
        generated_files: vec![
            records_path.display().to_string(),
            summary_path.display().to_string(),
            report_path.display().to_string(),
            blockers_path.display().to_string(),
            decision_path.display().to_string(),
        ],
        measurement_notes: vec![
            "helper variants use GEMMA4D_MLX_LM_PREFILL_CHUNK_TOKENS and GEMMA4D_MLX_LM_PREFILL_CLEAR_CACHE before target load".to_owned(),
            "native variants use GEMMA4D_REQUIRE_MLX=1, GEMMA4D_USE_NATIVE_GRAPH=1, and GEMMA4D_NATIVE_PREFILL_KV_EVAL before target load".to_owned(),
            "each variant/trial loads the target once and runs all selected workload prefills with fresh KV caches".to_owned(),
            "correctness compares output greedy token and greedy logit against the same-backend baseline variant for the same workload/trial".to_owned(),
            "low trial counts are retained in raw records and aggregates; acceptance still requires correctness plus the XR05 p50 or memory gate".to_owned(),
        ],
    };

    write_jsonl(&records_path, &summary.records_for_jsonl())?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR05 prefill/eval scheduling A/B: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR05 benchmark blocked; see blockers.md".into())
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
        if options.trials == 0 {
            return Err(CliError::Usage(
                "--trials must be greater than zero".to_owned(),
            ));
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
    requested_trials: usize,
    logit_tolerance: f64,
    prefill_improvement_gate_percent: f64,
    memory_improvement_gate_percent: f64,
    max_p95_regression_percent: f64,
    variants: Vec<Variant>,
    selected_workloads: Vec<SelectedWorkload>,
    record_count: usize,
    passed_records: usize,
    failed_records: usize,
    aggregates: Vec<Aggregate>,
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

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Backend {
    Helper,
    Native,
}

impl Backend {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Helper => "helper",
            Self::Native => "native",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Variant {
    name: String,
    backend: Backend,
    config: BTreeMap<String, String>,
    env: BTreeMap<String, String>,
    #[serde(skip_serializing)]
    prefill_chunk_policy: Option<PrefillChunkPolicy>,
    baseline_variant: Option<String>,
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
    model_load_ms: f64,
    prefill_ms: f64,
    ttft_ms: f64,
    prefill_tps: f64,
    output_token_id: Option<i32>,
    output_logit: Option<f64>,
    peak_mlx_gb: f64,
    rss_mb: f64,
    active_kv_bytes: u64,
    status: String,
    blocker: Option<String>,
    correctness: Correctness,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Correctness {
    status: String,
    reference_variant: Option<String>,
    token_match: Option<bool>,
    logit_match: Option<bool>,
    logit_abs_delta: Option<f64>,
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
            logit_abs_delta: None,
            logit_tolerance: Some(LOGIT_TOLERANCE),
            notes: Vec::new(),
        }
    }
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
    prefill_p50_ms: Option<f64>,
    prefill_p95_ms: Option<f64>,
    prefill_min_ms: Option<f64>,
    prefill_max_ms: Option<f64>,
    prefill_cv: Option<f64>,
    prefill_tps_p50: Option<f64>,
    peak_mlx_max_gb: Option<f64>,
    rss_max_mb: Option<f64>,
    active_kv_max_bytes: Option<u64>,
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
    correctness_passed: bool,
    candidate_trials: usize,
    baseline_trials: usize,
    prefill_p50_delta_percent: Option<f64>,
    prefill_p95_regression_percent: Option<f64>,
    peak_mlx_delta_percent: Option<f64>,
    accepted: bool,
    reason: String,
}

fn selected_variants(options: &Options) -> Result<Vec<Variant>, CliError> {
    let mut variants = vec![
        helper_variant(512),
        helper_variant(1024),
        helper_variant(2048),
        helper_variant_with_clear_cache(2048, false),
        helper_variant(4096),
        native_variant("native_eval_per_layer", "per_layer"),
        native_variant("native_eval_end_of_prefill", "end_of_prefill"),
        native_variant(
            "native_eval_selective_full_attention",
            "selective_full_attention",
        ),
        native_variant_with_extra_env(
            "native_chunked_prefill_256",
            "per_layer",
            [("GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS", "256")],
        ),
        native_variant_with_extra_env(
            "native_chunked_prefill_policy_long_context_256",
            "per_layer",
            [("GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY", "long_context_256")],
        ),
        native_variant_with_setter_policy(
            "native_chunked_prefill_setter_long_context_256",
            "per_layer",
            PrefillChunkPolicy::LongContext256,
            "long_context_256",
        ),
        native_variant_with_extra_env(
            "native_chunked_prefill_384",
            "per_layer",
            [("GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS", "384")],
        ),
        native_variant_with_extra_env(
            "native_chunked_prefill_512",
            "per_layer",
            [("GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS", "512")],
        ),
        native_variant_with_extra_env(
            "native_chunked_prefill_768",
            "per_layer",
            [("GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS", "768")],
        ),
        native_variant_with_extra_env(
            "native_chunked_prefill_1024",
            "per_layer",
            [("GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS", "1024")],
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

fn helper_variant(chunk_tokens: usize) -> Variant {
    helper_variant_with_clear_cache(chunk_tokens, true)
}

fn helper_variant_with_clear_cache(chunk_tokens: usize, prefill_clear_cache: bool) -> Variant {
    let mut config = BTreeMap::new();
    config.insert("prefill_chunk_tokens".to_owned(), chunk_tokens.to_string());
    config.insert(
        "prefill_clear_cache".to_owned(),
        prefill_clear_cache.to_string(),
    );
    let mut env = BTreeMap::new();
    env.insert(
        "GEMMA4D_MLX_LM_PREFILL_CHUNK_TOKENS".to_owned(),
        chunk_tokens.to_string(),
    );
    env.insert(
        "GEMMA4D_MLX_LM_PREFILL_CLEAR_CACHE".to_owned(),
        if prefill_clear_cache { "1" } else { "0" }.to_owned(),
    );
    Variant {
        name: if prefill_clear_cache {
            format!("helper_chunk_{chunk_tokens}")
        } else {
            format!("helper_chunk_{chunk_tokens}_no_clear_cache")
        },
        backend: Backend::Helper,
        config,
        env,
        prefill_chunk_policy: None,
        baseline_variant: if chunk_tokens == 2048 && prefill_clear_cache {
            None
        } else {
            Some("helper_chunk_2048".to_owned())
        },
    }
}

fn native_variant(name: &str, eval_mode: &str) -> Variant {
    native_variant_with_extra_env(name, eval_mode, [])
}

fn native_variant_with_extra_env<const N: usize>(
    name: &str,
    eval_mode: &str,
    extra_env: [(&str, &str); N],
) -> Variant {
    let mut config = BTreeMap::new();
    config.insert("prefill_kv_eval".to_owned(), eval_mode.to_owned());
    let mut env = BTreeMap::new();
    env.insert("GEMMA4D_REQUIRE_MLX".to_owned(), "1".to_owned());
    env.insert("GEMMA4D_USE_NATIVE_GRAPH".to_owned(), "1".to_owned());
    env.insert(
        "GEMMA4D_NATIVE_PREFILL_KV_EVAL".to_owned(),
        eval_mode.to_owned(),
    );
    for (key, value) in extra_env {
        config.insert(key.to_owned(), value.to_owned());
        env.insert(key.to_owned(), value.to_owned());
    }
    Variant {
        name: name.to_owned(),
        backend: Backend::Native,
        config,
        env,
        prefill_chunk_policy: None,
        baseline_variant: if name == "native_eval_per_layer" {
            None
        } else {
            Some("native_eval_per_layer".to_owned())
        },
    }
}

fn native_variant_with_setter_policy(
    name: &str,
    eval_mode: &str,
    policy: PrefillChunkPolicy,
    policy_name: &str,
) -> Variant {
    let mut variant = native_variant_with_extra_env(name, eval_mode, []);
    variant.config.insert(
        "prefill_chunk_policy_setter".to_owned(),
        policy_name.to_owned(),
    );
    variant.prefill_chunk_policy = Some(policy);
    variant
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

    let mut target = match target {
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
    if let Some(policy) = variant.prefill_chunk_policy {
        if let Err(error) = target.set_prefill_chunk_policy(policy) {
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
                    format!("prefill chunk policy setter failed: {error}"),
                ));
            }
            return Ok(());
        }
    }

    for workload in workload_inputs {
        records.push(run_prefill_record(
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
fn run_prefill_record(
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
    let mut cache = KvCache::create(&KvPolicy::default())?;
    let started = Instant::now();
    let step = prefill(target, &mut cache, &workload.token_ids);
    let prefill_duration = started.elapsed();
    let (status, blocker, output_token_id, output_logit, peak_mlx_gb, rss_mb, active_kv_bytes) =
        match step {
            Ok(step) => (
                "passed".to_owned(),
                None,
                Some(step.greedy_token),
                Some(f64::from(step.greedy_logit)),
                f64::from(step.peak_memory_gb),
                f64::from(step.peak_rss_mb),
                step.active_kv_bytes,
            ),
            Err(error) => (
                "failed".to_owned(),
                Some(format!("prefill failed: {error}")),
                None,
                None,
                0.0,
                0.0,
                0,
            ),
        };
    let prefill_ms = duration_ms(prefill_duration);
    let input_tokens = workload.token_ids.len();
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
        backend: variant.backend.as_str().to_owned(),
        config: variant.config.clone(),
        env: variant.env.clone(),
        trial_index,
        input_tokens,
        model_load_ms: duration_ms(model_load),
        prefill_ms,
        ttft_ms: prefill_ms,
        prefill_tps: if prefill_ms > 0.0 {
            input_tokens as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        },
        output_token_id,
        output_logit,
        peak_mlx_gb,
        rss_mb,
        active_kv_bytes,
        status,
        blocker,
        correctness: Correctness::pending(),
        notes: Vec::new(),
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
        backend: variant.backend.as_str().to_owned(),
        config: variant.config.clone(),
        env: variant.env.clone(),
        trial_index,
        input_tokens: workload.token_ids.len(),
        model_load_ms: duration_ms(model_load),
        prefill_ms: 0.0,
        ttft_ms: 0.0,
        prefill_tps: 0.0,
        output_token_id: None,
        output_logit: None,
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
            logit_abs_delta: None,
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
        .filter(|record| is_baseline_variant(&record.variant))
        .map(|record| {
            (
                (
                    record.backend.clone(),
                    record.workload_id.clone(),
                    record.trial_index,
                ),
                (
                    record.variant.clone(),
                    record.output_token_id,
                    record.output_logit,
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();

    for record in records {
        if record.status != "passed" {
            continue;
        }
        let key = (
            record.backend.clone(),
            record.workload_id.clone(),
            record.trial_index,
        );
        let Some((reference_variant, reference_token, reference_logit)) = references.get(&key)
        else {
            record.correctness = Correctness {
                status: "missing_reference".to_owned(),
                reference_variant: baseline_for_backend(&record.backend),
                token_match: None,
                logit_match: None,
                logit_abs_delta: None,
                logit_tolerance: Some(LOGIT_TOLERANCE),
                notes: vec!["same-backend baseline record was unavailable".to_owned()],
            };
            continue;
        };
        let token_match = record.output_token_id == *reference_token;
        let logit_abs_delta = match (record.output_logit, *reference_logit) {
            (Some(left), Some(right)) => Some((left - right).abs()),
            _ => None,
        };
        let logit_match = logit_abs_delta
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
            logit_abs_delta,
            logit_tolerance: Some(LOGIT_TOLERANCE),
            notes: Vec::new(),
        };
    }
}

fn is_baseline_variant(variant: &str) -> bool {
    variant == "helper_chunk_2048" || variant == "native_eval_per_layer"
}

fn baseline_for_backend(backend: &str) -> Option<String> {
    match backend {
        "helper" => Some("helper_chunk_2048".to_owned()),
        "native" => Some("native_eval_per_layer".to_owned()),
        _ => None,
    }
}

fn build_aggregates(records: &[Record]) -> Vec<Aggregate> {
    let mut keys = records
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
    for (variant, backend, workload_id, family) in std::mem::take(&mut keys) {
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
        let prefill = passed
            .iter()
            .map(|record| record.prefill_ms)
            .collect::<Vec<_>>();
        let tps = passed
            .iter()
            .map(|record| record.prefill_tps)
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
            prefill_p50_ms: percentile(prefill.clone(), 0.50),
            prefill_p95_ms: percentile(prefill.clone(), 0.95),
            prefill_min_ms: min_value(&prefill),
            prefill_max_ms: max_value(&prefill),
            prefill_cv: coefficient_of_variation(&prefill),
            prefill_tps_p50: percentile(tps, 0.50),
            peak_mlx_max_gb: max_value(&peak),
            rss_max_mb: max_value(&rss),
            active_kv_max_bytes: active_kv.into_iter().max(),
            low_n: passed.len() < 3,
            records: group,
        });
    }
    out
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
                    correctness_passed: false,
                    candidate_trials: candidate.passed_trials,
                    baseline_trials: 0,
                    prefill_p50_delta_percent: None,
                    prefill_p95_regression_percent: None,
                    peak_mlx_delta_percent: None,
                    accepted: false,
                    reason: "missing baseline aggregate".to_owned(),
                });
                continue;
            };

            let p50_delta = percent_improvement(baseline.prefill_p50_ms, candidate.prefill_p50_ms);
            let p95_regression =
                percent_regression(baseline.prefill_p95_ms, candidate.prefill_p95_ms);
            let peak_delta =
                percent_improvement(baseline.peak_mlx_max_gb, candidate.peak_mlx_max_gb);
            let correctness_passed = candidate.passed_trials > 0
                && baseline.passed_trials > 0
                && candidate.correctness_passed_trials == candidate.trial_count;
            let candidate_variant_clean = *variant_correctness_clean
                .get(&candidate.variant)
                .unwrap_or(&false);
            let enough_trials = candidate.passed_trials >= 3 && baseline.passed_trials >= 3;
            let prefill_gate = p50_delta
                .map(|delta| delta >= PREFILL_IMPROVEMENT_GATE_PERCENT)
                .unwrap_or(false);
            let memory_gate = peak_delta
                .map(|delta| delta >= MEMORY_IMPROVEMENT_GATE_PERCENT)
                .unwrap_or(false);
            let p95_ok = p95_regression
                .map(|regression| regression <= MAX_P95_REGRESSION_PERCENT)
                .unwrap_or(false);
            let accepted = correctness_passed
                && candidate_variant_clean
                && enough_trials
                && p95_ok
                && (prefill_gate || memory_gate);
            let reason = if !correctness_passed {
                "correctness gate failed or records missing".to_owned()
            } else if !candidate_variant_clean {
                "candidate variant had a correctness regression on another selected workload"
                    .to_owned()
            } else if !enough_trials {
                "fewer than three passed trials for candidate or baseline".to_owned()
            } else if !p95_ok {
                "prefill p95 regression exceeded gate".to_owned()
            } else if accepted {
                "accepted by XR05 prefill or memory gate".to_owned()
            } else {
                "no XR05 prefill or memory gate met".to_owned()
            };
            out.push(Comparison {
                candidate_variant: candidate.variant.clone(),
                baseline_variant: baseline.variant.clone(),
                backend: candidate.backend.clone(),
                workload_id: candidate.workload_id.clone(),
                family: candidate.family.clone(),
                correctness_passed,
                candidate_trials: candidate.passed_trials,
                baseline_trials: baseline.passed_trials,
                prefill_p50_delta_percent: p50_delta,
                prefill_p95_regression_percent: p95_regression,
                peak_mlx_delta_percent: peak_delta,
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
    } else if records
        .iter()
        .any(|record| record.correctness.status != "passed")
    {
        "reject_candidate".to_owned()
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
            // SAFETY: the XR05 benchmark is single-threaded and mutates process
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

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR05 Prefill And Eval Scheduling A/B Report\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Summary\n\n");
    out.push_str(&format!(
        "- Records: `{}` passed `{}` failed `{}`\n",
        summary.record_count, summary.passed_records, summary.failed_records
    ));
    out.push_str(&format!(
        "- Trials requested: `{}`; logit tolerance: `{:.3}`\n",
        summary.requested_trials, summary.logit_tolerance
    ));
    out.push_str(&format!(
        "- Prefill gate: p50 improvement `>= {:.1}%` or peak memory improvement `>= {:.1}%`, with p95 regression `<= {:.1}%`\n\n",
        summary.prefill_improvement_gate_percent,
        summary.memory_improvement_gate_percent,
        summary.max_p95_regression_percent
    ));

    out.push_str("## Variants\n\n");
    out.push_str("| Variant | Backend | Baseline | Config |\n");
    out.push_str("|---|---|---|---|\n");
    for variant in &summary.variants {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` |\n",
            variant.name,
            variant.backend.as_str(),
            variant.baseline_variant.as_deref().unwrap_or("self"),
            serde_json::to_string(&variant.config).unwrap_or_else(|_| "{}".to_owned())
        ));
    }

    out.push_str("\n## Aggregates\n\n");
    out.push_str("| Workload | Variant | Trials | Correct | Prefill p50 ms | Prefill p95 ms | Prefill tps p50 | Peak MLX GB | RSS MB | Low N |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for aggregate in &summary.aggregates {
        out.push_str(&format!(
            "| `{}` | `{}` | {}/{} | {} | {} | {} | {} | {} | {} | `{}` |\n",
            aggregate.workload_id,
            aggregate.variant,
            aggregate.passed_trials,
            aggregate.trial_count,
            aggregate.correctness_passed_trials,
            fmt_opt(aggregate.prefill_p50_ms),
            fmt_opt(aggregate.prefill_p95_ms),
            fmt_opt(aggregate.prefill_tps_p50),
            fmt_opt(aggregate.peak_mlx_max_gb),
            fmt_opt(aggregate.rss_max_mb),
            aggregate.low_n
        ));
    }

    out.push_str("\n## Comparisons\n\n");
    out.push_str("| Workload | Candidate | Baseline | Correct | p50 improvement % | p95 regression % | Peak MLX improvement % | Accepted | Reason |\n");
    out.push_str("|---|---|---|---|---:|---:|---:|---|---|\n");
    for comparison in &summary.comparisons {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | {} | {} | {} | `{}` | {} |\n",
            comparison.workload_id,
            comparison.candidate_variant,
            comparison.baseline_variant,
            comparison.correctness_passed,
            fmt_opt(comparison.prefill_p50_delta_percent),
            fmt_opt(comparison.prefill_p95_regression_percent),
            fmt_opt(comparison.peak_mlx_delta_percent),
            comparison.accepted,
            comparison.reason
        ));
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR05 Prefill And Eval Scheduling A/B Blockers\n\n");
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
    out.push_str("# XR05 Prefill And Eval Scheduling A/B Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Evidence\n\n");
    for file in &summary.generated_files {
        out.push_str(&format!("- `{file}`\n"));
    }
    out.push_str("\n## Accepted Comparisons\n\n");
    let accepted = summary
        .comparisons
        .iter()
        .filter(|comparison| comparison.accepted)
        .collect::<Vec<_>>();
    if accepted.is_empty() {
        out.push_str("No comparisons met the XR05 acceptance gate.\n");
    } else {
        for comparison in accepted {
            out.push_str(&format!(
                "- `{}` vs `{}` on `{}`: p50 improvement {}, peak MLX improvement {}, p95 regression {}\n",
                comparison.candidate_variant,
                comparison.baseline_variant,
                comparison.workload_id,
                fmt_opt(comparison.prefill_p50_delta_percent),
                fmt_opt(comparison.peak_mlx_delta_percent),
                fmt_opt(comparison.prefill_p95_regression_percent)
            ));
        }
    }
    out.push_str("\n## Claim Boundary\n\n");
    out.push_str("- XR05 is prefill/eval scheduling evidence only; it does not change defaults unless a later goal adopts a policy.\n");
    out.push_str("- Helper and native candidates are compared against same-backend baselines, not against each other.\n");
    out.push_str("- Low-N aggregates are marked in `summary.json` and `report.md`.\n");
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
            "no workloads selected for XR05 benchmark".to_owned(),
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

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
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
    format!("xr05-{}-{}", now.as_secs(), now.subsec_nanos())
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
        "usage: GEMMA4D_REQUIRE_MLX=1 cargo run -p gemma4d-bench --example xr05_prefill_eval_scheduling_ab -- [--out-dir PATH] [--workloads PATH] [--model-path PATH] [--python PATH] [--trials N] [--workload-id ID] [--clear-workload-ids] [--max-workloads N] [--variant NAME|--variants CSV]\n\ndefault out-dir: {DEFAULT_OUT_DIR}\ndefault workloads: {DEFAULT_WORKLOADS}"
    )
}
