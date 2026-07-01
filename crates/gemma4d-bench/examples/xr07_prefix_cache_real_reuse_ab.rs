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
use gemma4d_ffi::{KvCache, KvPolicy, LoadConfig, Target, decode_one, prefill, runtime_version};
use gemma4d_kv::{
    CacheAccountingSnapshot, CacheMode, Error as KvError, KV_LAYOUT_VERSION, KvBlockKey,
    KvNamespace, PrefillObservation, RamPrefixBlock, RamPrefixCache,
};
use gemma4d_tokenizer::sha256_hex;
use serde::Serialize;

const GOAL: &str = "XR07-prefix-cache-real-reuse-ab";
const MODE: &str = "native_ram_prefix_cache_real_reuse_ab";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR07-prefix-cache-real-reuse-ab";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const DEFAULT_TRIALS: usize = 2;
const DEFAULT_SUFFIX_TOKENS: usize = 16;
const DEFAULT_SUFFIX_EDIT_TOKENS: usize = 4;
const DEFAULT_CONTINUED_DECODE_TOKENS: usize = 4;
const DEFAULT_RAM_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 32_768;
const LOGIT_TOLERANCE: f64 = 0.000_001;
const WARM_SPEEDUP_GATE: f64 = 1.25;
const WARM_IMPROVEMENT_GATE_MS: f64 = 100.0;
const MEMORY_CLIFF_GB: f64 = 14.0;
const TINY16_CAP_MEMORY_FRACTION: f64 = 0.15;
const ENV_KEYS: &[&str] = &["GEMMA4D_REQUIRE_MLX", "GEMMA4D_USE_NATIVE_GRAPH"];

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
    let environment = capture_environment(&options.python);
    let model_identity =
        manifest::capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION");
    let workloads = load_workloads(&options.workloads_path)?;
    let mut blockers = startup_blockers(&options);
    let mut tokenizer_backend = "not_started".to_owned();
    let mut selected_cases = Vec::new();
    let mut records = Vec::new();

    if blockers.is_empty() {
        let mut tokenizer = TokenizerHelper::start(&options.python, &options.model_path)?;
        tokenizer_backend = tokenizer.backend().to_owned();
        selected_cases = prepare_cases(&options, &workloads, &mut tokenizer)?;
        tokenizer.shutdown();

        let load_started = Instant::now();
        let target = Target::load(&target_config(&options));
        let model_load_ms = duration_ms(load_started.elapsed());
        match target {
            Ok(target) => {
                for case in &selected_cases {
                    for trial_index in 0..options.trials {
                        eprintln!(
                            "XR07 running context={} trial={} workload={} suffix={}",
                            case.context_tokens(),
                            trial_index,
                            case.workload_id(),
                            case.suffix_tokens()
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
                            Ok(record) => records.push(record),
                            Err(error) => records.push(failed_record(
                                &run_id,
                                &git_sha,
                                &git_status_short,
                                &command,
                                case,
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
                    for trial_index in 0..options.trials {
                        records.push(failed_record(
                            &run_id,
                            &git_sha,
                            &git_status_short,
                            &command,
                            case,
                            trial_index,
                            model_load_ms,
                            format!("target load failed: {error}"),
                        ));
                    }
                }
            }
        }
    }

    let aggregates = build_aggregates(&records);
    let failed_hypotheses = failed_hypotheses(&records, &aggregates);
    blockers.extend(blockers_for_records(&records, &selected_cases, &options));
    blockers.sort();
    blockers.dedup();
    let mut cap_recommendation = cap_recommendation(
        &records,
        environment.hw_memsize_bytes,
        TINY16_CAP_MEMORY_FRACTION,
    );
    let decision = decision_for(&blockers, &records, &aggregates, &cap_recommendation);
    apply_decision_to_cap_recommendation(&decision, &mut cap_recommendation);
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
    let selected_case_records = selected_cases
        .iter()
        .map(|case| case.selected.clone())
        .collect::<Vec<_>>();
    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        generated_at_unix_seconds: unix_now(),
        status: status.to_owned(),
        decision: decision.clone(),
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
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        decision_path: decision_path.display().to_string(),
        requested_trials: options.trials,
        contexts: options.contexts.clone(),
        suffix_tokens: options.suffix_tokens,
        suffix_edit_tokens: options.suffix_edit_tokens,
        continued_decode_tokens: options.continued_decode_tokens,
        ram_budget_bytes: options.ram_budget_bytes,
        logit_tolerance: LOGIT_TOLERANCE,
        warm_speedup_gate: WARM_SPEEDUP_GATE,
        warm_improvement_gate_ms: WARM_IMPROVEMENT_GATE_MS,
        memory_cliff_gb: MEMORY_CLIFF_GB,
        tiny16_cap_memory_fraction: TINY16_CAP_MEMORY_FRACTION,
        selected_cases: selected_case_records,
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
        cap_recommendation,
        failed_hypotheses,
        blockers,
        generated_files,
        records,
        measurement_notes: vec![
            "baseline fresh_full_prefill_ms is a native prefill of the full edited prompt".to_owned(),
            "candidate warm_ttft_ms includes RAM namespace lookup, native snapshot import, and replay of the edited suffix tokens".to_owned(),
            "prefix prefill and snapshot export are reported separately because they are cache population costs, not warm lookup costs".to_owned(),
            "continued decode parity compares deterministic greedy tokens and logits after the full edited prompt state has been reached by both paths".to_owned(),
            "adapter isolation is measured as namespace safety only; XR07 does not load or execute adapter weights".to_owned(),
            "low_n=true means fewer than three passed trials; expensive 16K runs may still be used as evidence when marked".to_owned(),
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR07 prefix cache real reuse A/B: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR07 benchmark blocked; see blockers.md".into())
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
    contexts: Vec<usize>,
    trials: usize,
    suffix_tokens: usize,
    suffix_edit_tokens: usize,
    continued_decode_tokens: usize,
    ram_budget_bytes: u64,
    max_context_tokens: usize,
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
            contexts: vec![4096, 8192, 16_384],
            trials: DEFAULT_TRIALS,
            suffix_tokens: DEFAULT_SUFFIX_TOKENS,
            suffix_edit_tokens: DEFAULT_SUFFIX_EDIT_TOKENS,
            continued_decode_tokens: DEFAULT_CONTINUED_DECODE_TOKENS,
            ram_budget_bytes: DEFAULT_RAM_BUDGET_BYTES,
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
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
                "--trials" => {
                    options.trials =
                        parse_positive_usize(&required_value(&mut args, "--trials")?, "--trials")?
                }
                "--suffix-tokens" => {
                    options.suffix_tokens = parse_positive_usize(
                        &required_value(&mut args, "--suffix-tokens")?,
                        "--suffix-tokens",
                    )?
                }
                "--suffix-edit-tokens" => {
                    options.suffix_edit_tokens = parse_positive_usize(
                        &required_value(&mut args, "--suffix-edit-tokens")?,
                        "--suffix-edit-tokens",
                    )?
                }
                "--continued-decode-tokens" => {
                    options.continued_decode_tokens = parse_positive_usize(
                        &required_value(&mut args, "--continued-decode-tokens")?,
                        "--continued-decode-tokens",
                    )?
                }
                "--ram-budget-bytes" => {
                    options.ram_budget_bytes = parse_positive_u64(
                        &required_value(&mut args, "--ram-budget-bytes")?,
                        "--ram-budget-bytes",
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
        if options.suffix_edit_tokens > options.suffix_tokens {
            return Err(CliError::Usage(
                "--suffix-edit-tokens cannot exceed --suffix-tokens".to_owned(),
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
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    requested_trials: usize,
    contexts: Vec<usize>,
    suffix_tokens: usize,
    suffix_edit_tokens: usize,
    continued_decode_tokens: usize,
    ram_budget_bytes: u64,
    logit_tolerance: f64,
    warm_speedup_gate: f64,
    warm_improvement_gate_ms: f64,
    memory_cliff_gb: f64,
    tiny16_cap_memory_fraction: f64,
    selected_cases: Vec<SelectedCase>,
    record_count: usize,
    passed_records: usize,
    failed_records: usize,
    aggregates: Vec<Aggregate>,
    cap_recommendation: CapRecommendation,
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
    derived_case_seed: u64,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    context_tokens: usize,
    shared_prefix_tokens: usize,
    suffix_tokens: usize,
    suffix_edit_distance: usize,
    suffix_b_source: String,
    prefix_token_hash: String,
    full_a_token_hash: String,
    full_b_token_hash: String,
    alternate_workload_id: Option<String>,
    alternate_prompt_sha256: Option<String>,
}

#[derive(Debug, Clone)]
struct WorkloadCase {
    selected: SelectedCase,
    full_tokens_b: Vec<i32>,
    shared_prefix_tokens: Vec<i32>,
    suffix_b_tokens: Vec<i32>,
}

impl WorkloadCase {
    fn case_id(&self) -> &str {
        &self.selected.case_id
    }

    fn workload_id(&self) -> &str {
        &self.selected.workload_id
    }

    fn context_tokens(&self) -> usize {
        self.selected.context_tokens
    }

    fn suffix_tokens(&self) -> usize {
        self.selected.suffix_tokens
    }
}

#[derive(Debug, Clone)]
struct EncodedWorkload {
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
    case_id: String,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    source_deterministic_seed: u64,
    derived_case_seed: u64,
    trial_index: usize,
    context_tokens: usize,
    shared_prefix_tokens: usize,
    suffix_tokens: usize,
    suffix_edit_distance: usize,
    suffix_b_source: String,
    prefix_token_hash: String,
    full_b_token_hash: String,
    mode: String,
    cache_mode: String,
    adapter_namespace_case: AdapterNamespaceCase,
    model_load_ms: f64,
    prefix_population: PrefixPopulation,
    fresh_full_prefill: FreshFullPrefill,
    warm_restore: WarmRestore,
    continued_decode: ContinuedDecode,
    namespace_safety: NamespaceSafety,
    accounting: CacheAccountingSnapshot,
    gate: GateOutcome,
    status: String,
    blocker: Option<String>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PrefixPopulation {
    prefill_ms: f64,
    snapshot_export_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    snapshot_sequence_len: u64,
    snapshot_token_count: u64,
    snapshot_has_last_step: bool,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
struct FreshFullPrefill {
    ttft_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WarmRestore {
    ttft_ms: f64,
    lookup_ms: f64,
    native_import_ms: f64,
    suffix_replay_ms: f64,
    replayed_suffix_tokens: usize,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    restored_block_id: String,
    restored_native_handle_id: Option<u64>,
    reuse_hit_rate: f64,
    ttft_improvement_ms: f64,
    ttft_speedup: f64,
    token_parity: bool,
    logit_delta: f64,
    sequence_len_parity: bool,
    active_kv_bytes_parity: bool,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ContinuedDecode {
    requested_tokens: usize,
    generated_tokens: usize,
    fresh_decode_ms: f64,
    restored_decode_ms: f64,
    fresh_output_token_ids: Vec<i32>,
    restored_output_token_ids: Vec<i32>,
    token_parity: bool,
    max_logit_delta: f64,
    sequence_len_parity: bool,
}

#[derive(Debug, Clone, Serialize)]
struct NamespaceSafety {
    base_namespace_hash: String,
    adapter_namespace_hash: String,
    base_block_id: String,
    adapter_block_id: String,
    base_and_adapter_namespaces_differ: bool,
    base_and_adapter_block_ids_differ: bool,
    base_to_adapter_rejected: bool,
    adapter_to_base_rejected: bool,
    wrong_cache_mode_rejected: bool,
    same_namespace_miss_recorded: bool,
}

#[derive(Debug, Clone, Serialize)]
struct AdapterNamespaceCase {
    base_adapter_id: Option<String>,
    adapter_id: String,
    adapter_weight_hash: String,
    execution_scope: String,
}

#[derive(Debug, Clone, Serialize)]
struct GateOutcome {
    passed: bool,
    restored_continuation_parity: bool,
    continued_decode_parity: bool,
    warm_ttft_meaningful: bool,
    no_cross_adapter_reuse: bool,
    no_cross_cache_mode_reuse: bool,
    cache_accounting_present: bool,
    memory_below_cliff: bool,
    logit_tolerance: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Aggregate {
    case_id: String,
    workload_id: String,
    family: String,
    context_tokens: usize,
    trial_count: usize,
    passed_trials: usize,
    low_n: bool,
    fresh_full_prefill_ms_median: Option<f64>,
    warm_ttft_ms_median: Option<f64>,
    warm_speedup_median: Option<f64>,
    warm_improvement_ms_median: Option<f64>,
    restore_lookup_ms_median: Option<f64>,
    native_import_ms_median: Option<f64>,
    suffix_replay_ms_median: Option<f64>,
    prefix_prefill_ms_median: Option<f64>,
    snapshot_export_ms_median: Option<f64>,
    active_kv_bytes_max: Option<u64>,
    cache_resident_bytes_max: Option<u64>,
    accounting_hit_rate_median: Option<f64>,
    reuse_hit_rate_median: Option<f64>,
    peak_mlx_max_gb: Option<f64>,
    correctness_passed: bool,
    namespace_safety_passed: bool,
    warm_ttft_meaningful: bool,
    memory_gate_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CapRecommendation {
    recommended_tiny16_cap_bytes: Option<u64>,
    recommended_tiny16_cap_mib: Option<u64>,
    max_base_only_entry_bytes: Option<u64>,
    max_observed_cache_resident_bytes: Option<u64>,
    hw_memsize_bytes: Option<u64>,
    memory_fraction_limit: f64,
    memory_fraction_limit_bytes: Option<u64>,
    cap_gate_passed: bool,
    default_policy: String,
    rationale: String,
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
) -> Result<Record, Box<dyn std::error::Error>> {
    let policy = KvPolicy::default();
    let block_size =
        NonZeroU64::new(case.shared_prefix_tokens.len() as u64).expect("prefix is non-zero");
    let base_namespace = namespace_for(
        model_identity,
        &case.shared_prefix_tokens,
        None,
        None,
        CacheMode::Bf16,
    )?;
    let adapter_namespace = namespace_for(
        model_identity,
        &case.shared_prefix_tokens,
        Some("xr07-fixture-lora"),
        Some(&adapter_weight_hash(case)),
        CacheMode::Bf16,
    )?;
    let base_namespace_hash = base_namespace.namespace_hash()?.0;
    let adapter_namespace_hash = adapter_namespace.namespace_hash()?.0;
    let mut ram_cache = RamPrefixCache::new(
        NonZeroU64::new(options.ram_budget_bytes).expect("RAM budget is non-zero"),
    );

    let mut prefix_cache = KvCache::create(&policy)?;
    let prefix_started = Instant::now();
    let prefix_step = prefill(target, &mut prefix_cache, &case.shared_prefix_tokens)?;
    let prefix_prefill_ms = duration_ms(prefix_started.elapsed());
    let export_started = Instant::now();
    let snapshot = prefix_cache.export_snapshot()?;
    let snapshot_export_ms = duration_ms(export_started.elapsed());
    let snapshot_info = snapshot.info()?;
    let observation = PrefillObservation {
        sequence_len: prefix_step.sequence_len,
        greedy_token: prefix_step.greedy_token as u32,
        greedy_logit_bits: prefix_step.greedy_logit.to_bits(),
    };
    let native_handle_id = native_handle_id(case.context_tokens(), trial_index);
    let base_block = RamPrefixBlock::from_observation(
        base_namespace.clone(),
        0,
        block_size,
        0,
        observation,
        snapshot_info.active_kv_bytes,
    )?
    .with_native_handle(native_handle_id);
    let base_key = base_block.key.clone();
    ram_cache.insert(base_block)?;

    let mut fresh_cache = KvCache::create(&policy)?;
    let fresh_started = Instant::now();
    let fresh_step = prefill(target, &mut fresh_cache, &case.full_tokens_b)?;
    let fresh_full_prefill_ms = duration_ms(fresh_started.elapsed());

    let mut restored_cache = KvCache::create(&policy)?;
    let warm_started = Instant::now();
    let lookup_started = Instant::now();
    let restored = ram_cache.restore(&base_key, &base_namespace)?;
    let lookup_ms = duration_ms(lookup_started.elapsed());
    let native_import_started = Instant::now();
    restored_cache.import_snapshot(&snapshot)?;
    let native_import_ms = duration_ms(native_import_started.elapsed());
    let suffix_replay_started = Instant::now();
    let mut warm_step = restored_cache.last_step()?;
    let mut warm_peak_mlx_gb = f64::from(warm_step.peak_memory_gb);
    let mut warm_peak_rss_mb = f64::from(warm_step.peak_rss_mb);
    for token in &case.suffix_b_tokens {
        warm_step = decode_one(target, &mut restored_cache, *token)?;
        warm_peak_mlx_gb = warm_peak_mlx_gb.max(f64::from(warm_step.peak_memory_gb));
        warm_peak_rss_mb = warm_peak_rss_mb.max(f64::from(warm_step.peak_rss_mb));
    }
    let suffix_replay_ms = duration_ms(suffix_replay_started.elapsed());
    let warm_ttft_ms = duration_ms(warm_started.elapsed());

    let (continued_decode, continued_peak_mlx, continued_peak_rss) = run_continued_decode(
        options,
        target,
        &mut fresh_cache,
        &mut restored_cache,
        fresh_step.greedy_token,
    )?;
    warm_peak_mlx_gb = warm_peak_mlx_gb.max(continued_peak_mlx);
    warm_peak_rss_mb = warm_peak_rss_mb.max(continued_peak_rss);

    let adapter_weight_hash = adapter_weight_hash(case);
    let adapter_block = RamPrefixBlock::from_observation(
        adapter_namespace.clone(),
        0,
        block_size,
        0,
        observation,
        snapshot_info.active_kv_bytes,
    )?
    .with_native_handle(native_handle_id + 1);
    let adapter_key = adapter_block.key.clone();
    ram_cache.insert(adapter_block)?;

    let base_to_adapter_rejected =
        namespace_rejected(&mut ram_cache, &base_key, adapter_namespace.clone());
    let adapter_to_base_rejected =
        namespace_rejected(&mut ram_cache, &adapter_key, base_namespace.clone());
    let wrong_cache_mode_rejected = namespace_rejected(
        &mut ram_cache,
        &base_key,
        base_namespace
            .clone()
            .with_cache_mode(CacheMode::MlxAffineQ8),
    );
    let missing_key = KvBlockKey::new(
        &base_namespace,
        99,
        block_size,
        0,
        case.shared_prefix_tokens.len() as u64,
    )?;
    let same_namespace_miss_recorded = matches!(
        ram_cache.restore(&missing_key, &base_namespace),
        Err(KvError::NotFound { .. })
    );
    let accounting = ram_cache.accounting();

    let warm_logit_delta =
        (f64::from(fresh_step.greedy_logit) - f64::from(warm_step.greedy_logit)).abs();
    let warm_speedup = if warm_ttft_ms > 0.0 {
        fresh_full_prefill_ms / warm_ttft_ms
    } else {
        0.0
    };
    let warm_improvement_ms = fresh_full_prefill_ms - warm_ttft_ms;
    let warm_ttft_meaningful =
        warm_speedup >= WARM_SPEEDUP_GATE && warm_improvement_ms >= WARM_IMPROVEMENT_GATE_MS;
    let restored_continuation_parity = fresh_step.greedy_token == warm_step.greedy_token
        && warm_logit_delta <= LOGIT_TOLERANCE
        && fresh_step.sequence_len == warm_step.sequence_len;
    let namespace_safety = NamespaceSafety {
        base_namespace_hash: base_namespace_hash.clone(),
        adapter_namespace_hash: adapter_namespace_hash.clone(),
        base_block_id: base_key.block_id.0.clone(),
        adapter_block_id: adapter_key.block_id.0.clone(),
        base_and_adapter_namespaces_differ: base_namespace_hash != adapter_namespace_hash,
        base_and_adapter_block_ids_differ: base_key.block_id.0 != adapter_key.block_id.0,
        base_to_adapter_rejected,
        adapter_to_base_rejected,
        wrong_cache_mode_rejected,
        same_namespace_miss_recorded,
    };
    let no_cross_adapter_reuse = namespace_safety.base_and_adapter_namespaces_differ
        && namespace_safety.base_and_adapter_block_ids_differ
        && namespace_safety.base_to_adapter_rejected
        && namespace_safety.adapter_to_base_rejected;
    let no_cross_cache_mode_reuse = namespace_safety.wrong_cache_mode_rejected;
    let cache_accounting_present = accounting.hits >= 1
        && accounting.misses >= 1
        && accounting.restore_failures >= 3
        && accounting.resident_bytes >= snapshot_info.active_kv_bytes;
    let peak_memory = f64::from(prefix_step.peak_memory_gb)
        .max(f64::from(fresh_step.peak_memory_gb))
        .max(warm_peak_mlx_gb);
    let memory_below_cliff = peak_memory < MEMORY_CLIFF_GB;
    let mut gate = GateOutcome {
        passed: false,
        restored_continuation_parity,
        continued_decode_parity: continued_decode.token_parity
            && continued_decode.sequence_len_parity
            && continued_decode.max_logit_delta <= LOGIT_TOLERANCE,
        warm_ttft_meaningful,
        no_cross_adapter_reuse,
        no_cross_cache_mode_reuse,
        cache_accounting_present,
        memory_below_cliff,
        logit_tolerance: LOGIT_TOLERANCE,
    };
    gate.passed = gate.restored_continuation_parity
        && gate.continued_decode_parity
        && gate.warm_ttft_meaningful
        && gate.no_cross_adapter_reuse
        && gate.no_cross_cache_mode_reuse
        && gate.cache_accounting_present
        && gate.memory_below_cliff;

    let mut notes = Vec::new();
    if !gate.warm_ttft_meaningful {
        notes.push(format!(
            "warm TTFT speedup {:.2}x and improvement {:.3} ms did not meet {:.2}x/{:.1} ms gate",
            warm_speedup, warm_improvement_ms, WARM_SPEEDUP_GATE, WARM_IMPROVEMENT_GATE_MS
        ));
    }
    if !gate.restored_continuation_parity {
        notes
            .push("restored full-context continuation did not match fresh full prefill".to_owned());
    }
    if !gate.continued_decode_parity {
        notes.push(
            "continued greedy decode after restore did not match fresh continuation".to_owned(),
        );
    }
    if !gate.no_cross_adapter_reuse {
        notes.push("adapter namespace isolation check failed".to_owned());
    }
    if !gate.no_cross_cache_mode_reuse {
        notes.push("cache-mode namespace isolation check failed".to_owned());
    }
    if !gate.cache_accounting_present {
        notes.push(
            "RAM prefix cache accounting did not record expected hit/miss/failure metrics"
                .to_owned(),
        );
    }
    if !gate.memory_below_cliff {
        notes.push(format!(
            "peak MLX memory {:.3} GB crossed {:.1} GB cliff",
            peak_memory, MEMORY_CLIFF_GB
        ));
    }

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
        derived_case_seed: case.selected.derived_case_seed,
        trial_index,
        context_tokens: case.selected.context_tokens,
        shared_prefix_tokens: case.selected.shared_prefix_tokens,
        suffix_tokens: case.selected.suffix_tokens,
        suffix_edit_distance: case.selected.suffix_edit_distance,
        suffix_b_source: case.selected.suffix_b_source.clone(),
        prefix_token_hash: case.selected.prefix_token_hash.clone(),
        full_b_token_hash: case.selected.full_b_token_hash.clone(),
        mode: MODE.to_owned(),
        cache_mode: CacheMode::Bf16.label().to_owned(),
        adapter_namespace_case: AdapterNamespaceCase {
            base_adapter_id: None,
            adapter_id: "xr07-fixture-lora".to_owned(),
            adapter_weight_hash,
            execution_scope: "namespace_isolation_only_no_adapter_weights_loaded".to_owned(),
        },
        model_load_ms,
        prefix_population: PrefixPopulation {
            prefill_ms: prefix_prefill_ms,
            snapshot_export_ms,
            greedy_token: prefix_step.greedy_token,
            greedy_logit: prefix_step.greedy_logit,
            sequence_len: prefix_step.sequence_len,
            active_kv_bytes: prefix_step.active_kv_bytes,
            snapshot_sequence_len: snapshot_info.sequence_len,
            snapshot_token_count: snapshot_info.token_count,
            snapshot_has_last_step: snapshot_info.has_last_step,
            peak_memory_gb: f64::from(prefix_step.peak_memory_gb),
            peak_rss_mb: f64::from(prefix_step.peak_rss_mb),
        },
        fresh_full_prefill: FreshFullPrefill {
            ttft_ms: fresh_full_prefill_ms,
            greedy_token: fresh_step.greedy_token,
            greedy_logit: fresh_step.greedy_logit,
            sequence_len: fresh_step.sequence_len,
            active_kv_bytes: fresh_step.active_kv_bytes,
            peak_memory_gb: f64::from(fresh_step.peak_memory_gb),
            peak_rss_mb: f64::from(fresh_step.peak_rss_mb),
        },
        warm_restore: WarmRestore {
            ttft_ms: warm_ttft_ms,
            lookup_ms,
            native_import_ms,
            suffix_replay_ms,
            replayed_suffix_tokens: case.suffix_b_tokens.len(),
            greedy_token: warm_step.greedy_token,
            greedy_logit: warm_step.greedy_logit,
            sequence_len: warm_step.sequence_len,
            active_kv_bytes: warm_step.active_kv_bytes,
            restored_block_id: restored.block_id.0,
            restored_native_handle_id: restored.native_handle.map(|handle| handle.handle_id),
            reuse_hit_rate: 1.0,
            ttft_improvement_ms: warm_improvement_ms,
            ttft_speedup: warm_speedup,
            token_parity: fresh_step.greedy_token == warm_step.greedy_token,
            logit_delta: warm_logit_delta,
            sequence_len_parity: fresh_step.sequence_len == warm_step.sequence_len,
            active_kv_bytes_parity: fresh_step.active_kv_bytes == warm_step.active_kv_bytes,
            peak_memory_gb: warm_peak_mlx_gb,
            peak_rss_mb: warm_peak_rss_mb,
        },
        continued_decode,
        namespace_safety,
        accounting,
        gate,
        status: "passed".to_owned(),
        blocker: None,
        notes,
    })
}

fn run_continued_decode(
    options: &Options,
    target: &Target,
    fresh_cache: &mut KvCache,
    restored_cache: &mut KvCache,
    first_input_token: i32,
) -> Result<(ContinuedDecode, f64, f64), Box<dyn std::error::Error>> {
    let mut fresh_input = first_input_token;
    let mut restored_input = first_input_token;
    let mut fresh_tokens = Vec::with_capacity(options.continued_decode_tokens);
    let mut restored_tokens = Vec::with_capacity(options.continued_decode_tokens);
    let mut fresh_logits = Vec::with_capacity(options.continued_decode_tokens);
    let mut restored_logits = Vec::with_capacity(options.continued_decode_tokens);
    let mut sequence_len_parity = true;
    let mut peak_mlx = 0.0_f64;
    let mut peak_rss = 0.0_f64;

    let fresh_started = Instant::now();
    let mut fresh_steps = Vec::with_capacity(options.continued_decode_tokens);
    for _ in 0..options.continued_decode_tokens {
        let step = decode_one(target, fresh_cache, fresh_input)?;
        fresh_input = step.greedy_token;
        peak_mlx = peak_mlx.max(f64::from(step.peak_memory_gb));
        peak_rss = peak_rss.max(f64::from(step.peak_rss_mb));
        fresh_steps.push(step);
    }
    let fresh_decode_ms = duration_ms(fresh_started.elapsed());

    let restored_started = Instant::now();
    for fresh_step in &fresh_steps {
        let step = decode_one(target, restored_cache, restored_input)?;
        restored_input = step.greedy_token;
        peak_mlx = peak_mlx.max(f64::from(step.peak_memory_gb));
        peak_rss = peak_rss.max(f64::from(step.peak_rss_mb));
        sequence_len_parity &= fresh_step.sequence_len == step.sequence_len;
        restored_tokens.push(step.greedy_token);
        restored_logits.push(f64::from(step.greedy_logit));
    }
    let restored_decode_ms = duration_ms(restored_started.elapsed());

    for step in &fresh_steps {
        fresh_tokens.push(step.greedy_token);
        fresh_logits.push(f64::from(step.greedy_logit));
    }
    let token_parity = fresh_tokens == restored_tokens;
    let max_logit_delta = max_logit_delta(&fresh_logits, &restored_logits).unwrap_or(f64::INFINITY);
    Ok((
        ContinuedDecode {
            requested_tokens: options.continued_decode_tokens,
            generated_tokens: restored_tokens.len(),
            fresh_decode_ms,
            restored_decode_ms,
            fresh_output_token_ids: fresh_tokens,
            restored_output_token_ids: restored_tokens,
            token_parity,
            max_logit_delta,
            sequence_len_parity,
        },
        peak_mlx,
        peak_rss,
    ))
}

fn failed_record(
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    case: &WorkloadCase,
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
        derived_case_seed: case.selected.derived_case_seed,
        trial_index,
        context_tokens: case.selected.context_tokens,
        shared_prefix_tokens: case.selected.shared_prefix_tokens,
        suffix_tokens: case.selected.suffix_tokens,
        suffix_edit_distance: case.selected.suffix_edit_distance,
        suffix_b_source: case.selected.suffix_b_source.clone(),
        prefix_token_hash: case.selected.prefix_token_hash.clone(),
        full_b_token_hash: case.selected.full_b_token_hash.clone(),
        mode: MODE.to_owned(),
        cache_mode: CacheMode::Bf16.label().to_owned(),
        adapter_namespace_case: AdapterNamespaceCase {
            base_adapter_id: None,
            adapter_id: "xr07-fixture-lora".to_owned(),
            adapter_weight_hash: adapter_weight_hash(case),
            execution_scope: "namespace_isolation_only_no_adapter_weights_loaded".to_owned(),
        },
        model_load_ms,
        prefix_population: PrefixPopulation {
            prefill_ms: 0.0,
            snapshot_export_ms: 0.0,
            greedy_token: 0,
            greedy_logit: 0.0,
            sequence_len: 0,
            active_kv_bytes: 0,
            snapshot_sequence_len: 0,
            snapshot_token_count: 0,
            snapshot_has_last_step: false,
            peak_memory_gb: 0.0,
            peak_rss_mb: 0.0,
        },
        fresh_full_prefill: FreshFullPrefill {
            ttft_ms: 0.0,
            greedy_token: 0,
            greedy_logit: 0.0,
            sequence_len: 0,
            active_kv_bytes: 0,
            peak_memory_gb: 0.0,
            peak_rss_mb: 0.0,
        },
        warm_restore: WarmRestore {
            ttft_ms: 0.0,
            lookup_ms: 0.0,
            native_import_ms: 0.0,
            suffix_replay_ms: 0.0,
            replayed_suffix_tokens: 0,
            greedy_token: 0,
            greedy_logit: 0.0,
            sequence_len: 0,
            active_kv_bytes: 0,
            restored_block_id: "unavailable".to_owned(),
            restored_native_handle_id: None,
            reuse_hit_rate: 0.0,
            ttft_improvement_ms: 0.0,
            ttft_speedup: 0.0,
            token_parity: false,
            logit_delta: f64::INFINITY,
            sequence_len_parity: false,
            active_kv_bytes_parity: false,
            peak_memory_gb: 0.0,
            peak_rss_mb: 0.0,
        },
        continued_decode: ContinuedDecode {
            requested_tokens: 0,
            generated_tokens: 0,
            fresh_decode_ms: 0.0,
            restored_decode_ms: 0.0,
            fresh_output_token_ids: Vec::new(),
            restored_output_token_ids: Vec::new(),
            token_parity: false,
            max_logit_delta: f64::INFINITY,
            sequence_len_parity: false,
        },
        namespace_safety: NamespaceSafety {
            base_namespace_hash: "unavailable".to_owned(),
            adapter_namespace_hash: "unavailable".to_owned(),
            base_block_id: "unavailable".to_owned(),
            adapter_block_id: "unavailable".to_owned(),
            base_and_adapter_namespaces_differ: false,
            base_and_adapter_block_ids_differ: false,
            base_to_adapter_rejected: false,
            adapter_to_base_rejected: false,
            wrong_cache_mode_rejected: false,
            same_namespace_miss_recorded: false,
        },
        accounting: CacheAccountingSnapshot {
            budget_bytes: 0,
            resident_bytes: 0,
            resident_blocks: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            restore_failures: 0,
            hit_rate: 0.0,
            ssd_enabled: false,
        },
        gate: GateOutcome {
            passed: false,
            restored_continuation_parity: false,
            continued_decode_parity: false,
            warm_ttft_meaningful: false,
            no_cross_adapter_reuse: false,
            no_cross_cache_mode_reuse: false,
            cache_accounting_present: false,
            memory_below_cliff: false,
            logit_tolerance: LOGIT_TOLERANCE,
        },
        status: "failed".to_owned(),
        blocker: Some(blocker),
        notes: Vec::new(),
    }
}

fn prepare_cases(
    options: &Options,
    workloads: &[WorkloadRecord],
    tokenizer: &mut TokenizerHelper,
) -> Result<Vec<WorkloadCase>, CliError> {
    let mut encoded = BTreeMap::new();
    let required_ids = required_workload_ids(&options.contexts);
    for record in workloads {
        if required_ids.contains(&record.workload_id) {
            encoded.insert(
                record.workload_id.clone(),
                encode_workload(tokenizer, record)?,
            );
        }
    }

    let mut cases = Vec::with_capacity(options.contexts.len());
    for context_tokens in &options.contexts {
        let source_id = source_workload_id(*context_tokens)?;
        let source = encoded.get(source_id).ok_or_else(|| {
            CliError::Runtime(format!(
                "required workload {source_id} missing from {}",
                options.workloads_path.display()
            ))
        })?;
        if source.token_ids.len() < *context_tokens {
            return Err(CliError::Runtime(format!(
                "{} has {} tokens, fewer than requested context {}",
                source.record.workload_id,
                source.token_ids.len(),
                context_tokens
            )));
        }
        if *context_tokens <= options.suffix_tokens {
            return Err(CliError::Usage(format!(
                "context {context_tokens} must exceed suffix token count {}",
                options.suffix_tokens
            )));
        }
        let source_tokens = source.token_ids[..*context_tokens].to_vec();
        let prefix_len = context_tokens - options.suffix_tokens;
        let shared_prefix_tokens = source_tokens[..prefix_len].to_vec();
        let suffix_a = source_tokens[prefix_len..].to_vec();
        let derived_case_seed =
            source.record.deterministic_seed ^ ((*context_tokens as u64) << 16) ^ 0x5852_3037_u64;
        let alternate = alternate_workload(source_id).and_then(|alternate_id| {
            encoded.get(alternate_id).filter(|candidate| {
                candidate.token_ids.len() >= *context_tokens
                    && candidate.token_ids[..prefix_len] == shared_prefix_tokens
            })
        });
        let (suffix_b_tokens, suffix_b_source, alternate_workload_id, alternate_prompt_sha256) =
            if let Some(alternate) = alternate {
                (
                    alternate.token_ids[prefix_len..*context_tokens].to_vec(),
                    "alternate_common_prefix".to_owned(),
                    Some(alternate.record.workload_id.clone()),
                    Some(alternate.prompt_sha256.clone()),
                )
            } else {
                let edit_tokens = tokenizer.encode(&suffix_edit_text(derived_case_seed))?;
                (
                    edited_suffix(&suffix_a, &edit_tokens, options.suffix_edit_tokens),
                    "deterministic_token_suffix_edit".to_owned(),
                    None,
                    None,
                )
            };
        let full_tokens_b = [&shared_prefix_tokens[..], &suffix_b_tokens[..]].concat();
        let suffix_edit_distance = suffix_a
            .iter()
            .zip(&suffix_b_tokens)
            .filter(|(left, right)| left != right)
            .count();
        if suffix_edit_distance == 0 {
            return Err(CliError::Runtime(format!(
                "{} derived suffix edit had zero token distance",
                source.record.workload_id
            )));
        }
        let full_a_token_hash = token_hash("xr07-full-a-token-ids-v1", &source_tokens);
        let full_b_token_hash = token_hash("xr07-full-b-token-ids-v1", &full_tokens_b);
        let prefix_token_hash = token_hash("xr07-prefix-token-ids-v1", &shared_prefix_tokens);
        let case_id = format!(
            "xr07_{}k_{}",
            context_tokens / 1024,
            source.record.workload_id
        );
        cases.push(WorkloadCase {
            selected: SelectedCase {
                case_id,
                workload_id: source.record.workload_id.clone(),
                family: source.record.family.clone(),
                prompt_path: source.record.prompt_path.clone(),
                prompt_sha256: source.prompt_sha256.clone(),
                source_deterministic_seed: source.record.deterministic_seed,
                derived_case_seed,
                target_context_tokens: source.record.target_context_tokens,
                actual_context_tokens: source.record.actual_context_tokens,
                context_tokens: *context_tokens,
                shared_prefix_tokens: shared_prefix_tokens.len(),
                suffix_tokens: suffix_b_tokens.len(),
                suffix_edit_distance,
                suffix_b_source,
                prefix_token_hash,
                full_a_token_hash,
                full_b_token_hash,
                alternate_workload_id,
                alternate_prompt_sha256,
            },
            full_tokens_b,
            shared_prefix_tokens,
            suffix_b_tokens,
        });
    }
    Ok(cases)
}

fn source_workload_id(context_tokens: usize) -> Result<&'static str, CliError> {
    match context_tokens {
        4096 => Ok("code_review_rust_4k_001"),
        8192 => Ok("prefix_reuse_edit_8k_a_001"),
        16_384 => Ok("long_repo_pack_16k_001"),
        other => Err(CliError::Usage(format!(
            "XR07 supports 4K/8K/16K real reuse contexts; unsupported context {other}"
        ))),
    }
}

fn alternate_workload(source_id: &str) -> Option<&'static str> {
    match source_id {
        "prefix_reuse_edit_8k_a_001" => Some("prefix_reuse_edit_8k_b_001"),
        _ => None,
    }
}

fn required_workload_ids(contexts: &[usize]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for context in contexts {
        if let Ok(source_id) = source_workload_id(*context) {
            out.insert(source_id.to_owned());
            if let Some(alternate_id) = alternate_workload(source_id) {
                out.insert(alternate_id.to_owned());
            }
        }
    }
    out
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

fn edited_suffix(suffix_a: &[i32], edit_tokens: &[i32], edit_count: usize) -> Vec<i32> {
    let mut suffix_b = suffix_a.to_vec();
    if suffix_b.is_empty() || edit_tokens.is_empty() || edit_count == 0 {
        return suffix_b;
    }
    let edit_count = edit_count.min(suffix_b.len());
    let start = suffix_b.len() - edit_count;
    for offset in 0..edit_count {
        let index = start + offset;
        let current = suffix_b[index];
        let replacement = edit_tokens
            .iter()
            .copied()
            .find(|token| *token != current)
            .unwrap_or(current);
        suffix_b[index] = replacement;
    }
    suffix_b
}

fn suffix_edit_text(seed: u64) -> String {
    format!(
        "\n\nXR07 deterministic suffix edit seed {seed}: preserve the shared repo context, then answer the cache safety question with blockers first."
    )
}

fn namespace_for(
    model_identity: &manifest::ArtifactIdentity,
    prefix_tokens: &[i32],
    adapter_id: Option<&str>,
    adapter_weight_hash: Option<&str>,
    cache_mode: CacheMode,
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
        prompt_token_hash: token_hash("xr07-prefix-token-ids-v1", prefix_tokens),
        raw_prompt_hash: token_hash("xr07-prefix-raw-token-ids-v1", prefix_tokens),
        adapter_id: adapter_id.map(str::to_owned),
        adapter_weight_hash: adapter_weight_hash.map(str::to_owned),
        kv_layout_version: KV_LAYOUT_VERSION,
        cache_mode,
        mlx_version: version.backend_version,
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

fn namespace_rejected(
    cache: &mut RamPrefixCache,
    key: &KvBlockKey,
    namespace: KvNamespace,
) -> bool {
    matches!(
        cache.restore(key, &namespace),
        Err(KvError::NamespaceMismatch { .. })
    )
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

fn build_aggregates(records: &[Record]) -> Vec<Aggregate> {
    let keys = records
        .iter()
        .map(|record| {
            (
                record.case_id.clone(),
                record.workload_id.clone(),
                record.family.clone(),
                record.context_tokens,
            )
        })
        .collect::<BTreeSet<_>>();
    let mut out = Vec::new();
    for (case_id, workload_id, family, context_tokens) in keys {
        let group = records
            .iter()
            .filter(|record| record.case_id == case_id)
            .collect::<Vec<_>>();
        let passed = group
            .iter()
            .filter(|record| record.status == "passed")
            .copied()
            .collect::<Vec<_>>();
        let fresh = passed
            .iter()
            .map(|record| record.fresh_full_prefill.ttft_ms)
            .collect::<Vec<_>>();
        let warm = passed
            .iter()
            .map(|record| record.warm_restore.ttft_ms)
            .collect::<Vec<_>>();
        let speedup = passed
            .iter()
            .map(|record| record.warm_restore.ttft_speedup)
            .collect::<Vec<_>>();
        let improvement = passed
            .iter()
            .map(|record| record.warm_restore.ttft_improvement_ms)
            .collect::<Vec<_>>();
        let lookup = passed
            .iter()
            .map(|record| record.warm_restore.lookup_ms)
            .collect::<Vec<_>>();
        let native_import = passed
            .iter()
            .map(|record| record.warm_restore.native_import_ms)
            .collect::<Vec<_>>();
        let suffix_replay = passed
            .iter()
            .map(|record| record.warm_restore.suffix_replay_ms)
            .collect::<Vec<_>>();
        let prefix_prefill = passed
            .iter()
            .map(|record| record.prefix_population.prefill_ms)
            .collect::<Vec<_>>();
        let snapshot_export = passed
            .iter()
            .map(|record| record.prefix_population.snapshot_export_ms)
            .collect::<Vec<_>>();
        let hit_rate = passed
            .iter()
            .map(|record| record.accounting.hit_rate)
            .collect::<Vec<_>>();
        let reuse_hit_rate = passed
            .iter()
            .map(|record| record.warm_restore.reuse_hit_rate)
            .collect::<Vec<_>>();
        let peak = passed
            .iter()
            .map(|record| {
                record
                    .prefix_population
                    .peak_memory_gb
                    .max(record.fresh_full_prefill.peak_memory_gb)
                    .max(record.warm_restore.peak_memory_gb)
            })
            .collect::<Vec<_>>();
        let active_kv_bytes_max = passed
            .iter()
            .map(|record| record.fresh_full_prefill.active_kv_bytes)
            .max();
        let cache_resident_bytes_max = passed
            .iter()
            .map(|record| record.accounting.resident_bytes)
            .max();
        let correctness_passed = !group.is_empty()
            && group.iter().all(|record| {
                record.status == "passed"
                    && record.gate.restored_continuation_parity
                    && record.gate.continued_decode_parity
            });
        let namespace_safety_passed = !group.is_empty()
            && group.iter().all(|record| {
                record.status == "passed"
                    && record.gate.no_cross_adapter_reuse
                    && record.gate.no_cross_cache_mode_reuse
            });
        let warm_ttft_meaningful =
            !passed.is_empty() && passed.iter().all(|record| record.gate.warm_ttft_meaningful);
        let memory_gate_passed =
            !passed.is_empty() && passed.iter().all(|record| record.gate.memory_below_cliff);
        out.push(Aggregate {
            case_id,
            workload_id,
            family,
            context_tokens,
            trial_count: group.len(),
            passed_trials: passed.len(),
            low_n: passed.len() < 3,
            fresh_full_prefill_ms_median: percentile(fresh, 0.50),
            warm_ttft_ms_median: percentile(warm, 0.50),
            warm_speedup_median: percentile(speedup, 0.50),
            warm_improvement_ms_median: percentile(improvement, 0.50),
            restore_lookup_ms_median: percentile(lookup, 0.50),
            native_import_ms_median: percentile(native_import, 0.50),
            suffix_replay_ms_median: percentile(suffix_replay, 0.50),
            prefix_prefill_ms_median: percentile(prefix_prefill, 0.50),
            snapshot_export_ms_median: percentile(snapshot_export, 0.50),
            active_kv_bytes_max,
            cache_resident_bytes_max,
            accounting_hit_rate_median: percentile(hit_rate, 0.50),
            reuse_hit_rate_median: percentile(reuse_hit_rate, 0.50),
            peak_mlx_max_gb: max_value(&peak),
            correctness_passed,
            namespace_safety_passed,
            warm_ttft_meaningful,
            memory_gate_passed,
        });
    }
    out
}

fn cap_recommendation(
    records: &[Record],
    hw_memsize_bytes: Option<u64>,
    memory_fraction_limit: f64,
) -> CapRecommendation {
    let max_base_only_entry_bytes = records
        .iter()
        .filter(|record| record.status == "passed")
        .map(|record| record.prefix_population.active_kv_bytes)
        .max();
    let recommended_tiny16_cap_bytes =
        max_base_only_entry_bytes.map(|bytes| round_up_mib((bytes as f64 * 1.10) as u64));
    let recommended_tiny16_cap_mib =
        recommended_tiny16_cap_bytes.map(|bytes| bytes / (1024 * 1024));
    let max_observed_cache_resident_bytes = records
        .iter()
        .filter(|record| record.status == "passed")
        .map(|record| record.accounting.resident_bytes)
        .max();
    let memory_fraction_limit_bytes =
        hw_memsize_bytes.map(|bytes| (bytes as f64 * memory_fraction_limit) as u64);
    let cap_gate_passed = recommended_tiny16_cap_bytes
        .zip(memory_fraction_limit_bytes)
        .map(|(recommended, limit)| recommended <= limit)
        .unwrap_or(false);
    let default_policy = if cap_gate_passed {
        "enable_base_only_ram_prefix_cache_by_default_for_tiny16_with_lru_cap".to_owned()
    } else {
        "keep_ram_prefix_cache_experimental_for_tiny16_until_cap_fits_memory_fraction".to_owned()
    };
    let rationale = match (recommended_tiny16_cap_mib, memory_fraction_limit_bytes) {
        (Some(cap_mib), Some(limit_bytes)) if cap_gate_passed => format!(
            "base-only 16K-capable entry fits within {:.0}% tiny16 memory guard: recommended cap {cap_mib} MiB <= {} MiB",
            memory_fraction_limit * 100.0,
            limit_bytes / (1024 * 1024)
        ),
        (Some(cap_mib), Some(limit_bytes)) => format!(
            "recommended cap {cap_mib} MiB exceeds {:.0}% tiny16 memory guard {} MiB",
            memory_fraction_limit * 100.0,
            limit_bytes / (1024 * 1024)
        ),
        _ => "insufficient passed records or machine memory metadata for default cap decision"
            .to_owned(),
    };
    CapRecommendation {
        recommended_tiny16_cap_bytes,
        recommended_tiny16_cap_mib,
        max_base_only_entry_bytes,
        max_observed_cache_resident_bytes,
        hw_memsize_bytes,
        memory_fraction_limit,
        memory_fraction_limit_bytes,
        cap_gate_passed,
        default_policy,
        rationale,
    }
}

fn apply_decision_to_cap_recommendation(
    decision: &str,
    cap_recommendation: &mut CapRecommendation,
) {
    if decision == "accept_candidate" {
        return;
    }
    let candidate_cap = cap_recommendation
        .recommended_tiny16_cap_mib
        .map(|cap| format!("{cap} MiB"))
        .unwrap_or_else(|| "unavailable".to_owned());
    cap_recommendation.default_policy =
        "do_not_enable_ram_prefix_cache_by_default_for_tiny16".to_owned();
    cap_recommendation.rationale = format!(
        "decision {decision} prevents default enablement; candidate cap would be {candidate_cap} if correctness, namespace, speed, and memory blockers are resolved"
    );
}

fn failed_hypotheses(records: &[Record], aggregates: &[Aggregate]) -> Vec<String> {
    let mut out = Vec::new();
    for record in records {
        if record.status != "passed" {
            out.push(format!(
                "{} trial {} failed: {}",
                record.case_id,
                record.trial_index,
                record
                    .blocker
                    .as_deref()
                    .unwrap_or("no blocker detail recorded")
            ));
            continue;
        }
        for note in &record.notes {
            out.push(format!(
                "{} trial {}: {}",
                record.case_id, record.trial_index, note
            ));
        }
    }
    for aggregate in aggregates {
        if !aggregate.warm_ttft_meaningful {
            out.push(format!(
                "{} did not meet warm TTFT gate across all passed trials",
                aggregate.case_id
            ));
        }
        if aggregate.low_n {
            out.push(format!(
                "{} is low-N evidence: {}/{} passed trials",
                aggregate.case_id, aggregate.passed_trials, aggregate.trial_count
            ));
        }
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
    for context in &options.contexts {
        if !selected_cases
            .iter()
            .any(|case| case.context_tokens() == *context)
        {
            blockers.push(format!("{context} token XR07 case is missing"));
        }
    }
    for case in selected_cases {
        let case_records = records
            .iter()
            .filter(|record| record.case_id == case.selected.case_id)
            .collect::<Vec<_>>();
        if case_records.len() != options.trials {
            blockers.push(format!(
                "{} has {} records; expected {}",
                case.case_id(),
                case_records.len(),
                options.trials
            ));
        }
        for record in case_records {
            if record.status != "passed" {
                blockers.push(format!(
                    "{} trial {} failed: {}",
                    record.case_id,
                    record.trial_index,
                    record
                        .blocker
                        .as_deref()
                        .unwrap_or("no blocker detail recorded")
                ));
            } else {
                if !record.gate.restored_continuation_parity {
                    blockers.push(format!(
                        "{} trial {} restored continuation parity failed",
                        record.case_id, record.trial_index
                    ));
                }
                if !record.gate.continued_decode_parity {
                    blockers.push(format!(
                        "{} trial {} continued decode parity failed",
                        record.case_id, record.trial_index
                    ));
                }
                if !record.gate.no_cross_adapter_reuse {
                    blockers.push(format!(
                        "{} trial {} cross-adapter namespace isolation failed",
                        record.case_id, record.trial_index
                    ));
                }
                if !record.gate.no_cross_cache_mode_reuse {
                    blockers.push(format!(
                        "{} trial {} cross-cache-mode namespace isolation failed",
                        record.case_id, record.trial_index
                    ));
                }
            }
        }
    }
    blockers
}

fn decision_for(
    blockers: &[String],
    records: &[Record],
    aggregates: &[Aggregate],
    cap_recommendation: &CapRecommendation,
) -> String {
    if !blockers.is_empty() || records.is_empty() {
        return "blocked_with_evidence".to_owned();
    }
    if aggregates.is_empty() {
        return "blocked_with_evidence".to_owned();
    }
    if !aggregates
        .iter()
        .all(|aggregate| aggregate.correctness_passed && aggregate.namespace_safety_passed)
    {
        return "blocked_with_evidence".to_owned();
    }
    if aggregates
        .iter()
        .all(|aggregate| aggregate.warm_ttft_meaningful && aggregate.memory_gate_passed)
        && cap_recommendation.cap_gate_passed
    {
        "accept_candidate".to_owned()
    } else if aggregates
        .iter()
        .any(|aggregate| !aggregate.warm_ttft_meaningful || !aggregate.memory_gate_passed)
    {
        "reject_candidate".to_owned()
    } else {
        "needs_more_data".to_owned()
    }
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR07 Prefix Cache Real Reuse A/B Report\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Run\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!("| Model | `{}` |\n", summary.model_identity.path));
    out.push_str(&format!(
        "| Tokenizer | `{}` |\n",
        summary.tokenizer_backend
    ));
    out.push_str(&format!(
        "| Trials | `{}`; suffix `{}`; continued decode `{}` |\n",
        summary.requested_trials, summary.suffix_tokens, summary.continued_decode_tokens
    ));
    out.push_str(&format!(
        "| Warm gate | `{:.2}x` and `{:.1} ms` |\n",
        summary.warm_speedup_gate, summary.warm_improvement_gate_ms
    ));
    out.push_str(&format!(
        "| Cap recommendation | `{}` |\n",
        escape_md(&summary.cap_recommendation.rationale)
    ));
    out.push('\n');

    out.push_str("## Workload Cases\n\n");
    out.push_str("| Case | Context | Source | Prefix Tokens | Suffix Tokens | Edit Distance | Seed | Suffix Source |\n");
    out.push_str("|---|---:|---|---:|---:|---:|---:|---|\n");
    for case in &summary.selected_cases {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | {} | {} | {} | {} | `{}` |\n",
            case.case_id,
            case.context_tokens,
            case.workload_id,
            case.shared_prefix_tokens,
            case.suffix_tokens,
            case.suffix_edit_distance,
            case.derived_case_seed,
            case.suffix_b_source
        ));
    }
    out.push('\n');

    out.push_str("## Aggregates\n\n");
    out.push_str("| Case | Trials | Fresh Full ms | Warm TTFT ms | Speedup | Lookup ms | Import ms | Suffix Replay ms | Active KV MiB | Resident MiB | Hit Rate | Peak MLX GB | Correct | Namespace | Meaningful | Low N |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|\n");
    for aggregate in &summary.aggregates {
        out.push_str(&format!(
            "| `{}` | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | `{}` | `{}` | `{}` | `{}` |\n",
            aggregate.case_id,
            aggregate.passed_trials,
            aggregate.trial_count,
            fmt_opt(aggregate.fresh_full_prefill_ms_median),
            fmt_opt(aggregate.warm_ttft_ms_median),
            fmt_opt(aggregate.warm_speedup_median),
            fmt_opt(aggregate.restore_lookup_ms_median),
            fmt_opt(aggregate.native_import_ms_median),
            fmt_opt(aggregate.suffix_replay_ms_median),
            fmt_bytes_mib(aggregate.active_kv_bytes_max),
            fmt_bytes_mib(aggregate.cache_resident_bytes_max),
            fmt_opt(aggregate.accounting_hit_rate_median),
            fmt_opt(aggregate.peak_mlx_max_gb),
            aggregate.correctness_passed,
            aggregate.namespace_safety_passed,
            aggregate.warm_ttft_meaningful,
            aggregate.low_n
        ));
    }
    out.push('\n');

    out.push_str("## Namespace Safety\n\n");
    out.push_str("| Case | Trial | Base/Adapter Hash Differ | Block IDs Differ | Base->Adapter Rejected | Adapter->Base Rejected | Cache Mode Rejected | Same Namespace Miss |\n");
    out.push_str("|---|---:|---|---|---|---|---|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            record.case_id,
            record.trial_index,
            record.namespace_safety.base_and_adapter_namespaces_differ,
            record.namespace_safety.base_and_adapter_block_ids_differ,
            record.namespace_safety.base_to_adapter_rejected,
            record.namespace_safety.adapter_to_base_rejected,
            record.namespace_safety.wrong_cache_mode_rejected,
            record.namespace_safety.same_namespace_miss_recorded
        ));
    }
    out.push('\n');

    out.push_str("## Verification Command\n\n");
    out.push_str("```sh\n");
    out.push_str("GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab -- --out-dir ");
    out.push_str(&summary.out_dir);
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
    out.push_str("# XR07 Prefix Cache Real Reuse A/B Blockers\n\n");
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
    out.push_str("# XR07 Prefix Cache Real Reuse A/B Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("## Evidence\n\n");
    for path in &summary.generated_files {
        out.push_str(&format!("- `{path}`\n"));
    }
    out.push_str("\n## Default Policy\n\n");
    out.push_str(&format!(
        "- Policy: `{}`\n",
        summary.cap_recommendation.default_policy
    ));
    out.push_str(&format!(
        "- Cap: `{}` MiB\n",
        summary
            .cap_recommendation
            .recommended_tiny16_cap_mib
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unavailable".to_owned())
    ));
    out.push_str(&format!(
        "- Rationale: {}\n",
        summary.cap_recommendation.rationale
    ));
    out.push_str("\n## Claim Boundary\n\n");
    out.push_str("- XR07 measures RAM prefix cache reuse for real local prompts with small suffix edits; it does not optimize runtime code.\n");
    out.push_str(
        "- Warm TTFT includes lookup, native snapshot import, and suffix replay overhead.\n",
    );
    out.push_str("- Adapter evidence covers namespace isolation only; adapter-qualified entries remain partitioned and should not share base-only cache keys.\n");
    out.push_str("- Low-N records are flagged in summary and blockers; repeat runs are required before treating the policy as production serving guidance.\n");
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
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for XR07 native MLX evidence".to_owned());
    }
    if env::var("GEMMA4D_USE_NATIVE_GRAPH").ok().as_deref() != Some("1") {
        blockers.push(
            "GEMMA4D_USE_NATIVE_GRAPH=1 is required for XR07 native graph evidence".to_owned(),
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

fn adapter_weight_hash(case: &WorkloadCase) -> String {
    sha256_hex(
        format!(
            "xr07-fixture-lora\0{}\0{}\0{}",
            case.selected.case_id, case.selected.derived_case_seed, case.selected.prefix_token_hash
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

fn native_handle_id(context_tokens: usize, trial_index: usize) -> u64 {
    ((context_tokens as u64) << 32) | trial_index as u64
}

fn round_up_mib(bytes: u64) -> u64 {
    let mib = 1024 * 1024;
    bytes.div_ceil(mib) * mib
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

fn max_value(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .reduce(f64::max)
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

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, CliError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| parse_positive_usize(part, "--contexts"))
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
    format!("xr07-{}-{}", now.as_secs(), now.subsec_nanos())
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
                "GEMMA4D_NATIVE_DECODE_KV_EVAL",
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

fn fmt_bytes_mib(value: Option<u64>) -> String {
    value
        .map(|bytes| format!("{:.3}", bytes as f64 / (1024.0 * 1024.0)))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn usage() -> String {
    format!(
        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr07_prefix_cache_real_reuse_ab -- [--out-dir PATH] [--workloads PATH] [--model-path PATH] [--python PATH] [--trials N] [--clear-contexts] [--context N] [--contexts CSV] [--suffix-tokens N] [--suffix-edit-tokens N] [--continued-decode-tokens N] [--ram-budget-bytes N]\n\ndefault out-dir: {DEFAULT_OUT_DIR}\ndefault workloads: {DEFAULT_WORKLOADS}"
    )
}
