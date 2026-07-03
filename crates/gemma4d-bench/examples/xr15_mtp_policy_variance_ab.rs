use std::{
    collections::BTreeSet,
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{
    BuildProvenance, CliError, capture_build_provenance, manifest, workload_corpus::WorkloadRecord,
};
use gemma4d_ffi::{
    Drafter, KvCache, KvPolicy, LoadConfig, MTP_MAX_DRAFT_TOKENS, Target, decode_one, draft_block,
    prefill, verify_tokens, verify_tokens_terminal_no_lookahead,
};
use gemma4d_tokenizer::sha256_hex;
use serde::{Deserialize, Serialize};

const GOAL: &str = "XR15-mtp-policy-variance-ab";
const MODE: &str = "native_mtp_policy_variance_ab";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR15-mtp-policy-variance-ab";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_ASSISTANT_MODEL: &str = "artifacts/models/gemma-4-12B-it-qat-assistant-4bit";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const DEFAULT_SOURCE_REPLAY: &str = "benchmarks/out/XR14-mtp-policy-autotune/summary.json";
const DEFAULT_TRIALS: usize = 3;
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_MAX_NEW_TOKENS: usize = 64;
const DEFAULT_MIN_SPEEDUP_PERCENT: f64 = 5.0;
const DEFAULT_REGRESSION_GATE_PERCENT: f64 = 5.0;
const DEFAULT_MEMORY_CLIFF_GB: f64 = 14.0;
const DEFAULT_WORKLOAD_IDS: &[&str] = &[
    "benchmark_qa_4k_001",
    "mtp_candidate_1k_001",
    "chat_short_1k_001",
    "code_review_rust_4k_001",
    "mtp_candidate_4k_001",
    "tool_json_1k_001",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1))?;
    fs::create_dir_all(&args.out_dir)?;

    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let decision_path = args.out_dir.join("decision.md");

    let run_id = run_id();
    let build_provenance = capture_build_provenance()?;
    let command = command_line();
    let model_identity =
        manifest::capture_artifact_identity(&args.model_path, "GEMMA4D_MODEL_REVISION");
    let assistant_identity =
        manifest::capture_artifact_identity(&args.assistant_model_path, "GEMMA4D_MTP_REVISION");
    let source_replay = load_source_replay(&args.source_replay_path);
    let source_replay_sha256 = file_sha256(&args.source_replay_path);
    let mut blockers = startup_blockers(&args, &source_replay);
    let workloads = select_workloads(load_workloads(&args.workloads_path)?, &args)?;
    let selected_workloads = selected_workload_rows(&args, &workloads);
    let mut records = Vec::new();
    let mut tokenizer_backend = "not_started".to_owned();

    if blockers.is_empty() {
        let mut tokenizer = TokenizerHelper::start(&args.python, &args.model_path)?;
        tokenizer_backend = tokenizer.backend().to_owned();
        let total_trials = args.warmups + args.trials;
        for workload in &workloads {
            let encoded = encode_workload(&args, &mut tokenizer, workload)?;
            for trial_index in 0..total_trials {
                let trial_kind = if trial_index < args.warmups {
                    "warmup"
                } else {
                    "measured"
                };
                let baseline = run_baseline(&args, &encoded)?;
                for block_size in &args.block_sizes {
                    records.push(run_mtp_record(
                        &args,
                        &run_id,
                        &build_provenance,
                        &encoded,
                        trial_index,
                        trial_kind,
                        *block_size,
                        baseline.clone(),
                    )?);
                }
            }
        }
    }

    blockers.extend(record_blockers(&records));
    blockers.sort();
    blockers.dedup();

    let policy_summaries = policy_summaries(&args, &records);
    let decision = decision_for(&args, &blockers, &records, &policy_summaries);
    let status = if blockers.is_empty() {
        "completed"
    } else {
        "blocked"
    };
    let failed_hypotheses = failed_hypotheses(&policy_summaries);
    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        mode: MODE.to_owned(),
        status: status.to_owned(),
        decision,
        run_id,
        generated_at_unix_seconds: unix_now(),
        command,
        git_sha: build_provenance.git_sha.clone(),
        git_status_short: build_provenance.git_status_short.clone(),
        build_provenance,
        model_identity,
        assistant_identity,
        tokenizer_backend,
        source_replay_summary_path: args.source_replay_path.display().to_string(),
        source_replay_sha256,
        source_replay_run_id: source_replay
            .as_ref()
            .map(|source| source.run_id.clone())
            .unwrap_or_else(|| "unavailable".to_owned()),
        source_replay_decision: source_replay
            .as_ref()
            .map(|source| source.decision.clone())
            .unwrap_or_else(|| "unavailable".to_owned()),
        source_policy_name: "net_latency_guarded_5pct".to_owned(),
        workloads_path: args.workloads_path.display().to_string(),
        out_dir: args.out_dir.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        decision_path: decision_path.display().to_string(),
        generated_files: vec![
            records_path.display().to_string(),
            summary_path.display().to_string(),
            report_path.display().to_string(),
            blockers_path.display().to_string(),
            decision_path.display().to_string(),
        ],
        selected_workloads,
        requested_trials: args.trials,
        warmup_trials: args.warmups,
        max_new_tokens: args.max_new_tokens,
        block_sizes: args.block_sizes.clone(),
        min_speedup_percent: args.min_speedup_percent,
        regression_gate_percent: args.regression_gate_percent,
        memory_cliff_gb: args.memory_cliff_gb,
        experimental_terminal_no_lookahead: args.experimental_terminal_no_lookahead,
        adaptive_zero_accept_run: args.adaptive_zero_accept_run,
        adaptive_min_generated_tokens: args.adaptive_min_generated_tokens,
        low_n: args.trials < 3,
        record_count: records.len(),
        measured_record_count: records.iter().filter(|record| record.measured).count(),
        exact_record_count: records
            .iter()
            .filter(|record| record.comparison.byte_identical)
            .count(),
        policy_summaries,
        blockers,
        failed_hypotheses,
        measurement_notes: measurement_notes(),
        records: records.clone(),
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR15 MTP policy variance A/B: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR15 blocked; see blockers.md".into())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Args {
    out_dir: PathBuf,
    workloads_path: PathBuf,
    model_path: PathBuf,
    assistant_model_path: PathBuf,
    python: PathBuf,
    source_replay_path: PathBuf,
    trials: usize,
    warmups: usize,
    max_new_tokens: usize,
    block_sizes: Vec<usize>,
    workload_ids: Vec<String>,
    max_workloads: Option<usize>,
    min_speedup_percent: f64,
    regression_gate_percent: f64,
    memory_cliff_gb: f64,
    experimental_terminal_no_lookahead: bool,
    adaptive_zero_accept_run: Option<usize>,
    adaptive_min_generated_tokens: usize,
}

impl Args {
    fn parse<I, S>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut out = Self {
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
            workloads_path: PathBuf::from(DEFAULT_WORKLOADS),
            model_path: PathBuf::from(DEFAULT_MODEL),
            assistant_model_path: PathBuf::from(DEFAULT_ASSISTANT_MODEL),
            python: PathBuf::from(DEFAULT_PYTHON),
            source_replay_path: PathBuf::from(DEFAULT_SOURCE_REPLAY),
            trials: DEFAULT_TRIALS,
            warmups: DEFAULT_WARMUPS,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            block_sizes: vec![1, 2],
            workload_ids: DEFAULT_WORKLOAD_IDS
                .iter()
                .map(|workload_id| (*workload_id).to_owned())
                .collect(),
            max_workloads: None,
            min_speedup_percent: DEFAULT_MIN_SPEEDUP_PERCENT,
            regression_gate_percent: DEFAULT_REGRESSION_GATE_PERCENT,
            memory_cliff_gb: DEFAULT_MEMORY_CLIFF_GB,
            experimental_terminal_no_lookahead: false,
            adaptive_zero_accept_run: None,
            adaptive_min_generated_tokens: 0,
        };
        let mut args = args.into_iter().map(Into::into).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => out.out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?),
                "--workloads" | "--workloads-path" => {
                    out.workloads_path = PathBuf::from(required_value(&mut args, "--workloads")?)
                }
                "--model-path" => {
                    out.model_path = PathBuf::from(required_value(&mut args, "--model-path")?)
                }
                "--assistant-model-path" => {
                    out.assistant_model_path =
                        PathBuf::from(required_value(&mut args, "--assistant-model-path")?)
                }
                "--python" => out.python = PathBuf::from(required_value(&mut args, "--python")?),
                "--source-replay" => {
                    out.source_replay_path =
                        PathBuf::from(required_value(&mut args, "--source-replay")?)
                }
                "--trials" => {
                    out.trials =
                        parse_positive_usize(&required_value(&mut args, "--trials")?, "--trials")?
                }
                "--warmups" | "--warmup-trials" => {
                    out.warmups =
                        parse_usize(&required_value(&mut args, "--warmups")?, "--warmups")?
                }
                "--max-new-tokens" => {
                    out.max_new_tokens = parse_positive_usize(
                        &required_value(&mut args, "--max-new-tokens")?,
                        "--max-new-tokens",
                    )?
                }
                "--block-sizes" => {
                    out.block_sizes = parse_usize_csv(&required_value(&mut args, "--block-sizes")?)?
                }
                "--workload-id" => {
                    out.workload_ids
                        .push(required_value(&mut args, "--workload-id")?);
                }
                "--clear-workload-ids" => out.workload_ids.clear(),
                "--max-workloads" => {
                    out.max_workloads = Some(parse_positive_usize(
                        &required_value(&mut args, "--max-workloads")?,
                        "--max-workloads",
                    )?)
                }
                "--min-speedup-percent" => {
                    out.min_speedup_percent = parse_finite_nonnegative(
                        &required_value(&mut args, "--min-speedup-percent")?,
                        "--min-speedup-percent",
                    )?
                }
                "--regression-gate-percent" => {
                    out.regression_gate_percent = parse_finite_nonnegative(
                        &required_value(&mut args, "--regression-gate-percent")?,
                        "--regression-gate-percent",
                    )?
                }
                "--memory-cliff-gb" => {
                    out.memory_cliff_gb = parse_finite_positive(
                        &required_value(&mut args, "--memory-cliff-gb")?,
                        "--memory-cliff-gb",
                    )?
                }
                "--experimental-terminal-no-lookahead" => {
                    out.experimental_terminal_no_lookahead = true
                }
                "--adaptive-zero-accept-run" => {
                    out.adaptive_zero_accept_run = Some(parse_positive_usize(
                        &required_value(&mut args, "--adaptive-zero-accept-run")?,
                        "--adaptive-zero-accept-run",
                    )?)
                }
                "--adaptive-min-generated-tokens" => {
                    out.adaptive_min_generated_tokens = parse_usize(
                        &required_value(&mut args, "--adaptive-min-generated-tokens")?,
                        "--adaptive-min-generated-tokens",
                    )?
                }
                "-h" | "--help" => return Err(CliError::Usage(usage())),
                other => {
                    return Err(CliError::Usage(format!(
                        "unknown option '{other}'\n{}",
                        usage()
                    )));
                }
            }
        }
        out.block_sizes.sort_unstable();
        out.block_sizes.dedup();
        if out.block_sizes.is_empty() {
            return Err(CliError::Usage(
                "--block-sizes must not be empty".to_owned(),
            ));
        }
        if out
            .block_sizes
            .iter()
            .any(|block_size| *block_size > MTP_MAX_DRAFT_TOKENS)
        {
            return Err(CliError::Usage(format!(
                "XR15 executable block sizes must be <= {MTP_MAX_DRAFT_TOKENS}"
            )));
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct EncodedWorkload {
    record: WorkloadRecord,
    prompt_sha256: String,
    token_ids: Vec<i32>,
    max_new_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: String,
    mode: String,
    status: String,
    decision: String,
    run_id: String,
    generated_at_unix_seconds: u64,
    command: String,
    git_sha: String,
    git_status_short: String,
    build_provenance: BuildProvenance,
    model_identity: manifest::ArtifactIdentity,
    assistant_identity: manifest::ArtifactIdentity,
    tokenizer_backend: String,
    source_replay_summary_path: String,
    source_replay_sha256: String,
    source_replay_run_id: String,
    source_replay_decision: String,
    source_policy_name: String,
    workloads_path: String,
    out_dir: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    generated_files: Vec<String>,
    selected_workloads: Vec<SelectedWorkload>,
    requested_trials: usize,
    warmup_trials: usize,
    max_new_tokens: usize,
    block_sizes: Vec<usize>,
    min_speedup_percent: f64,
    regression_gate_percent: f64,
    memory_cliff_gb: f64,
    experimental_terminal_no_lookahead: bool,
    adaptive_zero_accept_run: Option<usize>,
    adaptive_min_generated_tokens: usize,
    low_n: bool,
    record_count: usize,
    measured_record_count: usize,
    exact_record_count: usize,
    policy_summaries: Vec<PolicySummary>,
    blockers: Vec<String>,
    failed_hypotheses: Vec<String>,
    measurement_notes: Vec<String>,
    records: Vec<Record>,
}

#[derive(Debug, Clone, Serialize)]
struct SelectedWorkload {
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    selected_max_new_tokens: usize,
    deterministic_seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    schema_version: u32,
    goal: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    build_provenance: BuildProvenance,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    deterministic_seed: u64,
    max_new_tokens: usize,
    trial_index: usize,
    trial_kind: String,
    measured: bool,
    block_size: usize,
    baseline: GreedyRun,
    mtp: MtpRun,
    comparison: Comparison,
    status: String,
    blocker: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct GreedyRun {
    generated_tokens: Vec<i32>,
    model_load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    decode_token_latencies_ms: Vec<f64>,
    decode_p50_ms: f64,
    decode_p95_ms: f64,
    decode_p99_ms: f64,
    decode_tps: f64,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct MtpRun {
    generated_tokens: Vec<i32>,
    model_load_ms: f64,
    drafter_load_ms: f64,
    prefill_ms: f64,
    draft_ms: f64,
    verify_ms: f64,
    verify_stage_ms: f64,
    verify_forward_ms: f64,
    verify_repair_ms: f64,
    fallback_decode_ms: f64,
    total_ms: f64,
    decode_phase_ms: f64,
    attempted_draft_tokens: u64,
    accepted_draft_tokens: u64,
    acceptance_rate: f64,
    accepted_tokens_per_verify: f64,
    target_verify_passes: u64,
    rollback_count: u64,
    terminal_no_lookahead_count: u64,
    auto_disabled: bool,
    auto_disable_reason: Option<String>,
    auto_disable_pass: Option<u64>,
    auto_disable_generated_tokens: Option<usize>,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
    events: Vec<MtpEvent>,
}

#[derive(Debug, Clone, Serialize)]
struct MtpEvent {
    pass_index: u64,
    block_size: usize,
    draft_tokens: Vec<i32>,
    committed_tokens: Vec<i32>,
    accepted_draft_count: u32,
    rejected: bool,
    remaining_token_budget: usize,
    terminal_no_lookahead: bool,
    context_sequence_len: u64,
    sequence_len: u64,
    verify_ms: f64,
    verify_stage_ms: f64,
    verify_forward_ms: f64,
    verify_repair_ms: f64,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
    trace_position_count: u32,
    trace_top_k: u32,
    first_position: u64,
    target_tokens: Vec<i32>,
    draft_in_target_top_k: Vec<bool>,
}

#[derive(Debug, Clone, Serialize)]
struct Comparison {
    byte_identical: bool,
    first_mismatch: Option<TokenMismatch>,
}

#[derive(Debug, Clone, Serialize)]
struct TokenMismatch {
    index: usize,
    baseline_token: Option<i32>,
    mtp_token: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicySummary {
    policy_name: String,
    decision: String,
    selected_mtp_workloads: usize,
    workload_count: usize,
    exact_workloads: usize,
    regressed_workloads: usize,
    total_baseline_decode_ms: f64,
    total_selected_decode_phase_ms: f64,
    total_delta_ms: f64,
    aggregate_speedup_percent: f64,
    max_peak_memory_gb: f64,
    total_accepted_draft_tokens: u64,
    total_attempted_draft_tokens: u64,
    weighted_acceptance_rate: f64,
    selected_workloads: Vec<String>,
    regressed_workload_ids: Vec<String>,
    low_n: bool,
    reasons: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SourceReplaySummary {
    run_id: String,
    decision: String,
}

fn run_mtp_record(
    args: &Args,
    run_id: &str,
    build_provenance: &BuildProvenance,
    workload: &EncodedWorkload,
    trial_index: usize,
    trial_kind: &str,
    block_size: usize,
    baseline: GreedyRun,
) -> Result<Record, Box<dyn std::error::Error>> {
    let mtp = run_mtp(args, workload, block_size)?;
    let comparison = compare_tokens(&baseline.generated_tokens, &mtp.generated_tokens);
    let blocker = if comparison.byte_identical {
        None
    } else {
        Some(format!(
            "{} block_size={} trial={} MTP output differed from native baseline",
            workload.record.workload_id, block_size, trial_index
        ))
    };
    let status = if blocker.is_some() {
        "failed"
    } else if mtp.peak_memory_gb > args.memory_cliff_gb as f32 {
        "failed_memory"
    } else {
        "passed"
    };

    Ok(Record {
        schema_version: 1,
        goal: GOAL.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: build_provenance.git_sha.clone(),
        git_status_short: build_provenance.git_status_short.clone(),
        build_provenance: build_provenance.clone(),
        workload_id: workload.record.workload_id.clone(),
        family: workload.record.family.clone(),
        prompt_path: workload.record.prompt_path.clone(),
        prompt_sha256: workload.prompt_sha256.clone(),
        target_context_tokens: workload.record.target_context_tokens,
        actual_context_tokens: workload.token_ids.len(),
        deterministic_seed: workload.record.deterministic_seed,
        max_new_tokens: workload.max_new_tokens,
        trial_index,
        trial_kind: trial_kind.to_owned(),
        measured: trial_kind == "measured",
        block_size,
        baseline,
        mtp,
        comparison,
        status: status.to_owned(),
        blocker,
    })
}

fn run_baseline(
    args: &Args,
    workload: &EncodedWorkload,
) -> Result<GreedyRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let load_started = Instant::now();
    let target = Target::load(&target_config(args, workload))?;
    let model_load = load_started.elapsed();
    let mut cache = KvCache::create(&KvPolicy::default())?;

    let prefill_started = Instant::now();
    let mut step = prefill(&target, &mut cache, &workload.token_ids)?;
    let prefill = prefill_started.elapsed();
    let mut decode_duration = Duration::ZERO;
    let mut decode_token_latencies_ms = Vec::new();
    let mut generated = Vec::with_capacity(workload.max_new_tokens);
    let mut peak_memory_gb = step.peak_memory_gb;
    let mut active_kv_bytes = step.active_kv_bytes;

    for index in 0..workload.max_new_tokens {
        generated.push(step.greedy_token);
        if index + 1 < workload.max_new_tokens {
            let decode_started = Instant::now();
            step = decode_one(&target, &mut cache, step.greedy_token)?;
            let elapsed = decode_started.elapsed();
            decode_duration += elapsed;
            decode_token_latencies_ms.push(duration_ms(elapsed));
            peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
            active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
        }
    }

    let decode_ms = duration_ms(decode_duration);
    Ok(GreedyRun {
        generated_tokens: generated,
        model_load_ms: duration_ms(model_load),
        prefill_ms: duration_ms(prefill),
        decode_ms,
        total_ms: duration_ms(started.elapsed()),
        decode_p50_ms: percentile(decode_token_latencies_ms.clone(), 0.50),
        decode_p95_ms: percentile(decode_token_latencies_ms.clone(), 0.95),
        decode_p99_ms: percentile(decode_token_latencies_ms.clone(), 0.99),
        decode_token_latencies_ms,
        decode_tps: if decode_ms > 0.0 {
            workload.max_new_tokens as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        },
        peak_memory_gb,
        active_kv_bytes,
    })
}

fn run_mtp(
    args: &Args,
    workload: &EncodedWorkload,
    block_size: usize,
) -> Result<MtpRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let load_started = Instant::now();
    let target = Target::load(&target_config(args, workload))?;
    let model_load = load_started.elapsed();
    let drafter_started = Instant::now();
    let drafter = Drafter::load(&assistant_config(args, workload), &target)?;
    let drafter_load = drafter_started.elapsed();
    let mut cache = KvCache::create(&KvPolicy::default())?;

    let prefill_started = Instant::now();
    let first = prefill(&target, &mut cache, &workload.token_ids)?;
    let prefill = prefill_started.elapsed();
    let mut generated = Vec::with_capacity(workload.max_new_tokens);
    let mut draft_duration = Duration::ZERO;
    let mut verify_duration = Duration::ZERO;
    let mut verify_stage_ms = 0.0_f64;
    let mut verify_forward_ms = 0.0_f64;
    let mut verify_repair_ms = 0.0_f64;
    let mut fallback_decode_duration = Duration::ZERO;
    let mut attempted_draft_tokens = 0_u64;
    let mut accepted_draft_tokens = 0_u64;
    let mut target_verify_passes = 0_u64;
    let mut rollback_count = 0_u64;
    let mut terminal_no_lookahead_count = 0_u64;
    let mut consecutive_zero_accepts = 0_usize;
    let mut auto_disabled = false;
    let mut auto_disable_reason = None;
    let mut auto_disable_pass = None;
    let mut auto_disable_generated_tokens = None;
    let mut pending_greedy = None;
    let mut peak_memory_gb = first.peak_memory_gb;
    let mut active_kv_bytes = first.active_kv_bytes;
    let mut events = Vec::new();

    while generated.len() < workload.max_new_tokens {
        if auto_disabled {
            if let Some(token) = pending_greedy.take() {
                generated.push(token);
                continue;
            }
            let token = *generated
                .last()
                .ok_or("auto-disabled MTP has no committed token")?;
            let decode_started = Instant::now();
            let step = decode_one(&target, &mut cache, token)?;
            fallback_decode_duration += decode_started.elapsed();
            peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
            active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
            generated.push(step.greedy_token);
            continue;
        }

        let remaining = workload.max_new_tokens - generated.len();
        let current_block_size = block_size.min(remaining).max(1);
        let draft_started = Instant::now();
        let draft = draft_block(
            &drafter,
            &mut cache,
            NonZeroU32::new(current_block_size as u32).expect("block size is non-zero"),
        )?;
        draft_duration += draft_started.elapsed();
        if draft.is_empty() {
            return Err("native MTP drafter returned no tokens".into());
        }

        attempted_draft_tokens += draft.len() as u64;
        target_verify_passes += 1;
        let terminal_no_lookahead =
            args.experimental_terminal_no_lookahead && draft.len() == remaining;
        let verify_started = Instant::now();
        let step = if terminal_no_lookahead {
            verify_tokens_terminal_no_lookahead(&target, &mut cache, &draft, remaining)?
        } else {
            verify_tokens(&target, &mut cache, &draft)?
        };
        let verify_elapsed = verify_started.elapsed();
        verify_duration += verify_elapsed;
        verify_stage_ms += step.verify_stage_ms;
        verify_forward_ms += step.verify_forward_ms;
        verify_repair_ms += step.verify_repair_ms;
        peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
        active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
        let committed = step.committed_tokens().to_vec();
        if committed.is_empty() {
            return Err("native MTP verifier committed no tokens".into());
        }
        let accepted = u64::from(step.accepted_draft_count);
        accepted_draft_tokens += accepted;
        if accepted == 0 {
            consecutive_zero_accepts += 1;
        } else {
            consecutive_zero_accepts = 0;
        }
        let rejected =
            usize::try_from(step.accepted_draft_count).unwrap_or(usize::MAX) < draft.len();
        if rejected {
            rollback_count += 1;
        }
        let terminal_skip_applied = terminal_no_lookahead
            && committed.len() >= remaining
            && step.native_last_hidden.is_none();
        if terminal_skip_applied {
            terminal_no_lookahead_count += 1;
        }
        let accepted_usize = usize::try_from(step.accepted_draft_count).unwrap_or(usize::MAX);
        let expected_trace_positions = if terminal_skip_applied {
            accepted_usize.min(draft.len())
        } else if accepted_usize >= draft.len() {
            draft.len() + 1
        } else {
            accepted_usize + 2
        };
        if usize::try_from(step.mtp_trace.position_count).unwrap_or(0) < expected_trace_positions {
            return Err(format!(
                "native MTP trace incomplete: block_size={} draft_len={} accepted={} expected_at_least={} got={}",
                block_size,
                draft.len(),
                step.accepted_draft_count,
                expected_trace_positions,
                step.mtp_trace.position_count
            )
            .into());
        }
        for token in &committed {
            if generated.len() < workload.max_new_tokens {
                generated.push(*token);
            }
        }
        events.push(MtpEvent {
            pass_index: target_verify_passes,
            block_size,
            draft_tokens: draft.clone(),
            committed_tokens: committed,
            accepted_draft_count: step.accepted_draft_count,
            rejected,
            remaining_token_budget: remaining,
            terminal_no_lookahead: terminal_skip_applied,
            context_sequence_len: step.mtp_trace.context_sequence_len,
            sequence_len: step.sequence_len,
            verify_ms: duration_ms(verify_elapsed),
            verify_stage_ms: step.verify_stage_ms,
            verify_forward_ms: step.verify_forward_ms,
            verify_repair_ms: step.verify_repair_ms,
            peak_memory_gb: step.peak_memory_gb,
            active_kv_bytes: step.active_kv_bytes,
            trace_position_count: step.mtp_trace.position_count,
            trace_top_k: step.mtp_trace.top_k,
            first_position: step.mtp_trace.first_position,
            target_tokens: step.mtp_trace.target_tokens.clone(),
            draft_in_target_top_k: step.mtp_trace.draft_in_top_k.clone(),
        });

        if let Some(zero_accept_run) = args.adaptive_zero_accept_run {
            if consecutive_zero_accepts >= zero_accept_run
                && generated.len() >= args.adaptive_min_generated_tokens
            {
                auto_disabled = true;
                auto_disable_pass = Some(target_verify_passes);
                auto_disable_generated_tokens = Some(generated.len());
                auto_disable_reason = Some(format!(
                    "consecutive zero-accept passes {} reached threshold {} after {} generated tokens",
                    consecutive_zero_accepts,
                    zero_accept_run,
                    generated.len()
                ));
                pending_greedy = Some(step.greedy_token);
            }
        }
    }

    let draft_ms = duration_ms(draft_duration);
    let verify_ms = duration_ms(verify_duration);
    let fallback_decode_ms = duration_ms(fallback_decode_duration);
    Ok(MtpRun {
        generated_tokens: generated,
        model_load_ms: duration_ms(model_load),
        drafter_load_ms: duration_ms(drafter_load),
        prefill_ms: duration_ms(prefill),
        draft_ms,
        verify_ms,
        verify_stage_ms,
        verify_forward_ms,
        verify_repair_ms,
        fallback_decode_ms,
        total_ms: duration_ms(started.elapsed()),
        decode_phase_ms: draft_ms + verify_ms + fallback_decode_ms,
        attempted_draft_tokens,
        accepted_draft_tokens,
        acceptance_rate: ratio(accepted_draft_tokens, attempted_draft_tokens),
        accepted_tokens_per_verify: ratio(accepted_draft_tokens, target_verify_passes),
        target_verify_passes,
        rollback_count,
        terminal_no_lookahead_count,
        auto_disabled,
        auto_disable_reason,
        auto_disable_pass,
        auto_disable_generated_tokens,
        peak_memory_gb,
        active_kv_bytes,
        events,
    })
}

fn target_config(args: &Args, workload: &EncodedWorkload) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(workload.token_ids.len().max(1) as u32)
            .expect("context length is non-zero"),
        allow_unsupported_config: false,
    }
}

fn assistant_config(args: &Args, workload: &EncodedWorkload) -> LoadConfig {
    LoadConfig {
        model_path: args.assistant_model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4_mtp_assistant".to_owned()),
        max_context_tokens: NonZeroU32::new(workload.token_ids.len().max(1) as u32)
            .expect("context length is non-zero"),
        allow_unsupported_config: false,
    }
}

fn policy_summaries(args: &Args, records: &[Record]) -> Vec<PolicySummary> {
    let measured = records
        .iter()
        .filter(|record| record.measured)
        .collect::<Vec<_>>();
    if measured.is_empty() {
        return Vec::new();
    }
    let mut summaries = Vec::new();
    summaries.push(policy_summary_for(args, &measured, "disabled_baseline"));
    for block_size in &args.block_sizes {
        summaries.push(policy_summary_for(
            args,
            &measured,
            &format!("fixed_block_{block_size}"),
        ));
    }
    summaries.push(policy_summary_for(
        args,
        &measured,
        "acceptance_threshold_35pct",
    ));
    summaries.push(policy_summary_for(
        args,
        &measured,
        "net_latency_guarded_5pct",
    ));
    summaries
}

fn policy_summary_for(args: &Args, records: &[&Record], policy_name: &str) -> PolicySummary {
    let workload_ids = records
        .iter()
        .map(|record| record.workload_id.clone())
        .collect::<BTreeSet<_>>();
    let mut total_baseline_decode_ms = 0.0;
    let mut total_selected_decode_phase_ms = 0.0;
    let mut selected_workloads = Vec::new();
    let mut regressed_workload_ids = Vec::new();
    let mut exact_workloads = 0_usize;
    let mut max_peak_memory_gb = 0.0_f64;
    let mut total_accepted_draft_tokens = 0_u64;
    let mut total_attempted_draft_tokens = 0_u64;

    for workload_id in &workload_ids {
        let workload_records = records
            .iter()
            .copied()
            .filter(|record| &record.workload_id == workload_id)
            .collect::<Vec<_>>();
        let baseline = baseline_candidate(&workload_records);
        let selected = select_policy_candidate(args, policy_name, &workload_records);
        total_baseline_decode_ms += baseline.decode_phase_ms;
        total_selected_decode_phase_ms += selected.decode_phase_ms;
        max_peak_memory_gb = max_peak_memory_gb.max(selected.peak_memory_gb);
        if selected.exact {
            exact_workloads += 1;
        }
        if selected.mtp_enabled {
            selected_workloads.push(format!(
                "{}:{}",
                workload_id,
                selected.block_size.unwrap_or_default()
            ));
            total_accepted_draft_tokens += selected.accepted_draft_tokens;
            total_attempted_draft_tokens += selected.attempted_draft_tokens;
        }
        if selected.decode_phase_ms
            > baseline.decode_phase_ms * (1.0 + args.regression_gate_percent / 100.0)
        {
            regressed_workload_ids.push(workload_id.clone());
        }
    }

    let total_delta_ms = total_selected_decode_phase_ms - total_baseline_decode_ms;
    let aggregate_speedup_percent =
        speedup_percent(total_baseline_decode_ms, total_selected_decode_phase_ms);
    let selected_mtp_workloads = selected_workloads.len();
    let regressed_workloads = regressed_workload_ids.len();
    let low_n = args.trials < 3;
    let decision = policy_decision(
        policy_name,
        aggregate_speedup_percent,
        selected_mtp_workloads,
        regressed_workloads,
        low_n,
        max_peak_memory_gb,
        args.memory_cliff_gb,
    );
    let reasons = policy_reasons(
        policy_name,
        aggregate_speedup_percent,
        selected_mtp_workloads,
        regressed_workloads,
        low_n,
        max_peak_memory_gb,
        args.memory_cliff_gb,
    );

    PolicySummary {
        policy_name: policy_name.to_owned(),
        decision,
        selected_mtp_workloads,
        workload_count: workload_ids.len(),
        exact_workloads,
        regressed_workloads,
        total_baseline_decode_ms,
        total_selected_decode_phase_ms,
        total_delta_ms,
        aggregate_speedup_percent,
        max_peak_memory_gb,
        total_accepted_draft_tokens,
        total_attempted_draft_tokens,
        weighted_acceptance_rate: ratio(total_accepted_draft_tokens, total_attempted_draft_tokens),
        selected_workloads,
        regressed_workload_ids,
        low_n,
        reasons,
    }
}

#[derive(Debug, Clone)]
struct SelectedPolicyCandidate {
    block_size: Option<usize>,
    mtp_enabled: bool,
    decode_phase_ms: f64,
    exact: bool,
    peak_memory_gb: f64,
    accepted_draft_tokens: u64,
    attempted_draft_tokens: u64,
}

fn select_policy_candidate(
    args: &Args,
    policy_name: &str,
    records: &[&Record],
) -> SelectedPolicyCandidate {
    let baseline = baseline_candidate(records);
    if policy_name == "disabled_baseline" {
        return baseline;
    }
    if let Some(block) = policy_name.strip_prefix("fixed_block_") {
        if let Ok(block_size) = block.parse::<usize>() {
            return block_candidate(records, block_size).unwrap_or(baseline);
        }
    }
    if policy_name.starts_with("acceptance_threshold_") {
        return records
            .iter()
            .map(|record| record.block_size)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|block_size| block_candidate(records, block_size))
            .filter(|candidate| candidate.exact)
            .filter(|candidate| {
                ratio(
                    candidate.accepted_draft_tokens,
                    candidate.attempted_draft_tokens,
                ) >= 0.35
            })
            .max_by(|left, right| {
                ratio(left.accepted_draft_tokens, left.attempted_draft_tokens)
                    .partial_cmp(&ratio(
                        right.accepted_draft_tokens,
                        right.attempted_draft_tokens,
                    ))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(baseline);
    }
    if policy_name.starts_with("net_latency_guarded_") {
        return records
            .iter()
            .map(|record| record.block_size)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|block_size| block_candidate(records, block_size))
            .filter(|candidate| candidate.exact)
            .filter(|candidate| candidate.peak_memory_gb <= args.memory_cliff_gb)
            .filter(|candidate| {
                speedup_percent(baseline.decode_phase_ms, candidate.decode_phase_ms)
                    >= args.min_speedup_percent
            })
            .min_by(|left, right| {
                left.decode_phase_ms
                    .partial_cmp(&right.decode_phase_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(baseline);
    }
    baseline
}

fn baseline_candidate(records: &[&Record]) -> SelectedPolicyCandidate {
    SelectedPolicyCandidate {
        block_size: None,
        mtp_enabled: false,
        decode_phase_ms: median(
            records
                .iter()
                .map(|record| record.baseline.decode_ms)
                .collect(),
        ),
        exact: true,
        peak_memory_gb: records
            .iter()
            .map(|record| f64::from(record.baseline.peak_memory_gb))
            .fold(0.0, f64::max),
        accepted_draft_tokens: 0,
        attempted_draft_tokens: 0,
    }
}

fn block_candidate(records: &[&Record], block_size: usize) -> Option<SelectedPolicyCandidate> {
    let block_records = records
        .iter()
        .copied()
        .filter(|record| record.block_size == block_size)
        .collect::<Vec<_>>();
    if block_records.is_empty() {
        return None;
    }
    Some(SelectedPolicyCandidate {
        block_size: Some(block_size),
        mtp_enabled: true,
        decode_phase_ms: median(
            block_records
                .iter()
                .map(|record| record.mtp.decode_phase_ms)
                .collect(),
        ),
        exact: block_records
            .iter()
            .all(|record| record.comparison.byte_identical),
        peak_memory_gb: block_records
            .iter()
            .map(|record| f64::from(record.mtp.peak_memory_gb))
            .fold(0.0, f64::max),
        accepted_draft_tokens: block_records
            .iter()
            .map(|record| record.mtp.accepted_draft_tokens)
            .sum(),
        attempted_draft_tokens: block_records
            .iter()
            .map(|record| record.mtp.attempted_draft_tokens)
            .sum(),
    })
}

#[allow(clippy::too_many_arguments)]
fn policy_decision(
    policy_name: &str,
    aggregate_speedup_percent: f64,
    selected_mtp_workloads: usize,
    regressed_workloads: usize,
    low_n: bool,
    max_peak_memory_gb: f64,
    memory_cliff_gb: f64,
) -> String {
    if policy_name == "disabled_baseline" {
        "baseline".to_owned()
    } else if regressed_workloads > 0
        || selected_mtp_workloads == 0
        || aggregate_speedup_percent <= 0.0
        || max_peak_memory_gb > memory_cliff_gb
    {
        "reject_candidate".to_owned()
    } else if low_n {
        "needs_more_data".to_owned()
    } else {
        "keep_experimental".to_owned()
    }
}

#[allow(clippy::too_many_arguments)]
fn policy_reasons(
    policy_name: &str,
    aggregate_speedup_percent: f64,
    selected_mtp_workloads: usize,
    regressed_workloads: usize,
    low_n: bool,
    max_peak_memory_gb: f64,
    memory_cliff_gb: f64,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if policy_name == "disabled_baseline" {
        reasons.push("baseline reference, not a candidate".to_owned());
    }
    if selected_mtp_workloads == 0 && policy_name != "disabled_baseline" {
        reasons.push("policy selected no MTP workloads".to_owned());
    }
    if regressed_workloads > 0 {
        reasons.push(format!("regressed {regressed_workloads} workload(s)"));
    }
    if aggregate_speedup_percent != 0.0 && policy_name != "disabled_baseline" {
        reasons.push(format!(
            "aggregate measured-trial speedup was {:.3}%",
            aggregate_speedup_percent
        ));
    }
    if low_n && policy_name != "disabled_baseline" {
        reasons.push("fewer than 3 measured trials; policy claim is low-N".to_owned());
    }
    if max_peak_memory_gb > memory_cliff_gb {
        reasons.push(format!(
            "peak MLX memory {:.3} GB exceeded {:.3} GB gate",
            max_peak_memory_gb, memory_cliff_gb
        ));
    }
    reasons
}

fn decision_for(
    args: &Args,
    blockers: &[String],
    records: &[Record],
    policy_summaries: &[PolicySummary],
) -> String {
    if !blockers.is_empty() || records.is_empty() {
        "blocked_with_evidence".to_owned()
    } else if records
        .iter()
        .any(|record| !record.comparison.byte_identical)
    {
        "blocked_with_evidence".to_owned()
    } else if args.trials < 3 {
        "needs_more_data".to_owned()
    } else {
        policy_summaries
            .iter()
            .find(|summary| summary.policy_name == "net_latency_guarded_5pct")
            .map(|summary| summary.decision.clone())
            .unwrap_or_else(|| "needs_more_data".to_owned())
    }
}

fn record_blockers(records: &[Record]) -> Vec<String> {
    records
        .iter()
        .filter_map(|record| record.blocker.clone())
        .collect()
}

fn failed_hypotheses(policy_summaries: &[PolicySummary]) -> Vec<String> {
    policy_summaries
        .iter()
        .filter(|summary| summary.decision == "reject_candidate")
        .map(|summary| {
            format!(
                "{} rejected: {}",
                summary.policy_name,
                summary.reasons.join("; ")
            )
        })
        .collect()
}

fn compare_tokens(baseline: &[i32], mtp: &[i32]) -> Comparison {
    if baseline == mtp {
        return Comparison {
            byte_identical: true,
            first_mismatch: None,
        };
    }
    let max_len = baseline.len().max(mtp.len());
    let first_mismatch = (0..max_len)
        .find(|index| baseline.get(*index) != mtp.get(*index))
        .map(|index| TokenMismatch {
            index,
            baseline_token: baseline.get(index).copied(),
            mtp_token: mtp.get(index).copied(),
        });
    Comparison {
        byte_identical: false,
        first_mismatch,
    }
}

fn load_workloads(path: &Path) -> Result<Vec<WorkloadRecord>, CliError> {
    let text = fs::read_to_string(path)
        .map_err(|error| CliError::Runtime(format!("failed to read workloads JSONL: {error}")))?;
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str::<WorkloadRecord>(line).map_err(|error| {
                CliError::Runtime(format!(
                    "failed to parse workload line {} in {}: {error}",
                    index + 1,
                    path.display()
                ))
            })?,
        );
    }
    if out.is_empty() {
        return Err(CliError::Runtime(format!(
            "workload manifest is empty: {}",
            path.display()
        )));
    }
    Ok(out)
}

fn select_workloads(
    mut workloads: Vec<WorkloadRecord>,
    args: &Args,
) -> Result<Vec<WorkloadRecord>, CliError> {
    if !args.workload_ids.is_empty() {
        let wanted = args.workload_ids.iter().cloned().collect::<BTreeSet<_>>();
        workloads.retain(|workload| wanted.contains(&workload.workload_id));
        let found = workloads
            .iter()
            .map(|workload| workload.workload_id.clone())
            .collect::<BTreeSet<_>>();
        for id in wanted {
            if !found.contains(&id) {
                return Err(CliError::Runtime(format!(
                    "requested workload id not found: {id}"
                )));
            }
        }
    }
    if let Some(max_workloads) = args.max_workloads {
        workloads.truncate(max_workloads);
    }
    if workloads.is_empty() {
        return Err(CliError::Runtime("no workloads selected".to_owned()));
    }
    Ok(workloads)
}

fn encode_workload(
    args: &Args,
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
    let token_ids = tokenizer.encode(&prompt)?;
    if prompt_sha256 != record.prompt_sha256 {
        return Err(CliError::Runtime(format!(
            "{} prompt sha mismatch: manifest={} actual={}",
            record.workload_id, record.prompt_sha256, prompt_sha256
        )));
    }
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
        max_new_tokens: record.max_new_tokens.min(args.max_new_tokens),
    })
}

fn selected_workload_rows(args: &Args, workloads: &[WorkloadRecord]) -> Vec<SelectedWorkload> {
    workloads
        .iter()
        .map(|workload| SelectedWorkload {
            workload_id: workload.workload_id.clone(),
            family: workload.family.clone(),
            prompt_path: workload.prompt_path.clone(),
            prompt_sha256: workload.prompt_sha256.clone(),
            target_context_tokens: workload.target_context_tokens,
            actual_context_tokens: workload.actual_context_tokens,
            selected_max_new_tokens: workload.max_new_tokens.min(args.max_new_tokens),
            deterministic_seed: workload.deterministic_seed,
        })
        .collect()
}

fn startup_blockers(args: &Args, source_replay: &Option<SourceReplaySummary>) -> Vec<String> {
    let mut blockers = Vec::new();
    if !args.model_path.exists() {
        blockers.push(format!(
            "target model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if !args.assistant_model_path.exists() {
        blockers.push(format!(
            "assistant model path does not exist: {}",
            args.assistant_model_path.display()
        ));
    }
    if !args.python.exists() {
        blockers.push(format!(
            "python path does not exist: {}",
            args.python.display()
        ));
    }
    if !args.source_replay_path.exists() {
        blockers.push(format!(
            "source replay summary does not exist: {}",
            args.source_replay_path.display()
        ));
    }
    if source_replay.is_none() {
        blockers.push("source replay summary could not be parsed".to_owned());
    }
    if env::var_os("GEMMA4D_USE_NATIVE_GRAPH").is_none() {
        blockers.push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for XR15".to_owned());
    }
    if env::var_os("GEMMA4D_REQUIRE_MLX").is_none() {
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for XR15".to_owned());
    }
    if args.experimental_terminal_no_lookahead {
        for flag in [
            "GEMMA4D_EXPERIMENTAL_MTP_BATCH_VERIFY",
            "GEMMA4D_EXPERIMENTAL_MTP_SKIP_FINAL_PROJECTION",
            "GEMMA4D_EXPERIMENTAL_MTP_INPLACE_VERIFY",
        ] {
            if env::var_os(flag).is_some() {
                blockers.push(format!(
                    "--experimental-terminal-no-lookahead cannot be combined with {flag}"
                ));
            }
        }
    }
    blockers
}

fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<(), CliError> {
    let mut file = File::create(path)
        .map_err(|error| CliError::Runtime(format!("failed to create records.jsonl: {error}")))?;
    for record in records {
        serde_json::to_writer(&mut file, record)
            .map_err(|error| CliError::Runtime(format!("failed to serialize record: {error}")))?;
        writeln!(file)
            .map_err(|error| CliError::Runtime(format!("failed to write record: {error}")))?;
    }
    Ok(())
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR15 MTP Policy Variance A/B\n\n");
    out.push_str("## Summary\n\n| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Decision | `{}` |\n", summary.decision));
    out.push_str(&format!("| Status | `{}` |\n", summary.status));
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!(
        "| Dirty diff SHA-256 | `{}` |\n",
        summary.build_provenance.dirty_diff_sha256
    ));
    out.push_str(&format!(
        "| Runner binary link mtime | `{}` |\n",
        summary
            .build_provenance
            .runner_binary_link_mtime_unix_seconds
    ));
    out.push_str(&format!(
        "| Runner binary | `{}` |\n",
        summary.build_provenance.runner_binary_path
    ));
    out.push_str(&format!(
        "| Source replay | `{}` (`{}`) |\n",
        summary.source_replay_run_id, summary.source_replay_decision
    ));
    out.push_str(&format!("| Records | `{}` |\n", summary.record_count));
    out.push_str(&format!(
        "| Measured records | `{}` |\n",
        summary.measured_record_count
    ));
    out.push_str(&format!(
        "| Exact records | `{}` |\n",
        summary.exact_record_count
    ));
    out.push_str(&format!(
        "| Trials | `{}` measured, `{}` warmup |\n",
        summary.requested_trials, summary.warmup_trials
    ));
    out.push_str(&format!(
        "| Terminal no-lookahead | `{}` |\n",
        summary.experimental_terminal_no_lookahead
    ));
    out.push_str(&format!(
        "| Adaptive zero-accept run | `{}` |\n",
        summary
            .adaptive_zero_accept_run
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_owned())
    ));
    out.push_str(&format!(
        "| Adaptive min generated tokens | `{}` |\n",
        summary.adaptive_min_generated_tokens
    ));
    out.push_str(&format!("| Low-N | `{}` |\n\n", summary.low_n));

    out.push_str("## Policy Results\n\n");
    out.push_str("| Policy | Decision | MTP selections | Baseline decode ms | Selected decode ms | Speedup % | Regressions | Weighted acceptance | Peak GB |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for policy in &summary.policy_summaries {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | {:.3} | {:.3} | {:.3} | {} | {:.3} | {:.3} |\n",
            policy.policy_name,
            policy.decision,
            policy.selected_mtp_workloads,
            policy.total_baseline_decode_ms,
            policy.total_selected_decode_phase_ms,
            policy.aggregate_speedup_percent,
            policy.regressed_workloads,
            policy.weighted_acceptance_rate,
            policy.max_peak_memory_gb
        ));
    }
    out.push('\n');

    out.push_str("## Records\n\n");
    out.push_str("| Workload | Trial | Block | Exact | Baseline decode ms | MTP decode phase ms | Fallback decode ms | Speedup % | Accepted/Attempted | Terminal skips | Auto disabled | Peak GB | Status |\n");
    out.push_str("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | `{}` {} | {} | `{}` | {:.3} | {:.3} | {:.3} | {:.3} | {}/{} | {} | `{}` | {:.3} | `{}` |\n",
            record.workload_id,
            record.trial_kind,
            record.trial_index,
            record.block_size,
            record.comparison.byte_identical,
            record.baseline.decode_ms,
            record.mtp.decode_phase_ms,
            record.mtp.fallback_decode_ms,
            speedup_percent(record.baseline.decode_ms, record.mtp.decode_phase_ms),
            record.mtp.accepted_draft_tokens,
            record.mtp.attempted_draft_tokens,
            record.mtp.terminal_no_lookahead_count,
            record.mtp.auto_disabled,
            record.mtp.peak_memory_gb,
            record.status
        ));
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR15 Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No hard blockers recorded.\n\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
        out.push('\n');
    }
    if !summary.failed_hypotheses.is_empty() {
        out.push_str("## Failed Hypotheses\n\n");
        for hypothesis in &summary.failed_hypotheses {
            out.push_str(&format!("- {hypothesis}\n"));
        }
    }
    out
}

fn render_decision(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR15 Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("MTP remains opt-in. This goal validates a policy candidate with fresh measured trials; it does not change runtime defaults.\n\n");
    if let Some(policy) = summary
        .policy_summaries
        .iter()
        .find(|policy| policy.policy_name == "net_latency_guarded_5pct")
    {
        out.push_str(&format!(
            "The net-latency-guarded policy decision is `{}` with {:.3}% aggregate measured speedup over baseline decode phase.\n\n",
            policy.decision, policy.aggregate_speedup_percent
        ));
        for reason in &policy.reasons {
            out.push_str(&format!("- {reason}\n"));
        }
    }
    out
}

fn measurement_notes() -> Vec<String> {
    vec![
        "baseline is native non-MTP greedy decode with GEMMA4D_REQUIRE_MLX=1 and GEMMA4D_USE_NATIVE_GRAPH=1".to_owned(),
        "candidate MTP decode phase is draft_ms + verify_ms + fallback_decode_ms; model load and prefill are recorded but excluded from policy speed decisions".to_owned(),
        "terminal no-lookahead mode only calls the experimental verifier on a final draft block whose returned draft count can satisfy the remaining generation budget".to_owned(),
        "adaptive zero-accept fallback is disabled unless --adaptive-zero-accept-run is passed; when active it uses native decode_one for the remaining tail after the gate fires".to_owned(),
        "warmup records remain in records.jsonl and summary.json but policy summaries use measured records only".to_owned(),
        "each evidence summary and record stamps git SHA, dirty-diff SHA-256, dirty-diff byte count, runner binary path, and runner binary link mtime; missing provenance aborts before measurement".to_owned(),
        "the net-latency guard requires exact MTP output, at least 5% decode-phase speedup, and peak MLX memory under the configured gate".to_owned(),
    ]
}

fn load_source_replay(path: &Path) -> Option<SourceReplaySummary> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn parse_usize_csv(value: &str) -> Result<Vec<usize>, CliError> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_positive_usize(part.trim(), "--block-sizes"))
        .collect()
}

fn parse_usize(value: &str, option: &str) -> Result<usize, CliError> {
    value
        .parse::<usize>()
        .map_err(|error| CliError::Usage(format!("{option} must be an integer: {error}")))
}

fn parse_positive_usize(value: &str, option: &str) -> Result<usize, CliError> {
    let parsed = parse_usize(value, option)?;
    if parsed == 0 {
        return Err(CliError::Usage(format!(
            "{option} must be greater than zero"
        )));
    }
    Ok(parsed)
}

fn parse_finite_positive(value: &str, option: &str) -> Result<f64, CliError> {
    let parsed = parse_finite_nonnegative(value, option)?;
    if parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(CliError::Usage(format!(
            "{option} must be greater than zero"
        )))
    }
}

fn parse_finite_nonnegative(value: &str, option: &str) -> Result<f64, CliError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| CliError::Usage(format!("{option} must be a number: {error}")))?;
    if parsed.is_finite() && parsed >= 0.0 {
        Ok(parsed)
    } else {
        Err(CliError::Usage(format!(
            "{option} must be finite and nonnegative"
        )))
    }
}

fn required_value<I>(args: &mut std::iter::Peekable<I>, option: &str) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| CliError::Usage(format!("{option} requires a value")))
}

fn usage() -> String {
    format!(
        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr15_mtp_policy_variance_ab -- [--out-dir PATH] [--source-replay PATH] [--trials N] [--warmups N] [--max-new-tokens N] [--block-sizes 1,2] [--experimental-terminal-no-lookahead] [--adaptive-zero-accept-run N] [--adaptive-min-generated-tokens N] [--workload-id ID] [--clear-workload-ids] [--max-workloads N]\n\ndefault out-dir: {DEFAULT_OUT_DIR}"
    )
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

fn ratio(left: u64, right: u64) -> f64 {
    if right == 0 {
        0.0
    } else {
        left as f64 / right as f64
    }
}

fn speedup_percent(baseline_ms: f64, candidate_ms: f64) -> f64 {
    if baseline_ms <= 0.0 {
        0.0
    } else {
        (baseline_ms - candidate_ms) / baseline_ms * 100.0
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn command_line() -> String {
    env::args().collect::<Vec<_>>().join(" ")
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn run_id() -> String {
    format!("xr15-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn file_sha256(path: &Path) -> String {
    command_stdout("shasum", &["-a", "256", &path.display().to_string()])
        .and_then(|line| line.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "unavailable".to_owned())
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

try:
    tokenizer = load_tokenizer(Path(sys.argv[1]))
    print(json.dumps({"ok": True, "backend": "mlx_lm.utils.load_tokenizer", "tokenizer_class": type(tokenizer).__name__}, separators=(",", ":")), flush=True)
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")), flush=True)
    raise SystemExit(1)

for line in sys.stdin:
    request = json.loads(line)
    cmd = request.get("cmd")
    if cmd == "encode":
        print(json.dumps({"ok": True, "ids": tokenizer.encode(request["text"])}, separators=(",", ":")), flush=True)
    elif cmd == "shutdown":
        break
    else:
        print(json.dumps({"ok": False, "error": f"unknown cmd {cmd}"}), flush=True)
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
                CliError::Runtime(format!("failed to start tokenizer helper: {error}"))
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| CliError::Runtime("tokenizer helper stdin unavailable".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CliError::Runtime("tokenizer helper stdout unavailable".to_owned()))?;
        let mut helper = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            backend: "unknown".to_owned(),
        };
        let mut line = String::new();
        helper
            .stdout
            .read_line(&mut line)
            .map_err(|error| CliError::Runtime(format!("tokenizer helper failed: {error}")))?;
        let value = serde_json::from_str::<serde_json::Value>(line.trim()).map_err(|error| {
            CliError::Runtime(format!(
                "tokenizer helper emitted invalid JSON: {error}: {line}"
            ))
        })?;
        if !value
            .get("ok")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Err(CliError::Runtime(format!(
                "tokenizer helper failed to initialize: {line}"
            )));
        }
        helper.backend = value
            .get("backend")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        Ok(helper)
    }

    fn backend(&self) -> &str {
        &self.backend
    }

    fn encode(&mut self, text: &str) -> Result<Vec<i32>, CliError> {
        let value = self.request(&serde_json::json!({"cmd":"encode","text":text}))?;
        let ids = value
            .get("ids")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| CliError::Runtime("tokenizer encode response missing ids".to_owned()))?;
        ids.iter()
            .map(|id| {
                let value = id.as_i64().ok_or_else(|| {
                    CliError::Runtime(
                        "tokenizer encode response contained non-integer id".to_owned(),
                    )
                })?;
                i32::try_from(value).map_err(|_| {
                    CliError::Runtime(format!("tokenizer id out of i32 range: {value}"))
                })
            })
            .collect()
    }

    fn request(&mut self, value: &serde_json::Value) -> Result<serde_json::Value, CliError> {
        serde_json::to_writer(&mut self.stdin, value).map_err(|error| {
            CliError::Runtime(format!("failed to write tokenizer request: {error}"))
        })?;
        writeln!(self.stdin).map_err(|error| {
            CliError::Runtime(format!("failed to flush tokenizer request: {error}"))
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
        Ok(value)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_even_and_odd_samples() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn policy_decision_marks_low_n_guarded_win_as_needs_more_data() {
        let decision = policy_decision("net_latency_guarded_5pct", 10.0, 1, 0, true, 8.0, 14.0);
        assert_eq!(decision, "needs_more_data");
    }

    #[test]
    fn policy_decision_rejects_acceptance_only_regression() {
        let decision = policy_decision("acceptance_threshold_35pct", -10.0, 1, 1, false, 8.0, 14.0);
        assert_eq!(decision, "reject_candidate");
    }
}
