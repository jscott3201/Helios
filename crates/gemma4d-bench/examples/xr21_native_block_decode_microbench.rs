use std::{
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{CliError, manifest, workload_corpus::WorkloadRecord};
use gemma4d_ffi::{KvCache, KvPolicy, LoadConfig, Target, decode_block, decode_one, prefill};
use gemma4d_tokenizer::sha256_hex;
use serde::Serialize;

const GOAL: &str = "XR21-native-block-decode-microbench";
const MODE: &str = "native_block_decode_microbench";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR21-native-block-decode-microbench";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const DEFAULT_WORKLOAD_ID: &str = "mtp_candidate_1k_001";
const DEFAULT_TRIALS: usize = 8;
const DEFAULT_WARMUPS: usize = 2;
const DEFAULT_MEMORY_CLIFF_GB: f64 = 14.0;
const DEFAULT_LOGIT_TOLERANCE: f32 = 0.25;
const MIN_BLOCK_SPEEDUP_PERCENT: f64 = 10.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1))?;
    fs::create_dir_all(&args.out_dir)?;

    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let decision_path = args.out_dir.join("decision.md");

    let run_id = run_id();
    let git_sha =
        command_stdout("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let git_status_short =
        command_stdout("git", &["status", "--short"]).unwrap_or_else(|| "unknown".to_owned());
    let command = command_line();
    let model_identity =
        manifest::capture_artifact_identity(&args.model_path, "GEMMA4D_MODEL_REVISION");

    let mut blockers = startup_blockers(&args);
    let mut records = Vec::new();
    let mut tokenizer_backend = "not_started".to_owned();
    let mut selected_workload = None;

    if blockers.is_empty() {
        let workload = load_workload(&args.workloads_path, &args.workload_id)?;
        let mut tokenizer = TokenizerHelper::start(&args.python, &args.model_path)?;
        tokenizer_backend = tokenizer.backend().to_owned();
        let encoded = encode_workload(&mut tokenizer, &workload)?;
        selected_workload = Some(selected_workload_row(&encoded));

        let target = Target::load(&target_config(&args, encoded.token_ids.len()))?;
        for trial_index in 0..(args.warmups + args.trials) {
            let trial_kind = if trial_index < args.warmups {
                "warmup"
            } else {
                "measured"
            };
            records.push(run_record(
                &args,
                &target,
                &run_id,
                &git_sha,
                &git_status_short,
                &encoded,
                trial_index,
                trial_kind,
            )?);
        }
    }

    blockers.extend(record_blockers(
        &records,
        args.memory_cliff_gb,
        args.logit_tolerance,
    ));
    blockers.sort();
    blockers.dedup();

    let measured = records
        .iter()
        .filter(|record| record.measured)
        .collect::<Vec<_>>();
    let serial_decode_ms_median = median(measured.iter().map(|record| record.serial_decode_ms));
    let block_decode_ms_median = median(measured.iter().map(|record| record.block_decode_ms));
    let speedup_percent = speedup_percent(serial_decode_ms_median, block_decode_ms_median);
    let max_peak_memory_gb = records
        .iter()
        .map(|record| record.peak_memory_gb)
        .fold(0.0_f32, f32::max);
    let max_logit_abs_diff = records
        .iter()
        .map(|record| record.max_logit_abs_diff)
        .fold(0.0_f32, f32::max);
    let exact_record_count = records
        .iter()
        .filter(|record| record.greedy_tokens_match && record.logits_within_tolerance)
        .count();
    let decision = decision_for(
        &blockers,
        args.trials,
        speedup_percent,
        max_peak_memory_gb,
        args.memory_cliff_gb,
        exact_record_count,
        records.len(),
    );
    let status = if blockers.is_empty() {
        "completed"
    } else {
        "blocked"
    };

    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        mode: MODE.to_owned(),
        status: status.to_owned(),
        decision,
        run_id,
        generated_at_unix_seconds: unix_now(),
        command,
        git_sha,
        git_status_short,
        model_identity,
        tokenizer_backend,
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
        selected_workload,
        requested_trials: args.trials,
        warmup_trials: args.warmups,
        memory_cliff_gb: args.memory_cliff_gb,
        logit_tolerance: args.logit_tolerance,
        min_block_speedup_percent: MIN_BLOCK_SPEEDUP_PERCENT,
        record_count: records.len(),
        measured_record_count: measured.len(),
        exact_record_count,
        serial_decode_ms_median,
        block_decode_ms_median,
        speedup_percent,
        max_peak_memory_gb,
        max_logit_abs_diff,
        blockers,
        records: records.clone(),
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR21 native block decode microbench: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR21 blocked; see blockers.md".into())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Args {
    out_dir: PathBuf,
    workloads_path: PathBuf,
    model_path: PathBuf,
    python: PathBuf,
    workload_id: String,
    trials: usize,
    warmups: usize,
    memory_cliff_gb: f64,
    logit_tolerance: f32,
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
            python: PathBuf::from(DEFAULT_PYTHON),
            workload_id: DEFAULT_WORKLOAD_ID.to_owned(),
            trials: DEFAULT_TRIALS,
            warmups: DEFAULT_WARMUPS,
            memory_cliff_gb: DEFAULT_MEMORY_CLIFF_GB,
            logit_tolerance: DEFAULT_LOGIT_TOLERANCE,
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
                "--python" => out.python = PathBuf::from(required_value(&mut args, "--python")?),
                "--workload-id" => out.workload_id = required_value(&mut args, "--workload-id")?,
                "--trials" => {
                    out.trials =
                        parse_positive_usize(&required_value(&mut args, "--trials")?, "--trials")?
                }
                "--warmups" | "--warmup-trials" => {
                    out.warmups =
                        parse_usize(&required_value(&mut args, "--warmups")?, "--warmups")?
                }
                "--memory-cliff-gb" => {
                    out.memory_cliff_gb = parse_finite_positive(
                        &required_value(&mut args, "--memory-cliff-gb")?,
                        "--memory-cliff-gb",
                    )?
                }
                "--logit-tolerance" => {
                    out.logit_tolerance = parse_finite_positive_f32(
                        &required_value(&mut args, "--logit-tolerance")?,
                        "--logit-tolerance",
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
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct EncodedWorkload {
    record: WorkloadRecord,
    prompt_sha256: String,
    token_ids: Vec<i32>,
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
    model_identity: manifest::ArtifactIdentity,
    tokenizer_backend: String,
    workloads_path: String,
    out_dir: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    generated_files: Vec<String>,
    selected_workload: Option<SelectedWorkload>,
    requested_trials: usize,
    warmup_trials: usize,
    memory_cliff_gb: f64,
    logit_tolerance: f32,
    min_block_speedup_percent: f64,
    record_count: usize,
    measured_record_count: usize,
    exact_record_count: usize,
    serial_decode_ms_median: f64,
    block_decode_ms_median: f64,
    speedup_percent: f64,
    max_peak_memory_gb: f32,
    max_logit_abs_diff: f32,
    blockers: Vec<String>,
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
    deterministic_seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    schema_version: u32,
    goal: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    deterministic_seed: u64,
    trial_index: usize,
    trial_kind: String,
    measured: bool,
    serial_input_tokens: Vec<i32>,
    serial_greedy_tokens: Vec<i32>,
    serial_greedy_logits: Vec<f32>,
    block_greedy_tokens: Vec<i32>,
    block_greedy_logits: Vec<f32>,
    greedy_tokens_match: bool,
    max_logit_abs_diff: f32,
    logits_within_tolerance: bool,
    serial_decode_ms: f64,
    block_decode_ms: f64,
    speedup_percent: f64,
    serial_sequence_len: u64,
    block_sequence_len: u64,
    peak_memory_gb: f32,
    serial_active_kv_bytes: u64,
    block_active_kv_bytes: u64,
    status: String,
    blocker: Option<String>,
}

#[allow(clippy::too_many_arguments)]
fn run_record(
    args: &Args,
    target: &Target,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    workload: &EncodedWorkload,
    trial_index: usize,
    trial_kind: &str,
) -> Result<Record, Box<dyn std::error::Error>> {
    let mut serial_cache = KvCache::create(&KvPolicy::default())?;
    let serial_prefill = prefill(target, &mut serial_cache, &workload.token_ids)?;
    let token_one = serial_prefill.greedy_token;

    let serial_first_started = Instant::now();
    let serial_second = decode_one(target, &mut serial_cache, token_one)?;
    let serial_first_elapsed = serial_first_started.elapsed();
    let token_two = serial_second.greedy_token;

    let serial_second_started = Instant::now();
    let serial_third = decode_one(target, &mut serial_cache, token_two)?;
    let serial_second_elapsed = serial_second_started.elapsed();
    let serial_decode_ms = duration_ms(serial_first_elapsed + serial_second_elapsed);
    let serial_greedy_tokens = vec![serial_second.greedy_token, serial_third.greedy_token];
    let serial_greedy_logits = vec![serial_second.greedy_logit, serial_third.greedy_logit];

    let mut block_cache = KvCache::create(&KvPolicy::default())?;
    let block_prefill = prefill(target, &mut block_cache, &workload.token_ids)?;
    let serial_input_tokens = vec![token_one, token_two];
    let block_started = Instant::now();
    let (block_step, block_greedy_tokens, block_greedy_logits) =
        decode_block(target, &mut block_cache, &serial_input_tokens)?;
    let block_decode_ms = duration_ms(block_started.elapsed());

    let greedy_tokens_match =
        block_prefill.greedy_token == token_one && block_greedy_tokens == serial_greedy_tokens;
    let max_logit_abs_diff = serial_greedy_logits
        .iter()
        .zip(block_greedy_logits.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let logits_within_tolerance = max_logit_abs_diff <= args.logit_tolerance;
    let peak_memory_gb = serial_prefill
        .peak_memory_gb
        .max(serial_second.peak_memory_gb)
        .max(serial_third.peak_memory_gb)
        .max(block_prefill.peak_memory_gb)
        .max(block_step.peak_memory_gb);
    let blocker = if !greedy_tokens_match {
        Some("block greedy tokens differed from serial decode".to_owned())
    } else if !logits_within_tolerance {
        Some(format!(
            "block logits differed from serial by {:.6}, above tolerance {:.6}",
            max_logit_abs_diff, args.logit_tolerance
        ))
    } else if peak_memory_gb > args.memory_cliff_gb as f32 {
        Some(format!(
            "peak MLX memory {:.3} GB exceeded {:.3} GB gate",
            peak_memory_gb, args.memory_cliff_gb
        ))
    } else {
        None
    };
    let status = if blocker.is_some() {
        "failed"
    } else {
        "passed"
    };

    Ok(Record {
        schema_version: 1,
        goal: GOAL.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        workload_id: workload.record.workload_id.clone(),
        family: workload.record.family.clone(),
        prompt_path: workload.record.prompt_path.clone(),
        prompt_sha256: workload.prompt_sha256.clone(),
        target_context_tokens: workload.record.target_context_tokens,
        actual_context_tokens: workload.token_ids.len(),
        deterministic_seed: workload.record.deterministic_seed,
        trial_index,
        trial_kind: trial_kind.to_owned(),
        measured: trial_kind == "measured",
        serial_input_tokens,
        serial_greedy_tokens,
        serial_greedy_logits,
        block_greedy_tokens,
        block_greedy_logits,
        greedy_tokens_match,
        max_logit_abs_diff,
        logits_within_tolerance,
        serial_decode_ms,
        block_decode_ms,
        speedup_percent: speedup_percent(serial_decode_ms, block_decode_ms),
        serial_sequence_len: serial_third.sequence_len,
        block_sequence_len: block_step.sequence_len,
        peak_memory_gb,
        serial_active_kv_bytes: serial_third.active_kv_bytes,
        block_active_kv_bytes: block_step.active_kv_bytes,
        status: status.to_owned(),
        blocker,
    })
}

fn target_config(args: &Args, context_len: usize) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(context_len.max(1) as u32)
            .expect("context length is non-zero"),
        allow_unsupported_config: false,
    }
}

fn startup_blockers(args: &Args) -> Vec<String> {
    let mut blockers = Vec::new();
    if !args.model_path.exists() {
        blockers.push(format!(
            "target model path does not exist: {}",
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
    if env::var_os("GEMMA4D_USE_NATIVE_GRAPH").is_none() {
        blockers.push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for XR21".to_owned());
    }
    if env::var_os("GEMMA4D_REQUIRE_MLX").is_none() {
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for XR21".to_owned());
    }
    blockers
}

fn record_blockers(records: &[Record], memory_cliff_gb: f64, logit_tolerance: f32) -> Vec<String> {
    let mut blockers = records
        .iter()
        .filter_map(|record| record.blocker.clone())
        .collect::<Vec<_>>();
    if records
        .iter()
        .any(|record| record.peak_memory_gb > memory_cliff_gb as f32)
    {
        blockers.push(format!(
            "at least one record exceeded the {:.3} GB memory gate",
            memory_cliff_gb
        ));
    }
    if records
        .iter()
        .any(|record| record.max_logit_abs_diff > logit_tolerance)
    {
        blockers.push(format!(
            "at least one record exceeded the {:.6} logit tolerance",
            logit_tolerance
        ));
    }
    blockers
}

fn decision_for(
    blockers: &[String],
    trials: usize,
    speedup_percent: f64,
    max_peak_memory_gb: f32,
    memory_cliff_gb: f64,
    exact_record_count: usize,
    record_count: usize,
) -> String {
    if !blockers.is_empty() || record_count == 0 || exact_record_count != record_count {
        "blocked_with_evidence".to_owned()
    } else if trials < 3 {
        "needs_more_data".to_owned()
    } else if max_peak_memory_gb > memory_cliff_gb as f32
        || speedup_percent < MIN_BLOCK_SPEEDUP_PERCENT
    {
        "reject_candidate".to_owned()
    } else {
        "accept_candidate".to_owned()
    }
}

fn load_workload(path: &Path, workload_id: &str) -> Result<WorkloadRecord, CliError> {
    let text = fs::read_to_string(path)
        .map_err(|error| CliError::Runtime(format!("failed to read workloads JSONL: {error}")))?;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let workload = serde_json::from_str::<WorkloadRecord>(line).map_err(|error| {
            CliError::Runtime(format!(
                "failed to parse workload line {} in {}: {error}",
                index + 1,
                path.display()
            ))
        })?;
        if workload.workload_id == workload_id {
            return Ok(workload);
        }
    }
    Err(CliError::Runtime(format!(
        "requested workload id not found: {workload_id}"
    )))
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
    })
}

fn selected_workload_row(workload: &EncodedWorkload) -> SelectedWorkload {
    SelectedWorkload {
        workload_id: workload.record.workload_id.clone(),
        family: workload.record.family.clone(),
        prompt_path: workload.record.prompt_path.clone(),
        prompt_sha256: workload.prompt_sha256.clone(),
        target_context_tokens: workload.record.target_context_tokens,
        actual_context_tokens: workload.token_ids.len(),
        deterministic_seed: workload.record.deterministic_seed,
    }
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
    out.push_str("# XR21 Native Block Decode Microbenchmark\n\n");
    out.push_str("## Summary\n\n| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Decision | `{}` |\n", summary.decision));
    out.push_str(&format!("| Status | `{}` |\n", summary.status));
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!(
        "| Trials | `{}` measured, `{}` warmup |\n",
        summary.requested_trials, summary.warmup_trials
    ));
    out.push_str(&format!(
        "| Median serial decode ms | `{:.3}` |\n",
        summary.serial_decode_ms_median
    ));
    out.push_str(&format!(
        "| Median block decode ms | `{:.3}` |\n",
        summary.block_decode_ms_median
    ));
    out.push_str(&format!(
        "| Block speedup | `{:.3}%` |\n",
        summary.speedup_percent
    ));
    out.push_str(&format!(
        "| Max logit abs diff | `{:.6}` |\n",
        summary.max_logit_abs_diff
    ));
    out.push_str(&format!(
        "| Peak memory GB | `{:.3}` |\n\n",
        summary.max_peak_memory_gb
    ));

    out.push_str("## Records\n\n");
    out.push_str("| Trial | Kind | Serial ms | Block ms | Speedup % | Greedy match | Max logit diff | Peak GB | Status |\n");
    out.push_str("|---:|---|---:|---:|---:|---|---:|---:|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| {} | `{}` | {:.3} | {:.3} | {:.3} | `{}` | {:.6} | {:.3} | `{}` |\n",
            record.trial_index,
            record.trial_kind,
            record.serial_decode_ms,
            record.block_decode_ms,
            record.speedup_percent,
            record.greedy_tokens_match,
            record.max_logit_abs_diff,
            record.peak_memory_gb,
            record.status
        ));
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR21 Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No hard blockers recorded.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn render_decision(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR21 Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str(&format!(
        "Median block decode speedup was {:.3}% against the two-decode serial path. Acceptance requires at least {:.3}% with exact greedy/logit parity and memory under {:.3} GB.\n",
        summary.speedup_percent, summary.min_block_speedup_percent, summary.memory_cliff_gb
    ));
    out
}

fn median<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    if values.is_empty() {
        0.0
    } else if values.len() % 2 == 1 {
        values[values.len() / 2]
    } else {
        (values[(values.len() / 2) - 1] + values[values.len() / 2]) / 2.0
    }
}

fn speedup_percent(baseline_ms: f64, candidate_ms: f64) -> f64 {
    if baseline_ms <= 0.0 {
        0.0
    } else {
        ((baseline_ms - candidate_ms) / baseline_ms) * 100.0
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn run_id() -> String {
    format!("xr21-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn command_line() -> String {
    env::args().collect::<Vec<_>>().join(" ")
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
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
    let parsed = value
        .parse::<f64>()
        .map_err(|error| CliError::Usage(format!("{option} must be a number: {error}")))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(CliError::Usage(format!(
            "{option} must be a finite positive number"
        )))
    }
}

fn parse_finite_positive_f32(value: &str, option: &str) -> Result<f32, CliError> {
    let parsed = value
        .parse::<f32>()
        .map_err(|error| CliError::Usage(format!("{option} must be a number: {error}")))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(CliError::Usage(format!(
            "{option} must be a finite positive number"
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
        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr21_native_block_decode_microbench -- [--out-dir PATH] [--workload-id ID] [--trials N] [--warmups N]\n\ndefault out-dir: {DEFAULT_OUT_DIR}"
    )
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
        let _ = self.request(&serde_json::json!({"cmd":"shutdown"}));
        let _ = self.child.wait();
    }
}
