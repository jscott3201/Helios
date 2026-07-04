use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{CliError, manifest, workload_corpus::WorkloadRecord};
use gemma4d_ffi::{
    Drafter, KvCache, KvPolicy, LoadConfig, MtpTraceInfo, Target, decode_one, draft_block, prefill,
    verify_tokens,
};
use gemma4d_tokenizer::sha256_hex;
use serde::Serialize;

const GOAL: &str = "XR03-mtp-real-context-diagnosis";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR03-mtp-real-context-diagnosis";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_ASSISTANT_MODEL: &str = "artifacts/models/gemma-4-12B-it-qat-assistant-4bit";
const DEFAULT_PYTHON: &str = "/opt/homebrew/opt/mlx-lm/libexec/bin/python";
const DEFAULT_MAX_NEW_TOKENS: usize = 4;
const DEFAULT_WORKLOAD_IDS: &[&str] = &[
    "mtp_candidate_1k_001",
    "mtp_candidate_4k_001",
    "chat_short_1k_001",
    "code_review_rust_4k_001",
    "benchmark_qa_4k_001",
];
const MODE: &str = "native_mtp_real_context_trace";

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
    let target_identity =
        manifest::capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION");
    let assistant_identity =
        manifest::capture_artifact_identity(&options.assistant_model_path, "GEMMA4D_MTP_REVISION");
    let mut blockers = startup_blockers(&options);
    let workloads = select_workloads(load_workloads(&options.workloads_path)?, &options)?;
    let mut records = Vec::new();
    let mut tokenizer_backend = "not_started".to_owned();

    if blockers.is_empty() {
        let mut tokenizer = TokenizerHelper::start(&options.python, &options.model_path)?;
        tokenizer_backend = tokenizer.backend().to_owned();

        for workload in &workloads {
            let prompt = fs::read_to_string(&workload.prompt_path).map_err(|error| {
                CliError::Runtime(format!(
                    "failed to read prompt {}: {error}",
                    workload.prompt_path
                ))
            })?;
            let prompt_sha256 = sha256_hex(prompt.as_bytes());
            let token_ids = tokenizer.encode(&prompt)?;
            let selected_max_new_tokens = options.max_new_tokens.min(workload.max_new_tokens);
            if prompt_sha256 != workload.prompt_sha256 {
                blockers.push(format!(
                    "{} prompt sha mismatch: manifest={} actual={}",
                    workload.workload_id, workload.prompt_sha256, prompt_sha256
                ));
                continue;
            }
            if token_ids.len() != workload.actual_context_tokens {
                blockers.push(format!(
                    "{} tokenizer length mismatch: manifest={} actual={}",
                    workload.workload_id,
                    workload.actual_context_tokens,
                    token_ids.len()
                ));
                continue;
            }

            let workload_run = WorkloadRun {
                record: workload.clone(),
                prompt_sha256,
                token_ids,
                workload_max_new_tokens: workload.max_new_tokens,
                max_new_tokens: selected_max_new_tokens,
            };
            let baseline = run_baseline(&options, &workload_run)?;
            for block_size in &options.block_sizes {
                records.push(run_record(
                    &options,
                    &run_id,
                    &git_sha,
                    &git_status_short,
                    &target_identity,
                    &assistant_identity,
                    &workload_run,
                    *block_size,
                    baseline.clone(),
                    &mut tokenizer,
                )?);
            }
        }
    }

    blockers.extend(blockers_for_records(&records));
    blockers.sort();
    blockers.dedup();

    let summary = build_summary(
        &options,
        &run_id,
        &git_sha,
        &git_status_short,
        target_identity,
        assistant_identity,
        tokenizer_backend,
        &records,
        blockers,
        vec![
            records_path.display().to_string(),
            summary_path.display().to_string(),
            report_path.display().to_string(),
            blockers_path.display().to_string(),
            decision_path.display().to_string(),
        ],
    );

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR03 MTP real-context diagnosis: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    if summary.decision == "blocked_with_evidence" {
        Err("XR03 diagnosis blocked; see blockers.md".into())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Options {
    out_dir: PathBuf,
    workloads_path: PathBuf,
    model_path: PathBuf,
    assistant_model_path: PathBuf,
    python: PathBuf,
    max_new_tokens: usize,
    block_sizes: Vec<usize>,
    workload_ids: Vec<String>,
    max_workloads: Option<usize>,
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
            assistant_model_path: PathBuf::from(DEFAULT_ASSISTANT_MODEL),
            python: PathBuf::from(DEFAULT_PYTHON),
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            block_sizes: vec![1, 2],
            workload_ids: DEFAULT_WORKLOAD_IDS
                .iter()
                .map(|workload_id| (*workload_id).to_owned())
                .collect(),
            max_workloads: None,
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
                "--assistant-model-path" => {
                    options.assistant_model_path =
                        PathBuf::from(required_value(&mut args, "--assistant-model-path")?)
                }
                "--python" => {
                    options.python = PathBuf::from(required_value(&mut args, "--python")?)
                }
                "--max-new-tokens" => {
                    options.max_new_tokens = parse_positive_usize(
                        &required_value(&mut args, "--max-new-tokens")?,
                        "--max-new-tokens",
                    )?
                }
                "--block-sizes" => {
                    options.block_sizes =
                        parse_usize_csv(&required_value(&mut args, "--block-sizes")?)?;
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
                    )?)
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
        options.block_sizes.sort_unstable();
        options.block_sizes.dedup();
        if options.block_sizes.is_empty() {
            return Err(CliError::Usage(
                "--block-sizes must not be empty".to_owned(),
            ));
        }
        if options.block_sizes.iter().any(|block_size| *block_size > 2) {
            return Err(CliError::Usage(
                "XR03 records design notes for block sizes 3/4 only; executable block sizes must be <= 2"
                    .to_owned(),
            ));
        }
        Ok(options)
    }
}

#[derive(Debug, Clone)]
struct WorkloadRun {
    record: WorkloadRecord,
    prompt_sha256: String,
    token_ids: Vec<i32>,
    workload_max_new_tokens: usize,
    max_new_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: &'static str,
    decision: String,
    status: String,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    git_sha: String,
    git_status_short: String,
    model_identity: manifest::ArtifactIdentity,
    assistant_identity: manifest::ArtifactIdentity,
    artifact_compatibility: ArtifactCompatibility,
    tokenizer_backend: String,
    workloads_path: String,
    out_dir: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    selected_workloads: Vec<SelectedWorkload>,
    block_sizes: Vec<usize>,
    requested_max_new_tokens: usize,
    record_count: usize,
    exact_records: usize,
    nonzero_acceptance_records: usize,
    acceptance_summary: AcceptanceSummary,
    root_cause_assessment: RootCauseAssessment,
    fix_hypotheses: Vec<FixHypothesis>,
    failed_hypotheses: Vec<String>,
    generated_files: Vec<String>,
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
    workload_max_new_tokens: usize,
    selected_max_new_tokens: usize,
    deterministic_seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactCompatibility {
    target_exists: bool,
    assistant_exists: bool,
    target_revision: String,
    assistant_revision: String,
    target_local_artifact_sha256: String,
    assistant_local_artifact_sha256: String,
    assessment: String,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    schema_version: u32,
    goal: &'static str,
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
    workload_max_new_tokens: usize,
    max_new_tokens: usize,
    block_size: usize,
    baseline: GreedyRun,
    mtp: MtpRun,
    comparison: Comparison,
    diagnosis: RecordDiagnosis,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct GreedyRun {
    generated_tokens: Vec<i32>,
    model_load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
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
    total_ms: f64,
    attempted_draft_tokens: u64,
    accepted_draft_tokens: u64,
    acceptance_rate: f64,
    accepted_tokens_per_verify: f64,
    target_verify_passes: u64,
    rollback_count: u64,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
    events: Vec<MtpEvent>,
}

#[derive(Debug, Clone, Serialize)]
struct MtpEvent {
    pass_index: u64,
    block_size: usize,
    draft_tokens: Vec<TokenText>,
    committed_tokens: Vec<TokenText>,
    accepted_draft_count: u32,
    rejected: bool,
    context_sequence_len: u64,
    sequence_len: u64,
    verify_ms: f64,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
    trace_position_count: u32,
    trace_top_k: u32,
    first_position: u64,
    hidden_shape: Vec<u64>,
    full_attention_layer: u32,
    full_attention_key_shape: Vec<u64>,
    full_attention_value_shape: Vec<u64>,
    sliding_attention_layer: u32,
    sliding_attention_key_shape: Vec<u64>,
    sliding_attention_value_shape: Vec<u64>,
    per_draft_token: Vec<DraftTokenTrace>,
    lookahead: Option<TargetTokenTrace>,
}

#[derive(Debug, Clone, Serialize)]
struct DraftTokenTrace {
    draft_index: usize,
    position_offset: u64,
    draft: TokenText,
    target_greedy: TokenText,
    target_greedy_logit: f32,
    draft_logit: f32,
    logit_margin: f32,
    draft_in_target_top_k: bool,
    target_top_k: Vec<TopKToken>,
}

#[derive(Debug, Clone, Serialize)]
struct TargetTokenTrace {
    position_offset: u64,
    target_greedy: TokenText,
    target_greedy_logit: f32,
    target_top_k: Vec<TopKToken>,
}

#[derive(Debug, Clone, Serialize)]
struct TopKToken {
    rank: usize,
    token_id: i32,
    text: String,
    logit: f32,
}

#[derive(Debug, Clone, Serialize)]
struct TokenText {
    token_id: i32,
    text: String,
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
struct RecordDiagnosis {
    likely_cause: String,
    draft_in_top_k_rate: f64,
    mean_logit_margin: f64,
    max_logit_margin: f64,
    exactness_passed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct AcceptanceSummary {
    attempted_draft_tokens: u64,
    accepted_draft_tokens: u64,
    acceptance_rate: f64,
    draft_in_top_k_count: u64,
    draft_trace_count: u64,
    draft_in_top_k_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
struct RootCauseAssessment {
    classification: String,
    evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FixHypothesis {
    rank: u32,
    hypothesis: String,
    expected_payoff: String,
    risk: String,
    evidence: String,
}

#[allow(clippy::too_many_arguments)]
fn run_record(
    options: &Options,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    _target_identity: &manifest::ArtifactIdentity,
    _assistant_identity: &manifest::ArtifactIdentity,
    workload: &WorkloadRun,
    block_size: usize,
    baseline: GreedyRun,
    tokenizer: &mut TokenizerHelper,
) -> Result<Record, Box<dyn std::error::Error>> {
    let mtp = run_mtp(options, workload, block_size, tokenizer)?;
    let comparison = compare_tokens(&baseline.generated_tokens, &mtp.generated_tokens);
    let diagnosis = diagnose_record(&comparison, &mtp);
    let mut blockers = Vec::new();
    if !comparison.byte_identical {
        let detail = comparison
            .first_mismatch
            .as_ref()
            .map(|mismatch| {
                format!(
                    " at generated index {}: baseline={:?} mtp={:?}",
                    mismatch.index, mismatch.baseline_token, mismatch.mtp_token
                )
            })
            .unwrap_or_default();
        blockers.push(format!(
            "{} block_size={} MTP output differed from non-MTP native output{}",
            workload.record.workload_id, block_size, detail
        ));
    }
    Ok(Record {
        schema_version: 1,
        goal: GOAL,
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
        workload_max_new_tokens: workload.workload_max_new_tokens,
        max_new_tokens: workload.max_new_tokens,
        block_size,
        baseline,
        mtp,
        comparison,
        diagnosis,
        blockers,
    })
}

fn run_baseline(
    options: &Options,
    workload: &WorkloadRun,
) -> Result<GreedyRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let load_started = Instant::now();
    let target = Target::load(&target_config(options, workload))?;
    let model_load = load_started.elapsed();
    let mut cache = KvCache::create(&KvPolicy::default())?;

    let prefill_started = Instant::now();
    let mut step = prefill(&target, &mut cache, &workload.token_ids)?;
    let prefill_duration = prefill_started.elapsed();
    let mut decode_duration = Duration::ZERO;
    let mut generated = Vec::with_capacity(workload.max_new_tokens);
    let mut peak_memory_gb = step.peak_memory_gb;
    let mut active_kv_bytes = step.active_kv_bytes;

    for index in 0..workload.max_new_tokens {
        generated.push(step.greedy_token);
        if index + 1 < workload.max_new_tokens {
            let decode_started = Instant::now();
            step = decode_one(&target, &mut cache, step.greedy_token)?;
            decode_duration += decode_started.elapsed();
            peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
            active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
        }
    }

    Ok(GreedyRun {
        generated_tokens: generated,
        model_load_ms: duration_ms(model_load),
        prefill_ms: duration_ms(prefill_duration),
        decode_ms: duration_ms(decode_duration),
        total_ms: duration_ms(started.elapsed()),
        peak_memory_gb,
        active_kv_bytes,
    })
}

fn run_mtp(
    options: &Options,
    workload: &WorkloadRun,
    block_size: usize,
    tokenizer: &mut TokenizerHelper,
) -> Result<MtpRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let load_started = Instant::now();
    let target = Target::load(&target_config(options, workload))?;
    let model_load = load_started.elapsed();
    let drafter_started = Instant::now();
    let drafter = Drafter::load(&assistant_config(options, workload), &target)?;
    let drafter_load = drafter_started.elapsed();
    let mut cache = KvCache::create(&KvPolicy::default())?;

    let prefill_started = Instant::now();
    let first = prefill(&target, &mut cache, &workload.token_ids)?;
    let prefill_duration = prefill_started.elapsed();

    let mut generated = Vec::with_capacity(workload.max_new_tokens);
    let mut draft_duration = Duration::ZERO;
    let mut verify_duration = Duration::ZERO;
    let mut attempted_draft_tokens = 0_u64;
    let mut accepted_draft_tokens = 0_u64;
    let mut target_verify_passes = 0_u64;
    let mut rollback_count = 0_u64;
    let mut peak_memory_gb = first.peak_memory_gb;
    let mut active_kv_bytes = first.active_kv_bytes;
    let mut events = Vec::new();

    while generated.len() < workload.max_new_tokens {
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
        let verify_started = Instant::now();
        let step = verify_tokens(&target, &mut cache, &draft)?;
        let verify_elapsed = verify_started.elapsed();
        verify_duration += verify_elapsed;
        peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
        active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);

        let committed = step.committed_tokens().to_vec();
        if committed.is_empty() {
            return Err("native MTP verifier committed no tokens".into());
        }
        let accepted = u64::from(step.accepted_draft_count);
        accepted_draft_tokens += accepted;
        let rejected =
            usize::try_from(step.accepted_draft_count).unwrap_or(usize::MAX) < draft.len();
        if rejected {
            rollback_count += 1;
        }
        for token in &committed {
            if generated.len() < workload.max_new_tokens {
                generated.push(*token);
            }
        }

        events.push(event_from_step(
            target_verify_passes,
            block_size,
            &draft,
            &committed,
            step.accepted_draft_count,
            rejected,
            &step.mtp_trace,
            step.sequence_len,
            duration_ms(verify_elapsed),
            step.peak_memory_gb,
            step.active_kv_bytes,
            tokenizer,
        )?);
    }

    Ok(MtpRun {
        generated_tokens: generated,
        model_load_ms: duration_ms(model_load),
        drafter_load_ms: duration_ms(drafter_load),
        prefill_ms: duration_ms(prefill_duration),
        draft_ms: duration_ms(draft_duration),
        verify_ms: duration_ms(verify_duration),
        total_ms: duration_ms(started.elapsed()),
        attempted_draft_tokens,
        accepted_draft_tokens,
        acceptance_rate: ratio(accepted_draft_tokens, attempted_draft_tokens),
        accepted_tokens_per_verify: ratio(accepted_draft_tokens, target_verify_passes),
        target_verify_passes,
        rollback_count,
        peak_memory_gb,
        active_kv_bytes,
        events,
    })
}

#[allow(clippy::too_many_arguments)]
fn event_from_step(
    pass_index: u64,
    block_size: usize,
    draft_tokens: &[i32],
    committed_tokens: &[i32],
    accepted_draft_count: u32,
    rejected: bool,
    trace: &MtpTraceInfo,
    sequence_len: u64,
    verify_ms: f64,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
    tokenizer: &mut TokenizerHelper,
) -> Result<MtpEvent, Box<dyn std::error::Error>> {
    let mut per_draft_token = Vec::new();
    for (index, draft_token) in draft_tokens
        .iter()
        .copied()
        .enumerate()
        .take(draft_tokens.len().min(trace.target_tokens.len()))
    {
        per_draft_token.push(DraftTokenTrace {
            draft_index: index,
            position_offset: trace.position_offsets.get(index).copied().unwrap_or(0),
            draft: token_text(draft_token, tokenizer)?,
            target_greedy: token_text(trace.target_tokens[index], tokenizer)?,
            target_greedy_logit: trace.target_logits.get(index).copied().unwrap_or(0.0),
            draft_logit: trace.draft_logits.get(index).copied().unwrap_or(0.0),
            logit_margin: trace.logit_margins.get(index).copied().unwrap_or(0.0),
            draft_in_target_top_k: trace.draft_in_top_k.get(index).copied().unwrap_or(false),
            target_top_k: top_k_for_position(index, trace, tokenizer)?,
        });
    }

    let lookahead_index = draft_tokens.len();
    let lookahead = if lookahead_index < trace.target_tokens.len() {
        Some(TargetTokenTrace {
            position_offset: trace
                .position_offsets
                .get(lookahead_index)
                .copied()
                .unwrap_or(0),
            target_greedy: token_text(trace.target_tokens[lookahead_index], tokenizer)?,
            target_greedy_logit: trace
                .target_logits
                .get(lookahead_index)
                .copied()
                .unwrap_or(0.0),
            target_top_k: top_k_for_position(lookahead_index, trace, tokenizer)?,
        })
    } else {
        None
    };

    Ok(MtpEvent {
        pass_index,
        block_size,
        draft_tokens: tokens_text(draft_tokens, tokenizer)?,
        committed_tokens: tokens_text(committed_tokens, tokenizer)?,
        accepted_draft_count,
        rejected,
        context_sequence_len: trace.context_sequence_len,
        sequence_len,
        verify_ms,
        peak_memory_gb,
        active_kv_bytes,
        trace_position_count: trace.position_count,
        trace_top_k: trace.top_k,
        first_position: trace.first_position,
        hidden_shape: trace.hidden_shape.clone(),
        full_attention_layer: trace.full_attention_layer,
        full_attention_key_shape: trace.full_attention_key_shape.clone(),
        full_attention_value_shape: trace.full_attention_value_shape.clone(),
        sliding_attention_layer: trace.sliding_attention_layer,
        sliding_attention_key_shape: trace.sliding_attention_key_shape.clone(),
        sliding_attention_value_shape: trace.sliding_attention_value_shape.clone(),
        per_draft_token,
        lookahead,
    })
}

fn top_k_for_position(
    position: usize,
    trace: &MtpTraceInfo,
    tokenizer: &mut TokenizerHelper,
) -> Result<Vec<TopKToken>, Box<dyn std::error::Error>> {
    let Some(ids) = trace.top_token_ids.get(position) else {
        return Ok(Vec::new());
    };
    let Some(logits) = trace.top_logits.get(position) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (rank, (token_id, logit)) in ids.iter().zip(logits.iter()).enumerate() {
        if *token_id < 0 {
            continue;
        }
        out.push(TopKToken {
            rank: rank + 1,
            token_id: *token_id,
            text: tokenizer.decode(&[*token_id])?,
            logit: *logit,
        });
    }
    Ok(out)
}

fn token_text(
    token_id: i32,
    tokenizer: &mut TokenizerHelper,
) -> Result<TokenText, Box<dyn std::error::Error>> {
    if token_id < 0 {
        return Ok(TokenText {
            token_id,
            text: "<missing>".to_owned(),
        });
    }
    Ok(TokenText {
        token_id,
        text: tokenizer.decode(&[token_id])?,
    })
}

fn tokens_text(
    tokens: &[i32],
    tokenizer: &mut TokenizerHelper,
) -> Result<Vec<TokenText>, Box<dyn std::error::Error>> {
    tokens
        .iter()
        .copied()
        .map(|token| token_text(token, tokenizer))
        .collect()
}

fn target_config(options: &Options, workload: &WorkloadRun) -> LoadConfig {
    LoadConfig {
        model_path: options.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(workload.token_ids.len().max(1) as u32)
            .expect("context length is non-zero"),
        allow_unsupported_config: false,
    }
}

fn assistant_config(options: &Options, workload: &WorkloadRun) -> LoadConfig {
    LoadConfig {
        model_path: options.assistant_model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4_mtp_assistant".to_owned()),
        max_context_tokens: NonZeroU32::new(workload.token_ids.len().max(1) as u32)
            .expect("context length is non-zero"),
        allow_unsupported_config: false,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_summary(
    options: &Options,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    target_identity: manifest::ArtifactIdentity,
    assistant_identity: manifest::ArtifactIdentity,
    tokenizer_backend: String,
    records: &[Record],
    blockers: Vec<String>,
    generated_files: Vec<String>,
) -> Summary {
    let exact_records = records
        .iter()
        .filter(|record| record.comparison.byte_identical)
        .count();
    let nonzero_acceptance_records = records
        .iter()
        .filter(|record| record.mtp.accepted_draft_tokens > 0)
        .count();
    let acceptance_summary = acceptance_summary(records);
    let root_cause_assessment = root_cause_assessment(records, &acceptance_summary);
    let fix_hypotheses = fix_hypotheses(&root_cause_assessment, &acceptance_summary);
    let failed_hypotheses = failed_hypotheses(records, &root_cause_assessment);
    let decision = decision_for(&blockers, records, &acceptance_summary);
    let status = if blockers.is_empty() {
        if records.len() == exact_records {
            "passed"
        } else {
            "failed"
        }
    } else {
        "blocked"
    };

    Summary {
        schema_version: 1,
        goal: GOAL,
        decision,
        status: status.to_owned(),
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        mode: MODE,
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        artifact_compatibility: artifact_compatibility(&target_identity, &assistant_identity),
        model_identity: target_identity,
        assistant_identity,
        tokenizer_backend,
        workloads_path: options.workloads_path.display().to_string(),
        out_dir: options.out_dir.display().to_string(),
        records_path: options.out_dir.join("records.jsonl").display().to_string(),
        summary_path: options.out_dir.join("summary.json").display().to_string(),
        report_path: options.out_dir.join("report.md").display().to_string(),
        blockers_path: options.out_dir.join("blockers.md").display().to_string(),
        decision_path: options.out_dir.join("decision.md").display().to_string(),
        selected_workloads: selected_workloads(records),
        block_sizes: options.block_sizes.clone(),
        requested_max_new_tokens: options.max_new_tokens,
        record_count: records.len(),
        exact_records,
        nonzero_acceptance_records,
        acceptance_summary,
        root_cause_assessment,
        fix_hypotheses,
        failed_hypotheses,
        generated_files,
        blockers,
        records: records.to_vec(),
    }
}

fn selected_workloads(records: &[Record]) -> Vec<SelectedWorkload> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for record in records {
        if !seen.insert(record.workload_id.clone()) {
            continue;
        }
        out.push(SelectedWorkload {
            workload_id: record.workload_id.clone(),
            family: record.family.clone(),
            prompt_path: record.prompt_path.clone(),
            prompt_sha256: record.prompt_sha256.clone(),
            target_context_tokens: record.target_context_tokens,
            actual_context_tokens: record.actual_context_tokens,
            workload_max_new_tokens: record.workload_max_new_tokens,
            selected_max_new_tokens: record.max_new_tokens,
            deterministic_seed: record.deterministic_seed,
        });
    }
    out
}

fn artifact_compatibility(
    target: &manifest::ArtifactIdentity,
    assistant: &manifest::ArtifactIdentity,
) -> ArtifactCompatibility {
    let target_revision = target
        .revision
        .clone()
        .unwrap_or_else(|| "unavailable".to_owned());
    let assistant_revision = assistant
        .revision
        .clone()
        .unwrap_or_else(|| "unavailable".to_owned());
    let assessment = if !target.exists || !assistant.exists {
        "blocked: one or more artifacts are missing"
    } else if target.revision.is_none() || assistant.revision.is_none() {
        "local artifact hashes recorded; upstream target/assistant revision alignment remains unverified"
    } else if target_revision == assistant_revision {
        "target and assistant revisions match"
    } else {
        "target and assistant revisions differ; assistant artifact mismatch remains possible"
    };
    ArtifactCompatibility {
        target_exists: target.exists,
        assistant_exists: assistant.exists,
        target_revision,
        assistant_revision,
        target_local_artifact_sha256: target.local_artifact_sha256.clone(),
        assistant_local_artifact_sha256: assistant.local_artifact_sha256.clone(),
        assessment: assessment.to_owned(),
    }
}

fn acceptance_summary(records: &[Record]) -> AcceptanceSummary {
    let attempted_draft_tokens = records
        .iter()
        .map(|record| record.mtp.attempted_draft_tokens)
        .sum::<u64>();
    let accepted_draft_tokens = records
        .iter()
        .map(|record| record.mtp.accepted_draft_tokens)
        .sum::<u64>();
    let draft_trace_count = records
        .iter()
        .flat_map(|record| &record.mtp.events)
        .flat_map(|event| &event.per_draft_token)
        .count() as u64;
    let draft_in_top_k_count = records
        .iter()
        .flat_map(|record| &record.mtp.events)
        .flat_map(|event| &event.per_draft_token)
        .filter(|trace| trace.draft_in_target_top_k)
        .count() as u64;

    AcceptanceSummary {
        attempted_draft_tokens,
        accepted_draft_tokens,
        acceptance_rate: ratio(accepted_draft_tokens, attempted_draft_tokens),
        draft_in_top_k_count,
        draft_trace_count,
        draft_in_top_k_rate: ratio(draft_in_top_k_count, draft_trace_count),
    }
}

fn root_cause_assessment(
    records: &[Record],
    acceptance: &AcceptanceSummary,
) -> RootCauseAssessment {
    let exact = records
        .iter()
        .all(|record| record.comparison.byte_identical);
    let mut evidence = Vec::new();
    evidence.push(format!(
        "attempted={} accepted={} acceptance_rate={:.3}",
        acceptance.attempted_draft_tokens,
        acceptance.accepted_draft_tokens,
        acceptance.acceptance_rate
    ));
    evidence.push(format!(
        "draft_in_top_k={}/{} rate={:.3}",
        acceptance.draft_in_top_k_count,
        acceptance.draft_trace_count,
        acceptance.draft_in_top_k_rate
    ));
    evidence.push(format!("byte_identical_outputs={exact}"));
    let classification = if !exact {
        "verifier_exactness_failure"
    } else if acceptance.attempted_draft_tokens == 0 {
        "blocked_no_draft_data"
    } else if acceptance.accepted_draft_tokens == 0 && acceptance.draft_in_top_k_count == 0 {
        evidence.push("all observed assistant drafts missed target top-k".to_owned());
        "implementation_or_artifact_mismatch"
    } else if acceptance.acceptance_rate < 0.05 {
        "low_acceptance_mixed_workload_or_alignment"
    } else {
        "workload_dependent_acceptance"
    };
    RootCauseAssessment {
        classification: classification.to_owned(),
        evidence,
    }
}

fn fix_hypotheses(
    assessment: &RootCauseAssessment,
    acceptance: &AcceptanceSummary,
) -> Vec<FixHypothesis> {
    let mut out = Vec::new();
    if assessment.classification == "verifier_exactness_failure" {
        out.push(FixHypothesis {
            rank: 1,
            hypothesis: "Add a focused parity trace comparing target incremental decode with full verifier logits at the first divergent generated token".to_owned(),
            expected_payoff: "high".to_owned(),
            risk: "low".to_owned(),
            evidence: "MTP output diverged from non-MTP while verifier exactness is the hard gate; the repair must start with target-path parity, not drafter tuning".to_owned(),
        });
        out.push(FixHypothesis {
            rank: 2,
            hypothesis: "Audit native MTP verify position offsets and target KV state after fallback commits near the 4k context boundary".to_owned(),
            expected_payoff: "high".to_owned(),
            risk: "medium".to_owned(),
            evidence: "Only benchmark_qa_4k_001 failed exactness, and its manifest records actual_context_tokens below the 4096 target context length".to_owned(),
        });
        out.push(FixHypothesis {
            rank: 3,
            hypothesis: "Keep MTP disabled by default and reject any acceptance-rate optimization until byte-identical exactness is restored".to_owned(),
            expected_payoff: "policy_guardrail".to_owned(),
            risk: "low".to_owned(),
            evidence: format!(
                "nonzero acceptance was observed ({}/{}), but exactness failed",
                acceptance.accepted_draft_tokens, acceptance.attempted_draft_tokens
            ),
        });
    } else if assessment.classification == "implementation_or_artifact_mismatch" {
        out.push(FixHypothesis {
            rank: 1,
            hypothesis: "Compare Helios assistant hidden-state projection and position offsets against an MLX reference MTP path".to_owned(),
            expected_payoff: "high".to_owned(),
            risk: "medium".to_owned(),
            evidence: "assistant draft tokens missed every observed target top-k while target exactness held".to_owned(),
        });
        out.push(FixHypothesis {
            rank: 2,
            hypothesis: "Verify target and assistant artifact revision alignment with upstream metadata or a known-good paired checkpoint".to_owned(),
            expected_payoff: "high".to_owned(),
            risk: "low".to_owned(),
            evidence: "local artifact hashes are recorded but upstream revision compatibility is unavailable".to_owned(),
        });
        out.push(FixHypothesis {
            rank: 3,
            hypothesis:
                "Audit captured full/sliding shared KV layer order and shapes used by the assistant"
                    .to_owned(),
            expected_payoff: "medium".to_owned(),
            risk: "medium".to_owned(),
            evidence: "XR03 records shared KV layer and shape metadata for every verify pass"
                .to_owned(),
        });
    } else {
        out.push(FixHypothesis {
            rank: 1,
            hypothesis: "Expand the trace corpus before selecting a repair".to_owned(),
            expected_payoff: "medium".to_owned(),
            risk: "low".to_owned(),
            evidence: format!(
                "observed acceptance_rate={:.3}, draft_in_top_k_rate={:.3}",
                acceptance.acceptance_rate, acceptance.draft_in_top_k_rate
            ),
        });
    }
    out.push(FixHypothesis {
        rank: (out.len() + 1) as u32,
        hypothesis: "Do not enable block sizes 3/4 until block sizes 1/2 have exactness and non-trivial acceptance".to_owned(),
        expected_payoff: "policy_guardrail".to_owned(),
        risk: "low".to_owned(),
        evidence: "XR03 executable harness rejects block sizes above 2".to_owned(),
    });
    out
}

fn failed_hypotheses(records: &[Record], assessment: &RootCauseAssessment) -> Vec<String> {
    let mut out = Vec::new();
    if assessment.classification == "implementation_or_artifact_mismatch" {
        out.push("workload-only explanation is weak: low acceptance reproduced across selected real workload families".to_owned());
    }
    if assessment.classification == "verifier_exactness_failure" {
        out.push("assistant-only mismatch is insufficient: exactness failed after target verification committed tokens".to_owned());
    }
    if records
        .iter()
        .all(|record| record.comparison.byte_identical)
    {
        out.push("verifier exactness failure did not reproduce: MTP outputs remained byte-identical to non-MTP native outputs".to_owned());
    }
    out
}

fn decision_for(blockers: &[String], records: &[Record], acceptance: &AcceptanceSummary) -> String {
    if !blockers.is_empty()
        || records.is_empty()
        || records
            .iter()
            .any(|record| !record.comparison.byte_identical)
    {
        "blocked_with_evidence".to_owned()
    } else if acceptance.attempted_draft_tokens > 0 {
        "accept_candidate".to_owned()
    } else {
        "needs_more_data".to_owned()
    }
}

fn diagnose_record(comparison: &Comparison, mtp: &MtpRun) -> RecordDiagnosis {
    let traces = mtp
        .events
        .iter()
        .flat_map(|event| &event.per_draft_token)
        .collect::<Vec<_>>();
    let draft_in_top_k = traces
        .iter()
        .filter(|trace| trace.draft_in_target_top_k)
        .count();
    let mean_logit_margin = if traces.is_empty() {
        0.0
    } else {
        traces
            .iter()
            .map(|trace| f64::from(trace.logit_margin))
            .sum::<f64>()
            / traces.len() as f64
    };
    let max_logit_margin = traces
        .iter()
        .map(|trace| f64::from(trace.logit_margin))
        .fold(0.0_f64, f64::max);
    let draft_in_top_k_rate = ratio(draft_in_top_k as u64, traces.len() as u64);
    let likely_cause = if !comparison.byte_identical {
        "verifier_exactness_failure"
    } else if mtp.accepted_draft_tokens == 0 && draft_in_top_k == 0 {
        "assistant_draft_distribution_mismatch"
    } else if mtp.acceptance_rate < 0.05 {
        "low_acceptance_real_workload"
    } else {
        "workload_dependent_acceptance"
    };
    RecordDiagnosis {
        likely_cause: likely_cause.to_owned(),
        draft_in_top_k_rate,
        mean_logit_margin,
        max_logit_margin,
        exactness_passed: comparison.byte_identical,
    }
}

fn blockers_for_records(records: &[Record]) -> Vec<String> {
    records
        .iter()
        .flat_map(|record| record.blockers.clone())
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
    let mismatch = (0..max_len)
        .find(|index| baseline.get(*index) != mtp.get(*index))
        .map(|index| TokenMismatch {
            index,
            baseline_token: baseline.get(index).copied(),
            mtp_token: mtp.get(index).copied(),
        });
    Comparison {
        byte_identical: false,
        first_mismatch: mismatch,
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
    options: &Options,
) -> Result<Vec<WorkloadRecord>, CliError> {
    if !options.workload_ids.is_empty() {
        let wanted = options
            .workload_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
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
    if let Some(max_workloads) = options.max_workloads {
        workloads.truncate(max_workloads);
    }
    if workloads.is_empty() {
        return Err(CliError::Runtime("no workloads selected".to_owned()));
    }
    Ok(workloads)
}

fn startup_blockers(options: &Options) -> Vec<String> {
    let mut blockers = Vec::new();
    if !options.model_path.exists() {
        blockers.push(format!(
            "target model path does not exist: {}",
            options.model_path.display()
        ));
    }
    if !options.assistant_model_path.exists() {
        blockers.push(format!(
            "assistant model path does not exist: {}",
            options.assistant_model_path.display()
        ));
    }
    if !options.python.exists() {
        blockers.push(format!(
            "python path does not exist: {}",
            options.python.display()
        ));
    }
    if env::var_os("GEMMA4D_USE_NATIVE_GRAPH").is_none() {
        blockers.push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for XR03".to_owned());
    }
    if env::var_os("GEMMA4D_REQUIRE_MLX").is_none() {
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for XR03".to_owned());
    }
    blockers
}

fn write_jsonl(path: &Path, records: &[Record]) -> Result<(), CliError> {
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
    out.push_str("# XR03 MTP Real-Context Diagnosis Report\n\n");
    out.push_str("## Summary\n\n| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Decision | `{}` |\n", summary.decision));
    out.push_str(&format!("| Status | `{}` |\n", summary.status));
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!("| Records | `{}` |\n", summary.record_count));
    out.push_str(&format!(
        "| Exact records | `{}` |\n",
        summary.exact_records
    ));
    out.push_str(&format!(
        "| Acceptance | `{}/{}` = `{:.3}` |\n",
        summary.acceptance_summary.accepted_draft_tokens,
        summary.acceptance_summary.attempted_draft_tokens,
        summary.acceptance_summary.acceptance_rate
    ));
    out.push_str(&format!(
        "| Draft in target top-k | `{}/{}` = `{:.3}` |\n",
        summary.acceptance_summary.draft_in_top_k_count,
        summary.acceptance_summary.draft_trace_count,
        summary.acceptance_summary.draft_in_top_k_rate
    ));
    out.push_str(&format!(
        "| Root cause classification | `{}` |\n\n",
        markdown_escape(&summary.root_cause_assessment.classification)
    ));

    out.push_str("## Artifact Compatibility\n\n| Field | Value |\n|---|---|\n");
    out.push_str(&format!(
        "| Target artifact SHA | `{}` |\n",
        markdown_escape(&summary.artifact_compatibility.target_local_artifact_sha256)
    ));
    out.push_str(&format!(
        "| Assistant artifact SHA | `{}` |\n",
        markdown_escape(
            &summary
                .artifact_compatibility
                .assistant_local_artifact_sha256
        )
    ));
    out.push_str(&format!(
        "| Assessment | `{}` |\n\n",
        markdown_escape(&summary.artifact_compatibility.assessment)
    ));

    out.push_str("## Records\n\n");
    out.push_str("| Workload | Family | Block | Exact | Accepted/Attempted | Top-k Hit Rate | Mean Margin | Verify ms | Peak GB | Cause |\n");
    out.push_str("|---|---|---:|---|---:|---:|---:|---:|---:|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | `{}` | {}/{} | {:.3} | {:.3} | {:.3} | {:.3} | `{}` |\n",
            markdown_escape(&record.workload_id),
            markdown_escape(&record.family),
            record.block_size,
            record.comparison.byte_identical,
            record.mtp.accepted_draft_tokens,
            record.mtp.attempted_draft_tokens,
            record.diagnosis.draft_in_top_k_rate,
            record.diagnosis.mean_logit_margin,
            record.mtp.verify_ms,
            record.mtp.peak_memory_gb,
            markdown_escape(&record.diagnosis.likely_cause)
        ));
    }

    out.push_str("\n## Fix Hypotheses\n\n");
    for item in &summary.fix_hypotheses {
        out.push_str(&format!(
            "{}. {} Payoff: `{}`. Risk: `{}`. Evidence: {}\n",
            item.rank,
            item.hypothesis,
            markdown_escape(&item.expected_payoff),
            markdown_escape(&item.risk),
            item.evidence
        ));
    }

    out.push_str("\n## Generated Files\n\n");
    for path in &summary.generated_files {
        out.push_str(&format!("- `{}`\n", markdown_escape(path)));
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR03 MTP Real-Context Diagnosis Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No blockers recorded.\n\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
        out.push('\n');
    }
    if !summary.failed_hypotheses.is_empty() {
        out.push_str("## Failed Hypotheses / Rejected Explanations\n\n");
        for hypothesis in &summary.failed_hypotheses {
            out.push_str(&format!("- {hypothesis}\n"));
        }
        out.push('\n');
    }
    out.push_str("## Fix Hypotheses\n\n");
    for item in &summary.fix_hypotheses {
        out.push_str(&format!("- rank {}: {}\n", item.rank, item.hypothesis));
    }
    out
}

fn render_decision(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR03 MTP Real-Context Diagnosis Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("XR03 is a diagnosis goal. It does not enable MTP by default and does not apply runtime fixes.\n\n");
    out.push_str("## Root Cause Assessment\n\n");
    out.push_str(&format!(
        "Classification: `{}`\n\n",
        markdown_escape(&summary.root_cause_assessment.classification)
    ));
    for evidence in &summary.root_cause_assessment.evidence {
        out.push_str(&format!("- {evidence}\n"));
    }
    out.push_str("\n## Fix Hypotheses\n\n");
    for item in &summary.fix_hypotheses {
        out.push_str(&format!(
            "- rank {}: {} (payoff `{}`, risk `{}`)\n",
            item.rank,
            item.hypothesis,
            markdown_escape(&item.expected_payoff),
            markdown_escape(&item.risk)
        ));
    }
    out.push_str("\n## Evidence\n\n");
    out.push_str(&format!("- Records: `{}`\n", summary.records_path));
    out.push_str(&format!("- Summary: `{}`\n", summary.summary_path));
    out.push_str(&format!("- Report: `{}`\n", summary.report_path));
    out.push_str(&format!("- Blockers: `{}`\n", summary.blockers_path));
    out
}

struct TokenizerHelper {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    backend: String,
    decode_cache: BTreeMap<i32, String>,
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
    if cmd == "shutdown":
        break
    if cmd == "encode":
        print(json.dumps({"ok": True, "ids": tokenizer.encode(request["text"])}, separators=(",", ":")), flush=True)
    elif cmd == "decode":
        print(json.dumps({"ok": True, "text": tokenizer.decode(request["ids"])}, separators=(",", ":")), flush=True)
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
            decode_cache: BTreeMap::new(),
        })
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

    fn decode(&mut self, ids: &[i32]) -> Result<String, CliError> {
        if ids.len() == 1
            && let Some(text) = self.decode_cache.get(&ids[0])
        {
            return Ok(text.clone());
        }
        let value = self.request(&serde_json::json!({"cmd":"decode","ids":ids}))?;
        let text = value
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| CliError::Runtime("tokenizer decode response missing text".to_owned()))?
            .to_owned();
        if ids.len() == 1 {
            self.decode_cache.insert(ids[0], text.clone());
        }
        Ok(text)
    }

    fn request(&mut self, request: &serde_json::Value) -> Result<serde_json::Value, CliError> {
        serde_json::to_writer(&mut self.stdin, request).map_err(|error| {
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

fn parse_usize_csv(value: &str) -> Result<Vec<usize>, CliError> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_positive_usize(part.trim(), "--block-sizes"))
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

fn usage() -> String {
    format!(
        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr03_mtp_real_context_diagnosis -- [--out-dir PATH] [--workloads PATH] [--model-path PATH] [--assistant-model-path PATH] [--python PATH] [--max-new-tokens N] [--block-sizes 1,2] [--workload-id ID] [--clear-workload-ids] [--max-workloads N]\n\ndefault out-dir: {DEFAULT_OUT_DIR}\ndefault workloads: {DEFAULT_WORKLOADS}"
    )
}

fn ratio(left: u64, right: u64) -> f64 {
    if right == 0 {
        0.0
    } else {
        left as f64 / right as f64
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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
    format!("xr03-{}-{}", now.as_secs(), now.subsec_nanos())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
