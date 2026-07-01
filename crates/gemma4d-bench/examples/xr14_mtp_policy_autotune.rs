use std::{
    cmp::Ordering,
    collections::BTreeMap,
    env, fs,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

const GOAL: &str = "XR14-mtp-policy-autotune";
const MODE: &str = "xr04_mtp_policy_replay";
const DEFAULT_SOURCE_SUMMARY: &str = "benchmarks/out/XR04-mtp-repair-and-autotune/summary.json";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR14-mtp-policy-autotune";
const DEFAULT_MIN_SPEEDUP_PERCENT: f64 = 5.0;
const DEFAULT_REGRESSION_GATE_PERCENT: f64 = 5.0;
const DEFAULT_ACCEPTANCE_THRESHOLD: f64 = 0.35;
const DEFAULT_MEMORY_CLIFF_GB: f64 = 14.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = Options::parse(env::args().skip(1))?;
    fs::create_dir_all(&options.out_dir)?;

    let records_path = options.out_dir.join("records.jsonl");
    let summary_path = options.out_dir.join("summary.json");
    let report_path = options.out_dir.join("report.md");
    let blockers_path = options.out_dir.join("blockers.md");
    let decision_path = options.out_dir.join("decision.md");

    let source_summary = load_source_summary(&options.source_summary)?;
    let source_records = build_workload_evidence(&source_summary)?;
    let policies = policy_specs(&options);
    let run_id = run_id();
    let generated_at_unix_seconds = unix_now();
    let git_sha =
        command_stdout("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let git_status_short =
        command_stdout("git", &["status", "--short"]).unwrap_or_else(|| "unknown".to_owned());
    let command = command_line();
    let source_summary_sha256 = file_sha256(&options.source_summary);

    let mut policy_records = Vec::new();
    for policy in &policies {
        for workload in source_records.values() {
            policy_records.push(build_policy_record(
                &options,
                &source_summary,
                &run_id,
                &git_sha,
                &git_status_short,
                &command,
                policy,
                workload,
            ));
        }
    }

    let aggregates = build_policy_aggregates(&policy_records);
    let hard_blockers = hard_blockers(&source_summary, &source_records);
    let failed_hypotheses = failed_hypotheses(&aggregates);
    let replay_limitations = replay_limitations(&source_summary);
    let recommendation = recommended_next_action(&aggregates);
    let decision = decision_for(&hard_blockers);
    let status = if hard_blockers.is_empty() {
        "completed"
    } else {
        "blocked"
    };

    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        mode: MODE.to_owned(),
        status: status.to_owned(),
        decision: decision.to_owned(),
        run_id,
        generated_at_unix_seconds,
        command,
        git_sha,
        git_status_short,
        source_summary_path: options.source_summary.display().to_string(),
        source_summary_sha256,
        source_goal: source_summary.goal.clone(),
        source_run_id: source_summary.run_id.clone(),
        source_git_sha: source_summary.git_sha.clone(),
        source_git_status_short: source_summary.git_status_short.clone(),
        source_decision: source_summary.decision.clone(),
        source_status: source_summary.status.clone(),
        source_mode: source_summary.mode.clone(),
        source_workloads_path: source_summary.workloads_path.clone(),
        source_generated_files: source_summary.generated_files.clone(),
        source_model_identity: source_summary.model_identity.clone(),
        source_assistant_identity: source_summary.assistant_identity.clone(),
        source_requested_max_new_tokens: source_summary.requested_max_new_tokens,
        source_block_sizes: source_summary.block_sizes.clone(),
        out_dir: options.out_dir.display().to_string(),
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
        min_speedup_percent: options.min_speedup_percent,
        regression_gate_percent: options.regression_gate_percent,
        acceptance_threshold: options.acceptance_threshold,
        memory_cliff_gb: options.memory_cliff_gb,
        selected_workloads: source_summary.selected_workloads.clone(),
        record_count: policy_records.len(),
        workload_count: source_records.len(),
        policies,
        policy_aggregates: aggregates,
        records: policy_records,
        hard_blockers,
        replay_limitations,
        failed_hypotheses,
        recommendation,
        measurement_notes: measurement_notes(),
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR14 MTP policy autotune replay: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());

    Ok(())
}

#[derive(Debug, Clone)]
struct Options {
    source_summary: PathBuf,
    out_dir: PathBuf,
    min_speedup_percent: f64,
    regression_gate_percent: f64,
    acceptance_threshold: f64,
    memory_cliff_gb: f64,
}

impl Options {
    fn parse<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut options = Self {
            source_summary: PathBuf::from(DEFAULT_SOURCE_SUMMARY),
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
            min_speedup_percent: DEFAULT_MIN_SPEEDUP_PERCENT,
            regression_gate_percent: DEFAULT_REGRESSION_GATE_PERCENT,
            acceptance_threshold: DEFAULT_ACCEPTANCE_THRESHOLD,
            memory_cliff_gb: DEFAULT_MEMORY_CLIFF_GB,
        };
        let mut args = args.into_iter().map(Into::into).peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--source-summary" => {
                    options.source_summary =
                        PathBuf::from(required_value(&mut args, "--source-summary")?)
                }
                "--out-dir" => {
                    options.out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?)
                }
                "--min-speedup-percent" => {
                    options.min_speedup_percent = parse_finite_nonnegative(
                        &required_value(&mut args, "--min-speedup-percent")?,
                        "--min-speedup-percent",
                    )?
                }
                "--regression-gate-percent" => {
                    options.regression_gate_percent = parse_finite_nonnegative(
                        &required_value(&mut args, "--regression-gate-percent")?,
                        "--regression-gate-percent",
                    )?
                }
                "--acceptance-threshold" => {
                    options.acceptance_threshold = parse_unit_interval(
                        &required_value(&mut args, "--acceptance-threshold")?,
                        "--acceptance-threshold",
                    )?
                }
                "--memory-cliff-gb" => {
                    options.memory_cliff_gb = parse_finite_positive(
                        &required_value(&mut args, "--memory-cliff-gb")?,
                        "--memory-cliff-gb",
                    )?
                }
                "-h" | "--help" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'\n{}", usage())),
            }
        }
        Ok(options)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct SourceSummary {
    goal: String,
    status: String,
    decision: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    mode: String,
    workloads_path: String,
    requested_max_new_tokens: usize,
    block_sizes: Vec<usize>,
    selected_workloads: Vec<SelectedWorkload>,
    generated_files: Vec<String>,
    blockers: Vec<String>,
    records: Vec<SourceRecord>,
    model_identity: Value,
    assistant_identity: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
struct SourceRecord {
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
    baseline: BaselineRun,
    mtp: MtpRun,
    comparison: Comparison,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct BaselineRun {
    generated_tokens: Vec<i32>,
    model_load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    peak_memory_gb: f64,
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct MtpRun {
    generated_tokens: Vec<i32>,
    model_load_ms: f64,
    drafter_load_ms: f64,
    prefill_ms: f64,
    draft_ms: f64,
    verify_ms: f64,
    total_ms: f64,
    attempted_draft_tokens: usize,
    accepted_draft_tokens: usize,
    acceptance_rate: f64,
    accepted_tokens_per_verify: f64,
    target_verify_passes: usize,
    rollback_count: usize,
    peak_memory_gb: f64,
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct Comparison {
    byte_identical: bool,
}

#[derive(Debug, Clone)]
struct WorkloadEvidence {
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    deterministic_seed: u64,
    workload_max_new_tokens: usize,
    max_new_tokens: usize,
    baseline: CandidateObservation,
    candidates: BTreeMap<usize, CandidateObservation>,
}

#[derive(Debug, Clone)]
struct CandidateObservation {
    variant: String,
    block_size: Option<usize>,
    decode_phase_ms: f64,
    exact: bool,
    generated_tokens: usize,
    model_load_ms: f64,
    drafter_load_ms: f64,
    prefill_ms: f64,
    total_ms: f64,
    draft_ms: f64,
    verify_ms: f64,
    peak_memory_gb: f64,
    active_kv_bytes: u64,
    attempted_draft_tokens: usize,
    accepted_draft_tokens: usize,
    acceptance_rate: f64,
    accepted_tokens_per_verify: f64,
    target_verify_passes: usize,
    rollback_count: usize,
    source_blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicySpec {
    name: String,
    kind: String,
    description: String,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyRecord {
    schema_version: u32,
    goal: String,
    mode: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    source_run_id: String,
    source_git_sha: String,
    source_git_status_short: String,
    source_decision: String,
    source_status: String,
    policy_name: String,
    policy_kind: String,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    deterministic_seed: u64,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    workload_max_new_tokens: usize,
    max_new_tokens: usize,
    selected_variant: String,
    selected_block_size: Option<usize>,
    selected_mtp_enabled: bool,
    baseline_decode_ms: f64,
    selected_decode_phase_ms: f64,
    delta_ms: f64,
    speedup_percent: f64,
    exact: bool,
    status: String,
    decision_reason: String,
    baseline_generated_tokens: usize,
    selected_generated_tokens: usize,
    baseline_model_load_ms: f64,
    selected_model_load_ms: f64,
    selected_drafter_load_ms: f64,
    baseline_prefill_ms: f64,
    selected_prefill_ms: f64,
    selected_draft_ms: f64,
    selected_verify_ms: f64,
    baseline_total_ms: f64,
    selected_total_ms: f64,
    baseline_peak_memory_gb: f64,
    selected_peak_memory_gb: f64,
    memory_delta_gb: f64,
    baseline_active_kv_bytes: u64,
    selected_active_kv_bytes: u64,
    active_kv_delta_bytes: i64,
    accepted_draft_tokens: usize,
    attempted_draft_tokens: usize,
    acceptance_rate: f64,
    accepted_tokens_per_verify: f64,
    target_verify_passes: usize,
    rollback_count: usize,
    source_blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyAggregate {
    policy_name: String,
    policy_kind: String,
    decision: String,
    workload_count: usize,
    selected_mtp_workloads: usize,
    exact_workloads: usize,
    regressed_workloads: usize,
    total_baseline_decode_ms: f64,
    total_selected_decode_phase_ms: f64,
    total_delta_ms: f64,
    aggregate_speedup_percent: f64,
    max_peak_memory_gb: f64,
    max_memory_delta_gb: f64,
    total_accepted_draft_tokens: usize,
    total_attempted_draft_tokens: usize,
    weighted_acceptance_rate: f64,
    selected_workloads: Vec<String>,
    regressed_workload_ids: Vec<String>,
    reasons: Vec<String>,
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
    source_summary_path: String,
    source_summary_sha256: String,
    source_goal: String,
    source_run_id: String,
    source_git_sha: String,
    source_git_status_short: String,
    source_decision: String,
    source_status: String,
    source_mode: String,
    source_workloads_path: String,
    source_generated_files: Vec<String>,
    source_model_identity: Value,
    source_assistant_identity: Value,
    source_requested_max_new_tokens: usize,
    source_block_sizes: Vec<usize>,
    out_dir: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    generated_files: Vec<String>,
    min_speedup_percent: f64,
    regression_gate_percent: f64,
    acceptance_threshold: f64,
    memory_cliff_gb: f64,
    selected_workloads: Vec<SelectedWorkload>,
    record_count: usize,
    workload_count: usize,
    policies: Vec<PolicySpec>,
    policy_aggregates: Vec<PolicyAggregate>,
    records: Vec<PolicyRecord>,
    hard_blockers: Vec<String>,
    replay_limitations: Vec<String>,
    failed_hypotheses: Vec<String>,
    recommendation: String,
    measurement_notes: Vec<String>,
}

fn load_source_summary(path: &Path) -> Result<SourceSummary, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn build_workload_evidence(
    source: &SourceSummary,
) -> Result<BTreeMap<String, WorkloadEvidence>, String> {
    if source.records.is_empty() {
        return Err("source summary has no records".to_owned());
    }

    let mut workloads = BTreeMap::new();
    for record in &source.records {
        let baseline = CandidateObservation {
            variant: "baseline_native_no_mtp".to_owned(),
            block_size: None,
            decode_phase_ms: record.baseline.decode_ms,
            exact: true,
            generated_tokens: record.baseline.generated_tokens.len(),
            model_load_ms: record.baseline.model_load_ms,
            drafter_load_ms: 0.0,
            prefill_ms: record.baseline.prefill_ms,
            total_ms: record.baseline.total_ms,
            draft_ms: 0.0,
            verify_ms: 0.0,
            peak_memory_gb: record.baseline.peak_memory_gb,
            active_kv_bytes: record.baseline.active_kv_bytes,
            attempted_draft_tokens: 0,
            accepted_draft_tokens: 0,
            acceptance_rate: 0.0,
            accepted_tokens_per_verify: 0.0,
            target_verify_passes: 0,
            rollback_count: 0,
            source_blockers: Vec::new(),
        };
        let entry = workloads
            .entry(record.workload_id.clone())
            .or_insert_with(|| WorkloadEvidence {
                workload_id: record.workload_id.clone(),
                family: record.family.clone(),
                prompt_path: record.prompt_path.clone(),
                prompt_sha256: record.prompt_sha256.clone(),
                target_context_tokens: record.target_context_tokens,
                actual_context_tokens: record.actual_context_tokens,
                deterministic_seed: record.deterministic_seed,
                workload_max_new_tokens: record.workload_max_new_tokens,
                max_new_tokens: record.max_new_tokens,
                baseline: baseline.clone(),
                candidates: BTreeMap::new(),
            });
        validate_repeated_baseline(entry, record)?;
        entry.candidates.insert(
            record.block_size,
            CandidateObservation {
                variant: format!("mtp_block_{}", record.block_size),
                block_size: Some(record.block_size),
                decode_phase_ms: record.mtp.draft_ms + record.mtp.verify_ms,
                exact: record.comparison.byte_identical,
                generated_tokens: record.mtp.generated_tokens.len(),
                model_load_ms: record.mtp.model_load_ms,
                drafter_load_ms: record.mtp.drafter_load_ms,
                prefill_ms: record.mtp.prefill_ms,
                total_ms: record.mtp.total_ms,
                draft_ms: record.mtp.draft_ms,
                verify_ms: record.mtp.verify_ms,
                peak_memory_gb: record.mtp.peak_memory_gb,
                active_kv_bytes: record.mtp.active_kv_bytes,
                attempted_draft_tokens: record.mtp.attempted_draft_tokens,
                accepted_draft_tokens: record.mtp.accepted_draft_tokens,
                acceptance_rate: record.mtp.acceptance_rate,
                accepted_tokens_per_verify: record.mtp.accepted_tokens_per_verify,
                target_verify_passes: record.mtp.target_verify_passes,
                rollback_count: record.mtp.rollback_count,
                source_blockers: record.blockers.clone(),
            },
        );
    }
    Ok(workloads)
}

fn validate_repeated_baseline(
    evidence: &WorkloadEvidence,
    record: &SourceRecord,
) -> Result<(), String> {
    if evidence.family != record.family
        || evidence.prompt_sha256 != record.prompt_sha256
        || evidence.deterministic_seed != record.deterministic_seed
        || evidence.actual_context_tokens != record.actual_context_tokens
        || evidence.max_new_tokens != record.max_new_tokens
    {
        return Err(format!(
            "{} has inconsistent workload metadata across source block records",
            evidence.workload_id
        ));
    }
    if !approx_eq(evidence.baseline.decode_phase_ms, record.baseline.decode_ms) {
        return Err(format!(
            "{} has inconsistent baseline decode_ms across source block records",
            evidence.workload_id
        ));
    }
    Ok(())
}

fn policy_specs(options: &Options) -> Vec<PolicySpec> {
    vec![
        PolicySpec {
            name: "disabled_baseline".to_owned(),
            kind: "baseline".to_owned(),
            description: "Never enable MTP; replay the native non-MTP decode baseline.".to_owned(),
        },
        PolicySpec {
            name: "fixed_block_1".to_owned(),
            kind: "fixed_block".to_owned(),
            description: "Always use MTP block size 1 when present in the source evidence."
                .to_owned(),
        },
        PolicySpec {
            name: "fixed_block_2".to_owned(),
            kind: "fixed_block".to_owned(),
            description: "Always use MTP block size 2 when present in the source evidence."
                .to_owned(),
        },
        PolicySpec {
            name: format!(
                "acceptance_threshold_{:.0}pct",
                options.acceptance_threshold * 100.0
            ),
            kind: "acceptance_threshold".to_owned(),
            description: format!(
                "Choose the highest-acceptance exact MTP block when acceptance is at least {:.3}.",
                options.acceptance_threshold
            ),
        },
        PolicySpec {
            name: format!(
                "net_latency_guarded_{:.0}pct",
                options.min_speedup_percent
            ),
            kind: "net_latency_guarded".to_owned(),
            description: format!(
                "Choose the fastest exact MTP block only when draft_ms + verify_ms beats baseline decode_ms by at least {:.1}%.",
                options.min_speedup_percent
            ),
        },
        PolicySpec {
            name: "oracle_fastest_exact".to_owned(),
            kind: "oracle_upper_bound".to_owned(),
            description:
                "Choose the fastest exact option per workload; this is an upper bound, not a deployable policy."
                    .to_owned(),
        },
    ]
}

fn build_policy_record(
    options: &Options,
    source: &SourceSummary,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
    policy: &PolicySpec,
    workload: &WorkloadEvidence,
) -> PolicyRecord {
    let selection = select_candidate(policy, workload, options);
    let baseline = &workload.baseline;
    let candidate = selection.candidate;
    let delta_ms = candidate.decode_phase_ms - baseline.decode_phase_ms;
    let speedup_percent = speedup_percent(baseline.decode_phase_ms, candidate.decode_phase_ms);
    let memory_delta_gb = candidate.peak_memory_gb - baseline.peak_memory_gb;
    let active_kv_delta_bytes =
        candidate.active_kv_bytes as i128 - baseline.active_kv_bytes as i128;
    let exact = !selection.mtp_enabled || candidate.exact;
    let status = if !exact {
        "failed_exactness"
    } else if candidate.peak_memory_gb > options.memory_cliff_gb {
        "failed_memory"
    } else if candidate.decode_phase_ms
        > baseline.decode_phase_ms * (1.0 + options.regression_gate_percent / 100.0)
    {
        "regressed"
    } else {
        "passed"
    };

    PolicyRecord {
        schema_version: 1,
        goal: GOAL.to_owned(),
        mode: MODE.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        command: command.to_owned(),
        source_run_id: source.run_id.clone(),
        source_git_sha: source.git_sha.clone(),
        source_git_status_short: source.git_status_short.clone(),
        source_decision: source.decision.clone(),
        source_status: source.status.clone(),
        policy_name: policy.name.clone(),
        policy_kind: policy.kind.clone(),
        workload_id: workload.workload_id.clone(),
        family: workload.family.clone(),
        prompt_path: workload.prompt_path.clone(),
        prompt_sha256: workload.prompt_sha256.clone(),
        deterministic_seed: workload.deterministic_seed,
        target_context_tokens: workload.target_context_tokens,
        actual_context_tokens: workload.actual_context_tokens,
        workload_max_new_tokens: workload.workload_max_new_tokens,
        max_new_tokens: workload.max_new_tokens,
        selected_variant: candidate.variant.clone(),
        selected_block_size: candidate.block_size,
        selected_mtp_enabled: selection.mtp_enabled,
        baseline_decode_ms: baseline.decode_phase_ms,
        selected_decode_phase_ms: candidate.decode_phase_ms,
        delta_ms,
        speedup_percent,
        exact,
        status: status.to_owned(),
        decision_reason: selection.reason,
        baseline_generated_tokens: baseline.generated_tokens,
        selected_generated_tokens: candidate.generated_tokens,
        baseline_model_load_ms: baseline.model_load_ms,
        selected_model_load_ms: candidate.model_load_ms,
        selected_drafter_load_ms: if selection.mtp_enabled {
            candidate.drafter_load_ms
        } else {
            0.0
        },
        baseline_prefill_ms: baseline.prefill_ms,
        selected_prefill_ms: candidate.prefill_ms,
        selected_draft_ms: candidate.draft_ms,
        selected_verify_ms: candidate.verify_ms,
        baseline_total_ms: baseline.total_ms,
        selected_total_ms: candidate.total_ms,
        baseline_peak_memory_gb: baseline.peak_memory_gb,
        selected_peak_memory_gb: candidate.peak_memory_gb,
        memory_delta_gb,
        baseline_active_kv_bytes: baseline.active_kv_bytes,
        selected_active_kv_bytes: candidate.active_kv_bytes,
        active_kv_delta_bytes: active_kv_delta_bytes.clamp(i64::MIN as i128, i64::MAX as i128)
            as i64,
        accepted_draft_tokens: candidate.accepted_draft_tokens,
        attempted_draft_tokens: candidate.attempted_draft_tokens,
        acceptance_rate: candidate.acceptance_rate,
        accepted_tokens_per_verify: candidate.accepted_tokens_per_verify,
        target_verify_passes: candidate.target_verify_passes,
        rollback_count: candidate.rollback_count,
        source_blockers: candidate.source_blockers.clone(),
    }
}

#[derive(Debug, Clone)]
struct Selection<'a> {
    candidate: &'a CandidateObservation,
    mtp_enabled: bool,
    reason: String,
}

fn select_candidate<'a>(
    policy: &PolicySpec,
    workload: &'a WorkloadEvidence,
    options: &Options,
) -> Selection<'a> {
    match policy.kind.as_str() {
        "baseline" => Selection {
            candidate: &workload.baseline,
            mtp_enabled: false,
            reason: "baseline policy keeps MTP disabled".to_owned(),
        },
        "fixed_block" if policy.name.ends_with("_1") => fixed_block_selection(workload, 1),
        "fixed_block" if policy.name.ends_with("_2") => fixed_block_selection(workload, 2),
        "acceptance_threshold" => acceptance_threshold_selection(workload, options),
        "net_latency_guarded" => net_latency_guarded_selection(workload, options),
        "oracle_upper_bound" => oracle_selection(workload),
        _ => Selection {
            candidate: &workload.baseline,
            mtp_enabled: false,
            reason: format!("unknown policy kind {}; kept baseline", policy.kind),
        },
    }
}

fn fixed_block_selection(workload: &WorkloadEvidence, block_size: usize) -> Selection<'_> {
    if let Some(candidate) = workload.candidates.get(&block_size) {
        Selection {
            candidate,
            mtp_enabled: true,
            reason: format!("fixed block-size policy selected block {block_size}"),
        }
    } else {
        Selection {
            candidate: &workload.baseline,
            mtp_enabled: false,
            reason: format!("source evidence has no block {block_size}; kept baseline"),
        }
    }
}

fn acceptance_threshold_selection<'a>(
    workload: &'a WorkloadEvidence,
    options: &Options,
) -> Selection<'a> {
    let selected = workload
        .candidates
        .values()
        .filter(|candidate| candidate.exact)
        .filter(|candidate| candidate.acceptance_rate >= options.acceptance_threshold)
        .max_by(|left, right| {
            left.acceptance_rate
                .partial_cmp(&right.acceptance_rate)
                .unwrap_or(Ordering::Equal)
                .then_with(|| right.block_size.cmp(&left.block_size))
        });
    if let Some(candidate) = selected {
        Selection {
            candidate,
            mtp_enabled: true,
            reason: format!(
                "selected highest acceptance {:.3} at block {}",
                candidate.acceptance_rate,
                candidate.block_size.unwrap_or_default()
            ),
        }
    } else {
        Selection {
            candidate: &workload.baseline,
            mtp_enabled: false,
            reason: format!(
                "no exact MTP block met acceptance threshold {:.3}",
                options.acceptance_threshold
            ),
        }
    }
}

fn net_latency_guarded_selection<'a>(
    workload: &'a WorkloadEvidence,
    options: &Options,
) -> Selection<'a> {
    let selected = fastest_exact_mtp(workload).filter(|candidate| {
        speedup_percent(workload.baseline.decode_phase_ms, candidate.decode_phase_ms)
            >= options.min_speedup_percent
            && candidate.peak_memory_gb <= options.memory_cliff_gb
    });
    if let Some(candidate) = selected {
        Selection {
            candidate,
            mtp_enabled: true,
            reason: format!(
                "selected fastest exact block {} with {:.3}% replay speedup",
                candidate.block_size.unwrap_or_default(),
                speedup_percent(workload.baseline.decode_phase_ms, candidate.decode_phase_ms)
            ),
        }
    } else {
        Selection {
            candidate: &workload.baseline,
            mtp_enabled: false,
            reason: format!(
                "no exact MTP block beat baseline by {:.1}% under memory gate",
                options.min_speedup_percent
            ),
        }
    }
}

fn oracle_selection(workload: &WorkloadEvidence) -> Selection<'_> {
    let selected = fastest_exact_mtp(workload)
        .filter(|candidate| candidate.decode_phase_ms < workload.baseline.decode_phase_ms);
    if let Some(candidate) = selected {
        Selection {
            candidate,
            mtp_enabled: true,
            reason: format!(
                "oracle selected fastest exact block {}",
                candidate.block_size.unwrap_or_default()
            ),
        }
    } else {
        Selection {
            candidate: &workload.baseline,
            mtp_enabled: false,
            reason: "oracle kept baseline because no exact MTP block was faster".to_owned(),
        }
    }
}

fn fastest_exact_mtp(workload: &WorkloadEvidence) -> Option<&CandidateObservation> {
    workload
        .candidates
        .values()
        .filter(|candidate| candidate.exact)
        .min_by(|left, right| {
            left.decode_phase_ms
                .partial_cmp(&right.decode_phase_ms)
                .unwrap_or(Ordering::Equal)
        })
}

fn build_policy_aggregates(records: &[PolicyRecord]) -> Vec<PolicyAggregate> {
    let mut grouped: BTreeMap<String, Vec<&PolicyRecord>> = BTreeMap::new();
    for record in records {
        grouped
            .entry(record.policy_name.clone())
            .or_default()
            .push(record);
    }

    grouped
        .into_iter()
        .map(|(policy_name, records)| {
            let policy_kind = records
                .first()
                .map(|record| record.policy_kind.clone())
                .unwrap_or_default();
            let total_baseline_decode_ms =
                records.iter().map(|record| record.baseline_decode_ms).sum();
            let total_selected_decode_phase_ms = records
                .iter()
                .map(|record| record.selected_decode_phase_ms)
                .sum();
            let total_delta_ms = total_selected_decode_phase_ms - total_baseline_decode_ms;
            let aggregate_speedup_percent =
                speedup_percent(total_baseline_decode_ms, total_selected_decode_phase_ms);
            let regressed_workload_ids = records
                .iter()
                .filter(|record| record.status == "regressed")
                .map(|record| record.workload_id.clone())
                .collect::<Vec<_>>();
            let total_accepted_draft_tokens = records
                .iter()
                .map(|record| record.accepted_draft_tokens)
                .sum();
            let total_attempted_draft_tokens = records
                .iter()
                .map(|record| record.attempted_draft_tokens)
                .sum();
            let weighted_acceptance_rate = if total_attempted_draft_tokens == 0 {
                0.0
            } else {
                total_accepted_draft_tokens as f64 / total_attempted_draft_tokens as f64
            };
            let selected_workloads = records
                .iter()
                .filter(|record| record.selected_mtp_enabled)
                .map(|record| {
                    format!(
                        "{}:{}",
                        record.workload_id,
                        record.selected_block_size.unwrap_or_default()
                    )
                })
                .collect::<Vec<_>>();
            let selected_mtp_workloads = selected_workloads.len();
            let exact_workloads = records.iter().filter(|record| record.exact).count();
            let regressed_workloads = regressed_workload_ids.len();
            let max_peak_memory_gb = records
                .iter()
                .map(|record| record.selected_peak_memory_gb)
                .fold(0.0, f64::max);
            let max_memory_delta_gb = records
                .iter()
                .map(|record| record.memory_delta_gb)
                .fold(f64::NEG_INFINITY, f64::max)
                .max(0.0);
            let reasons = aggregate_reasons(
                &policy_kind,
                aggregate_speedup_percent,
                regressed_workloads,
                selected_mtp_workloads,
                &regressed_workload_ids,
            );
            let decision = aggregate_decision(
                &policy_kind,
                aggregate_speedup_percent,
                regressed_workloads,
                selected_mtp_workloads,
            );

            PolicyAggregate {
                policy_name,
                policy_kind,
                decision,
                workload_count: records.len(),
                selected_mtp_workloads,
                exact_workloads,
                regressed_workloads,
                total_baseline_decode_ms,
                total_selected_decode_phase_ms,
                total_delta_ms,
                aggregate_speedup_percent,
                max_peak_memory_gb,
                max_memory_delta_gb,
                total_accepted_draft_tokens,
                total_attempted_draft_tokens,
                weighted_acceptance_rate,
                selected_workloads,
                regressed_workload_ids,
                reasons,
            }
        })
        .collect()
}

fn aggregate_decision(
    policy_kind: &str,
    aggregate_speedup_percent: f64,
    regressed_workloads: usize,
    selected_mtp_workloads: usize,
) -> String {
    if policy_kind == "baseline" {
        "baseline".to_owned()
    } else if regressed_workloads > 0 || aggregate_speedup_percent <= 0.0 {
        "reject_candidate".to_owned()
    } else if selected_mtp_workloads == 0 {
        "reject_candidate".to_owned()
    } else {
        "needs_more_data".to_owned()
    }
}

fn aggregate_reasons(
    policy_kind: &str,
    aggregate_speedup_percent: f64,
    regressed_workloads: usize,
    selected_mtp_workloads: usize,
    regressed_workload_ids: &[String],
) -> Vec<String> {
    let mut reasons = Vec::new();
    if policy_kind == "baseline" {
        reasons.push("baseline reference, not a candidate".to_owned());
    }
    if selected_mtp_workloads == 0 && policy_kind != "baseline" {
        reasons.push("policy selected no MTP workloads".to_owned());
    }
    if regressed_workloads > 0 {
        reasons.push(format!(
            "regressed {} workload(s): {}",
            regressed_workloads,
            regressed_workload_ids.join(", ")
        ));
    }
    if aggregate_speedup_percent > 0.0 {
        reasons.push(format!(
            "aggregate replay speedup was {:.3}%",
            aggregate_speedup_percent
        ));
    } else if policy_kind != "baseline" {
        reasons.push(format!(
            "aggregate replay speedup was {:.3}%",
            aggregate_speedup_percent
        ));
    }
    if policy_kind == "net_latency_guarded" || policy_kind == "oracle_upper_bound" {
        reasons.push(
            "selection is calibrated on XR04 replay data and needs holdout variance before runtime use"
                .to_owned(),
        );
    }
    reasons
}

fn hard_blockers(
    source: &SourceSummary,
    workloads: &BTreeMap<String, WorkloadEvidence>,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if !source.blockers.is_empty() {
        blockers.push(format!(
            "source summary contains blockers: {}",
            source.blockers.join("; ")
        ));
    }
    if source.records.is_empty() {
        blockers.push("source summary contains no records".to_owned());
    }
    for workload in workloads.values() {
        if workload.candidates.is_empty() {
            blockers.push(format!(
                "{} has no MTP candidate records",
                workload.workload_id
            ));
        }
        let exact_count = workload
            .candidates
            .values()
            .filter(|candidate| candidate.exact)
            .count();
        if exact_count != workload.candidates.len() {
            blockers.push(format!(
                "{} has non-exact MTP candidate records; speed policy replay cannot claim those candidates",
                workload.workload_id
            ));
        }
    }
    blockers.sort();
    blockers.dedup();
    blockers
}

fn failed_hypotheses(aggregates: &[PolicyAggregate]) -> Vec<String> {
    let mut failed = Vec::new();
    for aggregate in aggregates {
        if aggregate.decision == "reject_candidate" {
            failed.push(format!(
                "{} rejected: {}",
                aggregate.policy_name,
                aggregate.reasons.join("; ")
            ));
        }
    }
    failed
}

fn replay_limitations(source: &SourceSummary) -> Vec<String> {
    let mut limitations = vec![
        "This is a replay over XR04 root records, not a fresh model execution.".to_owned(),
        "XR04 contains one root run per workload/block and has no variance trials.".to_owned(),
        "Selection policies are calibrated on the same records they are evaluated on.".to_owned(),
        "XR04 incremental trace records only top-1 target tokens, so rank/top-k drafter claims remain out of scope.".to_owned(),
        "Server path, adapters, compressed active KV, sampling, and MTP block sizes above 2 remain out of scope.".to_owned(),
    ];
    if source.decision != "accept_candidate" {
        limitations.push(format!(
            "Source decision was {}, so replay cannot recommend MTP runtime policy.",
            source.decision
        ));
    }
    limitations
}

fn recommended_next_action(aggregates: &[PolicyAggregate]) -> String {
    let guarded = aggregates
        .iter()
        .find(|aggregate| aggregate.policy_kind == "net_latency_guarded");
    match guarded {
        Some(aggregate)
            if aggregate.decision == "needs_more_data" && aggregate.selected_mtp_workloads > 0 =>
        {
            format!(
                "Run XR14-mtp-policy-variance-ab with the latency-guarded policy as the candidate; replay selected {} workload/block pair(s) and showed {:.3}% aggregate decode-phase speedup, but no holdout variance exists.",
                aggregate.selected_mtp_workloads, aggregate.aggregate_speedup_percent
            )
        }
        Some(aggregate) => format!(
            "Do not run a runtime policy change yet; latency-guarded replay decision was {}.",
            aggregate.decision
        ),
        None => "Do not run a runtime policy change yet; latency-guarded policy was not evaluated."
            .to_owned(),
    }
}

fn decision_for(hard_blockers: &[String]) -> String {
    if hard_blockers.is_empty() {
        "needs_more_data".to_owned()
    } else {
        "blocked_with_evidence".to_owned()
    }
}

fn measurement_notes() -> Vec<String> {
    vec![
        "decode-phase comparison uses baseline.decode_ms versus mtp.draft_ms + mtp.verify_ms from XR04 records".to_owned(),
        "model load, drafter load, and prefill timings are preserved in records but excluded from policy speed decisions".to_owned(),
        "deterministic workload seeds and tokenizer-measured context lengths are copied from XR04 selected_workloads and records".to_owned(),
        "MTP remains opt-in because this replay has no fresh variance trials or holdout workloads".to_owned(),
    ]
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR14 MTP Policy Autotune Replay\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Decision | `{}` |\n", summary.decision));
    out.push_str(&format!("| Status | `{}` |\n", summary.status));
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!("| Source run | `{}` |\n", summary.source_run_id));
    out.push_str(&format!(
        "| Source summary SHA-256 | `{}` |\n",
        summary.source_summary_sha256
    ));
    out.push_str(&format!("| Workloads | `{}` |\n", summary.workload_count));
    out.push_str(&format!(
        "| Source max new tokens | `{}` |\n",
        summary.source_requested_max_new_tokens
    ));
    out.push_str(&format!(
        "| Block sizes | `{}` |\n\n",
        join_usize(&summary.source_block_sizes)
    ));

    out.push_str("## Policy Results\n\n");
    out.push_str("| Policy | Decision | MTP selections | Baseline decode ms | Selected decode ms | Speedup % | Regressions | Weighted acceptance |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|\n");
    for aggregate in &summary.policy_aggregates {
        out.push_str(&format!(
            "| `{}` | `{}` | {} | {:.3} | {:.3} | {:.3} | {} | {:.3} |\n",
            aggregate.policy_name,
            aggregate.decision,
            aggregate.selected_mtp_workloads,
            aggregate.total_baseline_decode_ms,
            aggregate.total_selected_decode_phase_ms,
            aggregate.aggregate_speedup_percent,
            aggregate.regressed_workloads,
            aggregate.weighted_acceptance_rate,
        ));
    }
    out.push('\n');

    out.push_str("## Selected Workload Decisions\n\n");
    out.push_str("| Policy | Workload | Family | Selected | Baseline ms | Selected ms | Speedup % | Acceptance | Status | Reason |\n");
    out.push_str("|---|---|---|---|---:|---:|---:|---:|---|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | {:.3} | {:.3} | {:.3} | {:.3} | `{}` | {} |\n",
            record.policy_name,
            record.workload_id,
            record.family,
            record.selected_variant,
            record.baseline_decode_ms,
            record.selected_decode_phase_ms,
            record.speedup_percent,
            record.acceptance_rate,
            record.status,
            escape_md(&record.decision_reason),
        ));
    }
    out.push('\n');

    out.push_str("## Failed Hypotheses\n\n");
    if summary.failed_hypotheses.is_empty() {
        out.push_str("- None.\n\n");
    } else {
        for item in &summary.failed_hypotheses {
            out.push_str(&format!("- {}\n", escape_md(item)));
        }
        out.push('\n');
    }

    out.push_str("## Limitations\n\n");
    for item in &summary.replay_limitations {
        out.push_str(&format!("- {}\n", escape_md(item)));
    }
    out.push('\n');

    out.push_str("## Recommendation\n\n");
    out.push_str(&summary.recommendation);
    out.push('\n');
    out
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR14 Blockers\n\n");
    if summary.hard_blockers.is_empty() {
        out.push_str("No hard replay blockers were recorded.\n\n");
    } else {
        out.push_str("## Hard Blockers\n\n");
        for blocker in &summary.hard_blockers {
            out.push_str(&format!("- {}\n", escape_md(blocker)));
        }
        out.push('\n');
    }

    out.push_str("## Decision Blockers\n\n");
    for limitation in &summary.replay_limitations {
        out.push_str(&format!("- {}\n", escape_md(limitation)));
    }
    out.push('\n');

    out.push_str("## Failed Hypotheses\n\n");
    if summary.failed_hypotheses.is_empty() {
        out.push_str("- None.\n");
    } else {
        for hypothesis in &summary.failed_hypotheses {
            out.push_str(&format!("- {}\n", escape_md(hypothesis)));
        }
    }
    out
}

fn render_decision(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR14 Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str("MTP remains opt-in. This replay found that fixed block-size and acceptance-only policies are not safe enough for a runtime default because high acceptance can still regress net decode latency.\n\n");
    out.push_str(&format!("{}\n\n", summary.recommendation));
    out.push_str("Required next evidence before a runtime policy change:\n\n");
    out.push_str("- Fresh native non-MTP versus native MTP variance run on holdout real-context workloads.\n");
    out.push_str(
        "- Exact byte/token parity at temperature 0 for every selected MTP workload/block.\n",
    );
    out.push_str("- Memory under the tiny16 gate and no non-selected workload regression over the policy gate.\n");
    out.push_str("- Restored top-k trace depth before making rank/top-k acceptance claims.\n");
    out
}

fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn run_id() -> String {
    format!("xr14-{}", unix_now())
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

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn required_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .filter(|value| !value.starts_with("--"))
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_finite_positive(value: &str, flag: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("{flag} must be a number: {error}"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err(format!("{flag} must be finite and greater than zero"))
    }
}

fn parse_finite_nonnegative(value: &str, flag: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("{flag} must be a number: {error}"))?;
    if parsed.is_finite() && parsed >= 0.0 {
        Ok(parsed)
    } else {
        Err(format!("{flag} must be finite and nonnegative"))
    }
}

fn parse_unit_interval(value: &str, flag: &str) -> Result<f64, String> {
    let parsed = parse_finite_nonnegative(value, flag)?;
    if parsed <= 1.0 {
        Ok(parsed)
    } else {
        Err(format!("{flag} must be between 0 and 1"))
    }
}

fn usage() -> &'static str {
    "usage: cargo run -p gemma4d-bench --example xr14_mtp_policy_autotune -- [--source-summary PATH] [--out-dir PATH] [--min-speedup-percent N] [--regression-gate-percent N] [--acceptance-threshold 0..1] [--memory-cliff-gb N]"
}

fn speedup_percent(baseline_ms: f64, candidate_ms: f64) -> f64 {
    if baseline_ms <= 0.0 {
        0.0
    } else {
        (baseline_ms - candidate_ms) / baseline_ms * 100.0
    }
}

fn approx_eq(left: f64, right: f64) -> bool {
    (left - right).abs() <= 0.000_001
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn escape_md(input: &str) -> String {
    input.replace('|', "\\|").replace('\n', " ")
}

fn file_sha256(path: &Path) -> String {
    command_stdout("shasum", &["-a", "256", &path.display().to_string()])
        .and_then(|line| line.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "unavailable".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_threshold_can_select_net_slow_candidate() {
        let options = test_options();
        let policy = PolicySpec {
            name: "acceptance_threshold_35pct".to_owned(),
            kind: "acceptance_threshold".to_owned(),
            description: String::new(),
        };
        let workload = synthetic_workload(vec![candidate(1, 125.0, 0.80, true)]);

        let selection = select_candidate(&policy, &workload, &options);

        assert!(selection.mtp_enabled);
        assert_eq!(selection.candidate.block_size, Some(1));
        assert!(selection.candidate.decode_phase_ms > workload.baseline.decode_phase_ms);
    }

    #[test]
    fn net_latency_guard_can_select_low_acceptance_fast_candidate() {
        let options = test_options();
        let policy = PolicySpec {
            name: "net_latency_guarded_5pct".to_owned(),
            kind: "net_latency_guarded".to_owned(),
            description: String::new(),
        };
        let workload = synthetic_workload(vec![candidate(1, 75.0, 0.25, true)]);

        let selection = select_candidate(&policy, &workload, &options);

        assert!(selection.mtp_enabled);
        assert_eq!(selection.candidate.block_size, Some(1));
    }

    #[test]
    fn aggregate_rejects_policy_with_regression() {
        let mut record = synthetic_policy_record("fixed_block_1", "fixed_block", 125.0);
        record.status = "regressed".to_owned();

        let aggregates = build_policy_aggregates(&[record]);

        assert_eq!(aggregates[0].decision, "reject_candidate");
        assert_eq!(aggregates[0].regressed_workloads, 1);
    }

    fn test_options() -> Options {
        Options {
            source_summary: PathBuf::from(DEFAULT_SOURCE_SUMMARY),
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
            min_speedup_percent: DEFAULT_MIN_SPEEDUP_PERCENT,
            regression_gate_percent: DEFAULT_REGRESSION_GATE_PERCENT,
            acceptance_threshold: DEFAULT_ACCEPTANCE_THRESHOLD,
            memory_cliff_gb: DEFAULT_MEMORY_CLIFF_GB,
        }
    }

    fn synthetic_workload(candidates: Vec<CandidateObservation>) -> WorkloadEvidence {
        WorkloadEvidence {
            workload_id: "workload".to_owned(),
            family: "family".to_owned(),
            prompt_path: "prompt.txt".to_owned(),
            prompt_sha256: "sha".to_owned(),
            target_context_tokens: 1024,
            actual_context_tokens: 1024,
            deterministic_seed: 1,
            workload_max_new_tokens: 32,
            max_new_tokens: 32,
            baseline: CandidateObservation {
                variant: "baseline_native_no_mtp".to_owned(),
                block_size: None,
                decode_phase_ms: 100.0,
                exact: true,
                generated_tokens: 32,
                model_load_ms: 1.0,
                drafter_load_ms: 0.0,
                prefill_ms: 10.0,
                total_ms: 111.0,
                draft_ms: 0.0,
                verify_ms: 0.0,
                peak_memory_gb: 7.0,
                active_kv_bytes: 1,
                attempted_draft_tokens: 0,
                accepted_draft_tokens: 0,
                acceptance_rate: 0.0,
                accepted_tokens_per_verify: 0.0,
                target_verify_passes: 0,
                rollback_count: 0,
                source_blockers: Vec::new(),
            },
            candidates: candidates
                .into_iter()
                .map(|candidate| (candidate.block_size.unwrap_or_default(), candidate))
                .collect(),
        }
    }

    fn candidate(
        block_size: usize,
        decode_phase_ms: f64,
        acceptance_rate: f64,
        exact: bool,
    ) -> CandidateObservation {
        CandidateObservation {
            variant: format!("mtp_block_{block_size}"),
            block_size: Some(block_size),
            decode_phase_ms,
            exact,
            generated_tokens: 32,
            model_load_ms: 1.0,
            drafter_load_ms: 1.0,
            prefill_ms: 10.0,
            total_ms: 111.0,
            draft_ms: 1.0,
            verify_ms: decode_phase_ms - 1.0,
            peak_memory_gb: 7.0,
            active_kv_bytes: 1,
            attempted_draft_tokens: 10,
            accepted_draft_tokens: (acceptance_rate * 10.0).round() as usize,
            acceptance_rate,
            accepted_tokens_per_verify: acceptance_rate,
            target_verify_passes: 10,
            rollback_count: 1,
            source_blockers: Vec::new(),
        }
    }

    fn synthetic_policy_record(
        policy_name: &str,
        policy_kind: &str,
        selected_decode_phase_ms: f64,
    ) -> PolicyRecord {
        PolicyRecord {
            schema_version: 1,
            goal: GOAL.to_owned(),
            mode: MODE.to_owned(),
            run_id: "run".to_owned(),
            git_sha: "git".to_owned(),
            git_status_short: String::new(),
            command: String::new(),
            source_run_id: "source".to_owned(),
            source_git_sha: "source-git".to_owned(),
            source_git_status_short: String::new(),
            source_decision: "accept_candidate".to_owned(),
            source_status: "passed".to_owned(),
            policy_name: policy_name.to_owned(),
            policy_kind: policy_kind.to_owned(),
            workload_id: "workload".to_owned(),
            family: "family".to_owned(),
            prompt_path: "prompt".to_owned(),
            prompt_sha256: "sha".to_owned(),
            deterministic_seed: 1,
            target_context_tokens: 1024,
            actual_context_tokens: 1024,
            workload_max_new_tokens: 32,
            max_new_tokens: 32,
            selected_variant: "mtp_block_1".to_owned(),
            selected_block_size: Some(1),
            selected_mtp_enabled: true,
            baseline_decode_ms: 100.0,
            selected_decode_phase_ms,
            delta_ms: selected_decode_phase_ms - 100.0,
            speedup_percent: speedup_percent(100.0, selected_decode_phase_ms),
            exact: true,
            status: "passed".to_owned(),
            decision_reason: String::new(),
            baseline_generated_tokens: 32,
            selected_generated_tokens: 32,
            baseline_model_load_ms: 1.0,
            selected_model_load_ms: 1.0,
            selected_drafter_load_ms: 1.0,
            baseline_prefill_ms: 10.0,
            selected_prefill_ms: 10.0,
            selected_draft_ms: 1.0,
            selected_verify_ms: selected_decode_phase_ms - 1.0,
            baseline_total_ms: 111.0,
            selected_total_ms: 111.0,
            baseline_peak_memory_gb: 7.0,
            selected_peak_memory_gb: 7.0,
            memory_delta_gb: 0.0,
            baseline_active_kv_bytes: 1,
            selected_active_kv_bytes: 1,
            active_kv_delta_bytes: 0,
            accepted_draft_tokens: 8,
            attempted_draft_tokens: 10,
            acceptance_rate: 0.8,
            accepted_tokens_per_verify: 0.8,
            target_verify_passes: 10,
            rollback_count: 1,
            source_blockers: Vec::new(),
        }
    }
}
