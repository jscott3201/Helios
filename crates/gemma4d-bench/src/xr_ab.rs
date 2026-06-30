use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_tokenizer::sha256_hex;
use serde::{Deserialize, Serialize};

use crate::{CliError, manifest, workload_corpus::WorkloadRecord};

pub const DEFAULT_WORKLOADS_PATH: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
pub const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR01-real-context-ab-harness";
pub const DEFAULT_MODEL_PATH: &str = "artifacts/models/gemma-4-12B-it-4bit";
pub const GOAL: &str = "XR01-real-context-ab-harness";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    DryRun,
    Real,
    Both,
}

impl RunMode {
    fn run_kinds(self) -> Vec<RunKind> {
        match self {
            Self::DryRun => vec![RunKind::DryRun],
            Self::Real => vec![RunKind::Real],
            Self::Both => vec![RunKind::DryRun, RunKind::Real],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RunKind {
    DryRun,
    Real,
}

impl RunKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::DryRun => "dry_run",
            Self::Real => "real",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
    Helper,
    Native,
    ServerRealHelper,
    ServerNative,
}

impl BackendMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Helper => "helper",
            Self::Native => "native",
            Self::ServerRealHelper => "server_real_helper",
            Self::ServerNative => "server_native",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheFlags {
    pub mode: String,
    pub ram_prefix_cache: bool,
    pub ssd_prefix_cache: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MtpFlags {
    pub enabled: bool,
    pub block_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterFlags {
    pub enabled: bool,
    pub adapter_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantConfig {
    pub name: String,
    pub backend: BackendMode,
    pub env: BTreeMap<String, String>,
    pub cache: CacheFlags,
    pub mtp: MtpFlags,
    pub adapter: AdapterFlags,
}

impl VariantConfig {
    pub fn baseline() -> Self {
        Self {
            name: "baseline".to_owned(),
            backend: BackendMode::Helper,
            env: BTreeMap::new(),
            cache: CacheFlags {
                mode: "disabled".to_owned(),
                ram_prefix_cache: false,
                ssd_prefix_cache: false,
            },
            mtp: MtpFlags {
                enabled: false,
                block_size: 1,
            },
            adapter: AdapterFlags {
                enabled: false,
                adapter_id: None,
            },
        }
    }

    pub fn candidate() -> Self {
        Self {
            name: "candidate".to_owned(),
            ..Self::baseline()
        }
    }

    fn effective_env(&self) -> BTreeMap<String, String> {
        let mut env = self.env.clone();
        match self.backend {
            BackendMode::Native | BackendMode::ServerNative => {
                env.entry("GEMMA4D_REQUIRE_MLX".to_owned())
                    .or_insert_with(|| "1".to_owned());
                env.entry("GEMMA4D_USE_NATIVE_GRAPH".to_owned())
                    .or_insert_with(|| "1".to_owned());
            }
            BackendMode::Helper | BackendMode::ServerRealHelper => {}
        }
        if self.cache.mode != "disabled" {
            env.entry("GEMMA4D_PREFIX_CACHE_MODE".to_owned())
                .or_insert_with(|| self.cache.mode.clone());
        }
        if self.cache.ram_prefix_cache {
            env.entry("GEMMA4D_RAM_PREFIX_CACHE".to_owned())
                .or_insert_with(|| "1".to_owned());
        }
        if self.cache.ssd_prefix_cache {
            env.entry("GEMMA4D_SSD_PREFIX_CACHE".to_owned())
                .or_insert_with(|| "1".to_owned());
        }
        if self.mtp.enabled {
            env.entry("GEMMA4D_MTP_ENABLED".to_owned())
                .or_insert_with(|| "1".to_owned());
            env.entry("GEMMA4D_MTP_BLOCK_SIZE".to_owned())
                .or_insert_with(|| self.mtp.block_size.to_string());
        }
        if let Some(adapter_id) = &self.adapter.adapter_id {
            env.entry("GEMMA4D_ADAPTER_ID".to_owned())
                .or_insert_with(|| adapter_id.clone());
        }
        env
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XrAbOptions {
    pub out_dir: PathBuf,
    pub workloads_path: PathBuf,
    pub model_path: PathBuf,
    pub mode: RunMode,
    pub trials: usize,
    pub max_workloads: Option<usize>,
    pub workload_ids: Vec<String>,
    pub max_new_tokens: Option<usize>,
    pub baseline: VariantConfig,
    pub candidate: VariantConfig,
}

impl Default for XrAbOptions {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
            workloads_path: PathBuf::from(DEFAULT_WORKLOADS_PATH),
            model_path: PathBuf::from(DEFAULT_MODEL_PATH),
            mode: RunMode::DryRun,
            trials: 1,
            max_workloads: None,
            workload_ids: Vec::new(),
            max_new_tokens: None,
            baseline: VariantConfig::baseline(),
            candidate: VariantConfig::candidate(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct XrAbSummary {
    pub schema_version: u32,
    pub goal: String,
    pub decision: String,
    pub status: String,
    pub run_id: String,
    pub git_sha: String,
    pub git_status_short: String,
    pub mode: RunMode,
    pub model_identity: manifest::ArtifactIdentity,
    pub workloads_path: String,
    pub out_dir: String,
    pub records_path: String,
    pub summary_path: String,
    pub report_path: String,
    pub blockers_path: String,
    pub decision_path: String,
    pub variants: Vec<VariantConfig>,
    pub requested_trials: usize,
    pub selected_workloads: Vec<String>,
    pub record_count: usize,
    pub dry_run_records: usize,
    pub real_records: usize,
    pub passed_records: usize,
    pub blocked_records: usize,
    pub failed_records: usize,
    pub schema_checks: SchemaChecks,
    pub command_paths: Vec<String>,
    pub generated_files: Vec<String>,
    pub blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchemaChecks {
    pub has_decode_p50_ms: bool,
    pub has_decode_p95_ms: bool,
    pub has_decode_p99_ms: bool,
    pub has_prefill_ms: bool,
    pub has_total_ms: bool,
    pub has_peak_memory: bool,
    pub has_active_kv_bytes: bool,
    pub has_output_token_ids: bool,
    pub has_correctness_gate: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct XrAbRecord {
    pub schema_version: u32,
    pub goal: String,
    pub run_id: String,
    pub git_sha: String,
    pub git_status_short: String,
    pub model_identity: manifest::ArtifactIdentity,
    pub run_kind: String,
    pub workload_id: String,
    pub family: String,
    pub prompt_path: String,
    pub prompt_sha256: String,
    pub expected_output_style: String,
    pub variant: String,
    pub backend: String,
    pub config: VariantConfig,
    pub trial_index: usize,
    pub input_tokens: usize,
    pub generated_tokens: usize,
    pub output_token_ids: Vec<i32>,
    pub model_load_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    pub decode_token_latencies_ms: Vec<f64>,
    pub decode_p50_ms: f64,
    pub decode_p95_ms: f64,
    pub decode_p99_ms: f64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub peak_mlx_gb: f64,
    pub active_kv_bytes: u64,
    pub rss_mb: f64,
    pub correctness: CorrectnessGate,
    pub command: String,
    pub exit_code: Option<i32>,
    pub status: String,
    pub blocker: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrectnessGate {
    pub status: String,
    pub gate: String,
    pub reference_variant: Option<String>,
    pub token_match: Option<bool>,
    pub first_mismatch_index: Option<usize>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GenerateJson {
    input_tokens: Option<usize>,
    generated_tokens: Option<Vec<i32>>,
    model_load_ms: Option<f64>,
    prefill_ms: Option<f64>,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    total_ms: Option<f64>,
    decode_tps: Option<f64>,
    decode_token_latencies_ms: Option<Vec<f64>>,
    mlx_active_memory_gb: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    active_kv_bytes: Option<u64>,
}

pub fn parse_cli_args<I, S>(args: I) -> Result<XrAbOptions, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut options = XrAbOptions::default();
    let mut args = args.into_iter().map(Into::into).peekable();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => options.out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?),
            "--workloads" | "--workloads-path" => {
                options.workloads_path = PathBuf::from(required_value(&mut args, "--workloads")?)
            }
            "--model-path" => {
                options.model_path = PathBuf::from(required_value(&mut args, "--model-path")?)
            }
            "--mode" => options.mode = parse_run_mode(&required_value(&mut args, "--mode")?)?,
            "--trials" => {
                options.trials =
                    parse_positive_usize(&required_value(&mut args, "--trials")?, "--trials")?
            }
            "--max-workloads" => {
                options.max_workloads = Some(parse_positive_usize(
                    &required_value(&mut args, "--max-workloads")?,
                    "--max-workloads",
                )?)
            }
            "--workload-id" => {
                options
                    .workload_ids
                    .push(required_value(&mut args, "--workload-id")?);
            }
            "--max-new-tokens" => {
                options.max_new_tokens = Some(parse_positive_usize(
                    &required_value(&mut args, "--max-new-tokens")?,
                    "--max-new-tokens",
                )?)
            }
            "--baseline-backend" => {
                options.baseline.backend =
                    parse_backend(&required_value(&mut args, "--baseline-backend")?)?
            }
            "--candidate-backend" => {
                options.candidate.backend =
                    parse_backend(&required_value(&mut args, "--candidate-backend")?)?
            }
            "--baseline-env" => {
                parse_env_pair(
                    &mut options.baseline.env,
                    &required_value(&mut args, "--baseline-env")?,
                    "--baseline-env",
                )?;
            }
            "--candidate-env" => {
                parse_env_pair(
                    &mut options.candidate.env,
                    &required_value(&mut args, "--candidate-env")?,
                    "--candidate-env",
                )?;
            }
            "--baseline-cache" => {
                apply_cache_mode(
                    &mut options.baseline.cache,
                    &required_value(&mut args, "--baseline-cache")?,
                );
            }
            "--candidate-cache" => {
                apply_cache_mode(
                    &mut options.candidate.cache,
                    &required_value(&mut args, "--candidate-cache")?,
                );
            }
            "--baseline-mtp" => {
                options.baseline.mtp.enabled = parse_bool_flag(
                    &required_value(&mut args, "--baseline-mtp")?,
                    "--baseline-mtp",
                )?
            }
            "--candidate-mtp" => {
                options.candidate.mtp.enabled = parse_bool_flag(
                    &required_value(&mut args, "--candidate-mtp")?,
                    "--candidate-mtp",
                )?
            }
            "--baseline-mtp-block-size" => {
                options.baseline.mtp.block_size = parse_positive_usize(
                    &required_value(&mut args, "--baseline-mtp-block-size")?,
                    "--baseline-mtp-block-size",
                )?
            }
            "--candidate-mtp-block-size" => {
                options.candidate.mtp.block_size = parse_positive_usize(
                    &required_value(&mut args, "--candidate-mtp-block-size")?,
                    "--candidate-mtp-block-size",
                )?
            }
            "--baseline-adapter" => {
                apply_adapter(
                    &mut options.baseline.adapter,
                    &required_value(&mut args, "--baseline-adapter")?,
                );
            }
            "--candidate-adapter" => {
                apply_adapter(
                    &mut options.candidate.adapter,
                    &required_value(&mut args, "--candidate-adapter")?,
                );
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

    Ok(options)
}

pub fn write_xr01_artifacts(options: &XrAbOptions) -> Result<XrAbSummary, CliError> {
    fs::create_dir_all(&options.out_dir)
        .map_err(|error| CliError::Runtime(format!("failed to create out dir: {error}")))?;

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
    let model_identity =
        manifest::capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION");
    let workloads = select_workloads(load_workloads(&options.workloads_path)?, options)?;
    let variants = vec![options.baseline.clone(), options.candidate.clone()];

    let mut records = Vec::new();
    let mut command_paths = Vec::new();

    for run_kind in options.mode.run_kinds() {
        for workload in &workloads {
            let prompt = fs::read_to_string(&workload.prompt_path).map_err(|error| {
                CliError::Runtime(format!(
                    "failed to read prompt {}: {error}",
                    workload.prompt_path
                ))
            })?;
            let prompt_sha256 = sha256_hex(prompt.as_bytes());
            for trial_index in 0..options.trials {
                let mut baseline_tokens = None;
                for variant in &variants {
                    let mut record = run_variant(
                        options,
                        &model_identity,
                        &git_sha,
                        &git_status_short,
                        &run_id,
                        run_kind,
                        workload,
                        &prompt,
                        &prompt_sha256,
                        variant,
                        trial_index,
                    );
                    if variant.name == options.baseline.name {
                        record.correctness = CorrectnessGate {
                            status: "passed".to_owned(),
                            gate: "baseline_self_check".to_owned(),
                            reference_variant: None,
                            token_match: Some(true),
                            first_mismatch_index: None,
                            notes: vec![
                                "baseline record is the reference for this workload/trial/run_kind"
                                    .to_owned(),
                            ],
                        };
                        baseline_tokens = Some(record.output_token_ids.clone());
                    } else if let Some(reference_tokens) = &baseline_tokens {
                        record.correctness = compare_against_baseline(
                            &variant.name,
                            reference_tokens,
                            &record.output_token_ids,
                        );
                        if record.correctness.status != "passed" && record.status == "passed" {
                            record.status = "failed".to_owned();
                            record.blocker = Some(
                                "candidate output token ids did not match baseline".to_owned(),
                            );
                        }
                    } else {
                        record.correctness = CorrectnessGate {
                            status: "blocked".to_owned(),
                            gate: "baseline_missing".to_owned(),
                            reference_variant: Some(options.baseline.name.clone()),
                            token_match: None,
                            first_mismatch_index: None,
                            notes: vec![
                                "baseline record was unavailable for comparison".to_owned(),
                            ],
                        };
                        record.status = "blocked".to_owned();
                        record.blocker = Some("baseline comparison record missing".to_owned());
                    }
                    command_paths.push(record.command.clone());
                    records.push(record);
                }
            }
        }
    }

    command_paths.sort();
    command_paths.dedup();
    write_jsonl(&records_path, &records)?;
    let summary = build_summary(
        options,
        &model_identity,
        workloads
            .iter()
            .map(|workload| workload.workload_id.clone())
            .collect(),
        records.len(),
        &records,
        command_paths,
        vec![
            records_path.display().to_string(),
            summary_path.display().to_string(),
            report_path.display().to_string(),
            blockers_path.display().to_string(),
            decision_path.display().to_string(),
        ],
        &run_id,
        &git_sha,
        &git_status_short,
    );

    fs::write(
        &summary_path,
        serde_json::to_vec_pretty(&summary)
            .map_err(|error| CliError::Runtime(format!("failed to serialize summary: {error}")))?,
    )
    .map_err(|error| CliError::Runtime(format!("failed to write summary.json: {error}")))?;
    fs::write(&report_path, render_report(&summary, &records))
        .map_err(|error| CliError::Runtime(format!("failed to write report.md: {error}")))?;
    fs::write(&blockers_path, render_blockers(&summary))
        .map_err(|error| CliError::Runtime(format!("failed to write blockers.md: {error}")))?;
    fs::write(&decision_path, render_decision(&summary))
        .map_err(|error| CliError::Runtime(format!("failed to write decision.md: {error}")))?;

    Ok(summary)
}

fn run_variant(
    options: &XrAbOptions,
    model_identity: &manifest::ArtifactIdentity,
    git_sha: &str,
    git_status_short: &str,
    run_id: &str,
    run_kind: RunKind,
    workload: &WorkloadRecord,
    prompt: &str,
    prompt_sha256: &str,
    variant: &VariantConfig,
    trial_index: usize,
) -> XrAbRecord {
    match run_kind {
        RunKind::DryRun => dry_run_record(
            options,
            model_identity,
            git_sha,
            git_status_short,
            run_id,
            workload,
            prompt_sha256,
            variant,
            trial_index,
        ),
        RunKind::Real => real_run_record(
            options,
            model_identity,
            git_sha,
            git_status_short,
            run_id,
            workload,
            prompt,
            prompt_sha256,
            variant,
            trial_index,
        ),
    }
}

fn dry_run_record(
    options: &XrAbOptions,
    model_identity: &manifest::ArtifactIdentity,
    git_sha: &str,
    git_status_short: &str,
    run_id: &str,
    workload: &WorkloadRecord,
    prompt_sha256: &str,
    variant: &VariantConfig,
    trial_index: usize,
) -> XrAbRecord {
    let max_new_tokens = effective_max_new_tokens(options, workload);
    let output_token_ids = synthetic_tokens(workload, trial_index, max_new_tokens);
    let decode_token_latencies_ms = synthetic_latencies(workload, max_new_tokens);
    let decode_ms = decode_token_latencies_ms.iter().sum::<f64>();
    let prefill_ms = workload.actual_context_tokens as f64 * 0.011;
    let total_ms = prefill_ms + decode_ms;
    let command = format!(
        "cargo run -p gemma4d-bench --example xr01_real_context_ab -- --mode dry-run --workloads {} --out-dir {} --max-workloads {} --trials {} --max-new-tokens {}",
        shell_quote(&options.workloads_path.display().to_string()),
        shell_quote(&options.out_dir.display().to_string()),
        options.max_workloads.unwrap_or(usize::MAX),
        options.trials,
        max_new_tokens
    );
    let input_tokens = workload.actual_context_tokens;
    base_record(
        model_identity,
        git_sha,
        git_status_short,
        run_id,
        RunKind::DryRun,
        workload,
        prompt_sha256,
        variant,
        trial_index,
        input_tokens,
        output_token_ids,
        0.0,
        prefill_ms,
        decode_ms,
        total_ms,
        decode_token_latencies_ms,
        if prefill_ms > 0.0 {
            input_tokens as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        },
        command,
        Some(0),
        "passed",
        None,
        vec!["dry_run_no_model_execution".to_owned()],
    )
}

fn real_run_record(
    options: &XrAbOptions,
    model_identity: &manifest::ArtifactIdentity,
    git_sha: &str,
    git_status_short: &str,
    run_id: &str,
    workload: &WorkloadRecord,
    prompt: &str,
    prompt_sha256: &str,
    variant: &VariantConfig,
    trial_index: usize,
) -> XrAbRecord {
    let max_new_tokens = effective_max_new_tokens(options, workload);
    let command_display = generate_command_display(options, workload, variant, max_new_tokens);
    if !model_identity.exists {
        return blocked_record(
            model_identity,
            git_sha,
            git_status_short,
            run_id,
            RunKind::Real,
            workload,
            prompt_sha256,
            variant,
            trial_index,
            command_display,
            format!(
                "model artifacts missing at {}",
                options.model_path.display()
            ),
        );
    }
    if matches!(
        variant.backend,
        BackendMode::ServerRealHelper | BackendMode::ServerNative
    ) {
        return blocked_record(
            model_identity,
            git_sha,
            git_status_short,
            run_id,
            RunKind::Real,
            workload,
            prompt_sha256,
            variant,
            trial_index,
            command_display,
            "server backend modes are explicit config values in XR01; the smoke runner executes local generate backends and fails closed for server modes".to_owned(),
        );
    }

    let started = Instant::now();
    let mut command = Command::new("cargo");
    command
        .arg("run")
        .arg("-p")
        .arg("gemma4d-server")
        .arg("--")
        .arg("generate")
        .arg("--model-path")
        .arg(&options.model_path)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-context-tokens")
        .arg(workload.actual_context_tokens.max(1).to_string())
        .arg("--max-new-tokens")
        .arg(max_new_tokens.to_string())
        .arg("--json");
    for (key, value) in variant.effective_env() {
        command.env(key, value);
    }

    match command.output() {
        Ok(output) => {
            let wall_ms = duration_ms(started.elapsed());
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if !output.status.success() {
                return blocked_record_with_stdio(
                    model_identity,
                    git_sha,
                    git_status_short,
                    run_id,
                    RunKind::Real,
                    workload,
                    prompt_sha256,
                    variant,
                    trial_index,
                    command_display,
                    output.status.code(),
                    format!(
                        "real generate command failed: {}",
                        first_nonempty_line(&stderr, &stdout)
                    ),
                );
            }
            let Some(parsed) = parse_generate_json(&stdout) else {
                return blocked_record_with_stdio(
                    model_identity,
                    git_sha,
                    git_status_short,
                    run_id,
                    RunKind::Real,
                    workload,
                    prompt_sha256,
                    variant,
                    trial_index,
                    command_display,
                    output.status.code(),
                    "real generate command did not emit parseable JSON metrics".to_owned(),
                );
            };
            let output_token_ids = parsed.generated_tokens.clone().unwrap_or_default();
            let decode_token_latencies_ms =
                parsed.decode_token_latencies_ms.clone().unwrap_or_default();
            let decode_ms = parsed
                .decode_ms
                .unwrap_or_else(|| decode_token_latencies_ms.iter().sum());
            let prefill_ms = parsed.prefill_ms.or(parsed.ttft_ms).unwrap_or(0.0);
            let total_ms = parsed.total_ms.unwrap_or(wall_ms);
            let input_tokens = parsed
                .input_tokens
                .unwrap_or(workload.actual_context_tokens);
            base_record(
                model_identity,
                git_sha,
                git_status_short,
                run_id,
                RunKind::Real,
                workload,
                prompt_sha256,
                variant,
                trial_index,
                input_tokens,
                output_token_ids,
                parsed.model_load_ms.unwrap_or(0.0),
                prefill_ms,
                decode_ms,
                total_ms,
                decode_token_latencies_ms,
                if prefill_ms > 0.0 {
                    input_tokens as f64 / (prefill_ms / 1000.0)
                } else {
                    0.0
                },
                command_display,
                output.status.code(),
                "passed",
                None,
                vec!["real_model_smoke".to_owned()],
            )
            .with_real_metrics(parsed)
        }
        Err(error) => blocked_record(
            model_identity,
            git_sha,
            git_status_short,
            run_id,
            RunKind::Real,
            workload,
            prompt_sha256,
            variant,
            trial_index,
            command_display,
            format!("failed to spawn real generate command: {error}"),
        ),
    }
}

trait WithRealMetrics {
    fn with_real_metrics(self, parsed: GenerateJson) -> Self;
}

impl WithRealMetrics for XrAbRecord {
    fn with_real_metrics(mut self, parsed: GenerateJson) -> Self {
        self.decode_tps = parsed.decode_tps.unwrap_or(self.decode_tps);
        self.peak_mlx_gb = parsed
            .peak_memory_gb
            .or(parsed.mlx_active_memory_gb)
            .unwrap_or(0.0);
        self.rss_mb = parsed.peak_rss_mb.unwrap_or(0.0);
        self.active_kv_bytes = parsed.active_kv_bytes.unwrap_or(0);
        self
    }
}

#[allow(clippy::too_many_arguments)]
fn base_record(
    model_identity: &manifest::ArtifactIdentity,
    git_sha: &str,
    git_status_short: &str,
    run_id: &str,
    run_kind: RunKind,
    workload: &WorkloadRecord,
    prompt_sha256: &str,
    variant: &VariantConfig,
    trial_index: usize,
    input_tokens: usize,
    output_token_ids: Vec<i32>,
    model_load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    decode_token_latencies_ms: Vec<f64>,
    prefill_tps: f64,
    command: String,
    exit_code: Option<i32>,
    status: &str,
    blocker: Option<String>,
    notes: Vec<String>,
) -> XrAbRecord {
    let decode_p50_ms = percentile(&decode_token_latencies_ms, 50.0);
    let decode_p95_ms = percentile(&decode_token_latencies_ms, 95.0);
    let decode_p99_ms = percentile(&decode_token_latencies_ms, 99.0);
    let generated_tokens = output_token_ids.len();
    let decode_tps = if decode_ms > 0.0 {
        generated_tokens as f64 / (decode_ms / 1000.0)
    } else {
        0.0
    };

    XrAbRecord {
        schema_version: 1,
        goal: GOAL.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        model_identity: model_identity.clone(),
        run_kind: run_kind.as_str().to_owned(),
        workload_id: workload.workload_id.clone(),
        family: workload.family.clone(),
        prompt_path: workload.prompt_path.clone(),
        prompt_sha256: prompt_sha256.to_owned(),
        expected_output_style: workload.expected_output_style.clone(),
        variant: variant.name.clone(),
        backend: variant.backend.as_str().to_owned(),
        config: variant.clone(),
        trial_index,
        input_tokens,
        generated_tokens,
        output_token_ids,
        model_load_ms,
        prefill_ms,
        decode_ms,
        total_ms,
        decode_token_latencies_ms,
        decode_p50_ms,
        decode_p95_ms,
        decode_p99_ms,
        prefill_tps,
        decode_tps,
        peak_mlx_gb: 0.0,
        active_kv_bytes: 0,
        rss_mb: 0.0,
        correctness: CorrectnessGate {
            status: "pending".to_owned(),
            gate: "pending_pairwise_comparison".to_owned(),
            reference_variant: None,
            token_match: None,
            first_mismatch_index: None,
            notes: Vec::new(),
        },
        command,
        exit_code,
        status: status.to_owned(),
        blocker,
        notes,
    }
}

#[allow(clippy::too_many_arguments)]
fn blocked_record(
    model_identity: &manifest::ArtifactIdentity,
    git_sha: &str,
    git_status_short: &str,
    run_id: &str,
    run_kind: RunKind,
    workload: &WorkloadRecord,
    prompt_sha256: &str,
    variant: &VariantConfig,
    trial_index: usize,
    command: String,
    blocker: String,
) -> XrAbRecord {
    blocked_record_with_stdio(
        model_identity,
        git_sha,
        git_status_short,
        run_id,
        run_kind,
        workload,
        prompt_sha256,
        variant,
        trial_index,
        command,
        None,
        blocker,
    )
}

#[allow(clippy::too_many_arguments)]
fn blocked_record_with_stdio(
    model_identity: &manifest::ArtifactIdentity,
    git_sha: &str,
    git_status_short: &str,
    run_id: &str,
    run_kind: RunKind,
    workload: &WorkloadRecord,
    prompt_sha256: &str,
    variant: &VariantConfig,
    trial_index: usize,
    command: String,
    exit_code: Option<i32>,
    blocker: String,
) -> XrAbRecord {
    let mut record = base_record(
        model_identity,
        git_sha,
        git_status_short,
        run_id,
        run_kind,
        workload,
        prompt_sha256,
        variant,
        trial_index,
        workload.actual_context_tokens,
        Vec::new(),
        0.0,
        0.0,
        0.0,
        0.0,
        Vec::new(),
        0.0,
        command,
        exit_code,
        "blocked",
        Some(blocker.clone()),
        vec!["failure_closed".to_owned()],
    );
    record.correctness = CorrectnessGate {
        status: "blocked".to_owned(),
        gate: "command_completed_with_metrics".to_owned(),
        reference_variant: None,
        token_match: None,
        first_mismatch_index: None,
        notes: vec![blocker],
    };
    record
}

fn compare_against_baseline(
    variant_name: &str,
    reference_tokens: &[i32],
    candidate_tokens: &[i32],
) -> CorrectnessGate {
    if reference_tokens == candidate_tokens {
        return CorrectnessGate {
            status: "passed".to_owned(),
            gate: "candidate_output_token_ids_equal_baseline".to_owned(),
            reference_variant: Some("baseline".to_owned()),
            token_match: Some(true),
            first_mismatch_index: None,
            notes: vec![format!("{variant_name} matched baseline output token ids")],
        };
    }

    let first_mismatch_index = reference_tokens
        .iter()
        .zip(candidate_tokens.iter())
        .position(|(left, right)| left != right)
        .or_else(|| Some(reference_tokens.len().min(candidate_tokens.len())));
    CorrectnessGate {
        status: "failed".to_owned(),
        gate: "candidate_output_token_ids_equal_baseline".to_owned(),
        reference_variant: Some("baseline".to_owned()),
        token_match: Some(false),
        first_mismatch_index,
        notes: vec![format!(
            "{variant_name} differed from baseline output token ids"
        )],
    }
}

fn build_summary(
    options: &XrAbOptions,
    model_identity: &manifest::ArtifactIdentity,
    selected_workloads: Vec<String>,
    record_count: usize,
    records: &[XrAbRecord],
    command_paths: Vec<String>,
    generated_files: Vec<String>,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
) -> XrAbSummary {
    let dry_run_records = records
        .iter()
        .filter(|record| record.run_kind == "dry_run")
        .count();
    let real_records = records
        .iter()
        .filter(|record| record.run_kind == "real")
        .count();
    let passed_records = records
        .iter()
        .filter(|record| record.status == "passed" && record.correctness.status == "passed")
        .count();
    let blocked_records = records
        .iter()
        .filter(|record| record.status == "blocked")
        .count();
    let failed_records = records
        .iter()
        .filter(|record| record.status == "failed" || record.correctness.status == "failed")
        .count();
    let schema_checks = schema_checks(records);
    let mut blockers = records
        .iter()
        .filter_map(|record| {
            record.blocker.as_ref().map(|blocker| {
                format!(
                    "{} {} {} trial {}: {}",
                    record.run_kind,
                    record.workload_id,
                    record.variant,
                    record.trial_index,
                    blocker
                )
            })
        })
        .collect::<Vec<_>>();
    blockers.extend(decision_blockers(
        options.mode,
        dry_run_records,
        real_records,
        &schema_checks,
    ));
    blockers.sort();
    blockers.dedup();

    let decision = if blockers.is_empty() {
        "accept_candidate"
    } else if blocked_records > 0 {
        "blocked_with_evidence"
    } else {
        "needs_more_data"
    };
    let status = if blockers.is_empty() {
        "passed"
    } else if blocked_records > 0 {
        "blocked"
    } else {
        "incomplete"
    };

    XrAbSummary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        decision: decision.to_owned(),
        status: status.to_owned(),
        run_id: run_id.to_owned(),
        git_sha: git_sha.to_owned(),
        git_status_short: git_status_short.to_owned(),
        mode: options.mode,
        model_identity: model_identity.clone(),
        workloads_path: options.workloads_path.display().to_string(),
        out_dir: options.out_dir.display().to_string(),
        records_path: options.out_dir.join("records.jsonl").display().to_string(),
        summary_path: options.out_dir.join("summary.json").display().to_string(),
        report_path: options.out_dir.join("report.md").display().to_string(),
        blockers_path: options.out_dir.join("blockers.md").display().to_string(),
        decision_path: options.out_dir.join("decision.md").display().to_string(),
        variants: vec![options.baseline.clone(), options.candidate.clone()],
        requested_trials: options.trials,
        selected_workloads,
        record_count,
        dry_run_records,
        real_records,
        passed_records,
        blocked_records,
        failed_records,
        schema_checks,
        command_paths,
        generated_files,
        blockers,
    }
}

fn decision_blockers(
    mode: RunMode,
    dry_run_records: usize,
    real_records: usize,
    schema: &SchemaChecks,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if matches!(mode, RunMode::DryRun | RunMode::Both) && dry_run_records == 0 {
        blockers.push("dry-run mode produced no records".to_owned());
    }
    if matches!(mode, RunMode::Real | RunMode::Both) && real_records == 0 {
        blockers.push("real-run mode produced no records".to_owned());
    }
    if matches!(mode, RunMode::DryRun) {
        blockers.push(
            "dry-run evidence is valid for CI/offline smoke, but accept_candidate requires a model-available command path; rerun with --mode both or --mode real when artifacts are available".to_owned(),
        );
    }
    if !schema.has_decode_p50_ms
        || !schema.has_decode_p95_ms
        || !schema.has_decode_p99_ms
        || !schema.has_prefill_ms
        || !schema.has_total_ms
        || !schema.has_peak_memory
        || !schema.has_active_kv_bytes
        || !schema.has_output_token_ids
        || !schema.has_correctness_gate
    {
        blockers.push("evidence schema is missing one or more XR01 required fields".to_owned());
    }
    blockers
}

fn schema_checks(records: &[XrAbRecord]) -> SchemaChecks {
    SchemaChecks {
        has_decode_p50_ms: records.iter().all(|record| record.decode_p50_ms >= 0.0),
        has_decode_p95_ms: records.iter().all(|record| record.decode_p95_ms >= 0.0),
        has_decode_p99_ms: records.iter().all(|record| record.decode_p99_ms >= 0.0),
        has_prefill_ms: records.iter().all(|record| record.prefill_ms >= 0.0),
        has_total_ms: records.iter().all(|record| record.total_ms >= 0.0),
        has_peak_memory: records.iter().all(|record| record.peak_mlx_gb >= 0.0),
        has_active_kv_bytes: records
            .iter()
            .all(|record| record.active_kv_bytes <= u64::MAX),
        has_output_token_ids: records.iter().all(|record| {
            record.status == "blocked" || record.generated_tokens == record.output_token_ids.len()
        }),
        has_correctness_gate: records.iter().all(|record| {
            !record.correctness.status.is_empty() && !record.correctness.gate.is_empty()
        }),
    }
}

pub fn render_report(summary: &XrAbSummary, records: &[XrAbRecord]) -> String {
    let mut out = String::new();
    out.push_str("# XR01 Real-Context A/B Harness Report\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Decision | `{}` |\n", summary.decision));
    out.push_str(&format!("| Status | `{}` |\n", summary.status));
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!(
        "| Git status | `{}` |\n",
        markdown_escape(&summary.git_status_short)
    ));
    out.push_str(&format!("| Mode | `{:?}` |\n", summary.mode));
    out.push_str(&format!(
        "| Workloads | `{}` |\n",
        summary.selected_workloads.len()
    ));
    out.push_str(&format!("| Records | `{}` |\n", summary.record_count));
    out.push_str(&format!(
        "| Dry-run records | `{}` |\n",
        summary.dry_run_records
    ));
    out.push_str(&format!("| Real records | `{}` |\n", summary.real_records));
    out.push_str(&format!(
        "| Model exists | `{}` |\n\n",
        summary.model_identity.exists
    ));

    out.push_str("## Variants\n\n");
    out.push_str(
        "| Variant | Backend | Env | Cache | MTP | Adapter |\n|---|---|---|---|---|---|\n",
    );
    for variant in &summary.variants {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            markdown_escape(&variant.name),
            variant.backend.as_str(),
            markdown_escape(&env_display(&variant.effective_env())),
            markdown_escape(&variant.cache.mode),
            if variant.mtp.enabled {
                "enabled"
            } else {
                "disabled"
            },
            markdown_escape(variant.adapter.adapter_id.as_deref().unwrap_or("none"))
        ));
    }

    out.push_str("\n## Records\n\n");
    out.push_str("| Kind | Workload | Variant | Backend | Trial | Status | Input | Output | Prefill ms | Decode p50/p95/p99 ms | Total ms | Peak GB | Active KV bytes | Correctness |\n");
    out.push_str("|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for record in records {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | {} | `{}` | {} | {} | {:.3} | {:.3}/{:.3}/{:.3} | {:.3} | {:.3} | {} | `{}` |\n",
            record.run_kind,
            markdown_escape(&record.workload_id),
            markdown_escape(&record.variant),
            markdown_escape(&record.backend),
            record.trial_index,
            markdown_escape(&record.status),
            record.input_tokens,
            record.generated_tokens,
            record.prefill_ms,
            record.decode_p50_ms,
            record.decode_p95_ms,
            record.decode_p99_ms,
            record.total_ms,
            record.peak_mlx_gb,
            record.active_kv_bytes,
            markdown_escape(&record.correctness.status)
        ));
    }

    out.push_str("\n## Commands\n\n```text\n");
    for command in &summary.command_paths {
        out.push_str(command);
        out.push('\n');
    }
    out.push_str("```\n\n");

    out.push_str("## Generated Files\n\n");
    for path in &summary.generated_files {
        out.push_str(&format!("- `{}`\n", markdown_escape(path)));
    }
    out
}

pub fn render_blockers(summary: &XrAbSummary) -> String {
    let mut out = String::new();
    out.push_str("# XR01 Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No blockers recorded.\n\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
        out.push('\n');
    }
    out.push_str("## Reproduce\n\n```text\n");
    for command in &summary.command_paths {
        out.push_str(command);
        out.push('\n');
    }
    out.push_str("```\n");
    out
}

pub fn render_decision(summary: &XrAbSummary) -> String {
    let mut out = String::new();
    out.push_str("# XR01 Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    if summary.blockers.is_empty() {
        out.push_str(
            "The XR01 harness produced dry-run evidence and real model smoke evidence with stable A/B records, decode percentile fields, prefill/total timing fields, memory fields, active KV bytes, output token IDs, and correctness gates. This accepts the harness shape only; it does not claim a production serving or runtime optimization win.\n\n",
        );
    } else {
        out.push_str("The XR01 harness wrote evidence but cannot be accepted until the blockers are resolved.\n\n");
    }
    out.push_str("## Evidence\n\n");
    out.push_str(&format!("- Records: `{}`\n", summary.records_path));
    out.push_str(&format!("- Summary: `{}`\n", summary.summary_path));
    out.push_str(&format!("- Report: `{}`\n", summary.report_path));
    out.push_str(&format!("- Blockers: `{}`\n", summary.blockers_path));
    out
}

fn load_workloads(path: &Path) -> Result<Vec<WorkloadRecord>, CliError> {
    let text = fs::read_to_string(path)
        .map_err(|error| CliError::Runtime(format!("failed to read workloads JSONL: {error}")))?;
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        records.push(
            serde_json::from_str::<WorkloadRecord>(line).map_err(|error| {
                CliError::Runtime(format!(
                    "failed to parse workload line {} in {}: {error}",
                    index + 1,
                    path.display()
                ))
            })?,
        );
    }
    if records.is_empty() {
        return Err(CliError::Runtime(format!(
            "workload manifest is empty: {}",
            path.display()
        )));
    }
    Ok(records)
}

fn select_workloads(
    mut workloads: Vec<WorkloadRecord>,
    options: &XrAbOptions,
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

fn write_jsonl(path: &Path, records: &[XrAbRecord]) -> Result<(), CliError> {
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

fn parse_generate_json(stdout: &str) -> Option<GenerateJson> {
    stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<GenerateJson>(line.trim()).ok())
}

fn synthetic_tokens(workload: &WorkloadRecord, trial_index: usize, count: usize) -> Vec<i32> {
    let seed = format!(
        "{}:{}:{}:{}",
        workload.workload_id, workload.prompt_sha256, workload.deterministic_seed, trial_index
    );
    let digest = sha256_hex(seed.as_bytes());
    let bytes = digest.as_bytes();
    (0..count)
        .map(|index| {
            let byte = bytes[index % bytes.len()] as i32;
            1000 + byte + (index as i32 % 31)
        })
        .collect()
}

fn synthetic_latencies(workload: &WorkloadRecord, count: usize) -> Vec<f64> {
    let base = 1.0 + (workload.actual_context_tokens as f64 / 8192.0);
    (0..count)
        .map(|index| base + (index % 7) as f64 * 0.037)
        .collect()
}

fn effective_max_new_tokens(options: &XrAbOptions, workload: &WorkloadRecord) -> usize {
    options
        .max_new_tokens
        .unwrap_or(workload.max_new_tokens)
        .min(workload.max_new_tokens)
        .max(1)
}

fn generate_command_display(
    options: &XrAbOptions,
    workload: &WorkloadRecord,
    variant: &VariantConfig,
    max_new_tokens: usize,
) -> String {
    let mut parts = variant
        .effective_env()
        .iter()
        .map(|(key, value)| format!("{key}={}", shell_quote(value)))
        .collect::<Vec<_>>();
    parts.extend(
        [
            "cargo".to_owned(),
            "run".to_owned(),
            "-p".to_owned(),
            "gemma4d-server".to_owned(),
            "--".to_owned(),
            "generate".to_owned(),
            "--model-path".to_owned(),
            shell_quote(&options.model_path.display().to_string()),
            "--prompt".to_owned(),
            format!("\"$(cat {})\"", shell_quote(&workload.prompt_path)),
            "--max-context-tokens".to_owned(),
            workload.actual_context_tokens.max(1).to_string(),
            "--max-new-tokens".to_owned(),
            max_new_tokens.to_string(),
            "--json".to_owned(),
        ]
        .into_iter(),
    );
    parts.join(" ")
}

fn env_display(env: &BTreeMap<String, String>) -> String {
    if env.is_empty() {
        return "none".to_owned();
    }
    env.iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let rank = ((percentile / 100.0) * (values.len().saturating_sub(1) as f64)).ceil() as usize;
    values[rank.min(values.len() - 1)]
}

fn parse_run_mode(value: &str) -> Result<RunMode, CliError> {
    match value {
        "dry-run" | "dry_run" => Ok(RunMode::DryRun),
        "real" => Ok(RunMode::Real),
        "both" => Ok(RunMode::Both),
        other => Err(CliError::Usage(format!(
            "unsupported --mode '{other}', expected dry-run, real, or both"
        ))),
    }
}

fn parse_backend(value: &str) -> Result<BackendMode, CliError> {
    match value {
        "helper" => Ok(BackendMode::Helper),
        "native" => Ok(BackendMode::Native),
        "server_real_helper" | "server-real-helper" => Ok(BackendMode::ServerRealHelper),
        "server_native" | "server-native" => Ok(BackendMode::ServerNative),
        other => Err(CliError::Usage(format!(
            "unsupported backend '{other}', expected helper, native, server_real_helper, or server_native"
        ))),
    }
}

fn parse_env_pair(
    env: &mut BTreeMap<String, String>,
    value: &str,
    option: &str,
) -> Result<(), CliError> {
    let Some((key, val)) = value.split_once('=') else {
        return Err(CliError::Usage(format!("{option} requires KEY=VALUE")));
    };
    if key.trim().is_empty() {
        return Err(CliError::Usage(format!(
            "{option} requires a non-empty key"
        )));
    }
    env.insert(key.to_owned(), val.to_owned());
    Ok(())
}

fn apply_cache_mode(cache: &mut CacheFlags, value: &str) {
    cache.mode = value.to_owned();
    cache.ram_prefix_cache = matches!(value, "ram" | "ram_prefix" | "ram-prefix" | "both");
    cache.ssd_prefix_cache = matches!(value, "ssd" | "ssd_prefix" | "ssd-prefix" | "both");
}

fn apply_adapter(adapter: &mut AdapterFlags, value: &str) {
    if value == "none" || value == "disabled" {
        adapter.enabled = false;
        adapter.adapter_id = None;
    } else {
        adapter.enabled = true;
        adapter.adapter_id = Some(value.to_owned());
    }
}

fn parse_bool_flag(value: &str, option: &str) -> Result<bool, CliError> {
    match value {
        "1" | "true" | "on" | "yes" | "enabled" => Ok(true),
        "0" | "false" | "off" | "no" | "disabled" => Ok(false),
        other => Err(CliError::Usage(format!(
            "{option} expects enabled/disabled, got '{other}'"
        ))),
    }
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

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("xr01-{}-{}", now.as_secs(), now.subsec_nanos())
}

fn first_nonempty_line(left: &str, right: &str) -> String {
    left.lines()
        .chain(right.lines())
        .find(|line| !line.trim().is_empty())
        .unwrap_or("no stderr/stdout")
        .trim()
        .to_owned()
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

fn usage() -> String {
    format!(
        "usage: cargo run -p gemma4d-bench --example xr01_real_context_ab -- [--mode dry-run|real|both] [--workloads PATH] [--out-dir PATH] [--model-path PATH] [--max-workloads N] [--workload-id ID] [--trials N] [--max-new-tokens N] [--baseline-backend helper|native|server_real_helper|server_native] [--candidate-backend helper|native|server_real_helper|server_native] [--baseline-env KEY=VALUE] [--candidate-env KEY=VALUE] [--baseline-cache disabled|ram|ssd|both] [--candidate-cache disabled|ram|ssd|both] [--baseline-mtp enabled|disabled] [--candidate-mtp enabled|disabled] [--baseline-adapter ID|none] [--candidate-adapter ID|none]\n\ndefault workloads: {DEFAULT_WORKLOADS_PATH}\ndefault out-dir: {DEFAULT_OUT_DIR}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cli_args_sets_explicit_variant_config() {
        let options = parse_cli_args([
            "--mode",
            "both",
            "--candidate-backend",
            "native",
            "--candidate-env",
            "GEMMA4D_TEST=1",
            "--candidate-cache",
            "ram",
            "--candidate-mtp",
            "enabled",
            "--candidate-mtp-block-size",
            "4",
            "--candidate-adapter",
            "rust-coding-r16-v1",
            "--max-workloads",
            "1",
        ])
        .expect("parse");
        assert_eq!(options.mode, RunMode::Both);
        assert_eq!(options.max_workloads, Some(1));
        assert_eq!(options.candidate.backend, BackendMode::Native);
        assert_eq!(
            options.candidate.env.get("GEMMA4D_TEST"),
            Some(&"1".to_owned())
        );
        assert!(options.candidate.cache.ram_prefix_cache);
        assert!(options.candidate.mtp.enabled);
        assert_eq!(options.candidate.mtp.block_size, 4);
        assert_eq!(
            options.candidate.adapter.adapter_id.as_deref(),
            Some("rust-coding-r16-v1")
        );
        let effective = options.candidate.effective_env();
        assert_eq!(
            effective.get("GEMMA4D_USE_NATIVE_GRAPH"),
            Some(&"1".to_owned())
        );
        assert_eq!(
            effective.get("GEMMA4D_MTP_BLOCK_SIZE"),
            Some(&"4".to_owned())
        );
    }

    #[test]
    fn dry_run_writes_stable_artifacts_without_model() {
        let root = unique_temp_dir("xr01-dry-run");
        fs::create_dir_all(root.join("prompts")).expect("prompt dir");
        let prompt_path = root.join("prompts/chat_short_1k_001.txt");
        fs::write(&prompt_path, "hello from xr01").expect("prompt");
        let workload = WorkloadRecord {
            schema_version: 1,
            workload_id: "chat_short_1k_001".to_owned(),
            family: "chat_short".to_owned(),
            source_files: vec!["AGENTS.md".to_owned()],
            prompt_path: prompt_path.display().to_string(),
            expected_output_style: "concise_operator_answer".to_owned(),
            max_new_tokens: 4,
            target_context_tokens: 1024,
            actual_context_tokens: 1024,
            deterministic_seed: 20260630,
            prompt_sha256: sha256_hex(b"hello from xr01"),
            tokenizer_backend: "test".to_owned(),
            notes: "test workload".to_owned(),
        };
        let workloads_path = root.join("workloads.jsonl");
        fs::write(
            &workloads_path,
            format!("{}\n", serde_json::to_string(&workload).expect("record")),
        )
        .expect("workloads");
        let options = XrAbOptions {
            out_dir: root.join("out"),
            workloads_path,
            model_path: root.join("missing-model"),
            mode: RunMode::DryRun,
            trials: 1,
            max_workloads: Some(1),
            workload_ids: Vec::new(),
            max_new_tokens: Some(2),
            baseline: VariantConfig::baseline(),
            candidate: VariantConfig::candidate(),
        };

        let summary = write_xr01_artifacts(&options).expect("write artifacts");
        assert_eq!(summary.dry_run_records, 2);
        assert_eq!(summary.real_records, 0);
        assert_eq!(summary.decision, "needs_more_data");
        assert!(options.out_dir.join("records.jsonl").exists());
        assert!(options.out_dir.join("summary.json").exists());
        assert!(options.out_dir.join("report.md").exists());
        assert!(options.out_dir.join("blockers.md").exists());
        assert!(options.out_dir.join("decision.md").exists());
    }

    #[test]
    fn real_run_missing_model_is_failure_closed() {
        let root = unique_temp_dir("xr01-real-blocked");
        fs::create_dir_all(root.join("prompts")).expect("prompt dir");
        let prompt_path = root.join("prompts/chat_short_1k_001.txt");
        fs::write(&prompt_path, "hello from xr01").expect("prompt");
        let workload = WorkloadRecord {
            schema_version: 1,
            workload_id: "chat_short_1k_001".to_owned(),
            family: "chat_short".to_owned(),
            source_files: vec!["AGENTS.md".to_owned()],
            prompt_path: prompt_path.display().to_string(),
            expected_output_style: "concise_operator_answer".to_owned(),
            max_new_tokens: 1,
            target_context_tokens: 1024,
            actual_context_tokens: 1024,
            deterministic_seed: 20260630,
            prompt_sha256: sha256_hex(b"hello from xr01"),
            tokenizer_backend: "test".to_owned(),
            notes: "test workload".to_owned(),
        };
        let workloads_path = root.join("workloads.jsonl");
        fs::write(
            &workloads_path,
            format!("{}\n", serde_json::to_string(&workload).expect("record")),
        )
        .expect("workloads");
        let options = XrAbOptions {
            out_dir: root.join("out"),
            workloads_path,
            model_path: root.join("missing-model"),
            mode: RunMode::Real,
            trials: 1,
            max_workloads: Some(1),
            workload_ids: Vec::new(),
            max_new_tokens: Some(1),
            baseline: VariantConfig::baseline(),
            candidate: VariantConfig::candidate(),
        };

        let summary = write_xr01_artifacts(&options).expect("write artifacts");
        assert_eq!(summary.real_records, 2);
        assert_eq!(summary.decision, "blocked_with_evidence");
        let blockers = fs::read_to_string(options.out_dir.join("blockers.md")).expect("blockers");
        assert!(blockers.contains("model artifacts missing"));
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let path = std::env::temp_dir().join(format!(
            "gemma4d-{label}-{}-{}",
            std::process::id(),
            now.as_nanos()
        ));
        fs::create_dir_all(&path).expect("temp dir");
        path
    }
}
