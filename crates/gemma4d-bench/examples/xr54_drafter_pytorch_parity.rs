#![recursion_limit = "256"]

use std::{
    env, fs,
    fs::File,
    io::{BufRead, BufReader, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{
    CliError, capture_build_provenance, manifest, workload_corpus::WorkloadRecord,
};
use gemma4d_ffi::{
    Drafter, KvCache, KvPolicy, LoadConfig, Target, draft_block_with_scores, prefill,
};
use gemma4d_tokenizer::sha256_hex;
use serde_json::json;

const GOAL: &str = "XR54-drafter-pytorch-parity";
const MODE: &str = "drafter_only_pytorch_parity";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR54-mtp-position-pin/pytorch-parity";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_ASSISTANT_MODEL: &str = "artifacts/models/gemma-4-12B-it-qat-assistant-4bit";
const DEFAULT_PYTORCH_ASSISTANT_MODEL: &str =
    "artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32";
const DEFAULT_PYTHON: &str = "/Users/justin/venvs/xr54-parity/bin/python";
const DEFAULT_PYTHONPATH: &str =
    "/opt/homebrew/Cellar/mlx-lm/0.31.3_2/libexec/lib/python3.14/site-packages";
const DEFAULT_SCRIPT: &str = "scripts/xr54_drafter_pytorch_parity.py";
const DEFAULT_REFERENCE_RECORDS: &str =
    "benchmarks/out/XR54-mtp-position-pin/xr54-r-mtp-candidate-one-trial/records.jsonl";
const DEFAULT_WORKLOAD_ID: &str = "mtp_candidate_1k_001";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1))?;
    if !args.python.exists() {
        return Err(CliError::Runtime(format!(
            "XR54 parity python interpreter not found: {}; pass --python PATH or create the default environment at {DEFAULT_PYTHON}",
            args.python.display()
        ))
        .into());
    }
    fs::create_dir_all(&args.out_dir)?;

    let payload_path = args.out_dir.join("payload.safetensors");
    let request_path = args.out_dir.join("request.json");
    let parity_path = args.out_dir.join("parity.json");
    let stdout_path = args.out_dir.join("python.stdout.txt");
    let stderr_path = args.out_dir.join("python.stderr.txt");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");

    let run_id = run_id();
    let build_provenance = capture_build_provenance()?;
    let git_sha = build_provenance.git_sha.clone();
    let git_status_short = build_provenance.git_status_short.clone();
    let command = command_line();
    let model_identity =
        manifest::capture_artifact_identity(&args.model_path, "GEMMA4D_MODEL_REVISION");
    let assistant_identity =
        manifest::capture_artifact_identity(&args.assistant_model_path, "GEMMA4D_MTP_REVISION");
    let pytorch_assistant_identity = manifest::capture_artifact_identity(
        &args.pytorch_assistant_model_path,
        "GEMMA4D_MTP_PYTORCH_REVISION",
    );

    let workload = load_workload(&args.workloads_path, &args.workload_id)?;
    let mut tokenizer =
        TokenizerHelper::start(&args.python, args.pythonpath.as_deref(), &args.model_path)?;
    let prompt = fs::read_to_string(&workload.prompt_path).map_err(|error| {
        CliError::Runtime(format!(
            "failed to read prompt {}: {error}",
            workload.prompt_path
        ))
    })?;
    let prompt_sha256 = sha256_hex(prompt.as_bytes());
    let token_ids = tokenizer.encode(&prompt)?;
    validate_workload_tokens(&workload, &prompt_sha256, &token_ids)?;

    let target = Target::load(&target_config(&args, token_ids.len()))?;
    let mut cache = KvCache::create(&KvPolicy::default())?;
    let prefill_step = prefill(&target, &mut cache, &token_ids)?;
    let drafter = Drafter::load(&assistant_config(&args, token_ids.len()), &target)?;
    let native_draft = draft_block_with_scores(
        &drafter,
        &mut cache,
        NonZeroU32::new(args.block_size).expect("block size is non-zero"),
    )?;
    let native_draft_tokens = native_draft
        .iter()
        .map(|draft| draft.token)
        .collect::<Vec<_>>();
    let native_draft_logits = native_draft
        .iter()
        .map(|draft| draft.logit)
        .collect::<Vec<_>>();
    let native_logit_margins = native_draft
        .iter()
        .map(|draft| draft.margin)
        .collect::<Vec<_>>();

    let last_context_token = *token_ids
        .last()
        .ok_or_else(|| CliError::Runtime("workload token list is empty".to_owned()))?;
    let mut parity_token_ids = vec![last_context_token];
    parity_token_ids.extend(
        native_draft_tokens
            .iter()
            .take(args.block_size.saturating_sub(1) as usize)
            .copied(),
    );
    let snapshot = cache.export_snapshot()?;
    snapshot.save_mtp_parity_to_path(&target, &parity_token_ids, &payload_path)?;

    let reference_draft_tokens =
        read_reference_draft_tokens(&args.reference_records_path, &args.workload_id);
    let native_matches_reference = reference_draft_tokens
        .as_ref()
        .map(|tokens| tokens == &native_draft_tokens)
        .unwrap_or(false);

    let request = json!({
        "schema_version": 1,
        "goal": GOAL,
        "mode": MODE,
        "run_id": run_id,
        "payload_path": payload_path.display().to_string(),
        "assistant_model_path": args.assistant_model_path.display().to_string(),
        "pytorch_assistant_model_path": args.pytorch_assistant_model_path.display().to_string(),
        "workload_id": workload.workload_id,
        "prompt_path": workload.prompt_path,
        "prompt_sha256": prompt_sha256,
        "actual_context_tokens": token_ids.len(),
        "context_sequence_len": prefill_step.sequence_len,
        "first_position": prefill_step.sequence_len.saturating_sub(1),
        "last_context_token": last_context_token,
        "parity_token_ids": parity_token_ids,
        "block_size": args.block_size,
        "native_draft_tokens": native_draft_tokens,
        "native_draft_logits": native_draft_logits,
        "native_logit_margins": native_logit_margins,
        "reference_records_path": args.reference_records_path.display().to_string(),
        "reference_draft_tokens": reference_draft_tokens,
        "native_matches_reference": native_matches_reference,
    });
    fs::write(&request_path, serde_json::to_vec_pretty(&request)?)?;

    let python = run_python_parity(&args, &request_path, &payload_path, &parity_path);
    fs::write(&stdout_path, &python.stdout)?;
    fs::write(&stderr_path, &python.stderr)?;

    let mut blockers = Vec::new();
    let mut warnings = Vec::new();
    if !native_matches_reference {
        warnings.push(
            "native draft tokens did not match the supplied XR54-R reference record; XR57 score validation uses the fresh native block".to_owned(),
        );
    }
    let parity_result = match fs::read_to_string(&parity_path) {
        Ok(body) => match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(result) => Some(result),
            Err(error) => {
                blockers.push(format!(
                    "PyTorch parity result is unparseable JSON at {}: {error}",
                    parity_path.display()
                ));
                None
            }
        },
        Err(error) => {
            blockers.push(format!(
                "PyTorch parity result is missing or unreadable at {}: {error}",
                parity_path.display()
            ));
            None
        }
    };
    if let Some(result) = &parity_result {
        if result
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("blocked")
            != "completed"
        {
            blockers.push(format!(
                "PyTorch parity result is not completed: {}",
                result
                    .get("blocker")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
            ));
        } else if !parity_matches_native(result) {
            blockers.push(
                "PyTorch parity completed but pinned tokens or pinned score fields did not match native"
                    .to_owned(),
            );
        }
    }
    if !python.status_success {
        blockers.push(format!(
            "PyTorch parity script did not complete: {}",
            python.blocker
        ));
    }
    blockers.sort();
    blockers.dedup();
    warnings.sort();
    warnings.dedup();

    let decision = if blockers.is_empty() {
        "completed"
    } else {
        "blocked_with_evidence"
    };
    let summary = json!({
        "schema_version": 1,
        "goal": GOAL,
        "mode": MODE,
        "decision": decision,
        "run_id": run_id,
        "generated_at_unix_seconds": unix_now(),
        "command": command,
        "git_sha": git_sha,
        "git_status_short": git_status_short,
        "build_provenance": build_provenance,
        "model_identity": model_identity,
        "assistant_identity": assistant_identity,
        "pytorch_assistant_identity": pytorch_assistant_identity,
        "tokenizer_backend": tokenizer.backend(),
        "workload_id": args.workload_id,
        "block_size": args.block_size,
        "model_path": args.model_path.display().to_string(),
        "assistant_model_path": args.assistant_model_path.display().to_string(),
        "pytorch_assistant_model_path": args.pytorch_assistant_model_path.display().to_string(),
        "payload_path": payload_path.display().to_string(),
        "request_path": request_path.display().to_string(),
        "parity_path": parity_path.display().to_string(),
        "python_stdout_path": stdout_path.display().to_string(),
        "python_stderr_path": stderr_path.display().to_string(),
        "summary_path": summary_path.display().to_string(),
        "report_path": report_path.display().to_string(),
        "blockers_path": blockers_path.display().to_string(),
        "generated_files": [
            payload_path.display().to_string(),
            request_path.display().to_string(),
            parity_path.display().to_string(),
            stdout_path.display().to_string(),
            stderr_path.display().to_string(),
            summary_path.display().to_string(),
            report_path.display().to_string(),
            blockers_path.display().to_string(),
        ],
        "context_sequence_len": prefill_step.sequence_len,
        "first_position": prefill_step.sequence_len.saturating_sub(1),
        "last_context_token": last_context_token,
        "parity_token_ids": request["parity_token_ids"],
        "native_draft_tokens": request["native_draft_tokens"],
        "native_draft_logits": request["native_draft_logits"],
        "native_logit_margins": request["native_logit_margins"],
        "reference_draft_tokens": request["reference_draft_tokens"],
        "native_matches_reference": native_matches_reference,
        "python_command": python.command,
        "pythonpath": args.pythonpath,
        "python_exit_status": python.exit_status,
        "python_status_success": python.status_success,
        "parity_result": parity_result,
        "warnings": warnings,
        "blockers": blockers,
    });
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("XR54 drafter PyTorch parity: {decision}");
    println!("payload: {}", payload_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if decision == "blocked_with_evidence" {
        Err("XR54 drafter PyTorch parity blocked; see blockers.md".into())
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
    pytorch_assistant_model_path: PathBuf,
    python: PathBuf,
    pythonpath: Option<String>,
    parity_script: PathBuf,
    reference_records_path: PathBuf,
    workload_id: String,
    block_size: u32,
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
            pytorch_assistant_model_path: PathBuf::from(DEFAULT_PYTORCH_ASSISTANT_MODEL),
            python: PathBuf::from(DEFAULT_PYTHON),
            pythonpath: Some(DEFAULT_PYTHONPATH.to_owned()),
            parity_script: PathBuf::from(DEFAULT_SCRIPT),
            reference_records_path: PathBuf::from(DEFAULT_REFERENCE_RECORDS),
            workload_id: DEFAULT_WORKLOAD_ID.to_owned(),
            block_size: 2,
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
                option @ ("--pytorch-assistant-model-path" | "--dense-assistant-model-path") => {
                    out.pytorch_assistant_model_path =
                        PathBuf::from(required_value(&mut args, option)?)
                }
                "--python" => out.python = PathBuf::from(required_value(&mut args, "--python")?),
                "--pythonpath" => out.pythonpath = Some(required_value(&mut args, "--pythonpath")?),
                "--no-pythonpath" => out.pythonpath = None,
                "--parity-script" => {
                    out.parity_script = PathBuf::from(required_value(&mut args, "--parity-script")?)
                }
                "--reference-records" => {
                    out.reference_records_path =
                        PathBuf::from(required_value(&mut args, "--reference-records")?)
                }
                "--workload-id" => {
                    out.workload_id = required_value(&mut args, "--workload-id")?;
                }
                "--block-size" => {
                    out.block_size = required_value(&mut args, "--block-size")?
                        .parse::<u32>()
                        .map_err(|error| {
                            CliError::Usage(format!("--block-size must be an integer: {error}"))
                        })?;
                }
                "--help" | "-h" => return Err(CliError::Usage(usage())),
                value => {
                    return Err(CliError::Usage(format!(
                        "unknown option {value}\n\n{}",
                        usage()
                    )));
                }
            }
        }
        if out.block_size == 0 {
            return Err(CliError::Usage("--block-size must be > 0".to_owned()));
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct PythonRun {
    command: String,
    status_success: bool,
    exit_status: String,
    stdout: String,
    stderr: String,
    blocker: String,
}

fn run_python_parity(
    args: &Args,
    request_path: &Path,
    payload_path: &Path,
    parity_path: &Path,
) -> PythonRun {
    let mut command = Vec::new();
    if let Some(pythonpath) = args.pythonpath.as_deref() {
        command.push(format!("PYTHONPATH={pythonpath}"));
    }
    command.extend([
        args.python.display().to_string(),
        args.parity_script.display().to_string(),
        "--assistant-model-path".to_owned(),
        args.pytorch_assistant_model_path.display().to_string(),
        "--payload".to_owned(),
        payload_path.display().to_string(),
        "--request".to_owned(),
        request_path.display().to_string(),
        "--out".to_owned(),
        parity_path.display().to_string(),
    ]);
    if parity_path.exists() {
        if let Err(error) = fs::remove_file(parity_path) {
            return PythonRun {
                command: command.join(" "),
                status_success: false,
                exit_status: "not_started".to_owned(),
                stdout: String::new(),
                stderr: String::new(),
                blocker: format!(
                    "failed to remove stale Python parity result {} before run: {error}",
                    parity_path.display()
                ),
            };
        }
    }
    let mut process = Command::new(&args.python);
    if let Some(pythonpath) = args.pythonpath.as_deref() {
        process.env("PYTHONPATH", pythonpath);
    }
    match process
        .arg(&args.parity_script)
        .arg("--assistant-model-path")
        .arg(&args.pytorch_assistant_model_path)
        .arg("--payload")
        .arg(payload_path)
        .arg("--request")
        .arg(request_path)
        .arg("--out")
        .arg(parity_path)
        .output()
    {
        Ok(output) => PythonRun {
            command: command.join(" "),
            status_success: output.status.success(),
            exit_status: output.status.to_string(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            blocker: if output.status.success() {
                "none".to_owned()
            } else {
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .last()
                    .unwrap_or("script exited non-zero")
                    .to_owned()
            },
        },
        Err(error) => PythonRun {
            command: command.join(" "),
            status_success: false,
            exit_status: "not_started".to_owned(),
            stdout: String::new(),
            stderr: String::new(),
            blocker: format!("failed to start Python parity script: {error}"),
        },
    }
}

fn load_workload(path: &Path, workload_id: &str) -> Result<WorkloadRecord, CliError> {
    let file = File::open(path).map_err(|error| {
        CliError::Runtime(format!(
            "failed to open workloads {}: {error}",
            path.display()
        ))
    })?;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| {
            CliError::Runtime(format!(
                "failed to read workload line {} in {}: {error}",
                index + 1,
                path.display()
            ))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<WorkloadRecord>(&line).map_err(|error| {
            CliError::Runtime(format!(
                "failed to parse workload line {} in {}: {error}",
                index + 1,
                path.display()
            ))
        })?;
        if record.workload_id == workload_id {
            return Ok(record);
        }
    }
    Err(CliError::Runtime(format!(
        "requested workload id not found: {workload_id}"
    )))
}

fn validate_workload_tokens(
    workload: &WorkloadRecord,
    prompt_sha256: &str,
    token_ids: &[i32],
) -> Result<(), CliError> {
    if prompt_sha256 != workload.prompt_sha256 {
        return Err(CliError::Runtime(format!(
            "{} prompt sha mismatch: manifest={} actual={}",
            workload.workload_id, workload.prompt_sha256, prompt_sha256
        )));
    }
    if token_ids.len() != workload.actual_context_tokens {
        return Err(CliError::Runtime(format!(
            "{} tokenizer length mismatch: manifest={} actual={}",
            workload.workload_id,
            workload.actual_context_tokens,
            token_ids.len()
        )));
    }
    Ok(())
}

fn read_reference_draft_tokens(path: &Path, workload_id: &str) -> Option<Vec<i32>> {
    let file = File::open(path).ok()?;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<serde_json::Value>(&line).ok()?;
        if value.get("workload_id").and_then(serde_json::Value::as_str) != Some(workload_id) {
            continue;
        }
        return value
            .get("mtp")?
            .get("events")?
            .as_array()?
            .first()?
            .get("draft_tokens")?
            .as_array()?
            .iter()
            .map(|token| i32::try_from(token.as_i64()?).ok())
            .collect();
    }
    None
}

fn target_config(args: &Args, token_count: usize) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(token_count.max(1) as u32)
            .expect("context length is non-zero"),
        allow_unsupported_config: false,
    }
}

fn assistant_config(args: &Args, token_count: usize) -> LoadConfig {
    LoadConfig {
        model_path: args.assistant_model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4_mtp_assistant".to_owned()),
        max_context_tokens: NonZeroU32::new(token_count.max(1) as u32)
            .expect("context length is non-zero"),
        allow_unsupported_config: false,
    }
}

fn parity_matches_native(result: &serde_json::Value) -> bool {
    let tokens_match = result
        .get("matches_native")
        .and_then(|matches| matches.get("pinned")?.as_bool())
        .unwrap_or(false);
    let scores_match = result
        .get("matches_native_scores")
        .and_then(|matches| matches.get("pinned")?.as_bool())
        .unwrap_or(false);
    tokens_match && scores_match
}

fn render_report(summary: &serde_json::Value) -> String {
    let mut out = String::new();
    out.push_str("# XR54 Drafter PyTorch Parity\n\n");
    out.push_str("## Summary\n\n| Field | Value |\n|---|---|\n");
    for field in [
        "decision",
        "run_id",
        "git_sha",
        "workload_id",
        "block_size",
        "payload_path",
        "parity_path",
        "pytorch_assistant_model_path",
        "native_matches_reference",
        "python_exit_status",
    ] {
        out.push_str(&format!(
            "| {} | `{}` |\n",
            field,
            summary
                .get(field)
                .map(value_to_inline)
                .unwrap_or_else(|| "unavailable".to_owned())
        ));
    }
    out.push_str("\n## Tokens\n\n");
    out.push_str(&format!(
        "- Native draft tokens: `{}`\n",
        value_to_inline(&summary["native_draft_tokens"])
    ));
    out.push_str(&format!(
        "- Reference draft tokens: `{}`\n",
        value_to_inline(&summary["reference_draft_tokens"])
    ));
    out.push_str(&format!(
        "- Exported target embedding token IDs: `{}`\n",
        value_to_inline(&summary["parity_token_ids"])
    ));
    if let Some(blockers) = summary
        .get("blockers")
        .and_then(serde_json::Value::as_array)
    {
        out.push_str("\n## Blockers\n\n");
        if blockers.is_empty() {
            out.push_str("No blockers recorded.\n");
        } else {
            for blocker in blockers {
                out.push_str(&format!("- {}\n", value_to_inline(blocker)));
            }
        }
    }
    if let Some(warnings) = summary
        .get("warnings")
        .and_then(serde_json::Value::as_array)
    {
        out.push_str("\n## Warnings\n\n");
        if warnings.is_empty() {
            out.push_str("No warnings recorded.\n");
        } else {
            for warning in warnings {
                out.push_str(&format!("- {}\n", value_to_inline(warning)));
            }
        }
    }
    out
}

fn render_blockers(summary: &serde_json::Value) -> String {
    let blockers = summary
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let warnings = summary
        .get("warnings")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if blockers.is_empty() {
        let mut out = String::from("No hard blockers recorded.\n");
        if !warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for warning in warnings {
                out.push_str(&format!("- {}\n", value_to_inline(&warning)));
            }
        }
        out
    } else {
        let mut out = String::new();
        for blocker in blockers {
            out.push_str(&format!("- {}\n", value_to_inline(&blocker)));
        }
        out
    }
}

fn value_to_inline(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn command_line() -> String {
    env::args().collect::<Vec<_>>().join(" ")
}

fn run_id() -> String {
    format!("xr54-parity-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn usage() -> String {
    format!(
        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example xr54_drafter_pytorch_parity -- [--out-dir PATH] [--workload-id ID] [--block-size N] [--python PATH] [--pythonpath PATH|--no-pythonpath] [--pytorch-assistant-model-path PATH]\n\ndefault out-dir: {DEFAULT_OUT_DIR}\ndefault python: {DEFAULT_PYTHON}\ndefault pythonpath: {DEFAULT_PYTHONPATH}"
    )
}

fn required_value<I>(args: &mut std::iter::Peekable<I>, option: &str) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| CliError::Usage(format!("{option} requires a value")))
}

struct TokenizerHelper {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    backend: String,
}

impl TokenizerHelper {
    fn start(python: &Path, pythonpath: Option<&str>, model_path: &Path) -> Result<Self, CliError> {
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
        let mut process = Command::new(python);
        if let Some(pythonpath) = pythonpath {
            process.env("PYTHONPATH", pythonpath);
        }
        let mut child = process
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

#[cfg(test)]
mod tests {
    use super::parity_matches_native;
    use serde_json::json;

    #[test]
    fn parity_match_guard_requires_pinned_position_mode() {
        let ok = json!({
            "matches_native": {"pinned": true, "incremented": false},
            "matches_native_scores": {"pinned": true, "incremented": false}
        });
        let bad = json!({
            "matches_native": {"pinned": false, "incremented": true},
            "matches_native_scores": {"pinned": true, "incremented": true}
        });
        let bad_scores = json!({
            "matches_native": {"pinned": true, "incremented": true},
            "matches_native_scores": {"pinned": false, "incremented": true}
        });

        assert!(parity_matches_native(&ok));
        assert!(!parity_matches_native(&bad));
        assert!(!parity_matches_native(&bad_scores));
    }
}

impl Drop for TokenizerHelper {
    fn drop(&mut self) {
        let _ = self.request(&serde_json::json!({"cmd":"shutdown"}));
        let _ = self.child.wait();
    }
}
