use std::{
    collections::BTreeMap,
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P00-performance-baseline";
const MODE: &str = "target_greedy_mlx_lm_helper_via_c_abi";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let environment = capture_environment();
    let model_identity = capture_model_identity(&args.model_path);
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");

    let mut records = Vec::new();
    if model_identity.exists {
        for context_tokens in &args.contexts {
            records.push(run_case(&args, &run_id, *context_tokens)?);
        }
    } else {
        for context_tokens in &args.contexts {
            records.push(blocked_case(&args, &run_id, *context_tokens));
        }
    }

    let status = aggregate_status(&records);
    let blockers = blockers_for(&records);
    let summary = BaselineSummary {
        schema_version: 1,
        goal: "P00-performance-baseline",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        environment,
        model_identity,
        relevant_environment: capture_relevant_environment(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        cases: records.clone(),
        blockers,
        measurement_notes: vec![
            "command_wall_ms measures the outer cargo command invocation used by this harness.",
            "total_ms is reported by gemma4d-server generate after CLI dispatch enters generation.",
            "process_command_overhead_ms is command_wall_ms minus total_ms when total_ms is available.",
            "ttft_ms is preserved as the existing prefill timing field; prefill_ms is emitted explicitly for P00.",
            "mlx_active_memory_gb and mlx_cache_memory_gb are null until the helper/native boundary exposes them.",
        ],
    };

    write_jsonl(&records_path, &records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P00 performance baseline: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if summary.status == "failed" {
        Err("P00 performance baseline failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    contexts: Vec<usize>,
    max_new_tokens: usize,
    max_context_tokens: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut contexts = vec![1024, 4096, 8192, 16_384];
        let mut max_new_tokens = 128;
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
                "--contexts" => {
                    let value = args.next().ok_or("--contexts requires a comma list")?;
                    contexts = parse_contexts(&value)?;
                }
                "--max-new-tokens" => {
                    let value = args.next().ok_or("--max-new-tokens requires a value")?;
                    max_new_tokens = parse_positive_usize(&value, "--max-new-tokens")?;
                }
                "--max-context-tokens" => {
                    let value = args.next().ok_or("--max-context-tokens requires a value")?;
                    max_context_tokens = parse_positive_usize(&value, "--max-context-tokens")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example p00_performance_baseline -- [--out-dir PATH] [--model-path PATH] [--contexts 1024,4096,8192,16384] [--max-new-tokens N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        if contexts.is_empty() {
            return Err("--contexts must include at least one context".into());
        }

        Ok(Self {
            out_dir,
            model_path,
            contexts,
            max_new_tokens,
            max_context_tokens,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct BaselineSummary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    environment: Environment,
    model_identity: ModelIdentity,
    relevant_environment: BTreeMap<String, Option<String>>,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    cases: Vec<BaselineRecord>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct Environment {
    machine: String,
    macos: String,
    rustc: String,
    cargo: String,
    mlx_version: String,
    git_commit: String,
    git_status_short: String,
    hw_memsize_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct ModelIdentity {
    model_path: String,
    exists: bool,
    configured_revision: String,
    revision_source: String,
    local_artifact_sha256: String,
    config_sha256: String,
    tokenizer_sha256: String,
    tokenizer_config_sha256: String,
    chat_template_sha256: String,
    safetensors_inventory_sha256: String,
    safetensors_file_count: usize,
    safetensors_total_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct BaselineRecord {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    workload: &'static str,
    context_tokens: usize,
    generated_tokens_requested: usize,
    generated_tokens_observed: usize,
    mode: &'static str,
    command: String,
    exit_code: Option<i32>,
    status: String,
    command_wall_ms: f64,
    process_command_overhead_ms: Option<f64>,
    input_tokens: Option<usize>,
    model_load_ms: Option<f64>,
    prefill_ms: Option<f64>,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    total_ms: Option<f64>,
    prefill_tokens_per_second: Option<f64>,
    decode_tokens_per_second: Option<f64>,
    decode_latency_ms: DecodeLatencySummary,
    memory: MemoryMetrics,
    raw_stdout: String,
    raw_stderr: String,
    blocker: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DecodeLatencySummary {
    samples: Vec<f64>,
    count: usize,
    p50_ms: Option<f64>,
    p95_ms: Option<f64>,
    max_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MemoryMetrics {
    mlx_active_memory_gb: Option<f64>,
    mlx_cache_memory_gb: Option<f64>,
    mlx_peak_memory_gb: Option<f64>,
    process_peak_rss_mb: Option<f64>,
}

#[derive(Debug, Deserialize)]
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
    mlx_cache_memory_gb: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
}

fn run_case(
    args: &Args,
    run_id: &str,
    context_tokens: usize,
) -> Result<BaselineRecord, Box<dyn std::error::Error>> {
    let mut command = Command::new("cargo");
    command.args([
        "run",
        "-p",
        "gemma4d-server",
        "--",
        "generate",
        "--model-path",
    ]);
    command.arg(&args.model_path);
    command.args([
        "--context-tokens",
        &context_tokens.to_string(),
        "--repeat-token",
        "1",
        "--max-context-tokens",
        &args.max_context_tokens.to_string(),
        "--max-new-tokens",
        &args.max_new_tokens.to_string(),
        "--json",
    ]);
    let display = format!(
        "cargo run -p gemma4d-server -- generate --model-path {} --context-tokens {context_tokens} --repeat-token 1 --max-context-tokens {} --max-new-tokens {} --json",
        args.model_path.display(),
        args.max_context_tokens,
        args.max_new_tokens
    );

    let started = Instant::now();
    let output = command.output()?;
    let command_wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let parsed = parse_generate_json(&stdout);
    let status = if output.status.success() && parsed.is_some() {
        "passed"
    } else if is_graceful_failure(&stderr) {
        "graceful_rejection"
    } else {
        "failed"
    };
    let blocker = if status == "passed" {
        None
    } else {
        Some(if stderr.is_empty() {
            "generate command did not produce parseable JSON".to_owned()
        } else {
            stderr.clone()
        })
    };

    Ok(record_from_result(
        run_id,
        context_tokens,
        args.max_new_tokens,
        display,
        output.status.code(),
        status,
        command_wall_ms,
        parsed,
        stdout,
        stderr,
        blocker,
    ))
}

fn blocked_case(args: &Args, run_id: &str, context_tokens: usize) -> BaselineRecord {
    let display = format!(
        "cargo run -p gemma4d-server -- generate --model-path {} --context-tokens {context_tokens} --repeat-token 1 --max-context-tokens {} --max-new-tokens {} --json",
        args.model_path.display(),
        args.max_context_tokens,
        args.max_new_tokens
    );
    record_from_result(
        run_id,
        context_tokens,
        args.max_new_tokens,
        display,
        None,
        "blocked_missing_model",
        0.0,
        None,
        String::new(),
        String::new(),
        Some(format!(
            "model path does not exist: {}",
            args.model_path.display()
        )),
    )
}

#[allow(clippy::too_many_arguments)]
fn record_from_result(
    run_id: &str,
    context_tokens: usize,
    max_new_tokens: usize,
    command: String,
    exit_code: Option<i32>,
    status: &str,
    command_wall_ms: f64,
    parsed: Option<GenerateJson>,
    raw_stdout: String,
    raw_stderr: String,
    blocker: Option<String>,
) -> BaselineRecord {
    let generated_tokens_observed = parsed
        .as_ref()
        .and_then(|parsed| parsed.generated_tokens.as_ref())
        .map(Vec::len)
        .unwrap_or(0);
    let total_ms = parsed.as_ref().and_then(|parsed| parsed.total_ms);
    let process_command_overhead_ms =
        total_ms.map(|total_ms| (command_wall_ms - total_ms).max(0.0));
    let input_tokens = parsed.as_ref().and_then(|parsed| parsed.input_tokens);
    let prefill_ms = parsed
        .as_ref()
        .and_then(|parsed| parsed.prefill_ms.or(parsed.ttft_ms));
    let prefill_tokens_per_second = match (input_tokens, prefill_ms) {
        (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens as f64 / (ms / 1000.0)),
        _ => None,
    };
    let samples = parsed
        .as_ref()
        .and_then(|parsed| parsed.decode_token_latencies_ms.clone())
        .unwrap_or_default();

    BaselineRecord {
        schema_version: 1,
        goal: "P00-performance-baseline",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        workload: "simple_chat_repeated_token",
        context_tokens,
        generated_tokens_requested: max_new_tokens,
        generated_tokens_observed,
        mode: MODE,
        command,
        exit_code,
        status: status.to_owned(),
        command_wall_ms,
        process_command_overhead_ms,
        input_tokens,
        model_load_ms: parsed.as_ref().and_then(|parsed| parsed.model_load_ms),
        prefill_ms,
        ttft_ms: parsed.as_ref().and_then(|parsed| parsed.ttft_ms),
        decode_ms: parsed.as_ref().and_then(|parsed| parsed.decode_ms),
        total_ms,
        prefill_tokens_per_second,
        decode_tokens_per_second: parsed.as_ref().and_then(|parsed| parsed.decode_tps),
        decode_latency_ms: decode_latency_summary(samples),
        memory: MemoryMetrics {
            mlx_active_memory_gb: parsed
                .as_ref()
                .and_then(|parsed| parsed.mlx_active_memory_gb),
            mlx_cache_memory_gb: parsed
                .as_ref()
                .and_then(|parsed| parsed.mlx_cache_memory_gb),
            mlx_peak_memory_gb: parsed.as_ref().and_then(|parsed| parsed.peak_memory_gb),
            process_peak_rss_mb: parsed.as_ref().and_then(|parsed| parsed.peak_rss_mb),
        },
        raw_stdout,
        raw_stderr,
        blocker,
    }
}

fn parse_generate_json(stdout: &str) -> Option<GenerateJson> {
    stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<GenerateJson>(line).ok())
}

fn decode_latency_summary(mut samples: Vec<f64>) -> DecodeLatencySummary {
    samples.sort_by(|a, b| a.total_cmp(b));
    let count = samples.len();
    DecodeLatencySummary {
        p50_ms: percentile(&samples, 0.50),
        p95_ms: percentile(&samples, 0.95),
        max_ms: samples.last().copied(),
        samples,
        count,
    }
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values.get(index).copied()
}

fn aggregate_status(records: &[BaselineRecord]) -> &'static str {
    if records
        .iter()
        .any(|record| record.status == "blocked_missing_model")
    {
        "blocked"
    } else if records.iter().all(|record| record.status == "passed") {
        "passed"
    } else {
        "failed"
    }
}

fn blockers_for(records: &[BaselineRecord]) -> Vec<String> {
    records
        .iter()
        .filter_map(|record| {
            record.blocker.as_ref().map(|blocker| {
                format!(
                    "{} tokens: {}; command: {}",
                    record.context_tokens, blocker, record.command
                )
            })
        })
        .collect()
}

fn capture_environment() -> Environment {
    Environment {
        machine: command_stdout("uname", &["-a"]).unwrap_or_else(|| "unknown".to_owned()),
        macos: command_stdout("sw_vers", &[]).unwrap_or_else(|| "unknown".to_owned()),
        rustc: command_stdout("rustc", &["-Vv"]).unwrap_or_else(|| "unknown".to_owned()),
        cargo: command_stdout("cargo", &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
        mlx_version: mlx_version(),
        git_commit: command_stdout("git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|| "unknown".to_owned()),
        git_status_short: command_stdout("git", &["status", "--short"])
            .unwrap_or_else(|| "unknown".to_owned()),
        hw_memsize_bytes: command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.trim().parse::<u64>().ok()),
    }
}

fn capture_model_identity(model_path: &Path) -> ModelIdentity {
    let safetensors = safetensors_inventory(model_path);
    let configured_revision = env::var("GEMMA4D_MODEL_REVISION")
        .unwrap_or_else(|_| "unavailable:GEMMA4D_MODEL_REVISION not set".to_owned());
    let revision_source = if configured_revision.starts_with("unavailable:") {
        "local_artifact_hash".to_owned()
    } else {
        "env:GEMMA4D_MODEL_REVISION".to_owned()
    };
    let config_sha256 = sha256_file_or_unavailable(&model_path.join("config.json"));
    let tokenizer_sha256 = sha256_file_or_unavailable(&model_path.join("tokenizer.json"));
    let tokenizer_config_sha256 =
        sha256_file_or_unavailable(&model_path.join("tokenizer_config.json"));
    let chat_template_sha256 = if model_path.join("chat_template.json").exists() {
        sha256_file_or_unavailable(&model_path.join("chat_template.json"))
    } else {
        sha256_file_or_unavailable(&model_path.join("tokenizer_config.json"))
    };
    let local_artifact_sha256 = sha256_hex(
        format!(
            "gemma4d:artifact:v1\nconfig={config_sha256}\ntokenizer={tokenizer_sha256}\ntokenizer_config={tokenizer_config_sha256}\nchat_template={chat_template_sha256}\nsafetensors={}\n",
            safetensors.inventory_sha256
        )
        .as_bytes(),
    );
    ModelIdentity {
        model_path: model_path.display().to_string(),
        exists: model_path.exists(),
        configured_revision,
        revision_source,
        local_artifact_sha256,
        config_sha256,
        tokenizer_sha256,
        tokenizer_config_sha256,
        chat_template_sha256,
        safetensors_inventory_sha256: safetensors.inventory_sha256,
        safetensors_file_count: safetensors.file_count,
        safetensors_total_bytes: safetensors.total_bytes,
    }
}

struct SafetensorsInventory {
    inventory_sha256: String,
    file_count: usize,
    total_bytes: u64,
}

fn safetensors_inventory(model_path: &Path) -> SafetensorsInventory {
    let mut entries = Vec::new();
    collect_safetensors(model_path, model_path, &mut entries);
    entries.sort();
    let total_bytes = entries
        .iter()
        .filter_map(|entry| entry.rsplit_once('\t'))
        .filter_map(|(_, bytes)| bytes.parse::<u64>().ok())
        .sum();
    let body = entries.join("\n");
    SafetensorsInventory {
        inventory_sha256: if entries.is_empty() {
            "unavailable:no safetensors files found".to_owned()
        } else {
            sha256_hex(body.as_bytes())
        },
        file_count: entries.len(),
        total_bytes,
    }
}

fn collect_safetensors(root: &Path, current: &Path, entries: &mut Vec<String>) {
    let Ok(read_dir) = fs::read_dir(current) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_safetensors(root, &path, entries);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let bytes = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            entries.push(format!("{}\t{}", relative.display(), bytes));
        }
    }
}

fn capture_relevant_environment() -> BTreeMap<String, Option<String>> {
    [
        "GEMMA4D_MLX_LM_PYTHON",
        "GEMMA4D_MODEL_PATH",
        "GEMMA4D_MODEL_REVISION",
        "GEMMA4D_USE_NATIVE_GRAPH",
        "GEMMA4D_REQUIRE_MLX",
        "GEMMA4D_FULL_MODEL_TESTS",
        "RUSTFLAGS",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), env::var(key).ok()))
    .collect()
}

fn mlx_version() -> String {
    let python = env::var("GEMMA4D_MLX_LM_PYTHON")
        .unwrap_or_else(|_| "/opt/homebrew/opt/mlx-lm/libexec/bin/python".to_owned());
    command_stdout(
        &python,
        &[
            "-c",
            "import mlx.core as mx; import mlx_lm; print(f'mlx={mx.__version__} mlx_lm={getattr(mlx_lm, \"__version__\", \"unknown\")}')",
        ],
    )
    .or_else(|| {
        command_stdout(
            "python3",
            &[
                "-c",
                "import mlx.core as mx; import mlx_lm; print(f'mlx={mx.__version__} mlx_lm={getattr(mlx_lm, \"__version__\", \"unknown\")}')",
            ],
        )
    })
    .unwrap_or_else(|| "unknown".to_owned())
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn sha256_file_or_unavailable(path: &Path) -> String {
    file_sha256(path).unwrap_or_else(|error| format!("unavailable:{error}"))
}

fn is_graceful_failure(stderr: &str) -> bool {
    stderr.contains("memory")
        || stderr.contains("Memory")
        || stderr.contains("unsupported")
        || stderr.contains("context")
}

fn write_jsonl(path: &Path, records: &[BaselineRecord]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn render_report(summary: &BaselineSummary) -> String {
    let mut out = String::new();
    out.push_str("# P00 Performance Baseline\n\n");
    out.push_str("## Status\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Mode: `{}`\n- Records: `{}`\n- Summary: `{}`\n\n",
        summary.status, summary.mode, summary.records_path, summary.summary_path
    ));
    out.push_str("## Environment\n\n");
    out.push_str("| Item | Value |\n|---|---|\n");
    out.push_str(&format!(
        "| Machine | `{}` |\n",
        escape_md(&summary.environment.machine)
    ));
    out.push_str(&format!(
        "| macOS | `{}` |\n",
        escape_md(&summary.environment.macos)
    ));
    out.push_str(&format!(
        "| Rust | `{}` |\n",
        escape_md(&summary.environment.rustc)
    ));
    out.push_str(&format!(
        "| Cargo | `{}` |\n",
        escape_md(&summary.environment.cargo)
    ));
    out.push_str(&format!(
        "| MLX | `{}` |\n",
        escape_md(&summary.environment.mlx_version)
    ));
    out.push_str(&format!(
        "| Git commit | `{}` |\n",
        escape_md(&summary.environment.git_commit)
    ));
    out.push_str(&format!(
        "| Git status | `{}` |\n",
        escape_md(&summary.environment.git_status_short)
    ));
    out.push_str(&format!(
        "| Hardware memory bytes | `{}` |\n\n",
        summary
            .environment
            .hw_memsize_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_owned())
    ));
    out.push_str("## Model Identity\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!(
        "| Model path | `{}` |\n",
        escape_md(&summary.model_identity.model_path)
    ));
    out.push_str(&format!(
        "| Configured revision | `{}` |\n",
        escape_md(&summary.model_identity.configured_revision)
    ));
    out.push_str(&format!(
        "| Revision source | `{}` |\n",
        escape_md(&summary.model_identity.revision_source)
    ));
    out.push_str(&format!(
        "| Local artifact SHA-256 | `{}` |\n",
        escape_md(&summary.model_identity.local_artifact_sha256)
    ));
    out.push_str(&format!(
        "| Config SHA-256 | `{}` |\n",
        escape_md(&summary.model_identity.config_sha256)
    ));
    out.push_str(&format!(
        "| Tokenizer SHA-256 | `{}` |\n",
        escape_md(&summary.model_identity.tokenizer_sha256)
    ));
    out.push_str(&format!(
        "| Tokenizer config SHA-256 | `{}` |\n",
        escape_md(&summary.model_identity.tokenizer_config_sha256)
    ));
    out.push_str(&format!(
        "| Chat template SHA-256 | `{}` |\n",
        escape_md(&summary.model_identity.chat_template_sha256)
    ));
    out.push_str(&format!(
        "| Safetensors inventory SHA-256 | `{}` |\n",
        escape_md(&summary.model_identity.safetensors_inventory_sha256)
    ));
    out.push_str(&format!(
        "| Safetensors files | `{}` |\n",
        summary.model_identity.safetensors_file_count
    ));
    out.push_str(&format!(
        "| Safetensors bytes | `{}` |\n\n",
        summary.model_identity.safetensors_total_bytes
    ));
    out.push_str("## Results\n\n");
    out.push_str("| Context | Generated | Status | Load ms | Prefill ms | Decode ms | Total ms | Command wall ms | Command overhead ms | Prefill tok/s | Decode tok/s | Decode p50 ms | Decode p95 ms | Peak MLX GB | Peak RSS MB |\n");
    out.push_str("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for case in &summary.cases {
        out.push_str(&format!(
            "| {} | {}/{} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            case.context_tokens,
            case.generated_tokens_observed,
            case.generated_tokens_requested,
            case.status,
            fmt_opt(case.model_load_ms),
            fmt_opt(case.prefill_ms),
            fmt_opt(case.decode_ms),
            fmt_opt(case.total_ms),
            fmt_num(case.command_wall_ms),
            fmt_opt(case.process_command_overhead_ms),
            fmt_opt(case.prefill_tokens_per_second),
            fmt_opt(case.decode_tokens_per_second),
            fmt_opt(case.decode_latency_ms.p50_ms),
            fmt_opt(case.decode_latency_ms.p95_ms),
            fmt_opt(case.memory.mlx_peak_memory_gb),
            fmt_opt(case.memory.process_peak_rss_mb),
        ));
    }
    out.push_str("\n## Relevant Environment\n\n");
    out.push_str("| Variable | Value |\n|---|---|\n");
    for (key, value) in &summary.relevant_environment {
        out.push_str(&format!(
            "| `{}` | `{}` |\n",
            escape_md(key),
            escape_md(value.as_deref().unwrap_or("unset"))
        ));
    }
    out.push_str("\n## Commands\n\n```text\n");
    for case in &summary.cases {
        out.push_str(&case.command);
        out.push('\n');
    }
    out.push_str("```\n\n## Measurement Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {}\n", escape_md(note)));
    }
    out.push_str("\n## Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("- None.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {}\n", escape_md(blocker)));
        }
    }
    out
}

fn render_blockers(summary: &BaselineSummary) -> String {
    let mut out = String::new();
    out.push_str("# P00 Blocker Report\n\n");
    out.push_str(&format!("- Status: `{}`\n", summary.status));
    if summary.blockers.is_empty() {
        out.push_str("- Blockers: none\n");
    } else {
        out.push_str("\n## Blockers\n\n");
        for blocker in &summary.blockers {
            out.push_str(&format!("- {}\n", escape_md(blocker)));
        }
        out.push_str("\n## Required Commands\n\n```text\n");
        for case in &summary.cases {
            out.push_str(&case.command);
            out.push('\n');
        }
        out.push_str("```\n");
    }
    out
}

fn parse_contexts(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| parse_positive_usize(part, "--contexts"))
        .collect()
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero").into());
    }
    Ok(parsed)
}

fn run_id() -> String {
    format!("p00-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn fmt_num(value: f64) -> String {
    format!("{value:.3}")
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(fmt_num).unwrap_or_else(|| "n/a".to_owned())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
