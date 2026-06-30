use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let report = run_matrix(&args)?;
    let jsonl_path = args.out_dir.join("records.jsonl");
    let report_path = args.out_dir.join("report.md");
    let summary_path = args.out_dir.join("summary.json");

    let mut jsonl = fs::File::create(&jsonl_path)?;
    for case in &report.cases {
        writeln!(jsonl, "{}", serde_json::to_string(case)?)?;
    }
    fs::write(&summary_path, serde_json::to_vec_pretty(&report)?)?;
    fs::write(
        &report_path,
        render_report(&report, &jsonl_path, &summary_path),
    )?;

    println!("M12 real tiny16 matrix: {}", report.status);
    println!("records: {}", jsonl_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());

    if report.status == "passed" {
        Ok(())
    } else {
        Err("M12 real tiny16 matrix failed".into())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from("benchmarks/out/M12/real-matrix");
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
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
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- [--out-dir PATH] [--model-path PATH]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }
        Ok(Self {
            out_dir,
            model_path,
        })
    }
}

#[derive(Debug, Serialize)]
struct MatrixReport {
    schema_version: u32,
    milestone: &'static str,
    status: &'static str,
    timestamp_unix: u64,
    environment: Environment,
    model_path: String,
    mode: &'static str,
    cases: Vec<MatrixCase>,
    known_limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct Environment {
    machine: String,
    macos: String,
    rustc: String,
    mlx_version: String,
    git_commit: String,
    hw_memsize_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct MatrixCase {
    timestamp_unix: u64,
    workload: &'static str,
    context_tokens: usize,
    generated_tokens_requested: usize,
    generated_tokens_observed: usize,
    mode: &'static str,
    command: String,
    exit_code: Option<i32>,
    status: String,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    prefill_tps: Option<f64>,
    decode_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    raw_stdout: String,
    raw_stderr: String,
    note: String,
}

#[derive(Debug, serde::Deserialize)]
struct GenerateJson {
    input_tokens: usize,
    generated_tokens: Vec<i32>,
    ttft_ms: f64,
    decode_ms: f64,
    decode_tps: f64,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
}

fn run_matrix(args: &Args) -> Result<MatrixReport, Box<dyn std::error::Error>> {
    let contexts = [
        (1024, 128, "standard_128_decode"),
        (4096, 128, "standard_128_decode"),
        (8192, 128, "standard_128_decode"),
        (16_384, 128, "standard_128_decode"),
        (32_768, 1, "32k_memory_probe_one_decode_token"),
    ];

    let mut cases = Vec::new();
    for (context_tokens, max_new_tokens, note) in contexts {
        cases.push(run_case(
            args,
            context_tokens,
            max_new_tokens,
            "simple_chat_repeated_token",
            note,
        )?);
    }

    let required_contexts_passed = cases.iter().all(|case| {
        if case.context_tokens == 32_768 {
            case.status == "passed" || case.status == "graceful_rejection"
        } else {
            case.status == "passed"
        }
    });

    Ok(MatrixReport {
        schema_version: 1,
        milestone: "M12",
        status: if required_contexts_passed {
            "passed"
        } else {
            "failed"
        },
        timestamp_unix: unix_now(),
        environment: capture_environment(),
        model_path: args.model_path.display().to_string(),
        mode: "target_greedy_mlx_lm_helper_via_c_abi",
        cases,
        known_limitations: vec![
            "32K is run as a one-token decode memory probe to protect tiny16 headroom.",
            "The current target path uses the MLX-LM helper through the C ABI; the hand-written native graph remains a tracked follow-up.",
        ],
    })
}

fn run_case(
    args: &Args,
    context_tokens: usize,
    max_new_tokens: usize,
    workload: &'static str,
    note: &'static str,
) -> Result<MatrixCase, Box<dyn std::error::Error>> {
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
        "32768",
        "--max-new-tokens",
        &max_new_tokens.to_string(),
        "--json",
    ]);
    let display = format!(
        "cargo run -p gemma4d-server -- generate --model-path {} --context-tokens {context_tokens} --repeat-token 1 --max-context-tokens 32768 --max-new-tokens {max_new_tokens} --json",
        args.model_path.display()
    );

    let started = Instant::now();
    let output = command.output()?;
    let wall_seconds = started.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let parsed = parse_generate_json(&stdout);
    let generated_tokens_observed = parsed
        .as_ref()
        .map(|parsed| parsed.generated_tokens.len())
        .unwrap_or(0);
    let status = if output.status.success() && parsed.is_some() {
        "passed"
    } else if is_graceful_failure(&stderr) {
        "graceful_rejection"
    } else {
        "failed"
    };
    let prefill_tps = parsed.as_ref().and_then(|parsed| {
        (parsed.ttft_ms > 0.0).then_some(parsed.input_tokens as f64 / (parsed.ttft_ms / 1000.0))
    });

    Ok(MatrixCase {
        timestamp_unix: unix_now(),
        workload,
        context_tokens,
        generated_tokens_requested: max_new_tokens,
        generated_tokens_observed,
        mode: "target_greedy_mlx_lm_helper_via_c_abi",
        command: display,
        exit_code: output.status.code(),
        status: status.to_owned(),
        ttft_ms: parsed.as_ref().map(|parsed| parsed.ttft_ms),
        decode_ms: parsed.as_ref().map(|parsed| parsed.decode_ms),
        prefill_tps,
        decode_tps: parsed.as_ref().map(|parsed| parsed.decode_tps),
        peak_memory_gb: parsed.as_ref().map(|parsed| parsed.peak_memory_gb),
        peak_rss_mb: parsed.as_ref().map(|parsed| parsed.peak_rss_mb),
        raw_stdout: stdout,
        raw_stderr: stderr,
        note: format!("{note}; wall_seconds={wall_seconds:.3}"),
    })
}

fn parse_generate_json(stdout: &str) -> Option<GenerateJson> {
    stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<GenerateJson>(line).ok())
}

fn is_graceful_failure(stderr: &str) -> bool {
    stderr.contains("memory")
        || stderr.contains("Memory")
        || stderr.contains("unsupported")
        || stderr.contains("context")
}

fn capture_environment() -> Environment {
    Environment {
        machine: command_stdout("uname", &["-a"]).unwrap_or_else(|| "unknown".to_owned()),
        macos: command_stdout("sw_vers", &[]).unwrap_or_else(|| "unknown".to_owned()),
        rustc: command_stdout("rustc", &["-Vv"]).unwrap_or_else(|| "unknown".to_owned()),
        mlx_version: mlx_version(),
        git_commit: command_stdout("git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|| "unknown".to_owned()),
        hw_memsize_bytes: command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.trim().parse::<u64>().ok()),
    }
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

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn render_report(report: &MatrixReport, jsonl_path: &Path, summary_path: &Path) -> String {
    let mut out = String::new();
    out.push_str("# M12 Real tiny16 Matrix\n\n");
    out.push_str("## Status\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Mode: `{}`\n- JSONL: `{}`\n- Summary: `{}`\n\n",
        report.status,
        report.mode,
        jsonl_path.display(),
        summary_path.display()
    ));
    out.push_str("## Environment\n\n");
    out.push_str("| Item | Value |\n|---|---|\n");
    out.push_str(&format!(
        "| Machine | `{}` |\n",
        escape_md(&report.environment.machine)
    ));
    out.push_str(&format!(
        "| macOS | `{}` |\n",
        escape_md(&report.environment.macos)
    ));
    out.push_str(&format!(
        "| Rust | `{}` |\n",
        escape_md(&report.environment.rustc)
    ));
    out.push_str(&format!(
        "| MLX | `{}` |\n",
        escape_md(&report.environment.mlx_version)
    ));
    out.push_str(&format!(
        "| Model | `{}` |\n\n",
        escape_md(&report.model_path)
    ));
    out.push_str("## Results\n\n");
    out.push_str("| Context | Generated | Status | TTFT ms | Prefill tok/s | Decode tok/s | Peak native GB | Peak RSS MB | Notes |\n");
    out.push_str("|---:|---:|---|---:|---:|---:|---:|---:|---|\n");
    for case in &report.cases {
        out.push_str(&format!(
            "| {} | {}/{} | `{}` | {} | {} | {} | {} | {} | {} |\n",
            case.context_tokens,
            case.generated_tokens_observed,
            case.generated_tokens_requested,
            case.status,
            fmt_opt(case.ttft_ms),
            fmt_opt(case.prefill_tps),
            fmt_opt(case.decode_tps),
            fmt_opt(case.peak_memory_gb),
            fmt_opt(case.peak_rss_mb),
            escape_md(&case.note)
        ));
    }
    out.push_str("\n## Commands\n\n```text\n");
    for case in &report.cases {
        out.push_str(&case.command);
        out.push('\n');
    }
    out.push_str("```\n\n## Known Limitations\n\n");
    for item in &report.known_limitations {
        out.push_str(&format!("- {}\n", escape_md(item)));
    }
    out
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
