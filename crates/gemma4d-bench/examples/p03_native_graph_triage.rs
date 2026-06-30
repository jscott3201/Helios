use std::{
    collections::BTreeMap,
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P03-native-graph-triage";
const MODE: &str = "native_graph_vs_helper_cli_triage";
const LOGIT_TOLERANCE: f64 = 0.5;
const MEMORY_CLIFF_GB: f64 = 12.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let environment = capture_environment();
    let probes = probes(args.include_16k);
    let mut records = Vec::new();
    let mut blockers = Vec::new();

    if !args.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            args.model_path.display()
        ));
    } else {
        for probe in &probes {
            records.push(run_probe(&args, &run_id, probe)?);
        }
    }

    blockers.extend(blockers_for_records(&records));
    let claims = claim_inventory(&records);
    let status = if blockers.is_empty() {
        "passed"
    } else {
        "triage_complete_with_blockers"
    };

    let summary = P03Summary {
        schema_version: 1,
        goal: "P03-native-graph-triage",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        model_path: args.model_path.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        environment,
        relevant_environment: capture_relevant_environment(),
        logit_tolerance: LOGIT_TOLERANCE,
        memory_cliff_gb: MEMORY_CLIFF_GB,
        probes_requested: probes.len(),
        claims,
        records,
        blockers,
        measurement_notes: vec![
            "helper runs use the default gemma4d generate path, which delegates through the MLX-LM helper behind the C ABI.",
            "native runs set GEMMA4D_REQUIRE_MLX=1 and GEMMA4D_USE_NATIVE_GRAPH=1 for the same tokenizer-controlled input.",
            "generated_logits are diagnostic values emitted by gemma4d generate from the FFI StepResult greedy_logit.",
            "1K/4K/8K probes request one generated token to triage native full-recompute memory and prefill cost without switching defaults.",
            "peak_rss_mb is currently meaningful for the helper process; native graph reports MLX peak memory and uses 0 RSS until native RSS reporting is added.",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P03 native graph triage: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    Ok(())
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    max_context_tokens: usize,
    include_16k: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut max_context_tokens = 32_768;
        let mut include_16k = false;

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
                "--max-context-tokens" => {
                    let value = args.next().ok_or("--max-context-tokens requires a value")?;
                    max_context_tokens = parse_positive_usize(&value, "--max-context-tokens")?;
                }
                "--include-16k" => {
                    include_16k = true;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example p03_native_graph_triage -- [--out-dir PATH] [--model-path PATH] [--max-context-tokens N] [--include-16k]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        Ok(Self {
            out_dir,
            model_path,
            max_context_tokens,
            include_16k,
        })
    }
}

#[derive(Debug, Clone)]
struct Probe {
    id: &'static str,
    description: &'static str,
    input: ProbeInput,
    max_new_tokens: usize,
}

#[derive(Debug, Clone)]
enum ProbeInput {
    TokenIds(Vec<i32>),
    RepeatToken {
        token_id: i32,
        context_tokens: usize,
    },
}

impl ProbeInput {
    fn nominal_tokens(&self) -> usize {
        match self {
            Self::TokenIds(tokens) => tokens.len(),
            Self::RepeatToken { context_tokens, .. } => *context_tokens,
        }
    }

    fn display(&self) -> String {
        match self {
            Self::TokenIds(tokens) => format!(
                "token_ids:{}",
                tokens
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::RepeatToken {
                token_id,
                context_tokens,
            } => format!("repeat_token:{token_id}x{context_tokens}"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct P03Summary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    model_path: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    logit_tolerance: f64,
    memory_cliff_gb: f64,
    probes_requested: usize,
    claims: ClaimInventory,
    records: Vec<P03Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct P03Record {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    probe_id: String,
    description: String,
    input_spec: String,
    nominal_input_tokens: usize,
    max_new_tokens: usize,
    mode: &'static str,
    helper: RunRecord,
    native: RunRecord,
    comparison: Comparison,
}

#[derive(Debug, Clone, Serialize)]
struct RunRecord {
    backend: &'static str,
    command: String,
    exit_code: Option<i32>,
    status: String,
    wall_ms: f64,
    input_tokens: Option<usize>,
    generated_tokens: Vec<i32>,
    generated_logits: Vec<f64>,
    model_load_ms: Option<f64>,
    prefill_ms: Option<f64>,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    total_ms: Option<f64>,
    decode_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    raw_stdout: String,
    raw_stderr: String,
}

#[derive(Debug, Clone, Serialize)]
struct Comparison {
    status: String,
    token_match: bool,
    first_token_mismatch: Option<TokenMismatch>,
    logit_count_compared: usize,
    max_logit_abs_delta: Option<f64>,
    mean_logit_abs_delta: Option<f64>,
    native_total_minus_helper_total_ms: Option<f64>,
    native_prefill_minus_helper_prefill_ms: Option<f64>,
    native_decode_minus_helper_decode_ms: Option<f64>,
    native_peak_minus_helper_peak_gb: Option<f64>,
    native_hotspot: Option<String>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TokenMismatch {
    index: usize,
    helper_token: Option<i32>,
    native_token: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct ClaimInventory {
    confirmed_parity: Vec<String>,
    numerical_drift: Vec<String>,
    unsupported_ops: Vec<String>,
    memory_cliffs: Vec<String>,
    measured_hotspots: Vec<String>,
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

#[derive(Debug, Default, Deserialize)]
struct GenerateJson {
    input_tokens: Option<usize>,
    generated_tokens: Option<Vec<i32>>,
    generated_logits: Option<Vec<f64>>,
    model_load_ms: Option<f64>,
    prefill_ms: Option<f64>,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    total_ms: Option<f64>,
    decode_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
}

fn probes(include_16k: bool) -> Vec<Probe> {
    let mut probes = vec![
        Probe {
            id: "hello_smoke",
            description: "M04 tokenizer-controlled Hello prompt; eight-token greedy parity.",
            input: ProbeInput::TokenIds(vec![9259]),
            max_new_tokens: 8,
        },
        Probe {
            id: "hello_reference_prefix",
            description: "M04 Hello plus two reference generated tokens; one-token continuation parity.",
            input: ProbeInput::TokenIds(vec![9259, 236772, 236772]),
            max_new_tokens: 1,
        },
        Probe {
            id: "repeat_9259_1k",
            description: "Native full-recompute 1K prefill/memory probe.",
            input: ProbeInput::RepeatToken {
                token_id: 9259,
                context_tokens: 1024,
            },
            max_new_tokens: 1,
        },
        Probe {
            id: "repeat_9259_4k",
            description: "Native full-recompute 4K prefill/memory probe.",
            input: ProbeInput::RepeatToken {
                token_id: 9259,
                context_tokens: 4096,
            },
            max_new_tokens: 1,
        },
        Probe {
            id: "repeat_9259_8k",
            description: "Native full-recompute 8K prefill/memory probe.",
            input: ProbeInput::RepeatToken {
                token_id: 9259,
                context_tokens: 8192,
            },
            max_new_tokens: 1,
        },
    ];
    if include_16k {
        probes.push(Probe {
            id: "repeat_9259_16k",
            description: "Optional native full-recompute 16K prefill/memory cliff probe.",
            input: ProbeInput::RepeatToken {
                token_id: 9259,
                context_tokens: 16_384,
            },
            max_new_tokens: 1,
        });
    }
    probes
}

fn run_probe(
    args: &Args,
    run_id: &str,
    probe: &Probe,
) -> Result<P03Record, Box<dyn std::error::Error>> {
    let helper = run_backend(args, probe, Backend::Helper)?;
    let native = run_backend(args, probe, Backend::Native)?;
    let comparison = compare_runs(probe, &helper, &native);

    Ok(P03Record {
        schema_version: 1,
        goal: "P03-native-graph-triage",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        probe_id: probe.id.to_owned(),
        description: probe.description.to_owned(),
        input_spec: probe.input.display(),
        nominal_input_tokens: probe.input.nominal_tokens(),
        max_new_tokens: probe.max_new_tokens,
        mode: MODE,
        helper,
        native,
        comparison,
    })
}

#[derive(Debug, Clone, Copy)]
enum Backend {
    Helper,
    Native,
}

fn run_backend(
    args: &Args,
    probe: &Probe,
    backend: Backend,
) -> Result<RunRecord, Box<dyn std::error::Error>> {
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
    append_probe_args(&mut command, probe, args.max_context_tokens);
    if matches!(backend, Backend::Native) {
        command.env("GEMMA4D_REQUIRE_MLX", "1");
        command.env("GEMMA4D_USE_NATIVE_GRAPH", "1");
    }

    let display = command_display(args, probe, backend);
    let started = Instant::now();
    let output = command.output()?;
    let wall_ms = duration_ms(started.elapsed());
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let parsed = parse_generate_json(&stdout);
    let status = if output.status.success() && parsed.is_some() {
        "ok"
    } else {
        "failed"
    };
    let parsed = parsed.unwrap_or_default();

    Ok(RunRecord {
        backend: match backend {
            Backend::Helper => "helper",
            Backend::Native => "native",
        },
        command: display,
        exit_code: output.status.code(),
        status: status.to_owned(),
        wall_ms,
        input_tokens: parsed.input_tokens,
        generated_tokens: parsed.generated_tokens.unwrap_or_default(),
        generated_logits: parsed.generated_logits.unwrap_or_default(),
        model_load_ms: parsed.model_load_ms,
        prefill_ms: parsed.prefill_ms,
        ttft_ms: parsed.ttft_ms,
        decode_ms: parsed.decode_ms,
        total_ms: parsed.total_ms,
        decode_tps: parsed.decode_tps,
        peak_memory_gb: parsed.peak_memory_gb,
        peak_rss_mb: parsed.peak_rss_mb,
        raw_stdout: stdout,
        raw_stderr: stderr,
    })
}

fn append_probe_args(command: &mut Command, probe: &Probe, max_context_tokens: usize) {
    match &probe.input {
        ProbeInput::TokenIds(tokens) => {
            command.arg("--token-ids").arg(
                tokens
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        ProbeInput::RepeatToken {
            token_id,
            context_tokens,
        } => {
            command
                .arg("--repeat-token")
                .arg(token_id.to_string())
                .arg("--context-tokens")
                .arg(context_tokens.to_string());
        }
    }
    command
        .arg("--max-context-tokens")
        .arg(max_context_tokens.to_string())
        .arg("--max-new-tokens")
        .arg(probe.max_new_tokens.to_string())
        .arg("--json");
}

fn command_display(args: &Args, probe: &Probe, backend: Backend) -> String {
    let mut parts = Vec::new();
    if matches!(backend, Backend::Native) {
        parts.push("GEMMA4D_REQUIRE_MLX=1".to_owned());
        parts.push("GEMMA4D_USE_NATIVE_GRAPH=1".to_owned());
    }
    parts.extend([
        "cargo".to_owned(),
        "run".to_owned(),
        "-p".to_owned(),
        "gemma4d-server".to_owned(),
        "--".to_owned(),
        "generate".to_owned(),
        "--model-path".to_owned(),
        args.model_path.display().to_string(),
    ]);
    match &probe.input {
        ProbeInput::TokenIds(tokens) => {
            parts.push("--token-ids".to_owned());
            parts.push(
                tokens
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        ProbeInput::RepeatToken {
            token_id,
            context_tokens,
        } => {
            parts.push("--repeat-token".to_owned());
            parts.push(token_id.to_string());
            parts.push("--context-tokens".to_owned());
            parts.push(context_tokens.to_string());
        }
    }
    parts.extend([
        "--max-context-tokens".to_owned(),
        args.max_context_tokens.to_string(),
        "--max-new-tokens".to_owned(),
        probe.max_new_tokens.to_string(),
        "--json".to_owned(),
    ]);
    parts.join(" ")
}

fn compare_runs(probe: &Probe, helper: &RunRecord, native: &RunRecord) -> Comparison {
    let mut blockers = Vec::new();
    if helper.status != "ok" {
        blockers.push(format!("{} helper run failed", probe.id));
    }
    if native.status != "ok" {
        blockers.push(format!("{} native run failed", probe.id));
    }
    let first_token_mismatch =
        first_token_mismatch(&helper.generated_tokens, &native.generated_tokens);
    let token_match = first_token_mismatch.is_none()
        && helper.generated_tokens.len() == native.generated_tokens.len();
    if helper.status == "ok" && native.status == "ok" && !token_match {
        blockers.push(format!(
            "{} native tokens differ from helper tokens",
            probe.id
        ));
    }
    if let Some(peak) = native.peak_memory_gb
        && peak >= MEMORY_CLIFF_GB
    {
        blockers.push(format!(
            "{} native peak memory {:.3} GB crosses {:.1} GB tiny16 cliff threshold",
            probe.id, peak, MEMORY_CLIFF_GB
        ));
    }

    let logit_stats = logit_delta_stats(&helper.generated_logits, &native.generated_logits);
    let logit_count_compared = logit_stats.count;
    if token_match
        && logit_stats
            .max_abs_delta
            .is_some_and(|delta| delta > LOGIT_TOLERANCE)
    {
        blockers.push(format!(
            "{} token parity holds but greedy logit delta exceeds tolerance {:.3}",
            probe.id, LOGIT_TOLERANCE
        ));
    }

    let status = if helper.status != "ok" || native.status != "ok" {
        "unsupported_or_runtime_failure"
    } else if !token_match {
        "token_mismatch"
    } else if logit_stats
        .max_abs_delta
        .is_some_and(|delta| delta > LOGIT_TOLERANCE)
    {
        "logit_drift"
    } else if native
        .peak_memory_gb
        .is_some_and(|peak| peak >= MEMORY_CLIFF_GB)
    {
        "memory_cliff"
    } else {
        "parity_confirmed"
    };

    Comparison {
        status: status.to_owned(),
        token_match,
        first_token_mismatch,
        logit_count_compared,
        max_logit_abs_delta: logit_stats.max_abs_delta,
        mean_logit_abs_delta: logit_stats.mean_abs_delta,
        native_total_minus_helper_total_ms: delta(helper.total_ms, native.total_ms),
        native_prefill_minus_helper_prefill_ms: delta(helper.prefill_ms, native.prefill_ms),
        native_decode_minus_helper_decode_ms: delta(helper.decode_ms, native.decode_ms),
        native_peak_minus_helper_peak_gb: delta(helper.peak_memory_gb, native.peak_memory_gb),
        native_hotspot: native_hotspot(native),
        blockers,
    }
}

fn first_token_mismatch(helper: &[i32], native: &[i32]) -> Option<TokenMismatch> {
    let max_len = helper.len().max(native.len());
    (0..max_len).find_map(|index| {
        let helper_token = helper.get(index).copied();
        let native_token = native.get(index).copied();
        (helper_token != native_token).then_some(TokenMismatch {
            index,
            helper_token,
            native_token,
        })
    })
}

#[derive(Debug, Default)]
struct LogitStats {
    count: usize,
    max_abs_delta: Option<f64>,
    mean_abs_delta: Option<f64>,
}

fn logit_delta_stats(helper: &[f64], native: &[f64]) -> LogitStats {
    let count = helper.len().min(native.len());
    if count == 0 {
        return LogitStats::default();
    }
    let deltas = helper
        .iter()
        .zip(native.iter())
        .map(|(helper, native)| (native - helper).abs())
        .collect::<Vec<_>>();
    let sum = deltas.iter().sum::<f64>();
    LogitStats {
        count,
        max_abs_delta: deltas.iter().copied().reduce(f64::max),
        mean_abs_delta: Some(sum / count as f64),
    }
}

fn native_hotspot(native: &RunRecord) -> Option<String> {
    let mut parts = [
        ("model_load_ms", native.model_load_ms),
        ("prefill_ms", native.prefill_ms),
        ("decode_ms", native.decode_ms),
    ]
    .into_iter()
    .filter_map(|(label, value)| value.map(|value| (label, value)))
    .collect::<Vec<_>>();
    parts.sort_by(|left, right| right.1.total_cmp(&left.1));
    parts
        .first()
        .map(|(label, value)| format!("{label} dominates at {value:.3} ms"))
}

fn parse_generate_json(stdout: &str) -> Option<GenerateJson> {
    stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<GenerateJson>(line).ok())
}

fn claim_inventory(records: &[P03Record]) -> ClaimInventory {
    let mut confirmed_parity = Vec::new();
    let mut numerical_drift = Vec::new();
    let mut unsupported_ops = Vec::new();
    let mut memory_cliffs = Vec::new();
    let mut measured_hotspots = Vec::new();

    for record in records {
        match record.comparison.status.as_str() {
            "parity_confirmed" => confirmed_parity.push(format!(
                "{}: native tokens and logits match helper within tolerance",
                record.probe_id
            )),
            "logit_drift" => numerical_drift.push(format!(
                "{}: token parity holds, max greedy logit delta {}",
                record.probe_id,
                fmt_opt(record.comparison.max_logit_abs_delta)
            )),
            "unsupported_or_runtime_failure" => unsupported_ops.push(format!(
                "{}: helper status `{}` native status `{}`",
                record.probe_id, record.helper.status, record.native.status
            )),
            "memory_cliff" => memory_cliffs.push(format!(
                "{}: native peak {} GB",
                record.probe_id,
                fmt_opt(record.native.peak_memory_gb)
            )),
            "token_mismatch" => numerical_drift.push(format!(
                "{}: token mismatch at {:?}",
                record.probe_id, record.comparison.first_token_mismatch
            )),
            _ => {}
        }
        if let Some(hotspot) = &record.comparison.native_hotspot {
            measured_hotspots.push(format!("{}: {}", record.probe_id, hotspot));
        }
        if let Some(peak) = record.native.peak_memory_gb
            && peak >= MEMORY_CLIFF_GB
            && !memory_cliffs
                .iter()
                .any(|claim| claim.starts_with(&record.probe_id))
        {
            memory_cliffs.push(format!("{}: native peak {peak:.3} GB", record.probe_id));
        }
    }

    ClaimInventory {
        confirmed_parity,
        numerical_drift,
        unsupported_ops,
        memory_cliffs,
        measured_hotspots,
    }
}

fn blockers_for_records(records: &[P03Record]) -> Vec<String> {
    records
        .iter()
        .flat_map(|record| record.comparison.blockers.iter().cloned())
        .collect()
}

fn write_jsonl(path: &Path, records: &[P03Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn render_report(summary: &P03Summary) -> String {
    let mut out = String::new();
    out.push_str("# P03 Native Graph Triage\n\n");
    out.push_str("## Status\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Mode: `{}`\n- Records: `{}`\n- Summary: `{}`\n- Blockers: `{}`\n- Logit tolerance: `{:.3}`\n- Memory cliff threshold: `{:.1} GB`\n\n",
        summary.status,
        summary.mode,
        summary.records_path,
        summary.summary_path,
        summary.blockers_path,
        summary.logit_tolerance,
        summary.memory_cliff_gb,
    ));
    out.push_str("## Claim Inventory\n\n");
    render_claim_list(
        &mut out,
        "Confirmed Parity",
        &summary.claims.confirmed_parity,
    );
    render_claim_list(&mut out, "Numerical Drift", &summary.claims.numerical_drift);
    render_claim_list(
        &mut out,
        "Unsupported Ops / Runtime Failures",
        &summary.claims.unsupported_ops,
    );
    render_claim_list(&mut out, "Memory Cliffs", &summary.claims.memory_cliffs);
    render_claim_list(
        &mut out,
        "Measured Hotspots",
        &summary.claims.measured_hotspots,
    );

    out.push_str("## Probe Results\n\n");
    out.push_str("| Probe | Tokens | Gen | Status | Token Match | Max Logit Delta | Helper Total ms | Native Total ms | Total Delta ms | Helper Prefill ms | Native Prefill ms | Helper Decode ms | Native Decode ms | Helper Peak GB | Native Peak GB | Native Hotspot |\n");
    out.push_str("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | {} | {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            record.probe_id,
            record.nominal_input_tokens,
            record.max_new_tokens,
            record.comparison.status,
            record.comparison.token_match,
            fmt_opt(record.comparison.max_logit_abs_delta),
            fmt_opt(record.helper.total_ms),
            fmt_opt(record.native.total_ms),
            fmt_opt(record.comparison.native_total_minus_helper_total_ms),
            fmt_opt(record.helper.prefill_ms),
            fmt_opt(record.native.prefill_ms),
            fmt_opt(record.helper.decode_ms),
            fmt_opt(record.native.decode_ms),
            fmt_opt(record.helper.peak_memory_gb),
            fmt_opt(record.native.peak_memory_gb),
            record.comparison.native_hotspot.as_deref().unwrap_or("n/a"),
        ));
    }

    out.push_str("\n## Token And Logit Detail\n\n");
    out.push_str("| Probe | Helper Tokens | Native Tokens | Helper Logits | Native Logits | Mean Logit Delta | First Mismatch |\n");
    out.push_str("|---|---|---|---|---|---:|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | `{}` |\n",
            record.probe_id,
            tokens_short(&record.helper.generated_tokens),
            tokens_short(&record.native.generated_tokens),
            logits_short(&record.helper.generated_logits),
            logits_short(&record.native.generated_logits),
            fmt_opt(record.comparison.mean_logit_abs_delta),
            record
                .comparison
                .first_token_mismatch
                .as_ref()
                .map(|mismatch| format!(
                    "index={} helper={:?} native={:?}",
                    mismatch.index, mismatch.helper_token, mismatch.native_token
                ))
                .unwrap_or_else(|| "none".to_owned()),
        ));
    }

    out.push_str("\n## Commands\n\n```text\n");
    for record in &summary.records {
        out.push_str(&record.helper.command);
        out.push('\n');
        out.push_str(&record.native.command);
        out.push('\n');
    }
    out.push_str("```\n\n");
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
    out.push_str("\n## Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {note}\n"));
    }
    if !summary.blockers.is_empty() {
        out.push_str("\n## Blockers\n\n");
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn render_claim_list(out: &mut String, title: &str, claims: &[String]) {
    out.push_str(&format!("### {title}\n\n"));
    if claims.is_empty() {
        out.push_str("- None recorded in this run.\n\n");
    } else {
        for claim in claims {
            out.push_str(&format!("- {claim}\n"));
        }
        out.push('\n');
    }
}

fn render_blockers(summary: &P03Summary) -> String {
    if summary.blockers.is_empty() {
        return "No blockers recorded.\n".to_owned();
    }
    let mut out = String::new();
    out.push_str("# P03 Blockers\n\n");
    for blocker in &summary.blockers {
        out.push_str(&format!("- {blocker}\n"));
    }
    out
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("{flag} must be a positive integer: {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero").into());
    }
    Ok(parsed)
}

fn delta(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(right - left),
        _ => None,
    }
}

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
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

fn tokens_short(tokens: &[i32]) -> String {
    const LIMIT: usize = 12;
    if tokens.len() <= LIMIT {
        return format!("{tokens:?}");
    }
    let head = tokens
        .iter()
        .take(LIMIT)
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{head}, ...; len={}]", tokens.len())
}

fn logits_short(logits: &[f64]) -> String {
    const LIMIT: usize = 8;
    let values = logits
        .iter()
        .take(LIMIT)
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",");
    if logits.len() <= LIMIT {
        format!("[{values}]")
    } else {
        format!("[{values}, ...; len={}]", logits.len())
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', "<br>")
}

fn run_id() -> String {
    format!("p03-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after Unix epoch")
        .as_secs()
}
