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
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P04-incremental-native-kv";
const MODE: &str = "incremental_native_kv_vs_helper_cli";
const LOGIT_TOLERANCE: f64 = 0.5;
const MEMORY_CLIFF_GB: f64 = 14.0;
const MAX_P50_8K_TO_1K_RATIO: f64 = 3.0;
const MAX_P95_8K_TO_1K_RATIO: f64 = 4.0;
const WARMUP_DECODE_SAMPLES: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let environment = capture_environment();
    let probes = probes(&args);
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
    let decode_growth = decode_growth(&records);
    if let Some(growth) = &decode_growth {
        blockers.extend(growth.blockers.clone());
    } else if args.contexts.len() >= 2 && args.max_new_tokens > 1 && blockers.is_empty() {
        blockers.push("native decode growth evidence is unavailable".to_owned());
    }

    let claims = claim_inventory(&records, decode_growth.as_ref());
    let status = if blockers.is_empty() {
        "passed"
    } else {
        "failed"
    };

    let summary = P04Summary {
        schema_version: 1,
        goal: "P04-incremental-native-kv",
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
        contexts: args.contexts.clone(),
        small_max_new_tokens: args.small_max_new_tokens,
        context_max_new_tokens: args.max_new_tokens,
        max_context_tokens: args.max_context_tokens,
        logit_tolerance: LOGIT_TOLERANCE,
        memory_cliff_gb: MEMORY_CLIFF_GB,
        max_p50_8k_to_1k_ratio: MAX_P50_8K_TO_1K_RATIO,
        max_p95_8k_to_1k_ratio: MAX_P95_8K_TO_1K_RATIO,
        warmup_decode_samples_discarded: WARMUP_DECODE_SAMPLES,
        decode_growth,
        probes_requested: probes.len(),
        claims,
        records,
        blockers,
        measurement_notes: vec![
            "helper runs use the default gemma4d generate path, preserving the helper-backed fallback.",
            "native runs set GEMMA4D_REQUIRE_MLX=1 and GEMMA4D_USE_NATIVE_GRAPH=1 to exercise the hand-written graph.",
            "prefill_ms includes native KV materialization; decode_token_latencies_ms contains one entry for each decode_one call after the first greedy token.",
            "active_kv_bytes is the native graph estimate of resident per-layer KV state after prefill/decode.",
            "raw decode latency samples are retained; decode growth uses steady-state p50/p95 after discarding the first four decode_one samples for MLX/JIT/cache warmup.",
            "greedy logit deltas are diagnostic unless generated token IDs diverge.",
            "peak_rss_mb is currently meaningful for the helper process; native graph reports MLX peak memory and uses 0 RSS until native RSS reporting is added.",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P04 incremental native KV: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if summary.status == "failed" {
        Err("P04 incremental native KV checks failed".into())
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
    small_max_new_tokens: usize,
    max_context_tokens: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut contexts = vec![1024, 4096, 8192];
        let mut max_new_tokens = 16;
        let mut small_max_new_tokens = 8;
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
                "--small-max-new-tokens" => {
                    let value = args
                        .next()
                        .ok_or("--small-max-new-tokens requires a value")?;
                    small_max_new_tokens = parse_positive_usize(&value, "--small-max-new-tokens")?;
                }
                "--max-context-tokens" => {
                    let value = args.next().ok_or("--max-context-tokens requires a value")?;
                    max_context_tokens = parse_positive_usize(&value, "--max-context-tokens")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example p04_incremental_native_kv -- [--out-dir PATH] [--model-path PATH] [--contexts 1024,4096,8192] [--max-new-tokens N] [--small-max-new-tokens N] [--max-context-tokens N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        if contexts.is_empty() {
            return Err("--contexts must include at least one context".into());
        }
        contexts.sort_unstable();
        contexts.dedup();

        Ok(Self {
            out_dir,
            model_path,
            contexts,
            max_new_tokens,
            small_max_new_tokens,
            max_context_tokens,
        })
    }
}

#[derive(Debug, Clone)]
struct Probe {
    id: String,
    description: String,
    input: ProbeInput,
    max_new_tokens: usize,
    kind: ProbeKind,
}

#[derive(Debug, Clone)]
enum ProbeKind {
    Small,
    Context,
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
struct P04Summary {
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
    contexts: Vec<usize>,
    small_max_new_tokens: usize,
    context_max_new_tokens: usize,
    max_context_tokens: usize,
    logit_tolerance: f64,
    memory_cliff_gb: f64,
    max_p50_8k_to_1k_ratio: f64,
    max_p95_8k_to_1k_ratio: f64,
    warmup_decode_samples_discarded: usize,
    decode_growth: Option<DecodeGrowth>,
    probes_requested: usize,
    claims: ClaimInventory,
    records: Vec<P04Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct P04Record {
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
    probe_kind: &'static str,
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
    decode_token_latencies_ms: Vec<f64>,
    decode_latency_stats: Option<LatencyStats>,
    steady_decode_latency_stats: Option<LatencyStats>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    active_kv_bytes: Option<u64>,
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
    native_active_kv_bytes: Option<u64>,
    native_decode_latency_stats: Option<LatencyStats>,
    native_steady_decode_latency_stats: Option<LatencyStats>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct TokenMismatch {
    index: usize,
    helper_token: Option<i32>,
    native_token: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct LatencyStats {
    count: usize,
    min_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    mean_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct DecodeGrowth {
    baseline_probe_id: String,
    largest_probe_id: String,
    baseline_context_tokens: usize,
    largest_context_tokens: usize,
    context_ratio: f64,
    native_p50_ratio: f64,
    native_p95_ratio: f64,
    native_p50_vs_linear_ratio: f64,
    native_p95_vs_linear_ratio: f64,
    baseline_native_p50_ms: f64,
    largest_native_p50_ms: f64,
    baseline_native_p95_ms: f64,
    largest_native_p95_ms: f64,
    status: String,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ClaimInventory {
    confirmed_parity: Vec<String>,
    decode_latency: Vec<String>,
    kv_memory: Vec<String>,
    numerical_drift: Vec<String>,
    runtime_failures: Vec<String>,
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
    decode_token_latencies_ms: Option<Vec<f64>>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    active_kv_bytes: Option<u64>,
}

fn probes(args: &Args) -> Vec<Probe> {
    let mut probes = vec![
        Probe {
            id: "hello_smoke".to_owned(),
            description: "M04 tokenizer-controlled Hello prompt; short greedy parity and incremental decode smoke.".to_owned(),
            input: ProbeInput::TokenIds(vec![9259]),
            max_new_tokens: args.small_max_new_tokens,
            kind: ProbeKind::Small,
        },
        Probe {
            id: "hello_reference_prefix".to_owned(),
            description: "Hello plus two reference generated tokens; continuation parity from a non-empty prefix.".to_owned(),
            input: ProbeInput::TokenIds(vec![9259, 236772, 236772]),
            max_new_tokens: args.small_max_new_tokens,
            kind: ProbeKind::Small,
        },
    ];

    for context_tokens in &args.contexts {
        probes.push(Probe {
            id: format!("repeat_9259_{}k", context_tokens / 1024),
            description: format!(
                "Repeated-token context probe for native incremental decode latency at {context_tokens} tokens."
            ),
            input: ProbeInput::RepeatToken {
                token_id: 9259,
                context_tokens: *context_tokens,
            },
            max_new_tokens: args.max_new_tokens,
            kind: ProbeKind::Context,
        });
    }

    probes
}

fn run_probe(
    args: &Args,
    run_id: &str,
    probe: &Probe,
) -> Result<P04Record, Box<dyn std::error::Error>> {
    let helper = run_backend(args, probe, Backend::Helper)?;
    let native = run_backend(args, probe, Backend::Native)?;
    let comparison = compare_runs(probe, &helper, &native);

    Ok(P04Record {
        schema_version: 1,
        goal: "P04-incremental-native-kv",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        probe_id: probe.id.clone(),
        description: probe.description.clone(),
        input_spec: probe.input.display(),
        nominal_input_tokens: probe.input.nominal_tokens(),
        max_new_tokens: probe.max_new_tokens,
        mode: MODE,
        probe_kind: match probe.kind {
            ProbeKind::Small => "small",
            ProbeKind::Context => "context",
        },
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
    match backend {
        Backend::Helper => {
            command.env_remove("GEMMA4D_REQUIRE_MLX");
            command.env_remove("GEMMA4D_USE_NATIVE_GRAPH");
        }
        Backend::Native => {
            command.env("GEMMA4D_REQUIRE_MLX", "1");
            command.env("GEMMA4D_USE_NATIVE_GRAPH", "1");
        }
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
    let decode_token_latencies_ms = parsed.decode_token_latencies_ms.unwrap_or_default();
    let decode_latency_stats = latency_stats(&decode_token_latencies_ms);
    let steady_decode_latency_stats = steady_latency_stats(&decode_token_latencies_ms);

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
        decode_latency_stats,
        steady_decode_latency_stats,
        decode_token_latencies_ms,
        peak_memory_gb: parsed.peak_memory_gb,
        peak_rss_mb: parsed.peak_rss_mb,
        active_kv_bytes: parsed.active_kv_bytes,
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

    let logit_stats = logit_delta_stats(&helper.generated_logits, &native.generated_logits);
    if native.status == "ok" && native.active_kv_bytes.unwrap_or_default() == 0 {
        blockers.push(format!(
            "{} native run did not report active KV bytes",
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
    if matches!(probe.kind, ProbeKind::Context)
        && probe.max_new_tokens > 1
        && native.steady_decode_latency_stats.is_none()
        && native.status == "ok"
    {
        blockers.push(format!(
            "{} native run did not emit steady-state decode latency samples",
            probe.id
        ));
    }

    let status = if helper.status != "ok" || native.status != "ok" {
        "runtime_failure"
    } else if !token_match {
        "token_mismatch"
    } else if native.active_kv_bytes.unwrap_or_default() == 0 {
        "missing_kv_state"
    } else if native
        .peak_memory_gb
        .is_some_and(|peak| peak >= MEMORY_CLIFF_GB)
    {
        "memory_cliff"
    } else if logit_stats
        .max_abs_delta
        .is_some_and(|delta| delta > LOGIT_TOLERANCE)
    {
        "parity_with_logit_drift"
    } else {
        "parity_confirmed"
    };

    Comparison {
        status: status.to_owned(),
        token_match,
        first_token_mismatch,
        logit_count_compared: logit_stats.count,
        max_logit_abs_delta: logit_stats.max_abs_delta,
        mean_logit_abs_delta: logit_stats.mean_abs_delta,
        native_total_minus_helper_total_ms: delta(helper.total_ms, native.total_ms),
        native_prefill_minus_helper_prefill_ms: delta(helper.prefill_ms, native.prefill_ms),
        native_decode_minus_helper_decode_ms: delta(helper.decode_ms, native.decode_ms),
        native_peak_minus_helper_peak_gb: delta(helper.peak_memory_gb, native.peak_memory_gb),
        native_active_kv_bytes: native.active_kv_bytes,
        native_decode_latency_stats: native.decode_latency_stats.clone(),
        native_steady_decode_latency_stats: native.steady_decode_latency_stats.clone(),
        blockers,
    }
}

fn decode_growth(records: &[P04Record]) -> Option<DecodeGrowth> {
    let mut context_records = records
        .iter()
        .filter(|record| record.probe_kind == "context")
        .filter_map(|record| {
            let stats = record.native.steady_decode_latency_stats.as_ref()?;
            Some((record, stats))
        })
        .collect::<Vec<_>>();
    context_records.sort_by_key(|(record, _)| record.nominal_input_tokens);
    let (baseline, baseline_stats) = context_records.first().copied()?;
    let (largest, largest_stats) = context_records.last().copied()?;
    if baseline.probe_id == largest.probe_id {
        return None;
    }

    let context_ratio = largest.nominal_input_tokens as f64 / baseline.nominal_input_tokens as f64;
    let native_p50_ratio = largest_stats.p50_ms / baseline_stats.p50_ms;
    let native_p95_ratio = largest_stats.p95_ms / baseline_stats.p95_ms;
    let native_p50_vs_linear_ratio = native_p50_ratio / context_ratio;
    let native_p95_vs_linear_ratio = native_p95_ratio / context_ratio;
    let mut blockers = Vec::new();
    if native_p50_ratio > MAX_P50_8K_TO_1K_RATIO {
        blockers.push(format!(
            "native decode p50 ratio {:.3} exceeds threshold {:.3} from {} to {} tokens",
            native_p50_ratio,
            MAX_P50_8K_TO_1K_RATIO,
            baseline.nominal_input_tokens,
            largest.nominal_input_tokens
        ));
    }
    if native_p95_ratio > MAX_P95_8K_TO_1K_RATIO {
        blockers.push(format!(
            "native decode p95 ratio {:.3} exceeds threshold {:.3} from {} to {} tokens",
            native_p95_ratio,
            MAX_P95_8K_TO_1K_RATIO,
            baseline.nominal_input_tokens,
            largest.nominal_input_tokens
        ));
    }

    let status = if blockers.is_empty() {
        "sublinear_steady_decode_growth"
    } else {
        "linear_or_unbounded_steady_decode_growth"
    };

    Some(DecodeGrowth {
        baseline_probe_id: baseline.probe_id.clone(),
        largest_probe_id: largest.probe_id.clone(),
        baseline_context_tokens: baseline.nominal_input_tokens,
        largest_context_tokens: largest.nominal_input_tokens,
        context_ratio,
        native_p50_ratio,
        native_p95_ratio,
        native_p50_vs_linear_ratio,
        native_p95_vs_linear_ratio,
        baseline_native_p50_ms: baseline_stats.p50_ms,
        largest_native_p50_ms: largest_stats.p50_ms,
        baseline_native_p95_ms: baseline_stats.p95_ms,
        largest_native_p95_ms: largest_stats.p95_ms,
        status: status.to_owned(),
        blockers,
    })
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

fn latency_stats(latencies: &[f64]) -> Option<LatencyStats> {
    if latencies.is_empty() {
        return None;
    }
    let mut sorted = latencies.to_vec();
    sorted.sort_by(f64::total_cmp);
    let sum = sorted.iter().sum::<f64>();
    Some(LatencyStats {
        count: sorted.len(),
        min_ms: sorted[0],
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        max_ms: sorted[sorted.len() - 1],
        mean_ms: sum / sorted.len() as f64,
    })
}

fn steady_latency_stats(latencies: &[f64]) -> Option<LatencyStats> {
    if latencies.len() > WARMUP_DECODE_SAMPLES {
        latency_stats(&latencies[WARMUP_DECODE_SAMPLES..])
    } else {
        latency_stats(latencies)
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let index = ((sorted.len() as f64 - 1.0) * percentile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn parse_generate_json(stdout: &str) -> Option<GenerateJson> {
    stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<GenerateJson>(line).ok())
}

fn claim_inventory(records: &[P04Record], growth: Option<&DecodeGrowth>) -> ClaimInventory {
    let mut confirmed_parity = Vec::new();
    let mut decode_latency = Vec::new();
    let mut kv_memory = Vec::new();
    let mut numerical_drift = Vec::new();
    let mut runtime_failures = Vec::new();

    for record in records {
        match record.comparison.status.as_str() {
            "parity_confirmed" => confirmed_parity.push(format!(
                "{}: native tokens and greedy logits match helper within tolerance",
                record.probe_id
            )),
            "parity_with_logit_drift" => {
                confirmed_parity.push(format!(
                    "{}: native generated token IDs match helper",
                    record.probe_id
                ));
                numerical_drift.push(format!(
                    "{}: token parity holds; max greedy logit delta {} and mean delta {}",
                    record.probe_id,
                    fmt_opt(record.comparison.max_logit_abs_delta),
                    fmt_opt(record.comparison.mean_logit_abs_delta)
                ));
            }
            "token_mismatch" => numerical_drift.push(format!(
                "{}: status `{}` max greedy logit delta {} first mismatch {:?}",
                record.probe_id,
                record.comparison.status,
                fmt_opt(record.comparison.max_logit_abs_delta),
                record.comparison.first_token_mismatch
            )),
            "runtime_failure" => runtime_failures.push(format!(
                "{}: helper status `{}` native status `{}`",
                record.probe_id, record.helper.status, record.native.status
            )),
            "missing_kv_state" => runtime_failures.push(format!(
                "{}: native did not expose active KV bytes",
                record.probe_id
            )),
            "memory_cliff" => runtime_failures.push(format!(
                "{}: native peak {} GB",
                record.probe_id,
                fmt_opt(record.native.peak_memory_gb)
            )),
            _ => {}
        }

        if let Some(stats) = &record.native.steady_decode_latency_stats {
            decode_latency.push(format!(
                "{}: native steady decode p50 {:.3} ms p95 {:.3} ms over {} samples",
                record.probe_id, stats.p50_ms, stats.p95_ms, stats.count
            ));
        }
        if let Some(active_kv_bytes) = record.native.active_kv_bytes {
            kv_memory.push(format!(
                "{}: native active KV {:.3} MiB, peak MLX {} GB",
                record.probe_id,
                active_kv_bytes as f64 / 1024.0 / 1024.0,
                fmt_opt(record.native.peak_memory_gb)
            ));
        }
    }

    if let Some(growth) = growth {
        decode_latency.push(format!(
            "{}: steady p50 ratio {:.3} and steady p95 ratio {:.3} across {:.1}x context growth",
            growth.status, growth.native_p50_ratio, growth.native_p95_ratio, growth.context_ratio
        ));
    }

    ClaimInventory {
        confirmed_parity,
        decode_latency,
        kv_memory,
        numerical_drift,
        runtime_failures,
    }
}

fn blockers_for_records(records: &[P04Record]) -> Vec<String> {
    records
        .iter()
        .flat_map(|record| record.comparison.blockers.iter().cloned())
        .collect()
}

fn write_jsonl(path: &Path, records: &[P04Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn render_report(summary: &P04Summary) -> String {
    let mut out = String::new();
    out.push_str("# P04 Incremental Native KV Decode\n\n");
    out.push_str("## Status\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Mode: `{}`\n- Records: `{}`\n- Summary: `{}`\n- Blockers: `{}`\n- Logit warning threshold: `{:.3}`\n- Memory cliff threshold: `{:.1} GB`\n- Warmup decode samples discarded for steady-state growth: `{}`\n- p50 growth threshold: `{:.3}`\n- p95 growth threshold: `{:.3}`\n\n",
        summary.status,
        summary.mode,
        summary.records_path,
        summary.summary_path,
        summary.blockers_path,
        summary.logit_tolerance,
        summary.memory_cliff_gb,
        summary.warmup_decode_samples_discarded,
        summary.max_p50_8k_to_1k_ratio,
        summary.max_p95_8k_to_1k_ratio,
    ));

    out.push_str("## Claim Inventory\n\n");
    render_claim_list(
        &mut out,
        "Confirmed Parity",
        &summary.claims.confirmed_parity,
    );
    render_claim_list(&mut out, "Decode Latency", &summary.claims.decode_latency);
    render_claim_list(&mut out, "KV Memory", &summary.claims.kv_memory);
    render_claim_list(&mut out, "Numerical Drift", &summary.claims.numerical_drift);
    render_claim_list(
        &mut out,
        "Runtime Failures",
        &summary.claims.runtime_failures,
    );

    out.push_str("## Decode Growth\n\n");
    if let Some(growth) = &summary.decode_growth {
        out.push_str("| Baseline | Largest | Context Ratio | p50 Ratio | p95 Ratio | p50 vs Linear | p95 vs Linear | Status |\n");
        out.push_str("|---|---|---:|---:|---:|---:|---:|---|\n");
        out.push_str(&format!(
            "| `{}` {} tokens | `{}` {} tokens | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | `{}` |\n\n",
            growth.baseline_probe_id,
            growth.baseline_context_tokens,
            growth.largest_probe_id,
            growth.largest_context_tokens,
            growth.context_ratio,
            growth.native_p50_ratio,
            growth.native_p95_ratio,
            growth.native_p50_vs_linear_ratio,
            growth.native_p95_vs_linear_ratio,
            growth.status,
        ));
    } else {
        out.push_str("No decode growth summary was available.\n\n");
    }

    out.push_str("## Probe Results\n\n");
    out.push_str("| Probe | Kind | Tokens | Gen | Status | Token Match | Max Logit Delta | Native Active KV MiB | Helper Total ms | Native Total ms | Native Prefill ms | Native Decode ms | Native steady p50 ms | Native steady p95 ms | Native raw p95 ms | Native Peak GB |\n");
    out.push_str("|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.records {
        let raw_stats = record.native.decode_latency_stats.as_ref();
        let steady_stats = record.native.steady_decode_latency_stats.as_ref();
        out.push_str(&format!(
            "| `{}` | `{}` | {} | {} | `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            record.probe_id,
            record.probe_kind,
            record.nominal_input_tokens,
            record.max_new_tokens,
            record.comparison.status,
            record.comparison.token_match,
            fmt_opt(record.comparison.max_logit_abs_delta),
            fmt_mib(record.native.active_kv_bytes),
            fmt_opt(record.helper.total_ms),
            fmt_opt(record.native.total_ms),
            fmt_opt(record.native.prefill_ms),
            fmt_opt(record.native.decode_ms),
            steady_stats
                .map(|stats| format!("{:.3}", stats.p50_ms))
                .unwrap_or_else(|| "n/a".to_owned()),
            steady_stats
                .map(|stats| format!("{:.3}", stats.p95_ms))
                .unwrap_or_else(|| "n/a".to_owned()),
            raw_stats
                .map(|stats| format!("{:.3}", stats.p95_ms))
                .unwrap_or_else(|| "n/a".to_owned()),
            fmt_opt(record.native.peak_memory_gb),
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

fn render_blockers(summary: &P04Summary) -> String {
    if summary.blockers.is_empty() {
        return "No blockers recorded.\n".to_owned();
    }
    let mut out = String::new();
    out.push_str("# P04 Blockers\n\n");
    for blocker in &summary.blockers {
        out.push_str(&format!("- {blocker}\n"));
    }
    out
}

fn parse_contexts(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| parse_positive_usize(part.trim(), "--contexts"))
        .collect()
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

fn fmt_mib(value: Option<u64>) -> String {
    value
        .map(|value| format!("{:.3}", value as f64 / 1024.0 / 1024.0))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', "<br>")
}

fn run_id() -> String {
    format!("p04-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after Unix epoch")
        .as_secs()
}
