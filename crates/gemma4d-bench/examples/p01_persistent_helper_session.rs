use std::{
    collections::BTreeMap,
    env, fs,
    io::{BufRead, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_ffi::{self as ffi, KvCache, KvPolicy, LoadConfig, Target};
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P01-persistent-helper-session";
const DEFAULT_COLD_RECORDS: &str = "benchmarks/out/M12/real-matrix/records.jsonl";
const MODE: &str = "target_greedy_mlx_lm_helper_via_c_abi";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let environment = capture_environment();
    let cold_records = load_cold_records(&args.cold_records_path)?;
    let mut blockers = Vec::new();

    if !args.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if cold_records.is_empty() {
        blockers.push(format!(
            "no cold M12 records were loaded from {}",
            args.cold_records_path.display()
        ));
    }

    let mut warm_records = Vec::new();
    let mut session = None;
    let benchmark_process_rss_before_mb = process_rss_mb();
    if blockers.is_empty() {
        session = Some(run_warm_session(&args, &run_id, &cold_records)?);
        warm_records = session
            .as_ref()
            .expect("session was just set")
            .records
            .clone();
    }

    let comparisons = compare_cold_warm(
        &args.contexts,
        &cold_records,
        &warm_records,
        session.as_ref().and_then(|session| session.model_load_ms),
    );
    let blockers = if blockers.is_empty() {
        missing_cold_blockers(&comparisons)
    } else {
        blockers
    };
    let status = if !blockers.is_empty() {
        "blocked"
    } else if warm_records.iter().all(|record| record.status == "passed")
        && comparisons
            .iter()
            .all(|comparison| comparison.output_stable)
    {
        "passed"
    } else {
        "failed"
    };
    let benchmark_process_rss_after_mb = process_rss_mb();
    let session = session.unwrap_or(WarmSession {
        model_load_ms: None,
        benchmark_process_rss_before_mb,
        benchmark_process_rss_after_mb,
        records: warm_records,
    });
    let load_amortization = load_amortization(&session, &comparisons, &args);
    let summary = PersistentSummary {
        schema_version: 1,
        goal: "P01-persistent-helper-session",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        model_path: args.model_path.display().to_string(),
        cold_records_path: args.cold_records_path.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        environment,
        relevant_environment: capture_relevant_environment(),
        contexts: args.contexts.clone(),
        rounds: args.rounds,
        generated_tokens_requested: args.max_new_tokens,
        explicit_reset: "single KvCache handle; KvCache::reset before every warm case; helper prefill recreates the Python prompt cache".to_owned(),
        session,
        comparisons,
        load_amortization,
        blockers,
        measurement_notes: vec![
            "warm session loads one Target once, then runs all cases in one process.",
            "warm total_case_ms excludes the one-time model_load_ms and includes reset_ms, prefill_ms, and decode_ms.",
            "warm total_with_amortized_load_ms adds model_load_ms divided by warm case count.",
            "cold values come from M12 cold CLI records and include their existing cargo/process behavior.",
            "output_stable compares warm generated token sequences against cold M12 raw_stdout tokens.",
            "helper peak RSS and MLX peak memory are cumulative helper-reported peaks; growth fields compare against the first warm case.",
        ],
    };

    write_jsonl(&records_path, &summary.session.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P01 persistent helper session: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if summary.status == "failed" {
        Err("P01 persistent helper session failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    cold_records_path: PathBuf,
    contexts: Vec<usize>,
    max_new_tokens: usize,
    max_context_tokens: NonZeroU32,
    rounds: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut cold_records_path = PathBuf::from(DEFAULT_COLD_RECORDS);
        let mut contexts = vec![1024, 4096, 8192, 16_384];
        let mut max_new_tokens = 128;
        let mut max_context_tokens = NonZeroU32::new(32_768).expect("non-zero default");
        let mut rounds = 2;

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
                "--cold-records" => {
                    cold_records_path = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--cold-records requires a path")?;
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
                    max_context_tokens =
                        NonZeroU32::new(parse_positive_u32(&value, "--max-context-tokens")?)
                            .ok_or("--max-context-tokens must be greater than zero")?;
                }
                "--rounds" => {
                    let value = args.next().ok_or("--rounds requires a value")?;
                    rounds = parse_positive_usize(&value, "--rounds")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example p01_persistent_helper_session -- [--out-dir PATH] [--model-path PATH] [--cold-records PATH] [--contexts 1024,4096,8192,16384] [--rounds N]"
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
            cold_records_path,
            contexts,
            max_new_tokens,
            max_context_tokens,
            rounds,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct PersistentSummary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    model_path: String,
    cold_records_path: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    contexts: Vec<usize>,
    rounds: usize,
    generated_tokens_requested: usize,
    explicit_reset: String,
    session: WarmSession,
    comparisons: Vec<ComparisonRow>,
    load_amortization: LoadAmortization,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct WarmSession {
    model_load_ms: Option<f64>,
    benchmark_process_rss_before_mb: Option<f64>,
    benchmark_process_rss_after_mb: Option<f64>,
    records: Vec<WarmRecord>,
}

#[derive(Debug, Clone, Serialize)]
struct WarmRecord {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    round: usize,
    case_index: usize,
    workload: &'static str,
    context_tokens: usize,
    generated_tokens_requested: usize,
    generated_tokens_observed: usize,
    mode: &'static str,
    status: String,
    reset_kind: &'static str,
    reset_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_case_ms: f64,
    total_with_amortized_load_ms: Option<f64>,
    prefill_tokens_per_second: f64,
    decode_tokens_per_second: f64,
    decode_latency_ms: DecodeLatencySummary,
    memory: WarmMemory,
    generated_tokens: Vec<i32>,
    cold_output_stable: Option<bool>,
    cold_generated_tokens_observed: Option<usize>,
    cold_command: Option<String>,
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
struct WarmMemory {
    mlx_peak_memory_gb: f64,
    helper_peak_rss_mb: f64,
    mlx_peak_growth_from_first_case_gb: Option<f64>,
    helper_peak_rss_growth_from_first_case_mb: Option<f64>,
    benchmark_process_rss_before_case_mb: Option<f64>,
    benchmark_process_rss_after_case_mb: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ComparisonRow {
    context_tokens: usize,
    cold_status: String,
    warm_status: String,
    output_stable: bool,
    cold_generated_tokens: Option<usize>,
    warm_generated_tokens: Option<usize>,
    cold_model_load_ms: Option<f64>,
    warm_session_model_load_ms: Option<f64>,
    warm_amortized_model_load_ms: Option<f64>,
    cold_prefill_ms: Option<f64>,
    warm_prefill_ms_mean: Option<f64>,
    prefill_delta_ms: Option<f64>,
    cold_decode_ms: Option<f64>,
    warm_decode_ms_mean: Option<f64>,
    decode_delta_ms: Option<f64>,
    cold_total_ms: Option<f64>,
    cold_command_wall_ms: Option<f64>,
    cold_process_command_overhead_ms: Option<f64>,
    warm_case_ms_mean: Option<f64>,
    warm_total_with_amortized_load_ms_mean: Option<f64>,
    total_with_amortized_load_delta_ms: Option<f64>,
    cold_decode_tps: Option<f64>,
    warm_decode_tps_mean: Option<f64>,
    cold_peak_memory_gb: Option<f64>,
    warm_peak_memory_gb_max: Option<f64>,
    cold_peak_rss_mb: Option<f64>,
    warm_peak_rss_mb_max: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct LoadAmortization {
    warm_case_count: usize,
    single_warm_model_load_ms: Option<f64>,
    cold_model_load_ms_sum_for_equivalent_cases: Option<f64>,
    model_load_ms_saved: Option<f64>,
    model_load_saved_percent: Option<f64>,
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

#[derive(Debug, Clone)]
struct ColdRecord {
    context_tokens: usize,
    status: String,
    command: String,
    generated_tokens_observed: usize,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    decode_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    command_wall_ms: Option<f64>,
    generated_tokens: Vec<i32>,
    model_load_ms: Option<f64>,
    total_ms: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ColdRecordJson {
    context_tokens: usize,
    status: String,
    command: String,
    generated_tokens_observed: usize,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    decode_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    note: Option<String>,
    raw_stdout: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ColdGenerateJson {
    generated_tokens: Option<Vec<i32>>,
    model_load_ms: Option<f64>,
    total_ms: Option<f64>,
}

fn run_warm_session(
    args: &Args,
    run_id: &str,
    cold_records: &[ColdRecord],
) -> Result<WarmSession, Box<dyn std::error::Error>> {
    let load_config = LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: env::var("GEMMA4D_MODEL_REVISION").ok(),
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: args.max_context_tokens,
        allow_unsupported_config: false,
    };

    let benchmark_process_rss_before_mb = process_rss_mb();
    let load_started = Instant::now();
    let target = Target::load(&load_config)?;
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let mut cache = KvCache::create(&KvPolicy::default())?;
    let mut records = Vec::new();
    let mut first_mlx_peak = None;
    let mut first_helper_rss = None;
    let warm_case_count = args.contexts.len() * args.rounds;

    for round in 1..=args.rounds {
        for context_tokens in &args.contexts {
            let case_index = records.len() + 1;
            let tokens = vec![1; *context_tokens];
            let cold = cold_records
                .iter()
                .find(|record| record.context_tokens == *context_tokens);
            let benchmark_process_rss_before_case_mb = process_rss_mb();
            let reset_started = Instant::now();
            cache.reset()?;
            let reset_ms = reset_started.elapsed().as_secs_f64() * 1000.0;

            let prefill_started = Instant::now();
            let mut step = ffi::prefill(&target, &mut cache, &tokens)?;
            let prefill = prefill_started.elapsed();
            let mut peak_memory_gb = f64::from(step.peak_memory_gb);
            let mut peak_rss_mb = f64::from(step.peak_rss_mb);
            let mut generated_tokens = Vec::with_capacity(args.max_new_tokens);
            let mut decode_latencies = Vec::with_capacity(args.max_new_tokens.saturating_sub(1));
            let decode_started = Instant::now();
            for index in 0..args.max_new_tokens {
                generated_tokens.push(step.greedy_token);
                if index + 1 < args.max_new_tokens {
                    let token_started = Instant::now();
                    step = ffi::decode_one(&target, &mut cache, step.greedy_token)?;
                    decode_latencies.push(token_started.elapsed());
                    peak_memory_gb = peak_memory_gb.max(f64::from(step.peak_memory_gb));
                    peak_rss_mb = peak_rss_mb.max(f64::from(step.peak_rss_mb));
                }
            }
            let decode = decode_started.elapsed();
            let prefill_ms = duration_ms(prefill);
            let decode_ms = duration_ms(decode);
            let total_case_ms = reset_ms + prefill_ms + decode_ms;
            let benchmark_process_rss_after_case_mb = process_rss_mb();
            let first_mlx = *first_mlx_peak.get_or_insert(peak_memory_gb);
            let first_rss = *first_helper_rss.get_or_insert(peak_rss_mb);
            let output_stable = cold.map(|cold| cold.generated_tokens == generated_tokens);
            let status = if output_stable == Some(false) {
                "output_mismatch"
            } else {
                "passed"
            };

            records.push(WarmRecord {
                schema_version: 1,
                goal: "P01-persistent-helper-session",
                run_id: run_id.to_owned(),
                timestamp_unix: unix_now(),
                round,
                case_index,
                workload: "simple_chat_repeated_token",
                context_tokens: *context_tokens,
                generated_tokens_requested: args.max_new_tokens,
                generated_tokens_observed: generated_tokens.len(),
                mode: MODE,
                status: status.to_owned(),
                reset_kind: "KvCache::reset before case",
                reset_ms,
                prefill_ms,
                decode_ms,
                total_case_ms,
                total_with_amortized_load_ms: Some(
                    total_case_ms + model_load_ms / warm_case_count as f64,
                ),
                prefill_tokens_per_second: *context_tokens as f64 / (prefill_ms / 1000.0),
                decode_tokens_per_second: decode_tps(args.max_new_tokens, decode),
                decode_latency_ms: decode_latency_summary(decode_latencies),
                memory: WarmMemory {
                    mlx_peak_memory_gb: peak_memory_gb,
                    helper_peak_rss_mb: peak_rss_mb,
                    mlx_peak_growth_from_first_case_gb: Some(peak_memory_gb - first_mlx),
                    helper_peak_rss_growth_from_first_case_mb: Some(peak_rss_mb - first_rss),
                    benchmark_process_rss_before_case_mb,
                    benchmark_process_rss_after_case_mb,
                },
                generated_tokens,
                cold_output_stable: output_stable,
                cold_generated_tokens_observed: cold.map(|cold| cold.generated_tokens_observed),
                cold_command: cold.map(|cold| cold.command.clone()),
                blocker: (output_stable == Some(false))
                    .then(|| "warm generated tokens differ from cold M12 output".to_owned()),
            });
        }
    }

    Ok(WarmSession {
        model_load_ms: Some(model_load_ms),
        benchmark_process_rss_before_mb,
        benchmark_process_rss_after_mb: process_rss_mb(),
        records,
    })
}

fn load_cold_records(path: &Path) -> Result<Vec<ColdRecord>, Box<dyn std::error::Error>> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut records = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: ColdRecordJson = serde_json::from_str(&line)?;
        let generated = raw
            .raw_stdout
            .as_deref()
            .and_then(parse_cold_generate_json)
            .unwrap_or_default();
        records.push(ColdRecord {
            context_tokens: raw.context_tokens,
            status: raw.status,
            command: raw.command,
            generated_tokens_observed: raw.generated_tokens_observed,
            ttft_ms: raw.ttft_ms,
            decode_ms: raw.decode_ms,
            decode_tps: raw.decode_tps,
            peak_memory_gb: raw.peak_memory_gb,
            peak_rss_mb: raw.peak_rss_mb,
            command_wall_ms: raw.note.as_deref().and_then(parse_wall_seconds),
            generated_tokens: generated.generated_tokens.unwrap_or_default(),
            model_load_ms: generated.model_load_ms,
            total_ms: generated.total_ms,
        });
    }
    Ok(records)
}

fn parse_cold_generate_json(stdout: &str) -> Option<ColdGenerateJson> {
    stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<ColdGenerateJson>(line).ok())
}

fn parse_wall_seconds(note: &str) -> Option<f64> {
    let (_, tail) = note.rsplit_once("wall_seconds=")?;
    tail.split(|ch: char| ch == ';' || ch.is_whitespace())
        .next()
        .and_then(|value| value.parse::<f64>().ok())
        .map(|seconds| seconds * 1000.0)
}

fn compare_cold_warm(
    contexts: &[usize],
    cold_records: &[ColdRecord],
    warm_records: &[WarmRecord],
    session_model_load_ms: Option<f64>,
) -> Vec<ComparisonRow> {
    let warm_case_count = warm_records.len().max(1);
    let warm_amortized_model_load_ms =
        session_model_load_ms.map(|model_load_ms| model_load_ms / warm_case_count as f64);
    contexts
        .iter()
        .map(|context_tokens| {
            let cold = cold_records
                .iter()
                .find(|record| record.context_tokens == *context_tokens);
            let warm = warm_records
                .iter()
                .filter(|record| record.context_tokens == *context_tokens)
                .collect::<Vec<_>>();
            let warm_status = if warm.is_empty() {
                "missing".to_owned()
            } else if warm.iter().all(|record| record.status == "passed") {
                "passed".to_owned()
            } else {
                "failed".to_owned()
            };
            let warm_prefill_ms_mean = mean(warm.iter().map(|record| record.prefill_ms));
            let warm_decode_ms_mean = mean(warm.iter().map(|record| record.decode_ms));
            let warm_case_ms_mean = mean(warm.iter().map(|record| record.total_case_ms));
            let warm_total_with_amortized_load_ms_mean = mean(
                warm.iter()
                    .filter_map(|record| record.total_with_amortized_load_ms),
            );
            let warm_decode_tps_mean =
                mean(warm.iter().map(|record| record.decode_tokens_per_second));
            let warm_peak_memory_gb_max =
                max(warm.iter().map(|record| record.memory.mlx_peak_memory_gb));
            let warm_peak_rss_mb_max =
                max(warm.iter().map(|record| record.memory.helper_peak_rss_mb));
            let cold_total_ms = cold.and_then(|record| record.total_ms);
            let cold_command_wall_ms = cold.and_then(|record| record.command_wall_ms);
            let cold_process_command_overhead_ms = match (cold_command_wall_ms, cold_total_ms) {
                (Some(wall), Some(total)) => Some((wall - total).max(0.0)),
                _ => None,
            };
            let cold_status = cold
                .map(|record| record.status.clone())
                .unwrap_or_else(|| "missing".to_owned());
            let output_stable = !warm.is_empty()
                && warm
                    .iter()
                    .all(|record| record.cold_output_stable == Some(true));

            ComparisonRow {
                context_tokens: *context_tokens,
                cold_status,
                warm_status,
                output_stable,
                cold_generated_tokens: cold.map(|record| record.generated_tokens_observed),
                warm_generated_tokens: warm.first().map(|record| record.generated_tokens_observed),
                cold_model_load_ms: cold.and_then(|record| record.model_load_ms),
                warm_session_model_load_ms: session_model_load_ms,
                warm_amortized_model_load_ms,
                cold_prefill_ms: cold.and_then(|record| record.ttft_ms),
                warm_prefill_ms_mean,
                prefill_delta_ms: delta(
                    cold.and_then(|record| record.ttft_ms),
                    warm_prefill_ms_mean,
                ),
                cold_decode_ms: cold.and_then(|record| record.decode_ms),
                warm_decode_ms_mean,
                decode_delta_ms: delta(
                    cold.and_then(|record| record.decode_ms),
                    warm_decode_ms_mean,
                ),
                cold_total_ms,
                cold_command_wall_ms,
                cold_process_command_overhead_ms,
                warm_case_ms_mean,
                warm_total_with_amortized_load_ms_mean,
                total_with_amortized_load_delta_ms: delta(
                    cold_total_ms,
                    warm_total_with_amortized_load_ms_mean,
                ),
                cold_decode_tps: cold.and_then(|record| record.decode_tps),
                warm_decode_tps_mean,
                cold_peak_memory_gb: cold.and_then(|record| record.peak_memory_gb),
                warm_peak_memory_gb_max,
                cold_peak_rss_mb: cold.and_then(|record| record.peak_rss_mb),
                warm_peak_rss_mb_max,
            }
        })
        .collect()
}

fn load_amortization(
    session: &WarmSession,
    comparisons: &[ComparisonRow],
    args: &Args,
) -> LoadAmortization {
    let warm_case_count = args.contexts.len() * args.rounds;
    let cold_model_load_ms_sum_for_equivalent_cases = comparisons
        .iter()
        .map(|comparison| comparison.cold_model_load_ms)
        .try_fold(0.0, |acc, value| {
            value.map(|value| acc + value * args.rounds as f64)
        });
    let model_load_ms_saved = match (
        cold_model_load_ms_sum_for_equivalent_cases,
        session.model_load_ms,
    ) {
        (Some(cold_sum), Some(warm_load)) => Some(cold_sum - warm_load),
        _ => None,
    };
    let model_load_saved_percent = match (
        model_load_ms_saved,
        cold_model_load_ms_sum_for_equivalent_cases,
    ) {
        (Some(saved), Some(cold_sum)) if cold_sum > 0.0 => Some(saved / cold_sum * 100.0),
        _ => None,
    };

    LoadAmortization {
        warm_case_count,
        single_warm_model_load_ms: session.model_load_ms,
        cold_model_load_ms_sum_for_equivalent_cases,
        model_load_ms_saved,
        model_load_saved_percent,
    }
}

fn missing_cold_blockers(comparisons: &[ComparisonRow]) -> Vec<String> {
    comparisons
        .iter()
        .filter_map(|comparison| {
            if comparison.cold_status == "missing" {
                Some(format!(
                    "missing cold M12 record for {} tokens",
                    comparison.context_tokens
                ))
            } else {
                None
            }
        })
        .collect()
}

fn write_jsonl(path: &Path, records: &[WarmRecord]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn decode_latency_summary(samples: Vec<Duration>) -> DecodeLatencySummary {
    let mut samples = samples
        .iter()
        .map(|duration| duration_ms(*duration))
        .collect::<Vec<_>>();
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

fn mean<I>(values: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum / count as f64)
}

fn max<I>(values: I) -> Option<f64>
where
    I: Iterator<Item = f64>,
{
    values.reduce(f64::max)
}

fn delta(cold: Option<f64>, warm: Option<f64>) -> Option<f64> {
    match (cold, warm) {
        (Some(cold), Some(warm)) => Some(warm - cold),
        _ => None,
    }
}

fn decode_tps(max_new_tokens: usize, decode: Duration) -> f64 {
    let decode_tokens = max_new_tokens.saturating_sub(1);
    if decode_tokens == 0 || decode.is_zero() {
        0.0
    } else {
        decode_tokens as f64 / decode.as_secs_f64()
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn process_rss_mb() -> Option<f64> {
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", pid.as_str()])
        .output()
        .ok()?;
    let rss_kb = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .ok()?;
    Some(rss_kb / 1024.0)
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

fn render_report(summary: &PersistentSummary) -> String {
    let mut out = String::new();
    out.push_str("# P01 Persistent Helper Session\n\n");
    out.push_str("## Status\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Mode: `{}`\n- Records: `{}`\n- Summary: `{}`\n- Cold source: `{}`\n- Explicit reset: `{}`\n\n",
        summary.status,
        summary.mode,
        summary.records_path,
        summary.summary_path,
        summary.cold_records_path,
        summary.explicit_reset
    ));
    out.push_str("## Load Amortization\n\n");
    out.push_str(
        "| Warm cases | Warm load once ms | Equivalent cold load ms | Load ms saved | Saved % |\n",
    );
    out.push_str("|---:|---:|---:|---:|---:|\n");
    out.push_str(&format!(
        "| {} | {} | {} | {} | {} |\n\n",
        summary.load_amortization.warm_case_count,
        fmt_opt(summary.load_amortization.single_warm_model_load_ms),
        fmt_opt(
            summary
                .load_amortization
                .cold_model_load_ms_sum_for_equivalent_cases
        ),
        fmt_opt(summary.load_amortization.model_load_ms_saved),
        fmt_opt(summary.load_amortization.model_load_saved_percent),
    ));
    out.push_str("## Cold vs Warm\n\n");
    out.push_str("| Context | Stable | Cold total ms | Warm case ms | Warm amortized total ms | Delta ms | Cold load ms | Warm amortized load ms | Cold prefill ms | Warm prefill ms | Cold decode ms | Warm decode ms | Cold decode tok/s | Warm decode tok/s | Cold peak GB | Warm peak GB | Cold RSS MB | Warm RSS MB |\n");
    out.push_str(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for comparison in &summary.comparisons {
        out.push_str(&format!(
            "| {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            comparison.context_tokens,
            comparison.output_stable,
            fmt_opt(comparison.cold_total_ms),
            fmt_opt(comparison.warm_case_ms_mean),
            fmt_opt(comparison.warm_total_with_amortized_load_ms_mean),
            fmt_opt(comparison.total_with_amortized_load_delta_ms),
            fmt_opt(comparison.cold_model_load_ms),
            fmt_opt(comparison.warm_amortized_model_load_ms),
            fmt_opt(comparison.cold_prefill_ms),
            fmt_opt(comparison.warm_prefill_ms_mean),
            fmt_opt(comparison.cold_decode_ms),
            fmt_opt(comparison.warm_decode_ms_mean),
            fmt_opt(comparison.cold_decode_tps),
            fmt_opt(comparison.warm_decode_tps_mean),
            fmt_opt(comparison.cold_peak_memory_gb),
            fmt_opt(comparison.warm_peak_memory_gb_max),
            fmt_opt(comparison.cold_peak_rss_mb),
            fmt_opt(comparison.warm_peak_rss_mb_max),
        ));
    }
    out.push_str("\n## Warm Session Cases\n\n");
    out.push_str("| Round | Context | Generated | Status | Reset ms | Prefill ms | Decode ms | Case ms | Amortized case ms | Decode p50 ms | Decode p95 ms | Peak GB | Peak GB growth | RSS MB | RSS growth MB |\n");
    out.push_str("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.session.records {
        out.push_str(&format!(
            "| {} | {} | {}/{} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            record.round,
            record.context_tokens,
            record.generated_tokens_observed,
            record.generated_tokens_requested,
            record.status,
            fmt_num(record.reset_ms),
            fmt_num(record.prefill_ms),
            fmt_num(record.decode_ms),
            fmt_num(record.total_case_ms),
            fmt_opt(record.total_with_amortized_load_ms),
            fmt_opt(record.decode_latency_ms.p50_ms),
            fmt_opt(record.decode_latency_ms.p95_ms),
            fmt_num(record.memory.mlx_peak_memory_gb),
            fmt_opt(record.memory.mlx_peak_growth_from_first_case_gb),
            fmt_num(record.memory.helper_peak_rss_mb),
            fmt_opt(record.memory.helper_peak_rss_growth_from_first_case_mb),
        ));
    }
    out.push_str("\n## Environment\n\n");
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
    out.push_str(
        "cargo run -p gemma4d-bench --example p01_persistent_helper_session -- --out-dir benchmarks/out/P01-persistent-helper-session --model-path artifacts/models/gemma-4-12B-it-4bit --cold-records benchmarks/out/M12/real-matrix/records.jsonl\n",
    );
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

fn render_blockers(summary: &PersistentSummary) -> String {
    let mut out = String::new();
    out.push_str("# P01 Blocker Report\n\n");
    out.push_str(&format!("- Status: `{}`\n", summary.status));
    if summary.blockers.is_empty() {
        out.push_str("- Blockers: none\n");
    } else {
        out.push_str("\n## Blockers\n\n");
        for blocker in &summary.blockers {
            out.push_str(&format!("- {}\n", escape_md(blocker)));
        }
        out.push_str("\n## Required Commands\n\n```text\n");
        out.push_str(
            "cargo run -p gemma4d-bench --example m12_real_tiny16_matrix -- --out-dir benchmarks/out/M12/real-matrix --model-path artifacts/models/gemma-4-12B-it-4bit\n",
        );
        out.push_str(
            "cargo run -p gemma4d-bench --example p01_persistent_helper_session -- --out-dir benchmarks/out/P01-persistent-helper-session --model-path artifacts/models/gemma-4-12B-it-4bit --cold-records benchmarks/out/M12/real-matrix/records.jsonl\n",
        );
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

fn parse_positive_u32(value: &str, name: &str) -> Result<u32, Box<dyn std::error::Error>> {
    let parsed = value.parse::<u32>()?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero").into());
    }
    Ok(parsed)
}

fn run_id() -> String {
    format!("p01-{}", unix_now())
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
