use std::{
    collections::BTreeMap,
    env, fs,
    io::Write,
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_server::http::{
    ServerConfig, ServerRuntime, http_request, parse_bind_addr, serve_listener,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P02-real-server-inference";
const DEFAULT_P01_SUMMARY: &str = "benchmarks/out/P01-persistent-helper-session/summary.json";
const MODE: &str = "server_openai_http_real_helper_generate_per_request";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let curl_fixtures_path = args.out_dir.join("curl-fixtures.md");
    let environment = capture_environment();
    let p01_summary = load_p01_summary(&args.p01_summary_path)?;
    let mut blockers = Vec::new();

    if !args.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if p01_summary.is_none() {
        blockers.push(format!(
            "P01 summary is unavailable: {}",
            args.p01_summary_path.display()
        ));
    }

    let mut records = Vec::new();
    if blockers.is_empty() {
        records = run_server_cases(&args, &run_id)?;
    } else {
        for context_tokens in &args.contexts {
            records.push(blocked_record(&args, &run_id, *context_tokens));
        }
    }

    let comparisons = compare_to_p01(&args.contexts, &records, p01_summary.as_ref());
    let mut all_blockers = blockers;
    all_blockers.extend(
        records
            .iter()
            .filter_map(|record| record.blocker.clone())
            .collect::<Vec<_>>(),
    );
    let status = if !all_blockers.is_empty() {
        "blocked"
    } else if records.iter().all(|record| record.status == "passed") {
        "passed"
    } else {
        "failed"
    };

    let summary = P02Summary {
        schema_version: 1,
        goal: "P02-real-server-inference",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        model_path: args.model_path.display().to_string(),
        p01_summary_path: args.p01_summary_path.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        curl_fixtures_path: curl_fixtures_path.display().to_string(),
        environment,
        relevant_environment: capture_relevant_environment(),
        contexts: args.contexts.clone(),
        generated_tokens_requested: args.max_new_tokens,
        max_context_tokens: args.max_context_tokens,
        memory_budget_mb: args.memory_budget_mb,
        p01_status: p01_summary
            .as_ref()
            .and_then(|summary| summary.status.clone()),
        p01_warm_model_load_ms: p01_summary
            .as_ref()
            .and_then(|summary| summary.session.model_load_ms),
        cases: records,
        comparisons,
        blockers: all_blockers,
        measurement_notes: vec![
            "server cases use an actual localhost TcpListener and the OpenAI-compatible HTTP route.",
            "P02 real-helper mode calls the existing helper-backed generate path per request, so model_load_ms is per server request.",
            "P01 warm-session rows load the helper target once and reset KV cache between cases; P02 vs P01 deltas quantify the remaining persistent-server gap.",
            "nominal_context_tokens is generated as repeated text; actual_prompt_tokens comes from the server response usage after tokenizer encoding.",
            "curl-fixtures.md records exact manual curl commands for non-streaming, streaming SSE, and Prometheus metrics smoke checks.",
        ],
    };

    write_jsonl(&records_path, &summary.cases)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&curl_fixtures_path, render_curl_fixtures(&args))?;

    println!("P02 real server inference: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("curl fixtures: {}", curl_fixtures_path.display());

    if summary.status == "failed" {
        Err("P02 real server inference failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    p01_summary_path: PathBuf,
    contexts: Vec<usize>,
    max_new_tokens: usize,
    max_context_tokens: usize,
    memory_budget_mb: u64,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut p01_summary_path = PathBuf::from(DEFAULT_P01_SUMMARY);
        let mut contexts = vec![1024, 4096, 8192, 16_384];
        let mut max_new_tokens = 128;
        let mut max_context_tokens = 32_768;
        let mut memory_budget_mb = 12 * 1024;

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
                "--p01-summary" => {
                    p01_summary_path = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--p01-summary requires a path")?;
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
                "--memory-budget-mb" => {
                    let value = args.next().ok_or("--memory-budget-mb requires a value")?;
                    memory_budget_mb = value.parse::<u64>().map_err(|error| {
                        format!("--memory-budget-mb must be an integer: {error}")
                    })?;
                    if memory_budget_mb == 0 {
                        return Err("--memory-budget-mb must be greater than zero".into());
                    }
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example p02_real_server_inference -- [--out-dir PATH] [--model-path PATH] [--p01-summary PATH] [--contexts 1024,4096,8192,16384] [--max-new-tokens N]"
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
            p01_summary_path,
            contexts,
            max_new_tokens,
            max_context_tokens,
            memory_budget_mb,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct P02Summary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    model_path: String,
    p01_summary_path: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    curl_fixtures_path: String,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    contexts: Vec<usize>,
    generated_tokens_requested: usize,
    max_context_tokens: usize,
    memory_budget_mb: u64,
    p01_status: Option<String>,
    p01_warm_model_load_ms: Option<f64>,
    cases: Vec<P02Record>,
    comparisons: Vec<P02Comparison>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct P02Record {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    case_index: usize,
    workload: &'static str,
    nominal_context_tokens: usize,
    actual_prompt_tokens: Option<usize>,
    generated_tokens_requested: usize,
    generated_tokens_observed: Option<usize>,
    mode: &'static str,
    status: String,
    request_wall_ms: f64,
    response_status: u16,
    response_content_type: String,
    server_metrics: Option<ResponseMetrics>,
    prometheus: Option<PrometheusSnapshot>,
    assistant_content_preview: Option<String>,
    raw_error_body: Option<String>,
    blocker: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ResponseMetrics {
    input_tokens: usize,
    generated_tokens: usize,
    model_load_ms: f64,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    decode_tps: f64,
    decode_latency_ms: DecodeLatencySummary,
    mlx_active_memory_gb: Option<f64>,
    mlx_cache_memory_gb: Option<f64>,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
    request_overhead_ms: f64,
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
struct PrometheusSnapshot {
    requests_total: Option<f64>,
    model_load_seconds: Option<f64>,
    prefill_tokens_total: Option<f64>,
    decode_tokens_total: Option<f64>,
    prefill_seconds: Option<f64>,
    decode_seconds: Option<f64>,
    ttft_seconds: Option<f64>,
    tokens_per_second: Option<f64>,
    memory_process_rss_bytes: Option<f64>,
    memory_peak_mlx_bytes: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct P02Comparison {
    context_tokens: usize,
    p02_status: String,
    p01_warm_status: String,
    actual_prompt_tokens: Option<usize>,
    generated_tokens: Option<usize>,
    p02_request_wall_ms: Option<f64>,
    p02_model_load_ms: Option<f64>,
    p02_prefill_ms: Option<f64>,
    p02_decode_ms: Option<f64>,
    p02_total_ms: Option<f64>,
    p02_decode_tps: Option<f64>,
    p02_peak_memory_gb: Option<f64>,
    p02_peak_rss_mb: Option<f64>,
    p01_warm_case_ms_mean: Option<f64>,
    p01_warm_amortized_total_ms_mean: Option<f64>,
    p01_warm_prefill_ms_mean: Option<f64>,
    p01_warm_decode_ms_mean: Option<f64>,
    p01_warm_decode_tps_mean: Option<f64>,
    p01_warm_peak_memory_gb_max: Option<f64>,
    p01_warm_peak_rss_mb_max: Option<f64>,
    p02_total_minus_p01_warm_case_ms: Option<f64>,
    p02_wall_minus_p01_warm_amortized_ms: Option<f64>,
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

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Usage,
    gemma4d_metrics: ChatMetrics,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Message {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ChatMetrics {
    input_tokens: usize,
    generated_tokens: usize,
    model_load_ms: f64,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    decode_tps: f64,
    decode_token_latencies_ms: Vec<f64>,
    mlx_active_memory_gb: Option<f64>,
    mlx_cache_memory_gb: Option<f64>,
    peak_memory_gb: f64,
    peak_rss_mb: f64,
}

#[derive(Debug, Deserialize)]
struct P01Summary {
    status: Option<String>,
    session: P01Session,
}

#[derive(Debug, Deserialize)]
struct P01Session {
    model_load_ms: Option<f64>,
    records: Vec<P01Record>,
}

#[derive(Debug, Deserialize)]
struct P01Record {
    context_tokens: usize,
    status: String,
    total_case_ms: f64,
    total_with_amortized_load_ms: Option<f64>,
    prefill_ms: f64,
    decode_ms: f64,
    decode_tokens_per_second: f64,
    memory: P01Memory,
}

#[derive(Debug, Deserialize)]
struct P01Memory {
    mlx_peak_memory_gb: f64,
    helper_peak_rss_mb: f64,
}

fn run_server_cases(
    args: &Args,
    run_id: &str,
) -> Result<Vec<P02Record>, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let mut config = ServerConfig::localhost_default()
        .with_bind_addr(addr)
        .with_real_helper(args.model_path.clone());
    config.max_context_tokens = args.max_context_tokens;
    config.memory_budget_bytes = args.memory_budget_mb.saturating_mul(1024 * 1024);
    let runtime = ServerRuntime::new(config);
    let handle = thread::spawn(move || serve_listener(listener, runtime, server_shutdown));

    let mut records = Vec::new();
    for (index, context_tokens) in args.contexts.iter().enumerate() {
        records.push(run_case(args, run_id, addr, index + 1, *context_tokens));
    }

    shutdown.store(true, Ordering::SeqCst);
    let _ = TcpStream::connect(addr);
    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => return Err(format!("server listener failed: {error}").into()),
        Err(_) => return Err("server listener thread panicked".into()),
    }

    Ok(records)
}

fn run_case(
    args: &Args,
    run_id: &str,
    addr: std::net::SocketAddr,
    case_index: usize,
    context_tokens: usize,
) -> P02Record {
    let body = json!({
        "model": ServerConfig::default().model_id,
        "messages": [{"role": "user", "content": prompt_for_context(context_tokens)}],
        "stream": false,
        "temperature": 0,
        "max_tokens": args.max_new_tokens,
    })
    .to_string();
    let started = Instant::now();
    let response = http_request(addr, "POST", "/v1/chat/completions", Some(&body));
    let request_wall_ms = duration_ms(started.elapsed());

    match response {
        Ok(response) if response.status == 200 => passed_record(
            args,
            run_id,
            addr,
            case_index,
            context_tokens,
            request_wall_ms,
            response,
        ),
        Ok(response) => {
            let status = response.status;
            error_record(
                args,
                run_id,
                case_index,
                context_tokens,
                request_wall_ms,
                Some(response),
                format!("server returned HTTP {status}"),
            )
        }
        Err(error) => error_record(
            args,
            run_id,
            case_index,
            context_tokens,
            request_wall_ms,
            None,
            format!("server request failed: {error}"),
        ),
    }
}

fn passed_record(
    args: &Args,
    run_id: &str,
    addr: std::net::SocketAddr,
    case_index: usize,
    context_tokens: usize,
    request_wall_ms: f64,
    response: gemma4d_server::http::HttpResponse,
) -> P02Record {
    let parsed = serde_json::from_str::<ChatResponse>(&response.body);
    let metrics_response = http_request(addr, "GET", "/metrics", None).ok();
    let prometheus = metrics_response
        .as_ref()
        .map(|response| parse_prometheus(&response.body));

    match parsed {
        Ok(parsed) => {
            let metrics = response_metrics(parsed.gemma4d_metrics, request_wall_ms);
            let status = if parsed.usage.prompt_tokens != metrics.input_tokens {
                "failed"
            } else {
                "passed"
            };
            let blocker = (status == "failed").then(|| {
                format!(
                    "response usage prompt_tokens {} did not match gemma4d_metrics input_tokens {}",
                    parsed.usage.prompt_tokens, metrics.input_tokens
                )
            });
            P02Record {
                schema_version: 1,
                goal: "P02-real-server-inference",
                run_id: run_id.to_owned(),
                timestamp_unix: unix_now(),
                case_index,
                workload: "server_chat_repeated_text",
                nominal_context_tokens: context_tokens,
                actual_prompt_tokens: Some(parsed.usage.prompt_tokens),
                generated_tokens_requested: args.max_new_tokens,
                generated_tokens_observed: Some(parsed.usage.completion_tokens),
                mode: MODE,
                status: status.to_owned(),
                request_wall_ms,
                response_status: response.status,
                response_content_type: response.content_type,
                server_metrics: Some(metrics),
                prometheus,
                assistant_content_preview: parsed
                    .choices
                    .first()
                    .map(|choice| preview(&choice.message.content, 240)),
                raw_error_body: None,
                blocker,
            }
        }
        Err(error) => error_record(
            args,
            run_id,
            case_index,
            context_tokens,
            request_wall_ms,
            Some(response),
            format!("failed to parse chat response JSON: {error}"),
        ),
    }
}

fn response_metrics(metrics: ChatMetrics, request_wall_ms: f64) -> ResponseMetrics {
    let decode_latency_ms = decode_latency_summary(metrics.decode_token_latencies_ms);
    ResponseMetrics {
        input_tokens: metrics.input_tokens,
        generated_tokens: metrics.generated_tokens,
        model_load_ms: metrics.model_load_ms,
        prefill_ms: metrics.prefill_ms,
        ttft_ms: metrics.ttft_ms,
        decode_ms: metrics.decode_ms,
        total_ms: metrics.total_ms,
        decode_tps: metrics.decode_tps,
        decode_latency_ms,
        mlx_active_memory_gb: metrics.mlx_active_memory_gb,
        mlx_cache_memory_gb: metrics.mlx_cache_memory_gb,
        peak_memory_gb: metrics.peak_memory_gb,
        peak_rss_mb: metrics.peak_rss_mb,
        request_overhead_ms: (request_wall_ms - metrics.total_ms).max(0.0),
    }
}

fn error_record(
    args: &Args,
    run_id: &str,
    case_index: usize,
    context_tokens: usize,
    request_wall_ms: f64,
    response: Option<gemma4d_server::http::HttpResponse>,
    blocker: String,
) -> P02Record {
    let (response_status, response_content_type, raw_error_body) = response
        .map(|response| (response.status, response.content_type, Some(response.body)))
        .unwrap_or((0, String::new(), None));
    P02Record {
        schema_version: 1,
        goal: "P02-real-server-inference",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        case_index,
        workload: "server_chat_repeated_text",
        nominal_context_tokens: context_tokens,
        actual_prompt_tokens: None,
        generated_tokens_requested: args.max_new_tokens,
        generated_tokens_observed: None,
        mode: MODE,
        status: "failed".to_owned(),
        request_wall_ms,
        response_status,
        response_content_type,
        server_metrics: None,
        prometheus: None,
        assistant_content_preview: None,
        raw_error_body,
        blocker: Some(blocker),
    }
}

fn blocked_record(args: &Args, run_id: &str, context_tokens: usize) -> P02Record {
    P02Record {
        schema_version: 1,
        goal: "P02-real-server-inference",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        case_index: 0,
        workload: "server_chat_repeated_text",
        nominal_context_tokens: context_tokens,
        actual_prompt_tokens: None,
        generated_tokens_requested: args.max_new_tokens,
        generated_tokens_observed: None,
        mode: MODE,
        status: "blocked".to_owned(),
        request_wall_ms: 0.0,
        response_status: 0,
        response_content_type: String::new(),
        server_metrics: None,
        prometheus: None,
        assistant_content_preview: None,
        raw_error_body: None,
        blocker: Some("prerequisite artifact is missing".to_owned()),
    }
}

fn load_p01_summary(path: &Path) -> Result<Option<P01Summary>, Box<dyn std::error::Error>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn compare_to_p01(
    contexts: &[usize],
    records: &[P02Record],
    p01_summary: Option<&P01Summary>,
) -> Vec<P02Comparison> {
    contexts
        .iter()
        .map(|context_tokens| {
            let p02 = records
                .iter()
                .find(|record| record.nominal_context_tokens == *context_tokens);
            let p01 = p01_summary
                .map(|summary| {
                    summary
                        .session
                        .records
                        .iter()
                        .filter(|record| record.context_tokens == *context_tokens)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let p02_metrics = p02.and_then(|record| record.server_metrics.as_ref());
            let p01_warm_case_ms_mean = mean(p01.iter().map(|record| record.total_case_ms));
            let p01_warm_amortized_total_ms_mean = mean(
                p01.iter()
                    .filter_map(|record| record.total_with_amortized_load_ms),
            );
            let p02_total_ms = p02_metrics.map(|metrics| metrics.total_ms);
            let p02_request_wall_ms = p02.map(|record| record.request_wall_ms);

            P02Comparison {
                context_tokens: *context_tokens,
                p02_status: p02
                    .map(|record| record.status.clone())
                    .unwrap_or_else(|| "missing".to_owned()),
                p01_warm_status: p01_status(&p01),
                actual_prompt_tokens: p02.and_then(|record| record.actual_prompt_tokens),
                generated_tokens: p02.and_then(|record| record.generated_tokens_observed),
                p02_request_wall_ms,
                p02_model_load_ms: p02_metrics.map(|metrics| metrics.model_load_ms),
                p02_prefill_ms: p02_metrics.map(|metrics| metrics.prefill_ms),
                p02_decode_ms: p02_metrics.map(|metrics| metrics.decode_ms),
                p02_total_ms,
                p02_decode_tps: p02_metrics.map(|metrics| metrics.decode_tps),
                p02_peak_memory_gb: p02_metrics.map(|metrics| metrics.peak_memory_gb),
                p02_peak_rss_mb: p02_metrics.map(|metrics| metrics.peak_rss_mb),
                p01_warm_case_ms_mean,
                p01_warm_amortized_total_ms_mean,
                p01_warm_prefill_ms_mean: mean(p01.iter().map(|record| record.prefill_ms)),
                p01_warm_decode_ms_mean: mean(p01.iter().map(|record| record.decode_ms)),
                p01_warm_decode_tps_mean: mean(
                    p01.iter().map(|record| record.decode_tokens_per_second),
                ),
                p01_warm_peak_memory_gb_max: max(p01
                    .iter()
                    .map(|record| record.memory.mlx_peak_memory_gb)),
                p01_warm_peak_rss_mb_max: max(p01
                    .iter()
                    .map(|record| record.memory.helper_peak_rss_mb)),
                p02_total_minus_p01_warm_case_ms: delta(p01_warm_case_ms_mean, p02_total_ms),
                p02_wall_minus_p01_warm_amortized_ms: delta(
                    p01_warm_amortized_total_ms_mean,
                    p02_request_wall_ms,
                ),
            }
        })
        .collect()
}

fn p01_status(records: &[&P01Record]) -> String {
    if records.is_empty() {
        "missing".to_owned()
    } else if records.iter().all(|record| record.status == "passed") {
        "passed".to_owned()
    } else {
        "failed".to_owned()
    }
}

fn parse_prometheus(body: &str) -> PrometheusSnapshot {
    PrometheusSnapshot {
        requests_total: metric_value(body, "gemma4d_requests_total"),
        model_load_seconds: metric_value(body, "gemma4d_model_load_seconds"),
        prefill_tokens_total: metric_value(body, "gemma4d_prefill_tokens_total"),
        decode_tokens_total: metric_value(body, "gemma4d_decode_tokens_total"),
        prefill_seconds: metric_value(body, "gemma4d_prefill_seconds"),
        decode_seconds: metric_value(body, "gemma4d_decode_seconds"),
        ttft_seconds: metric_value(body, "gemma4d_ttft_seconds"),
        tokens_per_second: metric_value(body, "gemma4d_tokens_per_second"),
        memory_process_rss_bytes: metric_value(body, "gemma4d_memory_process_rss_bytes"),
        memory_peak_mlx_bytes: metric_value(body, "gemma4d_memory_peak_mlx_bytes"),
    }
}

fn metric_value(body: &str, name: &str) -> Option<f64> {
    body.lines().find_map(|line| {
        let (metric, value) = line.split_once(' ')?;
        (metric == name)
            .then(|| value.parse::<f64>().ok())
            .flatten()
    })
}

fn prompt_for_context(context_tokens: usize) -> String {
    std::iter::repeat_n("hello", context_tokens)
        .collect::<Vec<_>>()
        .join(" ")
}

fn write_jsonl(path: &Path, records: &[P02Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
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

fn delta(baseline: Option<f64>, observed: Option<f64>) -> Option<f64> {
    match (baseline, observed) {
        (Some(baseline), Some(observed)) => Some(observed - baseline),
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

fn render_report(summary: &P02Summary) -> String {
    let mut out = String::new();
    out.push_str("# P02 Real Server Inference\n\n");
    out.push_str("## Status\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Mode: `{}`\n- Records: `{}`\n- Summary: `{}`\n- P01 source: `{}`\n- Curl fixtures: `{}`\n\n",
        summary.status,
        summary.mode,
        summary.records_path,
        summary.summary_path,
        summary.p01_summary_path,
        summary.curl_fixtures_path,
    ));
    out.push_str("## Server vs P01 Warm Session\n\n");
    out.push_str("| Context | Actual prompt | Generated | P02 status | P02 wall ms | P02 load ms | P02 prefill ms | P02 decode ms | P02 total ms | P02 tok/s | P01 warm case ms | P01 warm amortized ms | Total delta ms | Wall delta ms | P02 peak GB | P01 peak GB | P02 RSS MB | P01 RSS MB |\n");
    out.push_str("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for comparison in &summary.comparisons {
        out.push_str(&format!(
            "| {} | {} | {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            comparison.context_tokens,
            fmt_opt_usize(comparison.actual_prompt_tokens),
            fmt_opt_usize(comparison.generated_tokens),
            comparison.p02_status,
            fmt_opt(comparison.p02_request_wall_ms),
            fmt_opt(comparison.p02_model_load_ms),
            fmt_opt(comparison.p02_prefill_ms),
            fmt_opt(comparison.p02_decode_ms),
            fmt_opt(comparison.p02_total_ms),
            fmt_opt(comparison.p02_decode_tps),
            fmt_opt(comparison.p01_warm_case_ms_mean),
            fmt_opt(comparison.p01_warm_amortized_total_ms_mean),
            fmt_opt(comparison.p02_total_minus_p01_warm_case_ms),
            fmt_opt(comparison.p02_wall_minus_p01_warm_amortized_ms),
            fmt_opt(comparison.p02_peak_memory_gb),
            fmt_opt(comparison.p01_warm_peak_memory_gb_max),
            fmt_opt(comparison.p02_peak_rss_mb),
            fmt_opt(comparison.p01_warm_peak_rss_mb_max),
        ));
    }
    out.push_str("\n## Server Cases\n\n");
    out.push_str("| Case | Context | Prompt tokens | Generated | Status | HTTP | Wall ms | Load ms | Prefill ms | Decode ms | Total ms | Decode p50 ms | Decode p95 ms | Overhead ms | Peak GB | RSS MB |\n");
    out.push_str(
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    for record in &summary.cases {
        let metrics = record.server_metrics.as_ref();
        out.push_str(&format!(
            "| {} | {} | {} | {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            record.case_index,
            record.nominal_context_tokens,
            fmt_opt_usize(record.actual_prompt_tokens),
            fmt_opt_usize(record.generated_tokens_observed),
            record.status,
            record.response_status,
            fmt_num(record.request_wall_ms),
            fmt_opt(metrics.map(|metrics| metrics.model_load_ms)),
            fmt_opt(metrics.map(|metrics| metrics.prefill_ms)),
            fmt_opt(metrics.map(|metrics| metrics.decode_ms)),
            fmt_opt(metrics.map(|metrics| metrics.total_ms)),
            fmt_opt(metrics.and_then(|metrics| metrics.decode_latency_ms.p50_ms)),
            fmt_opt(metrics.and_then(|metrics| metrics.decode_latency_ms.p95_ms)),
            fmt_opt(metrics.map(|metrics| metrics.request_overhead_ms)),
            fmt_opt(metrics.map(|metrics| metrics.peak_memory_gb)),
            fmt_opt(metrics.map(|metrics| metrics.peak_rss_mb)),
        ));
    }
    out.push_str("\n## Prometheus Snapshot After Each Case\n\n");
    out.push_str("| Context | Requests | Load s | Prefill tokens | Decode tokens | Prefill s | Decode s | Tok/s | Peak MLX bytes | RSS bytes |\n");
    out.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.cases {
        let metrics = record.prometheus.as_ref();
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            record.nominal_context_tokens,
            fmt_opt(metrics.and_then(|metrics| metrics.requests_total)),
            fmt_opt(metrics.and_then(|metrics| metrics.model_load_seconds)),
            fmt_opt(metrics.and_then(|metrics| metrics.prefill_tokens_total)),
            fmt_opt(metrics.and_then(|metrics| metrics.decode_tokens_total)),
            fmt_opt(metrics.and_then(|metrics| metrics.prefill_seconds)),
            fmt_opt(metrics.and_then(|metrics| metrics.decode_seconds)),
            fmt_opt(metrics.and_then(|metrics| metrics.tokens_per_second)),
            fmt_opt(metrics.and_then(|metrics| metrics.memory_peak_mlx_bytes)),
            fmt_opt(metrics.and_then(|metrics| metrics.memory_process_rss_bytes)),
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

fn render_blockers(summary: &P02Summary) -> String {
    if summary.blockers.is_empty() {
        return "No blockers recorded.\n".to_owned();
    }
    let mut out = String::new();
    out.push_str("# P02 Blockers\n\n");
    for blocker in &summary.blockers {
        out.push_str(&format!("- {blocker}\n"));
    }
    out
}

fn render_curl_fixtures(args: &Args) -> String {
    let model_path = args.model_path.display();
    let default_config = ServerConfig::default();
    let model_id = &default_config.model_id;
    let addr = parse_bind_addr("127.0.0.1:18082").expect("fixture addr");
    format!(
        r#"# P02 Curl Fixtures

Start the real-helper server:

```sh
cargo run -p gemma4d-server -- serve --bind {addr} --backend real-helper --model-path {model_path} --max-context-tokens {max_context_tokens} --memory-budget-mb {memory_budget_mb}
```

Non-streaming smoke:

```sh
curl -sS -i -X POST http://{addr}/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{{"model":"{model_id}","messages":[{{"role":"user","content":"Say hello in one sentence."}}],"temperature":0,"max_tokens":8}}'
```

Streaming SSE smoke:

```sh
curl -sS -i -N -X POST http://{addr}/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{{"model":"{model_id}","messages":[{{"role":"user","content":"Say hello in one sentence."}}],"stream":true,"temperature":0,"max_tokens":8}}'
```

Prometheus metrics smoke:

```sh
curl -sS http://{addr}/metrics
```
"#,
        addr = addr,
        model_path = model_path,
        max_context_tokens = args.max_context_tokens,
        memory_budget_mb = args.memory_budget_mb,
        model_id = model_id,
    )
}

fn parse_contexts(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| parse_positive_usize(value, "--contexts"))
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

fn preview(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn fmt_num(value: f64) -> String {
    format!("{value:.3}")
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map(fmt_num).unwrap_or_else(|| "n/a".to_owned())
}

fn fmt_opt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_owned())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', "<br>")
}

fn run_id() -> String {
    format!("p02-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after Unix epoch")
        .as_secs()
}
