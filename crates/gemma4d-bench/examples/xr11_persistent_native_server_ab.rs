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
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::manifest;
use gemma4d_server::{
    config_from_serve_options,
    http::{ServerBackend, ServerConfig, ServerRuntime, http_request, serve_listener},
    parse_serve_options,
};
use gemma4d_tokenizer::sha256_hex;
use serde::{Deserialize, Serialize};
use serde_json::json;

const GOAL: &str = "XR11-persistent-native-server-ab";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR11-persistent-native-server-ab";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_WORKLOADS: &str = "benchmarks/workloads/real-contexts/workloads.jsonl";
const MODE: &str = "server_real_helper_vs_default_persistent_native_real_contexts";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let decision_path = args.out_dir.join("decision.md");
    let run_id = run_id();
    let command = command_display(&args);
    let environment = capture_environment();
    let relevant_environment = capture_relevant_environment();
    let model_identity =
        manifest::capture_artifact_identity(&args.model_path, "GEMMA4D_MODEL_REVISION");
    let mut blockers = Vec::new();

    if !args.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if !args.workloads_path.exists() {
        blockers.push(format!(
            "workload manifest does not exist: {}",
            args.workloads_path.display()
        ));
    }

    let workloads = if args.workloads_path.exists() {
        load_workloads(&args.workloads_path)?
    } else {
        Vec::new()
    };
    let selected = select_workloads(&workloads, &args.workload_ids, args.max_workloads);
    for workload_id in &args.workload_ids {
        if !selected
            .iter()
            .any(|workload| workload.workload_id == *workload_id)
        {
            blockers.push(format!(
                "selected workload id is unavailable in {}: {workload_id}",
                args.workloads_path.display()
            ));
        }
    }
    for workload in &selected {
        if !Path::new(&workload.prompt_path).exists() {
            blockers.push(format!(
                "prompt path does not exist for {}: {}",
                workload.workload_id, workload.prompt_path
            ));
        }
    }

    let mut records = Vec::new();
    let mut candidate_warmups = Vec::new();
    let mut final_metrics = FinalMetrics::default();
    if blockers.is_empty() {
        match run_cases(&args, &selected, &run_id, &environment, &model_identity) {
            Ok(run) => {
                records = run.records;
                candidate_warmups = run.candidate_warmups;
                final_metrics = run.final_metrics;
            }
            Err(error) => blockers.push(error.to_string()),
        }
    }
    if !blockers.is_empty() && records.is_empty() {
        records = blocked_records(
            &args,
            &selected,
            &run_id,
            &blockers,
            &environment,
            &model_identity,
        )?;
    }

    let comparison_blockers = records
        .iter()
        .flat_map(|record| record.blockers.iter().cloned())
        .collect::<Vec<_>>();
    let warmup_blockers = candidate_warmups
        .iter()
        .filter(|warmup| warmup.status != "passed")
        .map(|warmup| {
            format!(
                "{} candidate prefix warmup failed: {}",
                warmup.workload_id,
                warmup.error.as_deref().unwrap_or("unknown error")
            )
        })
        .collect::<Vec<_>>();
    let mut all_blockers = blockers;
    all_blockers.extend(comparison_blockers);
    all_blockers.extend(warmup_blockers);
    all_blockers.sort();
    all_blockers.dedup();

    let token_text_matches = records
        .iter()
        .all(|record| record.comparison_status == "tokens_and_text_match");
    let candidate_warmups_ok = args.candidate_prefix_warmup_tokens.is_none()
        || candidate_warmups
            .iter()
            .all(|warmup| warmup.status == "passed");
    let candidate_load_count_ok = final_metrics
        .candidate
        .as_ref()
        .is_some_and(|metrics| metrics.model_load_count == Some(1.0));
    let baseline_load_count_ok = baseline_load_count_ok(
        args.baseline_backend,
        final_metrics.baseline.as_ref(),
        &records,
    );
    let decision = if !all_blockers.is_empty() {
        "blocked_with_evidence"
    } else if token_text_matches
        && candidate_warmups_ok
        && candidate_load_count_ok
        && baseline_load_count_ok
    {
        "accept_candidate"
    } else if token_text_matches {
        "needs_more_data"
    } else {
        "reject_candidate"
    };
    let status = if all_blockers.is_empty() && decision == "accept_candidate" {
        "passed"
    } else if decision.starts_with("blocked") {
        "blocked"
    } else {
        "failed"
    };

    let generated_files = vec![
        records_path.display().to_string(),
        summary_path.display().to_string(),
        report_path.display().to_string(),
        blockers_path.display().to_string(),
        decision_path.display().to_string(),
    ];
    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        status: status.to_owned(),
        decision: decision.to_owned(),
        run_id,
        mode: mode_for_args(&args).to_owned(),
        command,
        model_path: args.model_path.display().to_string(),
        model_identity,
        workloads_path: args.workloads_path.display().to_string(),
        out_dir: args.out_dir.display().to_string(),
        repeats: args.repeats,
        max_new_tokens: args.max_new_tokens,
        max_context_tokens: args.max_context_tokens,
        memory_budget_mb: args.memory_budget_mb,
        baseline_backend: args.baseline_backend,
        candidate_prefix_warmup_tokens: args.candidate_prefix_warmup_tokens,
        selected_workloads: selected
            .iter()
            .map(|workload| workload.workload_id.clone())
            .collect(),
        generated_files,
        environment,
        relevant_environment,
        final_metrics,
        blockers: all_blockers,
        candidate_warmups,
        records: records.clone(),
        measurement_notes: vec![
            "The baseline backend is selected by --baseline-backend. The default baseline is ServerBackend::RealHelper; persistent-native baseline mode passes an explicit --backend persistent-native flag.",
            "Candidate is built from parse_serve_options with --model-path and no --backend flag; XR53 defaults that path to ServerBackend::PersistentNative.",
            "The persistent-native worker owns one ResidentTarget and creates fresh KV per request.",
            "Optional candidate prefix warmup uses the local /v1/runtime/warmup/prefix control endpoint before measured candidate requests; warmup cost is recorded separately and is not included in request gemma4d_metrics.",
            "The benchmark compares response text and generated token ids from gemma4d_metrics.generated_token_ids.",
            "Tokenizer and detokenizer helper costs remain in request handling for both backends; XR11 only evaluates target-load residency.",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, render_decision(&summary))?;

    println!("XR11 persistent native server A/B: {}", summary.status);
    println!("decision: {}", summary.decision);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision path: {}", decision_path.display());

    if summary.status == "failed" {
        Err("XR11 persistent native server A/B failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    workloads_path: PathBuf,
    workload_ids: Vec<String>,
    max_workloads: Option<usize>,
    repeats: usize,
    max_new_tokens: usize,
    max_context_tokens: usize,
    memory_budget_mb: u64,
    baseline_backend: ServerBackend,
    candidate_prefix_warmup_tokens: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut workloads_path = PathBuf::from(DEFAULT_WORKLOADS);
        let mut workload_ids = vec![
            "chat_short_1k_001".to_owned(),
            "tool_json_1k_001".to_owned(),
            "mtp_candidate_1k_001".to_owned(),
        ];
        let mut max_workloads = None;
        let mut repeats = 2usize;
        let mut max_new_tokens = 8usize;
        let mut max_context_tokens = 32_768usize;
        let mut memory_budget_mb = 12 * 1024u64;
        let mut baseline_backend = ServerBackend::RealHelper;
        let mut candidate_prefix_warmup_tokens = None;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => out_dir = required_path(&mut args, "--out-dir")?,
                "--model-path" => model_path = required_path(&mut args, "--model-path")?,
                "--workloads" => workloads_path = required_path(&mut args, "--workloads")?,
                "--workload-id" => workload_ids.push(required(&mut args, "--workload-id")?),
                "--workload-ids" => {
                    workload_ids = parse_csv(&required(&mut args, "--workload-ids")?)
                }
                "--clear-workload-ids" => workload_ids.clear(),
                "--max-workloads" => {
                    max_workloads = Some(parse_positive_usize(
                        &required(&mut args, "--max-workloads")?,
                        "--max-workloads",
                    )?);
                }
                "--repeats" => {
                    repeats =
                        parse_positive_usize(&required(&mut args, "--repeats")?, "--repeats")?;
                }
                "--max-new-tokens" => {
                    max_new_tokens = parse_positive_usize(
                        &required(&mut args, "--max-new-tokens")?,
                        "--max-new-tokens",
                    )?;
                }
                "--max-context-tokens" => {
                    max_context_tokens = parse_positive_usize(
                        &required(&mut args, "--max-context-tokens")?,
                        "--max-context-tokens",
                    )?;
                }
                "--memory-budget-mb" => {
                    memory_budget_mb = required(&mut args, "--memory-budget-mb")?
                        .parse::<u64>()
                        .map_err(|error| {
                            format!("--memory-budget-mb must be an integer: {error}")
                        })?;
                    if memory_budget_mb == 0 {
                        return Err("--memory-budget-mb must be greater than zero".into());
                    }
                }
                "--baseline-backend" => {
                    baseline_backend =
                        parse_baseline_backend(&required(&mut args, "--baseline-backend")?)?;
                }
                "--candidate-prefix-warmup-tokens" => {
                    candidate_prefix_warmup_tokens = Some(parse_positive_usize(
                        &required(&mut args, "--candidate-prefix-warmup-tokens")?,
                        "--candidate-prefix-warmup-tokens",
                    )?);
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- [--out-dir PATH] [--model-path PATH] [--workloads PATH] [--workload-ids CSV] [--repeats N] [--max-new-tokens N] [--baseline-backend real-helper|persistent-native] [--candidate-prefix-warmup-tokens N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        if workload_ids.is_empty() {
            return Err("at least one workload id is required".into());
        }

        Ok(Self {
            out_dir,
            model_path,
            workloads_path,
            workload_ids,
            max_workloads,
            repeats,
            max_new_tokens,
            max_context_tokens,
            memory_budget_mb,
            baseline_backend,
            candidate_prefix_warmup_tokens,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkloadRecord {
    schema_version: u32,
    workload_id: String,
    family: String,
    source_files: Vec<String>,
    prompt_path: String,
    expected_output_style: String,
    max_new_tokens: usize,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    deterministic_seed: u64,
    prompt_sha256: String,
    tokenizer_backend: String,
    notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Xr11Record {
    schema_version: u32,
    goal: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    model_identity: manifest::ArtifactIdentity,
    timestamp_unix: u64,
    workload_id: String,
    family: String,
    prompt_path: String,
    prompt_sha256: String,
    prompt_bytes: usize,
    deterministic_seed: u64,
    target_context_tokens: usize,
    actual_context_tokens: usize,
    tokenizer_backend: String,
    repeat_index: usize,
    max_new_tokens: usize,
    comparison_status: String,
    baseline: BackendRun,
    candidate: BackendRun,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendRun {
    backend: String,
    status: String,
    http_status: Option<u16>,
    request_wall_ms: Option<f64>,
    response_text: String,
    response_text_sha256: String,
    generated_token_ids: Vec<i32>,
    usage: Option<UsageRecord>,
    metrics: Option<ChatMetrics>,
    prometheus: Option<PrometheusSnapshot>,
    runtime_snapshot: Option<serde_json::Value>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixWarmupRun {
    workload_id: String,
    backend: String,
    status: String,
    http_status: Option<u16>,
    request_wall_ms: Option<f64>,
    requested_prefix_tokens: Option<usize>,
    prompt_tokens: Option<usize>,
    warmup_context_tokens: Option<usize>,
    tokenize_ms: Option<f64>,
    prefill_ms: Option<f64>,
    decode_ms: Option<f64>,
    total_ms: Option<f64>,
    peak_memory_gb: Option<f64>,
    active_kv_bytes: Option<u64>,
    runtime_snapshot: Option<serde_json::Value>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UsageRecord {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMetrics {
    input_tokens: usize,
    generated_tokens: usize,
    #[serde(default)]
    generated_token_ids: Vec<i32>,
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
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PrometheusSnapshot {
    requests_total: Option<f64>,
    model_load_seconds: Option<f64>,
    model_load_count: Option<f64>,
    resident_model_loaded: Option<f64>,
    persistent_worker_requests_total: Option<f64>,
    prefill_tokens_total: Option<f64>,
    decode_tokens_total: Option<f64>,
    prefill_seconds: Option<f64>,
    decode_seconds: Option<f64>,
    tokens_per_second: Option<f64>,
    memory_peak_mlx_bytes: Option<f64>,
    memory_process_rss_bytes: Option<f64>,
    prefix_warmups_total: Option<f64>,
    prefix_warmup_tokens_total: Option<f64>,
    prefix_warmup_seconds: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct FinalMetrics {
    baseline: Option<PrometheusSnapshot>,
    candidate: Option<PrometheusSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: String,
    status: String,
    decision: String,
    run_id: String,
    mode: String,
    command: String,
    model_path: String,
    model_identity: manifest::ArtifactIdentity,
    workloads_path: String,
    out_dir: String,
    repeats: usize,
    max_new_tokens: usize,
    max_context_tokens: usize,
    memory_budget_mb: u64,
    baseline_backend: ServerBackend,
    candidate_prefix_warmup_tokens: Option<usize>,
    selected_workloads: Vec<String>,
    generated_files: Vec<String>,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    final_metrics: FinalMetrics,
    blockers: Vec<String>,
    candidate_warmups: Vec<PrefixWarmupRun>,
    records: Vec<Xr11Record>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct Environment {
    os: String,
    arch: String,
    rustc: String,
    git_sha: String,
    git_status_short: String,
}

struct RunOutput {
    records: Vec<Xr11Record>,
    candidate_warmups: Vec<PrefixWarmupRun>,
    final_metrics: FinalMetrics,
}

struct RunningServer {
    addr: std::net::SocketAddr,
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<gemma4d_server::http::HttpResult<()>>>,
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_cases(
    args: &Args,
    workloads: &[WorkloadRecord],
    run_id: &str,
    environment: &Environment,
    model_identity: &manifest::ArtifactIdentity,
) -> Result<RunOutput, Box<dyn std::error::Error>> {
    let baseline_server = start_server(args, args.baseline_backend, true)?;
    let mut baseline = Vec::new();
    for workload in workloads {
        let prompt = fs::read_to_string(&workload.prompt_path)?;
        for repeat_index in 0..args.repeats {
            baseline.push((
                workload.workload_id.clone(),
                repeat_index,
                run_backend_request(
                    baseline_server.addr,
                    args.baseline_backend,
                    &prompt,
                    args.max_new_tokens,
                ),
            ));
        }
    }
    let baseline_final_metrics = fetch_prometheus(baseline_server.addr).unwrap_or_default();
    drop(baseline_server);

    let candidate_server = start_server(args, ServerBackend::PersistentNative, false)?;
    let mut candidate_warmups = Vec::new();
    let mut records = Vec::new();
    let mut baseline_iter = baseline.into_iter();
    for workload in workloads {
        let prompt = fs::read_to_string(&workload.prompt_path)?;
        let prompt_bytes = prompt.len();
        let computed_sha = sha256_hex(prompt.as_bytes());
        if let Some(prefix_tokens) = args.candidate_prefix_warmup_tokens {
            candidate_warmups.push(run_prefix_warmup(
                candidate_server.addr,
                &workload.workload_id,
                &prompt,
                prefix_tokens,
            ));
        }
        for repeat_index in 0..args.repeats {
            let Some((_, _, baseline_run)) = baseline_iter.next() else {
                return Err("baseline record count did not match candidate loop".into());
            };
            let candidate_run = run_backend_request(
                candidate_server.addr,
                ServerBackend::PersistentNative,
                &prompt,
                args.max_new_tokens,
            );
            let blockers = compare_runs(workload, &baseline_run, &candidate_run);
            let comparison_status = if blockers.is_empty() {
                "tokens_and_text_match"
            } else {
                "mismatch_or_request_error"
            };
            records.push(Xr11Record {
                schema_version: 1,
                goal: GOAL.to_owned(),
                run_id: run_id.to_owned(),
                git_sha: environment.git_sha.clone(),
                git_status_short: environment.git_status_short.clone(),
                model_identity: model_identity.clone(),
                timestamp_unix: unix_now(),
                workload_id: workload.workload_id.clone(),
                family: workload.family.clone(),
                prompt_path: workload.prompt_path.clone(),
                prompt_sha256: computed_sha.clone(),
                prompt_bytes,
                deterministic_seed: workload.deterministic_seed,
                target_context_tokens: workload.target_context_tokens,
                actual_context_tokens: workload.actual_context_tokens,
                tokenizer_backend: workload.tokenizer_backend.clone(),
                repeat_index,
                max_new_tokens: args.max_new_tokens,
                comparison_status: comparison_status.to_owned(),
                baseline: baseline_run,
                candidate: candidate_run,
                blockers,
            });
        }
    }
    let candidate_final_metrics = fetch_prometheus(candidate_server.addr).unwrap_or_default();
    drop(candidate_server);

    Ok(RunOutput {
        records,
        candidate_warmups,
        final_metrics: FinalMetrics {
            baseline: Some(baseline_final_metrics),
            candidate: Some(candidate_final_metrics),
        },
    })
}

fn start_server(
    args: &Args,
    backend: ServerBackend,
    explicit_backend_flag: bool,
) -> Result<RunningServer, Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let config = server_config_from_args(args, backend, explicit_backend_flag, addr)?;
    let runtime = ServerRuntime::new(config);
    let handle = thread::spawn(move || serve_listener(listener, runtime, server_shutdown));
    wait_for_health(addr)?;
    Ok(RunningServer {
        addr,
        shutdown,
        handle: Some(handle),
    })
}

fn server_config_from_args(
    args: &Args,
    backend: ServerBackend,
    explicit_backend_flag: bool,
    addr: std::net::SocketAddr,
) -> Result<ServerConfig, Box<dyn std::error::Error>> {
    let mut serve_args = vec![
        "--bind".to_owned(),
        addr.to_string(),
        "--model-path".to_owned(),
        args.model_path.display().to_string(),
        "--max-context-tokens".to_owned(),
        args.max_context_tokens.to_string(),
        "--memory-budget-mb".to_owned(),
        args.memory_budget_mb.to_string(),
    ];
    if explicit_backend_flag {
        serve_args.extend(["--backend".to_owned(), backend.cli_name().to_owned()]);
    }
    let config = config_from_serve_options(parse_serve_options(serve_args)?);
    if config.backend != backend {
        return Err(format!(
            "parsed server backend mismatch: expected {}, got {}",
            backend.as_str(),
            config.backend.as_str()
        )
        .into());
    }
    Ok(config)
}

fn wait_for_health(addr: std::net::SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    loop {
        if let Ok(response) = http_request(addr, "GET", "/health", None)
            && response.status == 200
        {
            return Ok(());
        }
        if started.elapsed() > Duration::from_secs(5) {
            return Err(format!("server at {addr} did not become healthy within 5s").into());
        }
        thread::sleep(Duration::from_millis(25));
    }
}

fn run_backend_request(
    addr: std::net::SocketAddr,
    backend: ServerBackend,
    prompt: &str,
    max_new_tokens: usize,
) -> BackendRun {
    let body = json!({
        "model": ServerConfig::default().model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_new_tokens,
    });
    let body = serde_json::to_string(&body).expect("request serializes");
    let started = Instant::now();
    let response = http_request(addr, "POST", "/v1/chat/completions", Some(&body));
    let request_wall_ms = duration_ms(started.elapsed());
    let prometheus = fetch_prometheus(addr);
    let runtime_snapshot = fetch_runtime_snapshot(addr);

    match response {
        Ok(response) if response.status == 200 => {
            match serde_json::from_str::<serde_json::Value>(&response.body) {
                Ok(value) => {
                    let response_text = value["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .to_owned();
                    let usage = value.get("usage").map(|usage| UsageRecord {
                        prompt_tokens: usize_at(usage, "prompt_tokens"),
                        completion_tokens: usize_at(usage, "completion_tokens"),
                        total_tokens: usize_at(usage, "total_tokens"),
                    });
                    let metrics = value
                        .get("gemma4d_metrics")
                        .and_then(|metrics| serde_json::from_value(metrics.clone()).ok());
                    let generated_token_ids = metrics
                        .as_ref()
                        .map(|metrics: &ChatMetrics| metrics.generated_token_ids.clone())
                        .unwrap_or_default();
                    BackendRun {
                        backend: backend.as_str().to_owned(),
                        status: "passed".to_owned(),
                        http_status: Some(response.status),
                        request_wall_ms: Some(request_wall_ms),
                        response_text_sha256: sha256_hex(response_text.as_bytes()),
                        response_text,
                        generated_token_ids,
                        usage,
                        metrics,
                        prometheus,
                        runtime_snapshot,
                        error: None,
                    }
                }
                Err(error) => BackendRun::error(
                    backend,
                    Some(response.status),
                    Some(request_wall_ms),
                    format!(
                        "response JSON parse failed: {error}; body={}",
                        response.body
                    ),
                    prometheus,
                    runtime_snapshot,
                ),
            }
        }
        Ok(response) => BackendRun::error(
            backend,
            Some(response.status),
            Some(request_wall_ms),
            response.body,
            prometheus,
            runtime_snapshot,
        ),
        Err(error) => BackendRun::error(
            backend,
            None,
            Some(request_wall_ms),
            error.to_string(),
            prometheus,
            runtime_snapshot,
        ),
    }
}

fn run_prefix_warmup(
    addr: std::net::SocketAddr,
    workload_id: &str,
    prompt: &str,
    prefix_tokens: usize,
) -> PrefixWarmupRun {
    let body = json!({
        "model": ServerConfig::default().model_id,
        "messages": [{"role": "user", "content": prompt}],
        "prefix_tokens": prefix_tokens,
    });
    let body = serde_json::to_string(&body).expect("warmup request serializes");
    let started = Instant::now();
    let response = http_request(addr, "POST", "/v1/runtime/warmup/prefix", Some(&body));
    let request_wall_ms = duration_ms(started.elapsed());
    let runtime_snapshot = fetch_runtime_snapshot(addr);

    match response {
        Ok(response) if response.status == 200 => {
            match serde_json::from_str::<serde_json::Value>(&response.body) {
                Ok(value) => PrefixWarmupRun {
                    workload_id: workload_id.to_owned(),
                    backend: ServerBackend::PersistentNative.as_str().to_owned(),
                    status: "passed".to_owned(),
                    http_status: Some(response.status),
                    request_wall_ms: Some(request_wall_ms),
                    requested_prefix_tokens: value
                        .get("requested_prefix_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok()),
                    prompt_tokens: value
                        .get("prompt_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok()),
                    warmup_context_tokens: value
                        .get("warmup_context_tokens")
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok()),
                    tokenize_ms: value.get("tokenize_ms").and_then(serde_json::Value::as_f64),
                    prefill_ms: value.get("prefill_ms").and_then(serde_json::Value::as_f64),
                    decode_ms: value.get("decode_ms").and_then(serde_json::Value::as_f64),
                    total_ms: value.get("total_ms").and_then(serde_json::Value::as_f64),
                    peak_memory_gb: value
                        .get("peak_memory_gb")
                        .and_then(serde_json::Value::as_f64),
                    active_kv_bytes: value
                        .get("active_kv_bytes")
                        .and_then(serde_json::Value::as_u64),
                    runtime_snapshot,
                    error: None,
                },
                Err(error) => PrefixWarmupRun::error(
                    workload_id,
                    Some(response.status),
                    Some(request_wall_ms),
                    format!(
                        "warmup response JSON parse failed: {error}; body={}",
                        response.body
                    ),
                    runtime_snapshot,
                ),
            }
        }
        Ok(response) => PrefixWarmupRun::error(
            workload_id,
            Some(response.status),
            Some(request_wall_ms),
            response.body,
            runtime_snapshot,
        ),
        Err(error) => PrefixWarmupRun::error(
            workload_id,
            None,
            Some(request_wall_ms),
            error.to_string(),
            runtime_snapshot,
        ),
    }
}

impl PrefixWarmupRun {
    fn error(
        workload_id: &str,
        http_status: Option<u16>,
        request_wall_ms: Option<f64>,
        error: String,
        runtime_snapshot: Option<serde_json::Value>,
    ) -> Self {
        Self {
            workload_id: workload_id.to_owned(),
            backend: ServerBackend::PersistentNative.as_str().to_owned(),
            status: "blocked".to_owned(),
            http_status,
            request_wall_ms,
            requested_prefix_tokens: None,
            prompt_tokens: None,
            warmup_context_tokens: None,
            tokenize_ms: None,
            prefill_ms: None,
            decode_ms: None,
            total_ms: None,
            peak_memory_gb: None,
            active_kv_bytes: None,
            runtime_snapshot,
            error: Some(error),
        }
    }
}

impl BackendRun {
    fn error(
        backend: ServerBackend,
        http_status: Option<u16>,
        request_wall_ms: Option<f64>,
        error: String,
        prometheus: Option<PrometheusSnapshot>,
        runtime_snapshot: Option<serde_json::Value>,
    ) -> Self {
        Self {
            backend: backend.as_str().to_owned(),
            status: "blocked".to_owned(),
            http_status,
            request_wall_ms,
            response_text: String::new(),
            response_text_sha256: sha256_hex(b""),
            generated_token_ids: Vec::new(),
            usage: None,
            metrics: None,
            prometheus,
            runtime_snapshot,
            error: Some(error),
        }
    }
}

fn compare_runs(
    workload: &WorkloadRecord,
    baseline: &BackendRun,
    candidate: &BackendRun,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if baseline.status != "passed" {
        blockers.push(format!(
            "{} baseline request failed: {}",
            workload.workload_id,
            baseline.error.as_deref().unwrap_or("unknown error")
        ));
    }
    if candidate.status != "passed" {
        blockers.push(format!(
            "{} candidate request failed: {}",
            workload.workload_id,
            candidate.error.as_deref().unwrap_or("unknown error")
        ));
    }
    if baseline.generated_token_ids != candidate.generated_token_ids {
        blockers.push(format!(
            "{} generated token ids differed: baseline={:?} candidate={:?}",
            workload.workload_id, baseline.generated_token_ids, candidate.generated_token_ids
        ));
    }
    if baseline.response_text != candidate.response_text {
        blockers.push(format!(
            "{} response text differed: baseline_sha={} candidate_sha={}",
            workload.workload_id, baseline.response_text_sha256, candidate.response_text_sha256
        ));
    }
    blockers
}

fn blocked_records(
    args: &Args,
    workloads: &[WorkloadRecord],
    run_id: &str,
    blockers: &[String],
    environment: &Environment,
    model_identity: &manifest::ArtifactIdentity,
) -> Result<Vec<Xr11Record>, Box<dyn std::error::Error>> {
    let mut records = Vec::new();
    for workload in workloads {
        let prompt = fs::read_to_string(&workload.prompt_path).unwrap_or_default();
        records.push(Xr11Record {
            schema_version: 1,
            goal: GOAL.to_owned(),
            run_id: run_id.to_owned(),
            git_sha: environment.git_sha.clone(),
            git_status_short: environment.git_status_short.clone(),
            model_identity: model_identity.clone(),
            timestamp_unix: unix_now(),
            workload_id: workload.workload_id.clone(),
            family: workload.family.clone(),
            prompt_path: workload.prompt_path.clone(),
            prompt_sha256: workload.prompt_sha256.clone(),
            prompt_bytes: prompt.len(),
            deterministic_seed: workload.deterministic_seed,
            target_context_tokens: workload.target_context_tokens,
            actual_context_tokens: workload.actual_context_tokens,
            tokenizer_backend: workload.tokenizer_backend.clone(),
            repeat_index: 0,
            max_new_tokens: args.max_new_tokens,
            comparison_status: "blocked".to_owned(),
            baseline: BackendRun::error(
                args.baseline_backend,
                None,
                None,
                "pre-run blocker".to_owned(),
                None,
                None,
            ),
            candidate: BackendRun::error(
                ServerBackend::PersistentNative,
                None,
                None,
                "pre-run blocker".to_owned(),
                None,
                None,
            ),
            blockers: blockers.to_vec(),
        });
    }
    Ok(records)
}

fn load_workloads(path: &Path) -> Result<Vec<WorkloadRecord>, Box<dyn std::error::Error>> {
    let body = fs::read_to_string(path)?;
    let mut workloads = Vec::new();
    for (index, line) in body.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let workload = serde_json::from_str::<WorkloadRecord>(trimmed)
            .map_err(|error| format!("invalid workload JSON at line {}: {error}", index + 1))?;
        workloads.push(workload);
    }
    Ok(workloads)
}

fn select_workloads(
    workloads: &[WorkloadRecord],
    workload_ids: &[String],
    max_workloads: Option<usize>,
) -> Vec<WorkloadRecord> {
    let mut selected = workload_ids
        .iter()
        .filter_map(|id| {
            workloads
                .iter()
                .find(|workload| workload.workload_id == *id)
        })
        .cloned()
        .collect::<Vec<_>>();
    if let Some(max_workloads) = max_workloads {
        selected.truncate(max_workloads);
    }
    selected
}

fn fetch_prometheus(addr: std::net::SocketAddr) -> Option<PrometheusSnapshot> {
    let response = http_request(addr, "GET", "/metrics", None).ok()?;
    (response.status == 200).then(|| parse_prometheus(&response.body))
}

fn fetch_runtime_snapshot(addr: std::net::SocketAddr) -> Option<serde_json::Value> {
    let response = http_request(addr, "GET", "/v1/runtime/snapshot", None).ok()?;
    if response.status != 200 {
        return None;
    }
    serde_json::from_str(&response.body).ok()
}

fn parse_prometheus(body: &str) -> PrometheusSnapshot {
    let mut values = BTreeMap::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next().and_then(|value| value.parse::<f64>().ok()) else {
            continue;
        };
        let metric_name = name.split('{').next().unwrap_or(name);
        values.insert(metric_name.to_owned(), value);
    }
    PrometheusSnapshot {
        requests_total: values.get("gemma4d_requests_total").copied(),
        model_load_seconds: values.get("gemma4d_model_load_seconds").copied(),
        model_load_count: values.get("gemma4d_model_load_count").copied(),
        resident_model_loaded: values.get("gemma4d_resident_model_loaded").copied(),
        persistent_worker_requests_total: values
            .get("gemma4d_persistent_worker_requests_total")
            .copied(),
        prefill_tokens_total: values.get("gemma4d_prefill_tokens_total").copied(),
        decode_tokens_total: values.get("gemma4d_decode_tokens_total").copied(),
        prefill_seconds: values.get("gemma4d_prefill_seconds").copied(),
        decode_seconds: values.get("gemma4d_decode_seconds").copied(),
        tokens_per_second: values.get("gemma4d_tokens_per_second").copied(),
        memory_peak_mlx_bytes: values.get("gemma4d_memory_peak_mlx_bytes").copied(),
        memory_process_rss_bytes: values.get("gemma4d_memory_process_rss_bytes").copied(),
        prefix_warmups_total: values.get("gemma4d_prefix_warmups_total").copied(),
        prefix_warmup_tokens_total: values.get("gemma4d_prefix_warmup_tokens_total").copied(),
        prefix_warmup_seconds: values.get("gemma4d_prefix_warmup_seconds").copied(),
    }
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR11 Persistent Native Server A/B\n\n");
    out.push_str(&format!("- Status: `{}`\n", summary.status));
    out.push_str(&format!("- Decision: `{}`\n", summary.decision));
    out.push_str(&format!("- Run ID: `{}`\n", summary.run_id));
    out.push_str(&format!("- Mode: `{}`\n", summary.mode));
    out.push_str(&format!(
        "- Baseline backend: `{}`\n",
        summary.baseline_backend.as_str()
    ));
    if let Some(tokens) = summary.candidate_prefix_warmup_tokens {
        out.push_str(&format!("- Candidate prefix warmup tokens: `{tokens}`\n"));
    }
    out.push_str(&format!("- Command: `{}`\n\n", summary.command));
    out.push_str("## Model Identity\n\n");
    out.push_str("| Field | Value |\n");
    out.push_str("|---|---|\n");
    out.push_str(&format!("| Path | `{}` |\n", summary.model_identity.path));
    out.push_str(&format!(
        "| Exists | `{}` |\n",
        summary.model_identity.exists
    ));
    out.push_str(&format!(
        "| Revision source | `{}` |\n",
        summary.model_identity.revision_source
    ));
    out.push_str(&format!(
        "| Config SHA-256 | `{}` |\n",
        summary.model_identity.config_sha256
    ));
    out.push_str(&format!(
        "| Tokenizer SHA-256 | `{}` |\n",
        summary.model_identity.tokenizer_sha256
    ));
    out.push_str(&format!(
        "| Safetensors inventory SHA-256 | `{}` |\n\n",
        summary.model_identity.safetensors_inventory_sha256
    ));
    out.push_str("## Load Residency\n\n");
    out.push_str("| Role | Backend | Requests | Model load count | Model load seconds | Resident loaded | Worker requests |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|\n");
    for (role, backend, metrics) in [
        (
            "baseline",
            summary.baseline_backend.as_str(),
            summary.final_metrics.baseline.as_ref(),
        ),
        (
            "candidate_default",
            ServerBackend::PersistentNative.as_str(),
            summary.final_metrics.candidate.as_ref(),
        ),
    ] {
        out.push_str(&format!(
            "| `{role}` | `{backend}` | {} | {} | {} | {} | {} |\n",
            fmt_opt(metrics.and_then(|metrics| metrics.requests_total)),
            fmt_opt(metrics.and_then(|metrics| metrics.model_load_count)),
            fmt_opt(metrics.and_then(|metrics| metrics.model_load_seconds)),
            fmt_opt(metrics.and_then(|metrics| metrics.resident_model_loaded)),
            fmt_opt(metrics.and_then(|metrics| metrics.persistent_worker_requests_total)),
        ));
    }
    if !summary.candidate_warmups.is_empty() {
        out.push_str("\n## Candidate Prefix Warmups\n\n");
        out.push_str("| Workload | Status | Request wall ms | Prefix tokens | Prompt tokens | Warm context tokens | Tokenize ms | Prefill ms | Decode ms | Total ms | Peak MLX GB |\n");
        out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for warmup in &summary.candidate_warmups {
            out.push_str(&format!(
                "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                warmup.workload_id,
                warmup.status,
                fmt_opt(warmup.request_wall_ms),
                fmt_opt_usize(warmup.requested_prefix_tokens),
                fmt_opt_usize(warmup.prompt_tokens),
                fmt_opt_usize(warmup.warmup_context_tokens),
                fmt_opt(warmup.tokenize_ms),
                fmt_opt(warmup.prefill_ms),
                fmt_opt(warmup.decode_ms),
                fmt_opt(warmup.total_ms),
                fmt_opt(warmup.peak_memory_gb),
            ));
        }
    }
    out.push_str("\n## Comparisons\n\n");
    out.push_str("| Workload | Repeat | Status | Baseline wall ms | Candidate wall ms | Baseline load ms | Candidate load ms | Tokens |\n");
    out.push_str("|---|---:|---|---:|---:|---:|---:|---:|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | {} | {} | {} | {} | {} |\n",
            record.workload_id,
            record.repeat_index,
            record.comparison_status,
            fmt_opt(record.baseline.request_wall_ms),
            fmt_opt(record.candidate.request_wall_ms),
            fmt_opt(
                record
                    .baseline
                    .metrics
                    .as_ref()
                    .map(|metrics| metrics.model_load_ms)
            ),
            fmt_opt(
                record
                    .candidate
                    .metrics
                    .as_ref()
                    .map(|metrics| metrics.model_load_ms)
            ),
            record.candidate.generated_token_ids.len(),
        ));
    }
    out.push_str("\n## Runtime Snapshot Evidence\n\n");
    out.push_str("| Workload | Repeat | Baseline backend | Baseline policy | Candidate backend | Candidate policy | Candidate policy reason | Candidate loaded |\n");
    out.push_str("|---|---:|---|---|---|---|---|---:|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            record.workload_id,
            record.repeat_index,
            record.baseline.backend,
            runtime_policy_status(&record.baseline),
            record.candidate.backend,
            runtime_policy_status(&record.candidate),
            runtime_policy_reason(&record.candidate),
            runtime_model_loaded(&record.candidate),
        ));
    }
    out.push_str("\n## Workload Metadata\n\n");
    out.push_str("| Workload | Seed | Target tokens | Actual tokens | Prompt SHA-256 |\n");
    out.push_str("|---|---:|---:|---:|---|\n");
    for record in &summary.records {
        if record.repeat_index == 0 {
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | `{}` |\n",
                record.workload_id,
                record.deterministic_seed,
                record.target_context_tokens,
                record.actual_context_tokens,
                record.prompt_sha256,
            ));
        }
    }
    if !summary.blockers.is_empty() {
        out.push_str("\n## Blockers\n\n");
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn runtime_policy_status(run: &BackendRun) -> String {
    run.runtime_snapshot
        .as_ref()
        .and_then(|snapshot| {
            snapshot
                .pointer("/persistent_backend/native_prefill_policy/status")
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("n/a")
        .to_owned()
}

fn runtime_policy_reason(run: &BackendRun) -> String {
    run.runtime_snapshot
        .as_ref()
        .and_then(|snapshot| {
            snapshot
                .pointer("/persistent_backend/native_prefill_policy/reason")
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("n/a")
        .replace('|', "\\|")
}

fn runtime_model_loaded(run: &BackendRun) -> String {
    run.runtime_snapshot
        .as_ref()
        .and_then(|snapshot| {
            snapshot
                .pointer("/health/model_loaded")
                .and_then(serde_json::Value::as_bool)
        })
        .map(|loaded| loaded.to_string())
        .unwrap_or_else(|| "n/a".to_owned())
}

fn render_blockers(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR11 Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No blockers recorded.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn render_decision(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR11 Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str(&format!("Status: `{}`\n\n", summary.status));
    out.push_str("Evidence files:\n\n");
    for file in &summary.generated_files {
        out.push_str(&format!("- `{file}`\n"));
    }
    if !summary.blockers.is_empty() {
        out.push_str("\nBlockers:\n\n");
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn baseline_load_count_ok(
    baseline_backend: ServerBackend,
    metrics: Option<&PrometheusSnapshot>,
    records: &[Xr11Record],
) -> bool {
    let count = metrics.and_then(|metrics| metrics.model_load_count);
    match baseline_backend {
        ServerBackend::RealHelper => count.is_none_or(|count| count >= records.len() as f64),
        ServerBackend::PersistentNative => count == Some(1.0),
        ServerBackend::Stub => false,
    }
}

fn mode_for_args(args: &Args) -> &'static str {
    if args.candidate_prefix_warmup_tokens.is_some() {
        return "server_persistent_native_prefix_warmup_policy";
    }
    match args.baseline_backend {
        ServerBackend::RealHelper => MODE,
        ServerBackend::PersistentNative => {
            "server_explicit_persistent_native_vs_default_persistent_native"
        }
        ServerBackend::Stub => "invalid_stub_baseline",
    }
}

fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn command_display(args: &Args) -> String {
    let candidate_warmup = args
        .candidate_prefix_warmup_tokens
        .map(|tokens| format!(" --candidate-prefix-warmup-tokens {tokens}"))
        .unwrap_or_default();
    format!(
        "cargo run -p gemma4d-bench --example xr11_persistent_native_server_ab -- --out-dir {} --model-path {} --workloads {} --workload-ids {} --repeats {} --max-new-tokens {} --max-context-tokens {} --memory-budget-mb {} --baseline-backend {}{}",
        args.out_dir.display(),
        args.model_path.display(),
        args.workloads_path.display(),
        args.workload_ids.join(","),
        args.repeats,
        args.max_new_tokens,
        args.max_context_tokens,
        args.memory_budget_mb,
        args.baseline_backend.cli_name(),
        candidate_warmup,
    )
}

fn capture_environment() -> Environment {
    Environment {
        os: env::consts::OS.to_owned(),
        arch: env::consts::ARCH.to_owned(),
        rustc: command_stdout("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_owned()),
        git_sha: command_stdout("git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|| "unknown".to_owned()),
        git_status_short: command_stdout("git", &["status", "--short"])
            .unwrap_or_else(|| "unknown".to_owned()),
    }
}

fn capture_relevant_environment() -> BTreeMap<String, Option<String>> {
    [
        "GEMMA4D_REQUIRE_MLX",
        "GEMMA4D_USE_NATIVE_GRAPH",
        "GEMMA4D_NATIVE_PREFILL_CHUNK_TOKENS",
        "GEMMA4D_NATIVE_PREFILL_CHUNK_POLICY",
        "GEMMA4D_MLX_LM_PYTHON",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), env::var(key).ok()))
    .collect()
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn required<I>(args: &mut I, flag: &str) -> Result<String, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn required_path<I>(args: &mut I, flag: &str) -> Result<PathBuf, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    Ok(PathBuf::from(required(args, flag)?))
}

fn parse_csv(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
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

fn parse_baseline_backend(value: &str) -> Result<ServerBackend, Box<dyn std::error::Error>> {
    match value {
        "real-helper" | "real_helper" => Ok(ServerBackend::RealHelper),
        "persistent-native" | "persistent_native" => Ok(ServerBackend::PersistentNative),
        "stub" => Err("--baseline-backend stub is not valid for XR11 comparison evidence".into()),
        other => Err(format!(
            "--baseline-backend must be real-helper or persistent-native, got '{other}'"
        )
        .into()),
    }
}

fn usize_at(value: &serde_json::Value, key: &str) -> usize {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize
}

fn fmt_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn fmt_opt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_owned())
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn run_id() -> String {
    format!("xr11-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after Unix epoch")
        .as_secs()
}
