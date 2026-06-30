use std::{
    env, fs,
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use gemma4d_server::http::{
    ServerConfig, ServerRuntime, http_request, parse_bind_addr, serve_listener,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    milestone: &'static str,
    status: &'static str,
    commands: Vec<&'static str>,
    bind_addr: String,
    bind_localhost_default: bool,
    openai_chat_non_streaming: bool,
    openai_chat_streaming: bool,
    adapter_selection_routed: bool,
    models_endpoint: bool,
    adapters_endpoint: bool,
    health_endpoint: bool,
    metrics_endpoint_core_counters: bool,
    memory_guard_error: bool,
    context_guard_error: bool,
    unsafe_remote_adapter_loading_not_exposed: bool,
    tui_control_endpoints: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = parse_out_path()?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let runtime = ServerRuntime::new(ServerConfig::default().with_bind_addr(addr));
    let handle = thread::spawn(move || serve_listener(listener, runtime, server_shutdown));

    let health = http_request(addr, "GET", "/health", None)?;
    let models = http_request(addr, "GET", "/v1/models", None)?;
    let adapters = http_request(addr, "GET", "/v1/adapters", None)?;
    let non_stream = http_request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat_body(false, None)),
    )?;
    let stream = http_request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat_body(true, None)),
    )?;
    let adapter = http_request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&chat_body(false, Some("rust-coding-r16-v1"))),
    )?;
    let context = http_request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&large_context_body()),
    )?;
    let remote_load = http_request(
        addr,
        "POST",
        "/v1/adapters/load",
        Some(r#"{"adapter_id":"rust-coding-r16-v1","url":"https://example.com/a.safetensors"}"#),
    )?;
    let metrics = http_request(addr, "GET", "/metrics", None)?;
    let runtime_snapshot = http_request(addr, "GET", "/v1/runtime/snapshot", None)?;
    let cache = http_request(addr, "GET", "/v1/cache/summary", None)?;
    let benchmark = http_request(addr, "GET", "/v1/benchmarks/runs/stub-current", None)?;
    let events = http_request(addr, "GET", "/v1/runtime/events", None)?;

    let memory_guard_error = memory_guard_rejects()?;
    let metrics_endpoint_core_counters = [
        "gemma4d_requests_total",
        "gemma4d_active_generations",
        "gemma4d_errors_total",
        "gemma4d_prefill_tokens_total",
        "gemma4d_decode_tokens_total",
        "gemma4d_adapters_loaded",
    ]
    .iter()
    .all(|metric| metrics.body.contains(metric));

    let report = Report {
        schema_version: 1,
        milestone: "M11",
        status: "passed",
        commands: vec![
            "cargo test -p gemma4d-server -p gemma4d-tui --all-targets",
            "cargo run -p gemma4d-server --example m11_server_smoke -- --out benchmarks/out/M11/server-smoke.json",
            "cargo run -p gemma4d-tui -- --provider mock snapshot --out-dir benchmarks/out/M11/tui-snapshots",
        ],
        bind_addr: addr.to_string(),
        bind_localhost_default: parse_bind_addr("127.0.0.1:8080").is_ok()
            && parse_bind_addr("0.0.0.0:8080").is_err(),
        openai_chat_non_streaming: non_stream.status == 200
            && non_stream.body.contains("\"object\":\"chat.completion\"")
            && non_stream.body.contains("\"choices\""),
        openai_chat_streaming: stream.status == 200
            && stream.content_type.starts_with("text/event-stream")
            && stream.body.contains("chat.completion.chunk")
            && stream.body.contains("data: [DONE]"),
        adapter_selection_routed: adapter.status == 200
            && adapter.body.contains("stub adapter rust-coding-r16-v1"),
        models_endpoint: models.status == 200 && models.body.contains("\"object\":\"list\""),
        adapters_endpoint: adapters.status == 200 && adapters.body.contains("rust-coding-r16-v1"),
        health_endpoint: health.status == 200 && health.body.contains("\"status\":\"ok\""),
        metrics_endpoint_core_counters,
        memory_guard_error,
        context_guard_error: context.status == 400 && context.body.contains("context_too_large"),
        unsafe_remote_adapter_loading_not_exposed: remote_load.status == 400
            && remote_load.body.contains("adapter_manifest_mismatch")
            && remote_load.body.contains("not exposed"),
        tui_control_endpoints: runtime_snapshot.status == 200
            && cache.status == 200
            && benchmark.status == 200
            && events.status == 200
            && events.content_type.starts_with("text/event-stream"),
    };

    let passed = report.bind_localhost_default
        && report.openai_chat_non_streaming
        && report.openai_chat_streaming
        && report.adapter_selection_routed
        && report.models_endpoint
        && report.adapters_endpoint
        && report.health_endpoint
        && report.metrics_endpoint_core_counters
        && report.memory_guard_error
        && report.context_guard_error
        && report.unsafe_remote_adapter_loading_not_exposed
        && report.tui_control_endpoints;

    let report = Report {
        status: if passed { "passed" } else { "failed" },
        ..report
    };
    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
    println!("M11 server smoke: {}", report.status);
    println!("evidence: {}", out_path.display());

    shutdown.store(true, Ordering::SeqCst);
    let _ = TcpStream::connect(addr);
    handle
        .join()
        .map_err(|_| "server thread panicked")?
        .map_err(|error| error.to_string())?;

    if passed {
        Ok(())
    } else {
        Err("M11 server smoke failed".into())
    }
}

fn memory_guard_rejects() -> Result<bool, Box<dyn std::error::Error>> {
    let runtime = ServerRuntime::new(ServerConfig {
        memory_budget_bytes: 16,
        ..ServerConfig::default()
    });
    let response = runtime.handle_request(
        "POST",
        "/v1/chat/completions",
        chat_body(false, None).as_bytes(),
    );
    Ok(response.status == 400 && response.body.contains("memory_guard_rejected"))
}

fn chat_body(stream: bool, adapter: Option<&str>) -> String {
    let adapter = adapter
        .map(|adapter| format!(r#","adapter":"{adapter}""#))
        .unwrap_or_default();
    format!(
        r#"{{
  "model":"mlx-community/gemma-4-12B-it-4bit",
  "messages":[{{"role":"user","content":"hello from m11 smoke"}}],
  "stream":{stream},
  "temperature":0,
  "max_tokens":8{adapter}
}}"#
    )
}

fn large_context_body() -> String {
    let prompt = (0..40_000).map(|_| "token").collect::<Vec<_>>().join(" ");
    format!(
        r#"{{
  "model":"mlx-community/gemma-4-12B-it-4bit",
  "messages":[{{"role":"user","content":"{prompt}"}}],
  "stream":false,
  "temperature":0,
  "max_tokens":8
}}"#
    )
}

fn parse_out_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut out = None;
    while let Some(arg) = args.next() {
        if arg == "--out" {
            out = args.next().map(PathBuf::from);
        }
    }
    out.ok_or_else(|| "usage: m11_server_smoke --out <path>".into())
}
