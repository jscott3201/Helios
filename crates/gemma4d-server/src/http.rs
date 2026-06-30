use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub const DEFAULT_BIND_ADDR: &str = "127.0.0.1:8080";
const SERVER_NAME: &str = "gemma4d";

pub type HttpResult<T> = std::result::Result<T, HttpError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    Io(String),
    Json(String),
    BadRequest(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message) | Self::Json(message) | Self::BadRequest(message) => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for HttpError {}

impl From<std::io::Error> for HttpError {
    fn from(source: std::io::Error) -> Self {
        Self::Io(source.to_string())
    }
}

impl From<serde_json::Error> for HttpError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    pub model_id: String,
    pub max_context_tokens: usize,
    pub memory_budget_bytes: u64,
    pub queue_capacity: usize,
    pub adapters: Vec<ServerAdapter>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: DEFAULT_BIND_ADDR
                .parse()
                .expect("default bind address is valid"),
            model_id: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
            max_context_tokens: 32_768,
            memory_budget_bytes: 12 * 1024 * 1024 * 1024,
            queue_capacity: 0,
            adapters: vec![ServerAdapter::stub_loaded("rust-coding-r16-v1")],
        }
    }
}

impl ServerConfig {
    pub fn localhost_default() -> Self {
        Self::default()
    }

    pub fn with_bind_addr(mut self, bind_addr: SocketAddr) -> Self {
        self.bind_addr = bind_addr;
        self
    }

    pub fn binds_localhost_by_default() -> bool {
        Self::default().bind_addr.ip().is_loopback()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerAdapter {
    pub id: String,
    pub object: String,
    pub loaded: bool,
    pub pinned: bool,
    pub resident_bytes: u64,
    pub source: String,
    pub supports_mtp: String,
}

impl ServerAdapter {
    pub fn stub_loaded(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "adapter".to_owned(),
            loaded: true,
            pinned: true,
            resident_bytes: 2551,
            source: "trusted-local-stub".to_owned(),
            supports_mtp: "unknown".to_owned(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerMetrics {
    pub requests_total: u64,
    pub chat_completions_total: u64,
    pub streaming_chat_completions_total: u64,
    pub active_generations: u64,
    pub queue_depth: u64,
    pub errors_total: BTreeMap<String, u64>,
    pub memory_process_rss_bytes: u64,
    pub memory_guard_rejections_total: u64,
    pub prefill_tokens_total: u64,
    pub decode_tokens_total: u64,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub ttft_seconds: f64,
    pub tokens_per_second: f64,
    pub adapters_loaded: u64,
    pub adapter_resident_bytes: u64,
    pub adapter_requests_total: BTreeMap<String, u64>,
}

#[derive(Clone)]
pub struct ServerRuntime {
    config: ServerConfig,
    metrics: Arc<Mutex<ServerMetrics>>,
    adapters: Arc<Mutex<Vec<ServerAdapter>>>,
    active_generation: Arc<AtomicBool>,
}

impl ServerRuntime {
    pub fn new(config: ServerConfig) -> Self {
        let adapters = config.adapters.clone();
        let runtime = Self {
            config,
            metrics: Arc::new(Mutex::new(ServerMetrics::default())),
            adapters: Arc::new(Mutex::new(adapters)),
            active_generation: Arc::new(AtomicBool::new(false)),
        };
        runtime.refresh_adapter_metrics();
        runtime
    }

    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    pub fn metrics_snapshot(&self) -> ServerMetrics {
        self.metrics.lock().expect("metrics lock").clone()
    }

    pub fn handle_request(&self, method: &str, path: &str, body: &[u8]) -> HttpResponse {
        self.increment_requests();
        match (method, path) {
            ("GET", "/health") => self.json_ok(json!({
                "status": "ok",
                "server": SERVER_NAME,
                "model_loaded": true,
                "bind": self.config.bind_addr.to_string(),
                "localhost_only": self.config.bind_addr.ip().is_loopback(),
            })),
            ("GET", "/v1/models") => self.json_ok(json!({
                "object": "list",
                "data": [{
                    "id": self.config.model_id,
                    "object": "model",
                    "owned_by": "local",
                }]
            })),
            ("GET", "/v1/adapters") => self.adapters_response(),
            ("POST", "/v1/adapters/load") => self.adapter_mutation_response(body, true),
            ("POST", "/v1/adapters/unload") => self.adapter_mutation_response(body, false),
            ("POST", "/v1/chat/completions") => self.chat_completions_response(body),
            ("GET", "/metrics") => {
                HttpResponse::ok("text/plain; version=0.0.4", self.metrics_text())
            }
            ("GET", "/v1/runtime/snapshot") => self.runtime_snapshot_response(),
            ("GET", "/v1/runtime/events") => self.runtime_events_response(),
            ("GET", "/v1/config") => self.json_ok(json!({
                "status": "stub",
                "bind": self.config.bind_addr.to_string(),
                "model": self.config.model_id,
                "max_context_tokens": self.config.max_context_tokens,
                "localhost_only": self.config.bind_addr.ip().is_loopback(),
            })),
            ("POST", "/v1/config/validate") => self.json_ok(json!({
                "status": "valid",
                "summary": "stub server config accepted",
                "diagnostics": [],
            })),
            ("POST", "/v1/config/apply") => self.json_ok(json!({
                "status": "read_only_stub",
                "applied": false,
            })),
            ("GET", "/v1/cache/summary") => self.cache_summary_response(),
            ("POST", "/v1/cache/evict") => self.json_ok(json!({
                "status": "read_only_stub",
                "evicted": 0,
            })),
            ("POST", "/v1/benchmarks/run") => self.json_ok(json!({
                "id": "stub-current",
                "status": "ready",
                "started": false,
                "report_path": "benchmarks/out/M11/stub-report.md",
            })),
            _ if method == "GET" && path.starts_with("/v1/benchmarks/runs/") => {
                let id = path.trim_start_matches("/v1/benchmarks/runs/");
                self.json_ok(json!({
                    "id": id,
                    "status": "ready",
                    "report_path": "benchmarks/out/M11/stub-report.md",
                    "note": "stub backend has not spawned benchmark work",
                }))
            }
            _ => self.error_response(404, "not_found", format!("no route for {method} {path}")),
        }
    }

    fn chat_completions_response(&self, body: &[u8]) -> HttpResponse {
        let request = match serde_json::from_slice::<ChatCompletionRequest>(body) {
            Ok(request) => request,
            Err(source) => {
                return self.error_response(
                    400,
                    "unsupported_model_config",
                    format!("invalid chat completion JSON: {source}"),
                );
            }
        };
        let admitted = match self.admit_chat_request(&request) {
            Ok(admitted) => admitted,
            Err(error) => return self.error_response(error.status, error.code, error.message),
        };
        let _guard = ActiveGenerationGuard::enter(self);
        let response_text = stub_chat_response(&request, admitted.adapter_id.as_deref());
        let completion_id = format!("chatcmpl-gemma4d-stub-{}", now_unix_seconds());
        let created = now_unix_seconds();
        self.record_generation(&request, &response_text, admitted.adapter_id.as_deref());

        if request.stream.unwrap_or(false) {
            self.increment_streaming();
            HttpResponse::ok(
                "text/event-stream",
                streaming_chat_body(&completion_id, created, &request.model, &response_text),
            )
        } else {
            self.increment_chat();
            let prompt_tokens = estimate_prompt_tokens(&request.messages);
            let completion_tokens = estimate_text_tokens(&response_text);
            self.json_ok(json!({
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "system_fingerprint": admitted.adapter_id.unwrap_or_else(|| "base".to_owned()),
            }))
        }
    }

    fn admit_chat_request(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<AdmittedRequest, ApiError> {
        if request.model != self.config.model_id {
            return Err(ApiError::new(
                400,
                "unsupported_model_config",
                format!("unsupported model {}", request.model),
            ));
        }
        if request.temperature.unwrap_or(0.0) != 0.0 {
            return Err(ApiError::new(
                400,
                "unsupported_model_config",
                "temperature must be 0 for the M11 stub backend",
            ));
        }
        let max_tokens = request.max_tokens.unwrap_or(16);
        let prompt_tokens = estimate_prompt_tokens(&request.messages);
        let total_context = prompt_tokens.saturating_add(max_tokens);
        if total_context > self.config.max_context_tokens {
            return Err(ApiError::new(
                400,
                "context_too_large",
                format!(
                    "request uses {total_context} estimated tokens, max is {}",
                    self.config.max_context_tokens
                ),
            ));
        }
        let estimated_bytes = (total_context as u64).saturating_mul(4096);
        if estimated_bytes > self.config.memory_budget_bytes {
            self.record_error("memory_guard_rejected");
            self.metrics
                .lock()
                .expect("metrics lock")
                .memory_guard_rejections_total += 1;
            return Err(ApiError::new(
                400,
                "memory_guard_rejected",
                format!(
                    "request predicts {estimated_bytes} bytes, budget is {}",
                    self.config.memory_budget_bytes
                ),
            ));
        }
        if self.active_generation.load(Ordering::SeqCst) && self.config.queue_capacity == 0 {
            return Err(ApiError::new(
                429,
                "native_backend_error",
                "single active generation already running and queue is full",
            ));
        }
        let adapter_id = request
            .adapter
            .as_deref()
            .filter(|adapter| !adapter.is_empty() && *adapter != "none")
            .map(str::to_owned);
        if let Some(adapter_id) = adapter_id.as_deref() {
            let adapters = self.adapters.lock().expect("adapters lock");
            let Some(adapter) = adapters.iter().find(|adapter| adapter.id == adapter_id) else {
                return Err(ApiError::new(
                    400,
                    "adapter_not_loaded",
                    format!("adapter is unavailable: {adapter_id}"),
                ));
            };
            if !adapter.loaded {
                return Err(ApiError::new(
                    400,
                    "adapter_not_loaded",
                    format!("adapter is not loaded: {adapter_id}"),
                ));
            }
        }
        Ok(AdmittedRequest { adapter_id })
    }

    fn adapters_response(&self) -> HttpResponse {
        let adapters = self.adapters.lock().expect("adapters lock").clone();
        self.json_ok(json!({
            "object": "list",
            "data": adapters,
        }))
    }

    fn adapter_mutation_response(&self, body: &[u8], loaded: bool) -> HttpResponse {
        let value = match serde_json::from_slice::<serde_json::Value>(body) {
            Ok(value) => value,
            Err(source) => {
                return self.error_response(
                    400,
                    "adapter_manifest_mismatch",
                    format!("invalid adapter request JSON: {source}"),
                );
            }
        };
        if value.get("source").is_some()
            || value.get("path").is_some()
            || value.get("url").is_some()
        {
            return self.error_response(
                400,
                "adapter_manifest_mismatch",
                "remote or caller-supplied adapter paths are not exposed by the M11 API",
            );
        }
        let Some(adapter_id) = value.get("adapter_id").and_then(serde_json::Value::as_str) else {
            return self.error_response(400, "adapter_not_loaded", "adapter_id is required");
        };
        let mut adapters = self.adapters.lock().expect("adapters lock");
        let Some(adapter) = adapters.iter_mut().find(|adapter| adapter.id == adapter_id) else {
            return self.error_response(
                400,
                "adapter_not_loaded",
                format!("adapter is unavailable: {adapter_id}"),
            );
        };
        adapter.loaded = loaded;
        let adapter = adapter.clone();
        drop(adapters);
        self.refresh_adapter_metrics();
        self.json_ok(json!({
            "object": "adapter",
            "data": adapter,
        }))
    }

    fn runtime_snapshot_response(&self) -> HttpResponse {
        let metrics = self.metrics_snapshot();
        let adapters = self.adapters.lock().expect("adapters lock").clone();
        self.json_ok(json!({
            "health": {
                "status": "ok",
                "model_loaded": true,
                "localhost_only": self.config.bind_addr.ip().is_loopback(),
            },
            "metrics": metrics,
            "adapters": adapters,
            "cache": cache_summary_json(),
            "benchmark": {
                "id": "stub-current",
                "status": "ready",
                "report_path": "benchmarks/out/M11/stub-report.md",
            },
            "chat": {
                "status": "idle",
                "streaming": true,
                "last_stream": "stub-ready",
            },
        }))
    }

    fn runtime_events_response(&self) -> HttpResponse {
        let event = json!({
            "type": "runtime.snapshot",
            "health": "ok",
            "streaming_chat_status": "idle",
        });
        HttpResponse::ok(
            "text/event-stream",
            format!("data: {}\n\ndata: [DONE]\n\n", compact_json(&event)),
        )
    }

    fn cache_summary_response(&self) -> HttpResponse {
        self.json_ok(cache_summary_json())
    }

    fn metrics_text(&self) -> String {
        let metrics = self.metrics_snapshot();
        let mut lines = Vec::new();
        lines.push(format!("gemma4d_requests_total {}", metrics.requests_total));
        lines.push(format!(
            "gemma4d_active_generations {}",
            metrics.active_generations
        ));
        lines.push(format!("gemma4d_queue_depth {}", metrics.queue_depth));
        for (code, count) in &metrics.errors_total {
            lines.push(format!("gemma4d_errors_total{{code=\"{code}\"}} {count}"));
        }
        if metrics.errors_total.is_empty() {
            lines.push("gemma4d_errors_total{code=\"none\"} 0".to_owned());
        }
        lines.push(format!(
            "gemma4d_memory_process_rss_bytes {}",
            metrics.memory_process_rss_bytes
        ));
        lines.push(format!(
            "gemma4d_memory_guard_rejections_total {}",
            metrics.memory_guard_rejections_total
        ));
        lines.push(format!(
            "gemma4d_prefill_tokens_total {}",
            metrics.prefill_tokens_total
        ));
        lines.push(format!(
            "gemma4d_decode_tokens_total {}",
            metrics.decode_tokens_total
        ));
        lines.push(format!(
            "gemma4d_prefill_seconds {:.6}",
            metrics.prefill_seconds
        ));
        lines.push(format!(
            "gemma4d_decode_seconds {:.6}",
            metrics.decode_seconds
        ));
        lines.push(format!("gemma4d_ttft_seconds {:.6}", metrics.ttft_seconds));
        lines.push(format!(
            "gemma4d_tokens_per_second {:.6}",
            metrics.tokens_per_second
        ));
        lines.push("gemma4d_mtp_attempted_tokens_total 0".to_owned());
        lines.push("gemma4d_mtp_accepted_tokens_total 0".to_owned());
        lines.push("gemma4d_mtp_acceptance_rate 0".to_owned());
        lines.push("gemma4d_mtp_rollbacks_total 0".to_owned());
        lines.push("gemma4d_mtp_auto_disabled_total 0".to_owned());
        lines.push("gemma4d_kv_active_bytes 0".to_owned());
        lines.push("gemma4d_prefix_cache_hits_total{tier=\"ram\"} 0".to_owned());
        lines.push("gemma4d_prefix_cache_hits_total{tier=\"ssd\"} 0".to_owned());
        lines.push("gemma4d_prefix_cache_misses_total 0".to_owned());
        lines.push("gemma4d_ssd_cache_read_bytes_total 0".to_owned());
        lines.push("gemma4d_ssd_cache_write_bytes_total 0".to_owned());
        lines.push("gemma4d_cache_restore_failures_total 0".to_owned());
        lines.push(format!(
            "gemma4d_adapters_loaded {}",
            metrics.adapters_loaded
        ));
        lines.push("gemma4d_adapter_load_seconds 0".to_owned());
        lines.push(format!(
            "gemma4d_adapter_resident_bytes {}",
            metrics.adapter_resident_bytes
        ));
        lines.push("gemma4d_adapter_evictions_total 0".to_owned());
        for (adapter_id, count) in &metrics.adapter_requests_total {
            lines.push(format!(
                "gemma4d_adapter_requests_total{{adapter_id=\"{adapter_id}\"}} {count}"
            ));
        }
        lines.push(String::new());
        lines.join("\n")
    }

    fn json_ok(&self, value: serde_json::Value) -> HttpResponse {
        HttpResponse::ok("application/json", compact_json(&value))
    }

    fn error_response(
        &self,
        status: u16,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> HttpResponse {
        let code = code.into();
        self.record_error(&code);
        let body = json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "code": code,
            }
        });
        HttpResponse::new(status, "application/json", compact_json(&body))
    }

    fn increment_requests(&self) {
        self.metrics.lock().expect("metrics lock").requests_total += 1;
    }

    fn increment_chat(&self) {
        self.metrics
            .lock()
            .expect("metrics lock")
            .chat_completions_total += 1;
    }

    fn increment_streaming(&self) {
        self.metrics
            .lock()
            .expect("metrics lock")
            .streaming_chat_completions_total += 1;
    }

    fn record_generation(
        &self,
        request: &ChatCompletionRequest,
        response_text: &str,
        adapter_id: Option<&str>,
    ) {
        let mut metrics = self.metrics.lock().expect("metrics lock");
        let prompt_tokens = estimate_prompt_tokens(&request.messages) as u64;
        let completion_tokens = estimate_text_tokens(response_text) as u64;
        metrics.prefill_tokens_total = metrics.prefill_tokens_total.saturating_add(prompt_tokens);
        metrics.decode_tokens_total = metrics
            .decode_tokens_total
            .saturating_add(completion_tokens);
        metrics.prefill_seconds += prompt_tokens as f64 / 100_000.0;
        metrics.decode_seconds += completion_tokens as f64 / 100_000.0;
        metrics.ttft_seconds += 0.001;
        metrics.tokens_per_second = 1000.0;
        if let Some(adapter_id) = adapter_id {
            *metrics
                .adapter_requests_total
                .entry(adapter_id.to_owned())
                .or_insert(0) += 1;
        }
    }

    fn record_error(&self, code: &str) {
        let mut metrics = self.metrics.lock().expect("metrics lock");
        *metrics.errors_total.entry(code.to_owned()).or_insert(0) += 1;
    }

    fn refresh_adapter_metrics(&self) {
        let adapters = self.adapters.lock().expect("adapters lock");
        let mut metrics = self.metrics.lock().expect("metrics lock");
        metrics.adapters_loaded = adapters.iter().filter(|adapter| adapter.loaded).count() as u64;
        metrics.adapter_resident_bytes = adapters
            .iter()
            .filter(|adapter| adapter.loaded)
            .map(|adapter| adapter.resident_bytes)
            .sum();
    }
}

struct ActiveGenerationGuard<'a> {
    runtime: &'a ServerRuntime,
}

impl<'a> ActiveGenerationGuard<'a> {
    fn enter(runtime: &'a ServerRuntime) -> Self {
        runtime.active_generation.store(true, Ordering::SeqCst);
        runtime
            .metrics
            .lock()
            .expect("metrics lock")
            .active_generations = 1;
        Self { runtime }
    }
}

impl Drop for ActiveGenerationGuard<'_> {
    fn drop(&mut self) {
        self.runtime
            .active_generation
            .store(false, Ordering::SeqCst);
        self.runtime
            .metrics
            .lock()
            .expect("metrics lock")
            .active_generations = 0;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdmittedRequest {
    adapter_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ApiError {
    status: u16,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub adapter: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub content_type: String,
    pub body: String,
}

impl HttpResponse {
    pub fn ok(content_type: impl Into<String>, body: impl Into<String>) -> Self {
        Self::new(200, content_type, body)
    }

    pub fn new(status: u16, content_type: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: content_type.into(),
            body: body.into(),
        }
    }

    fn status_text(&self) -> &'static str {
        match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            429 => "Too Many Requests",
            _ => "Internal Server Error",
        }
    }
}

pub fn serve_blocking(config: ServerConfig) -> HttpResult<()> {
    let listener = TcpListener::bind(config.bind_addr)?;
    let runtime = ServerRuntime::new(config);
    let shutdown = Arc::new(AtomicBool::new(false));
    serve_listener(listener, runtime, shutdown)
}

pub fn serve_listener(
    listener: TcpListener,
    runtime: ServerRuntime,
    shutdown: Arc<AtomicBool>,
) -> HttpResult<()> {
    listener.set_nonblocking(true)?;
    while !shutdown.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let runtime = runtime.clone();
                thread::spawn(move || {
                    let _ = handle_connection(stream, runtime);
                });
            }
            Err(source) if source.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(source) => return Err(HttpError::Io(source.to_string())),
        }
    }
    Ok(())
}

pub fn parse_bind_addr(value: &str) -> Result<SocketAddr, String> {
    let parsed = value
        .parse::<SocketAddr>()
        .map_err(|error| format!("invalid bind address {value}: {error}"))?;
    if !parsed.ip().is_loopback() {
        return Err(format!(
            "M11 binds localhost by default; non-local bind {parsed} requires a future security review"
        ));
    }
    Ok(parsed)
}

fn handle_connection(mut stream: TcpStream, runtime: ServerRuntime) -> HttpResult<()> {
    let request = read_http_request(&mut stream)?;
    let response = runtime.handle_request(&request.method, &request.path, &request.body);
    write_http_response(&mut stream, &response)?;
    Ok(())
}

#[derive(Debug)]
struct ParsedHttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_http_request(stream: &mut TcpStream) -> HttpResult<ParsedHttpRequest> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut buffer = Vec::new();
    let mut temp = [0_u8; 1024];
    let header_end;
    loop {
        let read = stream.read(&mut temp)?;
        if read == 0 {
            return Err(HttpError::BadRequest(
                "connection closed before headers".to_owned(),
            ));
        }
        buffer.extend_from_slice(&temp[..read]);
        if let Some(index) = find_header_end(&buffer) {
            header_end = index;
            break;
        }
        if buffer.len() > 64 * 1024 {
            return Err(HttpError::BadRequest(
                "request headers too large".to_owned(),
            ));
        }
    }

    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let mut lines = headers.lines();
    let Some(request_line) = lines.next() else {
        return Err(HttpError::BadRequest("missing request line".to_owned()));
    };
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| HttpError::BadRequest("missing method".to_owned()))?
        .to_owned();
    let raw_path = parts
        .next()
        .ok_or_else(|| HttpError::BadRequest("missing path".to_owned()))?;
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_owned();
    let mut content_length = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse::<usize>().map_err(|error| {
                HttpError::BadRequest(format!("invalid content-length header: {error}"))
            })?;
        }
    }
    let body_start = header_end + 4;
    while buffer.len().saturating_sub(body_start) < content_length {
        let read = stream.read(&mut temp)?;
        if read == 0 {
            return Err(HttpError::BadRequest(
                "connection closed before body".to_owned(),
            ));
        }
        buffer.extend_from_slice(&temp[..read]);
    }
    let body = buffer[body_start..body_start + content_length].to_vec();
    Ok(ParsedHttpRequest { method, path, body })
}

fn write_http_response(stream: &mut TcpStream, response: &HttpResponse) -> HttpResult<()> {
    let bytes = response.body.as_bytes();
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.status_text(),
        response.content_type,
        bytes.len()
    )?;
    stream.write_all(bytes)?;
    Ok(())
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn stub_chat_response(request: &ChatCompletionRequest, adapter_id: Option<&str>) -> String {
    let last_user = request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .map(|message| message.content.as_str())
        .unwrap_or("empty prompt");
    match adapter_id {
        Some(adapter_id) => format!("stub adapter {adapter_id}: {last_user}"),
        None => format!("stub response: {last_user}"),
    }
}

fn streaming_chat_body(id: &str, created: u64, model: &str, response_text: &str) -> String {
    let role = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": null,
        }]
    });
    let content = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": response_text},
            "finish_reason": null,
        }]
    });
    let done = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }]
    });
    format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
        compact_json(&role),
        compact_json(&content),
        compact_json(&done)
    )
}

fn cache_summary_json() -> serde_json::Value {
    json!({
        "status": "stub",
        "cache_mode": "bf16",
        "namespace_hash": null,
        "active_kv_bytes": 0,
        "ram": {
            "resident_bytes": 0,
            "resident_blocks": 0,
            "hits": 0,
            "misses": 0,
            "restore_failures": 0,
        },
        "ssd": {
            "stored_bytes": 0,
            "stored_blocks": 0,
            "reads": 0,
            "writes": 0,
            "restore_failures": 0,
            "namespace_rejections": 0,
        }
    })
}

fn estimate_prompt_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|message| estimate_text_tokens(&message.content).saturating_add(1))
        .sum::<usize>()
        .max(1)
}

fn estimate_text_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(1)
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).expect("JSON value serializes")
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after Unix epoch")
        .as_secs()
}

pub fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> HttpResult<HttpResponse> {
    let mut stream = TcpStream::connect(addr)?;
    let body = body.unwrap_or("");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw)?;
    parse_raw_http_response(&raw)
}

fn parse_raw_http_response(raw: &str) -> HttpResult<HttpResponse> {
    let Some((headers, body)) = raw.split_once("\r\n\r\n") else {
        return Err(HttpError::BadRequest("invalid HTTP response".to_owned()));
    };
    let status_line = headers.lines().next().unwrap_or_default();
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| HttpError::BadRequest("missing response status".to_owned()))?
        .parse::<u16>()
        .map_err(|error| HttpError::BadRequest(format!("invalid response status: {error}")))?;
    let content_type = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-type")
                    .then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_default();
    Ok(HttpResponse {
        status,
        content_type,
        body: body.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> ServerRuntime {
        ServerRuntime::new(ServerConfig::default())
    }

    fn chat_body(stream: bool, adapter: Option<&str>) -> String {
        let adapter = adapter
            .map(|adapter| format!(r#","adapter":"{adapter}""#))
            .unwrap_or_default();
        format!(
            r#"{{
  "model":"mlx-community/gemma-4-12B-it-4bit",
  "messages":[{{"role":"user","content":"hello from m11"}}],
  "stream":{stream},
  "temperature":0,
  "max_tokens":8{adapter}
}}"#
        )
    }

    #[test]
    fn default_bind_is_localhost() {
        assert!(ServerConfig::binds_localhost_by_default());
        assert_eq!(
            ServerConfig::default().bind_addr.to_string(),
            DEFAULT_BIND_ADDR
        );
        assert!(parse_bind_addr("127.0.0.1:8081").is_ok());
        assert!(parse_bind_addr("0.0.0.0:8081").is_err());
    }

    #[test]
    fn chat_completion_non_streaming_matches_openai_shape() {
        let response = runtime().handle_request(
            "POST",
            "/v1/chat/completions",
            chat_body(false, None).as_bytes(),
        );
        assert_eq!(response.status, 200);
        let value: serde_json::Value = serde_json::from_str(&response.body).expect("json");
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert!(
            value["choices"][0]["message"]["content"]
                .as_str()
                .expect("content")
                .contains("hello from m11")
        );
        assert!(value["usage"]["total_tokens"].as_u64().expect("usage") > 0);
    }

    #[test]
    fn streaming_chat_completion_uses_sse_done() {
        let response = runtime().handle_request(
            "POST",
            "/v1/chat/completions",
            chat_body(true, None).as_bytes(),
        );
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "text/event-stream");
        assert!(response.body.contains("chat.completion.chunk"));
        assert!(response.body.contains("data: [DONE]"));
    }

    #[test]
    fn adapter_selection_field_routes_or_rejects() {
        let routed = runtime().handle_request(
            "POST",
            "/v1/chat/completions",
            chat_body(false, Some("rust-coding-r16-v1")).as_bytes(),
        );
        assert_eq!(routed.status, 200);
        assert!(routed.body.contains("stub adapter rust-coding-r16-v1"));

        let rejected = runtime().handle_request(
            "POST",
            "/v1/chat/completions",
            chat_body(false, Some("missing-adapter")).as_bytes(),
        );
        assert_eq!(rejected.status, 400);
        assert!(rejected.body.contains("adapter_not_loaded"));
    }

    #[test]
    fn metrics_endpoint_exposes_core_counters() {
        let runtime = runtime();
        let _ = runtime.handle_request(
            "POST",
            "/v1/chat/completions",
            chat_body(false, Some("rust-coding-r16-v1")).as_bytes(),
        );
        let response = runtime.handle_request("GET", "/metrics", b"");
        assert_eq!(response.status, 200);
        for metric in [
            "gemma4d_requests_total",
            "gemma4d_active_generations",
            "gemma4d_errors_total",
            "gemma4d_prefill_tokens_total",
            "gemma4d_decode_tokens_total",
            "gemma4d_adapters_loaded",
            "gemma4d_adapter_requests_total{adapter_id=\"rust-coding-r16-v1\"}",
        ] {
            assert!(response.body.contains(metric), "missing metric {metric}");
        }
    }

    #[test]
    fn admission_and_memory_guard_return_stable_error_codes() {
        let config = ServerConfig {
            max_context_tokens: 4,
            memory_budget_bytes: 16,
            ..ServerConfig::default()
        };
        let runtime = ServerRuntime::new(config);
        let context = runtime.handle_request(
            "POST",
            "/v1/chat/completions",
            chat_body(false, None).as_bytes(),
        );
        assert_eq!(context.status, 400);
        assert!(context.body.contains("context_too_large"));

        let config = ServerConfig {
            max_context_tokens: 128,
            memory_budget_bytes: 16,
            ..ServerConfig::default()
        };
        let runtime = ServerRuntime::new(config);
        let memory = runtime.handle_request(
            "POST",
            "/v1/chat/completions",
            chat_body(false, None).as_bytes(),
        );
        assert_eq!(memory.status, 400);
        assert!(memory.body.contains("memory_guard_rejected"));
    }

    #[test]
    fn unsafe_remote_adapter_loading_is_not_exposed() {
        let body =
            br#"{"adapter_id":"rust-coding-r16-v1","url":"https://example.com/a.safetensors"}"#;
        let response = runtime().handle_request("POST", "/v1/adapters/load", body);
        assert_eq!(response.status, 400);
        assert!(response.body.contains("adapter_manifest_mismatch"));
        assert!(response.body.contains("not exposed"));
    }

    #[test]
    fn control_endpoints_return_stable_stub_shapes() {
        let runtime = runtime();
        for path in [
            "/health",
            "/v1/models",
            "/v1/adapters",
            "/v1/runtime/snapshot",
            "/v1/cache/summary",
            "/v1/benchmarks/runs/stub-current",
            "/v1/config",
        ] {
            let response = runtime.handle_request("GET", path, b"");
            assert_eq!(response.status, 200, "{path}");
            let _: serde_json::Value = serde_json::from_str(&response.body).expect(path);
        }
        let events = runtime.handle_request("GET", "/v1/runtime/events", b"");
        assert_eq!(events.status, 200);
        assert!(events.body.contains("runtime.snapshot"));
    }

    #[test]
    fn served_http_listener_smoke() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let shutdown = Arc::new(AtomicBool::new(false));
        let server_shutdown = Arc::clone(&shutdown);
        let runtime = runtime();
        let handle = thread::spawn(move || {
            serve_listener(listener, runtime, server_shutdown).expect("serve")
        });

        let health = http_request(addr, "GET", "/health", None).expect("health");
        assert_eq!(health.status, 200);
        assert!(health.body.contains("\"status\":\"ok\""));

        let stream = http_request(
            addr,
            "POST",
            "/v1/chat/completions",
            Some(&chat_body(true, None)),
        )
        .expect("stream");
        assert_eq!(stream.status, 200);
        assert!(stream.body.contains("data: [DONE]"));

        shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(addr);
        handle.join().expect("join");
    }
}
