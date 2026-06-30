use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    app::{
        AdapterEntrySnapshot, AdapterSnapshot, BackendEvent, BenchmarkRecord, BenchmarkStatus,
        CacheSnapshot, ChatSnapshot, DashboardSnapshot, LogEntry, MtpSnapshot, ProviderKind,
    },
    config::{ConfigValidation, validate_config_path},
};

pub trait RuntimeProvider {
    fn name(&self) -> String;
    fn dashboard_snapshot(&mut self) -> DashboardSnapshot;
    fn cache_snapshot(&mut self) -> CacheSnapshot;
    fn adapter_snapshot(&mut self) -> AdapterSnapshot;
    fn chat_snapshot(&mut self) -> ChatSnapshot;
    fn mtp_snapshot(&mut self) -> MtpSnapshot;
    fn backend_events(&mut self) -> Vec<BackendEvent>;
    fn validate_config(&mut self, path: &Path) -> ConfigValidation;
    fn start_benchmark(&mut self, out_dir: &Path) -> BenchmarkRecord;
}

pub fn create_provider(kind: ProviderKind, root: PathBuf) -> Box<dyn RuntimeProvider> {
    create_provider_with_server(kind, root, "http://127.0.0.1:8080".to_owned())
}

pub fn create_provider_with_server(
    kind: ProviderKind,
    root: PathBuf,
    server_url: String,
) -> Box<dyn RuntimeProvider> {
    match kind {
        ProviderKind::Mock => Box::new(MockProvider::default()),
        ProviderKind::File => Box::new(FileProvider::new(root)),
        ProviderKind::Http => Box::new(HttpProvider::new(server_url)),
    }
}

#[derive(Debug, Default)]
pub struct MockProvider {
    tick: u64,
}

impl RuntimeProvider for MockProvider {
    fn name(&self) -> String {
        "mock".to_owned()
    }

    fn dashboard_snapshot(&mut self) -> DashboardSnapshot {
        self.tick = self.tick.saturating_add(1);
        DashboardSnapshot {
            runtime_state: "mock-ready".to_owned(),
            provider: "mock".to_owned(),
            model_target: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
            context_window: "32768 tokens".to_owned(),
            memory_pressure: "42% mock pressure".to_owned(),
            active_task: format!("idle tick {}", self.tick),
            ttft_p50_ms: Some(118.4),
            decode_tps_p50: Some(31.7),
            cache_hit_rate: Some(0.0),
        }
    }

    fn backend_events(&mut self) -> Vec<BackendEvent> {
        vec![
            BackendEvent::Log(LogEntry::info(
                "mock provider attached; runtime mutation disabled for M05",
            )),
            BackendEvent::Log(LogEntry::info(
                "dashboard metrics are deterministic mock values",
            )),
            BackendEvent::Cache(self.cache_snapshot()),
            BackendEvent::Adapters(self.adapter_snapshot()),
            BackendEvent::Chat(self.chat_snapshot()),
            BackendEvent::Mtp(self.mtp_snapshot()),
        ]
    }

    fn cache_snapshot(&mut self) -> CacheSnapshot {
        CacheSnapshot::mock_m09()
    }

    fn adapter_snapshot(&mut self) -> AdapterSnapshot {
        AdapterSnapshot::mock_m10()
    }

    fn chat_snapshot(&mut self) -> ChatSnapshot {
        ChatSnapshot::mock_m11()
    }

    fn mtp_snapshot(&mut self) -> MtpSnapshot {
        MtpSnapshot::mock_m06()
    }

    fn validate_config(&mut self, path: &Path) -> ConfigValidation {
        validate_config_path(path)
    }

    fn start_benchmark(&mut self, out_dir: &Path) -> BenchmarkRecord {
        let mut record = BenchmarkRecord::ready(out_dir.to_path_buf());
        record.status = BenchmarkStatus::Completed;
        record.note = "mock benchmark recorded exact command, output directory, and report path; no process was spawned".to_owned();
        record
    }
}

#[derive(Debug, Clone)]
pub struct FileProvider {
    root: PathBuf,
    emitted_bootstrap_logs: bool,
}

impl FileProvider {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            emitted_bootstrap_logs: false,
        }
    }
}

impl RuntimeProvider for FileProvider {
    fn name(&self) -> String {
        "file".to_owned()
    }

    fn dashboard_snapshot(&mut self) -> DashboardSnapshot {
        let m04_report = self.root.join("docs/evidence/M04-benchmark-report.md");
        let benchmark_state = if m04_report.exists() {
            "last benchmark report found"
        } else {
            "no benchmark report found"
        };
        DashboardSnapshot {
            runtime_state: "file-offline".to_owned(),
            provider: "file".to_owned(),
            model_target: "from repository config files".to_owned(),
            context_window: "from tiny16-style config".to_owned(),
            memory_pressure: "not sampled; no daemon connected".to_owned(),
            active_task: benchmark_state.to_owned(),
            ttft_p50_ms: None,
            decode_tps_p50: None,
            cache_hit_rate: None,
        }
    }

    fn cache_snapshot(&mut self) -> CacheSnapshot {
        CacheSnapshot::disabled("file provider has no M07 cache accounting report loaded")
    }

    fn adapter_snapshot(&mut self) -> AdapterSnapshot {
        AdapterSnapshot::disabled("file provider has no M10 adapter registry report loaded")
    }

    fn chat_snapshot(&mut self) -> ChatSnapshot {
        ChatSnapshot::disabled("file provider has no M11 server stream attached")
    }

    fn backend_events(&mut self) -> Vec<BackendEvent> {
        if self.emitted_bootstrap_logs {
            return Vec::new();
        }
        self.emitted_bootstrap_logs = true;
        vec![
            BackendEvent::Log(LogEntry::info(
                "file provider attached; reading repository files only",
            )),
            BackendEvent::Log(LogEntry::warn(
                "runtime daemon absent; live Chat and mutation controls remain disabled",
            )),
        ]
    }

    fn mtp_snapshot(&mut self) -> MtpSnapshot {
        MtpSnapshot::disabled("file provider has no M06 MTP metrics report loaded")
    }

    fn validate_config(&mut self, path: &Path) -> ConfigValidation {
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root.join(path)
        };
        validate_config_path(&resolved)
    }

    fn start_benchmark(&mut self, out_dir: &Path) -> BenchmarkRecord {
        let resolved = if out_dir.is_absolute() {
            out_dir.to_path_buf()
        } else {
            self.root.join(out_dir)
        };
        let mut record = BenchmarkRecord::ready(resolved);
        record.status = BenchmarkStatus::Ready;
        record.note = "file provider prepared command only; M05 does not spawn benchmark processes"
            .to_owned();
        record
    }
}

#[derive(Debug, Clone)]
pub struct HttpProvider {
    client: SimpleHttpClient,
    emitted_bootstrap_logs: bool,
    last_error: Option<String>,
    last_chat: ChatSnapshot,
}

impl HttpProvider {
    pub fn new(server_url: String) -> Self {
        Self {
            client: SimpleHttpClient::new(server_url.clone()),
            emitted_bootstrap_logs: false,
            last_error: None,
            last_chat: ChatSnapshot::disabled(format!(
                "HTTP provider has not streamed chat from {server_url}"
            )),
        }
    }

    pub fn server_url(&self) -> &str {
        self.client.base_url()
    }

    fn request_json(
        &mut self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let response = self.client.request(method, path, body)?;
        if response.status != 200 {
            self.last_error = Some(format!(
                "{} {} returned HTTP {}: {}",
                method, path, response.status, response.body
            ));
            return Err(self.last_error.clone().expect("set"));
        }
        serde_json::from_str(&response.body).map_err(|error| error.to_string())
    }

    fn request_text(
        &mut self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> Result<HttpClientResponse, String> {
        let response = self.client.request(method, path, body)?;
        if response.status != 200 {
            self.last_error = Some(format!(
                "{} {} returned HTTP {}: {}",
                method, path, response.status, response.body
            ));
            return Err(self.last_error.clone().expect("set"));
        }
        Ok(response)
    }
}

impl RuntimeProvider for HttpProvider {
    fn name(&self) -> String {
        "http".to_owned()
    }

    fn dashboard_snapshot(&mut self) -> DashboardSnapshot {
        let health = match self.request_json("GET", "/health", None) {
            Ok(health) => health,
            Err(error) => {
                return DashboardSnapshot {
                    runtime_state: "http-disconnected".to_owned(),
                    provider: "http".to_owned(),
                    model_target: "unknown".to_owned(),
                    context_window: "unknown".to_owned(),
                    memory_pressure: "not sampled".to_owned(),
                    active_task: error,
                    ttft_p50_ms: None,
                    decode_tps_p50: None,
                    cache_hit_rate: None,
                };
            }
        };
        let metrics = self
            .client
            .request("GET", "/metrics", None)
            .ok()
            .map(|response| parse_prometheus_metrics(&response.body))
            .unwrap_or_default();
        DashboardSnapshot {
            runtime_state: format!(
                "http-{}",
                health
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown")
            ),
            provider: format!("http {}", self.client.base_url()),
            model_target: health
                .get("server")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("gemma4d")
                .to_owned(),
            context_window: "live control provider".to_owned(),
            memory_pressure: format!(
                "rss {} bytes",
                metrics
                    .get("gemma4d_memory_process_rss_bytes")
                    .copied()
                    .unwrap_or(0.0) as u64
            ),
            active_task: format!(
                "requests {} | active {}",
                metrics
                    .get("gemma4d_requests_total")
                    .copied()
                    .unwrap_or(0.0) as u64,
                metrics
                    .get("gemma4d_active_generations")
                    .copied()
                    .unwrap_or(0.0) as u64
            ),
            ttft_p50_ms: metrics
                .get("gemma4d_ttft_seconds")
                .map(|seconds| seconds * 1000.0),
            decode_tps_p50: metrics.get("gemma4d_tokens_per_second").copied(),
            cache_hit_rate: None,
        }
    }

    fn cache_snapshot(&mut self) -> CacheSnapshot {
        let value = match self.request_json("GET", "/v1/cache/summary", None) {
            Ok(value) => value,
            Err(error) => return CacheSnapshot::disabled(error),
        };
        let mut snapshot = CacheSnapshot::disabled("live HTTP cache summary");
        snapshot.status = string_at(&value, "status", "stub");
        snapshot.cache_mode = string_at(&value, "cache_mode", "bf16");
        snapshot.active_kv_bytes = u64_at(&value, "active_kv_bytes", 0);
        snapshot.note = format!("live cache summary from {}", self.client.base_url());
        if let Some(ram) = value.get("ram") {
            snapshot.ram.resident_bytes = u64_at(ram, "resident_bytes", 0);
            snapshot.ram.resident_blocks = u64_at(ram, "resident_blocks", 0) as usize;
            snapshot.ram.hits = u64_at(ram, "hits", 0);
            snapshot.ram.misses = u64_at(ram, "misses", 0);
            snapshot.ram.restore_failures = u64_at(ram, "restore_failures", 0);
        }
        if let Some(ssd) = value.get("ssd") {
            snapshot.ssd.stored_bytes = u64_at(ssd, "stored_bytes", 0);
            snapshot.ssd.stored_blocks = u64_at(ssd, "stored_blocks", 0) as usize;
            snapshot.ssd.reads = u64_at(ssd, "reads", 0);
            snapshot.ssd.writes = u64_at(ssd, "writes", 0);
            snapshot.ssd.restore_failures = u64_at(ssd, "restore_failures", 0);
            snapshot.ssd.namespace_rejections = u64_at(ssd, "namespace_rejections", 0);
        }
        snapshot
    }

    fn adapter_snapshot(&mut self) -> AdapterSnapshot {
        let value = match self.request_json("GET", "/v1/adapters", None) {
            Ok(value) => value,
            Err(error) => return AdapterSnapshot::disabled(error),
        };
        let entries = value
            .get("data")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .map(|item| AdapterEntrySnapshot {
                        adapter_id: string_at(item, "id", "unknown"),
                        display_name: None,
                        adapter_type: "lora".to_owned(),
                        source_path: PathBuf::from(string_at(item, "source", "trusted-local")),
                        loaded: bool_at(item, "loaded", false),
                        pinned: bool_at(item, "pinned", false),
                        active: false,
                        resident_bytes: u64_at(item, "resident_bytes", 0),
                        load_latency_us: 0,
                        target_modules: Vec::new(),
                        supports_mtp: string_at(item, "supports_mtp", "unknown"),
                        adapter_weight_hash: "server-reported".to_owned(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let loaded = entries.iter().filter(|entry| entry.loaded).count();
        let pinned = entries.iter().filter(|entry| entry.pinned).count();
        let total_resident_bytes = entries.iter().map(|entry| entry.resident_bytes).sum();
        AdapterSnapshot {
            status: "live".to_owned(),
            registry_root: PathBuf::from(self.client.base_url()),
            trusted_roots: vec![PathBuf::from("server-trusted-local")],
            loaded,
            pinned,
            active_adapter_id: entries
                .iter()
                .find(|entry| entry.loaded)
                .map(|entry| entry.adapter_id.clone()),
            total_resident_bytes,
            last_load_latency_us: None,
            mtp_disabled_active: loaded > 0,
            entries,
            note: format!("live adapter list from {}", self.client.base_url()),
        }
    }

    fn chat_snapshot(&mut self) -> ChatSnapshot {
        let adapter = self
            .adapter_snapshot()
            .active_adapter_id
            .unwrap_or_else(|| "none".to_owned());
        let adapter_field = if adapter == "none" {
            String::new()
        } else {
            format!(r#","adapter":"{adapter}""#)
        };
        let body = format!(
            r#"{{
  "model":"mlx-community/gemma-4-12B-it-4bit",
  "messages":[{{"role":"user","content":"hello from TUI"}}],
  "stream":true,
  "temperature":0,
  "max_tokens":8{adapter_field}
}}"#
        );
        let response = match self.request_text("POST", "/v1/chat/completions", Some(&body)) {
            Ok(response) => response,
            Err(error) => {
                self.last_chat = ChatSnapshot::disabled(error);
                return self.last_chat.clone();
            }
        };
        let events = parse_sse_events(&response.body);
        let preview = events
            .iter()
            .filter(|event| *event != "[DONE]")
            .filter_map(|event| serde_json::from_str::<serde_json::Value>(event).ok())
            .filter_map(|value| {
                value["choices"][0]["delta"]["content"]
                    .as_str()
                    .map(str::to_owned)
            })
            .collect::<Vec<_>>()
            .join("");
        self.last_chat = ChatSnapshot {
            status: "streaming-smoke-ok".to_owned(),
            server_url: self.client.base_url().to_owned(),
            model: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
            stream_enabled: response.content_type.starts_with("text/event-stream"),
            active_adapter_id: (adapter != "none").then_some(adapter),
            last_prompt: "hello from TUI".to_owned(),
            last_response_preview: preview,
            stream_events: events.len(),
            note: "live streaming chat smoke through /v1/chat/completions".to_owned(),
        };
        self.last_chat.clone()
    }

    fn mtp_snapshot(&mut self) -> MtpSnapshot {
        MtpSnapshot::disabled("HTTP provider exposes MTP metrics through /metrics in M11")
    }

    fn backend_events(&mut self) -> Vec<BackendEvent> {
        if self.emitted_bootstrap_logs {
            return Vec::new();
        }
        self.emitted_bootstrap_logs = true;
        vec![
            BackendEvent::Log(LogEntry::info(format!(
                "HTTP provider attached to {}",
                self.client.base_url()
            ))),
            BackendEvent::Dashboard(self.dashboard_snapshot()),
            BackendEvent::Cache(self.cache_snapshot()),
            BackendEvent::Adapters(self.adapter_snapshot()),
            BackendEvent::Chat(self.chat_snapshot()),
        ]
    }

    fn validate_config(&mut self, path: &Path) -> ConfigValidation {
        validate_config_path(path)
    }

    fn start_benchmark(&mut self, out_dir: &Path) -> BenchmarkRecord {
        let response = self
            .request_json("POST", "/v1/benchmarks/run", Some("{}"))
            .unwrap_or_else(|_| serde_json::json!({"status":"failed"}));
        let status = match string_at(&response, "status", "failed").as_str() {
            "ready" => BenchmarkStatus::Ready,
            "running" => BenchmarkStatus::Running,
            "completed" => BenchmarkStatus::Completed,
            _ => BenchmarkStatus::Failed,
        };
        BenchmarkRecord {
            status,
            command: "POST /v1/benchmarks/run".to_owned(),
            out_dir: out_dir.to_path_buf(),
            report_path: PathBuf::from(string_at(
                &response,
                "report_path",
                "benchmarks/out/M11/stub-report.md",
            )),
            note: format!(
                "benchmark status via HTTP provider {}",
                self.client.base_url()
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct SimpleHttpClient {
    base_url: String,
    addr: SocketAddr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HttpClientResponse {
    status: u16,
    content_type: String,
    body: String,
}

impl SimpleHttpClient {
    fn new(base_url: String) -> Self {
        let addr = parse_http_url_addr(&base_url).unwrap_or_else(|_| {
            "127.0.0.1:8080"
                .parse()
                .expect("fallback server address is valid")
        });
        Self { base_url, addr }
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> Result<HttpClientResponse, String> {
        let mut stream = TcpStream::connect_timeout(&self.addr, Duration::from_secs(2))
            .map_err(|error| format!("connect {} failed: {error}", self.base_url))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|error| error.to_string())?;
        let body = body.unwrap_or("");
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.addr,
            body.len(),
            body
        )
        .map_err(|error| error.to_string())?;
        let mut raw = String::new();
        stream
            .read_to_string(&mut raw)
            .map_err(|error| error.to_string())?;
        parse_http_response(&raw)
    }
}

fn parse_http_url_addr(url: &str) -> Result<SocketAddr, String> {
    let without_scheme = url
        .strip_prefix("http://")
        .ok_or_else(|| "HttpProvider only supports http:// URLs in M11".to_owned())?;
    let host_port = without_scheme.split('/').next().unwrap_or(without_scheme);
    host_port
        .parse::<SocketAddr>()
        .map_err(|error| format!("invalid server_url {url}: {error}"))
}

fn parse_http_response(raw: &str) -> Result<HttpClientResponse, String> {
    let Some((headers, body)) = raw.split_once("\r\n\r\n") else {
        return Err("invalid HTTP response".to_owned());
    };
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or_else(|| "missing HTTP status".to_owned())?
        .parse::<u16>()
        .map_err(|error| format!("invalid HTTP status: {error}"))?;
    let content_type = headers
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-type")
                    .then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_default();
    Ok(HttpClientResponse {
        status,
        content_type,
        body: body.to_owned(),
    })
}

fn parse_prometheus_metrics(body: &str) -> std::collections::BTreeMap<String, f64> {
    body.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let mut parts = line.split_whitespace();
            let name = parts.next()?.split('{').next()?.to_owned();
            let value = parts.next()?.parse::<f64>().ok()?;
            Some((name, value))
        })
        .collect()
}

fn parse_sse_events(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .map(str::trim)
        .filter(|event| !event.is_empty())
        .map(str::to_owned)
        .collect()
}

fn string_at(value: &serde_json::Value, key: &str, fallback: &str) -> String {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or(fallback)
        .to_owned()
}

fn u64_at(value: &serde_json::Value, key: &str, fallback: u64) -> u64 {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(fallback)
}

fn bool_at(value: &serde_json::Value, key: &str, fallback: bool) -> bool {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_http_url_addr() {
        assert_eq!(
            parse_http_url_addr("http://127.0.0.1:8080").expect("addr"),
            "127.0.0.1:8080".parse::<SocketAddr>().expect("addr")
        );
        assert!(parse_http_url_addr("https://127.0.0.1:8080").is_err());
    }

    #[test]
    fn parses_sse_data_events() {
        let events = parse_sse_events("data: {\"a\":1}\n\ndata: [DONE]\n\n");
        assert_eq!(events, vec!["{\"a\":1}", "[DONE]"]);
    }
}
