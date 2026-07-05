use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    app::{
        AdapterEntrySnapshot, AdapterSnapshot, BackendEvent, BenchmarkRecord, BenchmarkStatus,
        CacheSnapshot, ChatSnapshot, DashboardSnapshot, LiveMetricsSnapshot, LogEntry, MtpSnapshot,
        MtpStatus, ProviderKind,
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
            backend: "mock".to_owned(),
            model_target: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
            context_window: "32768 tokens".to_owned(),
            native_prefill_policy: "mock server policy unavailable".to_owned(),
            memory_pressure: "42% mock pressure".to_owned(),
            active_task: format!("idle tick {}", self.tick),
            ttft_p50_ms: Some(118.4),
            decode_tps_p50: Some(31.7),
            cache_hit_rate: Some(0.0),
            live: LiveMetricsSnapshot::mock(),
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
            backend: "file".to_owned(),
            model_target: "from repository config files".to_owned(),
            context_window: "from tiny16-style config".to_owned(),
            native_prefill_policy: "no live server policy".to_owned(),
            memory_pressure: "not sampled; no daemon connected".to_owned(),
            active_task: benchmark_state.to_owned(),
            ttft_p50_ms: None,
            decode_tps_p50: None,
            cache_hit_rate: None,
            live: LiveMetricsSnapshot::unavailable("file-offline"),
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

    fn request_metrics(&mut self) -> std::collections::BTreeMap<String, f64> {
        self.client
            .request("GET", "/metrics", None)
            .ok()
            .map(|response| parse_prometheus_metrics(&response.body))
            .unwrap_or_default()
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
                    backend: "unknown".to_owned(),
                    model_target: "unknown".to_owned(),
                    context_window: "unknown".to_owned(),
                    native_prefill_policy: "unknown".to_owned(),
                    memory_pressure: "not sampled".to_owned(),
                    active_task: error,
                    ttft_p50_ms: None,
                    decode_tps_p50: None,
                    cache_hit_rate: None,
                    live: LiveMetricsSnapshot::unavailable("http-disconnected"),
                };
            }
        };
        let metrics = self.request_metrics();
        let server_health = string_at(&health, "status", "unknown");
        let model_loaded = bool_at(&health, "model_loaded", false);
        let process_rss_bytes = metric_u64(&metrics, "gemma4d_memory_process_rss_bytes");
        let peak_mlx_bytes = metric_u64(&metrics, "gemma4d_memory_peak_mlx_bytes");
        let ttft_ms = metric_seconds_ms(&metrics, "gemma4d_ttft_seconds");
        let decode_tps = metric_option(&metrics, "gemma4d_tokens_per_second");
        let backend = string_at(&health, "backend", "unknown");
        let max_context_tokens = u64_at(&health, "max_context_tokens", 0);
        DashboardSnapshot {
            runtime_state: format!("http-{server_health}"),
            provider: format!("http {}", self.client.base_url()),
            backend,
            model_target: health
                .get("server")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("gemma4d")
                .to_owned(),
            context_window: if max_context_tokens == 0 {
                "live control provider".to_owned()
            } else {
                format!("{max_context_tokens} tokens")
            },
            native_prefill_policy: native_prefill_policy_summary(&health),
            memory_pressure: format!(
                "rss {process_rss_bytes} bytes | peak mlx {peak_mlx_bytes} bytes"
            ),
            active_task: format!(
                "requests {} | active {}",
                metric_u64(&metrics, "gemma4d_requests_total"),
                metric_u64(&metrics, "gemma4d_active_generations")
            ),
            ttft_p50_ms: ttft_ms,
            decode_tps_p50: decode_tps,
            cache_hit_rate: None,
            live: LiveMetricsSnapshot {
                server_health,
                model_loaded,
                requests_total: metric_u64(&metrics, "gemma4d_requests_total"),
                active_generations: metric_u64(&metrics, "gemma4d_active_generations"),
                model_load_ms: metric_seconds_ms(&metrics, "gemma4d_model_load_seconds"),
                prefill_ms: metric_seconds_ms(&metrics, "gemma4d_prefill_seconds"),
                decode_ms: metric_seconds_ms(&metrics, "gemma4d_decode_seconds"),
                ttft_ms,
                tokens_per_second: decode_tps,
                prefill_tokens_total: metric_u64(&metrics, "gemma4d_prefill_tokens_total"),
                decode_tokens_total: metric_u64(&metrics, "gemma4d_decode_tokens_total"),
                process_rss_bytes,
                peak_mlx_bytes,
            },
        }
    }

    fn cache_snapshot(&mut self) -> CacheSnapshot {
        let value = match self.request_json("GET", "/v1/cache/summary", None) {
            Ok(value) => value,
            Err(error) => return CacheSnapshot::disabled(error),
        };
        let metrics = self.request_metrics();
        let mut snapshot = CacheSnapshot::disabled("live HTTP cache summary");
        snapshot.status = string_at(&value, "status", "stub");
        snapshot.cache_mode = string_at(&value, "cache_mode", "bf16");
        snapshot.namespace_hash = value
            .get("namespace_hash")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        snapshot.active_kv_bytes = u64_at(
            &value,
            "active_kv_bytes",
            metric_u64(&metrics, "gemma4d_kv_active_bytes"),
        );
        snapshot.note = format!("live cache summary from {}", self.client.base_url());
        if let Some(ram) = value.get("ram") {
            snapshot.ram.resident_bytes = u64_at(ram, "resident_bytes", 0);
            snapshot.ram.resident_blocks = u64_at(ram, "resident_blocks", 0) as usize;
            snapshot.ram.hits = u64_at(
                ram,
                "hits",
                metric_u64(&metrics, "gemma4d_prefix_cache_hits_total"),
            );
            snapshot.ram.misses = u64_at(
                ram,
                "misses",
                metric_u64(&metrics, "gemma4d_prefix_cache_misses_total"),
            );
            snapshot.ram.restore_failures = u64_at(
                ram,
                "restore_failures",
                metric_u64(&metrics, "gemma4d_cache_restore_failures_total"),
            );
            snapshot.ram.hit_rate = hit_rate(snapshot.ram.hits, snapshot.ram.misses);
        }
        if let Some(ssd) = value.get("ssd") {
            snapshot.ssd.stored_bytes = u64_at(ssd, "stored_bytes", 0);
            snapshot.ssd.stored_blocks = u64_at(ssd, "stored_blocks", 0) as usize;
            snapshot.ssd.reads = u64_at(ssd, "reads", 0);
            snapshot.ssd.writes = u64_at(ssd, "writes", 0);
            snapshot.ssd.restore_failures = u64_at(ssd, "restore_failures", 0);
            snapshot.ssd.namespace_rejections = u64_at(ssd, "namespace_rejections", 0);
            snapshot.ssd.bytes_read = metric_u64(&metrics, "gemma4d_ssd_cache_read_bytes_total");
            snapshot.ssd.bytes_written =
                metric_u64(&metrics, "gemma4d_ssd_cache_write_bytes_total");
        }
        snapshot
    }

    fn adapter_snapshot(&mut self) -> AdapterSnapshot {
        let value = match self.request_json("GET", "/v1/adapters", None) {
            Ok(value) => value,
            Err(error) => return AdapterSnapshot::disabled(error),
        };
        let metrics = self.request_metrics();
        let mut entries = value
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
                        active: bool_at(item, "active", false),
                        resident_bytes: u64_at(item, "resident_bytes", 0),
                        load_latency_us: 0,
                        target_modules: Vec::new(),
                        supports_mtp: string_at(item, "supports_mtp", "unknown"),
                        adapter_weight_hash: "server-reported".to_owned(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if !entries.iter().any(|entry| entry.active)
            && let Some(entry) = entries.iter_mut().find(|entry| entry.loaded)
        {
            entry.active = true;
        }
        let loaded = entries.iter().filter(|entry| entry.loaded).count();
        let pinned = entries.iter().filter(|entry| entry.pinned).count();
        let total_resident_bytes = entries.iter().map(|entry| entry.resident_bytes).sum();
        let last_load_latency_us =
            metric_option(&metrics, "gemma4d_adapter_load_seconds").map(|seconds| {
                let micros = seconds * 1_000_000.0;
                micros.max(0.0).round() as u128
            });
        AdapterSnapshot {
            status: "live".to_owned(),
            registry_root: PathBuf::from(self.client.base_url()),
            trusted_roots: vec![PathBuf::from("server-trusted-local")],
            loaded,
            pinned,
            active_adapter_id: entries
                .iter()
                .find(|entry| entry.active)
                .map(|entry| entry.adapter_id.clone()),
            total_resident_bytes,
            last_load_latency_us,
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
        let metrics = self.request_metrics();
        let attempted = metric_u64(&metrics, "gemma4d_mtp_attempted_tokens_total");
        let accepted = metric_u64(&metrics, "gemma4d_mtp_accepted_tokens_total");
        let auto_disabled = metric_u64(&metrics, "gemma4d_mtp_auto_disabled_total");
        let rollbacks = metric_u64(&metrics, "gemma4d_mtp_rollbacks_total");
        let acceptance_rate = metric_option(&metrics, "gemma4d_mtp_acceptance_rate")
            .unwrap_or_else(|| hit_rate(accepted, attempted.saturating_sub(accepted)));
        let loaded_adapters = metric_u64(&metrics, "gemma4d_adapters_loaded");
        let active_kv_bytes = metric_u64(&metrics, "gemma4d_kv_active_bytes");
        let status = if auto_disabled > 0 {
            MtpStatus::AutoDisabled
        } else if attempted > 0 {
            MtpStatus::Enabled
        } else {
            MtpStatus::Disabled
        };
        MtpSnapshot {
            status,
            target: "server /metrics target".to_owned(),
            drafter: "server /metrics drafter".to_owned(),
            compatibility: "reported by provider metrics; no TUI native calls".to_owned(),
            exactness: "not asserted by TUI; see MTP benchmark artifacts".to_owned(),
            draft_block_size: 0,
            attempted_draft_tokens: attempted,
            accepted_draft_tokens: accepted,
            acceptance_rate,
            accepted_tokens_per_verify: 0.0,
            target_verify_passes: 0,
            rollback_count: rollbacks,
            auto_disable_reason: (auto_disabled > 0)
                .then(|| format!("{auto_disabled} provider auto-disable events")),
            failing_fixture: None,
            adapter_state: if loaded_adapters > 0 {
                format!("{loaded_adapters} loaded adapter(s); MTP gated")
            } else {
                "no loaded adapters reported".to_owned()
            },
            active_kv_mode: format!("provider active KV {active_kv_bytes} bytes"),
        }
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
            BackendEvent::Mtp(self.mtp_snapshot()),
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
    let mut metrics = std::collections::BTreeMap::<String, f64>::new();
    for (name, value) in body.lines().filter_map(|line| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let mut parts = line.split_whitespace();
        let name = parts.next()?.split('{').next()?.to_owned();
        let value = parts.next()?.parse::<f64>().ok()?;
        Some((name, value))
    }) {
        *metrics.entry(name).or_insert(0.0) += value;
    }
    metrics
}

fn metric_option(metrics: &std::collections::BTreeMap<String, f64>, name: &str) -> Option<f64> {
    metrics.get(name).copied()
}

fn metric_u64(metrics: &std::collections::BTreeMap<String, f64>, name: &str) -> u64 {
    metrics.get(name).copied().unwrap_or(0.0).max(0.0).round() as u64
}

fn metric_seconds_ms(metrics: &std::collections::BTreeMap<String, f64>, name: &str) -> Option<f64> {
    metric_option(metrics, name).map(|seconds| seconds * 1000.0)
}

fn hit_rate(hits: u64, misses: u64) -> f64 {
    let total = hits.saturating_add(misses);
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
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

fn native_prefill_policy_summary(value: &serde_json::Value) -> String {
    let Some(native_prefill) = value.get("native_prefill") else {
        return "native prefill policy unknown".to_owned();
    };
    let policy = native_prefill
        .get("server_default_policy")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("none");
    let admission = native_prefill
        .get("admission_prefill_chunked")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let source = native_prefill
        .get("state_source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    format!("policy {policy} | admission_chunked {admission} | source {source}")
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
