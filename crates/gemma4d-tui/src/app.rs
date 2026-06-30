use std::path::PathBuf;

use clap::ValueEnum;
use gemma4d_kv::{CacheAccountingSnapshot, SsdCacheAccountingSnapshot};
use serde::{Deserialize, Serialize};

use crate::config::ConfigValidation;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Mock,
    File,
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Mock => "mock",
            Self::File => "file",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageId {
    Dashboard,
    Config,
    Benchmarks,
    Chat,
    Cache,
    Adapters,
    Mtp,
    Logs,
    Help,
}

impl PageId {
    pub const ALL: [Self; 9] = [
        Self::Dashboard,
        Self::Config,
        Self::Benchmarks,
        Self::Chat,
        Self::Cache,
        Self::Adapters,
        Self::Mtp,
        Self::Logs,
        Self::Help,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Dashboard => "Dashboard",
            Self::Config => "Config",
            Self::Benchmarks => "Benchmarks",
            Self::Chat => "Chat",
            Self::Cache => "Cache",
            Self::Adapters => "Adapters",
            Self::Mtp => "MTP",
            Self::Logs => "Logs",
            Self::Help => "Help",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Dashboard => "dashboard",
            Self::Config => "config",
            Self::Benchmarks => "benchmarks",
            Self::Chat => "chat",
            Self::Cache => "cache",
            Self::Adapters => "adapters",
            Self::Mtp => "mtp",
            Self::Logs => "logs",
            Self::Help => "help",
        }
    }

    pub fn dependency_message(self) -> Option<&'static str> {
        match self {
            Self::Chat => Some("Disabled until M11 provides the local server/control provider."),
            _ => None,
        }
    }

    pub fn from_digit(ch: char) -> Option<Self> {
        let index = ch.to_digit(10)?;
        if index == 0 {
            None
        } else {
            Self::ALL.get((index - 1) as usize).copied()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    pub runtime_state: String,
    pub provider: String,
    pub model_target: String,
    pub context_window: String,
    pub memory_pressure: String,
    pub active_task: String,
    pub ttft_p50_ms: Option<f64>,
    pub decode_tps_p50: Option<f64>,
    pub cache_hit_rate: Option<f64>,
}

impl DashboardSnapshot {
    pub fn offline(provider: &str) -> Self {
        Self {
            runtime_state: "offline/mock".to_owned(),
            provider: provider.to_owned(),
            model_target: "Gemma 4 12B QAT target placeholder".to_owned(),
            context_window: "32768 tokens configured".to_owned(),
            memory_pressure: "not sampled".to_owned(),
            active_task: "no live runtime attached".to_owned(),
            ttft_p50_ms: None,
            decode_tps_p50: None,
            cache_hit_rate: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheSnapshot {
    pub status: String,
    pub cache_mode: String,
    pub block_size_tokens: u64,
    pub namespace_hash: Option<String>,
    pub ram: CacheAccountingSnapshot,
    pub ssd: SsdCacheAccountingSnapshot,
    pub compression: CompressionSnapshot,
    pub active_kv_bytes: u64,
    pub restored_tokens: u64,
    pub rejected_namespaces: u64,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionSnapshot {
    pub bf16_default: bool,
    pub q8_min_logit_cosine: f64,
    pub q4_min_logit_cosine: f64,
    pub q8_memory_reduction: f64,
    pub q4_memory_reduction: f64,
    pub namespace_hashes_unique_by_mode: bool,
    pub planar_iso_status: String,
}

impl CompressionSnapshot {
    pub fn disabled() -> Self {
        Self {
            bf16_default: true,
            q8_min_logit_cosine: 0.0,
            q4_min_logit_cosine: 0.0,
            q8_memory_reduction: 0.0,
            q4_memory_reduction: 0.0,
            namespace_hashes_unique_by_mode: true,
            planar_iso_status: "disabled".to_owned(),
        }
    }

    pub fn mock_m09() -> Self {
        Self {
            bf16_default: true,
            q8_min_logit_cosine: 0.999_941_945_454_655_4,
            q4_min_logit_cosine: 0.982_667_226_802_970_6,
            q8_memory_reduction: 0.499_999_903_142_452_13,
            q4_memory_reduction: 0.749_999_903_142_452_5,
            namespace_hashes_unique_by_mode: true,
            planar_iso_status: "feature_disabled_default".to_owned(),
        }
    }
}

impl CacheSnapshot {
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            status: "disabled".to_owned(),
            cache_mode: "none".to_owned(),
            block_size_tokens: 0,
            namespace_hash: None,
            ram: CacheAccountingSnapshot {
                budget_bytes: 0,
                resident_bytes: 0,
                resident_blocks: 0,
                hits: 0,
                misses: 0,
                evictions: 0,
                restore_failures: 0,
                hit_rate: 0.0,
                ssd_enabled: false,
            },
            ssd: SsdCacheAccountingSnapshot {
                budget_bytes: 0,
                stored_bytes: 0,
                stored_blocks: 0,
                hits: 0,
                misses: 0,
                writes: 0,
                reads: 0,
                evictions: 0,
                restore_failures: 0,
                namespace_rejections: 0,
                corruptions: 0,
                bytes_written: 0,
                bytes_read: 0,
                hit_rate: 0.0,
                mid_decode_fetches: 0,
                ssd_enabled: false,
            },
            compression: CompressionSnapshot::disabled(),
            active_kv_bytes: 0,
            restored_tokens: 0,
            rejected_namespaces: 0,
            note: reason.into(),
        }
    }

    pub fn mock_m09() -> Self {
        Self {
            status: "compression-evaluated".to_owned(),
            cache_mode: "bf16_default+mlx_affine_q8+mlx_affine_q4".to_owned(),
            block_size_tokens: 16 * 1024,
            namespace_hash: Some("m09-mock-namespace".to_owned()),
            ram: CacheAccountingSnapshot {
                budget_bytes: 64 * 1024 * 1024 * 1024,
                resident_bytes: 12 * 1024 * 1024 * 1024,
                resident_blocks: 4,
                hits: 4,
                misses: 0,
                evictions: 0,
                restore_failures: 12,
                hit_rate: 1.0,
                ssd_enabled: false,
            },
            ssd: SsdCacheAccountingSnapshot {
                budget_bytes: 64 * 1024 * 1024,
                stored_bytes: 164 * 1024,
                stored_blocks: 4,
                hits: 4,
                misses: 0,
                writes: 4,
                reads: 8,
                evictions: 0,
                restore_failures: 8,
                namespace_rejections: 4,
                corruptions: 4,
                bytes_written: 164 * 1024,
                bytes_read: 328 * 1024,
                hit_rate: 1.0,
                mid_decode_fetches: 0,
                ssd_enabled: true,
            },
            compression: CompressionSnapshot::mock_m09(),
            active_kv_bytes: 3 * 1024 * 1024 * 1024,
            restored_tokens: 16_384 + 32_768 + 65_536,
            rejected_namespaces: 16,
            note: "M09 q8/q4 fixture gates pass; Planar/Iso remains experimental".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterSnapshot {
    pub status: String,
    pub registry_root: PathBuf,
    pub trusted_roots: Vec<PathBuf>,
    pub loaded: usize,
    pub pinned: usize,
    pub active_adapter_id: Option<String>,
    pub total_resident_bytes: u64,
    pub last_load_latency_us: Option<u128>,
    pub mtp_disabled_active: bool,
    pub entries: Vec<AdapterEntrySnapshot>,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterEntrySnapshot {
    pub adapter_id: String,
    pub display_name: Option<String>,
    pub adapter_type: String,
    pub source_path: PathBuf,
    pub loaded: bool,
    pub pinned: bool,
    pub active: bool,
    pub resident_bytes: u64,
    pub load_latency_us: u128,
    pub target_modules: Vec<String>,
    pub supports_mtp: String,
    pub adapter_weight_hash: String,
}

impl AdapterSnapshot {
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            status: "disabled".to_owned(),
            registry_root: PathBuf::from("none"),
            trusted_roots: Vec::new(),
            loaded: 0,
            pinned: 0,
            active_adapter_id: None,
            total_resident_bytes: 0,
            last_load_latency_us: None,
            mtp_disabled_active: false,
            entries: Vec::new(),
            note: reason.into(),
        }
    }

    pub fn mock_m10() -> Self {
        Self {
            status: "ready".to_owned(),
            registry_root: PathBuf::from("benchmarks/out/M10/registry"),
            trusted_roots: vec![PathBuf::from("benchmarks/out/M10/fixtures/trusted")],
            loaded: 1,
            pinned: 1,
            active_adapter_id: Some("rust-coding-r16-v1".to_owned()),
            total_resident_bytes: 4096,
            last_load_latency_us: Some(1200),
            mtp_disabled_active: true,
            entries: vec![
                AdapterEntrySnapshot {
                    adapter_id: "rust-coding-r16-v1".to_owned(),
                    display_name: Some("Rust coding fixture".to_owned()),
                    adapter_type: "lora".to_owned(),
                    source_path: PathBuf::from(
                        "benchmarks/out/M10/fixtures/trusted/rust-coding-r16-v1",
                    ),
                    loaded: true,
                    pinned: true,
                    active: true,
                    resident_bytes: 4096,
                    load_latency_us: 1200,
                    target_modules: vec!["q_proj".to_owned(), "v_proj".to_owned()],
                    supports_mtp: "unknown".to_owned(),
                    adapter_weight_hash: "m10-fixture-adapter-weight-hash".to_owned(),
                },
                AdapterEntrySnapshot {
                    adapter_id: "sql-r16-v1".to_owned(),
                    display_name: Some("SQL fixture".to_owned()),
                    adapter_type: "lora".to_owned(),
                    source_path: PathBuf::from("benchmarks/out/M10/fixtures/trusted/sql-r16-v1"),
                    loaded: false,
                    pinned: false,
                    active: false,
                    resident_bytes: 0,
                    load_latency_us: 0,
                    target_modules: vec!["q_proj".to_owned(), "v_proj".to_owned()],
                    supports_mtp: "unknown".to_owned(),
                    adapter_weight_hash: "m10-fixture-sql-weight-hash".to_owned(),
                },
            ],
            note: "M10 standard LoRA registry is provider-backed; MTP stays disabled while an adapter is active".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtpStatus {
    Disabled,
    Enabled,
    AutoDisabled,
}

impl MtpStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Enabled => "enabled",
            Self::AutoDisabled => "auto-disabled",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MtpSnapshot {
    pub status: MtpStatus,
    pub target: String,
    pub drafter: String,
    pub compatibility: String,
    pub exactness: String,
    pub draft_block_size: usize,
    pub attempted_draft_tokens: u64,
    pub accepted_draft_tokens: u64,
    pub acceptance_rate: f64,
    pub accepted_tokens_per_verify: f64,
    pub target_verify_passes: u64,
    pub rollback_count: u64,
    pub auto_disable_reason: Option<String>,
    pub failing_fixture: Option<String>,
    pub adapter_state: String,
    pub active_kv_mode: String,
}

impl MtpSnapshot {
    pub fn disabled(reason: impl Into<String>) -> Self {
        Self {
            status: MtpStatus::Disabled,
            target: "none".to_owned(),
            drafter: "none".to_owned(),
            compatibility: "not checked".to_owned(),
            exactness: "not run".to_owned(),
            draft_block_size: 0,
            attempted_draft_tokens: 0,
            accepted_draft_tokens: 0,
            acceptance_rate: 0.0,
            accepted_tokens_per_verify: 0.0,
            target_verify_passes: 0,
            rollback_count: 0,
            auto_disable_reason: Some(reason.into()),
            failing_fixture: None,
            adapter_state: "adapters disabled for MTP in M06".to_owned(),
            active_kv_mode: "bf16 required".to_owned(),
        }
    }

    pub fn mock_m06() -> Self {
        Self {
            status: MtpStatus::AutoDisabled,
            target: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
            drafter: "mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned(),
            compatibility: "target/drafter ABI present; native assistant execution unsupported"
                .to_owned(),
            exactness:
                "block size 1 exact on fixture; block size 2 auto-disabled on failing fixture"
                    .to_owned(),
            draft_block_size: 2,
            attempted_draft_tokens: 2,
            accepted_draft_tokens: 0,
            acceptance_rate: 0.0,
            accepted_tokens_per_verify: 0.0,
            target_verify_passes: 1,
            rollback_count: 1,
            auto_disable_reason: Some(
                "acceptance rate fell below threshold after rejected draft".to_owned(),
            ),
            failing_fixture: Some("mtp_reject_second_token".to_owned()),
            adapter_state: "disabled; adapter != none is rejected for MTP in M06".to_owned(),
            active_kv_mode: "bf16 only; compressed active KV disabled".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkStatus {
    Ready,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl BenchmarkStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    pub status: BenchmarkStatus,
    pub command: String,
    pub out_dir: PathBuf,
    pub report_path: PathBuf,
    pub note: String,
}

impl BenchmarkRecord {
    pub fn ready(out_dir: PathBuf) -> Self {
        Self {
            status: BenchmarkStatus::Ready,
            command: format!(
                "gemma4d-bench run --model-path artifacts/models/gemma-4-12B-it-4bit --corpus benchmarks/prompts/M04-corpus.tsv --out-dir {} --reference mlx-helper --max-prompts 1",
                out_dir.display()
            ),
            report_path: out_dir.join("mock-report.md"),
            out_dir,
            note: "mock provider records launch details without spawning model/runtime work"
                .to_owned(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

impl LogLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
}

impl LogEntry {
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Info,
            message: message.into(),
        }
    }

    pub fn warn(message: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Warn,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BackendEvent {
    Dashboard(DashboardSnapshot),
    Cache(CacheSnapshot),
    Adapters(AdapterSnapshot),
    Log(LogEntry),
    Benchmark(BenchmarkRecord),
    Mtp(MtpSnapshot),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Navigate(PageId),
    NextPage,
    PreviousPage,
    Tick,
    Resize(u16, u16),
    RefreshRequested,
    ValidateCurrentConfig,
    ConfigValidated(ConfigValidation),
    StartBenchmark,
    BenchmarkRecorded(BenchmarkRecord),
    CacheUpdated(CacheSnapshot),
    AdaptersUpdated(AdapterSnapshot),
    MtpUpdated(MtpSnapshot),
    BackendEvent(BackendEvent),
    DashboardUpdated(DashboardSnapshot),
    AppendLog(LogEntry),
    QuitRequested,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppState {
    pub current_page: PageId,
    pub should_quit: bool,
    pub tick_count: u64,
    pub terminal_size: (u16, u16),
    pub provider_name: String,
    pub config_path: PathBuf,
    pub config_validation: ConfigValidation,
    pub dashboard: DashboardSnapshot,
    pub cache: CacheSnapshot,
    pub adapters: AdapterSnapshot,
    pub benchmark: BenchmarkRecord,
    pub mtp: MtpSnapshot,
    pub logs: Vec<LogEntry>,
    pub status_line: String,
}

impl AppState {
    pub fn new(provider_name: impl Into<String>, config_path: PathBuf) -> Self {
        let provider_name = provider_name.into();
        let benchmark = BenchmarkRecord::ready(PathBuf::from("benchmarks/out/M05"));
        Self {
            current_page: PageId::Dashboard,
            should_quit: false,
            tick_count: 0,
            terminal_size: (80, 24),
            provider_name: provider_name.clone(),
            config_path: config_path.clone(),
            config_validation: ConfigValidation::pending(config_path),
            dashboard: DashboardSnapshot::offline(&provider_name),
            cache: CacheSnapshot::disabled("cache provider snapshot pending"),
            adapters: AdapterSnapshot::disabled("adapter registry snapshot pending"),
            benchmark,
            mtp: MtpSnapshot::disabled("MTP provider snapshot pending"),
            logs: vec![LogEntry::info(
                "TUI initialized with offline provider boundary",
            )],
            status_line: "Tab/Shift-Tab navigate, 1-9 jump, ? help, q quit".to_owned(),
        }
    }

    pub fn current_index(&self) -> usize {
        PageId::ALL
            .iter()
            .position(|page| *page == self.current_page)
            .unwrap_or(0)
    }
}

pub fn reduce(state: &mut AppState, action: Action) {
    match action {
        Action::Navigate(page) => {
            state.current_page = page;
            state.status_line = format!("{} page", page.title());
        }
        Action::NextPage => {
            let next = (state.current_index() + 1) % PageId::ALL.len();
            reduce(state, Action::Navigate(PageId::ALL[next]));
        }
        Action::PreviousPage => {
            let index = state.current_index();
            let previous = if index == 0 {
                PageId::ALL.len() - 1
            } else {
                index - 1
            };
            reduce(state, Action::Navigate(PageId::ALL[previous]));
        }
        Action::Tick => {
            state.tick_count = state.tick_count.saturating_add(1);
        }
        Action::Resize(width, height) => {
            state.terminal_size = (width, height);
            state.status_line = format!("terminal resized to {width}x{height}");
        }
        Action::RefreshRequested => {
            state.status_line = "refresh requested".to_owned();
        }
        Action::ValidateCurrentConfig => {
            state.status_line = format!("validating {}", state.config_path.display());
        }
        Action::ConfigValidated(validation) => {
            state.status_line = validation.summary.clone();
            state.config_validation = validation;
        }
        Action::StartBenchmark => {
            state.status_line = "benchmark launch requested".to_owned();
        }
        Action::BenchmarkRecorded(record) => {
            state.status_line = format!(
                "benchmark {}: {}",
                record.status.label(),
                record.report_path.display()
            );
            state.benchmark = record;
        }
        Action::BackendEvent(event) => match event {
            BackendEvent::Dashboard(snapshot) => reduce(state, Action::DashboardUpdated(snapshot)),
            BackendEvent::Cache(snapshot) => reduce(state, Action::CacheUpdated(snapshot)),
            BackendEvent::Adapters(snapshot) => reduce(state, Action::AdaptersUpdated(snapshot)),
            BackendEvent::Log(entry) => reduce(state, Action::AppendLog(entry)),
            BackendEvent::Benchmark(record) => reduce(state, Action::BenchmarkRecorded(record)),
            BackendEvent::Mtp(snapshot) => reduce(state, Action::MtpUpdated(snapshot)),
        },
        Action::DashboardUpdated(snapshot) => {
            state.dashboard = snapshot;
        }
        Action::CacheUpdated(snapshot) => {
            state.status_line = format!("cache {}", snapshot.status);
            state.cache = snapshot;
        }
        Action::AdaptersUpdated(snapshot) => {
            state.status_line = format!("adapters {}", snapshot.status);
            state.adapters = snapshot;
        }
        Action::MtpUpdated(snapshot) => {
            state.status_line = format!("MTP {}", snapshot.status.label());
            state.mtp = snapshot;
        }
        Action::AppendLog(entry) => {
            state.logs.push(entry);
            if state.logs.len() > 5000 {
                let overflow = state.logs.len() - 5000;
                state.logs.drain(0..overflow);
            }
        }
        Action::QuitRequested => {
            state.should_quit = true;
            state.status_line = "quit requested".to_owned();
        }
    }
}
