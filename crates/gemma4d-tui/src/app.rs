use std::path::PathBuf;

use clap::ValueEnum;
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
            Self::Cache => {
                Some("Disabled until M07/M08 expose KV cache and prefix-cache providers.")
            }
            Self::Adapters => Some("Disabled until M10 dynamic adapter loading is implemented."),
            Self::Mtp => Some("Disabled until M06 speculative decoding exposes runtime telemetry."),
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
    Log(LogEntry),
    Benchmark(BenchmarkRecord),
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
    pub benchmark: BenchmarkRecord,
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
            benchmark,
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
            BackendEvent::Log(entry) => reduce(state, Action::AppendLog(entry)),
            BackendEvent::Benchmark(record) => reduce(state, Action::BenchmarkRecorded(record)),
        },
        Action::DashboardUpdated(snapshot) => {
            state.dashboard = snapshot;
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
