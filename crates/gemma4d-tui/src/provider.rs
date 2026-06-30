use std::path::{Path, PathBuf};

use crate::{
    app::{
        BackendEvent, BenchmarkRecord, BenchmarkStatus, DashboardSnapshot, LogEntry, MtpSnapshot,
        ProviderKind,
    },
    config::{ConfigValidation, validate_config_path},
};

pub trait RuntimeProvider {
    fn name(&self) -> String;
    fn dashboard_snapshot(&mut self) -> DashboardSnapshot;
    fn mtp_snapshot(&mut self) -> MtpSnapshot;
    fn backend_events(&mut self) -> Vec<BackendEvent>;
    fn validate_config(&mut self, path: &Path) -> ConfigValidation;
    fn start_benchmark(&mut self, out_dir: &Path) -> BenchmarkRecord;
}

pub fn create_provider(kind: ProviderKind, root: PathBuf) -> Box<dyn RuntimeProvider> {
    match kind {
        ProviderKind::Mock => Box::new(MockProvider::default()),
        ProviderKind::File => Box::new(FileProvider::new(root)),
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
            BackendEvent::Mtp(self.mtp_snapshot()),
        ]
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
                "runtime daemon absent; Chat/Cache/Adapters/MTP pages remain disabled",
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
