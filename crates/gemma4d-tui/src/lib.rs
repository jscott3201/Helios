#![doc = "Ratatui local operator console for gemma4d."]

pub mod app;
pub mod config;
pub mod provider;
pub mod terminal;
pub mod ui;

use std::{
    fs,
    io::{self, IsTerminal},
    path::PathBuf,
    time::Instant,
};

use app::{Action, AppState, PageId, ProviderKind, reduce};
use clap::{Parser, Subcommand};
use provider::{RuntimeProvider, create_provider};

pub const CRATE_NAME: &str = "gemma4d-tui";

pub fn bootstrap_status() -> &'static str {
    "ratatui-operator-console"
}

#[derive(Debug, Clone, Parser)]
#[command(
    name = CRATE_NAME,
    about = "Local operator console for offline gemma4d status, config, benchmarks, and logs"
)]
pub struct Cli {
    #[arg(long, value_enum, default_value_t = ProviderKind::Mock)]
    pub provider: ProviderKind,

    #[arg(long, default_value = "references/configs/tui.toml")]
    pub config: PathBuf,

    #[arg(long, default_value = "benchmarks/out/M05")]
    pub out_dir: PathBuf,

    #[arg(long, hide = true)]
    pub fail_after_init: bool,

    #[command(subcommand)]
    pub command: Option<CliCommand>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum CliCommand {
    /// Write deterministic text snapshots for all pages and exit.
    Snapshot {
        #[arg(long, default_value = "benchmarks/out/M05/snapshots")]
        out_dir: PathBuf,
    },
    /// Validate a TOML config with the selected provider and exit.
    Validate {
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Record a mock/file-backed benchmark launch surface and exit.
    Benchmark {
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
    /// Measure deterministic snapshot render latency and exit.
    ProfileRender {
        #[arg(long, default_value_t = 200)]
        iterations: usize,
        #[arg(long, default_value_t = 80)]
        width: u16,
        #[arg(long, default_value_t = 24)]
        height: u16,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    pub message: String,
    pub evidence_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiError {
    Io(String),
    Config(String),
    Provider(String),
    Render(String),
    Terminal(String),
}

impl std::fmt::Display for TuiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(message)
            | Self::Config(message)
            | Self::Provider(message)
            | Self::Render(message)
            | Self::Terminal(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for TuiError {}

impl From<io::Error> for TuiError {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

pub async fn run(cli: Cli) -> Result<RunOutcome, TuiError> {
    let mut provider = create_provider(cli.provider, PathBuf::from("."));
    let mut state = seed_state(provider.as_mut(), cli.config.clone());

    match &cli.command {
        Some(CliCommand::Snapshot { out_dir }) => {
            let paths = write_snapshots(&state, out_dir)?;
            Ok(RunOutcome {
                message: format!(
                    "wrote {} TUI snapshots to {}",
                    paths.len(),
                    out_dir.display()
                ),
                evidence_paths: paths,
            })
        }
        Some(CliCommand::Validate { config }) => {
            let path = config.clone().unwrap_or_else(|| cli.config.clone());
            let validation = provider.validate_config(&path);
            let message = serde_json::to_string_pretty(&validation)
                .map_err(|error| TuiError::Config(error.to_string()))?;
            Ok(RunOutcome {
                message,
                evidence_paths: vec![path],
            })
        }
        Some(CliCommand::Benchmark { out_dir }) => {
            let path = out_dir.clone().unwrap_or_else(|| cli.out_dir.clone());
            let record = provider.start_benchmark(&path);
            reduce(&mut state, Action::BenchmarkRecorded(record.clone()));
            let message = serde_json::to_string_pretty(&record)
                .map_err(|error| TuiError::Provider(error.to_string()))?;
            Ok(RunOutcome {
                message,
                evidence_paths: vec![path, record.report_path],
            })
        }
        Some(CliCommand::ProfileRender {
            iterations,
            width,
            height,
        }) => {
            let profile = profile_render(&state, *iterations, *width, *height)?;
            let message = serde_json::to_string_pretty(&profile)
                .map_err(|error| TuiError::Render(error.to_string()))?;
            Ok(RunOutcome {
                message,
                evidence_paths: Vec::new(),
            })
        }
        None => {
            if cli.fail_after_init || (io::stdin().is_terminal() && io::stdout().is_terminal()) {
                terminal::run_interactive(
                    state,
                    provider.as_mut(),
                    cli.fail_after_init,
                    cli.out_dir.clone(),
                )?;
                return Ok(RunOutcome {
                    message: "interactive TUI exited cleanly".to_owned(),
                    evidence_paths: Vec::new(),
                });
            }

            fs::create_dir_all(&cli.out_dir)?;
            let snapshot_path = cli.out_dir.join("headless-dashboard-80x24.snap");
            fs::write(&snapshot_path, ui::render_snapshot(&state, 80, 24)?)?;
            Ok(RunOutcome {
                message: format!(
                    "provider={} rendered 80x24 headless snapshot; stdin/stdout are not TTY, exiting cleanly",
                    state.provider_name
                ),
                evidence_paths: vec![snapshot_path],
            })
        }
    }
}

pub fn seed_state(provider: &mut dyn RuntimeProvider, config_path: PathBuf) -> AppState {
    let mut state = AppState::new(provider.name(), config_path.clone());
    reduce(
        &mut state,
        Action::DashboardUpdated(provider.dashboard_snapshot()),
    );
    reduce(&mut state, Action::CacheUpdated(provider.cache_snapshot()));
    reduce(
        &mut state,
        Action::AdaptersUpdated(provider.adapter_snapshot()),
    );
    reduce(&mut state, Action::MtpUpdated(provider.mtp_snapshot()));
    for event in provider.backend_events() {
        reduce(&mut state, Action::BackendEvent(event));
    }
    reduce(
        &mut state,
        Action::ConfigValidated(provider.validate_config(&config_path)),
    );
    state
}

pub fn write_snapshots(
    state: &AppState,
    out_dir: &std::path::Path,
) -> Result<Vec<PathBuf>, TuiError> {
    fs::create_dir_all(out_dir)?;
    let mut paths = Vec::new();
    for page in PageId::ALL {
        let mut page_state = state.clone();
        reduce(&mut page_state, Action::Navigate(page));
        for (width, height) in [(80, 24), (120, 40)] {
            let snapshot = ui::render_snapshot(&page_state, width, height)?;
            let file_name = format!("{}-{}x{}.snap", page.slug(), width, height);
            let path = out_dir.join(file_name);
            fs::write(&path, snapshot)?;
            paths.push(path);
        }
    }
    Ok(paths)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RenderProfile {
    pub iterations: usize,
    pub width: u16,
    pub height: u16,
    pub p50_us: u128,
    pub p95_us: u128,
}

pub fn profile_render(
    state: &AppState,
    iterations: usize,
    width: u16,
    height: u16,
) -> Result<RenderProfile, TuiError> {
    if iterations == 0 {
        return Err(TuiError::Render(
            "profile-render requires at least one iteration".to_owned(),
        ));
    }

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let _snapshot = ui::render_snapshot(state, width, height)?;
        samples.push(started.elapsed().as_micros());
    }
    samples.sort_unstable();
    let p50_index = percentile_index(iterations, 50);
    let p95_index = percentile_index(iterations, 95);
    Ok(RenderProfile {
        iterations,
        width,
        height,
        p50_us: samples[p50_index],
        p95_us: samples[p95_index],
    })
}

fn percentile_index(len: usize, percentile: usize) -> usize {
    let rank = (len * percentile).div_ceil(100);
    rank.saturating_sub(1).min(len - 1)
}
