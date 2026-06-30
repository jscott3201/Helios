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
