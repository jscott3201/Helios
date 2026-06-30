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
use provider::{RuntimeProvider, create_provider_with_server};

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

    #[arg(long, default_value = "http://127.0.0.1:8080")]
    pub server_url: String,

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
    /// Run the M12 non-interactive TUI release walkthrough and write evidence.
    ReleaseWalkthrough {
        #[arg(long, default_value = "benchmarks/out/M12/tui-walkthrough")]
        out_dir: PathBuf,
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
    let mut provider =
        create_provider_with_server(cli.provider, PathBuf::from("."), cli.server_url.clone());
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
        Some(CliCommand::ReleaseWalkthrough { out_dir }) => {
            let outcome = write_release_walkthrough(
                &mut state,
                provider.as_mut(),
                cli.config.clone(),
                out_dir,
            )?;
            Ok(outcome)
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
    reduce(&mut state, Action::ChatUpdated(provider.chat_snapshot()));
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
pub struct ReleaseWalkthroughReport {
    pub schema_version: u32,
    pub milestone: &'static str,
    pub provider: String,
    pub config_status: String,
    pub benchmark_status: String,
    pub dashboard_status: String,
    pub chat_status: String,
    pub cache_status: String,
    pub adapter_status: String,
    pub snapshot_count: usize,
    pub render_profile: RenderProfile,
    pub report_path: PathBuf,
    pub json_path: PathBuf,
}

pub fn write_release_walkthrough(
    state: &mut AppState,
    provider: &mut dyn RuntimeProvider,
    config_path: PathBuf,
    out_dir: &std::path::Path,
) -> Result<RunOutcome, TuiError> {
    fs::create_dir_all(out_dir)?;

    let validation = provider.validate_config(&config_path);
    reduce(state, Action::ConfigValidated(validation.clone()));

    reduce(
        state,
        Action::DashboardUpdated(provider.dashboard_snapshot()),
    );
    reduce(state, Action::CacheUpdated(provider.cache_snapshot()));
    reduce(state, Action::AdaptersUpdated(provider.adapter_snapshot()));
    reduce(state, Action::ChatUpdated(provider.chat_snapshot()));
    reduce(state, Action::MtpUpdated(provider.mtp_snapshot()));

    let benchmark_dir = out_dir.join("benchmark-launch");
    let benchmark = provider.start_benchmark(&benchmark_dir);
    reduce(state, Action::BenchmarkRecorded(benchmark.clone()));

    let snapshot_dir = out_dir.join("snapshots");
    let snapshots = write_snapshots(state, &snapshot_dir)?;
    let render_profile = profile_render(state, 120, 120, 40)?;

    let report_path = out_dir.join("tui-release-walkthrough.md");
    let json_path = out_dir.join("tui-release-walkthrough.json");
    let report = ReleaseWalkthroughReport {
        schema_version: 1,
        milestone: "M12",
        provider: state.provider_name.clone(),
        config_status: validation.status.label().to_owned(),
        benchmark_status: benchmark.status.label().to_owned(),
        dashboard_status: state.dashboard.runtime_state.clone(),
        chat_status: state.chat.status.clone(),
        cache_status: state.cache.status.clone(),
        adapter_status: state.adapters.status.clone(),
        snapshot_count: snapshots.len(),
        render_profile,
        report_path: report_path.clone(),
        json_path: json_path.clone(),
    };

    fs::write(
        &json_path,
        serde_json::to_vec_pretty(&report).map_err(|error| TuiError::Render(error.to_string()))?,
    )?;
    fs::write(
        &report_path,
        render_release_walkthrough_markdown(WalkthroughMarkdown {
            report: &report,
            validation: &validation,
            benchmark: &benchmark,
            dashboard: &state.dashboard,
            cache: &state.cache,
            adapters: &state.adapters,
            chat: &state.chat,
            snapshots: &snapshots,
        }),
    )?;

    let mut evidence_paths = vec![report_path.clone(), json_path.clone(), snapshot_dir];
    evidence_paths.push(benchmark_dir);
    Ok(RunOutcome {
        message: format!(
            "wrote M12 TUI release walkthrough to {}",
            report_path.display()
        ),
        evidence_paths,
    })
}

struct WalkthroughMarkdown<'a> {
    report: &'a ReleaseWalkthroughReport,
    validation: &'a config::ConfigValidation,
    benchmark: &'a app::BenchmarkRecord,
    dashboard: &'a app::DashboardSnapshot,
    cache: &'a app::CacheSnapshot,
    adapters: &'a app::AdapterSnapshot,
    chat: &'a app::ChatSnapshot,
    snapshots: &'a [PathBuf],
}

fn render_release_walkthrough_markdown(input: WalkthroughMarkdown<'_>) -> String {
    let WalkthroughMarkdown {
        report,
        validation,
        benchmark,
        dashboard,
        cache,
        adapters,
        chat,
        snapshots,
    } = input;
    let mut out = String::new();
    out.push_str("# M12 TUI Release Walkthrough\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Item | Value |\n|---|---|\n");
    out.push_str(&format!(
        "| Provider | `{}` |\n",
        markdown_escape(&report.provider)
    ));
    out.push_str(&format!(
        "| Config validation | `{}` |\n",
        markdown_escape(&report.config_status)
    ));
    out.push_str(&format!(
        "| Benchmark launch | `{}` |\n",
        markdown_escape(&report.benchmark_status)
    ));
    out.push_str(&format!(
        "| Dashboard | `{}` |\n",
        markdown_escape(&report.dashboard_status)
    ));
    out.push_str(&format!(
        "| Chat | `{}` |\n",
        markdown_escape(&report.chat_status)
    ));
    out.push_str(&format!(
        "| Cache | `{}` |\n",
        markdown_escape(&report.cache_status)
    ));
    out.push_str(&format!(
        "| Adapters | `{}` |\n",
        markdown_escape(&report.adapter_status)
    ));
    out.push_str(&format!(
        "| Render p50/p95 | `{}` us / `{}` us |\n\n",
        report.render_profile.p50_us, report.render_profile.p95_us
    ));

    out.push_str("## Workflow Evidence\n\n");
    out.push_str("| Step | Evidence |\n|---|---|\n");
    out.push_str(&format!(
        "| Config validation | `{}`: {} |\n",
        validation.path.display(),
        markdown_escape(&validation.summary)
    ));
    out.push_str(&format!(
        "| Benchmark launch | `{}` -> `{}` ({}) |\n",
        markdown_escape(&benchmark.command),
        benchmark.report_path.display(),
        markdown_escape(&benchmark.note)
    ));
    out.push_str(&format!(
        "| Metrics review | {} / {} / {:?} ms TTFT |\n",
        markdown_escape(&dashboard.runtime_state),
        markdown_escape(&dashboard.memory_pressure),
        dashboard.ttft_p50_ms
    ));
    out.push_str(&format!(
        "| Adapter status | loaded {} pinned {} active {:?} resident {} bytes |\n",
        adapters.loaded, adapters.pinned, adapters.active_adapter_id, adapters.total_resident_bytes
    ));
    out.push_str(&format!(
        "| Cache status | {} active_kv={} ram_blocks={} ssd_blocks={} |\n",
        markdown_escape(&cache.status),
        cache.active_kv_bytes,
        cache.ram.resident_blocks,
        cache.ssd.stored_blocks
    ));
    out.push_str(&format!(
        "| Streaming chat | {} events={} preview `{}` |\n",
        markdown_escape(&chat.status),
        chat.stream_events,
        markdown_escape(&chat.last_response_preview)
    ));
    out.push_str(&format!(
        "| Report export | `{}` and `{}` |\n\n",
        report.report_path.display(),
        report.json_path.display()
    ));

    out.push_str("## Snapshot Artifacts\n\n");
    for path in snapshots {
        out.push_str(&format!("- `{}`\n", path.display()));
    }
    out
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

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
