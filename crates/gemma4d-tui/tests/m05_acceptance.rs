use std::{
    cell::RefCell,
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use gemma4d_server::http::{ServerConfig, ServerRuntime, serve_listener};
use gemma4d_tui::{
    TuiError,
    app::{Action, AppState, BenchmarkStatus, PageId, reduce},
    config::{ValidationStatus, validate_config_path},
    profile_render,
    provider::{HttpProvider, MockProvider, RuntimeProvider},
    seed_state,
    terminal::{TerminalLifecycle, TerminalOps, key_to_action},
    ui::render_snapshot,
};

#[test]
fn reducer_navigates_resizes_and_quits() {
    let mut state = AppState::new("mock", PathBuf::from("references/configs/tui.toml"));

    reduce(&mut state, Action::NextPage);
    assert_eq!(state.current_page, PageId::Config);

    reduce(&mut state, Action::PreviousPage);
    assert_eq!(state.current_page, PageId::Dashboard);

    reduce(&mut state, Action::Navigate(PageId::Logs));
    assert_eq!(state.current_page, PageId::Logs);

    reduce(&mut state, Action::Resize(120, 40));
    assert_eq!(state.terminal_size, (120, 40));

    reduce(&mut state, Action::Tick);
    assert_eq!(state.tick_count, 1);

    reduce(&mut state, Action::QuitRequested);
    assert!(state.should_quit);
}

#[test]
fn keybindings_map_to_expected_actions() {
    assert_eq!(
        key_to_action(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
        Some(Action::NextPage)
    );
    assert_eq!(
        key_to_action(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)),
        Some(Action::PreviousPage)
    );
    assert_eq!(
        key_to_action(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE)),
        Some(Action::Navigate(PageId::Benchmarks))
    );
    assert_eq!(
        key_to_action(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE)),
        Some(Action::Navigate(PageId::Help))
    );
    assert_eq!(
        key_to_action(KeyEvent::new(KeyCode::Char('v'), KeyModifiers::NONE)),
        Some(Action::ValidateCurrentConfig)
    );
    assert_eq!(
        key_to_action(KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE)),
        Some(Action::StartBenchmark)
    );
    assert_eq!(
        key_to_action(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
        Some(Action::QuitRequested)
    );
}

#[test]
fn invalid_tiny16_fixture_is_caught() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/invalid-tiny16.toml");
    let validation = validate_config_path(&path);

    assert_eq!(validation.status, ValidationStatus::Invalid);
    assert!(
        validation
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("headroom"))
    );
    assert!(
        validation
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.path == "[model].target")
    );
}

#[test]
fn mock_benchmark_records_exact_command_and_output_path() {
    let mut provider = MockProvider::default();
    let out_dir = Path::new("benchmarks/out/M05/mock-bench");
    let record = provider.start_benchmark(out_dir);

    assert_eq!(record.status, BenchmarkStatus::Completed);
    assert_eq!(record.out_dir, out_dir);
    assert_eq!(record.report_path, out_dir.join("mock-report.md"));
    assert!(record.command.contains("gemma4d-bench run"));
    assert!(
        record
            .command
            .contains("--out-dir benchmarks/out/M05/mock-bench")
    );
    assert!(record.note.contains("no process was spawned"));
}

#[test]
fn snapshots_render_required_pages_at_required_sizes() {
    let mut provider = MockProvider::default();
    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/configs/tui.toml");
    let base = seed_state(&mut provider, config_path);

    for page in [
        PageId::Dashboard,
        PageId::Config,
        PageId::Benchmarks,
        PageId::Chat,
        PageId::Adapters,
        PageId::Logs,
        PageId::Help,
    ] {
        let mut state = base.clone();
        reduce(&mut state, Action::Navigate(page));

        for (width, height) in [(80, 24), (120, 40)] {
            let snapshot = render_snapshot(&state, width, height).unwrap();
            assert!(
                snapshot.contains(page.title()),
                "snapshot for {page:?} at {width}x{height} did not contain page title"
            );
        }
    }
}

#[test]
fn chat_page_renders_streaming_status() {
    let mut provider = MockProvider::default();
    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/configs/tui.toml");
    let base = seed_state(&mut provider, config_path);

    let mut state = base.clone();
    reduce(&mut state, Action::Navigate(PageId::Chat));
    let snapshot = render_snapshot(&state, 80, 24).unwrap();
    assert!(snapshot.contains("Chat"));
    assert!(snapshot.contains("streaming-smoke-ready"));
    assert!(snapshot.contains("Streaming yes"));
    assert!(snapshot.contains("stub adapter rust-coding-r16-v1"));
}

#[test]
fn cache_page_renders_cache_accounting_and_compression_summary() {
    let mut provider = MockProvider::default();
    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/configs/tui.toml");
    let mut state = seed_state(&mut provider, config_path);
    reduce(&mut state, Action::Navigate(PageId::Cache));

    let snapshot = render_snapshot(&state, 80, 24).unwrap();
    assert!(snapshot.contains("Cache"));
    assert!(snapshot.contains("bf16_default"));
    assert!(snapshot.contains("RAM resident"));
    assert!(snapshot.contains("restore failures"));
    assert!(snapshot.contains("SSD enabled"));
    assert!(snapshot.contains("bytes read"));
    assert!(snapshot.contains("mid-decode fetches 0"));
    assert!(snapshot.contains("Compression q8"));
    assert!(snapshot.contains("BF16 default yes"));
    assert!(snapshot.contains("Planar/Iso feature_disabled_default"));
}

#[test]
fn mtp_page_renders_acceptance_rollback_and_auto_disable_status() {
    let mut provider = MockProvider::default();
    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/configs/tui.toml");
    let mut state = seed_state(&mut provider, config_path);
    reduce(&mut state, Action::Navigate(PageId::Mtp));

    let snapshot = render_snapshot(&state, 80, 24).unwrap();
    assert!(snapshot.contains("MTP"));
    assert!(snapshot.contains("auto-disabled"));
    assert!(snapshot.contains("Acceptance rate"));
    assert!(snapshot.contains("Rollbacks"));
    assert!(snapshot.contains("Auto-disable reason"));
}

#[test]
fn adapter_page_renders_registry_summary_and_mtp_gate() {
    let mut provider = MockProvider::default();
    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/configs/tui.toml");
    let mut state = seed_state(&mut provider, config_path);
    reduce(&mut state, Action::Navigate(PageId::Adapters));

    let snapshot = render_snapshot(&state, 80, 24).unwrap();
    assert!(snapshot.contains("Adapters"));
    assert!(snapshot.contains("Registry benchmarks/out/M10/registry"));
    assert!(snapshot.contains("rust-coding-r16-v1"));
    assert!(snapshot.contains("loaded yes"));
    assert!(snapshot.contains("pinned yes"));
    assert!(snapshot.contains("MTP disabled with active adapter yes"));
}

#[test]
fn http_provider_attaches_to_live_server_and_streams_chat() {
    let (addr, shutdown, handle) = spawn_stub_server();
    let mut provider = HttpProvider::new(format!("http://{addr}"));
    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/configs/tui.toml");
    let mut state = seed_state(&mut provider, config_path);

    assert_eq!(state.dashboard.runtime_state, "http-ok");
    assert!(state.dashboard.provider.contains("http://"));
    assert_eq!(state.adapters.loaded, 1);
    assert_eq!(state.chat.status, "streaming-smoke-ok");
    assert!(state.chat.stream_enabled);
    assert!(state.chat.stream_events >= 4);
    assert!(state.chat.last_response_preview.contains("hello from TUI"));

    reduce(&mut state, Action::Navigate(PageId::Chat));
    let snapshot = render_snapshot(&state, 80, 24).unwrap();
    assert!(snapshot.contains("streaming-smoke-ok"));
    assert!(snapshot.contains("hello from TUI"));

    shutdown.store(true, Ordering::SeqCst);
    let _ = TcpStream::connect(addr);
    handle.join().expect("server thread");
}

#[test]
fn render_profile_reports_p50_and_p95() {
    let mut provider = MockProvider::default();
    let config_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../references/configs/tui.toml");
    let state = seed_state(&mut provider, config_path);
    let profile = profile_render(&state, 3, 80, 24).unwrap();

    assert_eq!(profile.iterations, 3);
    assert_eq!((profile.width, profile.height), (80, 24));
    assert!(profile.p95_us >= profile.p50_us);
}

#[test]
fn terminal_lifecycle_restores_after_normal_quit() {
    let events = Rc::new(RefCell::new(Vec::<&'static str>::new()));
    let ops = RecordingOps {
        events: Rc::clone(&events),
    };
    let mut lifecycle = TerminalLifecycle::new(ops);

    lifecycle.enter().unwrap();
    lifecycle.restore().unwrap();

    assert_eq!(
        events.borrow().as_slice(),
        ["enable_raw", "enter_alt", "leave_alt", "disable_raw"]
    );
}

#[test]
fn terminal_lifecycle_restores_after_controlled_error() {
    let events = Rc::new(RefCell::new(Vec::<&'static str>::new()));

    {
        let ops = RecordingOps {
            events: Rc::clone(&events),
        };
        let mut lifecycle = TerminalLifecycle::new(ops);
        let result: Result<(), TuiError> = (|| {
            lifecycle.enter()?;
            Err(TuiError::Terminal("controlled test error".to_owned()))
        })();
        assert!(result.is_err());
    }

    assert_eq!(
        events.borrow().as_slice(),
        ["enable_raw", "enter_alt", "leave_alt", "disable_raw"]
    );
}

fn spawn_stub_server() -> (
    std::net::SocketAddr,
    Arc<AtomicBool>,
    thread::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let shutdown = Arc::new(AtomicBool::new(false));
    let server_shutdown = Arc::clone(&shutdown);
    let runtime = ServerRuntime::new(ServerConfig::default().with_bind_addr(addr));
    let handle = thread::spawn(move || {
        serve_listener(listener, runtime, server_shutdown).expect("serve listener");
    });
    (addr, shutdown, handle)
}

#[derive(Clone)]
struct RecordingOps {
    events: Rc<RefCell<Vec<&'static str>>>,
}

impl TerminalOps for RecordingOps {
    fn enable_raw(&mut self) -> Result<(), TuiError> {
        self.events.borrow_mut().push("enable_raw");
        Ok(())
    }

    fn enter_alternate_screen(&mut self) -> Result<(), TuiError> {
        self.events.borrow_mut().push("enter_alt");
        Ok(())
    }

    fn leave_alternate_screen(&mut self) -> Result<(), TuiError> {
        self.events.borrow_mut().push("leave_alt");
        Ok(())
    }

    fn disable_raw(&mut self) -> Result<(), TuiError> {
        self.events.borrow_mut().push("disable_raw");
        Ok(())
    }
}
