use std::{
    io::{self, Write},
    path::PathBuf,
    time::Duration,
};

use crossterm::{
    event::{self, Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::{
    TuiError,
    app::{Action, AppState, PageId, reduce},
    provider::RuntimeProvider,
    ui,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppEvent {
    Input(KeyEvent),
    Resize(u16, u16),
    Tick,
}

pub trait TerminalOps {
    fn enable_raw(&mut self) -> Result<(), TuiError>;
    fn enter_alternate_screen(&mut self) -> Result<(), TuiError>;
    fn leave_alternate_screen(&mut self) -> Result<(), TuiError>;
    fn disable_raw(&mut self) -> Result<(), TuiError>;
}

#[derive(Debug, Default)]
pub struct CrosstermOps;

impl TerminalOps for CrosstermOps {
    fn enable_raw(&mut self) -> Result<(), TuiError> {
        enable_raw_mode().map_err(|error| TuiError::Terminal(error.to_string()))
    }

    fn enter_alternate_screen(&mut self) -> Result<(), TuiError> {
        execute!(io::stdout(), EnterAlternateScreen)
            .map_err(|error| TuiError::Terminal(error.to_string()))
    }

    fn leave_alternate_screen(&mut self) -> Result<(), TuiError> {
        execute!(io::stdout(), LeaveAlternateScreen)
            .map_err(|error| TuiError::Terminal(error.to_string()))
    }

    fn disable_raw(&mut self) -> Result<(), TuiError> {
        disable_raw_mode().map_err(|error| TuiError::Terminal(error.to_string()))
    }
}

#[derive(Debug)]
pub struct TerminalLifecycle<O: TerminalOps> {
    ops: O,
    raw_enabled: bool,
    alternate_screen: bool,
}

impl<O: TerminalOps> TerminalLifecycle<O> {
    pub fn new(ops: O) -> Self {
        Self {
            ops,
            raw_enabled: false,
            alternate_screen: false,
        }
    }

    pub fn enter(&mut self) -> Result<(), TuiError> {
        self.ops.enable_raw()?;
        self.raw_enabled = true;
        if let Err(error) = self.ops.enter_alternate_screen() {
            let _ = self.restore();
            return Err(error);
        }
        self.alternate_screen = true;
        Ok(())
    }

    pub fn restore(&mut self) -> Result<(), TuiError> {
        let mut first_error = None;
        if self.alternate_screen {
            if let Err(error) = self.ops.leave_alternate_screen() {
                first_error = Some(error);
            }
            self.alternate_screen = false;
        }
        if self.raw_enabled {
            if let Err(error) = self.ops.disable_raw() {
                first_error.get_or_insert(error);
            }
            self.raw_enabled = false;
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl<O: TerminalOps> Drop for TerminalLifecycle<O> {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

pub fn run_interactive(
    mut state: AppState,
    provider: &mut dyn RuntimeProvider,
    fail_after_init: bool,
    benchmark_out_dir: PathBuf,
) -> Result<(), TuiError> {
    let mut lifecycle = TerminalLifecycle::new(CrosstermOps);
    lifecycle.enter()?;
    if fail_after_init {
        return Err(TuiError::Terminal(
            "controlled failure after terminal initialization".to_owned(),
        ));
    }

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal =
        Terminal::new(backend).map_err(|error| TuiError::Terminal(error.to_string()))?;

    while !state.should_quit {
        terminal
            .draw(|frame| ui::render_app(frame, &state))
            .map_err(|error| TuiError::Render(error.to_string()))?;

        match next_event(Duration::from_millis(250))? {
            AppEvent::Input(key) => {
                if let Some(action) = key_to_action(key) {
                    dispatch_action(&mut state, provider, action, benchmark_out_dir.clone());
                }
            }
            AppEvent::Resize(width, height) => reduce(&mut state, Action::Resize(width, height)),
            AppEvent::Tick => {
                reduce(&mut state, Action::Tick);
                reduce(
                    &mut state,
                    Action::DashboardUpdated(provider.dashboard_snapshot()),
                );
                reduce(&mut state, Action::CacheUpdated(provider.cache_snapshot()));
                reduce(&mut state, Action::MtpUpdated(provider.mtp_snapshot()));
                for event in provider.backend_events() {
                    reduce(&mut state, Action::BackendEvent(event));
                }
            }
        }
    }

    terminal
        .show_cursor()
        .map_err(|error| TuiError::Terminal(error.to_string()))?;
    lifecycle.restore()
}

pub fn dispatch_action(
    state: &mut AppState,
    provider: &mut dyn RuntimeProvider,
    action: Action,
    benchmark_out_dir: PathBuf,
) {
    match action {
        Action::ValidateCurrentConfig => {
            reduce(state, Action::ValidateCurrentConfig);
            let validation = provider.validate_config(&state.config_path);
            reduce(state, Action::ConfigValidated(validation));
        }
        Action::StartBenchmark => {
            reduce(state, Action::StartBenchmark);
            let record = provider.start_benchmark(&benchmark_out_dir);
            reduce(state, Action::BenchmarkRecorded(record));
        }
        Action::RefreshRequested => {
            reduce(state, Action::RefreshRequested);
            reduce(
                state,
                Action::DashboardUpdated(provider.dashboard_snapshot()),
            );
            reduce(state, Action::CacheUpdated(provider.cache_snapshot()));
            reduce(state, Action::MtpUpdated(provider.mtp_snapshot()));
            for event in provider.backend_events() {
                reduce(state, Action::BackendEvent(event));
            }
        }
        other => reduce(state, other),
    }
}

pub fn next_event(timeout: Duration) -> Result<AppEvent, TuiError> {
    if event::poll(timeout).map_err(|error| TuiError::Terminal(error.to_string()))? {
        match event::read().map_err(|error| TuiError::Terminal(error.to_string()))? {
            CrosstermEvent::Key(key) => Ok(AppEvent::Input(key)),
            CrosstermEvent::Resize(width, height) => Ok(AppEvent::Resize(width, height)),
            _ => Ok(AppEvent::Tick),
        }
    } else {
        Ok(AppEvent::Tick)
    }
}

pub fn key_to_action(key: KeyEvent) -> Option<Action> {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            Some(Action::QuitRequested)
        }
        KeyCode::Char('q') => Some(Action::QuitRequested),
        KeyCode::Esc => Some(Action::Navigate(PageId::Dashboard)),
        KeyCode::Tab => Some(Action::NextPage),
        KeyCode::BackTab => Some(Action::PreviousPage),
        KeyCode::Char('?') => Some(Action::Navigate(PageId::Help)),
        KeyCode::Char('r') => Some(Action::RefreshRequested),
        KeyCode::Char('v') => Some(Action::ValidateCurrentConfig),
        KeyCode::Char('b') => Some(Action::StartBenchmark),
        KeyCode::Char(ch) => PageId::from_digit(ch).map(Action::Navigate),
        _ => None,
    }
}

pub fn flush_stdout() {
    let _ = io::stdout().flush();
}
