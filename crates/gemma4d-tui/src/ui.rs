use ratatui::{
    Frame, Terminal,
    backend::TestBackend,
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Tabs, Wrap},
};

use crate::{
    TuiError,
    app::{AppState, PageId},
    config::DiagnosticSeverity,
};

pub fn render_app(frame: &mut Frame<'_>, state: &AppState) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(frame.area());

    render_tabs(frame, root[0], state);
    render_page(frame, root[1], state);
    render_status(frame, root[2], state);
}

pub fn render_snapshot(state: &AppState, width: u16, height: u16) -> Result<String, TuiError> {
    let backend = TestBackend::new(width, height);
    let mut terminal =
        Terminal::new(backend).map_err(|error| TuiError::Render(error.to_string()))?;
    terminal
        .draw(|frame| render_app(frame, state))
        .map_err(|error| TuiError::Render(error.to_string()))?;
    Ok(buffer_to_string(terminal.backend().buffer(), width, height))
}

fn render_tabs(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let titles = PageId::ALL
        .iter()
        .enumerate()
        .map(|(index, page)| Line::from(format!("{} {}", index + 1, page.title())))
        .collect::<Vec<_>>();
    let tabs = Tabs::new(titles)
        .select(state.current_index())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("gemma4d operator"),
        )
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(tabs, area);
}

fn render_page(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    match state.current_page {
        PageId::Dashboard => render_dashboard(frame, area, state),
        PageId::Config => render_config(frame, area, state),
        PageId::Benchmarks => render_benchmarks(frame, area, state),
        PageId::Cache => render_cache(frame, area, state),
        PageId::Adapters => render_adapters(frame, area, state),
        PageId::Mtp => render_mtp(frame, area, state),
        PageId::Logs => render_logs(frame, area, state),
        PageId::Help => render_help(frame, area),
        PageId::Chat => render_chat(frame, area, state),
    }
}

fn render_dashboard(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(11),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

    let snapshot = &state.dashboard;
    let live = &snapshot.live;
    let lines = vec![
        Line::from(vec![
            Span::styled("Runtime ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.runtime_state.clone()),
        ]),
        Line::from(format!(
            "Health {} | model loaded {} | requests {} | active {}",
            live.server_health,
            yes_no(live.model_loaded),
            live.requests_total,
            live.active_generations
        )),
        Line::from(vec![
            Span::styled("Provider ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.provider.clone()),
        ]),
        Line::from(vec![
            Span::styled("Backend ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.backend.clone()),
        ]),
        Line::from(vec![
            Span::styled("Model ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.model_target.clone()),
        ]),
        Line::from(vec![
            Span::styled("Context ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.context_window.clone()),
        ]),
        Line::from(vec![
            Span::styled("Native prefill ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.native_prefill_policy.clone()),
        ]),
        Line::from(vec![
            Span::styled("Memory ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.memory_pressure.clone()),
        ]),
        Line::from(vec![
            Span::styled("Task ", Style::default().fg(Color::Gray)),
            Span::raw(snapshot.active_task.clone()),
        ]),
        Line::from(format!(
            "Tokens prefill {} | decode {} | tok/s {}",
            live.prefill_tokens_total,
            live.decode_tokens_total,
            option_tps(live.tokens_per_second)
        )),
        Line::from(vec![
            Span::styled("MTP ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "{} | block {} | accept {:.1}% | rollbacks {}",
                state.mtp.status.label(),
                state.mtp.draft_block_size,
                state.mtp.acceptance_rate * 100.0,
                state.mtp.rollback_count
            )),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Dashboard"))
            .wrap(Wrap { trim: true }),
        chunks[0],
    );

    let utilization = snapshot.cache_hit_rate.unwrap_or(0.0).clamp(0.0, 1.0);
    frame.render_widget(
        Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Prefix cache hit rate"),
            )
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(utilization)
            .label(format!("{:.0}%", utilization * 100.0)),
        chunks[1],
    );

    let perf = format!(
        "Live load {} | Prefill {} | TTFT {} | Decode {} | RSS {} | Peak MLX {} | redraw tick {}",
        option_ms(live.model_load_ms),
        option_ms(live.prefill_ms),
        option_ms(live.ttft_ms.or(snapshot.ttft_p50_ms)),
        option_ms(live.decode_ms),
        bytes(live.process_rss_bytes),
        bytes(live.peak_mlx_bytes),
        state.tick_count
    );
    frame.render_widget(
        Paragraph::new(perf)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Performance snapshot"),
            )
            .wrap(Wrap { trim: true }),
        chunks[2],
    );
}

fn render_config(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let validation = &state.config_validation;
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Path ", Style::default().fg(Color::Gray)),
            Span::raw(validation.path.display().to_string()),
        ]),
        Line::from(vec![
            Span::styled("Status ", Style::default().fg(Color::Gray)),
            Span::styled(
                validation.status.label(),
                Style::default().fg(status_color(validation.status)),
            ),
        ]),
        Line::from(validation.summary.clone()),
        Line::from(""),
    ];
    for diagnostic in validation.diagnostics.iter().take(12) {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", diagnostic.severity.label()),
                Style::default().fg(diagnostic_color(diagnostic.severity)),
            ),
            Span::styled(
                format!("{} ", diagnostic.path),
                Style::default().fg(Color::Gray),
            ),
            Span::raw(diagnostic.message.clone()),
        ]));
    }
    if validation.diagnostics.is_empty() {
        lines.push(Line::from("no diagnostics"));
    }

    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Config"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_benchmarks(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let record = &state.benchmark;
    let live = &state.dashboard.live;
    let lines = vec![
        Line::from(vec![
            Span::styled("Status ", Style::default().fg(Color::Gray)),
            Span::raw(record.status.label()),
        ]),
        Line::from(vec![
            Span::styled("Out dir ", Style::default().fg(Color::Gray)),
            Span::raw(record.out_dir.display().to_string()),
        ]),
        Line::from(vec![
            Span::styled("Report ", Style::default().fg(Color::Gray)),
            Span::raw(record.report_path.display().to_string()),
        ]),
        Line::from(format!(
            "Live load {} | prefill {} | decode {} | tok/s {}",
            option_ms(live.model_load_ms),
            option_ms(live.prefill_ms),
            option_ms(live.decode_ms),
            option_tps(live.tokens_per_second)
        )),
        Line::from(format!(
            "Live memory RSS {} | peak MLX {} | server {}",
            bytes(live.process_rss_bytes),
            bytes(live.peak_mlx_bytes),
            live.server_health
        )),
        Line::from(""),
        Line::from("Exact command"),
        Line::from(record.command.clone()),
        Line::from(""),
        Line::from(record.note.clone()),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Benchmarks"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_chat(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let chat = &state.chat;
    let lines = vec![
        Line::from(vec![
            Span::styled("Status ", Style::default().fg(Color::Gray)),
            Span::raw(chat.status.clone()),
        ]),
        Line::from(vec![
            Span::styled("Server ", Style::default().fg(Color::Gray)),
            Span::raw(chat.server_url.clone()),
        ]),
        Line::from(vec![
            Span::styled("Model ", Style::default().fg(Color::Gray)),
            Span::raw(chat.model.clone()),
        ]),
        Line::from(format!(
            "Streaming {} | events {} | adapter {}",
            yes_no(chat.stream_enabled),
            chat.stream_events,
            chat.active_adapter_id.as_deref().unwrap_or("none")
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("Prompt ", Style::default().fg(Color::Gray)),
            Span::raw(chat.last_prompt.clone()),
        ]),
        Line::from(vec![
            Span::styled("Preview ", Style::default().fg(Color::Gray)),
            Span::raw(chat.last_response_preview.clone()),
        ]),
        Line::from(""),
        Line::from(chat.note.clone()),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Chat"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_cache(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let cache = &state.cache;
    let ram = cache.ram;
    let ssd = cache.ssd;
    let compression = &cache.compression;
    let lines = vec![
        Line::from(vec![
            Span::styled("Status ", Style::default().fg(Color::Gray)),
            Span::raw(cache.status.clone()),
        ]),
        Line::from(vec![
            Span::styled("Mode ", Style::default().fg(Color::Gray)),
            Span::raw(cache.cache_mode.clone()),
        ]),
        Line::from(format!("Block size {} tokens", cache.block_size_tokens)),
        Line::from(format!(
            "Namespace {}",
            cache.namespace_hash.as_deref().unwrap_or("none")
        )),
        Line::from(format!(
            "RAM resident {} / {} across {} blocks",
            bytes(ram.resident_bytes),
            bytes(ram.budget_bytes),
            ram.resident_blocks
        )),
        Line::from(format!(
            "Hits {} | misses {} | hit rate {:.1}%",
            ram.hits,
            ram.misses,
            ram.hit_rate * 100.0
        )),
        Line::from(format!(
            "Evictions {} | restore failures {} | rejected namespaces {}",
            ram.evictions, ram.restore_failures, cache.rejected_namespaces
        )),
        Line::from(format!("Active KV {}", bytes(cache.active_kv_bytes))),
        Line::from(format!("Restored tokens {}", cache.restored_tokens)),
        Line::from(format!(
            "SSD {} | stored {} / {} across {} blocks",
            if ssd.ssd_enabled {
                "enabled"
            } else {
                "disabled"
            },
            bytes(ssd.stored_bytes),
            bytes(ssd.budget_bytes),
            ssd.stored_blocks
        )),
        Line::from(format!(
            "SSD reads {} | writes {} | bytes read {} | written {}",
            ssd.reads,
            ssd.writes,
            bytes(ssd.bytes_read),
            bytes(ssd.bytes_written)
        )),
        Line::from(format!(
            "SSD failures {} | corruptions {} | mid-decode fetches {}",
            ssd.restore_failures, ssd.corruptions, ssd.mid_decode_fetches
        )),
        Line::from(format!(
            "Compression q8 cosine {:.6} | q4 cosine {:.6}",
            compression.q8_min_logit_cosine, compression.q4_min_logit_cosine
        )),
        Line::from(format!(
            "Compression memory q8 {:.1}% | q4 {:.1}% | BF16 default {}",
            compression.q8_memory_reduction * 100.0,
            compression.q4_memory_reduction * 100.0,
            if compression.bf16_default {
                "yes"
            } else {
                "no"
            }
        )),
        Line::from(format!(
            "Namespace by mode {} | Planar/Iso {}",
            if compression.namespace_hashes_unique_by_mode {
                "isolated"
            } else {
                "shared"
            },
            compression.planar_iso_status
        )),
        Line::from(cache.note.clone()),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Cache"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_adapters(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let adapters = &state.adapters;
    let trusted = if adapters.trusted_roots.is_empty() {
        "none".to_owned()
    } else {
        adapters
            .trusted_roots
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Status ", Style::default().fg(Color::Gray)),
            Span::raw(adapters.status.clone()),
        ]),
        Line::from(vec![
            Span::styled("Registry ", Style::default().fg(Color::Gray)),
            Span::raw(adapters.registry_root.display().to_string()),
        ]),
        Line::from(vec![
            Span::styled("Trusted roots ", Style::default().fg(Color::Gray)),
            Span::raw(trusted),
        ]),
        Line::from(format!(
            "Loaded {} | pinned {} | active {}",
            adapters.loaded,
            adapters.pinned,
            adapters.active_adapter_id.as_deref().unwrap_or("none")
        )),
        Line::from(format!(
            "Resident {} | last load {}",
            bytes(adapters.total_resident_bytes),
            adapters
                .last_load_latency_us
                .map(|value| format!("{value} us"))
                .unwrap_or_else(|| "n/a".to_owned())
        )),
        Line::from(format!(
            "MTP disabled with active adapter {}",
            if adapters.mtp_disabled_active {
                "yes"
            } else {
                "no"
            }
        )),
        Line::from(""),
        Line::from("Adapter registry entries"),
    ];
    for entry in adapters.entries.iter().take(8) {
        lines.push(Line::from(format!(
            "{} | {} | loaded {} | pinned {} | active {} | {} | mtp {}",
            entry.adapter_id,
            entry.adapter_type,
            yes_no(entry.loaded),
            yes_no(entry.pinned),
            yes_no(entry.active),
            bytes(entry.resident_bytes),
            entry.supports_mtp
        )));
        lines.push(Line::from(format!(
            "  targets {} | source {}",
            entry.target_modules.join(","),
            entry.source_path.display()
        )));
    }
    if adapters.entries.is_empty() {
        lines.push(Line::from("no adapters registered"));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(adapters.note.clone()));
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Adapters"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_logs(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let items = state
        .logs
        .iter()
        .rev()
        .take(area.height.saturating_sub(2) as usize)
        .map(|entry| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{} ", entry.level.label()),
                    Style::default().fg(Color::Gray),
                ),
                Span::raw(entry.message.clone()),
            ]))
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(items).block(Block::default().borders(Borders::ALL).title("Logs")),
        area,
    );
}

fn render_mtp(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    let mtp = &state.mtp;
    let lines = vec![
        Line::from(vec![
            Span::styled("Status ", Style::default().fg(Color::Gray)),
            Span::raw(mtp.status.label()),
        ]),
        Line::from(vec![
            Span::styled("Target ", Style::default().fg(Color::Gray)),
            Span::raw(mtp.target.clone()),
        ]),
        Line::from(vec![
            Span::styled("Drafter ", Style::default().fg(Color::Gray)),
            Span::raw(mtp.drafter.clone()),
        ]),
        Line::from(vec![
            Span::styled("Compatibility ", Style::default().fg(Color::Gray)),
            Span::raw(mtp.compatibility.clone()),
        ]),
        Line::from(vec![
            Span::styled("Exactness ", Style::default().fg(Color::Gray)),
            Span::raw(mtp.exactness.clone()),
        ]),
        Line::from(format!(
            "Draft block size {} | attempted {} | accepted {}",
            mtp.draft_block_size, mtp.attempted_draft_tokens, mtp.accepted_draft_tokens
        )),
        Line::from(format!(
            "Acceptance rate {:.1}% | accepted/verify {:.2} | verify passes {}",
            mtp.acceptance_rate * 100.0,
            mtp.accepted_tokens_per_verify,
            mtp.target_verify_passes
        )),
        Line::from(format!("Rollbacks {}", mtp.rollback_count)),
        Line::from(format!(
            "Auto-disable reason {}",
            mtp.auto_disable_reason.as_deref().unwrap_or("none")
        )),
        Line::from(format!(
            "Failing fixture {}",
            mtp.failing_fixture.as_deref().unwrap_or("none")
        )),
        Line::from(format!("Adapter state {}", mtp.adapter_state)),
        Line::from(format!("Active KV {}", mtp.active_kv_mode)),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("MTP"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_help(frame: &mut Frame<'_>, area: Rect) {
    let lines = vec![
        Line::from("Navigation"),
        Line::from("Tab / Shift-Tab  next or previous page"),
        Line::from("1..9             direct page jump"),
        Line::from("?                Help"),
        Line::from("Esc              Dashboard"),
        Line::from(""),
        Line::from("Actions"),
        Line::from("r                refresh provider snapshot"),
        Line::from("v                validate current config"),
        Line::from("b                record benchmark launch surface"),
        Line::from("q / Ctrl-C        graceful quit"),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Help"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

fn render_status(frame: &mut Frame<'_>, area: Rect, state: &AppState) {
    frame.render_widget(
        Paragraph::new(state.status_line.clone()).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

fn buffer_to_string(buffer: &Buffer, width: u16, height: u16) -> String {
    let mut output = String::new();
    for y in 0..height {
        let mut line = String::new();
        for x in 0..width {
            line.push_str(buffer[(x, y)].symbol());
        }
        output.push_str(line.trim_end());
        output.push('\n');
    }
    output
}

fn option_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.1} ms"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn option_tps(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.1} tok/s"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn bytes(value: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if value >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", value as f64 / GIB)
    } else if value >= 1024 * 1024 {
        format!("{:.2} MiB", value as f64 / MIB)
    } else {
        format!("{value} B")
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn status_color(status: crate::config::ValidationStatus) -> Color {
    match status {
        crate::config::ValidationStatus::Pending => Color::Yellow,
        crate::config::ValidationStatus::Valid => Color::Green,
        crate::config::ValidationStatus::Invalid => Color::Red,
    }
}

fn diagnostic_color(severity: DiagnosticSeverity) -> Color {
    match severity {
        DiagnosticSeverity::Info => Color::Blue,
        DiagnosticSeverity::Warning => Color::Yellow,
        DiagnosticSeverity::Error => Color::Red,
    }
}
