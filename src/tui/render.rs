use super::*;

pub(super) fn render(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    app.begin_render();
    frame.render_widget(Clear, area);
    frame
        .buffer_mut()
        .set_style(area, Style::default().fg(TEXT).bg(BACKGROUND));

    if app.landing_visible() && !app.page_open() {
        render_landing(frame, area, app);
        return;
    }

    let regions = ui_regions(area, app);

    if app.model_picker_open() {
        render_model_picker(frame, regions.transcript, app);
    } else if app.provider_editor_open() {
        render_provider_editor(frame, regions.transcript, app);
    } else if app.session_picker_open() {
        render_session_picker(frame, regions.transcript, app);
    } else if app.help_open {
        render_help_page(frame, regions.transcript, app);
    } else {
        render_transcript(frame, regions.transcript, app);
        let completion = align_with_footer_input(regions.completion, regions.footer);
        render_completion(frame, completion, app);
    }
    render_working_line(frame, regions.working, app);
    render_footer(frame, regions.footer, app);
    render_keymap(frame, regions.keymap, app);
}

fn render_working_line(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.is_empty() || !app.busy {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    const FULL: &str = "Working... (Esc)";
    const COMPACT: &str = "Working...";
    let text = if area.width as usize >= FULL.chars().count() + 2 {
        FULL
    } else {
        COMPACT
    };
    let elapsed = app
        .turn_started
        .map(|started| started.elapsed().as_secs_f32())
        .unwrap_or(0.0);
    frame.render_widget(Paragraph::new(working_shimmer_line(text, elapsed)), area);
}

fn lerp_color(from: Color, to: Color, t: f32) -> Color {
    let (Color::Rgb(fr, fg, fb), Color::Rgb(tr, tg, tb)) = (from, to) else {
        return if t < 0.5 { from } else { to };
    };
    let t = t.clamp(0.0, 1.0);
    let blend = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    Color::Rgb(blend(fr, tr), blend(fg, tg), blend(fb, tb))
}

pub(super) fn working_shimmer_line(text: &str, elapsed_secs: f32) -> Line<'static> {
    const BAND: f32 = 5.0;
    const PERIOD_SECS: f32 = 1.2;
    let len = text.chars().count() as f32;
    let progress = (elapsed_secs % PERIOD_SECS) / PERIOD_SECS;
    let center = progress * (len + BAND * 2.0) - BAND;
    let spans = text
        .chars()
        .enumerate()
        .map(|(index, ch)| {
            let distance = index as f32 - center;
            let style = if distance.abs() > BAND / 2.0 {
                Style::default().fg(TEXT_DIM)
            } else if distance.abs() <= 1.0 {
                Style::default()
                    .fg(ACCENT_PRIMARY)
                    .add_modifier(Modifier::BOLD)
            } else if distance > 0.0 {
                // leading edge: solid primary
                Style::default().fg(ACCENT_PRIMARY)
            } else {
                // trailing edge: ease from primary toward secondary
                let t = (distance.abs() - 1.0) / (BAND / 2.0 - 1.0);
                Style::default().fg(lerp_color(ACCENT_PRIMARY, ACCENT_SECONDARY, t))
            };
            Span::styled(ch.to_string(), style)
        })
        .collect::<Vec<_>>();
    Line::from(spans)
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    let status_height = u16::from(area.height > 0);
    render_statusline(
        frame,
        Rect::new(area.x, area.y, area.width, status_height),
        app,
    );
    let input = Rect::new(
        area.x,
        area.y.saturating_add(status_height),
        area.width,
        area.height.saturating_sub(status_height),
    );
    render_input_frame(frame, input, app);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatuslineLayout {
    Full,
    WithoutIdentifier,
    ShortPath,
    WithoutGit,
    WithoutPath,
    WithoutSpeed,
    PointState,
    WithoutState,
    WithoutBrand,
    ShortContext,
}

const STATUSLINE_LADDER: [StatuslineLayout; 10] = [
    StatuslineLayout::Full,
    StatuslineLayout::WithoutIdentifier,
    StatuslineLayout::ShortPath,
    StatuslineLayout::WithoutGit,
    StatuslineLayout::WithoutPath,
    StatuslineLayout::WithoutSpeed,
    StatuslineLayout::PointState,
    StatuslineLayout::WithoutState,
    StatuslineLayout::WithoutBrand,
    StatuslineLayout::ShortContext,
];

struct StatuslineInputs {
    model: String,
    thinking: &'static str,
    cwd: String,
    project: String,
    git: Option<String>,
    identifier: Option<String>,
    speed: Option<String>,
    context: String,
    short_context: String,
    state: String,
}

fn statusline_inputs(app: &App) -> StatuslineInputs {
    let cwd = app.working_dir.display().to_string();
    let project = app
        .working_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(cwd.as_str())
        .to_owned();
    let percent = app.context_chars as f64 * 100.0 / app.max_context_chars.max(1) as f64;
    StatuslineInputs {
        model: model_short_name(&app.model),
        thinking: thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference)),
        cwd,
        project,
        git: app.git_status.as_ref().map(git_status_label),
        identifier: app
            .session_id
            .as_deref()
            .map(short_session_id)
            .or_else(|| app.git_status.as_ref().map(|git| git.commit.as_str()))
            .map(str::to_owned),
        speed: (!app.busy)
            .then_some(app.tokens_per_second)
            .flatten()
            .map(|rate| format!("{rate:.1} tok/s")),
        context: format!(
            "ctx {percent:.1}%/{}",
            format_char_budget(app.max_context_chars)
        ),
        short_context: format!("{percent:.1}%"),
        state: format!(
            "{} {}",
            app.status.symbol(),
            statusline_state_label(app.status)
        ),
    }
}

fn render_statusline(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    register_statusline_hits(app, area);
    frame.render_widget(
        Paragraph::new(statusline_line(app, area.width as usize))
            .style(Style::default().bg(BACKGROUND))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn statusline_line(app: &App, width: usize) -> Line<'static> {
    if width == 0 {
        return Line::default();
    }
    let inputs = statusline_inputs(app);
    for layout in STATUSLINE_LADDER {
        let left = statusline_left_parts(layout, &inputs);
        let right = statusline_right_parts(layout, &inputs, app.status.symbol());
        if statusline_parts_width(&left, &right) <= width {
            return styled_statusline(layout, left, right, width, app.status);
        }
    }

    compact_statusline(&inputs.model, inputs.thinking, &inputs.context, width)
}

fn register_statusline_hits(app: &mut App, area: Rect) {
    if area.width == 0 {
        return;
    }
    let width = area.width as usize;
    let inputs = statusline_inputs(app);
    let wide_separator = UnicodeWidthStr::width("  ·  ") as u16;
    let mut separator_width = wide_separator;
    let mut has_brand = true;
    let mut model = inputs.model.clone();
    let mut fitted = false;
    for layout in STATUSLINE_LADDER {
        let left = statusline_left_parts(layout, &inputs);
        let right = statusline_right_parts(layout, &inputs, app.status.symbol());
        if statusline_parts_width(&left, &right) <= width {
            has_brand = !matches!(
                layout,
                StatuslineLayout::WithoutBrand | StatuslineLayout::ShortContext
            );
            fitted = true;
            break;
        }
    }
    if !fitted {
        separator_width = UnicodeWidthStr::width(" · ") as u16;
        let fixed = UnicodeWidthStr::width("ZEX")
            + UnicodeWidthStr::width(inputs.thinking)
            + UnicodeWidthStr::width(inputs.context.as_str())
            + UnicodeWidthStr::width(" · ") * 3;
        model = truncate_inline(&inputs.model, width.saturating_sub(fixed).max(1));
    }
    let mut x = area.x;
    if has_brand {
        x = x
            .saturating_add(UnicodeWidthStr::width("ZEX") as u16)
            .saturating_add(separator_width);
    }
    let model_width =
        (UnicodeWidthStr::width(model.as_str()) as u16).min(area.right().saturating_sub(x));
    app.register_hit(Rect::new(x, area.y, model_width, 1), HitTarget::StatusModel);
    let thinking_x = x
        .saturating_add(model_width)
        .saturating_add(separator_width);
    let thinking_width = (UnicodeWidthStr::width(inputs.thinking) as u16)
        .min(area.right().saturating_sub(thinking_x));
    app.register_hit(
        Rect::new(thinking_x, area.y, thinking_width, 1),
        HitTarget::StatusThinking,
    );
}

fn statusline_left_parts(layout: StatuslineLayout, inputs: &StatuslineInputs) -> Vec<String> {
    let mut parts = Vec::new();
    if !matches!(
        layout,
        StatuslineLayout::WithoutBrand | StatuslineLayout::ShortContext
    ) {
        parts.push("ZEX".to_owned());
    }
    parts.push(inputs.model.clone());
    parts.push(inputs.thinking.to_owned());
    match layout {
        StatuslineLayout::Full => {
            parts.push(inputs.cwd.clone());
            parts.extend(inputs.git.clone());
            parts.extend(inputs.identifier.clone());
        }
        StatuslineLayout::WithoutIdentifier => {
            parts.push(inputs.cwd.clone());
            parts.extend(inputs.git.clone());
        }
        StatuslineLayout::ShortPath => {
            parts.push(inputs.project.clone());
            parts.extend(inputs.git.clone());
        }
        StatuslineLayout::WithoutGit => parts.push(inputs.project.clone()),
        _ => {}
    }
    parts
}

/// Right side parts as `(text, is_state)`; the state flag keeps the status
/// color on the state field even when it is no longer the last part.
fn statusline_right_parts(
    layout: StatuslineLayout,
    inputs: &StatuslineInputs,
    state_symbol: &str,
) -> Vec<(String, bool)> {
    let mut parts = Vec::new();
    if matches!(
        layout,
        StatuslineLayout::Full
            | StatuslineLayout::WithoutIdentifier
            | StatuslineLayout::ShortPath
            | StatuslineLayout::WithoutGit
            | StatuslineLayout::WithoutPath
    ) {
        parts.extend(inputs.speed.clone().map(|speed| (speed, false)));
    }
    let context = if matches!(layout, StatuslineLayout::ShortContext) {
        inputs.short_context.clone()
    } else {
        inputs.context.clone()
    };
    parts.push((context, false));
    match layout {
        StatuslineLayout::PointState => parts.push((state_symbol.to_owned(), true)),
        StatuslineLayout::WithoutState
        | StatuslineLayout::WithoutBrand
        | StatuslineLayout::ShortContext => {}
        _ => parts.push((inputs.state.clone(), true)),
    }
    parts
}

fn statusline_parts_width(left: &[String], right: &[(String, bool)]) -> usize {
    let separator_width = UnicodeWidthStr::width("  ·  ");
    let left_width = left
        .iter()
        .map(|part| UnicodeWidthStr::width(part.as_str()))
        .sum::<usize>()
        + separator_width.saturating_mul(left.len().saturating_sub(1));
    let right_width = right
        .iter()
        .map(|(part, _)| UnicodeWidthStr::width(part.as_str()))
        .sum::<usize>()
        + separator_width.saturating_mul(right.len().saturating_sub(1));
    left_width + right_width + usize::from(!left.is_empty() && !right.is_empty()) * 2
}

fn styled_statusline(
    layout: StatuslineLayout,
    left: Vec<String>,
    right: Vec<(String, bool)>,
    width: usize,
    status: Status,
) -> Line<'static> {
    let content_width = statusline_parts_width(&left, &right);
    let spacer = width.saturating_sub(content_width);
    let has_brand = !matches!(
        layout,
        StatuslineLayout::WithoutBrand | StatuslineLayout::ShortContext
    );
    let model_index = usize::from(has_brand);
    let separator = Color::Rgb(82, 82, 82);
    let mut spans = Vec::new();
    for (index, part) in left.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled("  ·  ", Style::default().fg(separator)));
        }
        let style = if has_brand && index == 0 {
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD)
        } else if index == model_index || index == model_index + 1 {
            Style::default()
                .fg(ACCENT_PRIMARY)
                .add_modifier(Modifier::UNDERLINED)
        } else {
            Style::default().fg(TEXT_DIM)
        };
        spans.push(Span::styled(part, style));
    }
    if !spans.is_empty() && !right.is_empty() {
        spans.push(Span::raw(" ".repeat(spacer.saturating_add(2))));
    }
    for (index, (part, is_state)) in right.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled("  ·  ", Style::default().fg(separator)));
        }
        let style = if is_state {
            Style::default().fg(status.color())
        } else {
            Style::default().fg(TEXT_DIM)
        };
        spans.push(Span::styled(part, style));
    }
    Line::from(spans)
}

fn compact_statusline(model: &str, thinking: &str, context: &str, width: usize) -> Line<'static> {
    let separator = " · ";
    let fixed = UnicodeWidthStr::width("ZEX")
        + UnicodeWidthStr::width(thinking)
        + UnicodeWidthStr::width(context)
        + UnicodeWidthStr::width(separator) * 3;
    let model = truncate_inline(model, width.saturating_sub(fixed).max(1));
    let separator_style = Style::default().fg(TEXT_DIM);
    Line::from(vec![
        Span::styled(
            "ZEX",
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(separator, separator_style),
        Span::styled(
            model,
            Style::default()
                .fg(ACCENT_PRIMARY)
                .add_modifier(Modifier::UNDERLINED),
        ),
        Span::styled(separator, separator_style),
        Span::styled(
            thinking.to_owned(),
            Style::default()
                .fg(ACCENT_PRIMARY)
                .add_modifier(Modifier::UNDERLINED),
        ),
        Span::styled(separator, separator_style),
        Span::styled(context.to_owned(), Style::default().fg(TEXT_DIM)),
    ])
}

fn model_short_name(model: &str) -> String {
    ModelRef::from_key(model)
        .map(|target| target.model_id)
        .unwrap_or_else(|| model.rsplit('/').next().unwrap_or(model).to_owned())
}

fn thinking_short_name(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "min",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "med",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhi",
        ThinkingLevel::Max => "max",
    }
}

fn git_status_label(git: &GitStatus) -> String {
    format!("{}{}", git.branch, dirty_suffix(git.dirty_count))
}

fn dirty_suffix(count: usize) -> String {
    if count == 0 {
        String::new()
    } else {
        format!(" *{count}")
    }
}

fn format_char_budget(chars: usize) -> String {
    if chars >= 1_000_000 {
        format!("{:.1}Mc", chars as f64 / 1_000_000.0)
    } else if chars >= 1_000 {
        format!("{}Kc", (chars + 500) / 1_000)
    } else {
        format!("{chars}c")
    }
}

fn statusline_state_label(status: Status) -> &'static str {
    match status {
        Status::Idle => "idle",
        Status::Thinking => "thinking",
        Status::RunningTool => "tool",
        Status::Cancelling => "stopping",
        Status::Error => "error",
    }
}

fn render_session_picker(frame: &mut Frame<'_>, viewport: Rect, app: &mut App) {
    let Some(picker) = app.session_picker.clone() else {
        return;
    };
    if viewport.is_empty() {
        return;
    }

    let area = horizontal_inset(viewport, HORIZONTAL_GUTTER);
    let inner_width = area.width.saturating_sub(2) as usize;
    let max_visible = area.height.saturating_sub(3).max(1) as usize / 2;
    let visible_count = picker.sessions.len().clamp(1, max_visible.max(1));
    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                "Session index",
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if picker.sessions.is_empty() {
                    "  Esc close"
                } else {
                    "  ↑↓ select · Enter resume · Esc cancel"
                },
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::default(),
    ];

    if picker.sessions.is_empty() {
        lines.push(Line::from(Span::styled(
            "No saved sessions",
            Style::default().fg(TEXT_DIM),
        )));
    } else {
        let start = picker
            .selected
            .saturating_sub(visible_count.saturating_sub(1));
        for (index, session) in picker
            .sessions
            .iter()
            .enumerate()
            .skip(start)
            .take(visible_count)
        {
            let target = HitTarget::Session(index);
            let selected = index == picker.selected;
            let hovered = app.hovered(&target);
            let armed = app.armed(&target);
            let background = if selected || armed {
                SURFACE_RAISED
            } else if hovered {
                SURFACE_HOVER
            } else {
                BACKGROUND
            };
            let foreground = TEXT_STRONG;
            let secondary = if selected { TEXT } else { TEXT_DIM };
            let marker = if selected { "▌" } else { " " };
            let timestamp = format_session_time(session.updated_at);
            let metadata = format!(
                "{} · {} message{}",
                timestamp,
                session.message_count,
                if session.message_count == 1 { "" } else { "s" }
            );
            let id = short_session_id(&session.id);
            let id_width = UnicodeWidthStr::width(id);
            let metadata_limit = inner_width
                .saturating_sub(id_width.saturating_add(4))
                .max(1);
            let preview_limit = inner_width.saturating_sub(2).max(1);

            lines.push(
                Line::from(vec![
                    Span::styled(format!("{marker} "), Style::default().fg(secondary)),
                    Span::styled(
                        id.to_owned(),
                        Style::default().fg(foreground).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("  ", Style::default().fg(secondary)),
                    Span::styled(
                        single_line(&metadata, metadata_limit),
                        Style::default().fg(secondary),
                    ),
                ])
                .style(Style::default().bg(background)),
            );
            lines.push(
                Line::from(vec![
                    Span::styled("  ", Style::default().fg(secondary)),
                    Span::styled(
                        single_line(&session.preview, preview_limit),
                        Style::default().fg(secondary),
                    ),
                ])
                .style(Style::default().bg(background)),
            );
        }
    }

    if !picker.sessions.is_empty() {
        let start = picker
            .selected
            .saturating_sub(visible_count.saturating_sub(1));
        for offset in 0..visible_count.min(picker.sessions.len().saturating_sub(start)) {
            app.register_hit(
                Rect::new(
                    area.x,
                    area.y.saturating_add(2 + offset as u16 * 2),
                    area.width,
                    2,
                ),
                HitTarget::Session(start + offset),
            );
        }
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
}

fn render_model_picker(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let Some(picker) = app.model_picker.clone() else {
        return;
    };
    if area.is_empty() {
        return;
    }
    let area = horizontal_inset(area, HORIZONTAL_GUTTER);
    let active = app.providers.active_model.clone();
    let current = active
        .as_ref()
        .and_then(|target| app.providers.model(target))
        .map(|(provider, model)| format!("{} / {}", provider.display_name, model.display_name))
        .unwrap_or_else(|| "none".to_owned());
    let mut lines = vec![
        Line::from(vec![
            Span::styled(
                "Models",
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Current: ", Style::default().fg(TEXT_DIM)),
            Span::styled(current, Style::default().fg(TEXT)),
        ]),
        Line::default(),
    ];

    if picker.choices.is_empty() {
        lines.extend([
            Line::from(Span::styled(
                "No configured models",
                Style::default().fg(TEXT_STRONG),
            )),
            Line::from(Span::styled(
                "Add a Provider and at least one model with /provider.",
                Style::default().fg(TEXT_DIM),
            )),
        ]);
    } else {
        let mut provider = "";
        for (index, choice) in picker.choices.iter().enumerate() {
            if choice.provider_name != provider {
                if !provider.is_empty() {
                    lines.push(Line::default());
                }
                provider = &choice.provider_name;
                lines.push(Line::from(Span::styled(
                    provider.to_owned(),
                    Style::default()
                        .fg(ACCENT_PRIMARY)
                        .add_modifier(Modifier::BOLD),
                )));
            }
            let target = HitTarget::Model(index);
            let selected = index == picker.selected;
            let current = active
                .as_ref()
                .is_some_and(|active| *active == choice.target);
            let hovered = app.hovered(&target);
            let armed = app.armed(&target);
            let background = if selected || armed {
                SURFACE_RAISED
            } else if hovered {
                SURFACE_HOVER
            } else {
                BACKGROUND
            };
            let marker = if selected { "▌" } else { " " };
            let current_marker = if current { "●" } else { " " };
            let thinking = choice.thinking.summary();
            let row = if area.width >= 76 {
                Line::from(vec![
                    Span::styled(
                        format!("{marker} {current_marker} "),
                        Style::default().fg(if selected {
                            ACCENT_PRIMARY
                        } else if current {
                            OK
                        } else {
                            TEXT_FAINT
                        }),
                    ),
                    Span::styled(
                        pad_display(&single_line(&choice.model_name, 30), 32),
                        Style::default().fg(TEXT_STRONG),
                    ),
                    Span::styled(
                        pad_display(&single_line(&choice.target.model_id, 28), 30),
                        Style::default().fg(TEXT_DIM),
                    ),
                    Span::styled(format!("think {thinking}"), Style::default().fg(TEXT_DIM)),
                ])
            } else {
                Line::from(vec![
                    Span::styled(
                        format!("{marker} {current_marker} "),
                        Style::default().fg(if selected {
                            ACCENT_PRIMARY
                        } else if current {
                            OK
                        } else {
                            TEXT_FAINT
                        }),
                    ),
                    Span::styled(
                        single_line(&choice.model_name, area.width.saturating_sub(18) as usize),
                        Style::default().fg(TEXT_STRONG),
                    ),
                    Span::styled(
                        format!(" · think {thinking}"),
                        Style::default().fg(TEXT_DIM),
                    ),
                ])
            };
            lines.push(row.style(Style::default().bg(background)));
        }
    }

    let mut y = area.y.saturating_add(2);
    let mut provider = "";
    for (index, choice) in picker.choices.iter().enumerate() {
        if choice.provider_name != provider {
            if !provider.is_empty() {
                y = y.saturating_add(1);
            }
            provider = &choice.provider_name;
            y = y.saturating_add(1);
        }
        if y < area.bottom() {
            app.register_hit(Rect::new(area.x, y, area.width, 1), HitTarget::Model(index));
        }
        y = y.saturating_add(1);
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
}

fn render_provider_editor(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let Some(editor) = app.provider_editor.clone() else {
        return;
    };
    if area.is_empty() {
        return;
    }
    let area = content_area(area);
    let left_width = if area.width < 50 {
        area.width
    } else {
        (area.width / 3).clamp(24, 38)
    };
    let left = Rect::new(area.x, area.y, left_width, area.height);
    let right = Rect::new(
        left.right(),
        area.y,
        area.width.saturating_sub(left_width),
        area.height,
    );
    let left_style = if editor.pane == ProviderPane::Providers {
        Style::default().fg(ACCENT_PRIMARY)
    } else {
        Style::default().fg(TEXT_FAINT)
    };
    let mut provider_lines = vec![
        Line::from(Span::styled(
            "Providers",
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        )),
        Line::default(),
    ];
    if editor.draft.providers.is_empty() {
        provider_lines.push(Line::from(Span::styled(
            "No Providers configured",
            Style::default().fg(TEXT_DIM),
        )));
        provider_lines.push(Line::from(Span::styled(
            "Press n to add one.",
            Style::default().fg(TEXT_DIM),
        )));
    } else {
        for (index, provider) in editor.draft.providers.iter().enumerate() {
            let target = HitTarget::Provider {
                pane: ProviderPane::Providers,
                index,
            };
            let selected = index == editor.provider_selected;
            let hovered = app.hovered(&target);
            let armed = app.armed(&target);
            provider_lines.push(
                Line::from(vec![
                    Span::styled(
                        if selected { "▌ " } else { "  " },
                        Style::default().fg(ACCENT_PRIMARY),
                    ),
                    Span::styled(
                        single_line(
                            &provider.display_name,
                            left_width.saturating_sub(5) as usize,
                        ),
                        Style::default().fg(if selected { TEXT_STRONG } else { TEXT }),
                    ),
                ])
                .style(Style::default().bg(
                    if (selected && editor.pane == ProviderPane::Providers) || armed {
                        SURFACE_RAISED
                    } else if hovered {
                        SURFACE_HOVER
                    } else {
                        BACKGROUND
                    },
                )),
            );
        }
    }
    frame.render_widget(
        Paragraph::new(provider_lines)
            .block(
                Block::default()
                    .borders(Borders::RIGHT)
                    .border_style(left_style),
            )
            .wrap(Wrap { trim: false }),
        left,
    );

    if right.is_empty() {
        return;
    }
    let right = horizontal_inset(right, 2);
    let Some(provider) = editor.selected_provider() else {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    "Provider details",
                    Style::default()
                        .fg(TEXT_STRONG)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::default(),
                Line::from(Span::styled(
                    "Create a Provider to configure its endpoint and models.",
                    Style::default().fg(TEXT_DIM),
                )),
            ]),
            right,
        );
        return;
    };

    for index in 0..editor.draft.providers.len() {
        let y = left.y.saturating_add(2 + index as u16);
        if y < left.bottom() {
            app.register_hit(
                Rect::new(left.x, y, left.width.saturating_sub(1), 1),
                HitTarget::Provider {
                    pane: ProviderPane::Providers,
                    index,
                },
            );
        }
    }

    let detail_active = editor.pane == ProviderPane::Details;
    let model_active = editor.pane == ProviderPane::Models;
    let api_key = if provider.api_key.is_empty() {
        "<required>".to_owned()
    } else {
        "••••••••••••".to_owned()
    };
    let fields = [
        ("ID", provider.id.clone()),
        ("Name", provider.display_name.clone()),
        ("Base URL", provider.base_url.clone()),
        ("API key", api_key),
        ("API", provider.openai_api.to_string()),
    ];
    let mut lines = vec![Line::from(Span::styled(
        "Provider details",
        Style::default()
            .fg(if detail_active {
                ACCENT_PRIMARY
            } else {
                TEXT_STRONG
            })
            .add_modifier(Modifier::BOLD),
    ))];
    for (index, (label, value)) in fields.into_iter().enumerate() {
        let target = HitTarget::Provider {
            pane: ProviderPane::Details,
            index,
        };
        let selected = detail_active && editor.detail_selected == index;
        let hovered = app.hovered(&target);
        let armed = app.armed(&target);
        lines.push(
            Line::from(vec![
                Span::styled(
                    format!("{} {:<10}", if selected { "▌" } else { " " }, label),
                    Style::default().fg(if selected { ACCENT_PRIMARY } else { TEXT_DIM }),
                ),
                Span::styled(single_line(&value, 70), Style::default().fg(TEXT_STRONG)),
            ])
            .style(Style::default().bg(if selected || armed {
                SURFACE_RAISED
            } else if hovered {
                SURFACE_HOVER
            } else {
                BACKGROUND
            })),
        );
    }
    lines.extend([
        Line::default(),
        Line::from(Span::styled(
            "Models",
            Style::default()
                .fg(if model_active {
                    ACCENT_PRIMARY
                } else {
                    TEXT_STRONG
                })
                .add_modifier(Modifier::BOLD),
        )),
    ]);
    if provider.models.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No models · focus Models and press n",
            Style::default().fg(TEXT_DIM),
        )));
    } else {
        for (index, model) in provider.models.iter().enumerate() {
            let target = HitTarget::Provider {
                pane: ProviderPane::Models,
                index,
            };
            let selected = model_active && editor.model_selected == index;
            let hovered = app.hovered(&target);
            let armed = app.armed(&target);
            let capabilities = editor.draft.thinking_capabilities(&ModelRef {
                provider_id: provider.id.clone(),
                model_id: model.id.clone(),
            });
            let thinking = capabilities.summary();
            let map = capabilities
                .reasoning_effort_map
                .iter()
                .filter(|(level, _)| capabilities.supported.contains(level))
                .map(|(level, value)| format!("{level}:{value}"))
                .collect::<Vec<_>>()
                .join(",");
            lines.push(
                Line::from(vec![
                    Span::styled(
                        if selected { "▌ " } else { "  " },
                        Style::default().fg(ACCENT_PRIMARY),
                    ),
                    Span::styled(
                        pad_display(&single_line(&model.display_name, 28), 30),
                        Style::default().fg(TEXT_STRONG),
                    ),
                    Span::styled(
                        pad_display(&single_line(&model.id, 24), 26),
                        Style::default().fg(TEXT_DIM),
                    ),
                    Span::styled(
                        format!("think {thinking}  map {map}"),
                        Style::default().fg(TEXT_DIM),
                    ),
                ])
                .style(Style::default().bg(if selected || armed {
                    SURFACE_RAISED
                } else if hovered {
                    SURFACE_HOVER
                } else {
                    BACKGROUND
                })),
            );
        }
    }
    if let Some(field_editor) = &editor.field_editor {
        lines.extend([
            Line::default(),
            Line::from(Span::styled(
                "Edit value",
                Style::default()
                    .fg(ACCENT_PRIMARY)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                if matches!(
                    field_editor.target,
                    ProviderEditTarget::Provider(ProviderField::ApiKey)
                ) {
                    "•".repeat(field_editor.input.content.chars().count())
                } else {
                    field_editor.input.content.clone()
                },
                Style::default().fg(TEXT_STRONG).bg(SURFACE_RAISED),
            )),
            Line::from(Span::styled(
                "Enter apply field · Esc cancel",
                Style::default().fg(TEXT_DIM),
            )),
        ]);
    }
    if let Some(dialog) = &editor.dialog {
        lines.extend([
            Line::default(),
            Line::from(Span::styled(
                match dialog {
                    ProviderDialog::Delete(_) => "Delete selected item?",
                    ProviderDialog::Discard => "Discard unsaved changes?",
                },
                Style::default().fg(BAD).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "Enter/y confirm · Esc/n cancel",
                Style::default().fg(TEXT_DIM),
            )),
        ]);
    }
    for index in 0..ProviderField::COUNT {
        let y = right.y.saturating_add(1 + index as u16);
        if y < right.bottom() {
            app.register_hit(
                Rect::new(right.x, y, right.width, 1),
                HitTarget::Provider {
                    pane: ProviderPane::Details,
                    index,
                },
            );
        }
    }
    let models_y = right.y.saturating_add(8);
    for index in 0..provider.models.len() {
        let y = models_y.saturating_add(index as u16);
        if y < right.bottom() {
            app.register_hit(
                Rect::new(right.x, y, right.width, 1),
                HitTarget::Provider {
                    pane: ProviderPane::Models,
                    index,
                },
            );
        }
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), right);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct UiRegions {
    pub(super) transcript: Rect,
    pub(super) completion: Rect,
    pub(super) working: Rect,
    pub(super) footer: Rect,
    pub(super) keymap: Rect,
}

pub(super) fn ui_regions(area: Rect, app: &App) -> UiRegions {
    let keymap_height = u16::from(area.height >= 3);
    let working_height = u16::from(app.busy && area.height >= 8);
    let requested_input_rows = if app.busy || app.page_open() {
        1
    } else {
        input_metrics(
            &app.input.content,
            app.input.cursor,
            footer_input_width(area.width)
                .saturating_sub(INPUT_HORIZONTAL_PADDING.saturating_mul(2))
                .max(1) as usize,
        )
        .total_rows
        .clamp(1, MAX_INPUT_ROWS)
    };
    let preferred_footer_height = (requested_input_rows as u16)
        .saturating_add(1)
        .saturating_add(INPUT_VERTICAL_PADDING.saturating_mul(2));
    let footer_height = if area.height
        >= keymap_height
            .saturating_add(working_height)
            .saturating_add(MIN_TRANSCRIPT_HEIGHT)
            .saturating_add(2)
    {
        preferred_footer_height.min(
            area.height
                .saturating_sub(keymap_height)
                .saturating_sub(working_height)
                .saturating_sub(MIN_TRANSCRIPT_HEIGHT),
        )
    } else {
        area.height
            .saturating_sub(keymap_height)
            .saturating_sub(working_height)
            .min(2)
    };
    let fixed_height = footer_height + keymap_height + working_height;
    let remaining = area.height.saturating_sub(fixed_height);
    let transcript_reserve = MIN_TRANSCRIPT_HEIGHT.min(remaining);
    let completion_width = footer_input_width(area.width);
    let completion_height =
        completion_height(app, completion_width).min(remaining.saturating_sub(transcript_reserve));
    let transcript_height = remaining.saturating_sub(completion_height);

    let [transcript, completion, working, footer, keymap] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(transcript_height),
            Constraint::Length(completion_height),
            Constraint::Length(working_height),
            Constraint::Length(footer_height),
            Constraint::Length(keymap_height),
        ])
        .areas(area);

    UiRegions {
        transcript,
        completion,
        working,
        footer,
        keymap,
    }
}

pub(super) fn align_with_footer_input(area: Rect, footer: Rect) -> Rect {
    let gutter = HORIZONTAL_GUTTER.min(footer.width / 2);
    Rect::new(
        footer.x.saturating_add(gutter),
        area.y,
        footer.width.saturating_sub(gutter.saturating_mul(2)),
        area.height,
    )
}

fn footer_input_width(width: u16) -> u16 {
    let gutter = HORIZONTAL_GUTTER.min(width / 2);
    width.saturating_sub(gutter.saturating_mul(2))
}

fn horizontal_inset(area: Rect, amount: u16) -> Rect {
    if area.width <= amount.saturating_mul(2) {
        Rect::new(area.x, area.y, 0, area.height)
    } else {
        Rect::new(
            area.x + amount,
            area.y,
            area.width - amount.saturating_mul(2),
            area.height,
        )
    }
}

fn render_transcript(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        app.transcript_page_height = 0;
        app.max_scroll = 0;
        app.scroll_top = 0;
        return;
    }
    let area = content_area(area);
    let area = Rect::new(
        area.x,
        area.y,
        area.width,
        area.height
            .saturating_sub(u16::from(area.height > MIN_TRANSCRIPT_HEIGHT)),
    );
    if app.transcript.is_empty() {
        app.transcript_page_height = area.height as usize;
        app.max_scroll = 0;
        app.scroll_top = 0;
        return;
    }

    app.register_hit(area, HitTarget::Transcript);
    let transcript = transcript_text(app, area.width as usize);
    let paragraph = Paragraph::new(transcript.text)
        .style(Style::default().fg(TEXT))
        .wrap(Wrap { trim: false });
    let line_count = paragraph.line_count(area.width.max(1));
    app.transcript_page_height = area.height as usize;
    app.max_scroll = line_count.saturating_sub(area.height as usize);
    if app.follow_output {
        app.scroll_top = app.max_scroll;
    } else {
        app.scroll_top = app.scroll_top.min(app.max_scroll);
    }

    let paragraph = paragraph.scroll((app.scroll_top.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(paragraph, area);
    for (index, line) in transcript.card_lines {
        let Some(y) = line
            .checked_sub(app.scroll_top)
            .and_then(|line| u16::try_from(line).ok())
            .filter(|line| *line < area.height)
            .map(|line| area.y.saturating_add(line))
        else {
            continue;
        };
        app.register_hit(Rect::new(area.x, y, area.width, 1), HitTarget::Card(index));
    }
    for (index, line) in transcript.output_lines {
        let Some(y) = line
            .checked_sub(app.scroll_top)
            .and_then(|line| u16::try_from(line).ok())
            .filter(|line| *line < area.height)
            .map(|line| area.y.saturating_add(line))
        else {
            continue;
        };
        app.register_hit(
            Rect::new(area.x, y, area.width, 1),
            HitTarget::ToolOutput(index),
        );
    }

    if !app.follow_output && app.max_scroll > 0 {
        let indicator = format!(
            " {}–{} / {} ",
            app.scroll_top.saturating_add(1),
            app.scroll_top
                .saturating_add(app.transcript_page_height)
                .min(line_count),
            line_count
        );
        let width = indicator.chars().count().min(area.width as usize) as u16;
        let x = area.right().saturating_sub(width);
        frame.render_widget(
            Paragraph::new(indicator).style(Style::default().fg(TEXT_DIM)),
            Rect::new(x, area.y, width, 1),
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LandingRegions {
    pub(super) brand: Rect,
    pub(super) card: Rect,
    pub(super) hint: Rect,
    pub(super) status: Rect,
}

pub(super) fn landing_regions(area: Rect, app: &App) -> LandingRegions {
    if area.is_empty() {
        return LandingRegions {
            brand: area,
            card: area,
            hint: area,
            status: area,
        };
    }

    let status_height = u16::from(area.height >= 4);
    let status = Rect::new(
        area.x,
        area.bottom().saturating_sub(status_height),
        area.width,
        status_height,
    );
    let stage = Rect::new(
        area.x,
        area.y,
        area.width,
        area.height.saturating_sub(status_height),
    );
    let card_width = landing_card_width(area.width);
    let input_rows = input_metrics(
        &app.input.content,
        app.input.cursor,
        card_width.saturating_sub(7).max(1) as usize,
    )
    .total_rows
    .clamp(1, MAX_INPUT_ROWS) as u16;
    let card_height = match stage.height {
        11.. => input_rows.saturating_add(4).max(5),
        5..=10 => 3,
        1..=4 => 1,
        _ => 0,
    };
    let brand_height = if stage.height >= 15 {
        LANDING_LOGO_ROWS.len() as u16
    } else {
        0
    };
    let brand_gap = if brand_height == 0 {
        0
    } else if stage.height >= 14 {
        2
    } else {
        1
    };
    let hint_height = u16::from(stage.height >= 7);
    let hint_gap = u16::from(hint_height > 0 && stage.height >= 10);
    let group_height = brand_height + brand_gap + card_height + hint_gap + hint_height;
    let group_y = stage.y + stage.height.saturating_sub(group_height) / 2;
    let card_x = area.x + area.width.saturating_sub(card_width) / 2;
    let brand = Rect::new(area.x, group_y, area.width, brand_height);
    let card_y = brand.bottom().saturating_add(brand_gap);
    let card = Rect::new(card_x, card_y, card_width, card_height);
    let hint = Rect::new(
        card.x,
        card.bottom().saturating_add(hint_gap),
        card.width,
        hint_height,
    );

    LandingRegions {
        brand,
        card,
        hint,
        status,
    }
}

fn render_landing(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let regions = landing_regions(area, app);

    if !regions.brand.is_empty() {
        for (row, logo_row) in LANDING_LOGO_ROWS.iter().enumerate() {
            frame.render_widget(
                Paragraph::new(landing_logo_line(logo_row)).alignment(Alignment::Center),
                Rect::new(
                    regions.brand.x,
                    regions.brand.y + row as u16,
                    regions.brand.width,
                    1,
                ),
            );
        }
    }

    render_landing_card(frame, regions.card, app);

    if !regions.hint.is_empty() {
        let hint = if regions.hint.width >= 62 {
            "Enter send  ·  Shift+Enter line break  ·  / actions"
        } else {
            "Enter send  ·  / actions"
        };
        let hint_area = Rect::new(
            regions.hint.x.saturating_add(4),
            regions.hint.y,
            regions.hint.width.saturating_sub(7),
            regions.hint.height,
        );
        frame.render_widget(
            Paragraph::new(Span::styled(hint, Style::default().fg(TEXT_FAINT)))
                .alignment(Alignment::Left),
            hint_area,
        );
    }

    render_landing_status(frame, regions.status, app);

    if app.completion_open() && !regions.card.is_empty() {
        let desired = completion_height(app, regions.card.width);
        let available = regions.card.y.saturating_sub(area.y);
        let height = desired.min(available);
        let completion = Rect::new(
            regions.card.x,
            regions.card.y.saturating_sub(height),
            regions.card.width,
            height,
        );
        render_completion(frame, completion, app);
    }
}

fn landing_logo_line(row: &str) -> Line<'static> {
    let width = UnicodeWidthStr::width(row).saturating_sub(1).max(1);
    Line::from(
        row.char_indices()
            .map(|(column, character)| {
                let position = UnicodeWidthStr::width(&row[..column]) as f32 / width as f32;
                let color = if position < 0.34 {
                    lerp_color(LANDING_LOGO_DARK, TEXT_DIM, position / 0.34)
                } else if position < 0.72 {
                    lerp_color(TEXT_DIM, TEXT_STRONG, (position - 0.34) / 0.38)
                } else {
                    lerp_color(TEXT_STRONG, TEXT_DIM, (position - 0.72) / 0.28)
                };
                Span::styled(
                    character.to_string(),
                    Style::default().fg(color).bg(BACKGROUND),
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn landing_card_width(width: u16) -> u16 {
    if width == 0 {
        return 0;
    }
    let max_width = width.saturating_sub(2).max(1);
    let target = (u32::from(width) * 11 / 20) as u16;
    target.clamp(max_width.min(32), max_width.min(66))
}

fn render_landing_card(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    app.register_hit(area, HitTarget::Input);
    frame.render_widget(
        Block::default()
            .borders(Borders::NONE)
            .style(Style::default().bg(SURFACE)),
        area,
    );
    if area.width <= 3 || area.height == 0 {
        return;
    }
    let inner_rows = area.height;
    for row in 0..inner_rows {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "▌",
                Style::default().fg(ACCENT_PRIMARY).bg(SURFACE),
            )),
            Rect::new(area.x, area.y + row, 1, 1),
        );
    }
    let full_layout = inner_rows >= 5;
    let model_row = u16::from(inner_rows >= 2);
    let metadata_y = area
        .bottom()
        .saturating_sub(if full_layout { 2 } else { 1 });
    let editor_y = area.y.saturating_add(u16::from(full_layout));
    let editor_bottom = if model_row == 1 {
        metadata_y.saturating_sub(u16::from(full_layout))
    } else {
        area.bottom()
    };
    let editor_area = Rect::new(
        area.x.saturating_add(4),
        editor_y,
        area.width.saturating_sub(7),
        editor_bottom.saturating_sub(editor_y),
    );
    if model_row == 1 && area.width > 7 {
        let model = model_short_name(&app.model);
        let thinking = thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference));
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    model,
                    Style::default()
                        .fg(ACCENT_PRIMARY)
                        .bg(SURFACE)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("    ", Style::default().fg(TEXT_FAINT).bg(SURFACE)),
                Span::styled("ZEX / ", Style::default().fg(TEXT_FAINT).bg(SURFACE)),
                Span::styled(thinking, Style::default().fg(TEXT_DIM).bg(SURFACE)),
            ]))
            .style(Style::default().bg(SURFACE)),
            Rect::new(editor_area.x, metadata_y, editor_area.width, 1),
        );
    }
    if !editor_area.is_empty() {
        let placeholder = if editor_area.width >= 40 {
            Line::from(vec![
                Span::styled("Ask anything...", Style::default().fg(TEXT_DIM).bg(SURFACE)),
                Span::styled(
                    "  “Explain this repo”",
                    Style::default().fg(TEXT_FAINT).bg(SURFACE),
                ),
            ])
        } else {
            Line::from(Span::styled(
                "Ask anything...",
                Style::default().fg(TEXT_DIM).bg(SURFACE),
            ))
        };
        render_input_buffer(frame, editor_area, app, "", Some(placeholder), SURFACE);
    }
}

fn render_landing_status(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let area = horizontal_inset(area, HORIZONTAL_GUTTER);
    if area.is_empty() {
        return;
    }
    let version = format!("v{}", env!("CARGO_PKG_VERSION"));
    let version_width = UnicodeWidthStr::width(version.as_str()).min(area.width as usize) as u16;
    let version_area = Rect::new(
        area.right().saturating_sub(version_width),
        area.y,
        version_width,
        1,
    );
    let path_area = Rect::new(
        area.x,
        area.y,
        version_area.x.saturating_sub(area.x).saturating_sub(2),
        1,
    );
    let path = truncate_inline(
        &app.working_dir.display().to_string(),
        path_area.width as usize,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(path, Style::default().fg(TEXT_FAINT)))
            .style(Style::default().bg(BACKGROUND)),
        path_area,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            version,
            Style::default().fg(TEXT_FAINT).bg(BACKGROUND),
        ))
        .alignment(Alignment::Right),
        version_area,
    );
}

fn completion_height(app: &App, width: u16) -> u16 {
    if app.page_open() {
        return 0;
    }
    if app.completion_open() {
        let matches = app.completion_matches();
        let usage_width = matches
            .iter()
            .map(|command| UnicodeWidthStr::width(command.usage))
            .max()
            .unwrap_or(0);
        let inner_width = width.saturating_sub(4).max(1) as usize;
        matches
            .iter()
            .map(|command| {
                let wide = inner_width.saturating_sub(2) >= usage_width + 2 + 18;
                let line_width = if wide {
                    2 + usage_width + 2 + UnicodeWidthStr::width(command.description)
                } else {
                    2 + UnicodeWidthStr::width(command.usage)
                        + 3
                        + UnicodeWidthStr::width(command.description)
                };
                line_width.div_ceil(inner_width).max(1) as u16
            })
            .sum::<u16>()
            .saturating_add(2)
    } else {
        0
    }
}

fn render_completion(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.height == 0 {
        return;
    }
    let matches = app.completion_matches();
    let inner_width = area.width.saturating_sub(2) as usize;
    let usage_width = matches
        .iter()
        .map(|command| UnicodeWidthStr::width(command.usage))
        .max()
        .unwrap_or(0);
    let lines = matches
        .iter()
        .take(area.height.saturating_sub(2) as usize)
        .enumerate()
        .map(|(index, command)| {
            let target = HitTarget::Completion(index);
            let selected = index == app.completion.selected;
            let hovered = app.hovered(&target);
            let armed = app.armed(&target);
            let marker = if selected { "›" } else { " " };
            let available = inner_width.saturating_sub(2);
            let wide = available >= usage_width + 2 + 18;
            let command_style = Style::default()
                .fg(if selected { TEXT_STRONG } else { TEXT })
                .bg(if selected || armed {
                    SURFACE_RAISED
                } else if hovered {
                    SURFACE_HOVER
                } else {
                    SURFACE
                })
                .add_modifier(if selected {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                });
            let marker_style = Style::default()
                .fg(if selected { ACCENT_PRIMARY } else { TEXT_FAINT })
                .bg(if selected || armed {
                    SURFACE_RAISED
                } else if hovered {
                    SURFACE_HOVER
                } else {
                    SURFACE
                });
            let description_style = Style::default()
                .fg(if selected { TEXT } else { TEXT_DIM })
                .bg(if selected || armed {
                    SURFACE_RAISED
                } else if hovered {
                    SURFACE_HOVER
                } else {
                    SURFACE
                });
            let row_style = Style::default().bg(if selected || armed {
                SURFACE_RAISED
            } else if hovered {
                SURFACE_HOVER
            } else {
                SURFACE
            });
            if wide {
                Line::from(vec![
                    Span::styled(format!("{marker} "), marker_style),
                    Span::styled(pad_display(command.usage, usage_width), command_style),
                    Span::raw("  "),
                    Span::styled(command.description, description_style),
                ])
                .style(row_style)
            } else {
                Line::from(vec![
                    Span::styled(format!("{marker} "), marker_style),
                    Span::styled(command.usage, command_style),
                    Span::styled(" · ", description_style),
                    Span::styled(command.description, description_style),
                ])
                .style(row_style)
            }
        })
        .collect::<Vec<_>>();
    let inner = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(TEXT_FAINT))
        .padding(ratatui::widgets::Padding::horizontal(1))
        .inner(area);
    for index in 0..matches.len().min(inner.height as usize) {
        app.register_hit(
            Rect::new(
                inner.x,
                inner.y.saturating_add(index as u16),
                inner.width,
                1,
            ),
            HitTarget::Completion(index),
        );
    }
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::TOP)
                    .border_style(Style::default().fg(TEXT_FAINT))
                    .padding(ratatui::widgets::Padding::horizontal(1)),
            )
            .style(Style::default().bg(SURFACE))
            .wrap(Wrap { trim: false }),
        area,
    );
}

struct TranscriptRender {
    text: Text<'static>,
    card_lines: Vec<(usize, usize)>,
    output_lines: Vec<(usize, usize)>,
}

fn transcript_text(app: &App, width: usize) -> TranscriptRender {
    let mut lines = Vec::new();
    let mut card_lines = Vec::new();
    let mut output_lines = Vec::new();
    for (index, entry) in app.transcript.iter().enumerate() {
        let next = app.transcript.get(index + 1);
        match entry {
            TranscriptEntry::Message { role, content } => {
                let final_answer = *role == MessageRole::Assistant
                    && index > 0
                    && matches!(
                        app.transcript.get(index - 1),
                        Some(TranscriptEntry::Turn(_))
                    );
                append_markdown_lines(&mut lines, content, *role, final_answer, width);
            }
            TranscriptEntry::Thinking(thinking) => {
                if app.show_thinking {
                    card_lines.push((index, lines.len()));
                    let status = if app.busy
                        && app.status == Status::Thinking
                        && index + 1 == app.transcript.len()
                    {
                        ThinkingStatus::Active
                    } else {
                        ThinkingStatus::Done
                    };
                    let level = app.thinking_level.unwrap_or(app.thinking_preference);
                    append_thinking_lines(
                        &mut lines,
                        thinking,
                        status,
                        level,
                        app.selected_card == Some(index),
                        app.hovered(&HitTarget::Card(index)),
                        width,
                    );
                }
            }
            TranscriptEntry::Tool(tool) => {
                card_lines.push((index, lines.len()));
                append_tool_lines(
                    &mut lines,
                    tool,
                    app.selected_card == Some(index),
                    app.hovered(&HitTarget::Card(index)),
                    width,
                );
                if tool.expanded && tool_output_body(tool).len() > TOOL_OUTPUT_PREVIEW_LINES {
                    output_lines.push((index, lines.len().saturating_sub(2)));
                }
            }
            TranscriptEntry::Error {
                summary,
                detail,
                expanded,
            } => {
                lines.push(Line::from(vec![
                    Span::styled("  × ", Style::default().fg(BAD)),
                    Span::styled(
                        summary.clone(),
                        Style::default()
                            .fg(TEXT_STRONG)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        if *expanded {
                            "  Ctrl+E hide"
                        } else {
                            "  Ctrl+E details"
                        },
                        Style::default().fg(TEXT_DIM),
                    ),
                ]));
                if *expanded {
                    for source_line in detail.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("    │ ", Style::default().fg(TEXT_FAINT)),
                            Span::styled(source_line.to_owned(), Style::default().fg(TEXT_DIM)),
                        ]));
                    }
                }
                lines.push(Line::default());
            }
            TranscriptEntry::Turn(turn) => {
                append_turn_line(&mut lines, turn, width);
            }
            TranscriptEntry::Sessions(sessions) => {
                append_session_lines(&mut lines, sessions);
            }
        }
        append_transcript_gap(&mut lines, entry, next);
    }
    if app.busy {
        if !lines.is_empty() && !lines.last().is_some_and(|line| line.spans.is_empty()) {
            lines.push(Line::default());
        }
        append_running_turn_line(&mut lines, app, width);
    }

    TranscriptRender {
        text: Text::from(lines),
        card_lines,
        output_lines,
    }
}

fn append_transcript_gap(
    lines: &mut Vec<Line<'static>>,
    entry: &TranscriptEntry,
    next: Option<&TranscriptEntry>,
) {
    let Some(next) = next else {
        return;
    };
    let compact_tool_sequence = matches!(
        (entry, next),
        (
            TranscriptEntry::Tool(ToolEntry {
                expanded: false,
                ..
            }),
            TranscriptEntry::Tool(_)
        ) | (TranscriptEntry::Thinking(_), TranscriptEntry::Tool(_))
    );
    if !compact_tool_sequence {
        lines.push(Line::default());
    }
}

fn render_help_page(frame: &mut Frame<'_>, viewport: Rect, app: &mut App) {
    if viewport.is_empty() {
        return;
    }

    let area = horizontal_inset(viewport, HORIZONTAL_GUTTER);
    let inner_width = area.width as usize;
    let lines = help_lines(app, inner_width);
    for index in 0..command_specs().len() {
        let y = area.y.saturating_add(1 + index as u16);
        if y < area.bottom() {
            app.register_hit(Rect::new(area.x, y, area.width, 1), HitTarget::Help(index));
        }
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), area);
}

fn help_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let usage_width = command_specs()
        .iter()
        .map(|command| command.usage.len())
        .max()
        .unwrap_or(0);
    let wide = width >= usage_width + 2 + 24;
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Command index",
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "  ↑↓ select · Enter run · Esc close",
            Style::default().fg(TEXT_DIM),
        ),
    ])];
    lines.extend(command_specs().iter().enumerate().map(|(index, command)| {
        let target = HitTarget::Help(index);
        let selected = app.help_selected == index;
        let hovered = app.hovered(&target);
        let armed = app.armed(&target);
        let background = if selected || armed {
            SURFACE_RAISED
        } else if hovered {
            SURFACE_HOVER
        } else {
            BACKGROUND
        };
        let marker = if selected { "▌ " } else { "  " };
        if wide {
            Line::from(vec![
                Span::styled(marker, Style::default().fg(ACCENT_PRIMARY)),
                Span::styled(
                    format!("{:<usage_width$}", command.usage),
                    Style::default().fg(if selected {
                        TEXT_STRONG
                    } else {
                        ACCENT_PRIMARY
                    }),
                ),
                Span::raw("  "),
                Span::styled(command.description, Style::default().fg(TEXT_DIM)),
            ])
            .style(Style::default().bg(background))
        } else {
            Line::from(vec![
                Span::styled(marker, Style::default().fg(ACCENT_PRIMARY)),
                Span::styled(
                    command.usage,
                    Style::default().fg(if selected {
                        TEXT_STRONG
                    } else {
                        ACCENT_PRIMARY
                    }),
                ),
            ])
            .style(Style::default().bg(background))
        }
    }));
    lines
}

fn append_session_lines(
    lines: &mut Vec<Line<'static>>,
    sessions: &[crate::session::SessionSummary],
) {
    if sessions.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("  · ", Style::default().fg(ACCENT_PRIMARY)),
            Span::styled("No saved sessions.", Style::default().fg(TEXT_DIM)),
        ]));
        return;
    }

    lines.push(Line::from(Span::styled(
        format!("  Saved sessions ({})", sessions.len()),
        Style::default()
            .fg(TEXT_STRONG)
            .add_modifier(Modifier::BOLD),
    )));
    for session in sessions {
        lines.push(Line::from(Span::styled(
            format!("  {}", session.id),
            Style::default()
                .fg(ACCENT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            format!(
                "    {} message{} · {}",
                session.message_count,
                if session.message_count == 1 { "" } else { "s" },
                session.preview
            ),
            Style::default().fg(TEXT_DIM),
        )));
    }
}

fn append_turn_line(lines: &mut Vec<Line<'static>>, turn: &TurnEntry, width: usize) {
    let (marker, label, color) = match turn.outcome {
        TurnOutcome::Done => ("✔", "turn done", OK),
        TurnOutcome::Failed => ("✗", "turn failed", BAD),
        TurnOutcome::Stopped => ("×", "turn stopped", TEXT_DIM),
    };
    let mut details = vec![format!(
        "{} tool{}",
        turn.tool_count,
        if turn.tool_count == 1 { "" } else { "s" }
    )];
    if let Some(elapsed) = turn.elapsed {
        details.push(format_turn_duration(elapsed));
    }
    if let Some(tokens) = turn.output_tokens {
        details.push(format_compact_count(tokens));
    }
    let prefix = format!("{label}  ");
    let details = details.join("  ·  ");
    let details_width = width.saturating_sub(4 + UnicodeWidthStr::width(prefix.as_str()));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            format!("{marker} "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(prefix, Style::default().fg(TEXT_DIM)),
        Span::styled(
            truncate_display(&details, details_width),
            Style::default().fg(TEXT_FAINT),
        ),
    ]));
}

fn append_running_turn_line(lines: &mut Vec<Line<'static>>, app: &App, width: usize) {
    let elapsed = app
        .turn_started
        .map(|started| format_turn_duration(started.elapsed()))
        .unwrap_or_else(|| "0s".to_owned());
    let details = [
        format!(
            "{} tool{}",
            app.turn_tool_count,
            if app.turn_tool_count == 1 { "" } else { "s" }
        ),
        elapsed,
    ]
    .join("  ·  ");
    let prefix = "running  ";
    let details_width = width.saturating_sub(4 + UnicodeWidthStr::width(prefix));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            "◌ ",
            Style::default()
                .fg(ACCENT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(prefix, Style::default().fg(TEXT_DIM)),
        Span::styled(
            truncate_display(&details, details_width),
            Style::default().fg(TEXT_FAINT),
        ),
    ]));
}

fn format_turn_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn format_compact_count(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn append_markdown_lines(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    role: MessageRole,
    final_answer: bool,
    width: usize,
) {
    let base_color = match role {
        MessageRole::User => TEXT_STRONG,
        MessageRole::Assistant if final_answer => TEXT,
        MessageRole::Assistant => TEXT_DIM,
    };
    let mut in_code_block = false;
    let mut first_visual_line = true;

    for source_line in content.split('\n') {
        let trimmed = source_line.trim_start();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            let language = trimmed.trim_start_matches('`').trim();
            if in_code_block {
                let rail = message_rail(role, final_answer, &mut first_visual_line);
                lines.push(
                    Line::from(vec![
                        Span::styled(rail, Style::default().fg(message_rail_color(role))),
                        Span::styled("▌ ", Style::default().fg(TEXT_DIM)),
                        Span::styled(
                            if language.is_empty() {
                                "code".to_owned()
                            } else {
                                language.to_owned()
                            },
                            Style::default().fg(TEXT_DIM),
                        ),
                    ])
                    .style(Style::default().bg(SURFACE)),
                );
            } else {
                let rail = message_rail(role, final_answer, &mut first_visual_line);
                lines.push(
                    Line::from(vec![
                        Span::styled(rail, Style::default().fg(message_rail_color(role))),
                        Span::styled("▌", Style::default().fg(TEXT_FAINT)),
                    ])
                    .style(Style::default().bg(SURFACE)),
                );
            }
            continue;
        }
        if in_code_block {
            let content_width = width.saturating_sub(4).max(1);
            for segment in wrap_display_hard(source_line, content_width) {
                let rail = message_rail(role, final_answer, &mut first_visual_line);
                lines.push(
                    Line::from(vec![
                        Span::styled(rail, Style::default().fg(message_rail_color(role))),
                        Span::styled("▌ ", Style::default().fg(TEXT_DIM)),
                        Span::styled(segment, Style::default().fg(TEXT_STRONG)),
                    ])
                    .style(Style::default().bg(SURFACE)),
                );
            }
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("### ") {
            append_wrapped_message_lines(
                lines,
                heading,
                role,
                final_answer,
                &mut first_visual_line,
                width,
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            );
        } else if let Some(heading) = trimmed
            .strip_prefix("## ")
            .or_else(|| trimmed.strip_prefix("# "))
        {
            append_wrapped_message_lines(
                lines,
                heading,
                role,
                final_answer,
                &mut first_visual_line,
                width,
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            );
        } else if let Some(item) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            append_output_block_lines(
                lines,
                item,
                "• ",
                role,
                &mut first_visual_line,
                width,
                base_color,
            );
        } else if let Some((number, item)) = numbered_list_item(trimmed) {
            append_output_block_lines(
                lines,
                item,
                &format!("{number}. "),
                role,
                &mut first_visual_line,
                width,
                base_color,
            );
        } else if let Some(quote) = trimmed.strip_prefix("> ") {
            append_output_block_lines(
                lines,
                quote,
                "│ ",
                role,
                &mut first_visual_line,
                width,
                TEXT_DIM,
            );
        } else if source_line.is_empty() {
            let rail = message_rail(role, final_answer, &mut first_visual_line);
            lines.push(Line::from(Span::styled(
                rail,
                Style::default().fg(message_rail_color(role)),
            )));
        } else {
            append_wrapped_message_lines(
                lines,
                source_line,
                role,
                final_answer,
                &mut first_visual_line,
                width,
                Style::default().fg(base_color),
            );
        }
    }
}

fn message_rail(
    role: MessageRole,
    final_answer: bool,
    first_visual_line: &mut bool,
) -> &'static str {
    let rail = match (role, final_answer, *first_visual_line) {
        (MessageRole::Assistant, true, _) => "  ",
        (MessageRole::User, _, true) => "▎  ",
        (MessageRole::User, _, false) => "▎  ",
        (MessageRole::Assistant, false, true) => "  ",
        (MessageRole::Assistant, false, false) => "  ",
    };
    *first_visual_line = false;
    rail
}

fn message_rail_color(role: MessageRole) -> Color {
    match role {
        MessageRole::User => ACCENT_PRIMARY,
        MessageRole::Assistant => TEXT_FAINT,
    }
}

fn append_wrapped_message_lines(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    role: MessageRole,
    final_answer: bool,
    first_visual_line: &mut bool,
    width: usize,
    content_style: Style,
) {
    let rail_width = match role {
        MessageRole::User => 3,
        MessageRole::Assistant => 2,
    };
    for segment in wrap_display_words(content, width.saturating_sub(rail_width).max(1)) {
        let rail = message_rail(role, final_answer, first_visual_line);
        lines.push(Line::from(vec![
            Span::styled(rail, Style::default().fg(message_rail_color(role))),
            Span::styled(segment, content_style),
        ]));
    }
}

fn append_output_block_lines(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    marker: &str,
    role: MessageRole,
    first_visual_line: &mut bool,
    width: usize,
    content_color: Color,
) {
    let marker_width = UnicodeWidthStr::width(marker);
    let rail_width = match role {
        MessageRole::User => 3,
        MessageRole::Assistant => 2,
    };
    let content_width = width.saturating_sub(rail_width + marker_width).max(1);
    for (index, segment) in wrap_display_words(content, content_width)
        .into_iter()
        .enumerate()
    {
        let rail = message_rail(role, false, first_visual_line);
        lines.push(Line::from(vec![
            Span::styled(rail, Style::default().fg(message_rail_color(role))),
            Span::styled(
                if index == 0 {
                    marker.to_owned()
                } else {
                    " ".repeat(marker_width)
                },
                Style::default().fg(TEXT_DIM),
            ),
            Span::styled(segment, Style::default().fg(content_color)),
        ]));
    }
}

fn append_thinking_lines(
    lines: &mut Vec<Line<'static>>,
    thinking: &ThinkingEntry,
    status: ThinkingStatus,
    level: ThinkingLevel,
    selected: bool,
    hovered: bool,
    width: usize,
) {
    let fold = if thinking.expanded { "▼" } else { "▶" };
    let rail_color = if selected {
        ACCENT_PRIMARY
    } else {
        ACCENT_SECONDARY
    };
    let header_style = if selected {
        Style::default().bg(SURFACE_RAISED)
    } else if hovered {
        Style::default().bg(SURFACE_HOVER)
    } else {
        Style::default().bg(SURFACE)
    };
    let (state, state_color) = match status {
        ThinkingStatus::Active => ("…", ACCENT_PRIMARY),
        ThinkingStatus::Done => ("done", TEXT_DIM),
    };
    let marker = fold;
    let mut header = vec![
        Span::styled(format!("{marker} "), Style::default().fg(rail_color)),
        Span::styled(
            pad_display("thinking", 8),
            Style::default()
                .fg(ACCENT_SECONDARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default().fg(TEXT_FAINT)),
        Span::styled(
            pad_display(&level.to_string(), 7),
            Style::default().fg(TEXT_DIM),
        ),
        Span::styled("  ", Style::default().fg(TEXT_FAINT)),
        Span::styled(state, Style::default().fg(state_color)),
    ];
    if !thinking.expanded {
        let used = 2 + 8 + 2 + 7 + 2 + UnicodeWidthStr::width(state) + 2;
        header.extend([
            Span::styled("  ", Style::default().fg(TEXT_FAINT)),
            Span::styled(
                truncate_display(
                    &single_line(&thinking.content, 240),
                    width.saturating_sub(used),
                ),
                Style::default().fg(TEXT_DIM),
            ),
        ]);
    }
    lines.push(Line::from(header).style(header_style));

    if thinking.expanded {
        for line in thinking.content.split('\n') {
            lines.push(
                Line::from(vec![
                    Span::styled("  │ ", Style::default().fg(rail_color)),
                    Span::styled(line.to_owned(), Style::default().fg(TEXT_DIM)),
                ])
                .style(Style::default().bg(SURFACE)),
            );
        }
        lines.push(
            Line::from(vec![
                Span::styled("  └ ", Style::default().fg(rail_color)),
                Span::styled(
                    format!(
                        "{} lines · Ctrl+O collapse",
                        thinking.content.lines().count().max(1)
                    ),
                    Style::default().fg(TEXT_FAINT),
                ),
            ])
            .style(Style::default().bg(SURFACE)),
        );
    }
    lines.push(Line::default());
}

mod tool;

use tool::append_tool_lines;

pub(super) use tool::tool_detail_line_count;

fn render_input_frame(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    app.register_hit(area, HitTarget::Input);
    let border = if app.input_focused && !app.page_open() && !app.busy {
        ACCENT_PRIMARY
    } else if app.hovered(&HitTarget::Input) && !app.page_open() && !app.busy {
        TEXT_DIM
    } else {
        TEXT_FAINT
    };
    let border_style = Style::default().fg(border).bg(BACKGROUND);
    frame.render_widget(
        Block::default()
            .borders(Borders::LEFT | Borders::RIGHT)
            .border_style(border_style)
            .style(Style::default().bg(BACKGROUND)),
        area,
    );
    if area.width > 2 {
        let shoulder = if app.input_focused && !app.page_open() && !app.busy {
            ACCENT_PRIMARY
        } else {
            TEXT_FAINT
        };
        frame.render_widget(
            Paragraph::new(Span::styled(
                "─",
                Style::default().fg(shoulder).bg(BACKGROUND),
            )),
            Rect::new(area.x + 1, area.y, 1, 1),
        );
        frame.render_widget(
            Paragraph::new(Span::styled(
                "─",
                Style::default().fg(shoulder).bg(BACKGROUND),
            )),
            Rect::new(area.right().saturating_sub(2), area.y, 1, 1),
        );
    }
    if area.width <= 2 || area.height == 0 {
        return;
    }
    let input_area = Rect::new(
        area.x.saturating_add(INPUT_HORIZONTAL_PADDING),
        area.y.saturating_add(INPUT_VERTICAL_PADDING),
        area.width
            .saturating_sub(INPUT_HORIZONTAL_PADDING.saturating_mul(2)),
        area.height
            .saturating_sub(INPUT_VERTICAL_PADDING.saturating_mul(2)),
    );
    if input_area.is_empty() {
        return;
    }

    if app.page_open() {
        let label = if app.model_picker_open() {
            "model index"
        } else if app.provider_editor_open() {
            "provider config"
        } else if app.session_picker_open() {
            "session index"
        } else {
            "command index"
        };
        let y = area.y + area.height.saturating_sub(1) / 2;
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("ZEX / ", Style::default().fg(ACCENT_PRIMARY).bg(BACKGROUND)),
                Span::styled(label, Style::default().fg(TEXT).bg(BACKGROUND)),
            ]))
            .style(Style::default().bg(BACKGROUND))
            .alignment(Alignment::Center),
            Rect::new(
                input_area.x,
                y.clamp(input_area.y, input_area.bottom().saturating_sub(1)),
                input_area.width,
                1,
            ),
        );
        return;
    }

    if app.busy {
        return;
    }

    render_input_buffer(
        frame,
        input_area,
        app,
        "",
        Some(Line::from(Span::styled(
            "ask anything…",
            Style::default().fg(TEXT_DIM).bg(BACKGROUND),
        ))),
        BACKGROUND,
    );
}

fn render_input_buffer(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &App,
    prompt: &str,
    placeholder: Option<Line<'static>>,
    background: Color,
) {
    if area.is_empty() {
        return;
    }
    let prompt_width = UnicodeWidthStr::width(prompt).min(area.width as usize) as u16;
    let prompt_area = Rect::new(area.x, area.y, prompt_width, area.height);
    let editor_area = Rect::new(
        prompt_area.right(),
        area.y,
        area.width.saturating_sub(prompt_width),
        area.height,
    );
    if editor_area.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(
                prompt.to_owned(),
                Style::default().fg(ACCENT_PRIMARY).bg(background),
            ))
            .style(Style::default().bg(background)),
            area,
        );
        return;
    }

    if prompt_width > 0 {
        frame.render_widget(
            Paragraph::new(Span::styled(
                prompt.to_owned(),
                Style::default().fg(ACCENT_PRIMARY).bg(background),
            ))
            .style(Style::default().bg(background)),
            prompt_area,
        );
    }

    let editor_width = editor_area.width.max(1) as usize;
    let visible_rows = editor_area.height.max(1) as usize;
    let metrics = input_metrics(&app.input.content, app.input.cursor, editor_width);
    let vertical_scroll = metrics.cursor_row.saturating_sub(visible_rows - 1);
    let editor = if app.input.is_empty() {
        Text::default()
    } else {
        Text::from(Line::from(Span::styled(
            app.input.content.clone(),
            Style::default().fg(TEXT_STRONG).bg(background),
        )))
    };
    frame.render_widget(
        Paragraph::new(editor)
            .style(Style::default().bg(background))
            .wrap(Wrap { trim: false })
            .scroll((vertical_scroll.min(u16::MAX as usize) as u16, 0)),
        editor_area,
    );
    if app.input.is_empty()
        && let Some(placeholder) = placeholder
        && editor_area.width > 1
    {
        frame.render_widget(
            Paragraph::new(placeholder)
                .style(Style::default().bg(background))
                .wrap(Wrap { trim: true }),
            Rect::new(
                editor_area.x + 1,
                editor_area.y,
                editor_area.width - 1,
                editor_area.height,
            ),
        );
    }

    let cursor_y = metrics.cursor_row.saturating_sub(vertical_scroll) as u16;
    frame.set_cursor_position((
        editor_area.x + metrics.cursor_column.min(editor_width - 1) as u16,
        editor_area.y + cursor_y.min(editor_area.height.saturating_sub(1)),
    ));
}

fn render_keymap(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.is_empty() {
        return;
    }
    let area = horizontal_inset(content_area(area), 2);
    if area.is_empty() {
        return;
    }
    let hint = if app.model_picker_open() {
        Line::from(Span::styled(
            "↑↓/jk select · Enter/Space switch · click twice · Esc/q cancel",
            Style::default().fg(TEXT_DIM),
        ))
    } else if app.provider_editor_open() {
        let text = if app.provider_editor_is_confirming() {
            "Enter/y confirm · Esc/n cancel"
        } else if app.provider_editor_is_editing() {
            "Enter apply field · Esc cancel · Ctrl+S save"
        } else {
            "Tab pane · ↑↓/jk select · f fetch models · Enter name · i ID · Space thinking · m/t min/max · r fill map · n new · d delete · Ctrl+S save · Esc exit"
        };
        Line::from(Span::styled(text, Style::default().fg(TEXT_DIM)))
    } else if app.session_picker_open() {
        Line::from(Span::styled(
            "↑↓/jk select · Enter/Space resume · click twice · Esc/q cancel",
            Style::default().fg(TEXT_DIM),
        ))
    } else if app.help_open {
        Line::from(Span::styled(
            "↑↓/jk select · Enter/Space run · click twice · Esc/q return",
            Style::default().fg(TEXT_DIM),
        ))
    } else if let Some(toast) = &app.toast {
        Line::from(vec![
            Span::styled("● ", Style::default().fg(toast.color())),
            Span::styled(toast.message.clone(), Style::default().fg(TEXT)),
        ])
    } else if app.completion_open() {
        Line::from(Span::styled(
            "↑↓ select · Tab complete · Enter run · Esc close",
            Style::default().fg(TEXT_DIM),
        ))
    } else if app.busy {
        let text = if area.width >= 72 {
            "Esc stop · wheel/PgUp/PgDn scroll · Ctrl+O cards"
        } else {
            "Esc stop · PgUp/PgDn scroll"
        };
        Line::from(Span::styled(text, Style::default().fg(TEXT_FAINT)))
    } else {
        let text = if area.width >= 92 {
            "Enter send · Shift+Enter newline · wheel scroll · Ctrl+O cards · / commands"
        } else {
            "Enter send · wheel scroll · / commands"
        };
        Line::from(Span::styled(text, Style::default().fg(TEXT_FAINT)))
    };
    frame.render_widget(Paragraph::new(hint), area);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct InputMetrics {
    pub(super) cursor_row: usize,
    pub(super) cursor_column: usize,
    pub(super) total_rows: usize,
}

pub(super) fn input_metrics(input: &str, cursor: usize, width: usize) -> InputMetrics {
    let width = width.max(1);
    let (cursor_row, cursor_column) = wrapped_position(&input[..cursor], width);
    let (last_row, last_column) = wrapped_position(input, width);
    let total_rows = last_row + usize::from(last_column > 0 || input.ends_with('\n')).max(1);
    InputMetrics {
        cursor_row,
        cursor_column,
        total_rows,
    }
}

fn wrapped_position(input: &str, width: usize) -> (usize, usize) {
    let mut row = 0;
    let mut column = 0;
    for character in input.chars() {
        if character == '\n' {
            row += 1;
            column = 0;
            continue;
        }
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if character_width > 0 && column + character_width > width {
            row += 1;
            column = 0;
        }
        column += character_width;
        if column >= width {
            row += column / width;
            column %= width;
        }
    }
    (row, column)
}

pub(super) fn format_json(value: &str) -> String {
    serde_json::from_str::<Value>(value)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| value.to_owned())
}

pub(super) fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let content: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{content}\n… truncated")
    } else {
        content
    }
}

pub(super) fn single_line(value: &str, max_chars: usize) -> String {
    truncate_chars(&value.replace(['\r', '\n'], " "), max_chars).replace("\n… truncated", " …")
}

/// Strip ANSI escape sequences, carriage returns and control characters so
/// captured tool output never renders as garbled terminal bytes. Runs of
/// replacement characters (from lossy decoding) collapse to a single marker.
pub(super) fn sanitize_terminal_text(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(character) = chars.next() {
        match character {
            '\u{1b}' => match chars.next() {
                Some('[') => {
                    for c in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&c) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    let mut escaped = false;
                    for c in chars.by_ref() {
                        if c == '\u{7}' || (escaped && c == '\\') {
                            break;
                        }
                        escaped = c == '\u{1b}';
                    }
                }
                _ => {}
            },
            '\n' | '\t' => cleaned.push(character),
            '\r' => {}
            '\u{fffd}' => {
                if !cleaned.ends_with('\u{fffd}') {
                    cleaned.push('\u{fffd}');
                }
            }
            c if c.is_control() => {}
            c => cleaned.push(c),
        }
    }
    cleaned
}

struct BashOutput<'a> {
    exit_code: Option<&'a str>,
    stdout: &'a str,
    stderr: &'a str,
}

/// Split the bash tool envelope (`exit_code: N\nstdout:\n…\nstderr:\n…`) into
/// parts so the card can render a product view instead of the raw shell.
fn parse_bash_output(output: &str) -> Option<BashOutput<'_>> {
    let rest = output.strip_prefix("exit_code: ")?;
    let (code, rest) = rest.split_once('\n')?;
    let rest = rest.strip_prefix("stdout:\n")?;
    let (stdout, stderr) = match rest.rfind("\nstderr:\n") {
        Some(index) => {
            let (stdout, tail) = rest.split_at(index);
            (stdout, &tail["\nstderr:\n".len()..])
        }
        None => (rest, ""),
    };
    Some(BashOutput {
        exit_code: Some(code.trim()),
        stdout,
        stderr,
    })
}

/// Output lines shown inside an expanded tool card: parsed bash stdout/stderr
/// for shell calls, plain output for everything else.
fn tool_output_body(tool: &ToolEntry) -> Vec<String> {
    if tool.name == "bash"
        && let Some(parsed) = parse_bash_output(&tool.output)
    {
        let failed = parsed.exit_code.is_some_and(|code| code != "0");
        let body = if failed && !parsed.stderr.trim().is_empty() {
            parsed.stderr
        } else {
            parsed.stdout
        };
        return body.lines().map(str::to_owned).collect();
    }
    tool.output.lines().map(str::to_owned).collect()
}

fn tool_arguments(tool: &ToolEntry) -> Option<Value> {
    serde_json::from_str::<Value>(&tool.arguments).ok()
}

fn edit_change_counts(tool: &ToolEntry) -> Option<(usize, usize)> {
    let arguments = tool_arguments(tool)?;
    let old = arguments.get("old_text")?.as_str()?;
    let new = arguments.get("new_text")?.as_str()?;
    let (_, old_changed, new_changed, _) = changed_line_ranges(old, new);
    Some((new_changed.len(), old_changed.len()))
}

fn file_change_body(tool: &ToolEntry) -> Option<Vec<String>> {
    let arguments = tool_arguments(tool)?;
    match tool.name.as_str() {
        "edit" => {
            let old = arguments.get("old_text")?.as_str()?;
            let new = arguments.get("new_text")?.as_str()?;
            let (prefix, old_changed, new_changed, suffix) = changed_line_ranges(old, new);
            let mut lines = Vec::new();
            if let Some(context) = prefix.last() {
                lines.push(format!("  {context}"));
            }
            lines.extend(old_changed.iter().map(|line| format!("- {line}")));
            lines.extend(new_changed.iter().map(|line| format!("+ {line}")));
            if let Some(context) = suffix.first() {
                lines.push(format!("  {context}"));
            }
            Some(lines)
        }
        "write" => arguments.get("content")?.as_str().map(|content| {
            content
                .lines()
                .map(|line| format!("+ {line}"))
                .collect::<Vec<_>>()
        }),
        _ => None,
    }
}

fn changed_line_ranges<'a>(
    old: &'a str,
    new: &'a str,
) -> (Vec<&'a str>, Vec<&'a str>, Vec<&'a str>, Vec<&'a str>) {
    let old_lines = old.lines().collect::<Vec<_>>();
    let new_lines = new.lines().collect::<Vec<_>>();
    let prefix_len = old_lines
        .iter()
        .zip(&new_lines)
        .take_while(|(old, new)| old == new)
        .count();
    let shared_remaining = old_lines
        .len()
        .saturating_sub(prefix_len)
        .min(new_lines.len().saturating_sub(prefix_len));
    let suffix_len = old_lines[prefix_len..]
        .iter()
        .rev()
        .zip(new_lines[prefix_len..].iter().rev())
        .take(shared_remaining)
        .take_while(|(old, new)| old == new)
        .count();
    let old_changed_end = old_lines.len().saturating_sub(suffix_len);
    let new_changed_end = new_lines.len().saturating_sub(suffix_len);
    (
        old_lines[..prefix_len].to_vec(),
        old_lines[prefix_len..old_changed_end].to_vec(),
        new_lines[prefix_len..new_changed_end].to_vec(),
        old_lines[old_changed_end..].to_vec(),
    )
}

fn truncate_inline(value: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut chars = value.chars();
    let content = chars
        .by_ref()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    if chars.next().is_some() {
        format!("{content}…")
    } else {
        value.to_owned()
    }
}

fn truncate_display(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if UnicodeWidthStr::width(value) <= width {
        return value.to_owned();
    }
    if width == 1 {
        return "…".to_owned();
    }

    let mut result = String::new();
    let mut used = 0;
    for character in value.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if used + character_width > width - 1 {
            break;
        }
        result.push(character);
        used += character_width;
    }
    result.push('…');
    result
}

fn wrap_display_words(value: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if value.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0;
    for word in value.split_whitespace() {
        let word_width = UnicodeWidthStr::width(word);
        if word_width > width {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
                current_width = 0;
            }
            let mut chunks = wrap_display_hard(word, width);
            if let Some(last) = chunks.pop() {
                lines.extend(chunks);
                current_width = UnicodeWidthStr::width(last.as_str());
                current = last;
            }
            continue;
        }
        let separator_width = usize::from(!current.is_empty());
        if current_width + separator_width + word_width > width {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        if !current.is_empty() {
            current.push(' ');
            current_width += 1;
        }
        current.push_str(word);
        current_width += word_width;
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn wrap_display_hard(value: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if value.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0;
    for character in value.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if current_width > 0 && current_width + character_width > width {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(character);
        current_width += character_width;
        if current_width >= width {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn short_session_id(id: &str) -> &str {
    id.rsplit('-').next().unwrap_or(id)
}

fn format_session_time(timestamp: time::OffsetDateTime) -> String {
    let local_offset = time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC);
    timestamp
        .to_offset(local_offset)
        .format(time::macros::format_description!(
            "[year]-[month]-[day] [hour]:[minute]"
        ))
        .unwrap_or_else(|_| timestamp.unix_timestamp().to_string())
}

fn content_area(area: Rect) -> Rect {
    let gutter = HORIZONTAL_GUTTER.min(area.width / 2);
    Rect::new(
        area.x + gutter,
        area.y,
        area.width.saturating_sub(gutter.saturating_mul(2)),
        area.height,
    )
}

fn pad_display(value: &str, width: usize) -> String {
    let value_width = UnicodeWidthStr::width(value);
    let mut padded = String::with_capacity(value.len() + width.saturating_sub(value_width));
    padded.push_str(value);
    padded.extend(std::iter::repeat_n(' ', width.saturating_sub(value_width)));
    padded
}

fn pad_display_left(value: &str, width: usize) -> String {
    let value_width = UnicodeWidthStr::width(value);
    let mut padded = String::with_capacity(value.len() + width.saturating_sub(value_width));
    padded.extend(std::iter::repeat_n(' ', width.saturating_sub(value_width)));
    padded.push_str(value);
    padded
}

fn numbered_list_item(line: &str) -> Option<(&str, &str)> {
    let (number, item) = line.split_once(". ")?;
    (!number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit()))
        .then_some((number, item))
}

pub(super) fn error_summary(message: &str) -> String {
    let first = message
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("Unknown error")
        .trim();
    single_line(first, 120)
}
