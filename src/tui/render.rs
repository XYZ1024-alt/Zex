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
    render_footer(frame, regions.footer, app);
    render_keymap(frame, regions.keymap, app);
    if app.output_panel_open() {
        render_output_panel(frame, area, app);
    }
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    let status_height = u16::from(area.height > 1);
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

struct StatuslineInputs {
    cwd: String,
    project: String,
    git: Option<String>,
    speed: Option<String>,
    context: String,
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
        cwd,
        project,
        git: app.git_status.as_ref().map(git_status_label),
        speed: (!app.busy)
            .then_some(app.tokens_per_second)
            .flatten()
            .map(|rate| format!("{rate:.1} tok/s")),
        context: format!("ctx {percent:.1}%"),
    }
}

fn render_statusline(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
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
    let separator = "  ·  ";
    let full_right = inputs
        .speed
        .iter()
        .chain(std::iter::once(&inputs.context))
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(separator);
    let full_left = inputs.git.as_ref().map_or_else(
        || inputs.cwd.clone(),
        |git| format!("{}  ·  {git}", inputs.cwd),
    );
    let short_left = inputs.git.as_ref().map_or_else(
        || inputs.project.clone(),
        |git| format!("{}  ·  {git}", inputs.project),
    );
    let right = if UnicodeWidthStr::width(full_right.as_str())
        + UnicodeWidthStr::width(short_left.as_str())
        < width
    {
        full_right
    } else {
        inputs.context.clone()
    };
    let right_width = UnicodeWidthStr::width(right.as_str());
    let available_left = width.saturating_sub(right_width + 1);
    let left = if UnicodeWidthStr::width(full_left.as_str()) <= available_left {
        full_left
    } else {
        truncate_inline(&short_left, available_left)
    };
    let spacer = width.saturating_sub(
        UnicodeWidthStr::width(left.as_str()) + UnicodeWidthStr::width(right.as_str()),
    );
    Line::from(vec![
        Span::styled(left, Style::default().fg(TEXT_FAINT)),
        Span::raw(" ".repeat(spacer)),
        Span::styled(right, Style::default().fg(TEXT_FAINT)),
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

fn render_session_picker(frame: &mut Frame<'_>, viewport: Rect, app: &mut App) {
    let Some(picker) = app.session_picker.clone() else {
        return;
    };
    if viewport.is_empty() {
        return;
    }

    let area = content_area(viewport);
    if area.is_empty() {
        return;
    }
    let header_height = if area.height >= 7 { 3 } else { 1 };
    let header = Rect::new(area.x, area.y, area.width, header_height);
    let list = Rect::new(
        area.x,
        header.bottom(),
        area.width,
        area.height.saturating_sub(header_height),
    );
    let inner_width = area.width.saturating_sub(6) as usize;
    let max_visible = list.height.saturating_add(1).max(1) as usize / 3;
    let visible_count = picker.sessions.len().clamp(1, max_visible.max(1));
    frame.render_widget(
        Paragraph::new(Line::from(vec![
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
        ])),
        Rect::new(header.x, header.y, header.width, 1),
    );
    if header.height >= 3 {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Recent conversations, ordered by last activity.",
                Style::default().fg(TEXT_FAINT),
            )),
            Rect::new(header.x, header.y + 1, header.width, 1),
        );
    }

    if picker.sessions.is_empty() {
        let empty_height = list.height.min(5);
        let empty_y = list
            .y
            .saturating_add(list.height.saturating_sub(empty_height) / 3);
        let empty = Rect::new(list.x, empty_y, list.width, empty_height);
        frame.render_widget(Block::default().style(Style::default().bg(SURFACE)), empty);
        render_accent(frame, empty, ACCENT_PRIMARY, SURFACE);
        if empty.height >= 1 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "No saved sessions",
                    Style::default()
                        .fg(TEXT_STRONG)
                        .add_modifier(Modifier::BOLD),
                )),
                Rect::new(
                    empty.x.saturating_add(3),
                    empty.y.saturating_add(u16::from(empty.height >= 4)),
                    empty.width.saturating_sub(6),
                    1,
                ),
            );
        }
        if empty.height >= 3 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "Start a conversation here; Zex will keep it ready to resume.",
                    Style::default().fg(TEXT_DIM),
                ))
                .wrap(Wrap { trim: true }),
                Rect::new(
                    empty.x.saturating_add(3),
                    empty.y.saturating_add(2),
                    empty.width.saturating_sub(6),
                    empty.height.saturating_sub(2),
                ),
            );
        }
    } else {
        let start = picker
            .selected
            .saturating_sub(visible_count.saturating_sub(1));
        for (offset, (index, session)) in picker
            .sessions
            .iter()
            .enumerate()
            .skip(start)
            .take(visible_count)
            .enumerate()
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
                SURFACE
            };
            let item = Rect::new(
                list.x,
                list.y.saturating_add(offset as u16 * 3),
                list.width,
                2.min(list.bottom().saturating_sub(list.y + offset as u16 * 3)),
            );
            if item.is_empty() {
                continue;
            }
            frame.render_widget(
                Block::default().style(Style::default().bg(background)),
                item,
            );
            if selected || armed {
                render_accent(frame, item, ACCENT_PRIMARY, background);
            }
            let foreground = if selected { TEXT_STRONG } else { TEXT };
            let secondary = if selected { TEXT } else { TEXT_DIM };
            let timestamp = format_session_time(session.updated_at);
            let metadata = format!(
                "{} · {} message{}",
                timestamp,
                session.message_count,
                if session.message_count == 1 { "" } else { "s" }
            );
            let id = short_session_id(&session.id);
            let id_width = UnicodeWidthStr::width(id).min(inner_width);
            let metadata_limit = inner_width.saturating_sub(id_width + 2);
            let metadata = truncate_display(&metadata, metadata_limit);
            let metadata_width = UnicodeWidthStr::width(metadata.as_str());
            let spacer = inner_width.saturating_sub(id_width + metadata_width);
            let text_x = item.x.saturating_add(3);
            let text_width = item.width.saturating_sub(6);
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(
                        truncate_display(id, id_width),
                        Style::default().fg(foreground).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" ".repeat(spacer)),
                    Span::styled(metadata, Style::default().fg(secondary)),
                ]))
                .style(Style::default().bg(background)),
                Rect::new(text_x, item.y, text_width, 1),
            );
            if item.height >= 2 {
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        single_line(&session.preview, inner_width),
                        Style::default().fg(secondary),
                    ))
                    .style(Style::default().bg(background)),
                    Rect::new(text_x, item.y + 1, text_width, 1),
                );
            }
            app.register_hit(item, HitTarget::Session(index));
        }
    }
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
    pub(super) footer: Rect,
    pub(super) keymap: Rect,
}

pub(super) fn ui_regions(area: Rect, app: &App) -> UiRegions {
    let keymap_height = u16::from(area.height >= 3);
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
    let preferred_footer_height = if app.page_open() {
        3
    } else {
        (requested_input_rows as u16)
            .saturating_add(2)
            .saturating_add(INPUT_VERTICAL_PADDING.saturating_mul(2))
    };
    let footer_height = if area.height
        >= keymap_height
            .saturating_add(MIN_TRANSCRIPT_HEIGHT)
            .saturating_add(2)
    {
        preferred_footer_height.min(
            area.height
                .saturating_sub(keymap_height)
                .saturating_sub(MIN_TRANSCRIPT_HEIGHT),
        )
    } else {
        area.height.saturating_sub(keymap_height).min(2)
    };
    let fixed_height = footer_height + keymap_height;
    let remaining = area.height.saturating_sub(fixed_height);
    let transcript_reserve = MIN_TRANSCRIPT_HEIGHT.min(remaining);
    let completion_width = footer_input_width(area.width);
    let completion_height =
        completion_height(app, completion_width).min(remaining.saturating_sub(transcript_reserve));
    let transcript_height = remaining.saturating_sub(completion_height);

    let [transcript, completion, footer, keymap] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(transcript_height),
            Constraint::Length(completion_height),
            Constraint::Length(footer_height),
            Constraint::Length(keymap_height),
        ])
        .areas(area);

    UiRegions {
        transcript,
        completion,
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

fn centered_rect(area: Rect, width_percent: u16, height_percent: u16) -> Rect {
    let width = (u32::from(area.width) * u32::from(width_percent) / 100)
        .clamp(1, u32::from(area.width)) as u16;
    let height = (u32::from(area.height) * u32::from(height_percent) / 100)
        .clamp(1, u32::from(area.height)) as u16;
    Rect::new(
        area.x.saturating_add(area.width.saturating_sub(width) / 2),
        area.y
            .saturating_add(area.height.saturating_sub(height) / 2),
        width,
        height,
    )
}

fn render_output_panel(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    frame
        .buffer_mut()
        .set_style(area, Style::default().fg(TEXT_DIM).bg(BACKGROUND));

    let panel_area = centered_rect(area, 84, 82);
    if panel_area.width < 4 || panel_area.height < 4 {
        return;
    }
    app.register_hit(panel_area, HitTarget::OutputPanel);
    frame.render_widget(Clear, panel_area);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT_PRIMARY))
            .style(Style::default().bg(SURFACE_RAISED)),
        panel_area,
    );

    let inner = Rect::new(
        panel_area.x + 1,
        panel_area.y + 1,
        panel_area.width - 2,
        panel_area.height - 2,
    );
    let [header, content, footer] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);
    app.register_hit(
        Rect::new(header.right().saturating_sub(3), header.y, 3, 1),
        HitTarget::OutputClose,
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "ZEX / ",
                Style::default()
                    .fg(ACCENT_PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("assistant output", Style::default().fg(TEXT_STRONG)),
        ]))
        .style(Style::default().bg(SURFACE_RAISED)),
        header,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(
            "[×]",
            Style::default()
                .fg(if app.hovered(&HitTarget::OutputClose) {
                    TEXT_STRONG
                } else {
                    ACCENT_PRIMARY
                })
                .bg(SURFACE_RAISED)
                .add_modifier(Modifier::BOLD),
        ))
        .alignment(Alignment::Right),
        header,
    );

    let (entry_index, requested_scroll) = app
        .output_panel
        .map(|panel| (panel.entry_index, panel.scroll_top))
        .unwrap_or((usize::MAX, 0));
    let Some(TranscriptEntry::Message {
        role: MessageRole::Assistant,
        content: output,
    }) = app.transcript.get(entry_index)
    else {
        return;
    };
    let mut lines = Vec::new();
    append_markdown_lines(
        &mut lines,
        output,
        MessageRole::Assistant,
        true,
        content.width.saturating_sub(2) as usize,
    );
    let paragraph = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT).bg(SURFACE))
        .wrap(Wrap { trim: false });
    let line_count = paragraph.line_count(content.width.max(1));
    let page_height = content.height as usize;
    let max_scroll = line_count.saturating_sub(page_height);
    let scroll_top = requested_scroll.min(max_scroll);
    if let Some(panel) = &mut app.output_panel {
        panel.page_height = page_height;
        panel.max_scroll = max_scroll;
        panel.scroll_top = scroll_top;
    }
    frame.render_widget(
        paragraph.scroll((scroll_top.min(u16::MAX as usize) as u16, 0)),
        content,
    );

    let position = if max_scroll == 0 {
        "full output".to_owned()
    } else {
        format!(
            "{}–{} / {}",
            scroll_top.saturating_add(1),
            scroll_top.saturating_add(page_height).min(line_count),
            line_count
        )
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                "↑↓ read · PgUp/PgDn jump · Space page · Esc timeline",
                Style::default().fg(TEXT_DIM),
            ),
            Span::styled(format!("  {position}"), Style::default().fg(TEXT_FAINT)),
        ]))
        .style(Style::default().bg(SURFACE_RAISED)),
        footer,
    );
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

    for panel in &transcript.panels {
        render_transcript_panel_background(frame, area, panel, app.scroll_top);
    }
    let paragraph = paragraph.scroll((app.scroll_top.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(paragraph, area);
    for panel in &transcript.panels {
        render_transcript_panel_accent(frame, area, panel, app.scroll_top);
    }
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
    for (index, start_line, end_line) in transcript.response_lines {
        let visible_start = start_line.max(app.scroll_top);
        let visible_end = end_line.min(app.scroll_top.saturating_add(area.height as usize));
        if visible_start >= visible_end {
            continue;
        }
        let y = area.y.saturating_add(
            u16::try_from(visible_start.saturating_sub(app.scroll_top)).unwrap_or(u16::MAX),
        );
        let height = u16::try_from(visible_end.saturating_sub(visible_start)).unwrap_or(u16::MAX);
        app.register_hit(
            Rect::new(area.x, y, area.width, height),
            HitTarget::Response(index),
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

fn visible_transcript_panel(
    area: Rect,
    panel: &TranscriptPanel,
    scroll_top: usize,
) -> Option<Rect> {
    let visible_start = panel.start_line.max(scroll_top);
    let visible_end = panel
        .end_line
        .min(scroll_top.saturating_add(area.height as usize));
    if visible_start >= visible_end {
        return None;
    }
    let y = area.y.saturating_add(
        u16::try_from(visible_start.saturating_sub(scroll_top)).unwrap_or(u16::MAX),
    );
    let height = u16::try_from(visible_end.saturating_sub(visible_start)).unwrap_or(u16::MAX);
    Some(Rect::new(
        area.x,
        y,
        area.width
            .min(u16::try_from(panel.width).unwrap_or(u16::MAX)),
        height,
    ))
}

fn render_transcript_panel_background(
    frame: &mut Frame<'_>,
    area: Rect,
    panel: &TranscriptPanel,
    scroll_top: usize,
) {
    let Some(panel_area) = visible_transcript_panel(area, panel, scroll_top) else {
        return;
    };
    frame
        .buffer_mut()
        .set_style(panel_area, Style::default().bg(panel.background));
}

fn render_transcript_panel_accent(
    frame: &mut Frame<'_>,
    area: Rect,
    panel: &TranscriptPanel,
    scroll_top: usize,
) {
    let Some(panel_area) = visible_transcript_panel(area, panel, scroll_top) else {
        return;
    };
    if let Some(accent) = panel.accent
        && panel_area.width > 0
    {
        render_accent(frame, panel_area, accent, panel.background);
    }
}

fn render_accent(frame: &mut Frame<'_>, area: Rect, accent: Color, background: Color) {
    if area.is_empty() {
        return;
    }
    frame.buffer_mut().set_style(
        Rect::new(area.x, area.y, 1, area.height),
        Style::default().fg(accent).bg(background),
    );
    for row in area.y..area.bottom() {
        frame.buffer_mut()[(area.x, row)].set_symbol("▎");
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

    let status_height = u16::from(area.height >= 2);
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
        card_width.saturating_sub(6).max(1) as usize,
    )
    .total_rows
    .clamp(1, MAX_INPUT_ROWS) as u16;
    let card_height = match stage.height {
        12.. => input_rows.saturating_add(4).max(5),
        6..=11 => 4,
        1..=4 => 1,
        5 => 3,
        _ => 0,
    };
    let full_logo = stage.height >= 16 && card_width >= 29;
    let brand_height = if full_logo {
        LANDING_LOGO_ROWS.len() as u16
    } else if stage.height >= 7 {
        1
    } else {
        0
    };
    let brand_gap = u16::from(brand_height > 0);
    let hint_height = if stage.height >= 14 {
        2
    } else if stage.height >= 10 {
        1
    } else {
        0
    };
    let hint_gap = u16::from(hint_height > 0);
    let group_height = brand_height + brand_gap + card_height + hint_gap + hint_height;
    let free_height = stage.height.saturating_sub(group_height);
    let group_y = stage.y + free_height / 2;
    let card_x = area.x + area.width.saturating_sub(card_width) / 2;
    let brand = Rect::new(card_x, group_y, card_width, brand_height);
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
        if regions.brand.height >= LANDING_LOGO_ROWS.len() as u16 {
            for (row, logo_row) in LANDING_LOGO_ROWS.iter().enumerate() {
                frame.render_widget(
                    Paragraph::new(landing_logo_line(logo_row)).alignment(Alignment::Center),
                    Rect::new(
                        regions.brand.x,
                        regions.brand.y.saturating_add(row as u16),
                        regions.brand.width,
                        1,
                    ),
                );
            }
        } else {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "ZEX",
                    Style::default()
                        .fg(TEXT_STRONG)
                        .add_modifier(Modifier::BOLD),
                ))
                .alignment(Alignment::Center),
                Rect::new(regions.brand.x, regions.brand.y, regions.brand.width, 1),
            );
        }
    }

    render_landing_card(frame, regions.card, app);
    if !regions.hint.is_empty() {
        let hint_width = regions.hint.width.saturating_sub(6) as usize;
        let actions = if hint_width >= 52 {
            "Enter send  ·  Shift+Enter newline  ·  / actions"
        } else if hint_width >= 24 {
            "Enter send  ·  / actions"
        } else {
            "Enter send"
        };
        frame.render_widget(
            Paragraph::new(Span::styled(actions, Style::default().fg(TEXT_DIM))),
            Rect::new(
                regions.hint.x.saturating_add(3),
                regions.hint.y,
                regions.hint.width.saturating_sub(6),
                1,
            ),
        );
        if regions.hint.height >= 2 {
            let hint = if regions.hint.width >= 70 {
                "Point to a file, describe an outcome, or ask a question."
            } else if regions.hint.width >= 48 {
                "Point to a file, describe an outcome, or ask."
            } else {
                "Point to a file, task, or question."
            };
            frame.render_widget(
                Paragraph::new(Span::styled(hint, Style::default().fg(TEXT_FAINT))),
                Rect::new(
                    regions.hint.x.saturating_add(3),
                    regions.hint.y + 1,
                    regions.hint.width.saturating_sub(6),
                    1,
                ),
            );
        }
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
    Line::from(
        row.chars()
            .enumerate()
            .map(|(column, character)| {
                let color = if column < 9 {
                    TEXT_FAINT
                } else if column < 18 {
                    TEXT_DIM
                } else {
                    TEXT_STRONG
                };
                Span::styled(character.to_string(), Style::default().fg(color))
            })
            .collect::<Vec<_>>(),
    )
}

fn landing_card_width(width: u16) -> u16 {
    if width == 0 {
        return 0;
    }
    let max_width = width.saturating_sub(2).max(1);
    let target = (u32::from(width) * 31 / 50) as u16;
    target.clamp(max_width.min(40), max_width.min(76))
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
    render_accent(frame, area, ACCENT_PRIMARY, SURFACE);
    if area.width <= 7 || area.height == 0 {
        return;
    }
    let compact = area.height < 5;
    let editor_y = area.y.saturating_add(u16::from(area.height >= 3));
    let metadata_y = area.bottom().saturating_sub(2);
    let editor_area = Rect::new(
        area.x.saturating_add(3),
        editor_y,
        area.width.saturating_sub(6),
        metadata_y
            .saturating_sub(editor_y)
            .max(u16::from(editor_y < area.bottom())),
    );
    if !compact && metadata_y < area.bottom() {
        let model = model_short_name(&app.model);
        let thinking = thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference));
        let right = format!("think {thinking}");
        let right_width = UnicodeWidthStr::width(right.as_str());
        let model_width = editor_area.width as usize;
        let model = truncate_display(&model, model_width.saturating_sub(right_width + 2));
        let spacer = model_width.saturating_sub(
            UnicodeWidthStr::width(model.as_str()) + UnicodeWidthStr::width(right.as_str()),
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    model,
                    Style::default()
                        .fg(TEXT_STRONG)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" ".repeat(spacer)),
                Span::styled(right, Style::default().fg(TEXT_FAINT)),
            ]))
            .style(Style::default().bg(SURFACE)),
            Rect::new(editor_area.x, metadata_y, editor_area.width, 1),
        );
    }
    if !editor_area.is_empty() {
        let placeholder = Line::from(Span::styled("Ask anything…", Style::default().fg(TEXT_DIM)));
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
    let version = env!("CARGO_PKG_VERSION").to_owned();
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
    panels: Vec<TranscriptPanel>,
    card_lines: Vec<(usize, usize)>,
    output_lines: Vec<(usize, usize)>,
    response_lines: Vec<(usize, usize, usize)>,
}

struct TranscriptPanel {
    start_line: usize,
    end_line: usize,
    width: usize,
    background: Color,
    accent: Option<Color>,
}

fn transcript_text(app: &App, width: usize) -> TranscriptRender {
    const PANEL_RIGHT_PADDING: usize = 3;
    let content_width = width.saturating_sub(PANEL_RIGHT_PADDING).max(1);
    let mut lines = Vec::new();
    let mut panels = Vec::new();
    let mut card_lines = Vec::new();
    let mut output_lines = Vec::new();
    let mut response_lines = Vec::new();
    for (index, entry) in app.transcript.iter().enumerate() {
        let next = app.transcript.get(index + 1);
        match entry {
            TranscriptEntry::Message { role, content } => {
                let final_answer = app.is_final_answer(index);
                let start_line = lines.len();
                lines.push(Line::default());
                append_markdown_lines(&mut lines, content, *role, final_answer, content_width);
                lines.push(Line::default());
                let selected = app.selected_entry == Some(index);
                let hovered = final_answer && app.hovered(&HitTarget::Response(index));
                panels.push(TranscriptPanel {
                    start_line,
                    end_line: lines.len(),
                    width,
                    background: if selected {
                        SURFACE_RAISED
                    } else if hovered {
                        SURFACE_HOVER
                    } else {
                        SURFACE
                    },
                    accent: Some(if *role == MessageRole::User {
                        ACCENT_PRIMARY
                    } else if final_answer {
                        ACCENT_SECONDARY
                    } else {
                        TEXT_FAINT
                    }),
                });
                if final_answer {
                    response_lines.push((index, start_line, lines.len()));
                }
            }
            TranscriptEntry::Thinking(thinking) => {
                if app.show_thinking {
                    let start_line = lines.len();
                    card_lines.push((index, start_line));
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
                        app.selected_entry == Some(index),
                        app.hovered(&HitTarget::Card(index)),
                        content_width,
                    );
                    panels.push(TranscriptPanel {
                        start_line,
                        end_line: lines.len().saturating_sub(1).max(start_line + 1),
                        width,
                        background: if app.selected_entry == Some(index) {
                            SURFACE_RAISED
                        } else if app.hovered(&HitTarget::Card(index)) {
                            SURFACE_HOVER
                        } else {
                            SURFACE
                        },
                        accent: Some(ACCENT_SECONDARY),
                    });
                }
            }
            TranscriptEntry::Tool(tool) => {
                let start_line = lines.len();
                card_lines.push((index, start_line));
                append_tool_lines(
                    &mut lines,
                    tool,
                    app.selected_entry == Some(index),
                    app.hovered(&HitTarget::Card(index)),
                    content_width,
                );
                panels.push(TranscriptPanel {
                    start_line,
                    end_line: lines.len(),
                    width,
                    background: if app.selected_entry == Some(index) {
                        SURFACE_RAISED
                    } else if app.hovered(&HitTarget::Card(index)) {
                        SURFACE_HOVER
                    } else {
                        SURFACE
                    },
                    accent: if app.selected_entry == Some(index)
                        || matches!(tool.status, ToolStatus::Running)
                    {
                        Some(ACCENT_PRIMARY)
                    } else {
                        None
                    },
                });
                if tool.expanded && tool_output_body(tool).len() > TOOL_OUTPUT_PREVIEW_LINES {
                    output_lines.push((index, lines.len().saturating_sub(2)));
                }
            }
            TranscriptEntry::Error {
                summary,
                detail,
                expanded,
            } => {
                let start_line = lines.len();
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
                panels.push(TranscriptPanel {
                    start_line,
                    end_line: lines.len(),
                    width,
                    background: SURFACE,
                    accent: Some(BAD),
                });
                lines.push(Line::default());
            }
            TranscriptEntry::Turn(turn) => {
                let start_line = lines.len();
                append_turn_line(&mut lines, turn, content_width);
                panels.push(TranscriptPanel {
                    start_line,
                    end_line: lines.len(),
                    width: width.min(48),
                    background: SURFACE,
                    accent: Some(match turn.outcome {
                        TurnOutcome::Done => OK,
                        TurnOutcome::Failed => BAD,
                        TurnOutcome::Stopped => TEXT_DIM,
                    }),
                });
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
        let start_line = lines.len();
        append_running_turn_line(&mut lines, app, content_width);
        panels.push(TranscriptPanel {
            start_line,
            end_line: lines.len(),
            width: width.min(48),
            background: SURFACE,
            accent: Some(ACCENT_PRIMARY),
        });
    }

    TranscriptRender {
        text: Text::from(lines),
        panels,
        card_lines,
        output_lines,
        response_lines,
    }
}

fn append_transcript_gap(
    lines: &mut Vec<Line<'static>>,
    _entry: &TranscriptEntry,
    next: Option<&TranscriptEntry>,
) {
    let Some(_) = next else {
        return;
    };
    lines.push(Line::default());
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
    let detail = match app.status {
        Status::Thinking => "Preparing the next step",
        Status::RunningTool => "Waiting for tool output",
        Status::Cancelling => "Stopping the active turn",
        Status::Error => "The turn needs attention",
        Status::Idle => "Ready",
    };
    lines.push(Line::from(vec![
        Span::raw("    "),
        Span::styled(detail, Style::default().fg(TEXT_FAINT)),
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
                lines.push(Line::from(vec![
                    Span::styled(
                        rail,
                        Style::default().fg(message_rail_color(role, final_answer)),
                    ),
                    Span::styled("▌ ", Style::default().fg(TEXT_DIM)),
                    Span::styled(
                        if language.is_empty() {
                            "code".to_owned()
                        } else {
                            language.to_owned()
                        },
                        Style::default().fg(TEXT_DIM),
                    ),
                ]));
            } else {
                let rail = message_rail(role, final_answer, &mut first_visual_line);
                lines.push(Line::from(vec![
                    Span::styled(
                        rail,
                        Style::default().fg(message_rail_color(role, final_answer)),
                    ),
                    Span::styled("▌", Style::default().fg(TEXT_FAINT)),
                ]));
            }
            continue;
        }
        if in_code_block {
            let content_width = width.saturating_sub(4).max(1);
            for segment in wrap_display_hard(source_line, content_width) {
                let rail = message_rail(role, final_answer, &mut first_visual_line);
                lines.push(Line::from(vec![
                    Span::styled(
                        rail,
                        Style::default().fg(message_rail_color(role, final_answer)),
                    ),
                    Span::styled("▌ ", Style::default().fg(TEXT_DIM)),
                    Span::styled(segment, Style::default().fg(TEXT_STRONG)),
                ]));
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
                Style::default().fg(message_rail_color(role, final_answer)),
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
        (MessageRole::Assistant, true, _) => "▎  ",
        (MessageRole::User, _, true) => "▎  ",
        (MessageRole::User, _, false) => "▎  ",
        (MessageRole::Assistant, false, true) => "▎  ",
        (MessageRole::Assistant, false, false) => "▎  ",
    };
    *first_visual_line = false;
    rail
}

fn message_rail_color(role: MessageRole, final_answer: bool) -> Color {
    match (role, final_answer) {
        (MessageRole::User, _) => ACCENT_PRIMARY,
        (MessageRole::Assistant, true) => ACCENT_SECONDARY,
        (MessageRole::Assistant, false) => TEXT_FAINT,
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
        MessageRole::Assistant => 3,
    };
    for segment in wrap_display_words(content, width.saturating_sub(rail_width).max(1)) {
        let rail = message_rail(role, final_answer, first_visual_line);
        lines.push(Line::from(vec![
            Span::styled(
                rail,
                Style::default().fg(message_rail_color(role, final_answer)),
            ),
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
        MessageRole::Assistant => 3,
    };
    let content_width = width.saturating_sub(rail_width + marker_width).max(1);
    for (index, segment) in wrap_display_words(content, content_width)
        .into_iter()
        .enumerate()
    {
        let final_answer = false;
        let rail = message_rail(role, final_answer, first_visual_line);
        lines.push(Line::from(vec![
            Span::styled(
                rail,
                Style::default().fg(message_rail_color(role, final_answer)),
            ),
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
    _hovered: bool,
    width: usize,
) {
    let fold = if thinking.expanded { "▼" } else { "▶" };
    let rail_color = if selected {
        ACCENT_PRIMARY
    } else {
        ACCENT_SECONDARY
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
    lines.push(Line::from(header));

    if thinking.expanded {
        for line in thinking.content.split('\n') {
            lines.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(rail_color)),
                Span::styled(line.to_owned(), Style::default().fg(TEXT_DIM)),
            ]));
        }
        lines.push(Line::from(vec![
            Span::styled("  └ ", Style::default().fg(rail_color)),
            Span::styled(
                format!(
                    "{} lines · Ctrl+O collapse",
                    thinking.content.lines().count().max(1)
                ),
                Style::default().fg(TEXT_FAINT),
            ),
        ]));
    }
}

mod tool;

use tool::append_tool_lines;

pub(super) use tool::tool_detail_line_count;

fn render_input_frame(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let input_hit_area = Rect::new(
        area.x,
        area.y.saturating_add(1),
        area.width,
        area.height.saturating_sub(1),
    );
    app.register_hit(input_hit_area, HitTarget::Input);
    frame.render_widget(
        Block::default()
            .borders(Borders::NONE)
            .style(Style::default().bg(SURFACE)),
        area,
    );
    if area.width > 0 {
        let accent = if app.input_focused && !app.page_open() {
            ACCENT_PRIMARY
        } else {
            TEXT_FAINT
        };
        frame.buffer_mut().set_style(
            Rect::new(area.x, area.y, 1, area.height),
            Style::default().fg(accent).bg(SURFACE),
        );
        for row in area.y..area.bottom() {
            frame.buffer_mut()[(area.x, row)].set_symbol("▎");
        }
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
        let y = area.y.saturating_add(area.height.saturating_sub(1) / 2);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("ZEX  ", Style::default().fg(ACCENT_PRIMARY)),
                Span::styled(label, Style::default().fg(TEXT)),
            ]))
            .style(Style::default().bg(SURFACE))
            .alignment(Alignment::Center),
            Rect::new(area.x.saturating_add(1), y, area.width.saturating_sub(1), 1),
        );
        return;
    }
    if area.width <= INPUT_HORIZONTAL_PADDING.saturating_mul(2) || area.height == 0 {
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

    let metadata = input_metadata_line(app, input_area.width as usize);
    frame.render_widget(
        Paragraph::new(metadata).style(Style::default().bg(SURFACE)),
        Rect::new(input_area.x, input_area.y, input_area.width, 1),
    );
    register_input_metadata_hits(app, input_area);

    let editor_area = Rect::new(
        input_area.x,
        input_area.y.saturating_add(1),
        input_area.width,
        input_area.height.saturating_sub(1),
    );
    if editor_area.is_empty() {
        return;
    }
    if app.busy {
        let busy_label = match app.status {
            Status::RunningTool => "working · waiting for tool output",
            Status::Cancelling => "stopping…",
            Status::Error => "turn needs attention",
            Status::Thinking | Status::Idle => "working…",
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("› ", Style::default().fg(ACCENT_PRIMARY)),
                Span::styled(busy_label, Style::default().fg(TEXT_DIM)),
            ]))
            .style(Style::default().bg(SURFACE)),
            editor_area,
        );
        return;
    }

    render_input_buffer(
        frame,
        editor_area,
        app,
        "",
        Some(Line::from(Span::styled(
            "ask anything…",
            Style::default().fg(TEXT_DIM),
        ))),
        SURFACE,
    );
}

fn input_metadata_line(app: &App, width: usize) -> Line<'static> {
    if width == 0 {
        return Line::default();
    }
    let model = model_short_name(&app.model);
    let thinking = thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference));
    let separator = "  ·  ";
    let fixed = UnicodeWidthStr::width("ZEX")
        + UnicodeWidthStr::width(thinking)
        + UnicodeWidthStr::width(separator) * 2;
    let model = truncate_inline(&model, width.saturating_sub(fixed).max(1));
    Line::from(vec![
        Span::styled(
            "ZEX",
            Style::default()
                .fg(ACCENT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(separator, Style::default().fg(TEXT_FAINT)),
        Span::styled(
            model,
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(separator, Style::default().fg(TEXT_FAINT)),
        Span::styled(thinking, Style::default().fg(TEXT_DIM)),
    ])
}

fn register_input_metadata_hits(app: &mut App, area: Rect) {
    if area.is_empty() {
        return;
    }
    let separator_width = UnicodeWidthStr::width("  ·  ") as u16;
    let model = model_short_name(&app.model);
    let model_x = area
        .x
        .saturating_add(UnicodeWidthStr::width("ZEX") as u16)
        .saturating_add(separator_width);
    let model_width =
        (UnicodeWidthStr::width(model.as_str()) as u16).min(area.right().saturating_sub(model_x));
    app.register_hit(
        Rect::new(model_x, area.y, model_width, 1),
        HitTarget::StatusModel,
    );
    let thinking_x = model_x
        .saturating_add(model_width)
        .saturating_add(separator_width);
    let thinking = thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference));
    let thinking_width =
        (UnicodeWidthStr::width(thinking) as u16).min(area.right().saturating_sub(thinking_x));
    app.register_hit(
        Rect::new(thinking_x, area.y, thinking_width, 1),
        HitTarget::StatusThinking,
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
                Style::default().fg(ACCENT_PRIMARY),
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
                Style::default().fg(ACCENT_PRIMARY),
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
            Style::default().fg(TEXT_STRONG),
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
        && !app.input_focused
        && let Some(placeholder) = placeholder
        && editor_area.width > 0
    {
        frame.render_widget(
            Paragraph::new(placeholder)
                .style(Style::default().bg(background))
                .wrap(Wrap { trim: true }),
            editor_area,
        );
    }

    let cursor_y = metrics.cursor_row.saturating_sub(vertical_scroll) as u16;
    if app.input_focused && !app.output_panel_open() {
        frame.set_cursor_position((
            editor_area.x + metrics.cursor_column.min(editor_width - 1) as u16,
            editor_area.y + cursor_y.min(editor_area.height.saturating_sub(1)),
        ));
    }
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
        let text = if app
            .session_picker
            .as_ref()
            .is_some_and(|picker| picker.sessions.is_empty())
        {
            "Esc/q return"
        } else {
            "↑↓/jk select · Enter/Space resume · click twice · Esc/q cancel"
        };
        Line::from(Span::styled(text, Style::default().fg(TEXT_DIM)))
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
        if area.width >= 72 {
            aligned_hint_line(
                "Esc interrupt  ·  Working",
                "Tab browse  ·  Ctrl+P commands  ·  / actions",
                area.width as usize,
            )
        } else if area.width >= 34 {
            Line::from(Span::styled(
                "Esc interrupt  ·  Working",
                Style::default().fg(TEXT_FAINT),
            ))
        } else {
            Line::from(Span::styled(
                "Esc interrupt",
                Style::default().fg(TEXT_FAINT),
            ))
        }
    } else if !app.input_focused && app.selected_entry.is_some() {
        let text = if area.width >= 58 {
            "Tab browse · Enter open/toggle · Space compose · Esc clear"
        } else if area.width >= 36 {
            "Enter open · Space compose · Esc clear"
        } else {
            "Space compose · Esc clear"
        };
        Line::from(Span::styled(text, Style::default().fg(TEXT_FAINT)))
    } else {
        if area.width >= 72 {
            aligned_hint_line(
                "Enter send  ·  Shift+Enter newline",
                "Tab browse  ·  / actions",
                area.width as usize,
            )
        } else if area.width >= 40 {
            Line::from(Span::styled(
                "Enter send  ·  Tab browse  ·  / actions",
                Style::default().fg(TEXT_FAINT),
            ))
        } else {
            Line::from(Span::styled(
                "Enter send  ·  Tab browse",
                Style::default().fg(TEXT_FAINT),
            ))
        }
    };
    frame.render_widget(Paragraph::new(hint), area);
}

fn aligned_hint_line(left: &str, right: &str, width: usize) -> Line<'static> {
    let left_width = UnicodeWidthStr::width(left);
    let right_width = UnicodeWidthStr::width(right);
    let spacer = width.saturating_sub(left_width + right_width);
    Line::from(vec![
        Span::styled(left.to_owned(), Style::default().fg(TEXT_FAINT)),
        Span::raw(" ".repeat(spacer)),
        Span::styled(right.to_owned(), Style::default().fg(TEXT_FAINT)),
    ])
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
