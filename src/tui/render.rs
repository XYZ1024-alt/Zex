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
    render_location_header(frame, regions.header, app);

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
        render_turn_status(frame, regions.turn_status, app);
    }
    render_footer(frame, regions.footer, app);
    render_keymap(frame, regions.keymap, app);
    if app.output_panel_open() {
        render_output_panel(frame, area, app);
    }
}

fn render_location_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.is_empty() {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    frame.render_widget(
        Paragraph::new(location_line(app, area.width as usize))
            .style(Style::default().bg(BACKGROUND)),
        area,
    );
}

fn location_line(app: &App, width: usize) -> Line<'static> {
    let mut spans = Vec::new();
    let show_branch = width >= 28;
    if show_branch && let Some(git) = &app.git_status {
        let branch = if width < 48 {
            truncate_display(&git.branch, 12)
        } else {
            git.branch.clone()
        };
        spans.push(Span::styled(
            branch,
            Style::default().fg(TEXT).add_modifier(Modifier::DIM),
        ));
        if git.dirty_count > 0 {
            spans.push(Span::styled(
                format!(" *{}", git.dirty_count),
                Style::default().fg(TEXT_FAINT),
            ));
        }
        spans.push(Span::raw("  "));
    }
    let path = collapse_home(&app.working_dir);
    let path = if width < 28 {
        truncate_display(&path, width)
    } else {
        path
    };
    spans.push(Span::styled(path, Style::default().fg(GRAY_DIM)));
    Line::from(truncate_spans(spans, width))
}

fn collapse_home(path: &Path) -> String {
    let displayed = path.display().to_string();
    let home = std::env::var("USERPROFILE")
        .ok()
        .or_else(|| std::env::var("HOME").ok());
    let Some(home) = home else {
        return displayed;
    };
    let home_path = PathBuf::from(&home);
    if let Ok(stripped) = path.strip_prefix(&home_path) {
        if stripped.as_os_str().is_empty() {
            return "~".to_owned();
        }
        return format!("~{}{}", std::path::MAIN_SEPARATOR, stripped.display());
    }
    if let Some(rest) = displayed.strip_prefix(&home) {
        if rest.is_empty() {
            return "~".to_owned();
        }
        return format!("~{rest}");
    }
    displayed
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    render_input_frame(frame, area, app);
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
    let header_height = if area.height >= 7 {
        4
    } else {
        2.min(area.height)
    };
    let header = Rect::new(area.x, area.y, area.width, header_height);
    let list = Rect::new(
        area.x,
        header.bottom(),
        area.width,
        area.height.saturating_sub(header_height),
    );
    let inner_width = area.width.saturating_sub(6) as usize;
    let max_visible = list.height.max(1) as usize;
    let visible_count = picker.sessions.len().clamp(1, max_visible.max(1));
    render_page_header(
        frame,
        header,
        "Session index",
        "Resume work from a recent conversation",
        format!("{} saved", picker.sessions.len()),
    );
    if header.height >= 4 {
        frame.render_widget(
            Paragraph::new(Span::styled(
                "Updated most recently first",
                Style::default().fg(TEXT_FAINT),
            )),
            Rect::new(header.x, header.y + 2, header.width, 1),
        );
    }

    if picker.sessions.is_empty() {
        if list.height >= 1 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "No saved sessions",
                    Style::default()
                        .fg(TEXT_STRONG)
                        .add_modifier(Modifier::BOLD),
                )),
                Rect::new(list.x, list.y, list.width, 1),
            );
        }
        if list.height >= 2 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "Start a conversation and it will appear here.",
                    Style::default().fg(TEXT_DIM),
                )),
                Rect::new(list.x, list.y + 1, list.width, 1),
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
            let background = row_highlight(selected || armed, hovered);
            let item = Rect::new(
                list.x,
                list.y.saturating_add(offset as u16),
                list.width,
                1,
            );
            if item.is_empty() {
                continue;
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
            let preview = single_line(&session.preview, inner_width.saturating_sub(24));
            let left = format!("{id}  {preview}");
            let left = truncate_display(&left, inner_width.saturating_sub(metadata.len() + 2));
            let spacer = inner_width
                .saturating_sub(UnicodeWidthStr::width(left.as_str()))
                .saturating_sub(UnicodeWidthStr::width(metadata.as_str()));
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(left, Style::default().fg(foreground).add_modifier(Modifier::BOLD)),
                    Span::raw(" ".repeat(spacer)),
                    Span::styled(metadata, Style::default().fg(secondary)),
                ]))
                .style(Style::default().bg(background)),
                item,
            );
            app.register_hit(item, HitTarget::Session(index));
        }
    }
}

fn row_highlight(selected: bool, hovered: bool) -> Color {
    if selected {
        SURFACE_RAISED
    } else if hovered {
        SURFACE_HOVER
    } else {
        BACKGROUND
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
    let header_height = if area.height >= 6 { 3 } else { 1 };
    let header = Rect::new(area.x, area.y, area.width, header_height);
    let body = Rect::new(
        area.x,
        header.bottom(),
        area.width,
        area.height.saturating_sub(header_height),
    );
    render_page_header(
        frame,
        header,
        "Models",
        "Choose the engine for the next turn",
        format!("Current: {current}"),
    );
    let mut lines = Vec::new();

    if picker.choices.is_empty() {
        lines.extend([
            Line::from(Span::styled(
                "No configured models",
                Style::default().fg(TEXT_STRONG),
            )),
            Line::from(Span::styled(
                "Configure a provider with /provider.",
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
                lines.push(Line::from(vec![
                    Span::styled("─ ", Style::default().fg(TEXT_FAINT)),
                    Span::styled(
                        provider.to_owned(),
                        Style::default()
                            .fg(ACCENT_PRIMARY)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
            }
            let target = HitTarget::Model(index);
            let selected = index == picker.selected;
            let current = active
                .as_ref()
                .is_some_and(|active| *active == choice.target);
            let hovered = app.hovered(&target);
            let armed = app.armed(&target);
            let background = row_highlight(selected || armed, hovered);
            let marker = if selected { " " } else { " " };
            let current_marker = if current { "●" } else { " " };
            let thinking = choice.thinking.summary();
            let row_width = area.width as usize;
            let row = if area.width >= 96 {
                let think = truncate_display(
                    &format!("think {thinking}"),
                    row_width.saturating_sub(4 + 32 + 30),
                );
                Line::from(truncate_spans(
                    vec![
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
                        Span::styled(think, Style::default().fg(TEXT_DIM)),
                    ],
                    row_width,
                ))
            } else if area.width >= 64 {
                let think = truncate_display(
                    &format!("think {thinking}"),
                    row_width.saturating_sub(4 + 22 + 16),
                );
                Line::from(truncate_spans(
                    vec![
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
                            pad_display(&single_line(&choice.model_name, 20), 22),
                            Style::default().fg(TEXT_STRONG),
                        ),
                        Span::styled(
                            pad_display(&single_line(&choice.target.model_id, 14), 16),
                            Style::default().fg(TEXT_DIM),
                        ),
                        Span::styled(think, Style::default().fg(TEXT_DIM)),
                    ],
                    row_width,
                ))
            } else {
                let name_budget = row_width.saturating_sub(18).max(4);
                Line::from(truncate_spans(
                    vec![
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
                            single_line(&choice.model_name, name_budget),
                            Style::default().fg(TEXT_STRONG),
                        ),
                        Span::styled(
                            format!(" · think {thinking}"),
                            Style::default().fg(TEXT_DIM),
                        ),
                    ],
                    row_width,
                ))
            };
            lines.push(row.style(Style::default().bg(background)));
        }
    }

    let mut y = body.y;
    let mut provider = "";
    for (index, choice) in picker.choices.iter().enumerate() {
        if choice.provider_name != provider {
            if !provider.is_empty() {
                y = y.saturating_add(1);
            }
            provider = &choice.provider_name;
            y = y.saturating_add(1);
        }
        if y < body.bottom() {
            app.register_hit(Rect::new(body.x, y, body.width, 1), HitTarget::Model(index));
        }
        y = y.saturating_add(1);
    }
    frame.render_widget(
        Paragraph::new(lines)
            .style(Style::default().bg(BACKGROUND))
            .wrap(Wrap { trim: false }),
        body,
    );
}

fn render_page_header(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    subtitle: &str,
    meta: String,
) {
    if area.is_empty() {
        return;
    }
    let width = area.width as usize;
    let meta = truncate_display(
        &meta,
        width.saturating_sub(UnicodeWidthStr::width(title).min(width) + 1),
    );
    let title = truncate_display(
        title,
        width.saturating_sub(UnicodeWidthStr::width(meta.as_str()) + 1),
    );
    let title_width = UnicodeWidthStr::width(title.as_str());
    let meta_width = UnicodeWidthStr::width(meta.as_str());
    let spacer = width.saturating_sub(title_width + meta_width);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                title,
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" ".repeat(spacer)),
            Span::styled(meta, Style::default().fg(TEXT_FAINT)),
        ]))
        .style(Style::default().bg(BACKGROUND)),
        Rect::new(area.x, area.y, area.width, 1),
    );
    if area.height >= 2 {
        frame.render_widget(
            Paragraph::new(Span::styled(
                truncate_display(subtitle, area.width as usize),
                Style::default().fg(TEXT_DIM),
            )),
            Rect::new(area.x, area.y + 1, area.width, 1),
        );
    }
}

fn render_provider_editor(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let Some(editor) = app.provider_editor.clone() else {
        return;
    };
    if area.is_empty() {
        return;
    }
    let area = content_area(area);
    let page_header_height = if area.height >= 7 { 3 } else { 1 };
    let page_header = Rect::new(area.x, area.y, area.width, page_header_height);
    render_page_header(
        frame,
        page_header,
        "Providers",
        "Manage endpoints, credentials, and available models",
        if editor.dirty() {
            "unsaved changes".to_owned()
        } else {
            "saved".to_owned()
        },
    );
    let area = Rect::new(
        area.x,
        page_header.bottom(),
        area.width,
        area.height.saturating_sub(page_header_height),
    );
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
            "Connections",
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("{} configured", editor.draft.providers.len()),
            Style::default().fg(TEXT_FAINT),
        )),
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
                        if selected { "  " } else { "  " },
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
    pub(super) header: Rect,
    pub(super) transcript: Rect,
    pub(super) turn_status: Rect,
    pub(super) completion: Rect,
    pub(super) footer: Rect,
    pub(super) keymap: Rect,
}

pub(super) fn ui_regions(area: Rect, app: &App) -> UiRegions {
    let header_height: u16 = u16::from(area.height >= 2);
    let keymap_height = u16::from(area.height >= 3);
    let compact_chrome = area.height < 10;
    let turn_status_height = u16::from(
        app.busy && !app.page_open() && area.height >= header_height + keymap_height + 6,
    );
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
    // Editor rows plus rounded prompt chrome (top + info).
    let preferred_footer_height = if app.page_open() {
        1
    } else if compact_chrome {
        2
    } else {
        (requested_input_rows as u16).saturating_add(2)
    };
    let footer_height = if area.height
        >= header_height
            .saturating_add(keymap_height)
            .saturating_add(turn_status_height)
            .saturating_add(MIN_TRANSCRIPT_HEIGHT)
            .saturating_add(1)
    {
        preferred_footer_height
            .min(
                area.height
                    .saturating_sub(header_height)
                    .saturating_sub(keymap_height)
                    .saturating_sub(turn_status_height)
                    .saturating_sub(MIN_TRANSCRIPT_HEIGHT),
            )
            .max(1)
    } else {
        area.height
            .saturating_sub(header_height.saturating_add(keymap_height))
            .min(2)
    };
    let fixed_height = header_height + footer_height + keymap_height + turn_status_height;
    let remaining = area.height.saturating_sub(fixed_height);
    let transcript_reserve = MIN_TRANSCRIPT_HEIGHT.min(remaining);
    let completion_width = footer_input_width(area.width);
    let completion_height =
        completion_height(app, completion_width).min(remaining.saturating_sub(transcript_reserve));
    let transcript_height = remaining.saturating_sub(completion_height);

    let [header, transcript, completion, turn_status, footer, keymap] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header_height),
            Constraint::Length(transcript_height),
            Constraint::Length(completion_height),
            Constraint::Length(turn_status_height),
            Constraint::Length(footer_height),
            Constraint::Length(keymap_height),
        ])
        .areas(area);

    UiRegions {
        header,
        transcript,
        turn_status,
        completion,
        footer,
        keymap,
    }
}

#[cfg(test)]
pub(super) fn regions_inside_frame(area: Rect, regions: &UiRegions) -> bool {
    [
        regions.header,
        regions.transcript,
        regions.turn_status,
        regions.completion,
        regions.footer,
        regions.keymap,
    ]
    .into_iter()
    .all(|region| region.is_empty() || rect_inside(area, region))
}

#[cfg(test)]
pub(super) fn regions_non_overlapping(regions: &UiRegions) -> bool {
    let occupied = [
        regions.header,
        regions.transcript,
        regions.completion,
        regions.footer,
        regions.keymap,
    ]
    .into_iter()
    .filter(|region| !region.is_empty())
    .collect::<Vec<_>>();
    occupied.iter().enumerate().all(|(index, left)| {
        occupied
            .iter()
            .skip(index + 1)
            .all(|right| !rects_overlap(*left, *right))
    })
}

#[cfg(test)]
fn rect_inside(outer: Rect, inner: Rect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && inner.right() <= outer.right()
        && inner.bottom() <= outer.bottom()
}

#[cfg(test)]
fn rects_overlap(left: Rect, right: Rect) -> bool {
    left.x < right.right()
        && right.x < left.right()
        && left.y < right.bottom()
        && right.y < left.bottom()
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
            .border_type(BorderType::Rounded)
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
                "◆ ZEX / ",
                Style::default()
                    .fg(ACCENT_PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "assistant output · Response reader",
                Style::default().fg(TEXT_STRONG),
            ),
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
                "↑↓ read  │  PgUp/PgDn jump  │  Space page  │  Esc timeline",
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
    // Pre-wrapped lines: do not wrap again or logical panel rows drift from
    // the painted rows (gray slabs / clipped bodies / smashed glyphs).
    let paragraph = Paragraph::new(transcript.text).style(Style::default().fg(TEXT).bg(BACKGROUND));
    let line_count = transcript.line_count;
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



#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LandingRegions {
    pub(super) top: Rect,
    pub(super) logo: Rect,
    pub(super) hero: Rect,
    pub(super) menu: Rect,
    pub(super) card: Rect,
    pub(super) status: Rect,
    pub(super) completion: Rect,
}

const PROMPT_HEIGHT: u16 = 3;
const HERO_V_PAD: u16 = 1;
const HERO_LOGO_PAD: u16 = 3;

pub(super) fn landing_regions(area: Rect, app: &App) -> LandingRegions {
    let empty = Rect::new(area.x, area.y, 0, 0);
    if area.is_empty() {
        return LandingRegions {
            top: area,
            logo: empty,
            hero: empty,
            menu: empty,
            card: area,
            status: empty,
            completion: empty,
        };
    }

    let top_height = u16::from(area.height >= 2);
    let status_height = u16::from(area.height >= 3);
    let top = Rect::new(area.x, area.y, area.width, top_height);
    let status = Rect::new(
        area.x,
        area.bottom().saturating_sub(status_height),
        area.width,
        status_height,
    );
    let stage = Rect::new(
        area.x,
        top.bottom(),
        area.width,
        area.height.saturating_sub(top_height + status_height),
    );
    let prompt_width = stage.width.saturating_sub(4).max(1);
    let input_rows = input_metrics(
        &app.input.content,
        app.input.cursor,
        prompt_width.saturating_sub(6).max(1) as usize,
    )
    .total_rows
    .clamp(1, MAX_INPUT_ROWS) as u16;
    let mut card_height = match stage.height {
        8.. => input_rows.saturating_add(2).max(PROMPT_HEIGHT),
        5..=7 => 3,
        3..=4 => 2,
        1..=2 => 1,
        _ => 0,
    }
    .min(stage.height);
    let completion_desired = if app.completion_open() {
        completion_height(app, prompt_width)
    } else {
        0
    };
    let mut completion_h = completion_desired.min(stage.height.saturating_sub(card_height));
    let mut above = stage
        .height
        .saturating_sub(card_height)
        .saturating_sub(completion_h)
        .saturating_sub(u16::from(completion_h > 0));

    let mut menu_height = if completion_desired > 0 {
        0
    } else if above >= 6 {
        5
    } else if above >= 3 {
        2
    } else {
        0
    };
    menu_height = menu_height.min(above);
    above = above.saturating_sub(menu_height);

    let window_height = area.height;
    let stacked_art = crate::tui::glyphs::logo_art(crate::tui::glyphs::logo_tier(window_height));
    let hero_art = crate::tui::glyphs::hero_logo_art();
    let hero_logo_rows = hero_art.map_or(0, crate::tui::glyphs::logo_line_count);
    let right_col = 1 + u16::from(menu_height > 0) + menu_height;
    let hero_inner = hero_logo_rows.max(right_col);
    let hero_box_height = 2 + HERO_V_PAD * 2 + hero_inner;
    let want_hero = stage.width >= crate::tui::glyphs::HERO_BOX_MIN_WIDTH
        && hero_art.is_some()
        && stage.height >= hero_box_height + card_height + 1
        && completion_desired == 0
        && menu_height > 0;

    let (hero, logo, menu, card, completion) = if want_hero {
        let box_width = stage.width.saturating_sub(6).min(120).max(1);
        let remaining = stage.height.saturating_sub(hero_box_height + card_height);
        let top_pad = remaining / 3;
        let hero_x = stage.x + stage.width.saturating_sub(box_width) / 2;
        let hero = Rect::new(hero_x, stage.y + top_pad, box_width, hero_box_height);
        let inner = Rect::new(
            hero.x + 1,
            hero.y + 1 + HERO_V_PAD,
            hero.width.saturating_sub(2),
            hero_inner,
        );
        let logo_w = hero_art
            .map(crate::tui::glyphs::logo_visual_width)
            .unwrap_or(0);
        let left_w = if logo_w == 0 {
            2
        } else {
            logo_w + HERO_LOGO_PAD
        };
        let logo = Rect::new(inner.x, inner.y, left_w.min(inner.width), hero_logo_rows.min(inner.height));
        let right_x = inner.x.saturating_add(left_w.min(inner.width));
        let inset = 2u16.min(inner.width.saturating_sub(left_w.min(inner.width)) / 2);
        let right_w = inner
            .width
            .saturating_sub(left_w.min(inner.width))
            .saturating_sub(inset);
        let menu = Rect::new(
            right_x,
            inner.y.saturating_add(2).min(inner.bottom()),
            right_w,
            menu_height.min(inner.bottom().saturating_sub(inner.y.saturating_add(2))),
        );
        let card = Rect::new(
            stage.x + 2,
            stage.bottom().saturating_sub(card_height),
            stage.width.saturating_sub(4),
            card_height,
        );
        (hero, logo, menu, card, empty)
    } else {
        loop {
            let logo_h = stacked_art
                .map(crate::tui::glyphs::logo_line_count)
                .unwrap_or(0);
            let logo_gap = u16::from(logo_h > 0 && menu_height > 0);
            let used = logo_h + logo_gap + menu_height + completion_h
                + u16::from(completion_h > 0)
                + card_height;
            if used <= stage.height {
                break;
            }
            if menu_height > 2 {
                menu_height = 2;
            } else if menu_height > 0 {
                menu_height = 0;
            } else if completion_h > 2 {
                completion_h -= 1;
            } else if card_height > 2 {
                card_height -= 1;
            } else if completion_h > 0 {
                completion_h -= 1;
            } else {
                break;
            }
        }
        let logo_h = if completion_h > 0 {
            0
        } else {
            stacked_art
                .map(crate::tui::glyphs::logo_line_count)
                .unwrap_or(0)
                .min(stage.height.saturating_sub(card_height + menu_height))
        };
        let card_x = stage.x + 2;
        let card_w = stage.width.saturating_sub(4);
        let _lower = completion_h + u16::from(completion_h > 0) + card_height;
        let card = Rect::new(
            card_x,
            stage.bottom().saturating_sub(card_height),
            card_w,
            card_height,
        );
        let completion = Rect::new(
            card_x,
            card.y.saturating_sub(completion_h + u16::from(completion_h > 0)),
            card_w,
            completion_h,
        );
        let upper_bottom = if completion_h > 0 {
            completion.y
        } else {
            card.y
        };
        let upper_h = upper_bottom.saturating_sub(stage.y);
        let logo_gap = u16::from(logo_h > 0 && menu_height > 0);
        let group = logo_h + logo_gap + menu_height;
        let top_pad = upper_h.saturating_sub(group) / 3;
        let logo_w = stacked_art
            .map(crate::tui::glyphs::logo_visual_width)
            .unwrap_or(24)
            .min(stage.width);
        let logo_x = stage.x + stage.width.saturating_sub(logo_w) / 2;
        let logo = Rect::new(logo_x, stage.y + top_pad, logo_w, logo_h);
        let menu_w = welcome_menu_width(welcome_actions_for_rows(menu_height), stage.width);
        let menu_x = stage.x + stage.width.saturating_sub(menu_w) / 2;
        let menu = Rect::new(
            menu_x,
            logo.bottom().saturating_add(logo_gap),
            menu_w,
            menu_height,
        );
        (empty, logo, menu, card, completion)
    };

    let _ = above;
    LandingRegions {
        top,
        logo,
        hero,
        menu,
        card,
        status,
        completion,
    }
}

fn render_landing(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let regions = landing_regions(area, app);
    app.welcome_menu_len = welcome_actions_for_rows(regions.menu.height).len();
    if app
        .welcome_menu_selected
        .is_some_and(|index| index >= app.welcome_menu_len)
    {
        app.welcome_menu_selected = app.welcome_menu_len.checked_sub(1);
    }

    if !regions.top.is_empty() {
        let top = content_area(regions.top);
        frame.render_widget(
            Paragraph::new(location_line(app, top.width as usize))
                .style(Style::default().bg(BACKGROUND)),
            top,
        );
    }

    if !regions.hero.is_empty() {
        render_hero_box(frame, &regions, app);
    } else if !regions.logo.is_empty() {
        let art = crate::tui::glyphs::logo_art(crate::tui::glyphs::logo_tier(area.height));
        render_logo_art(frame, regions.logo, art);
        render_landing_menu(frame, regions.menu, app);
    } else {
        render_landing_menu(frame, regions.menu, app);
    }

    render_landing_card(frame, regions.card, app);
    render_product_badge(frame, regions.status);

    if !regions.completion.is_empty() {
        render_completion(frame, regions.completion, app);
    }
}

fn render_hero_box(frame: &mut Frame<'_>, regions: &LandingRegions, app: &mut App) {
    let border = Style::default().fg(BORDER);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(border)
            .style(Style::default().bg(BACKGROUND)),
        regions.hero,
    );
    render_logo_art(frame, regions.logo, crate::tui::glyphs::hero_logo_art());
    let version_row = Rect::new(
        regions.menu.x,
        regions.logo.y,
        regions.menu.width,
        1,
    );
    if !version_row.is_empty() && regions.menu.y > regions.logo.y {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    PRODUCT_NAME,
                    Style::default()
                        .fg(TEXT_STRONG)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {}", env!("CARGO_PKG_VERSION")),
                    Style::default().fg(TEXT_FAINT),
                ),
            ])),
            version_row,
        );
    }
    render_landing_menu(frame, regions.menu, app);
}

fn render_logo_art(frame: &mut Frame<'_>, area: Rect, art: Option<&str>) {
    let Some(art) = art else {
        return;
    };
    if area.is_empty() {
        return;
    }
    let lines: Vec<Line> = art
        .lines()
        .filter(|line| !line.is_empty())
        .take(area.height as usize)
        .enumerate()
        .map(|(row, line)| {
            Line::from(
                line.chars()
                    .enumerate()
                    .map(|(column, character)| {
                        Span::styled(
                            character.to_string(),
                            Style::default().fg(logo_shimmer_color(row, column)),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
            .alignment(Alignment::Center)
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), area);
}

/// Grok-style logo shimmer: a raised-cosine sheen sweeps the braille mark
/// diagonally every 4s, layered over a slow 5s breathing pulse.
fn logo_shimmer_color(row: usize, column: usize) -> Color {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0);
    let tick = millis / 83; // ~12fps, grok's shimmer cadence
    let diagonal = (row + column) as u64;
    let sweep = (tick * 83) % 4000;
    let mut brightness = 0.3_f64;
    if sweep < 1300 {
        let center = sweep as f64 / 1300.0 * 34.0 - 4.0;
        let distance = (diagonal as f64 - center).abs();
        if distance < 4.0 {
            brightness += 0.7 * (0.5 + 0.5 * (std::f64::consts::PI * distance / 4.0).cos());
        }
    }
    let breath = ((tick * 83) % 5000) as f64 / 5000.0 * std::f64::consts::TAU;
    brightness *= 0.85 + 0.15 * breath.sin().abs();
    blend_color(GRAY_DIM, TEXT_STRONG, brightness)
}

/// Linear RGB blend, the single primitive behind all grok-style animation.
fn blend_color(from: Color, to: Color, t: f64) -> Color {
    let (Color::Rgb(fr, fg, fb), Color::Rgb(tr, tg, tb)) = (from, to) else {
        return to;
    };
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8| (a as f64 + (b as f64 - a as f64) * t).round() as u8;
    Color::Rgb(lerp(fr, tr), lerp(fg, tg), lerp(fb, tb))
}

fn render_product_badge(frame: &mut Frame<'_>, area: Rect) {
    if area.is_empty() {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                PRODUCT_NAME,
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {}", env!("CARGO_PKG_VERSION")),
                Style::default().fg(TEXT_FAINT),
            ),
        ]))
        .alignment(Alignment::Right)
        .style(Style::default().bg(BACKGROUND)),
        area,
    );
}

fn render_landing_menu(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    if area.is_empty() {
        return;
    }
    let actions = welcome_actions_for_rows(area.height);
    let width = area.width as usize;
    for (index, action) in actions.iter().enumerate() {
        if index as u16 >= area.height {
            break;
        }
        let row = Rect::new(area.x, area.y + index as u16, area.width, 1);
        app.register_hit(row, HitTarget::WelcomeAction(action.kind));
        let selected = app.welcome_menu_selected == Some(index)
            || app.hovered(&HitTarget::WelcomeAction(action.kind));
        let shortcut_width = UnicodeWidthStr::width(action.shortcut);
        let label_budget = width.saturating_sub(shortcut_width.saturating_add(2));
        let label = truncate_display(action.label, label_budget);
        let shortcut = if shortcut_width + 2 <= width {
            action.shortcut
        } else {
            ""
        };
        let spacer = width
            .saturating_sub(UnicodeWidthStr::width(label.as_str()))
            .saturating_sub(UnicodeWidthStr::width(shortcut));
        let (label_style, shortcut_style, background) = if selected {
            (
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
                Style::default().fg(TEXT_DIM),
                SURFACE,
            )
        } else {
            (
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                Style::default().fg(TEXT_FAINT),
                BACKGROUND,
            )
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(label, label_style),
                Span::raw(" ".repeat(spacer)),
                Span::styled(shortcut.to_owned(), shortcut_style),
            ]))
            .style(Style::default().bg(background)),
            row,
        );
    }
}

fn welcome_menu_width(actions: &[WelcomeAction], max_width: u16) -> u16 {
    let content = actions
        .iter()
        .map(|action| {
            UnicodeWidthStr::width(action.label) + 4 + UnicodeWidthStr::width(action.shortcut)
        })
        .max()
        .unwrap_or(24) as u16;
    content.max(24).min(max_width.max(1))
}

fn render_landing_card(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let placeholder = Line::from(Span::styled(
        "Type a message...",
        Style::default().fg(TEXT_FAINT),
    ));
    paint_composer(frame, area, app, Some(placeholder), false);
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
            let background = row_highlight(selected || armed, hovered);
            let command_style = Style::default()
                .fg(if selected { TEXT_STRONG } else { ACCENT_PRIMARY })
                .bg(background)
                .add_modifier(if selected {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                });
            let description_style = Style::default()
                .fg(if selected { TEXT } else { TEXT_DIM })
                .bg(background);
            let row_style = Style::default().bg(background);
            let available = inner_width.saturating_sub(2);
            let wide = available >= usage_width + 2 + 18;
            if wide {
                Line::from(truncate_spans(
                    vec![
                        Span::styled(pad_display(command.usage, usage_width), command_style),
                        Span::raw("  "),
                        Span::styled(command.description.to_owned(), description_style),
                    ],
                    inner_width,
                ))
                .style(row_style)
            } else {
                Line::from(truncate_spans(
                    vec![
                        Span::styled(command.usage.to_owned(), command_style),
                        Span::raw("  "),
                        Span::styled(command.description.to_owned(), description_style),
                    ],
                    inner_width,
                ))
                .style(row_style)
            }
        })
        .collect::<Vec<_>>();
    let palette = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(BACKGROUND))
        .padding(ratatui::widgets::Padding::horizontal(1));
    let inner = palette.inner(area);
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
            .block(palette)
            .style(Style::default().bg(BACKGROUND))
            .wrap(Wrap { trim: false }),
        area,
    );
}

struct TranscriptRender {
    text: Text<'static>,
    line_count: usize,
    #[allow(dead_code)]
    panels: Vec<TranscriptPanel>,
    card_lines: Vec<(usize, usize)>,
    output_lines: Vec<(usize, usize)>,
    response_lines: Vec<(usize, usize, usize)>,
}

#[allow(dead_code)]
struct TranscriptPanel {
    start_line: usize,
    end_line: usize,
    width: usize,
    background: Color,
    accent: Option<Color>,
}

fn rail_spans(color: Color) -> Vec<Span<'static>> {
    vec![
        Span::styled(
            crate::tui::glyphs::accent_bar().to_owned(),
            Style::default().fg(color),
        ),
        Span::raw(" "),
    ]
}

fn header_row_style(selected: bool, hovered: bool) -> Style {
    Style::default().bg(row_highlight(selected, hovered))
}

/// Extend a banded line to the full content width: ratatui only paints line
/// style over existing graphemes, so pad with background-colored spaces.
fn pad_line_band(line: &mut Line<'static>, width: usize) {
    let Some(bg) = line.style.bg else {
        return;
    };
    let used = line.width();
    if used < width {
        line.spans.push(Span::styled(
            " ".repeat(width - used),
            Style::default().bg(bg),
        ));
    }
}

fn transcript_text(app: &App, width: usize) -> TranscriptRender {
    let content_width = width.max(1);
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
                append_markdown_lines(&mut lines, content, *role, final_answer, content_width);
                if *role == MessageRole::User {
                    // Grok-style raised band: user turns get a full-width
                    // surface background so they stay scannable in long
                    // transcripts.
                    for line in &mut lines[start_line..] {
                        if line.style.bg.is_none() {
                            line.style = line.style.patch(Style::default().bg(SURFACE));
                        }
                        pad_line_band(line, content_width);
                    }
                }
                panels.push(TranscriptPanel {
                    start_line,
                    end_line: lines.len(),
                    width,
                    background: BACKGROUND,
                    accent: Some(role_accent(*role)),
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
                        end_line: lines.len(),
                        width,
                        background: BACKGROUND,
                        accent: Some(ACCENT_THINKING),
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
                    background: BACKGROUND,
                    accent: Some(ACCENT_TOOL),
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
                        for segment in
                            wrap_display_hard(source_line, width.saturating_sub(6).max(1))
                        {
                            lines.push(Line::from(vec![
                                Span::styled("    │ ", Style::default().fg(TEXT_FAINT)),
                                Span::styled(segment, Style::default().fg(TEXT_DIM)),
                            ]));
                        }
                    }
                }
                panels.push(TranscriptPanel {
                    start_line,
                    end_line: lines.len(),
                    width,
                    background: BACKGROUND,
                    accent: None,
                });
                lines.push(Line::default());
            }
            TranscriptEntry::Turn(turn) => {
                let start_line = lines.len();
                append_turn_line(&mut lines, turn, content_width);
                panels.push(TranscriptPanel {
                    start_line,
                    end_line: lines.len(),
                    width,
                    background: BACKGROUND,
                    accent: None,
                });
            }
        }
        append_transcript_gap(&mut lines, entry, next);
    }
    let line_count = lines.len();
    TranscriptRender {
        text: Text::from(lines),
        line_count,
        panels,
        card_lines,
        output_lines,
        response_lines,
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
    let compact = matches!(
        entry,
        TranscriptEntry::Thinking(_) | TranscriptEntry::Tool(_)
    ) && matches!(
        next,
        TranscriptEntry::Thinking(_) | TranscriptEntry::Tool(_)
    );
    if !compact {
        lines.push(Line::default());
    }
}

fn render_help_page(frame: &mut Frame<'_>, viewport: Rect, app: &mut App) {
    if viewport.is_empty() {
        return;
    }

    let area = horizontal_inset(viewport, HORIZONTAL_GUTTER);
    let header_height = if area.height >= 5 { 3 } else { 1 };
    let header = Rect::new(area.x, area.y, area.width, header_height);
    render_page_header(
        frame,
        header,
        "Command palette",
        "Run a command or inspect its purpose",
        format!("{} actions", command_specs().len()),
    );
    let body = Rect::new(
        area.x,
        header.bottom(),
        area.width,
        area.height.saturating_sub(header_height),
    );
    let inner_width = body.width as usize;
    let lines = help_lines(app, inner_width);
    for index in 0..command_specs().len() {
        let y = body.y.saturating_add(index as u16);
        if y < body.bottom() {
            app.register_hit(Rect::new(body.x, y, body.width, 1), HitTarget::Help(index));
        }
    }
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), body);
}

fn help_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let usage_width = command_specs()
        .iter()
        .map(|command| command.usage.len())
        .max()
        .unwrap_or(0);
    let wide = width >= usage_width + 2 + 24;
    command_specs()
        .iter()
        .enumerate()
        .map(|(index, command)| {
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
            let marker = if selected { "  " } else { "  " };
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
        })
        .collect()
}

fn append_turn_line(lines: &mut Vec<Line<'static>>, turn: &TurnEntry, width: usize) {
    let label = match turn.outcome {
        TurnOutcome::Done => "turn done",
        TurnOutcome::Failed => "turn failed",
        TurnOutcome::Stopped => "turn stopped",
    };
    let color = match turn.outcome {
        TurnOutcome::Done => TEXT_FAINT,
        TurnOutcome::Failed => BAD,
        TurnOutcome::Stopped => TEXT_FAINT,
    };
    let mut details = vec![label.to_owned()];
    details.push(format!(
        "{} tool{}",
        turn.tool_count,
        if turn.tool_count == 1 { "" } else { "s" }
    ));
    if let Some(elapsed) = turn.elapsed {
        details.push(format_turn_duration(elapsed));
    }
    if let Some(tokens) = turn.output_tokens {
        details.push(format_compact_count(tokens));
    }
    let summary = details.join("  ");
    lines.push(Line::from(truncate_spans(
        vec![
            Span::raw("  "),
            Span::styled(
                truncate_display(&summary, width.saturating_sub(2)),
                Style::default().fg(color),
            ),
        ],
        width,
    )));
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
                let mut band = Line::from(vec![
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
                ])
                .style(Style::default().bg(CODE_BG));
                pad_line_band(&mut band, width);
                lines.push(band);
            } else {
                let rail = message_rail(role, final_answer, &mut first_visual_line);
                let mut band = Line::from(vec![
                    Span::styled(
                        rail,
                        Style::default().fg(message_rail_color(role, final_answer)),
                    ),
                    Span::styled("▌", Style::default().fg(TEXT_FAINT)),
                ])
                .style(Style::default().bg(CODE_BG));
                pad_line_band(&mut band, width);
                lines.push(band);
            }
            continue;
        }
        if in_code_block {
            let content_width = width.saturating_sub(4).max(1);
            for segment in wrap_display_hard(source_line, content_width) {
                let rail = message_rail(role, final_answer, &mut first_visual_line);
                let mut band = Line::from(vec![
                    Span::styled(
                        rail,
                        Style::default().fg(message_rail_color(role, final_answer)),
                    ),
                    Span::styled("▌ ", Style::default().fg(TEXT_DIM)),
                    Span::styled(segment, Style::default().fg(TEXT_STRONG)),
                ])
                .style(Style::default().bg(CODE_BG));
                pad_line_band(&mut band, width);
                lines.push(band);
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
                    .fg(ACCENT_SECONDARY)
                    .add_modifier(Modifier::BOLD),
            );
        } else if let Some(heading) = trimmed.strip_prefix("## ") {
            append_wrapped_message_lines(
                lines,
                heading,
                role,
                final_answer,
                &mut first_visual_line,
                width,
                Style::default()
                    .fg(ACCENT_PRIMARY)
                    .add_modifier(Modifier::BOLD),
            );
        } else if let Some(heading) = trimmed.strip_prefix("# ") {
            append_wrapped_message_lines(
                lines,
                heading,
                role,
                final_answer,
                &mut first_visual_line,
                width,
                Style::default()
                    .fg(MODEL_ACCENT)
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

fn role_accent(role: MessageRole) -> Color {
    match role {
        MessageRole::User => ACCENT_USER,
        MessageRole::Assistant => ACCENT_ASSISTANT,
    }
}

/// User turns lead with a `❯` prompt (grok-style); continuation lines indent
/// to the same column. Assistant turns keep the accent rail.
fn message_rail(
    role: MessageRole,
    _final_answer: bool,
    first_visual_line: &mut bool,
) -> String {
    let first = *first_visual_line;
    *first_visual_line = false;
    match role {
        MessageRole::User if first => crate::tui::glyphs::prompt_arrow().to_owned(),
        MessageRole::User => "  ".to_owned(),
        MessageRole::Assistant => format!("{} ", crate::tui::glyphs::accent_bar()),
    }
}

fn message_rail_color(role: MessageRole, _final_answer: bool) -> Color {
    role_accent(role)
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
        MessageRole::User => 2,
        MessageRole::Assistant => 2,
    };
    let styled = inline_markdown_spans(content, content_style);
    let plain = strip_inline_markdown(content);
    let segments = wrap_display_words(&plain, width.saturating_sub(rail_width).max(1));
    if segments.len() == 1 {
        let rail = message_rail(role, final_answer, first_visual_line);
        let mut row = vec![Span::styled(
            rail,
            Style::default().fg(message_rail_color(role, final_answer)),
        )];
        row.extend(styled);
        lines.push(Line::from(row));
        return;
    }
    for segment in segments {
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

fn strip_inline_markdown(value: &str) -> String {
    value.replace("**", "").replace('`', "")
}

fn inline_markdown_spans(value: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let chars: Vec<char> = value.chars().collect();
    let mut index = 0;
    let mut buffer = String::new();
    while index < chars.len() {
        if chars[index] == '*'
            && chars.get(index + 1) == Some(&'*')
            && let Some(end) = chars[index + 2..]
                .windows(2)
                .position(|pair| pair == ['*', '*'])
        {
            if !buffer.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buffer), base));
            }
            let bold: String = chars[index + 2..index + 2 + end].iter().collect();
            spans.push(Span::styled(
                bold,
                base.fg(TEXT_STRONG).add_modifier(Modifier::BOLD),
            ));
            index += end + 4;
            continue;
        }
        if chars[index] == '`'
            && let Some(end) = chars[index + 1..].iter().position(|c| *c == '`')
        {
            if !buffer.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buffer), base));
            }
            let code: String = chars[index + 1..index + 1 + end].iter().collect();
            spans.push(Span::styled(
                code,
                base.fg(MD_CODE).add_modifier(Modifier::BOLD),
            ));
            index += end + 2;
            continue;
        }
        buffer.push(chars[index]);
        index += 1;
    }
    if !buffer.is_empty() {
        spans.push(Span::styled(buffer, base));
    }
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
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
        MessageRole::User => 2,
        MessageRole::Assistant => 2,
    };
    let content_width = width.saturating_sub(rail_width + marker_width).max(1);
    for (index, segment) in wrap_display_words(&strip_inline_markdown(content), content_width)
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
    _level: ThinkingLevel,
    selected: bool,
    _hovered: bool,
    width: usize,
) {
    let label_style = Style::default()
        .fg(if selected { TEXT_STRONG } else { TEXT })
        .add_modifier(Modifier::BOLD);
    let header = match status {
        ThinkingStatus::Active => "Thinking…".to_owned(),
        ThinkingStatus::Done => match thinking.elapsed.filter(|elapsed| *elapsed >= Duration::from_millis(100))
        {
            Some(elapsed) => format!("Thought for {}", format_turn_duration(elapsed)),
            None => "Thought".to_owned(),
        },
    };
    let mut header_spans = rail_spans(ACCENT_THINKING);
    header_spans.push(Span::styled(header, label_style));
    if status == ThinkingStatus::Done && !thinking.expanded {
        header_spans.push(Span::styled(
            "  (ctrl+e to expand)",
            Style::default().fg(GRAY_DIM),
        ));
    }
    lines.push(
        Line::from(truncate_spans(header_spans, width)).style(header_row_style(selected, false)),
    );

    if thinking.expanded {
        // Grok-style de-emphasis: thinking bodies render dim + italic.
        let body_style = Style::default()
            .fg(TEXT_FAINT)
            .add_modifier(Modifier::ITALIC);
        for line in thinking.content.split('\n') {
            for segment in wrap_display_hard(line, width.saturating_sub(2).max(1)) {
                let mut row = rail_spans(ACCENT_THINKING);
                row.push(Span::styled(segment, body_style));
                lines.push(Line::from(row));
            }
        }
    }
}

mod tool;

use tool::append_tool_lines;

pub(super) use tool::tool_detail_line_count;

fn render_input_frame(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    paint_composer(frame, area, app, None, !app.page_open());
}

fn composer_border_color(app: &App) -> Color {
    if app.busy || app.input_focused {
        BORDER_ACTIVE
    } else {
        BORDER
    }
}

#[cfg(test)]
pub(super) fn composer_editor_origin(area: Rect) -> (u16, u16) {
    if area.width >= 8 && area.height >= 3 {
        (area.x.saturating_add(4), area.y.saturating_add(1))
    } else if area.width >= 8 && area.height == 2 {
        (area.x.saturating_add(4), area.y)
    } else {
        (area.x.saturating_add(2), area.y)
    }
}

fn paint_composer(
    frame: &mut Frame<'_>,
    area: Rect,
    app: &mut App,
    placeholder: Option<Line<'static>>,
    show_info: bool,
) {
    if area.is_empty() {
        return;
    }
    app.register_hit(area, HitTarget::Input);
    let boxed = area.width >= 8 && area.height >= 2;
    if !boxed {
        if app.page_open() {
            paint_page_composer_label(frame, area, app);
        } else {
            paint_input_editor(frame, area, app, placeholder);
        }
        return;
    }

    let border = composer_border_color(app);
    let has_top = area.height >= 3;
    let has_bottom = area.height >= 2;
    let editor_y = area.y + u16::from(has_top);
    let editor_h = area
        .height
        .saturating_sub(u16::from(has_top) + u16::from(has_bottom))
        .max(1);
    let editor_row = Rect::new(area.x, editor_y, area.width, editor_h);
    if has_top {
        paint_box_edge(frame, area.x, area.y, area.width, '╭', '─', '╮', border);
        if !app.landing_visible() {
            paint_session_title(frame, area.x, area.y, area.width, app);
        }
    }
    paint_box_sides(frame, editor_row, border);
    let inner = Rect::new(
        editor_row.x.saturating_add(2),
        editor_row.y,
        editor_row.width.saturating_sub(4),
        editor_row.height,
    );
    if app.page_open() {
        paint_page_composer_label(frame, inner, app);
    } else {
        paint_input_editor(frame, inner, app, placeholder);
    }
    if has_bottom {
        let info_y = area.bottom().saturating_sub(1);
        paint_box_edge(frame, area.x, info_y, area.width, '╰', '─', '╯', border);
        if show_info && !app.page_open() {
            paint_composer_info(frame, area.x, info_y, area.width, app);
        }
    }
}

fn paint_box_edge(
    frame: &mut Frame<'_>,
    x: u16,
    y: u16,
    width: u16,
    left: char,
    mid: char,
    right: char,
    fg: Color,
) {
    if width == 0 {
        return;
    }
    let buf = frame.buffer_mut();
    for offset in 0..width {
        let ch = if offset == 0 {
            left
        } else if offset + 1 == width {
            right
        } else {
            mid
        };
        if let Some(cell) = buf.cell_mut((x + offset, y)) {
            cell.set_char(ch);
            cell.set_style(Style::default().fg(fg).bg(BACKGROUND));
        }
    }
}

fn paint_box_sides(frame: &mut Frame<'_>, area: Rect, fg: Color) {
    if area.width < 2 {
        return;
    }
    let buf = frame.buffer_mut();
    let style = Style::default().fg(fg).bg(BACKGROUND);
    for y in area.y..area.bottom() {
        if let Some(cell) = buf.cell_mut((area.x, y)) {
            cell.set_char('│');
            cell.set_style(style);
        }
        if let Some(cell) = buf.cell_mut((area.right().saturating_sub(1), y)) {
            cell.set_char('│');
            cell.set_style(style);
        }
        for x in area.x + 1..area.right().saturating_sub(1) {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_char(' ');
                cell.set_style(Style::default().fg(TEXT).bg(BACKGROUND));
            }
        }
    }
}

fn paint_page_composer_label(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.is_empty() {
        return;
    }
    let label = if app.model_picker_open() {
        "models"
    } else if app.provider_editor_open() {
        "providers"
    } else if app.session_picker_open() {
        "sessions"
    } else {
        "commands"
    };
    frame.render_widget(
        Paragraph::new(Span::styled(
            truncate_display(label, area.width as usize),
            Style::default().fg(TEXT_DIM),
        ))
        .style(Style::default().bg(BACKGROUND)),
        area,
    );
}

fn paint_input_editor(
    frame: &mut Frame<'_>,
    editor_area: Rect,
    app: &mut App,
    placeholder: Option<Line<'static>>,
) {
    if editor_area.is_empty() {
        return;
    }
    if app.busy {
        render_input_buffer(
            frame,
            editor_area,
            app,
            crate::tui::glyphs::prompt_arrow(),
            None,
            BACKGROUND,
        );
        return;
    }
    render_input_buffer(
        frame,
        editor_area,
        app,
        crate::tui::glyphs::prompt_arrow(),
        placeholder,
        BACKGROUND,
    );
}

fn paint_session_title(frame: &mut Frame<'_>, x: u16, y: u16, width: u16, app: &App) {
    let title = app.transcript.iter().find_map(|entry| match entry {
        TranscriptEntry::Message {
            role: MessageRole::User,
            content,
        } if !content.trim().is_empty() => Some(single_line(content, 48)),
        _ => None,
    });
    let Some(title) = title else {
        return;
    };
    let label = format!(" {title} ");
    let label_width = UnicodeWidthStr::width(label.as_str()) as u16;
    if label_width + 4 >= width {
        return;
    }
    frame.render_widget(
        Paragraph::new(Span::styled(
            label,
            Style::default().fg(TEXT_FAINT).bg(BACKGROUND),
        )),
        Rect::new(x + 2, y, label_width.min(width.saturating_sub(4)), 1),
    );
}

fn paint_composer_info(frame: &mut Frame<'_>, x: u16, y: u16, width: u16, app: &mut App) {
    if width < 8 {
        return;
    }
    let inner_x = x.saturating_add(2);
    let inner_width = width.saturating_sub(4);
    let meta = input_metadata_line(app, inner_width.saturating_sub(2) as usize);
    let mut spans = vec![Span::raw(" ")];
    spans.extend(meta.spans);
    spans.push(Span::raw(" "));
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(BACKGROUND)),
        Rect::new(inner_x, y, inner_width, 1),
    );
    register_input_metadata_hits(
        app,
        Rect::new(
            inner_x.saturating_add(1),
            y,
            inner_width.saturating_sub(2),
            1,
        ),
    );
}

fn input_metadata_line(app: &App, width: usize) -> Line<'static> {
    if width == 0 {
        return Line::default();
    }
    let model = model_short_name(&app.model);
    let thinking = thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference));
    let mode = truncate_display(&format!("{model} · think {thinking}"), width);
    let mode_width = UnicodeWidthStr::width(mode.as_str());
    // Grok-style chrome: the model name glows teal, the rest stays muted.
    let mode_spans = if mode == format!("{model} · think {thinking}") {
        vec![
            Span::styled(model, Style::default().fg(MODEL_ACCENT)),
            Span::styled(format!(" · think {thinking}"), Style::default().fg(TEXT_DIM)),
        ]
    } else {
        vec![Span::styled(mode, Style::default().fg(MODEL_ACCENT))]
    };
    let context = format!(
        "{:.0}%",
        app.context_chars as f64 * 100.0 / app.max_context_chars.max(1) as f64
    );
    let run = if app.landing_visible() || app.busy {
        String::new()
    } else {
        app.tokens_per_second
            .map(|rate| format!("{rate:.1}/s  {context}"))
            .unwrap_or(context)
    };
    let remaining = width.saturating_sub(mode_width);
    if remaining < 4 {
        return Line::from(mode_spans);
    }
    let detail = truncate_display(&run, remaining.saturating_sub(2));
    let spacer = remaining.saturating_sub(UnicodeWidthStr::width(detail.as_str()));
    let mut spans = mode_spans;
    spans.push(Span::raw(" ".repeat(spacer)));
    spans.push(Span::styled(detail, Style::default().fg(TEXT_FAINT)));
    Line::from(spans)
}

fn register_input_metadata_hits(app: &mut App, area: Rect) {
    if area.is_empty() {
        return;
    }
    let model = model_short_name(&app.model);
    let model_x = area.x;
    let model_width =
        (UnicodeWidthStr::width(model.as_str()) as u16).min(area.right().saturating_sub(model_x));
    app.register_hit(
        Rect::new(model_x, area.y, model_width, 1),
        HitTarget::StatusModel,
    );
    let thinking_x = model_x.saturating_add(model_width).saturating_add(2);
    let thinking = thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference));
    let thinking_width =
        (UnicodeWidthStr::width(thinking) as u16).min(area.right().saturating_sub(thinking_x));
    app.register_hit(
        Rect::new(thinking_x, area.y, thinking_width, 1),
        HitTarget::StatusThinking,
    );
}

/// Grok-style prompt arrow: bright user gray when focused, dim when not.
fn prompt_color(app: &App) -> Color {
    if app.input_focused || app.busy {
        ACCENT_USER
    } else {
        GRAY_DIM
    }
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
                Style::default().fg(prompt_color(app)),
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
                Style::default().fg(prompt_color(app)),
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
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    if let Some(toast) = &app.toast {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("● ", Style::default().fg(toast.color())),
                Span::styled(toast.message.clone(), Style::default().fg(TEXT)),
            ])),
            area,
        );
        return;
    }
    let items: &[(&str, &str)] = if app.model_picker_open() {
        &[("↑↓", "select"), ("enter", "switch"), ("esc", "close")]
    } else if app.provider_editor_open() {
        if app.provider_editor_is_confirming() {
            &[("enter", "confirm"), ("esc", "cancel")]
        } else if app.provider_editor_is_editing() {
            &[("enter", "apply"), ("esc", "cancel"), ("ctrl+s", "save")]
        } else {
            &[
                ("tab", "pane"),
                ("↑↓", "select"),
                ("enter", "edit"),
                ("n", "new"),
                ("d", "delete"),
                ("ctrl+s", "save"),
                ("esc", "close"),
            ]
        }
    } else if app.session_picker_open() {
        if app
            .session_picker
            .as_ref()
            .is_some_and(|picker| picker.sessions.is_empty())
        {
            &[("esc", "close")]
        } else {
            &[("↑↓", "select"), ("enter", "resume"), ("esc", "close")]
        }
    } else if app.help_open {
        &[("↑↓", "select"), ("enter", "run"), ("esc", "close")]
    } else if app.completion_open() {
        &[
            ("↑↓", "select"),
            ("tab", "complete"),
            ("enter", "run"),
            ("esc", "close"),
        ]
    } else if app.busy {
        &[("esc", "interrupt")]
    } else if !app.input_focused && app.selected_entry.is_some() {
        &[
            ("tab", "browse"),
            ("enter", "open"),
            ("space", "compose"),
            ("esc", "clear"),
        ]
    } else {
        &[
            ("enter", "send"),
            ("shift+enter", "newline"),
            ("/", "commands"),
        ]
    };
    frame.render_widget(
        Paragraph::new(shortcut_bar(items, area.width as usize)),
        area,
    );
}

fn shortcut_bar(items: &[(&str, &str)], width: usize) -> Line<'static> {
    let key_style = Style::default()
        .fg(TEXT_DIM)
        .add_modifier(Modifier::BOLD);
    let label_style = Style::default().fg(TEXT_FAINT);
    let sep_style = Style::default().fg(TEXT_FAINT);
    let mut spans = Vec::new();
    let mut used = 0;
    for (index, (key, label)) in items.iter().enumerate() {
        let piece = if index == 0 {
            format!("{key}  {label}")
        } else {
            format!("  │  {key}  {label}")
        };
        let piece_width = UnicodeWidthStr::width(piece.as_str());
        if used + piece_width > width {
            break;
        }
        if index > 0 {
            spans.push(Span::styled("  │  ", sep_style));
        }
        spans.push(Span::styled((*key).to_owned(), key_style));
        spans.push(Span::styled(format!("  {label}"), label_style));
        used += piece_width;
    }
    Line::from(truncate_spans(spans, width))
}

fn render_turn_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.is_empty() || !app.busy {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    let tick = app
        .turn_started
        .map(|started| started.elapsed().as_millis() as u64 / 33)
        .unwrap_or(0);
    let spinner = crate::tui::glyphs::spinner_frame(tick);
    // Grok-style activity coloring: tools green, thinking purple.
    let (activity, accent) = match app.status {
        Status::Cancelling => ("stopping", BAD),
        Status::Error => ("error", BAD),
        Status::RunningTool => ("working", OK),
        Status::Thinking => ("thinking", ACCENT_SECONDARY),
        Status::Idle => ("working", TEXT_FAINT),
    };
    let elapsed = app
        .turn_started
        .map(|started| format_turn_duration(started.elapsed()))
        .unwrap_or_else(|| "0s".to_owned());
    let tools = format!(
        "{} tool{}",
        app.turn_tool_count,
        if app.turn_tool_count == 1 { "" } else { "s" }
    );
    let left = format!("{spinner} {activity}");
    let detail = format!("{tools}  {elapsed}");
    let right = "esc  interrupt";
    let spacer = (area.width as usize)
        .saturating_sub(UnicodeWidthStr::width(left.as_str()))
        .saturating_sub(UnicodeWidthStr::width(detail.as_str()))
        .saturating_sub(2)
        .saturating_sub(UnicodeWidthStr::width(right));
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(left, Style::default().fg(accent)),
            Span::styled(format!("  {detail}"), Style::default().fg(TEXT_FAINT)),
            Span::raw(" ".repeat(spacer)),
            Span::styled(right, Style::default().fg(TEXT_FAINT)),
        ])),
        area,
    );
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
        if character_width > width {
            row += 1;
            column = 0;
            continue;
        }
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
        if character_width > width {
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
                current_width = 0;
            }
            lines.push(truncate_display(&character.to_string(), width));
            continue;
        }
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

fn truncate_spans(spans: Vec<Span<'static>>, width: usize) -> Vec<Span<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let mut fitted = Vec::new();
    let mut used = 0;
    for span in spans {
        if used >= width {
            break;
        }
        let content = span.content.as_ref();
        let span_width = UnicodeWidthStr::width(content);
        if used + span_width <= width {
            used += span_width;
            fitted.push(span);
            continue;
        }
        let clipped = truncate_display(content, width - used);
        if !clipped.is_empty() {
            fitted.push(Span::styled(clipped, span.style));
        }
        break;
    }
    fitted
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
