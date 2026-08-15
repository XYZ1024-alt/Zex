use super::*;

pub(super) fn append_tool_lines(
    lines: &mut Vec<Line<'static>>,
    tool: &ToolEntry,
    selected: bool,
    _hovered: bool,
    width: usize,
) {
    let subject = tool_subject(tool);
    let elapsed = tool_elapsed(tool);
    let result = tool_result(tool);
    let duration = format_duration(elapsed);
    let running_millis = match tool.status {
        ToolStatus::Running => Some(
            tool.started_at
                .map(|started| started.elapsed().as_millis() as u64)
                .unwrap_or(0),
        ),
        _ => None,
    };
    lines.push(tool_header_line(
        &tool_verb(&tool.name),
        tool_subject_spans(tool, &subject),
        &result,
        &duration,
        ToolHeaderState {
            selected,
            result_color: tool_result_color(tool),
            running_millis,
        },
        width,
    ));

    if tool.expanded {
        let label_style = Style::default().fg(theme().text_faint);
        let body_style = Style::default().fg(theme().text_dim);
        let added_style = Style::default().fg(theme().ok);
        let removed_style = Style::default().fg(theme().bad);
        let arguments = serde_json::from_str::<Value>(&tool.arguments).ok();
        let param_key = match tool.name.as_str() {
            "bash" => "command",
            "read" | "write" | "edit" => "path",
            "grep" | "glob" => "pattern",
            _ => "",
        };
        let param = arguments
            .as_ref()
            .and_then(|value| value.get(param_key))
            .and_then(Value::as_str)
            .unwrap_or("");
        let body_lines = tool_output_body(tool);
        let edit_diff = file_change_body(tool);
        let visible_lines = if tool.show_full_output {
            edit_diff
                .as_ref()
                .map_or_else(|| body_lines.len(), Vec::len)
        } else {
            edit_diff
                .as_ref()
                .map_or_else(|| body_lines.len(), Vec::len)
                .min(TOOL_OUTPUT_PREVIEW_LINES)
        };
        // Body rows indent two columns to nest under the header.
        let push = |lines: &mut Vec<Line<'static>>, spans: Vec<Span<'static>>| {
            let mut row = vec![Span::raw("  ")];
            row.extend(spans);
            lines.push(Line::from(row));
        };

        // Re-show the invocation only when the header subject could not carry
        // it (multi-line or truncated); otherwise that would repeat the header.
        let invocation_overflow = param.lines().count() > 1 || param.chars().count() > 100;
        if param_key.is_empty() {
            push(
                lines,
                vec![Span::styled(single_line(&tool.arguments, 200), label_style)],
            );
        } else if invocation_overflow {
            for line in param.lines() {
                for segment in wrap_display_hard(line, width.saturating_sub(2).max(1)) {
                    push(lines, vec![Span::styled(segment, label_style)]);
                }
            }
        }

        if let Some(diff_lines) = &edit_diff {
            for line in diff_lines.iter().take(visible_lines) {
                let (style, band) = if line.starts_with("+ ") {
                    (added_style, Some(theme().diff_add_bg))
                } else if line.starts_with("- ") {
                    (removed_style, Some(theme().diff_del_bg))
                } else {
                    (Style::default().fg(theme().text), None)
                };
                for segment in wrap_display_hard(line, width.saturating_sub(2).max(1)) {
                    let mut rendered =
                        Line::from(vec![Span::raw("  "), Span::styled(segment, style)]);
                    if let Some(band) = band {
                        // Full-width band, like grok's diff rows.
                        rendered = rendered.style(Style::default().bg(band));
                        pad_line_band(&mut rendered, width);
                    }
                    lines.push(rendered);
                }
            }
        } else if tool.output.is_empty() && matches!(tool.status, ToolStatus::Running) {
            push(lines, vec![Span::styled("running…", label_style)]);
        } else if body_lines.is_empty() {
            push(lines, vec![Span::styled("(no output)", label_style)]);
        } else {
            for line in body_lines.iter().take(visible_lines) {
                for segment in wrap_display_hard(line, width.saturating_sub(2).max(1)) {
                    push(lines, vec![Span::styled(segment, body_style)]);
                }
            }
        }

        let total = edit_diff
            .as_ref()
            .map_or_else(|| body_lines.len(), Vec::len);
        // Footer only earns its row when it adds information the header does
        // not already carry: hidden overflow or a timeout.
        if !tool.show_full_output && total > TOOL_OUTPUT_PREVIEW_LINES {
            push(
                lines,
                vec![Span::styled(
                    format!("… {} more · ctrl+o", total - TOOL_OUTPUT_PREVIEW_LINES),
                    label_style,
                )],
            );
        } else if tool.show_full_output && total > TOOL_OUTPUT_PREVIEW_LINES {
            push(lines, vec![Span::styled("ctrl+o to collapse", label_style)]);
        } else if total == 0 && !matches!(tool.status, ToolStatus::Running) {
            push(
                lines,
                vec![Span::styled(
                    format!("timeout {}", format_duration(tool.timeout)),
                    label_style,
                )],
            );
        }
    }
}

fn tool_verb(name: &str) -> String {
    match name {
        "bash" => "Bash".to_owned(),
        "read" => "Read".to_owned(),
        "write" => "Write".to_owned(),
        "edit" => "Edit".to_owned(),
        "grep" => "Grep".to_owned(),
        "glob" => "Glob".to_owned(),
        other => other.to_owned(),
    }
}

/// Quiet subject: paths and identifiers stay on the gray ramp, shell
/// commands get a dim `$` prompt and a muted sand tone.
fn tool_subject_spans(tool: &ToolEntry, subject: &str) -> Vec<Span<'static>> {
    if tool.name == "bash" {
        return vec![
            Span::styled("$ ", Style::default().fg(theme().gray_dim)),
            Span::styled(subject.to_owned(), Style::default().fg(theme().command)),
        ];
    }
    vec![Span::styled(
        subject.to_owned(),
        Style::default().fg(theme().text_dim),
    )]
}

struct ToolHeaderState {
    selected: bool,
    result_color: Color,
    running_millis: Option<u64>,
}

fn tool_header_line(
    verb: &str,
    subject: Vec<Span<'static>>,
    result: &str,
    duration: &str,
    state: ToolHeaderState,
    width: usize,
) -> Line<'static> {
    let selected = state.selected;
    let result_color = state.result_color;
    let verb_style = Style::default()
        .fg(if selected {
            theme().text_strong
        } else {
            theme().text
        })
        .add_modifier(Modifier::BOLD);
    // Status glyph: a live braille spinner while running, otherwise a quiet
    // dot colored by outcome (green ok, red failed, gray stopped).
    let glyph = match state.running_millis {
        Some(millis) => Span::styled(
            crate::tui::glyphs::spinner_frame(millis).to_owned(),
            Style::default().fg(theme().running),
        ),
        None => Span::styled(
            crate::tui::glyphs::status_dot().to_owned(),
            Style::default().fg(result_color),
        ),
    };
    let mut spans = vec![
        glyph,
        Span::raw(" "),
        Span::styled(verb.to_owned(), verb_style),
        Span::raw(" "),
    ];
    spans.extend(subject);
    // Meta tail: `· result · duration`, faint and red only on failure. The
    // word "running" is redundant next to the spinner, so it is dropped.
    let mut meta: Vec<String> = Vec::new();
    if state.running_millis.is_none() && !result.is_empty() {
        meta.push(result.to_owned());
    }
    if !duration.is_empty() {
        meta.push(duration.to_owned());
    }
    if !meta.is_empty() {
        let meta_color = if result_color == theme().bad {
            theme().bad
        } else {
            theme().text_faint
        };
        spans.push(Span::styled(
            format!(" · {}", meta.join(" · ")),
            Style::default().fg(meta_color),
        ));
    }
    super::card_header_line(spans, selected, width)
}

fn tool_output_line_count(output: &str) -> usize {
    output.split('\n').count().max(1)
}

pub(in crate::tui) fn tool_detail_line_count(tool: &ToolEntry) -> usize {
    file_change_body(tool)
        .map_or_else(|| tool_output_line_count(&tool.output), |lines| lines.len())
        .max(1)
}

fn tool_subject(tool: &ToolEntry) -> String {
    let Ok(arguments) = serde_json::from_str::<Value>(&tool.arguments) else {
        return single_line(&tool.arguments, 100);
    };
    let key = match tool.name.as_str() {
        "bash" => "command",
        "read" | "write" | "edit" => "path",
        "grep" | "glob" => "pattern",
        _ => "",
    };
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(|value| single_line(value, 100))
        .unwrap_or_else(|| single_line(&tool.arguments, 100))
}

fn tool_elapsed(tool: &ToolEntry) -> Duration {
    tool.elapsed
        .or_else(|| tool.started_at.map(|started| started.elapsed()))
        .unwrap_or(Duration::ZERO)
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.1}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

fn tool_result(tool: &ToolEntry) -> String {
    match tool.status {
        ToolStatus::Running => return "running".to_owned(),
        ToolStatus::Cancelled => return "stopped".to_owned(),
        ToolStatus::Failed if tool.name != "bash" => return "failed".to_owned(),
        ToolStatus::Done | ToolStatus::Failed => {}
    }

    if tool.name == "bash" {
        if let Some(code) = bash_exit_code(&tool.output) {
            return format!("exit {code}");
        }
        return tool.status.label().to_owned();
    }
    match tool.name.as_str() {
        "read" => format!("{} lines", tool.output.lines().count()),
        "edit" => edit_change_counts(tool)
            .map(|(added, removed)| format_change_counts(added, removed))
            .unwrap_or_else(|| "edited".to_owned()),
        "write" => tool
            .change
            .as_ref()
            .map(|change| {
                let (added, removed) = crate::agent::change_counts(change);
                format_change_counts(added, removed)
            })
            .or_else(|| {
                tool_arguments(tool)
                    .and_then(|arguments| {
                        arguments
                            .get("content")
                            .and_then(Value::as_str)
                            .map(|content| content.lines().count().max(1))
                    })
                    .map(|lines| format!("+{lines}"))
            })
            .unwrap_or_else(|| "written".to_owned()),
        "grep" if tool.output.starts_with("No matches found.") => "0 matches".to_owned(),
        "grep" => trailing_count_summary(&tool.output, "matching line(s)", "matches")
            .unwrap_or_else(|| format!("{} matches", content_line_count(&tool.output))),
        "glob" if tool.output.starts_with("No matching paths found.") => "0 paths".to_owned(),
        "glob" => trailing_count_summary(&tool.output, "matching path(s)", "paths")
            .unwrap_or_else(|| format!("{} paths", content_line_count(&tool.output))),
        _ => "ok".to_owned(),
    }
}

fn format_change_counts(added: usize, removed: usize) -> String {
    match (added, removed) {
        (added, 0) => format!("+{added}"),
        (0, removed) => format!("−{removed}"),
        (added, removed) => format!("+{added} −{removed}"),
    }
}

fn tool_result_color(tool: &ToolEntry) -> Color {
    if tool.name == "bash" && bash_exit_code(&tool.output).is_some_and(|code| code != "0") {
        return theme().bad;
    }
    tool.status.color()
}

fn bash_exit_code(output: &str) -> Option<&str> {
    output.strip_prefix("exit_code: ")?.lines().next()
}

fn content_line_count(output: &str) -> usize {
    output
        .lines()
        .take_while(|line| !line.trim().is_empty())
        .count()
        .max(1)
}

fn trailing_count_summary(output: &str, marker: &str, label: &str) -> Option<String> {
    let line = output.lines().rev().find(|line| line.contains(marker))?;
    Some(format!("{} {label}", line.split_whitespace().next()?))
}
