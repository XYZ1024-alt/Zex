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
    lines.push(tool_header_line(
        &tool_verb(&tool.name),
        tool_subject_spans(tool, &subject),
        &result,
        &duration,
        selected,
        tool_result_color(tool),
        width,
    ));

    if tool.expanded {
        let label_style = Style::default().fg(TEXT_DIM);
        let body_style = Style::default().fg(TEXT);
        let added_style = Style::default().fg(OK);
        let removed_style = Style::default().fg(BAD);
        let failed = tool_result_color(tool) == BAD;
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
        let mut push = |lines: &mut Vec<Line<'static>>, spans: Vec<Span<'static>>| {
            let mut row = rail_spans(ACCENT_TOOL);
            row.push(Span::raw("  "));
            row.extend(spans);
            lines.push(Line::from(row));
        };

        push(lines, Vec::new());
        if edit_diff.is_some() {
            push(lines, vec![
                Span::styled("diff  ", label_style),
                Span::styled(param.to_owned(), body_style),
            ]);
        } else if tool.name == "bash" && !failed {
            push(lines, vec![Span::styled("command", label_style)]);
            for line in param.lines() {
                push(lines, vec![
                    Span::raw("  "),
                    Span::styled(line.to_owned(), body_style),
                ]);
            }
            push(lines, Vec::new());
            push(lines, vec![Span::styled("output", label_style)]);
        } else if param.is_empty() {
            push(lines, vec![Span::styled(
                single_line(&tool.arguments, 200),
                label_style,
            )]);
        } else if tool.name == "bash" {
            push(lines, vec![Span::styled("command", label_style)]);
            for line in param.lines() {
                push(lines, vec![
                    Span::raw("  "),
                    Span::styled(line.to_owned(), body_style),
                ]);
            }
            push(lines, vec![]);
            push(lines, vec![Span::styled("stderr", label_style)]);
        } else {
            let indent = " ".repeat(param_key.len() + 2);
            for (index, line) in param.lines().enumerate() {
                push(lines, vec![
                    Span::styled(
                        if index == 0 {
                            format!("{param_key}  ")
                        } else {
                            indent.clone()
                        },
                        label_style,
                    ),
                    Span::styled(line.to_owned(), body_style),
                ]);
            }
        }

        if let Some(diff_lines) = &edit_diff {
            for line in diff_lines.iter().take(visible_lines) {
                let (style, band) = if line.starts_with("+ ") {
                    (added_style, Some(DIFF_ADD_BG))
                } else if line.starts_with("- ") {
                    (removed_style, Some(DIFF_DEL_BG))
                } else {
                    (body_style, None)
                };
                for segment in wrap_display_hard(line, width.saturating_sub(4).max(1)) {
                    let mut row = rail_spans(ACCENT_TOOL);
                    row.push(Span::raw("  "));
                    row.push(Span::styled(segment, style));
                    let mut rendered = Line::from(row);
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
                for segment in wrap_display_hard(line, width.saturating_sub(4).max(1)) {
                    push(lines, vec![Span::styled(segment, body_style)]);
                }
            }
        }

        let total = edit_diff
            .as_ref()
            .map_or_else(|| body_lines.len(), Vec::len);
        let unit = if total == 1 { "line" } else { "lines" };
        let footer = if !tool.show_full_output && total > TOOL_OUTPUT_PREVIEW_LINES {
            format!(
                "{total} {unit} · {} more · Ctrl+O expand",
                total - TOOL_OUTPUT_PREVIEW_LINES
            )
        } else if tool.show_full_output && total > TOOL_OUTPUT_PREVIEW_LINES {
            format!("{total} {unit} · Ctrl+O collapse")
        } else if total > 0 {
            format!("{total} {unit}")
        } else {
            format!("timeout {}", format_duration(tool.timeout))
        };
        push(lines, vec![Span::styled(footer, label_style)]);
    }
}

fn tool_verb(name: &str) -> String {
    match name {
        "read" => "Read".to_owned(),
        other => other.to_owned(),
    }
}

/// Grok-style subject: paths glow orange, shell commands get a dim `$` prompt.
fn tool_subject_spans(tool: &ToolEntry, subject: &str) -> Vec<Span<'static>> {
    if tool.name == "bash" {
        return vec![
            Span::styled("$ ", Style::default().fg(GRAY_DIM)),
            Span::styled(subject.to_owned(), Style::default().fg(COMMAND)),
        ];
    }
    let color = match tool.name.as_str() {
        "read" | "write" | "edit" | "grep" | "glob" => PATH,
        _ => TEXT_DIM,
    };
    vec![Span::styled(subject.to_owned(), Style::default().fg(color))]
}

fn tool_header_line(
    verb: &str,
    subject: Vec<Span<'static>>,
    result: &str,
    duration: &str,
    selected: bool,
    result_color: Color,
    width: usize,
) -> Line<'static> {
    let verb_style = Style::default()
        .fg(if selected { TEXT_STRONG } else { TEXT })
        .add_modifier(Modifier::BOLD);
    let mut spans = rail_spans(ACCENT_TOOL);
    spans.extend([Span::styled(verb.to_owned(), verb_style), Span::raw(" ")]);
    spans.extend(subject);
    if width >= 36 && !result.is_empty() {
        spans.extend([
            Span::styled("  ", Style::default().fg(TEXT_FAINT)),
            Span::styled(result.to_owned(), Style::default().fg(result_color)),
        ]);
    }
    if width >= 48 && !duration.is_empty() {
        spans.extend([
            Span::styled("  ", Style::default().fg(TEXT_FAINT)),
            Span::styled(duration.to_owned(), Style::default().fg(TEXT_FAINT)),
        ]);
    }
    Line::from(super::truncate_spans(spans, width)).style(header_row_style(selected, false))
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
            .map(|(added, removed)| match (added, removed) {
                (added, 0) => format!("+{added}"),
                (0, removed) => format!("−{removed}"),
                (added, removed) => format!("+{added} −{removed}"),
            })
            .unwrap_or_else(|| "edited".to_owned()),
        "write" => tool_arguments(tool)
            .and_then(|arguments| {
                arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .map(|content| content.lines().count().max(1))
            })
            .map(|lines| format!("+{lines}"))
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

fn tool_result_color(tool: &ToolEntry) -> Color {
    if tool.name == "bash" && bash_exit_code(&tool.output).is_some_and(|code| code != "0") {
        return BAD;
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
