use std::{
    collections::{HashMap, VecDeque},
    io::{self, IsTerminal},
    time::Duration,
};

use anyhow::{Context, Result};
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use ratatui::{
    DefaultTerminal, Frame, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use tokio::sync::mpsc;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{
    agent::{Agent, AgentEvent, Message, MessageRole},
    provider::Provider,
};

const MAX_TOOL_ENTRIES: usize = 200;
const MAX_ERRORS: usize = 5;
const MAX_TOOL_PREVIEW_CHARS: usize = 800;
const INPUT_HEIGHT: u16 = 3;

pub fn is_available() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub async fn run<P>(
    agent: &mut Agent<P>,
    event_receiver: mpsc::UnboundedReceiver<AgentEvent>,
) -> Result<()>
where
    P: Provider,
{
    let mut terminal = TerminalSession::start()?;
    let result = run_loop(
        terminal.terminal_mut(),
        agent,
        event_receiver,
        EventStream::new(),
    )
    .await;
    let restore_result = terminal.restore();

    match (result, restore_result) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

async fn run_loop<P>(
    terminal: &mut DefaultTerminal,
    agent: &mut Agent<P>,
    mut event_receiver: mpsc::UnboundedReceiver<AgentEvent>,
    mut terminal_events: EventStream,
) -> Result<()>
where
    P: Provider,
{
    let mut app = App::new(agent.messages());
    let mut redraw = tokio::time::interval(Duration::from_millis(80));
    redraw.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        terminal
            .draw(|frame| render(frame, &app))
            .context("failed to draw TUI")?;

        if app.should_quit {
            return Ok(());
        }

        tokio::select! {
            _ = redraw.tick() => {}
            event = event_receiver.recv() => {
                match event {
                    Some(event) => app.apply_agent_event(event),
                    None => app.status = Status::Idle,
                }
            }
            event = terminal_events.next() => {
                match event {
                    Some(Ok(event)) => match handle_terminal_event(event, &mut app) {
                        InputAction::None => {}
                        InputAction::Quit => app.should_quit = true,
                        InputAction::Submit(prompt) => {
                            run_turn(
                                terminal,
                                &mut app,
                                agent,
                                &mut event_receiver,
                                &mut terminal_events,
                                prompt,
                            )
                            .await?;
                        }
                    },
                    Some(Err(error)) => return Err(error).context("failed to read terminal event"),
                    None => return Ok(()),
                }
            }
        }
    }
}

async fn run_turn<P>(
    terminal: &mut DefaultTerminal,
    app: &mut App,
    agent: &mut Agent<P>,
    event_receiver: &mut mpsc::UnboundedReceiver<AgentEvent>,
    terminal_events: &mut EventStream,
    prompt: String,
) -> Result<()>
where
    P: Provider,
{
    app.busy = true;
    app.status = Status::Thinking;
    let prompt_future = agent.prompt(prompt);
    tokio::pin!(prompt_future);
    let mut redraw = tokio::time::interval(Duration::from_millis(80));
    redraw.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        terminal
            .draw(|frame| render(frame, app))
            .context("failed to draw TUI")?;

        tokio::select! {
            _ = redraw.tick() => {}
            result = &mut prompt_future => {
                while let Ok(event) = event_receiver.try_recv() {
                    app.apply_agent_event(event);
                }
                match result {
                    Ok(_) => {
                        if app.busy {
                            app.busy = false;
                            app.status = Status::Idle;
                        }
                    }
                    Err(error) => {
                        app.record_error_if_new(format!("{error:#}"));
                        app.busy = false;
                        app.status = Status::Error;
                    }
                }
                return Ok(());
            }
            event = event_receiver.recv() => {
                match event {
                    Some(event) => app.apply_agent_event(event),
                    None => return Ok(()),
                }
            }
            event = terminal_events.next() => {
                match event {
                    Some(Ok(Event::Key(key)))
                        if key.kind == crossterm::event::KeyEventKind::Press
                            && ((key.modifiers.contains(KeyModifiers::CONTROL)
                                && key.code == KeyCode::Char('c'))
                                || key.code == KeyCode::Char('q')) =>
                    {
                        app.should_quit = true;
                        return Ok(());
                    }
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        return Err(error).context("failed to read terminal event");
                    }
                    None => return Ok(()),
                }
            }
        }
    }
}

enum InputAction {
    None,
    Quit,
    Submit(String),
}

fn handle_terminal_event(event: Event, app: &mut App) -> InputAction {
    let Event::Key(key) = event else {
        return InputAction::None;
    };

    if key.kind != crossterm::event::KeyEventKind::Press {
        return InputAction::None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return InputAction::Quit;
    }

    if app.busy {
        if key.code == KeyCode::Char('q') {
            return InputAction::Quit;
        }
        return InputAction::None;
    }

    match key {
        KeyEvent {
            code: KeyCode::Esc, ..
        }
        | KeyEvent {
            code: KeyCode::Char('q'),
            ..
        } if app.input.is_empty() => app.should_quit = true,
        KeyEvent {
            code: KeyCode::Enter,
            modifiers,
            ..
        } if !modifiers.contains(KeyModifiers::SHIFT) => {
            let prompt = app.take_prompt();
            if prompt.is_empty() {
                return InputAction::None;
            }
            return InputAction::Submit(prompt);
        }
        KeyEvent {
            code: KeyCode::Backspace,
            ..
        } => {
            app.input.pop();
        }
        KeyEvent {
            code: KeyCode::Char(character),
            modifiers,
            ..
        } if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
            app.input.push(character);
        }
        _ => {}
    }
    if app.should_quit {
        InputAction::Quit
    } else {
        InputAction::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Thinking,
    RunningTool,
    Error,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Thinking => "thinking",
            Self::RunningTool => "running tool",
            Self::Error => "error",
        }
    }

    fn color(self) -> Color {
        match self {
            Self::Idle => Color::Green,
            Self::Thinking => Color::Yellow,
            Self::RunningTool => Color::Cyan,
            Self::Error => Color::Red,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolStatus {
    Running,
    Done,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolEntry {
    call_id: String,
    name: String,
    output: String,
    status: ToolStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TranscriptEntry {
    Message { role: MessageRole, content: String },
    Tool(ToolEntry),
    Error(String),
}

#[derive(Debug)]
struct App {
    transcript: Vec<TranscriptEntry>,
    input: String,
    active_tools: HashMap<String, String>,
    errors: VecDeque<String>,
    status: Status,
    busy: bool,
    should_quit: bool,
}

impl App {
    fn new(messages: &[Message]) -> Self {
        let transcript = messages
            .iter()
            .filter_map(|message| match message {
                Message::User { content } => Some(TranscriptEntry::Message {
                    role: MessageRole::User,
                    content: content.clone(),
                }),
                Message::Assistant { content, .. } if !content.is_empty() => {
                    Some(TranscriptEntry::Message {
                        role: MessageRole::Assistant,
                        content: content.clone(),
                    })
                }
                _ => None,
            })
            .collect();

        Self {
            transcript,
            input: String::new(),
            active_tools: HashMap::new(),
            errors: VecDeque::new(),
            status: Status::Idle,
            busy: false,
            should_quit: false,
        }
    }

    fn take_prompt(&mut self) -> String {
        let prompt = self.input.trim().to_owned();
        self.input.clear();
        prompt
    }

    fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::MessageDelta { role, delta } => {
                self.append_message(role, delta);
                if role == MessageRole::Assistant {
                    self.status = Status::Thinking;
                }
            }
            AgentEvent::ToolStart { call_id, name } => {
                self.active_tools.insert(call_id.clone(), name.clone());
                self.status = Status::RunningTool;
                self.transcript.push(TranscriptEntry::Tool(ToolEntry {
                    call_id,
                    name,
                    output: String::new(),
                    status: ToolStatus::Running,
                }));
                self.trim_tool_entries();
            }
            AgentEvent::ToolEnd {
                call_id,
                name,
                output,
                is_error,
            } => {
                let status = if is_error {
                    ToolStatus::Failed
                } else {
                    ToolStatus::Done
                };
                if let Some(entry) = self.transcript.iter_mut().rev().find_map(|entry| {
                    let TranscriptEntry::Tool(tool) = entry else {
                        return None;
                    };
                    (tool.call_id == call_id).then_some(tool)
                }) {
                    entry.name = name;
                    entry.output = truncate_chars(&output, MAX_TOOL_PREVIEW_CHARS);
                    entry.status = status;
                }
                self.active_tools.remove(&call_id);
                self.status = if !self.active_tools.is_empty() {
                    Status::RunningTool
                } else if is_error {
                    Status::Error
                } else {
                    Status::Thinking
                };
                if is_error {
                    self.record_error(output);
                }
            }
            AgentEvent::Error { message } => {
                self.record_error(message);
                self.status = Status::Error;
                self.busy = false;
            }
            AgentEvent::TurnEnd => {
                self.active_tools.clear();
                self.status = Status::Idle;
                self.busy = false;
            }
        }
    }

    fn append_message(&mut self, role: MessageRole, delta: String) {
        if let Some(TranscriptEntry::Message {
            role: last_role,
            content,
        }) = self.transcript.last_mut()
            && *last_role == role
        {
            content.push_str(&delta);
            return;
        }

        self.transcript.push(TranscriptEntry::Message {
            role,
            content: delta,
        });
    }

    fn record_error(&mut self, message: String) {
        if self.errors.len() == MAX_ERRORS {
            self.errors.pop_front();
        }
        self.errors.push_back(message.clone());
        self.transcript.push(TranscriptEntry::Error(message));
    }

    fn record_error_if_new(&mut self, message: String) {
        if self.errors.back().map(String::as_str) != Some(message.as_str()) {
            self.record_error(message);
        }
    }

    fn trim_tool_entries(&mut self) {
        let tool_count = self
            .transcript
            .iter()
            .filter(|entry| matches!(entry, TranscriptEntry::Tool(_)))
            .count();
        if tool_count <= MAX_TOOL_ENTRIES {
            return;
        }

        if let Some(index) = self
            .transcript
            .iter()
            .position(|entry| matches!(entry, TranscriptEntry::Tool(_)))
        {
            self.transcript.remove(index);
        }
    }
}

fn render(frame: &mut Frame<'_>, app: &App) {
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(INPUT_HEIGHT)])
        .split(frame.area());
    let upper = if areas[0].width >= 100 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(60), Constraint::Length(32)])
            .split(areas[0])
    } else {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(5), Constraint::Length(8)])
            .split(areas[0])
    };

    render_transcript(frame, upper[0], app);
    render_status(frame, upper[1], app);
    render_input(frame, areas[1], app);
}

fn render_transcript(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let mut lines = Vec::new();
    for entry in &app.transcript {
        match entry {
            TranscriptEntry::Message { role, content } => {
                let (label, color) = match role {
                    MessageRole::User => ("you", Color::Blue),
                    MessageRole::Assistant => ("assistant", Color::Green),
                };
                lines.push(Line::from(Span::styled(
                    label,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                )));
                lines.extend(content.lines().map(|line| Line::from(line.to_owned())));
                lines.push(Line::default());
            }
            TranscriptEntry::Tool(tool) => {
                let (status, color) = match tool.status {
                    ToolStatus::Running => ("running", Color::Yellow),
                    ToolStatus::Done => ("done", Color::Green),
                    ToolStatus::Failed => ("failed", Color::Red),
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        "tool ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(&tool.name),
                    Span::raw(" · "),
                    Span::styled(status, Style::default().fg(color)),
                ]));
                if !tool.output.is_empty() {
                    lines.extend(tool.output.lines().map(|line| {
                        Line::from(Span::styled(line, Style::default().fg(Color::DarkGray)))
                    }));
                }
                lines.push(Line::default());
            }
            TranscriptEntry::Error(message) => {
                lines.push(Line::from(Span::styled(
                    format!("error · {message}"),
                    Style::default().fg(Color::Red),
                )));
                lines.push(Line::default());
            }
        }
    }

    let available_height = area.height.saturating_sub(2) as usize;
    let scroll = lines.len().saturating_sub(available_height) as u16;
    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .title(" Conversation & tools ")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(paragraph, area);
}

fn render_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let tool = app
        .active_tools
        .values()
        .next()
        .map(String::as_str)
        .unwrap_or("none");
    let mut lines = vec![
        Line::from(vec![
            Span::raw("status  "),
            Span::styled(
                app.status.label(),
                Style::default()
                    .fg(app.status.color())
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(format!("tool    {tool}")),
        Line::default(),
        Line::from(Span::styled(
            "errors",
            Style::default().add_modifier(Modifier::BOLD),
        )),
    ];
    if app.errors.is_empty() {
        lines.push(Line::from(Span::styled(
            "none",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        lines.extend(app.errors.iter().rev().map(|error| {
            Line::from(Span::styled(
                single_line(error, 120),
                Style::default().fg(Color::Red),
            ))
        }));
    }

    let paragraph = Paragraph::new(lines)
        .block(Block::default().title(" Status ").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn render_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let title = if app.busy {
        " Working · q/Ctrl-C quits "
    } else {
        " Prompt · Enter sends · Esc quits "
    };
    let input_width = area.width.saturating_sub(2) as usize;
    let visible_input = (!app.busy).then(|| input_viewport(&app.input, input_width));
    let content = visible_input
        .as_ref()
        .map_or("Waiting for the current turn…", |viewport| viewport.text);
    let paragraph = Paragraph::new(content)
        .block(Block::default().title(title).borders(Borders::ALL))
        .style(if app.busy {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        });
    frame.render_widget(paragraph, area);

    if let Some(viewport) = visible_input {
        frame.set_cursor_position((area.x + 1 + viewport.cursor_column, area.y + 1));
    }
}

struct InputViewport<'a> {
    text: &'a str,
    cursor_column: u16,
}

fn input_viewport(input: &str, width: usize) -> InputViewport<'_> {
    if width == 0 {
        return InputViewport {
            text: "",
            cursor_column: 0,
        };
    }

    let input_width = UnicodeWidthStr::width(input);
    if input_width < width {
        return InputViewport {
            text: input,
            cursor_column: input_width as u16,
        };
    }

    let mut visible_width = 0;
    let mut start = input.len();
    for (index, character) in input.char_indices().rev() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if visible_width + character_width >= width {
            break;
        }
        visible_width += character_width;
        start = index;
    }

    InputViewport {
        text: &input[start..],
        cursor_column: visible_width as u16,
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let content: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{content}\n… output truncated")
    } else {
        content
    }
}

fn single_line(value: &str, max_chars: usize) -> String {
    truncate_chars(&value.replace(['\r', '\n'], " "), max_chars)
}

struct TerminalSession {
    terminal: DefaultTerminal,
    restored: bool,
}

impl TerminalSession {
    fn start() -> Result<Self> {
        enable_raw_mode().context("failed to enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error).context("failed to enter alternate screen");
        }

        let backend = CrosstermBackend::new(stdout);
        let terminal = ratatui::Terminal::with_options(
            backend,
            TerminalOptions {
                viewport: Viewport::Fullscreen,
            },
        )
        .context("failed to initialize terminal")?;
        Ok(Self {
            terminal,
            restored: false,
        })
    }

    fn terminal_mut(&mut self) -> &mut DefaultTerminal {
        &mut self.terminal
    }

    fn restore(&mut self) -> Result<()> {
        if self.restored {
            return Ok(());
        }
        self.restored = true;

        disable_raw_mode().context("failed to disable terminal raw mode")?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)
            .context("failed to leave alternate screen")?;
        self.terminal
            .show_cursor()
            .context("failed to restore terminal cursor")
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.restored {
            return;
        }
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[cfg(test)]
mod tests {
    use super::{App, Status, ToolStatus, TranscriptEntry, input_viewport};
    use crate::agent::{AgentEvent, MessageRole};

    #[test]
    fn folds_assistant_deltas_and_tracks_tool_lifecycle() {
        let mut app = App::new(&[]);
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "Hel".to_owned(),
        });
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "lo".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            output: "Cargo.toml".to_owned(),
            is_error: false,
        });
        app.apply_agent_event(AgentEvent::TurnEnd);

        assert_eq!(app.status, Status::Idle);
        assert!(!app.busy);
        assert_eq!(
            app.transcript[0],
            TranscriptEntry::Message {
                role: MessageRole::Assistant,
                content: "Hello".to_owned(),
            }
        );
        let TranscriptEntry::Tool(tool) = &app.transcript[1] else {
            panic!("expected tool entry");
        };
        assert_eq!(tool.status, ToolStatus::Done);
        assert_eq!(tool.output, "Cargo.toml");
    }

    #[test]
    fn retains_recent_errors_for_the_status_summary() {
        let mut app = App::new(&[]);
        app.apply_agent_event(AgentEvent::Error {
            message: "provider failed".to_owned(),
        });

        assert_eq!(app.status, Status::Error);
        assert_eq!(
            app.errors.back().map(String::as_str),
            Some("provider failed")
        );
        assert!(matches!(
            app.transcript.last(),
            Some(TranscriptEntry::Error(message)) if message == "provider failed"
        ));
    }

    #[test]
    fn keeps_another_tool_visible_when_one_call_finishes() {
        let mut app = App::new(&[]);
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-2".to_owned(),
            name: "bash".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            output: "done".to_owned(),
            is_error: false,
        });

        assert_eq!(app.status, Status::RunningTool);
        assert_eq!(
            app.active_tools.get("call-2").map(String::as_str),
            Some("bash")
        );
        assert!(!app.active_tools.contains_key("call-1"));
    }

    #[test]
    fn input_viewport_keeps_ascii_cursor_inside_the_box() {
        let viewport = input_viewport("123456789", 6);

        assert_eq!(viewport.text, "56789");
        assert_eq!(viewport.cursor_column, 5);
    }

    #[test]
    fn input_viewport_uses_display_width_for_chinese_text() {
        let viewport = input_viewport("ab你好", 6);

        assert_eq!(viewport.text, "b你好");
        assert_eq!(viewport.cursor_column, 5);
    }

    #[test]
    fn input_viewport_leaves_one_cell_for_the_cursor() {
        let viewport = input_viewport("abcdef", 6);

        assert_eq!(viewport.text, "bcdef");
        assert_eq!(viewport.cursor_column, 5);
    }
}
