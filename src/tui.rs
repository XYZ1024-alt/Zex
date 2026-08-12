use std::{
    collections::{BTreeMap, VecDeque},
    io::{self, IsTerminal},
    time::Duration,
};

use anyhow::{Context, Result};
use crossterm::{
    event::{
        DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEvent,
        KeyModifiers,
    },
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
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use unicode_width::UnicodeWidthChar;

use crate::{
    agent::{Agent, AgentEvent, Message, MessageRole, PromptOutcome},
    command::{CommandEffect, execute, parse},
    provider::Provider,
    session::SessionStore,
};

const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const MAX_ERRORS: usize = 3;
const MAX_TOOL_DETAIL_CHARS: usize = 4_000;
const MAX_TOOL_ARGUMENT_CHARS: usize = 2_000;
const MAX_INPUT_ROWS: u16 = 6;
const MIN_TRANSCRIPT_HEIGHT: u16 = 3;

pub fn is_available() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub async fn run<P>(
    agent: &mut Agent<P>,
    event_receiver: mpsc::UnboundedReceiver<AgentEvent>,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
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
        session_store,
        session_id,
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
    session_store: &SessionStore,
    session_id: &mut Option<String>,
) -> Result<()>
where
    P: Provider,
{
    let mut app = App::new(agent.messages(), agent.model().to_owned());
    let mut redraw = tokio::time::interval(FRAME_INTERVAL);
    redraw.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut dirty = true;

    loop {
        tokio::select! {
            _ = redraw.tick(), if dirty => {
                terminal
                    .draw(|frame| render(frame, &mut app))
                    .context("failed to draw TUI")?;
                dirty = false;
            }
            event = event_receiver.recv() => {
                match event {
                    Some(event) => {
                        app.apply_agent_event(event);
                        drain_agent_events(&mut event_receiver, &mut app);
                    }
                    None => app.finish_turn(Status::Idle),
                }
                dirty = true;
            }
            event = terminal_events.next() => {
                match event {
                    Some(Ok(event)) => match handle_terminal_event(event, &mut app, false) {
                        InputAction::None => {}
                        InputAction::Quit => return Ok(()),
                        InputAction::Interrupt => {}
                        InputAction::Submit(prompt) => {
                            match parse(&prompt) {
                                Ok(Some(command)) => {
                                    match execute(command, agent, session_store, session_id).await {
                                        Ok(result) => {
                                            app.model = agent.model().to_owned();
                                            match result.effect {
                                                CommandEffect::None => {}
                                                CommandEffect::ClearView => app.reset_transcript(),
                                                CommandEffect::ReplaceView => {
                                                    app.replace_transcript(agent.messages());
                                                }
                                            }
                                            app.transcript.push(TranscriptEntry::Notice(result.message));
                                            app.scroll_to_bottom();
                                        }
                                        Err(error) => app.record_error(format!("{error:#}")),
                                    }
                                }
                                Ok(None) => {
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
                                Err(error) => app.record_error(format!("{error:#}")),
                            }
                        }
                    },
                    Some(Err(error)) => return Err(error).context("failed to read terminal event"),
                    None => return Ok(()),
                }
                dirty = true;
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
    app.start_turn();
    let (cancel_sender, cancel_receiver) = watch::channel(false);
    let prompt_future = agent.prompt_cancellable(prompt, cancel_receiver);
    tokio::pin!(prompt_future);

    let mut redraw = tokio::time::interval(FRAME_INTERVAL);
    redraw.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut dirty = true;
    let mut cancellation_requested = false;

    loop {
        tokio::select! {
            _ = redraw.tick(), if dirty => {
                terminal
                    .draw(|frame| render(frame, app))
                    .context("failed to draw TUI")?;
                dirty = false;
            }
            result = &mut prompt_future => {
                drain_agent_events(event_receiver, app);
                match result {
                    Ok(PromptOutcome::Completed(_)) => {
                        if app.busy {
                            app.finish_turn(Status::Idle);
                        }
                    }
                    Ok(PromptOutcome::Cancelled) => {
                        if app.busy {
                            app.apply_agent_event(AgentEvent::TurnCancelled);
                        }
                    }
                    Err(error) => {
                        app.record_error_if_new(format!("{error:#}"));
                        app.finish_turn(Status::Error);
                    }
                }
                terminal
                    .draw(|frame| render(frame, app))
                    .context("failed to draw TUI")?;
                return Ok(());
            }
            event = event_receiver.recv() => {
                match event {
                    Some(event) => {
                        app.apply_agent_event(event);
                        drain_agent_events(event_receiver, app);
                    }
                    None => app.finish_turn(Status::Idle),
                }
                dirty = true;
            }
            event = terminal_events.next() => {
                match event {
                    Some(Ok(event)) => match handle_terminal_event(event, app, true) {
                        InputAction::Interrupt if !cancellation_requested => {
                            cancellation_requested = true;
                            app.status = Status::Cancelling;
                            let _ = cancel_sender.send(true);
                        }
                        InputAction::None | InputAction::Interrupt => {}
                        InputAction::Quit | InputAction::Submit(_) => {}
                    },
                    Some(Err(error)) => {
                        return Err(error).context("failed to read terminal event");
                    }
                    None => return Ok(()),
                }
                dirty = true;
            }
        }
    }
}

fn drain_agent_events(event_receiver: &mut mpsc::UnboundedReceiver<AgentEvent>, app: &mut App) {
    while let Ok(event) = event_receiver.try_recv() {
        app.apply_agent_event(event);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InputAction {
    None,
    Quit,
    Interrupt,
    Submit(String),
}

fn handle_terminal_event(event: Event, app: &mut App, turn_active: bool) -> InputAction {
    match event {
        Event::Paste(content) if !turn_active => {
            app.input.insert_str(&content);
            InputAction::None
        }
        Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
            handle_key_event(key, app, turn_active)
        }
        _ => InputAction::None,
    }
}

fn handle_key_event(key: KeyEvent, app: &mut App, turn_active: bool) -> InputAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return if turn_active {
            InputAction::Interrupt
        } else {
            InputAction::Quit
        };
    }

    match key.code {
        KeyCode::PageUp => {
            app.scroll_page_up();
            return InputAction::None;
        }
        KeyCode::PageDown => {
            app.scroll_page_down();
            return InputAction::None;
        }
        KeyCode::Home => {
            app.scroll_to_top();
            return InputAction::None;
        }
        KeyCode::End => {
            app.scroll_to_bottom();
            return InputAction::None;
        }
        KeyCode::Tab => {
            app.select_tool(false);
            return InputAction::None;
        }
        KeyCode::BackTab => {
            app.select_tool(true);
            return InputAction::None;
        }
        KeyCode::Esc => {
            return if app.cancel_ui_layer() || turn_active {
                InputAction::None
            } else {
                InputAction::Quit
            };
        }
        KeyCode::Char('e')
            if app.input.is_empty()
                && !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            app.toggle_selected_tool();
            return InputAction::None;
        }
        _ => {}
    }

    if turn_active {
        return InputAction::None;
    }

    match key {
        KeyEvent {
            code: KeyCode::Enter,
            modifiers,
            ..
        } if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) => {
            app.input.insert_char('\n');
        }
        KeyEvent {
            code: KeyCode::Enter,
            ..
        } => {
            let prompt = app.input.take_trimmed();
            if !prompt.is_empty() {
                return InputAction::Submit(prompt);
            }
        }
        KeyEvent {
            code: KeyCode::Backspace,
            ..
        } => app.input.backspace(),
        KeyEvent {
            code: KeyCode::Delete,
            ..
        } => app.input.delete(),
        KeyEvent {
            code: KeyCode::Left,
            ..
        } => app.input.move_left(),
        KeyEvent {
            code: KeyCode::Right,
            ..
        } => app.input.move_right(),
        KeyEvent {
            code: KeyCode::Char(character),
            modifiers,
            ..
        } if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) => {
            app.input.insert_char(character);
        }
        _ => {}
    }

    InputAction::None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Thinking,
    RunningTool,
    Cancelling,
    Error,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Self::Idle => "IDLE",
            Self::Thinking => "THINKING",
            Self::RunningTool => "TOOL",
            Self::Cancelling => "INTERRUPTING",
            Self::Error => "ERROR",
        }
    }

    fn color(self) -> Color {
        match self {
            Self::Idle => Color::Green,
            Self::Thinking => Color::Yellow,
            Self::RunningTool => Color::Cyan,
            Self::Cancelling => Color::Yellow,
            Self::Error => Color::Red,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolStatus {
    Running,
    Done,
    Failed,
    Cancelled,
}

impl ToolStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "interrupted",
        }
    }

    fn color(self) -> Color {
        match self {
            Self::Running => Color::Yellow,
            Self::Done => Color::Green,
            Self::Failed => Color::Red,
            Self::Cancelled => Color::Yellow,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolEntry {
    call_id: String,
    name: String,
    arguments: String,
    output: String,
    status: ToolStatus,
    expanded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TranscriptEntry {
    Message { role: MessageRole, content: String },
    Tool(ToolEntry),
    Error(String),
    Notice(String),
}

#[derive(Debug, Default)]
struct InputBuffer {
    content: String,
    cursor: usize,
}

impl InputBuffer {
    fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    fn insert_char(&mut self, character: char) {
        self.content.insert(self.cursor, character);
        self.cursor += character.len_utf8();
    }

    fn insert_str(&mut self, value: &str) {
        self.content.insert_str(self.cursor, value);
        self.cursor += value.len();
    }

    fn backspace(&mut self) {
        let Some(previous) = self.content[..self.cursor].char_indices().next_back() else {
            return;
        };
        self.content.drain(previous.0..self.cursor);
        self.cursor = previous.0;
    }

    fn delete(&mut self) {
        let Some(character) = self.content[self.cursor..].chars().next() else {
            return;
        };
        self.content
            .drain(self.cursor..self.cursor + character.len_utf8());
    }

    fn move_left(&mut self) {
        if let Some((index, _)) = self.content[..self.cursor].char_indices().next_back() {
            self.cursor = index;
        }
    }

    fn move_right(&mut self) {
        if let Some(character) = self.content[self.cursor..].chars().next() {
            self.cursor += character.len_utf8();
        }
    }

    fn clear(&mut self) {
        self.content.clear();
        self.cursor = 0;
    }

    fn take_trimmed(&mut self) -> String {
        let prompt = self.content.trim().to_owned();
        self.clear();
        prompt
    }
}

#[derive(Debug)]
struct App {
    model: String,
    transcript: Vec<TranscriptEntry>,
    input: InputBuffer,
    active_tools: BTreeMap<String, String>,
    errors: VecDeque<String>,
    selected_tool: Option<String>,
    status: Status,
    busy: bool,
    scroll_top: usize,
    max_scroll: usize,
    transcript_page_height: usize,
    follow_output: bool,
}

impl App {
    fn new(messages: &[Message], model: String) -> Self {
        let mut app = Self {
            model,
            transcript: Vec::new(),
            input: InputBuffer::default(),
            active_tools: BTreeMap::new(),
            errors: VecDeque::new(),
            selected_tool: None,
            status: Status::Idle,
            busy: false,
            scroll_top: 0,
            max_scroll: 0,
            transcript_page_height: 1,
            follow_output: true,
        };

        for message in messages {
            match message {
                Message::System { .. } => {}
                Message::User { content } => app.transcript.push(TranscriptEntry::Message {
                    role: MessageRole::User,
                    content: content.clone(),
                }),
                Message::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    if !content.is_empty() {
                        app.transcript.push(TranscriptEntry::Message {
                            role: MessageRole::Assistant,
                            content: content.clone(),
                        });
                    }
                    app.transcript.extend(tool_calls.iter().map(|call| {
                        TranscriptEntry::Tool(ToolEntry {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: truncate_chars(
                                &format_json(&call.arguments),
                                MAX_TOOL_ARGUMENT_CHARS,
                            ),
                            output: String::new(),
                            status: ToolStatus::Done,
                            expanded: false,
                        })
                    }));
                }
                Message::Tool {
                    tool_call_id,
                    content,
                } => {
                    if let Some(tool) = app.find_tool_mut(tool_call_id) {
                        tool.output = truncate_chars(content, MAX_TOOL_DETAIL_CHARS);
                        tool.status = if content.starts_with("tool error:") {
                            ToolStatus::Failed
                        } else {
                            ToolStatus::Done
                        };
                    }
                }
            }
        }

        app
    }

    fn start_turn(&mut self) {
        self.busy = true;
        self.status = Status::Thinking;
        self.scroll_to_bottom();
    }

    fn finish_turn(&mut self, status: Status) {
        self.busy = false;
        self.active_tools.clear();
        self.status = status;
    }

    fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::MessageDelta { role, delta } => {
                self.append_message(role, delta);
                if role == MessageRole::Assistant && self.status != Status::Cancelling {
                    self.status = Status::Thinking;
                }
            }
            AgentEvent::ToolStart {
                call_id,
                name,
                arguments,
            } => {
                self.active_tools.insert(call_id.clone(), name.clone());
                self.status = Status::RunningTool;
                self.transcript.push(TranscriptEntry::Tool(ToolEntry {
                    call_id,
                    name,
                    arguments: truncate_chars(&format_json(&arguments), MAX_TOOL_ARGUMENT_CHARS),
                    output: String::new(),
                    status: ToolStatus::Running,
                    expanded: false,
                }));
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
                if let Some(tool) = self.find_tool_mut(&call_id) {
                    tool.name = name;
                    tool.output = truncate_chars(&output, MAX_TOOL_DETAIL_CHARS);
                    tool.status = status;
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
                    self.remember_error(&output);
                }
            }
            AgentEvent::Error { message } => {
                self.record_error(message);
                self.status = Status::Error;
            }
            AgentEvent::ContextCompacted { stats } => {
                self.transcript.push(TranscriptEntry::Notice(format!(
                    "Auto-compacted context: freed approximately {} chars; kept {} recent turn(s).",
                    stats.freed_chars, stats.kept_turns
                )));
            }
            AgentEvent::TurnCancelled => {
                for entry in &mut self.transcript {
                    if let TranscriptEntry::Tool(tool) = entry
                        && tool.status == ToolStatus::Running
                    {
                        tool.status = ToolStatus::Cancelled;
                    }
                }
                self.transcript.push(TranscriptEntry::Notice(
                    "Turn interrupted. You can edit the next prompt and continue.".to_owned(),
                ));
                self.finish_turn(Status::Idle);
            }
            AgentEvent::TurnEnd => self.finish_turn(Status::Idle),
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

    fn reset_transcript(&mut self) {
        self.transcript.clear();
        self.active_tools.clear();
        self.errors.clear();
        self.selected_tool = None;
        self.status = Status::Idle;
        self.scroll_to_bottom();
    }

    fn replace_transcript(&mut self, messages: &[Message]) {
        let model = self.model.clone();
        *self = Self::new(messages, model);
    }

    fn find_tool_mut(&mut self, call_id: &str) -> Option<&mut ToolEntry> {
        self.transcript.iter_mut().rev().find_map(|entry| {
            let TranscriptEntry::Tool(tool) = entry else {
                return None;
            };
            (tool.call_id == call_id).then_some(tool)
        })
    }

    fn record_error(&mut self, message: String) {
        if !self.remember_error(&message) {
            return;
        }
        self.transcript.push(TranscriptEntry::Error(message));
    }

    fn remember_error(&mut self, message: &str) -> bool {
        if self.errors.back().map(String::as_str) == Some(message) {
            return false;
        }
        if self.errors.len() == MAX_ERRORS {
            self.errors.pop_front();
        }
        self.errors.push_back(message.to_owned());
        true
    }

    fn record_error_if_new(&mut self, message: String) {
        self.record_error(message);
    }

    fn scroll_page_up(&mut self) {
        self.follow_output = false;
        self.scroll_top = self
            .scroll_top
            .saturating_sub(self.transcript_page_height.max(1));
    }

    fn scroll_page_down(&mut self) {
        let next = self
            .scroll_top
            .saturating_add(self.transcript_page_height.max(1));
        if next >= self.max_scroll {
            self.scroll_to_bottom();
        } else {
            self.follow_output = false;
            self.scroll_top = next;
        }
    }

    fn scroll_to_top(&mut self) {
        self.follow_output = false;
        self.scroll_top = 0;
    }

    fn scroll_to_bottom(&mut self) {
        self.follow_output = true;
        self.scroll_top = self.max_scroll;
    }

    fn tool_ids(&self) -> Vec<&str> {
        self.transcript
            .iter()
            .filter_map(|entry| match entry {
                TranscriptEntry::Tool(tool) => Some(tool.call_id.as_str()),
                _ => None,
            })
            .collect()
    }

    fn select_tool(&mut self, reverse: bool) {
        let ids = self.tool_ids();
        if ids.is_empty() {
            self.selected_tool = None;
            return;
        }

        let index = self
            .selected_tool
            .as_deref()
            .and_then(|selected| ids.iter().position(|id| *id == selected));
        let next = match (index, reverse) {
            (Some(0), true) | (None, true) => ids.len() - 1,
            (Some(index), true) => index - 1,
            (Some(index), false) => (index + 1) % ids.len(),
            (None, false) => 0,
        };
        self.selected_tool = Some(ids[next].to_owned());
    }

    fn toggle_selected_tool(&mut self) {
        if self.selected_tool.is_none() {
            self.selected_tool = self.tool_ids().last().map(|id| (*id).to_owned());
        }
        let selected = self.selected_tool.clone();
        if let Some(selected) = selected
            && let Some(tool) = self.find_tool_mut(&selected)
        {
            tool.expanded = !tool.expanded;
        }
    }

    fn cancel_ui_layer(&mut self) -> bool {
        if let Some(selected) = self.selected_tool.clone() {
            if let Some(tool) = self.find_tool_mut(&selected)
                && tool.expanded
            {
                tool.expanded = false;
                return true;
            }
            self.selected_tool = None;
            return true;
        }
        if !self.input.is_empty() {
            self.input.clear();
            return true;
        }
        if !self.follow_output {
            self.scroll_to_bottom();
            return true;
        }
        false
    }

    fn active_tool_summary(&self) -> String {
        if self.active_tools.is_empty() {
            "—".to_owned()
        } else {
            self.active_tools
                .values()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

fn render(frame: &mut Frame<'_>, app: &mut App) {
    let input_height = input_height(&app.input, frame.area().width);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(MIN_TRANSCRIPT_HEIGHT),
            Constraint::Length(input_height),
            Constraint::Length(1),
        ])
        .split(frame.area());

    render_status(frame, areas[0], app);
    render_transcript(frame, areas[1], app);
    render_input(frame, areas[2], app);
    render_help(frame, areas[3], app);
}

fn render_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let mut spans = vec![
        Span::styled(
            " ZEX ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" model ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            single_line(&app.model, 48),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  turn ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            app.status.label(),
            Style::default()
                .fg(app.status.color())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  tool ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            single_line(&app.active_tool_summary(), 40),
            Style::default().fg(Color::Cyan),
        ),
    ];

    if app.status == Status::Error
        && let Some(error) = app.errors.back()
    {
        spans.push(Span::styled(
            "  error ",
            Style::default().fg(Color::DarkGray),
        ));
        spans.push(Span::styled(
            single_line(error, 80),
            Style::default().fg(Color::Red),
        ));
    }

    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(Color::Rgb(24, 27, 33))),
        area,
    );
}

fn render_transcript(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let text = transcript_text(app);
    let block = Block::default()
        .title(" Conversation ")
        .title_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let paragraph = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
    let line_count = paragraph.line_count(area.width);
    app.transcript_page_height = area.height.saturating_sub(2) as usize;
    app.max_scroll = line_count.saturating_sub(area.height as usize);
    if app.follow_output {
        app.scroll_top = app.max_scroll;
    } else {
        app.scroll_top = app.scroll_top.min(app.max_scroll);
    }

    let paragraph = paragraph.scroll((app.scroll_top.min(u16::MAX as usize) as u16, 0));
    frame.render_widget(paragraph, area);

    if !app.follow_output && app.max_scroll > 0 {
        let indicator = format!(
            " lines {}–{} / {} ",
            app.scroll_top.saturating_add(1),
            app.scroll_top
                .saturating_add(app.transcript_page_height)
                .min(line_count),
            line_count
        );
        let width = indicator.chars().count().min(area.width as usize) as u16;
        let x = area.right().saturating_sub(width + 1);
        frame.render_widget(
            Paragraph::new(indicator).style(Style::default().fg(Color::Yellow)),
            Rect::new(x, area.y, width, 1),
        );
    }
}

fn transcript_text(app: &App) -> Text<'static> {
    let mut lines = Vec::new();
    if app.transcript.is_empty() {
        lines.push(Line::from(Span::styled(
            "Start a conversation. Tool calls will appear here as folded summaries.",
            Style::default().fg(Color::DarkGray),
        )));
        return Text::from(lines);
    }

    for entry in &app.transcript {
        match entry {
            TranscriptEntry::Message { role, content } => {
                let (label, color) = match role {
                    MessageRole::User => (" YOU ", Color::Blue),
                    MessageRole::Assistant => (" ASSISTANT ", Color::Green),
                };
                lines.push(Line::from(Span::styled(
                    label,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                )));
                append_markdown_lines(&mut lines, content, *role);
                lines.push(Line::default());
            }
            TranscriptEntry::Tool(tool) => append_tool_lines(
                &mut lines,
                tool,
                app.selected_tool.as_deref() == Some(tool.call_id.as_str()),
            ),
            TranscriptEntry::Error(message) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        " ERROR ",
                        Style::default()
                            .fg(Color::White)
                            .bg(Color::Red)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(single_line(message, 320), Style::default().fg(Color::Red)),
                ]));
                lines.push(Line::default());
            }
            TranscriptEntry::Notice(message) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        " INFO ",
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" "),
                    Span::styled(message.clone(), Style::default().fg(Color::Yellow)),
                ]));
                lines.push(Line::default());
            }
        }
    }

    Text::from(lines)
}

fn append_markdown_lines(lines: &mut Vec<Line<'static>>, content: &str, role: MessageRole) {
    let base_color = match role {
        MessageRole::User => Color::White,
        MessageRole::Assistant => Color::Gray,
    };
    let mut in_code_block = false;

    for source_line in content.split('\n') {
        let trimmed = source_line.trim_start();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            let language = trimmed.trim_start_matches('`').trim();
            let label = if in_code_block {
                if language.is_empty() {
                    "  code".to_owned()
                } else {
                    format!("  code · {language}")
                }
            } else {
                "  end code".to_owned()
            };
            lines.push(Line::from(Span::styled(
                label,
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }
        if in_code_block {
            lines.push(Line::from(Span::styled(
                format!("  {source_line}"),
                Style::default().fg(Color::Yellow),
            )));
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("### ") {
            lines.push(Line::from(Span::styled(
                heading.to_owned(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if let Some(heading) = trimmed
            .strip_prefix("## ")
            .or_else(|| trimmed.strip_prefix("# "))
        {
            lines.push(Line::from(Span::styled(
                heading.to_owned(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if let Some(item) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(Color::Cyan)),
                Span::styled(item.to_owned(), Style::default().fg(base_color)),
            ]));
        } else if let Some(quote) = trimmed.strip_prefix("> ") {
            lines.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
                Span::styled(quote.to_owned(), Style::default().fg(Color::DarkGray)),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                source_line.to_owned(),
                Style::default().fg(base_color),
            )));
        }
    }
}

fn append_tool_lines(lines: &mut Vec<Line<'static>>, tool: &ToolEntry, selected: bool) {
    let marker = if selected { "▶" } else { " " };
    let fold = if tool.expanded { "−" } else { "+" };
    let summary = tool_summary(tool);
    let selection_style = if selected {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Gray)
    };
    lines.push(Line::from(vec![
        Span::styled(
            format!("{marker} {fold} TOOL "),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(tool.name.clone(), selection_style),
        Span::styled(" · ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            tool.status.label(),
            Style::default().fg(tool.status.color()),
        ),
        Span::styled(" · ", Style::default().fg(Color::DarkGray)),
        Span::styled(summary, Style::default().fg(Color::DarkGray)),
    ]));

    if tool.expanded {
        lines.push(Line::from(Span::styled(
            "    arguments",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        for line in tool.arguments.split('\n') {
            lines.push(Line::from(Span::styled(
                format!("      {line}"),
                Style::default().fg(Color::Gray),
            )));
        }
        lines.push(Line::from(Span::styled(
            "    result",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        if tool.output.is_empty() {
            lines.push(Line::from(Span::styled(
                "      waiting for result…",
                Style::default().fg(Color::DarkGray),
            )));
        } else {
            for line in tool.output.split('\n') {
                lines.push(Line::from(Span::styled(
                    format!("      {line}"),
                    Style::default().fg(Color::Gray),
                )));
            }
        }
    }
    lines.push(Line::default());
}

fn tool_summary(tool: &ToolEntry) -> String {
    let source = if tool.output.is_empty() {
        &tool.arguments
    } else {
        &tool.output
    };
    if source.trim().is_empty() {
        return if tool.status == ToolStatus::Running {
            "waiting for result…".to_owned()
        } else {
            "no output".to_owned()
        };
    }
    single_line(
        source
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or(source),
        120,
    )
}

fn render_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let title = if app.busy {
        " Prompt · locked during turn "
    } else {
        " Prompt "
    };
    let block = Block::default()
        .title(title)
        .title_style(Style::default().fg(if app.busy {
            Color::DarkGray
        } else {
            Color::Cyan
        }))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if app.busy {
            Color::DarkGray
        } else {
            Color::Gray
        }));
    let inner_width = area.width.saturating_sub(2).max(1) as usize;
    let visible_rows = area.height.saturating_sub(2).max(1) as usize;

    if app.busy {
        frame.render_widget(
            Paragraph::new("Turn in progress — Ctrl+C interrupts it.")
                .block(block)
                .style(Style::default().fg(Color::DarkGray)),
            area,
        );
        return;
    }

    let metrics = input_metrics(&app.input.content, app.input.cursor, inner_width);
    let vertical_scroll = metrics.cursor_row.saturating_sub(visible_rows - 1);
    let content = if app.input.is_empty() {
        Text::from(Line::from(Span::styled(
            "Type a message or /help…",
            Style::default().fg(Color::DarkGray),
        )))
    } else {
        Text::from(app.input.content.clone())
    };
    frame.render_widget(
        Paragraph::new(content)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((vertical_scroll.min(u16::MAX as usize) as u16, 0)),
        area,
    );

    let cursor_y = metrics.cursor_row.saturating_sub(vertical_scroll) as u16;
    frame.set_cursor_position((
        area.x + 1 + metrics.cursor_column.min(inner_width - 1) as u16,
        area.y + 1 + cursor_y.min(area.height.saturating_sub(3)),
    ));
}

fn render_help(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let help = if app.busy {
        if area.width >= 92 {
            " Ctrl+C interrupt  ·  PgUp/PgDn scroll  ·  Home/End history  ·  Tab select tool  ·  e expand  ·  Esc close "
        } else {
            " Ctrl+C interrupt · PgUp/PgDn scroll · Tab tool · e expand "
        }
    } else if area.width >= 110 {
        " Enter send  ·  Shift/Alt+Enter newline  ·  Ctrl+C exit  ·  PgUp/PgDn scroll  ·  Home/End history  ·  Tab tool  ·  e expand "
    } else {
        " Enter send · Shift/Alt+Enter newline · Ctrl+C exit · PgUp/PgDn scroll · Tab tool "
    };
    frame.render_widget(
        Paragraph::new(help).style(Style::default().fg(Color::DarkGray)),
        area,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputMetrics {
    cursor_row: usize,
    cursor_column: usize,
    total_rows: usize,
}

fn input_metrics(input: &str, cursor: usize, width: usize) -> InputMetrics {
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

fn input_height(input: &InputBuffer, terminal_width: u16) -> u16 {
    let inner_width = terminal_width.saturating_sub(2).max(1) as usize;
    let rows = input_metrics(&input.content, input.cursor, inner_width)
        .total_rows
        .min(MAX_INPUT_ROWS as usize) as u16;
    rows + 2
}

fn format_json(value: &str) -> String {
    serde_json::from_str::<Value>(value)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| value.to_owned())
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let content: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{content}\n… truncated")
    } else {
        content
    }
}

fn single_line(value: &str, max_chars: usize) -> String {
    truncate_chars(&value.replace(['\r', '\n'], " "), max_chars).replace("\n… truncated", " …")
}

struct TerminalSession {
    terminal: DefaultTerminal,
    restored: bool,
}

impl TerminalSession {
    fn start() -> Result<Self> {
        enable_raw_mode().context("failed to enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(stdout, EnterAlternateScreen, EnableBracketedPaste) {
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
        execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        )
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
        let _ = execute!(
            self.terminal.backend_mut(),
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = self.terminal.show_cursor();
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::{KeyEvent, KeyEventKind, KeyEventState};
    use ratatui::{Terminal, backend::TestBackend};

    use super::{
        App, InputAction, InputBuffer, Status, ToolStatus, TranscriptEntry, handle_key_event,
        input_metrics, render, truncate_chars,
    };
    use crate::agent::{AgentEvent, MessageRole};

    fn app() -> App {
        App::new(&[], "test-model".to_owned())
    }

    fn key(code: crossterm::event::KeyCode, modifiers: crossterm::event::KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn folds_streamed_assistant_deltas_and_tracks_tool_details() {
        let mut app = app();
        app.start_turn();
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
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
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
        assert_eq!(tool.arguments, "{\n  \"path\": \"Cargo.toml\"\n}");
        assert_eq!(tool.output, "Cargo.toml");
        assert!(!tool.expanded);
    }

    #[test]
    fn interruption_restores_idle_state_and_marks_running_tools() {
        let mut app = app();
        app.start_turn();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"sleep 30"}"#.to_owned(),
        });

        app.apply_agent_event(AgentEvent::TurnCancelled);

        assert_eq!(app.status, Status::Idle);
        assert!(!app.busy);
        assert!(app.active_tools.is_empty());
        let TranscriptEntry::Tool(tool) = &app.transcript[0] else {
            panic!("expected tool entry");
        };
        assert_eq!(tool.status, ToolStatus::Cancelled);
        assert!(matches!(
            app.transcript.last(),
            Some(TranscriptEntry::Notice(message)) if message.contains("interrupted")
        ));
    }

    #[test]
    fn duplicate_errors_are_shown_once() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::Error {
            message: "provider failed".to_owned(),
        });
        app.record_error_if_new("provider failed".to_owned());

        assert_eq!(app.errors.len(), 1);
        assert_eq!(
            app.transcript
                .iter()
                .filter(|entry| matches!(entry, TranscriptEntry::Error(_)))
                .count(),
            1
        );
    }

    #[test]
    fn tool_failures_use_the_tool_row_without_repeating_an_error_row() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"missing"}"#.to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            output: "tool error: file not found".to_owned(),
            is_error: true,
        });

        assert_eq!(
            app.errors.back().map(String::as_str),
            Some("tool error: file not found")
        );
        assert!(
            !app.transcript
                .iter()
                .any(|entry| matches!(entry, TranscriptEntry::Error(_)))
        );
    }

    #[test]
    fn multiline_input_edits_at_the_cursor_and_submits_trimmed_text() {
        let mut input = InputBuffer::default();
        input.insert_str("firstsecond");
        for _ in 0..6 {
            input.move_left();
        }
        input.insert_char('\n');
        input.insert_str("new ");
        input.backspace();
        input.insert_char(' ');

        assert_eq!(input.content, "first\nnew second");
        assert_eq!(input.take_trimmed(), "first\nnew second");
        assert!(input.is_empty());
    }

    #[test]
    fn input_metrics_wrap_wide_characters_and_track_newlines() {
        let metrics = input_metrics("ab你好\ncd", "ab你好\ncd".len(), 5);

        assert_eq!(metrics.cursor_row, 2);
        assert_eq!(metrics.cursor_column, 2);
        assert_eq!(metrics.total_rows, 3);
    }

    #[test]
    fn keymap_distinguishes_submit_newline_interrupt_and_exit() {
        let mut app = app();
        app.input.insert_str("hello");
        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Enter,
                    crossterm::event::KeyModifiers::SHIFT,
                ),
                &mut app,
                false,
            ),
            InputAction::None
        );
        assert_eq!(app.input.content, "hello\n");
        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Char('c'),
                    crossterm::event::KeyModifiers::CONTROL,
                ),
                &mut app,
                true,
            ),
            InputAction::Interrupt
        );
        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Char('c'),
                    crossterm::event::KeyModifiers::CONTROL,
                ),
                &mut app,
                false,
            ),
            InputAction::Quit
        );
        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Enter,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
            ),
            InputAction::Submit("hello".to_owned())
        );
    }

    #[test]
    fn tool_selection_and_expansion_are_explicit() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            arguments: "{}".to_owned(),
        });
        app.select_tool(false);
        app.toggle_selected_tool();

        assert_eq!(app.selected_tool.as_deref(), Some("call-1"));
        let TranscriptEntry::Tool(tool) = &app.transcript[0] else {
            panic!("expected tool entry");
        };
        assert!(tool.expanded);
        assert!(app.cancel_ui_layer());
        let TranscriptEntry::Tool(tool) = &app.transcript[0] else {
            panic!("expected tool entry");
        };
        assert!(!tool.expanded);
    }

    #[test]
    fn long_tool_details_are_truncated() {
        assert_eq!(truncate_chars("abcdef", 4), "abcd\n… truncated");
    }

    #[test]
    fn slash_command_notices_render_readably_without_user_messages() {
        let mut app = App::new(&[], "gpt-test".to_owned());
        app.transcript.push(TranscriptEntry::Notice(
            "/help\n/model\n/clear\n/sessions\n/resume [id]\n/compact".to_owned(),
        ));
        let backend = TestBackend::new(100, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("INFO"));
        assert!(screen.contains("/help"));
        assert!(screen.contains("/compact"));
        assert!(!screen.contains("YOU"));
    }

    #[test]
    fn renders_status_conversation_folded_tool_input_and_help_regions() {
        let mut app = App::new(&[], "gpt-test".to_owned());
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "Inspect the project".to_owned(),
        });
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "I will read the manifest.".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            output: "package zex".to_owned(),
            is_error: false,
        });
        app.input.insert_str("next prompt");
        let backend = TestBackend::new(120, 28);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("ZEX"));
        assert!(screen.contains("gpt-test"));
        assert!(screen.contains("Conversation"));
        assert!(screen.contains("YOU"));
        assert!(screen.contains("ASSISTANT"));
        assert!(screen.contains("+ TOOL read"));
        assert!(!screen.contains("\"path\": \"Cargo.toml\""));
        assert!(screen.contains("next prompt"));
        assert!(screen.contains("Enter send"));
    }
}
