use std::{
    collections::{BTreeMap, VecDeque},
    io::{self, IsTerminal},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, EventStream, KeyCode, KeyEvent, KeyModifiers, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use ratatui::{
    DefaultTerminal, Frame, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
};
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{
    agent::{Agent, AgentEvent, Message, MessageRole, PromptOutcome},
    command::{CommandEffect, CommandOutput, CommandSpec, command_specs, execute, parse},
    provider::{Provider, ThinkingLevel},
    session::{SessionStore, SessionSummary},
};

const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const TOAST_DURATION: Duration = Duration::from_secs(4);
const MAX_ERRORS: usize = 3;
const MAX_ERROR_DETAIL_CHARS: usize = 4_000;
const MAX_TOOL_DETAIL_CHARS: usize = 4_000;
const MAX_TOOL_ARGUMENT_CHARS: usize = 2_000;
const MAX_INPUT_HISTORY: usize = 100;
const MAX_INPUT_ROWS: u16 = 6;
const MIN_TRANSCRIPT_HEIGHT: u16 = 2;
const HORIZONTAL_GUTTER: u16 = 1;
const INPUT_PROMPT: &str = "› ";
const SCROLL_STEP: usize = 3;
const PASTE_BURST_WINDOW: Duration = Duration::from_millis(12);

const SURFACE: Color = Color::Rgb(20, 23, 28);
const SURFACE_RAISED: Color = Color::Rgb(27, 31, 37);
const TEXT: Color = Color::Rgb(207, 213, 220);
const TEXT_STRONG: Color = Color::Rgb(235, 239, 243);
const DIM: Color = Color::Rgb(111, 120, 131);
const MUTED: Color = Color::Rgb(69, 77, 87);
const ACCENT: Color = Color::Rgb(104, 165, 184);
const SUCCESS: Color = Color::Rgb(116, 171, 136);
const ERROR: Color = Color::Rgb(198, 111, 118);

pub fn is_available() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub async fn run<P>(
    agent: &mut Agent<P>,
    event_receiver: mpsc::UnboundedReceiver<AgentEvent>,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
    working_dir: &Path,
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
        working_dir,
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
    working_dir: &Path,
) -> Result<()>
where
    P: Provider,
{
    let mut app = App::new(
        agent.messages(),
        agent.model().to_owned(),
        session_id.clone(),
        AppContext {
            working_dir: working_dir.to_path_buf(),
            thinking_level: agent.thinking_level(),
            thinking_preference: agent.thinking_preference(),
            context_chars: agent.context_chars(),
            max_context_chars: agent.max_context_chars(),
            default_tool_timeout: agent.default_tool_timeout(),
        },
    );
    let mut redraw = tokio::time::interval(FRAME_INTERVAL);
    redraw.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut dirty = true;
    let mut burst = KeyBurst::default();

    loop {
        tokio::select! {
            _ = redraw.tick() => {
                dirty |= app.expire_toast(Instant::now());
                if dirty {
                    terminal
                        .draw(|frame| render(frame, &mut app))
                        .context("failed to draw TUI")?;
                    dirty = false;
                }
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
                    Some(Ok(event)) => match handle_terminal_event(
                        event,
                        &mut app,
                        false,
                        &mut burst,
                        Instant::now(),
                    ) {
                        InputAction::None => {}
                        InputAction::Quit => return Ok(()),
                        InputAction::Interrupt => {}
                        InputAction::Resume(requested_id) => {
                            match execute(
                                crate::command::SlashCommand::Resume(Some(requested_id)),
                                agent,
                                session_store,
                                session_id,
                                working_dir,
                            )
                            .await
                            {
                                Ok(result) => {
                                    app.sync_agent_status(agent, session_id.as_deref());
                                    match result.effect {
                                        CommandEffect::None => {}
                                        CommandEffect::ClearView => app.reset_transcript(),
                                        CommandEffect::ReplaceView => {
                                            app.replace_transcript(agent.messages());
                                        }
                                    }
                                    if app.push_command_output(result.output) {
                                        app.scroll_to_bottom();
                                    }
                                }
                                Err(error) => app.record_error(format!("{error:#}")),
                            }
                        }
                        InputAction::Submit(prompt) => {
                            app.remember_submission(&prompt);
                            match parse(&prompt) {
                                Ok(Some(command)) => {
                                    match execute(
                                        command,
                                        agent,
                                        session_store,
                                        session_id,
                                        working_dir,
                                    )
                                    .await
                                    {
                                        Ok(result) => {
                                            app.sync_agent_status(agent, session_id.as_deref());
                                            match result.effect {
                                                CommandEffect::None => {}
                                                CommandEffect::ClearView => app.reset_transcript(),
                                                CommandEffect::ReplaceView => {
                                                    app.replace_transcript(agent.messages());
                                                }
                                            }
                                            if app.push_command_output(result.output) {
                                                app.scroll_to_bottom();
                                            }
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
                                    app.sync_agent_status(agent, session_id.as_deref());
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
    let mut burst = KeyBurst::default();

    loop {
        tokio::select! {
            _ = redraw.tick() => {
                dirty |= app.expire_toast(Instant::now());
                if dirty {
                    terminal
                        .draw(|frame| render(frame, app))
                        .context("failed to draw TUI")?;
                    dirty = false;
                }
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
                    Some(Ok(event)) => match handle_terminal_event(
                        event,
                        app,
                        true,
                        &mut burst,
                        Instant::now(),
                    ) {
                        InputAction::Interrupt if !cancellation_requested => {
                            cancellation_requested = true;
                            app.status = Status::Cancelling;
                            let _ = cancel_sender.send(true);
                        }
                        InputAction::None | InputAction::Interrupt => {}
                        InputAction::Quit | InputAction::Resume(_) | InputAction::Submit(_) => {}
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
    Resume(String),
    Submit(String),
}

#[derive(Debug, Default)]
struct KeyBurst {
    last_plain_key: Option<Instant>,
}

impl KeyBurst {
    fn observe(&mut self, key: KeyEvent, now: Instant) -> bool {
        if key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
        {
            self.last_plain_key = None;
            return false;
        }
        let in_burst = self
            .last_plain_key
            .is_some_and(|previous| now.duration_since(previous) <= PASTE_BURST_WINDOW);
        if matches!(
            key.code,
            KeyCode::Char(_) | KeyCode::Enter | KeyCode::Tab | KeyCode::BackTab
        ) {
            self.last_plain_key = Some(now);
        } else {
            self.last_plain_key = None;
        }
        in_burst
    }

    fn reset(&mut self) {
        self.last_plain_key = None;
    }
}

fn handle_terminal_event(
    event: Event,
    app: &mut App,
    turn_active: bool,
    burst: &mut KeyBurst,
    now: Instant,
) -> InputAction {
    match event {
        Event::Paste(content) if !turn_active => {
            burst.reset();
            app.prepare_input_edit();
            app.input.insert_str(&content);
            app.refresh_completion();
            InputAction::None
        }
        Event::Mouse(mouse) if !app.completion_open() && !app.session_picker_open() => {
            match mouse.kind {
                MouseEventKind::ScrollUp => app.scroll_lines_up(SCROLL_STEP),
                MouseEventKind::ScrollDown => app.scroll_lines_down(SCROLL_STEP),
                _ => {}
            }
            InputAction::None
        }
        Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
            let in_paste_burst = burst.observe(key, now);
            handle_key_event(key, app, turn_active, in_paste_burst)
        }
        _ => InputAction::None,
    }
}

fn handle_key_event(
    key: KeyEvent,
    app: &mut App,
    turn_active: bool,
    in_paste_burst: bool,
) -> InputAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return if turn_active {
            InputAction::Interrupt
        } else {
            InputAction::Quit
        };
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        app.toggle_selected_tool();
        return InputAction::None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('e') {
        app.toggle_latest_error();
        return InputAction::None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
        return if turn_active {
            InputAction::None
        } else {
            InputAction::Submit("/think".to_owned())
        };
    }

    if app.session_picker_open() && !turn_active {
        match key.code {
            KeyCode::Up => app.select_session(true),
            KeyCode::Down => app.select_session(false),
            KeyCode::Enter => {
                if let Some(session_id) = app.take_selected_session() {
                    return InputAction::Resume(session_id);
                }
            }
            KeyCode::Esc => app.dismiss_session_picker(),
            _ => {}
        }
        return InputAction::None;
    }

    if app.completion_open() && !turn_active {
        match key.code {
            KeyCode::Up => {
                app.select_completion(true);
                return InputAction::None;
            }
            KeyCode::Down => {
                app.select_completion(false);
                return InputAction::None;
            }
            KeyCode::Tab => {
                app.accept_completion();
                return InputAction::None;
            }
            KeyCode::Enter => {
                if app.complete_or_execute_selected() {
                    return InputAction::None;
                }
            }
            KeyCode::Esc => {
                app.dismiss_completion();
                return InputAction::None;
            }
            _ => {}
        }
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
            if app.input.is_empty() {
                app.select_tool(false);
            }
            return InputAction::None;
        }
        KeyCode::BackTab => {
            if app.input.is_empty() {
                app.select_tool(true);
            }
            return InputAction::None;
        }
        KeyCode::Esc => {
            return if app.cancel_ui_layer() || turn_active {
                InputAction::None
            } else {
                InputAction::Quit
            };
        }
        _ => {}
    }

    if turn_active {
        return InputAction::None;
    }

    match key.code {
        KeyCode::Up => {
            app.navigate_history(true);
            return InputAction::None;
        }
        KeyCode::Down => {
            app.navigate_history(false);
            return InputAction::None;
        }
        _ => {}
    }

    match key {
        KeyEvent {
            code: KeyCode::Enter,
            modifiers,
            ..
        } if modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) || in_paste_burst => {
            app.prepare_input_edit();
            app.input.insert_char('\n');
            app.refresh_completion();
        }
        KeyEvent {
            code: KeyCode::Enter,
            ..
        } => {
            let prompt = app.take_input();
            if !prompt.is_empty() {
                return InputAction::Submit(prompt);
            }
        }
        KeyEvent {
            code: KeyCode::Backspace,
            ..
        } => {
            app.prepare_input_edit();
            app.input.backspace();
            app.refresh_completion();
        }
        KeyEvent {
            code: KeyCode::Delete,
            ..
        } => {
            app.prepare_input_edit();
            app.input.delete();
            app.refresh_completion();
        }
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
            app.prepare_input_edit();
            app.input.insert_char(character);
            app.refresh_completion();
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
            Self::Idle => "ready",
            Self::Thinking => "thinking",
            Self::RunningTool => "working",
            Self::Cancelling => "stopping",
            Self::Error => "error",
        }
    }

    fn color(self) -> Color {
        match self {
            Self::Idle => SUCCESS,
            Self::Thinking | Self::RunningTool | Self::Cancelling => ACCENT,
            Self::Error => ERROR,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::Idle => "●",
            Self::Thinking | Self::RunningTool => "◌",
            Self::Cancelling => "◍",
            Self::Error => "×",
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
            Self::Running | Self::Cancelled => ACCENT,
            Self::Done => SUCCESS,
            Self::Failed => ERROR,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::Running => "◌",
            Self::Done => "✓",
            Self::Failed => "×",
            Self::Cancelled => "−",
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
    started_at: Option<Instant>,
    elapsed: Option<Duration>,
    timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TranscriptEntry {
    Message {
        role: MessageRole,
        content: String,
    },
    Tool(ToolEntry),
    Error {
        summary: String,
        detail: String,
        expanded: bool,
    },
    Sessions(Vec<crate::session::SessionSummary>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToastTone {
    Neutral,
    Success,
}

#[derive(Debug)]
struct Toast {
    message: String,
    tone: ToastTone,
    expires_at: Instant,
}

impl Toast {
    fn new(message: String, tone: ToastTone) -> Self {
        Self {
            message,
            tone,
            expires_at: Instant::now() + TOAST_DURATION,
        }
    }

    fn color(&self) -> Color {
        match self.tone {
            ToastTone::Neutral => ACCENT,
            ToastTone::Success => SUCCESS,
        }
    }
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

    fn replace(&mut self, value: &str) {
        self.content.clear();
        self.content.push_str(value);
        self.cursor = self.content.len();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GitStatus {
    branch: String,
    commit: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompletionState {
    selected: usize,
    dismissed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionPicker {
    sessions: Vec<SessionSummary>,
    selected: usize,
}

#[derive(Debug, Clone)]
struct AppContext {
    working_dir: PathBuf,
    thinking_level: Option<ThinkingLevel>,
    thinking_preference: ThinkingLevel,
    context_chars: usize,
    max_context_chars: usize,
    default_tool_timeout: Duration,
}

#[derive(Debug)]
struct App {
    model: String,
    session_id: Option<String>,
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
    working_dir: PathBuf,
    git_status: Option<GitStatus>,
    thinking_level: Option<ThinkingLevel>,
    thinking_preference: ThinkingLevel,
    context_chars: usize,
    max_context_chars: usize,
    default_tool_timeout: Duration,
    completion: CompletionState,
    input_history: VecDeque<String>,
    history_cursor: Option<usize>,
    history_draft: String,
    toast: Option<Toast>,
    help_open: bool,
    session_picker: Option<SessionPicker>,
}

impl App {
    fn new(
        messages: &[Message],
        model: String,
        session_id: Option<String>,
        context: AppContext,
    ) -> Self {
        let git_status = load_git_status(&context.working_dir);
        let mut app = Self {
            model,
            session_id,
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
            working_dir: context.working_dir,
            git_status,
            thinking_level: context.thinking_level,
            thinking_preference: context.thinking_preference,
            context_chars: context.context_chars,
            max_context_chars: context.max_context_chars,
            default_tool_timeout: context.default_tool_timeout,
            completion: CompletionState {
                selected: 0,
                dismissed: false,
            },
            input_history: VecDeque::new(),
            history_cursor: None,
            history_draft: String::new(),
            toast: None,
            help_open: false,
            session_picker: None,
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
                            started_at: None,
                            elapsed: None,
                            timeout: app.default_tool_timeout,
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
        self.help_open = false;
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
                timeout,
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
                    started_at: Some(Instant::now()),
                    elapsed: None,
                    timeout,
                }));
            }
            AgentEvent::ToolEnd {
                call_id,
                name,
                output,
                is_error,
                elapsed,
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
                    tool.started_at = None;
                    tool.elapsed = Some(elapsed);
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
                self.show_toast(
                    format!(
                        "Context compacted · −{} chars · {} recent turns kept",
                        stats.freed_chars, stats.kept_turns
                    ),
                    ToastTone::Neutral,
                );
            }
            AgentEvent::TurnCancelled => {
                for entry in &mut self.transcript {
                    if let TranscriptEntry::Tool(tool) = entry
                        && tool.status == ToolStatus::Running
                    {
                        tool.status = ToolStatus::Cancelled;
                        tool.elapsed = tool.started_at.map(|started| started.elapsed());
                        tool.started_at = None;
                    }
                }
                self.finish_turn(Status::Idle);
                self.show_toast("Turn interrupted".to_owned(), ToastTone::Neutral);
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
        self.toast = None;
        self.help_open = false;
        self.session_picker = None;
        self.scroll_to_bottom();
    }

    fn replace_transcript(&mut self, messages: &[Message]) {
        let model = self.model.clone();
        *self = Self::new(
            messages,
            model,
            self.session_id.clone(),
            AppContext {
                working_dir: self.working_dir.clone(),
                thinking_level: self.thinking_level,
                thinking_preference: self.thinking_preference,
                context_chars: self.context_chars,
                max_context_chars: self.max_context_chars,
                default_tool_timeout: self.default_tool_timeout,
            },
        );
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
        let detail = truncate_chars(&message, MAX_ERROR_DETAIL_CHARS);
        self.transcript.push(TranscriptEntry::Error {
            summary: error_summary(&detail),
            detail,
            expanded: false,
        });
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

    fn push_command_output(&mut self, output: CommandOutput) -> bool {
        match output {
            CommandOutput::Help => {
                self.help_open = true;
                return false;
            }
            CommandOutput::Sessions(sessions) => {
                self.transcript.push(TranscriptEntry::Sessions(sessions));
            }
            CommandOutput::ResumePicker(sessions) => {
                self.input.clear();
                self.reset_history_navigation();
                self.open_session_picker(sessions);
                return false;
            }
            CommandOutput::Status(message) => {
                self.show_toast(message, ToastTone::Success);
                return false;
            }
        }
        true
    }

    fn scroll_page_up(&mut self) {
        self.scroll_lines_up(self.transcript_page_height.saturating_sub(1).max(1));
    }

    fn scroll_page_down(&mut self) {
        self.scroll_lines_down(self.transcript_page_height.saturating_sub(1).max(1));
    }

    fn scroll_lines_up(&mut self, lines: usize) {
        self.follow_output = false;
        self.scroll_top = self.scroll_top.saturating_sub(lines.max(1));
    }

    fn scroll_lines_down(&mut self, lines: usize) {
        let next = self.scroll_top.saturating_add(lines.max(1));
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

    fn toggle_latest_error(&mut self) {
        if let Some(TranscriptEntry::Error { expanded, .. }) = self
            .transcript
            .iter_mut()
            .rev()
            .find(|entry| matches!(entry, TranscriptEntry::Error { .. }))
        {
            *expanded = !*expanded;
        }
    }

    fn cancel_ui_layer(&mut self) -> bool {
        if self.session_picker_open() {
            self.dismiss_session_picker();
            return true;
        }
        if self.completion_open() {
            self.dismiss_completion();
            return true;
        }
        if self.help_open {
            self.help_open = false;
            return true;
        }
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
        if let Some(TranscriptEntry::Error { expanded, .. }) = self
            .transcript
            .iter_mut()
            .rev()
            .find(|entry| matches!(entry, TranscriptEntry::Error { .. }))
            && *expanded
        {
            *expanded = false;
            return true;
        }
        if !self.input.is_empty() {
            self.input.clear();
            self.reset_history_navigation();
            return true;
        }
        if !self.follow_output {
            self.scroll_to_bottom();
            return true;
        }
        false
    }

    fn sync_agent_status<P>(&mut self, agent: &Agent<P>, session_id: Option<&str>)
    where
        P: Provider,
    {
        self.model = agent.model().to_owned();
        self.session_id = session_id.map(str::to_owned);
        self.thinking_level = agent.thinking_level();
        self.thinking_preference = agent.thinking_preference();
        self.context_chars = agent.context_chars();
        self.max_context_chars = agent.max_context_chars();
    }

    fn open_session_picker(&mut self, sessions: Vec<SessionSummary>) {
        self.help_open = false;
        self.completion.dismissed = true;
        self.session_picker = Some(SessionPicker {
            selected: sessions
                .iter()
                .position(|session| Some(session.id.as_str()) == self.session_id.as_deref())
                .unwrap_or(0),
            sessions,
        });
    }

    fn session_picker_open(&self) -> bool {
        self.session_picker.is_some()
    }

    fn select_session(&mut self, reverse: bool) {
        let Some(picker) = &mut self.session_picker else {
            return;
        };
        let count = picker.sessions.len();
        if count == 0 {
            return;
        }
        picker.selected = if reverse {
            picker.selected.checked_sub(1).unwrap_or(count - 1)
        } else {
            (picker.selected + 1) % count
        };
    }

    fn take_selected_session(&mut self) -> Option<String> {
        let picker = self.session_picker.take()?;
        picker
            .sessions
            .get(picker.selected)
            .map(|session| session.id.clone())
    }

    fn dismiss_session_picker(&mut self) {
        self.session_picker = None;
    }

    fn completion_matches(&self) -> Vec<&'static CommandSpec> {
        let input = self.input.content.trim_start();
        if self.busy
            || self.completion.dismissed
            || !input.starts_with('/')
            || input.contains(char::is_whitespace)
        {
            return Vec::new();
        }
        command_specs()
            .iter()
            .filter(|command| command.name.starts_with(input))
            .collect()
    }

    fn completion_open(&self) -> bool {
        !self.completion_matches().is_empty()
    }

    fn refresh_completion(&mut self) {
        self.completion.dismissed = false;
        let count = self.completion_matches().len();
        self.completion.selected = self.completion.selected.min(count.saturating_sub(1));
    }

    fn select_completion(&mut self, reverse: bool) {
        let count = self.completion_matches().len();
        if count == 0 {
            return;
        }
        self.completion.selected = if reverse {
            self.completion.selected.checked_sub(1).unwrap_or(count - 1)
        } else {
            (self.completion.selected + 1) % count
        };
    }

    fn accept_completion(&mut self) {
        let matches = self.completion_matches();
        let Some(command) = matches.get(self.completion.selected) else {
            return;
        };
        let completed = format!(
            "{}{}",
            command.name,
            if command.accepts_arguments { " " } else { "" }
        );
        self.input.replace(&completed);
        self.reset_history_navigation();
        self.completion.dismissed = true;
    }

    fn complete_or_execute_selected(&mut self) -> bool {
        let matches = self.completion_matches();
        let Some(command) = matches.get(self.completion.selected) else {
            return false;
        };
        if self.input.content.trim() == command.name {
            if command.accepts_arguments {
                self.completion.dismissed = true;
            }
            return false;
        }
        self.accept_completion();
        true
    }

    fn dismiss_completion(&mut self) {
        self.completion.dismissed = true;
    }

    fn remember_submission(&mut self, prompt: &str) {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return;
        }
        if self.input_history.back().map(String::as_str) != Some(prompt) {
            if self.input_history.len() == MAX_INPUT_HISTORY {
                self.input_history.pop_front();
            }
            self.input_history.push_back(prompt.to_owned());
        }
        self.reset_history_navigation();
    }

    fn navigate_history(&mut self, older: bool) {
        if self.input_history.is_empty() {
            return;
        }
        if older {
            let index = match self.history_cursor {
                Some(0) => 0,
                Some(index) => index - 1,
                None => {
                    self.history_draft = self.input.content.clone();
                    self.input_history.len() - 1
                }
            };
            self.history_cursor = Some(index);
            if let Some(value) = self.input_history.get(index) {
                self.input.replace(value);
            }
        } else {
            let Some(index) = self.history_cursor else {
                return;
            };
            if index + 1 < self.input_history.len() {
                self.history_cursor = Some(index + 1);
                if let Some(value) = self.input_history.get(index + 1) {
                    self.input.replace(value);
                }
            } else {
                let draft = std::mem::take(&mut self.history_draft);
                self.history_cursor = None;
                self.input.replace(&draft);
            }
        }
        self.refresh_completion();
    }

    fn prepare_input_edit(&mut self) {
        self.help_open = false;
        self.reset_history_navigation();
    }

    fn reset_history_navigation(&mut self) {
        self.history_cursor = None;
        self.history_draft.clear();
    }

    fn take_input(&mut self) -> String {
        self.reset_history_navigation();
        self.input.take_trimmed()
    }

    fn show_toast(&mut self, message: String, tone: ToastTone) {
        self.toast = Some(Toast::new(single_line(&message, 160), tone));
    }

    fn expire_toast(&mut self, now: Instant) -> bool {
        if self
            .toast
            .as_ref()
            .is_some_and(|toast| now >= toast.expires_at)
        {
            self.toast = None;
            return true;
        }
        false
    }
}

fn load_git_status(working_dir: &Path) -> Option<GitStatus> {
    let branch = git_output(working_dir, &["branch", "--show-current"]).or_else(|| {
        git_output(working_dir, &["rev-parse", "--short", "HEAD"])
            .map(|commit| format!("detached:{commit}"))
    })?;
    let commit = git_output(working_dir, &["rev-parse", "--short", "HEAD"])?;
    Some(GitStatus { branch, commit })
}

fn git_output(working_dir: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(working_dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

fn render(frame: &mut Frame<'_>, app: &mut App) {
    frame.render_widget(Clear, frame.area());
    let regions = ui_regions(frame.area(), app);

    render_transcript(frame, regions.transcript, app);
    render_completion(frame, regions.completion, app);
    render_status(frame, regions.status, app);
    render_keymap(frame, regions.keymap, app);
    render_input(frame, regions.input, app);
    render_help_overlay(frame, regions.transcript, app);
    render_session_picker(frame, regions.transcript, app);
}

fn render_status(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.is_empty() {
        return;
    }
    let footer = content_area(area);
    let rail_color = if app.busy {
        ACCENT
    } else {
        status_rail_color(app.status)
    };
    let content = render_footer_rails(frame, footer, rail_color);
    let content = horizontal_inset(content, 1);
    if content.is_empty() {
        return;
    }
    let context_percent = app
        .context_chars
        .saturating_mul(100)
        .checked_div(app.max_context_chars.max(1))
        .unwrap_or(0)
        .min(999);
    let thinking = app.thinking_level.map_or_else(
        || format!("n/a ({})", app.thinking_preference),
        |level| level.to_string(),
    );
    let location = match &app.git_status {
        Some(git) => format!(
            "{}  {}@{}",
            short_path(&app.working_dir, 32),
            single_line(&git.branch, 18),
            git.commit
        ),
        None => short_path(&app.working_dir, 48),
    };
    let session = app
        .session_id
        .as_deref()
        .map(short_session_id)
        .unwrap_or("new");

    let right_text = if content.width >= 18 {
        format!("ctx {context_percent}%")
    } else {
        format!("{context_percent}%")
    };
    let right_width = UnicodeWidthStr::width(right_text.as_str()).min(content.width as usize);
    let gap = usize::from(content.width as usize > right_width + 1);
    let left_width = content
        .width
        .saturating_sub(right_width as u16)
        .saturating_sub(gap as u16);
    let left = Rect::new(content.x, content.y, left_width, content.height);
    let right = Rect::new(
        content.right().saturating_sub(right_width as u16),
        content.y,
        right_width as u16,
        content.height,
    );

    let model_limit = if left_width >= 80 {
        32
    } else if left_width >= 52 {
        24
    } else {
        14
    };
    let mut spans = vec![
        Span::styled(
            "ZEX",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(app.status.symbol(), Style::default().fg(app.status.color())),
        Span::styled(
            format!(" {}", app.status.label()),
            Style::default().fg(app.status.color()),
        ),
        Span::styled("  ·  ", Style::default().fg(MUTED)),
        Span::styled(
            single_line(&app.model, model_limit),
            Style::default().fg(TEXT_STRONG),
        ),
    ];
    if left_width >= 42 {
        spans.extend([
            Span::styled("  ·  think ", Style::default().fg(DIM)),
            Span::styled(thinking, Style::default().fg(TEXT)),
        ]);
    }
    if left_width >= 62 && app.session_id.is_some() {
        spans.extend([
            Span::styled("  ·  session ", Style::default().fg(DIM)),
            Span::styled(session, Style::default().fg(TEXT)),
        ]);
    }
    if left_width >= 76 {
        spans.extend([
            Span::styled("  ·  ", Style::default().fg(MUTED)),
            Span::styled(location, Style::default().fg(DIM)),
        ]);
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), left);
    if right.width > 0 {
        frame.render_widget(
            Paragraph::new(if content.width >= 18 {
                Line::from(vec![
                    Span::styled("ctx ", Style::default().fg(DIM)),
                    Span::styled(
                        format!("{context_percent}%"),
                        Style::default().fg(TEXT_STRONG),
                    ),
                ])
            } else {
                Line::from(Span::styled(
                    format!("{context_percent}%"),
                    Style::default().fg(TEXT_STRONG),
                ))
            })
            .alignment(Alignment::Right),
            right,
        );
    } else if content.width > 0 {
        frame.render_widget(
            Paragraph::new(right_text)
                .style(Style::default().fg(TEXT_STRONG))
                .alignment(Alignment::Right),
            content,
        );
    }
}

fn render_session_picker(frame: &mut Frame<'_>, viewport: Rect, app: &App) {
    let Some(picker) = &app.session_picker else {
        return;
    };
    if viewport.is_empty() {
        return;
    }

    let viewport = content_area(viewport);
    let width = viewport.width.clamp(1, 88);
    let max_visible = viewport.height.saturating_sub(4).max(1) as usize;
    let visible_count = picker.sessions.len().clamp(1, max_visible);
    let height = viewport
        .height
        .min((visible_count as u16).saturating_mul(2).saturating_add(4))
        .max(1);
    let area = Rect::new(
        viewport.x + viewport.width.saturating_sub(width) / 2,
        viewport.y + viewport.height.saturating_sub(height) / 2,
        width,
        height,
    );
    let inner_width = area.width.saturating_sub(4) as usize;
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Resume session",
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
            Style::default().fg(DIM),
        ),
    ])];

    if picker.sessions.is_empty() {
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "No saved sessions",
            Style::default().fg(DIM),
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
            let selected = index == picker.selected;
            let background = if selected { ACCENT } else { SURFACE_RAISED };
            let foreground = if selected { Color::Black } else { TEXT_STRONG };
            let secondary = if selected { Color::Black } else { DIM };
            let marker = if selected { "›" } else { " " };
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

    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT))
                    .style(Style::default().bg(SURFACE_RAISED))
                    .padding(ratatui::widgets::Padding::horizontal(1)),
            )
            .style(Style::default().bg(SURFACE_RAISED))
            .wrap(Wrap { trim: false }),
        area,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UiRegions {
    transcript: Rect,
    completion: Rect,
    status: Rect,
    keymap: Rect,
    input: Rect,
}

fn ui_regions(area: Rect, app: &App) -> UiRegions {
    let desired_input = input_height(&app.input, area.width);
    let input_height = desired_input
        .min(area.height.saturating_sub(2).max(1))
        .min(area.height);
    let status_height = u16::from(area.height > input_height);
    let keymap_height = u16::from(area.height > input_height + status_height);
    let fixed_height = input_height + status_height + keymap_height;
    let remaining = area.height.saturating_sub(fixed_height);
    let transcript_reserve = MIN_TRANSCRIPT_HEIGHT.min(remaining);
    let completion_height =
        completion_height(app).min(remaining.saturating_sub(transcript_reserve));
    let transcript_height = remaining.saturating_sub(completion_height);

    let transcript = Rect::new(area.x, area.y, area.width, transcript_height);
    let completion = Rect::new(area.x, transcript.bottom(), area.width, completion_height);
    let status = Rect::new(area.x, completion.bottom(), area.width, status_height);
    let keymap = Rect::new(area.x, status.bottom(), area.width, keymap_height);
    let input = Rect::new(area.x, keymap.bottom(), area.width, input_height);

    debug_assert_eq!(input.bottom(), area.bottom());
    debug_assert!(
        transcript.height + completion.height + status.height + keymap.height + input.height
            == area.height
    );

    UiRegions {
        transcript,
        completion,
        status,
        keymap,
        input,
    }
}

fn render_footer_rails(frame: &mut Frame<'_>, area: Rect, color: Color) -> Rect {
    if area.width < 2 || area.height == 0 {
        return area;
    }
    frame.render_widget(
        Block::default()
            .borders(Borders::LEFT | Borders::RIGHT)
            .border_type(BorderType::Thick)
            .border_style(Style::default().fg(color)),
        area,
    );
    Rect::new(
        area.x.saturating_add(1),
        area.y,
        area.width.saturating_sub(2),
        area.height,
    )
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
    if app.transcript.is_empty() {
        app.transcript_page_height = area.height as usize;
        app.max_scroll = 0;
        app.scroll_top = 0;
        render_empty_state(frame, area);
        return;
    }

    let text = transcript_text(app);
    let paragraph = Paragraph::new(text)
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
            Paragraph::new(indicator).style(Style::default().fg(DIM)),
            Rect::new(x, area.y, width, 1),
        );
    }
}

fn render_empty_state(frame: &mut Frame<'_>, area: Rect) {
    if area.is_empty() {
        return;
    }
    let height = area.height.min(2);
    let y = area.y + area.height.saturating_sub(height) / 2;
    let empty_area = Rect::new(area.x, y, area.width, height);
    let mut lines = vec![Line::from(Span::styled(
        "Zex",
        Style::default()
            .fg(TEXT_STRONG)
            .add_modifier(Modifier::BOLD),
    ))];
    if height > 1 {
        lines.push(Line::from(Span::styled(
            "Ask anything, or type / for commands",
            Style::default().fg(DIM),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true }),
        empty_area,
    );
}

fn completion_height(app: &App) -> u16 {
    if app.completion_open() {
        app.completion_matches().len() as u16 + 2
    } else {
        0
    }
}

fn render_completion(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.height == 0 {
        return;
    }
    let area = content_area(area);
    if area.is_empty() {
        return;
    }
    let matches = app.completion_matches();
    let inner_width = area.width.saturating_sub(4) as usize;
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
            let selected = index == app.completion.selected;
            let marker = if selected { "›" } else { " " };
            let available = inner_width.saturating_sub(2);
            let wide = available >= usage_width + 2 + 18;
            let command_style = Style::default()
                .fg(if selected { Color::Black } else { TEXT })
                .bg(if selected { ACCENT } else { SURFACE_RAISED })
                .add_modifier(if selected {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                });
            let marker_style = Style::default()
                .fg(if selected { Color::Black } else { MUTED })
                .bg(if selected { ACCENT } else { SURFACE_RAISED });
            let description_style = Style::default()
                .fg(if selected { Color::Black } else { DIM })
                .bg(if selected { ACCENT } else { SURFACE_RAISED });
            if wide {
                Line::from(vec![
                    Span::styled(format!("{marker} "), marker_style),
                    Span::styled(pad_display(command.usage, usage_width), command_style),
                    Span::raw("  "),
                    Span::styled(command.description, description_style),
                ])
                .style(if selected {
                    Style::default().bg(ACCENT)
                } else {
                    Style::default().bg(SURFACE_RAISED)
                })
            } else {
                Line::from(vec![
                    Span::styled(format!("{marker} "), marker_style),
                    Span::styled(command.usage, command_style),
                    Span::styled(" · ", description_style),
                    Span::styled(command.description, description_style),
                ])
                .style(if selected {
                    Style::default().bg(ACCENT)
                } else {
                    Style::default().bg(SURFACE_RAISED)
                })
            }
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT))
                    .padding(ratatui::widgets::Padding::horizontal(1)),
            )
            .style(Style::default().bg(SURFACE_RAISED))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn transcript_text(app: &App) -> Text<'static> {
    let mut lines = Vec::new();
    for entry in &app.transcript {
        match entry {
            TranscriptEntry::Message { role, content } => {
                if *role == MessageRole::User {
                    lines.push(Line::from(Span::styled(
                        "›",
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    )));
                }
                append_markdown_lines(&mut lines, content, *role);
                lines.push(Line::default());
            }
            TranscriptEntry::Tool(tool) => append_tool_lines(
                &mut lines,
                tool,
                app.selected_tool.as_deref() == Some(tool.call_id.as_str()),
            ),
            TranscriptEntry::Error {
                summary,
                detail,
                expanded,
            } => {
                lines.push(Line::from(vec![
                    Span::styled("  × ", Style::default().fg(ERROR)),
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
                        Style::default().fg(DIM),
                    ),
                ]));
                if *expanded {
                    for source_line in detail.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("    │ ", Style::default().fg(MUTED)),
                            Span::styled(source_line.to_owned(), Style::default().fg(DIM)),
                        ]));
                    }
                }
                lines.push(Line::default());
            }
            TranscriptEntry::Sessions(sessions) => {
                append_session_lines(&mut lines, sessions);
                lines.push(Line::default());
            }
        }
    }

    Text::from(lines)
}

fn render_help_overlay(frame: &mut Frame<'_>, viewport: Rect, app: &App) {
    if !app.help_open || viewport.is_empty() {
        return;
    }

    let viewport = content_area(viewport);
    let width = viewport.width.clamp(1, 76);
    let desired_height = command_specs().len() as u16 + 4;
    let height = viewport.height.min(desired_height).max(1);
    let area = Rect::new(
        viewport.x,
        viewport.bottom().saturating_sub(height),
        width,
        height,
    );
    let inner_width = area.width.saturating_sub(4) as usize;
    let lines = help_lines(inner_width);

    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(ACCENT))
                    .style(Style::default().bg(SURFACE_RAISED))
                    .padding(ratatui::widgets::Padding::horizontal(1)),
            )
            .style(Style::default().bg(SURFACE_RAISED))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn help_lines(width: usize) -> Vec<Line<'static>> {
    let usage_width = command_specs()
        .iter()
        .map(|command| command.usage.len())
        .max()
        .unwrap_or(0);
    let wide = width >= usage_width + 2 + 24;
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Commands",
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  Esc close", Style::default().fg(DIM)),
    ])];
    lines.extend(command_specs().iter().map(|command| {
        if wide {
            Line::from(vec![
                Span::styled(
                    format!("{:<usage_width$}", command.usage),
                    Style::default().fg(ACCENT),
                ),
                Span::raw("  "),
                Span::styled(command.description, Style::default().fg(DIM)),
            ])
        } else {
            Line::from(Span::styled(command.usage, Style::default().fg(ACCENT)))
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
            Span::styled("  · ", Style::default().fg(ACCENT)),
            Span::styled("No saved sessions.", Style::default().fg(DIM)),
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
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            format!(
                "    {} message{} · {}",
                session.message_count,
                if session.message_count == 1 { "" } else { "s" },
                session.preview
            ),
            Style::default().fg(DIM),
        )));
    }
}

fn append_markdown_lines(lines: &mut Vec<Line<'static>>, content: &str, role: MessageRole) {
    let base_color = match role {
        MessageRole::User => TEXT_STRONG,
        MessageRole::Assistant => TEXT,
    };
    let content_prefix = match role {
        MessageRole::User => "  ",
        MessageRole::Assistant => "",
    };
    let mut in_code_block = false;

    for source_line in content.split('\n') {
        let trimmed = source_line.trim_start();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            let language = trimmed.trim_start_matches('`').trim();
            if in_code_block {
                lines.push(
                    Line::from(vec![
                        Span::styled(
                            format!("{content_prefix}▌ "),
                            Style::default().fg(ACCENT).bg(SURFACE),
                        ),
                        Span::styled(
                            if language.is_empty() {
                                "code".to_owned()
                            } else {
                                language.to_owned()
                            },
                            Style::default().fg(DIM).bg(SURFACE),
                        ),
                    ])
                    .style(Style::default().bg(SURFACE)),
                );
            } else {
                lines.push(Line::default());
            }
            continue;
        }
        if in_code_block {
            lines.push(
                Line::from(vec![
                    Span::styled(
                        format!("{content_prefix}▌ "),
                        Style::default().fg(ACCENT).bg(SURFACE),
                    ),
                    Span::styled(
                        source_line.to_owned(),
                        Style::default().fg(TEXT_STRONG).bg(SURFACE),
                    ),
                ])
                .style(Style::default().bg(SURFACE)),
            );
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("### ") {
            lines.push(Line::from(Span::styled(
                format!("{content_prefix}{heading}"),
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if let Some(heading) = trimmed
            .strip_prefix("## ")
            .or_else(|| trimmed.strip_prefix("# "))
        {
            lines.push(Line::from(Span::styled(
                format!("{content_prefix}{heading}"),
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            )));
        } else if let Some(item) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            lines.push(Line::from(vec![
                Span::styled(format!("{content_prefix}• "), Style::default().fg(ACCENT)),
                Span::styled(item.to_owned(), Style::default().fg(base_color)),
            ]));
        } else if let Some((number, item)) = numbered_list_item(trimmed) {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{content_prefix}{number}. "),
                    Style::default().fg(ACCENT),
                ),
                Span::styled(item.to_owned(), Style::default().fg(base_color)),
            ]));
        } else if let Some(quote) = trimmed.strip_prefix("> ") {
            lines.push(Line::from(vec![
                Span::styled(format!("{content_prefix}│ "), Style::default().fg(MUTED)),
                Span::styled(quote.to_owned(), Style::default().fg(DIM)),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                format!("{content_prefix}{source_line}"),
                Style::default().fg(base_color),
            )));
        }
    }
}

fn append_tool_lines(lines: &mut Vec<Line<'static>>, tool: &ToolEntry, selected: bool) {
    let marker = if selected { "›" } else { " " };
    let fold = if tool.expanded { "▾" } else { "▸" };
    let title = tool_title(tool);
    let elapsed = tool_elapsed(tool);
    let card_style = Style::default().bg(SURFACE);
    let rail_color = if selected {
        ACCENT
    } else {
        Color::Rgb(55, 65, 74)
    };
    lines.push(
        Line::from(vec![
            Span::styled(
                format!("{marker} {fold} "),
                Style::default().fg(rail_color).bg(SURFACE),
            ),
            Span::styled(title, Style::default().fg(TEXT_STRONG).bg(SURFACE)),
            Span::styled("  ", card_style),
            Span::styled(
                format!("{} {}", tool.status.symbol(), tool.status.label()),
                Style::default().fg(tool.status.color()).bg(SURFACE),
            ),
            Span::styled(
                format!("  {}", format_duration(elapsed)),
                Style::default().fg(DIM).bg(SURFACE),
            ),
        ])
        .style(card_style),
    );

    if !tool.expanded {
        lines.push(
            Line::from(vec![
                Span::styled("  │  ", Style::default().fg(rail_color).bg(SURFACE)),
                Span::styled(tool_summary(tool), Style::default().fg(DIM).bg(SURFACE)),
            ])
            .style(card_style),
        );
    }

    if tool.expanded {
        lines.push(
            Line::from(vec![
                Span::styled("  │  ", Style::default().fg(rail_color).bg(SURFACE)),
                Span::styled("input", Style::default().fg(DIM).bg(SURFACE)),
            ])
            .style(card_style),
        );
        for line in tool.arguments.split('\n') {
            lines.push(
                Line::from(vec![
                    Span::styled("  │    ", Style::default().fg(rail_color).bg(SURFACE)),
                    Span::styled(line.to_owned(), Style::default().fg(TEXT).bg(SURFACE)),
                ])
                .style(card_style),
            );
        }
        lines.push(
            Line::from(vec![
                Span::styled("  │  ", Style::default().fg(rail_color).bg(SURFACE)),
                Span::styled("output", Style::default().fg(DIM).bg(SURFACE)),
            ])
            .style(card_style),
        );
        if tool.output.is_empty() {
            lines.push(
                Line::from(vec![
                    Span::styled("  │    ", Style::default().fg(rail_color).bg(SURFACE)),
                    Span::styled("waiting for result…", Style::default().fg(DIM).bg(SURFACE)),
                ])
                .style(card_style),
            );
        } else {
            for line in tool.output.split('\n') {
                lines.push(
                    Line::from(vec![
                        Span::styled("  │    ", Style::default().fg(rail_color).bg(SURFACE)),
                        Span::styled(line.to_owned(), Style::default().fg(TEXT).bg(SURFACE)),
                    ])
                    .style(card_style),
                );
            }
        }
        lines.push(
            Line::from(vec![
                Span::styled("  └  ", Style::default().fg(rail_color).bg(SURFACE)),
                Span::styled(
                    format!("timeout {}", format_duration(tool.timeout)),
                    Style::default().fg(DIM).bg(SURFACE),
                ),
            ])
            .style(card_style),
        );
    }
    lines.push(Line::default());
}

fn tool_title(tool: &ToolEntry) -> String {
    if tool.name == "bash"
        && let Ok(arguments) = serde_json::from_str::<Value>(&tool.arguments)
        && let Some(command) = arguments.get("command").and_then(Value::as_str)
    {
        return format!("$ {}", single_line(command, 100));
    }
    tool.name.clone()
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

fn tool_summary(tool: &ToolEntry) -> String {
    match tool.status {
        ToolStatus::Running => return "Working…".to_owned(),
        ToolStatus::Cancelled => return "Interrupted".to_owned(),
        ToolStatus::Failed => {}
        ToolStatus::Done if tool.output.trim().is_empty() => return "Completed".to_owned(),
        ToolStatus::Done => {}
    }

    if tool.name == "bash" {
        if let Some(summary) = bash_output_summary(&tool.output) {
            return summary;
        }
        return if tool.status == ToolStatus::Failed {
            "Command failed".to_owned()
        } else {
            "Completed".to_owned()
        };
    }
    let source = if tool.output.trim().is_empty() {
        &tool.arguments
    } else {
        &tool.output
    };
    single_line(
        source
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or(source),
        120,
    )
}

fn bash_output_summary(output: &str) -> Option<String> {
    let (_, body) = output
        .strip_prefix("exit_code: ")?
        .split_once("\nstdout:\n")?;
    let (stdout, stderr) = body.split_once("\nstderr:\n")?;
    let summary = stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .or_else(|| stderr.lines().find(|line| !line.trim().is_empty()))?;
    Some(single_line(summary, 120))
}

fn render_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    if area.is_empty() {
        return;
    }
    let footer = content_area(area);
    let rail_color = if app.busy {
        ACCENT
    } else {
        status_rail_color(app.status)
    };
    let input_area = horizontal_inset(render_footer_rails(frame, footer, rail_color), 1);
    if input_area.is_empty() {
        return;
    }

    if app.busy {
        let busy_text = match app.status {
            Status::Thinking => "Agent is thinking",
            Status::RunningTool => "Agent is working",
            Status::Cancelling => "Stopping agent",
            Status::Error => "Agent stopped with an error",
            Status::Idle => "Agent is working",
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("◌ ", Style::default().fg(ACCENT)),
                Span::styled(busy_text, Style::default().fg(TEXT)),
                Span::styled(" · Ctrl+C to stop", Style::default().fg(DIM)),
            ]))
            .wrap(Wrap { trim: false }),
            input_area,
        );
        return;
    }

    let prompt_width = UnicodeWidthStr::width(INPUT_PROMPT).min(input_area.width as usize) as u16;
    let prompt_area = Rect::new(input_area.x, input_area.y, prompt_width, input_area.height);
    let editor_area = Rect::new(
        prompt_area.right(),
        input_area.y,
        input_area.width.saturating_sub(prompt_width),
        input_area.height,
    );
    if editor_area.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled(INPUT_PROMPT, Style::default().fg(ACCENT))),
            input_area,
        );
        return;
    }

    frame.render_widget(
        Paragraph::new(Span::styled(INPUT_PROMPT, Style::default().fg(ACCENT))),
        prompt_area,
    );

    let editor_width = editor_area.width.max(1) as usize;
    let visible_rows = editor_area.height.max(1) as usize;
    let metrics = input_metrics(&app.input.content, app.input.cursor, editor_width);
    let vertical_scroll = metrics.cursor_row.saturating_sub(visible_rows - 1);
    let editor = if app.input.is_empty() {
        Text::from(Line::from(Span::styled(
            "Ask Zex...",
            Style::default().fg(DIM),
        )))
    } else {
        Text::from(Line::from(Span::styled(
            app.input.content.clone(),
            Style::default().fg(TEXT_STRONG),
        )))
    };
    frame.render_widget(
        Paragraph::new(editor)
            .wrap(Wrap { trim: false })
            .scroll((vertical_scroll.min(u16::MAX as usize) as u16, 0)),
        editor_area,
    );

    let cursor_y = metrics.cursor_row.saturating_sub(vertical_scroll) as u16;
    frame.set_cursor_position((
        editor_area.x + metrics.cursor_column.min(editor_width - 1) as u16,
        editor_area.y + cursor_y.min(editor_area.height.saturating_sub(1)),
    ));
}

fn status_rail_color(status: Status) -> Color {
    match status {
        Status::Idle => Color::Rgb(76, 128, 145),
        Status::Thinking | Status::RunningTool | Status::Cancelling => ACCENT,
        Status::Error => ERROR,
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
    let hint = if app.session_picker_open() {
        Line::from(Span::styled(
            "↑↓ select · Enter resume · Esc cancel",
            Style::default().fg(DIM),
        ))
    } else if let Some(toast) = &app.toast {
        Line::from(vec![
            Span::styled("● ", Style::default().fg(toast.color())),
            Span::styled(toast.message.clone(), Style::default().fg(TEXT)),
        ])
    } else if app.completion_open() {
        Line::from(Span::styled(
            "↑↓ select · Tab complete · Enter run · Esc close",
            Style::default().fg(DIM),
        ))
    } else if app.busy {
        let text = if area.width >= 72 {
            "Ctrl+C stop · PgUp/PgDn scroll · Ctrl+O tool details"
        } else {
            "Ctrl+C stop · PgUp/PgDn scroll"
        };
        Line::from(Span::styled(text, Style::default().fg(DIM)))
    } else {
        let text = if area.width >= 92 {
            "Enter send · Shift+Enter newline · ↑↓ history · / commands"
        } else {
            "Enter send · ↑↓ history · / commands"
        };
        Line::from(Span::styled(text, Style::default().fg(DIM)))
    };
    frame.render_widget(Paragraph::new(hint), area);
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
    let content_width = content_area(Rect::new(0, 0, terminal_width, 1)).width;
    let inner_width = content_width.saturating_sub(6).max(1) as usize;
    let rows = input_metrics(&input.content, input.cursor, inner_width)
        .total_rows
        .min(MAX_INPUT_ROWS as usize) as u16;
    rows.max(1)
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

fn short_path(path: &Path, max_chars: usize) -> String {
    let display = path.display().to_string();
    if display.chars().count() <= max_chars {
        return display;
    }
    let keep = max_chars.saturating_sub(2);
    let tail = display
        .chars()
        .rev()
        .take(keep)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("…{tail}")
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

fn error_summary(message: &str) -> String {
    let first = message
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("Unknown error")
        .trim();
    single_line(first, 120)
}

struct TerminalSession {
    terminal: DefaultTerminal,
    restored: bool,
}

impl TerminalSession {
    fn start() -> Result<Self> {
        enable_raw_mode().context("failed to enable terminal raw mode")?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        ) {
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
            DisableMouseCapture,
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
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = self.terminal.show_cursor();
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, time::Duration};

    use crossterm::event::{KeyEvent, KeyEventKind, KeyEventState};
    use ratatui::{Terminal, backend::TestBackend};

    use super::{
        ACCENT, App, AppContext, CommandOutput, InputAction, InputBuffer, KeyBurst, SCROLL_STEP,
        SURFACE, SURFACE_RAISED, Status, ToolStatus, TranscriptEntry, UiRegions, command_specs,
        handle_key_event, handle_terminal_event, input_metrics, render, truncate_chars, ui_regions,
    };
    use crate::agent::{AgentEvent, Message, MessageRole};
    use crate::provider::ThinkingLevel;

    fn app() -> App {
        App::new(
            &[],
            "test-model".to_owned(),
            None,
            AppContext {
                working_dir: PathBuf::from("."),
                thinking_level: None,
                thinking_preference: ThinkingLevel::Medium,
                context_chars: 0,
                max_context_chars: 120_000,
                default_tool_timeout: Duration::from_secs(60),
            },
        )
    }

    fn key(code: crossterm::event::KeyCode, modifiers: crossterm::event::KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn style_at(terminal: &Terminal<TestBackend>, x: u16, y: u16) -> ratatui::style::Style {
        terminal.backend().buffer()[(x, y)].style()
    }

    fn assert_regions_fill(area: ratatui::layout::Rect, regions: UiRegions) {
        assert_eq!(regions.transcript.y, area.y);
        assert_eq!(regions.completion.y, regions.transcript.bottom());
        assert_eq!(regions.status.y, regions.completion.bottom());
        assert_eq!(regions.keymap.y, regions.status.bottom());
        assert_eq!(regions.input.y, regions.keymap.bottom());
        assert_eq!(regions.input.bottom(), area.bottom());
    }

    #[test]
    fn empty_state_uses_native_background_and_full_width_footer_rails() {
        let mut app = app();
        app.working_dir = PathBuf::from("D:/workspaces/zex");
        app.git_status = Some(super::GitStatus {
            branch: "main".to_owned(),
            commit: "a1b2c3d".to_owned(),
        });
        app.thinking_level = Some(ThinkingLevel::High);
        app.context_chars = 30_000;
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let regions = ui_regions(ratatui::layout::Rect::new(0, 0, 100, 24), &app);

        assert!(screen.contains("Zex"));
        assert!(screen.contains("Ask anything, or type / for commands"));
        assert!(screen.contains("ZEX"));
        assert!(screen.contains("Ask Zex..."));
        assert!(screen.contains("Enter send"));
        assert!(screen.contains("think high"));
        assert!(screen.contains("main@a1b2c3d"));
        assert!(screen.contains("ctx 25%"));
        assert_ne!(style_at(&terminal, 0, 0).bg, Some(SURFACE));
        assert_ne!(style_at(&terminal, 0, 0).bg, Some(SURFACE_RAISED));
        assert_eq!(
            style_at(&terminal, 1, regions.status.y).fg,
            Some(super::status_rail_color(Status::Idle))
        );
        assert_eq!(
            style_at(&terminal, 98, regions.input.y).fg,
            Some(super::status_rail_color(Status::Idle))
        );
        assert_regions_fill(ratatui::layout::Rect::new(0, 0, 100, 24), regions);
    }

    #[test]
    fn empty_input_cursor_starts_on_the_placeholder_not_before_it() {
        let mut app = app();
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let input = ui_regions(ratatui::layout::Rect::new(0, 0, 80, 16), &app).input;

        terminal.backend_mut().assert_cursor_position((5, input.y));
        assert_eq!(
            terminal.backend().buffer()[(5, input.y)].symbol(),
            "A",
            "the cursor must overlay the first placeholder character"
        );
    }

    #[test]
    fn typed_input_replaces_the_placeholder_at_the_same_cursor_origin() {
        let mut app = app();
        app.input.insert_str("hello");
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let input = ui_regions(ratatui::layout::Rect::new(0, 0, 80, 16), &app).input;

        assert!(screen.contains("› hello"));
        assert!(!screen.contains("Ask Zex..."));
        terminal.backend_mut().assert_cursor_position((10, input.y));
    }

    #[test]
    fn layout_remains_edge_aligned_across_terminal_sizes() {
        for (width, height) in [(120, 32), (70, 18), (38, 12), (16, 6), (6, 3)] {
            let mut app = app();
            app.input
                .insert_str("first line\nsecond line that wraps on narrow terminals");
            app.refresh_completion();
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();

            terminal.draw(|frame| render(frame, &mut app)).unwrap();
            let area = ratatui::layout::Rect::new(0, 0, width, height);
            let regions = ui_regions(area, &app);
            assert_regions_fill(area, regions);
            assert!(regions.input.height >= 1);
            assert!(regions.input.height <= height);
            assert_ne!(style_at(&terminal, 0, 0).bg, Some(SURFACE));
            assert_ne!(style_at(&terminal, 0, 0).bg, Some(SURFACE_RAISED));
        }
    }

    #[test]
    fn tool_and_code_surfaces_do_not_color_the_transcript_background() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "Answer\n```rust\nfn main() {}\n```".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-surface".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"cargo check"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-surface".to_owned(),
            name: "bash".to_owned(),
            output: "Finished".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(24),
        });
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut saw_surface = false;
        let mut saw_native_background = false;
        for cell in buffer.content() {
            saw_surface |= cell.style().bg == Some(SURFACE);
            saw_native_background |= !matches!(cell.style().bg, Some(SURFACE | SURFACE_RAISED));
        }

        assert!(saw_surface);
        assert!(saw_native_background);
        assert!(format!("{}", terminal.backend()).contains("fn main() {}"));
        assert!(format!("{}", terminal.backend()).contains("$ cargo check"));
    }

    #[test]
    fn narrow_status_truncates_without_wrapping_or_losing_core_fields() {
        let mut app = app();
        app.model = "provider/very-long-model-name-that-cannot-fit".to_owned();
        app.working_dir =
            PathBuf::from("D:/very/long/workspace/path/that/should/not/wrap/across/the/footer");
        app.git_status = Some(super::GitStatus {
            branch: "feature/very-long-branch-name".to_owned(),
            commit: "deadbee".to_owned(),
        });
        app.thinking_level = Some(ThinkingLevel::Medium);
        app.context_chars = 60_000;
        let backend = TestBackend::new(38, 12);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let status_rows = screen
            .lines()
            .filter(|row| row.contains("ZEX") || row.contains("ready"))
            .collect::<Vec<_>>();

        assert_eq!(status_rows.len(), 1);
        assert!(status_rows[0].contains("ZEX"));
        assert!(status_rows[0].contains("ready"));
        assert!(status_rows[0].contains("50%"));
        assert!(!screen.contains("feature/very-long-branch-name"));
    }

    #[test]
    fn ready_thinking_and_running_states_are_clear_in_the_footer() {
        let mut app = app();
        let backend = TestBackend::new(100, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let ready = format!("{}", terminal.backend());
        assert!(ready.contains("ready"));
        assert!(ready.contains("Ask Zex..."));

        app.start_turn();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let thinking = format!("{}", terminal.backend());
        assert!(thinking.contains("thinking"));
        assert!(thinking.contains("Agent is thinking"));
        assert!(thinking.contains("Ctrl+C to stop"));

        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-running".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let running = format!("{}", terminal.backend());
        assert!(running.contains("working"));
        assert!(running.contains("Agent is working"));
        assert_eq!(running.matches("Agent is working").count(), 1);
    }

    #[test]
    fn multiline_input_grows_upward_and_keeps_the_footer_rails() {
        let mut app = app();
        let area = ratatui::layout::Rect::new(0, 0, 72, 18);
        let single = ui_regions(area, &app);
        app.input
            .insert_str("first line\nsecond line\nthird line\nfourth line");
        let multiline = ui_regions(area, &app);
        let backend = TestBackend::new(area.width, area.height);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();

        assert!(multiline.input.height > single.input.height);
        assert_eq!(multiline.input.bottom(), single.input.bottom());
        assert!(multiline.input.y < single.input.y);
        for y in multiline.input.y..multiline.input.bottom() {
            assert_eq!(
                style_at(&terminal, 1, y).fg,
                Some(super::status_rail_color(Status::Idle))
            );
            assert_eq!(
                style_at(&terminal, 70, y).fg,
                Some(super::status_rail_color(Status::Idle))
            );
        }
    }

    #[test]
    fn completion_panel_aligns_with_footer_and_highlights_selection() {
        let mut app = app();
        app.input.insert_str("/");
        app.refresh_completion();
        let backend = TestBackend::new(100, 28);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let regions = ui_regions(ratatui::layout::Rect::new(0, 0, 100, 28), &app);
        let panel_x = 1;
        let input_rail_x = 1;
        assert_eq!(
            style_at(&terminal, panel_x, regions.completion.y).fg,
            Some(ACCENT)
        );
        assert_eq!(
            style_at(&terminal, input_rail_x, regions.input.y).fg,
            Some(super::status_rail_color(Status::Idle))
        );
        let selected_row = regions.completion.y + 1;
        assert!((2..98).any(|x| style_at(&terminal, x, selected_row).bg == Some(ACCENT)));
        assert!(
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .any(|cell| cell.style().bg == Some(SURFACE_RAISED))
        );
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
            timeout: Duration::from_secs(60),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            output: "Cargo.toml".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(12),
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
            timeout: Duration::from_secs(60),
        });

        app.apply_agent_event(AgentEvent::TurnCancelled);

        assert_eq!(app.status, Status::Idle);
        assert!(!app.busy);
        assert!(app.active_tools.is_empty());
        let TranscriptEntry::Tool(tool) = &app.transcript[0] else {
            panic!("expected tool entry");
        };
        assert_eq!(tool.status, ToolStatus::Cancelled);
        assert!(
            app.toast
                .as_ref()
                .is_some_and(|toast| toast.message.contains("interrupted"))
        );
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
                .filter(|entry| matches!(entry, TranscriptEntry::Error { .. }))
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
            timeout: Duration::from_secs(60),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            output: "tool error: file not found".to_owned(),
            is_error: true,
            elapsed: Duration::from_millis(8),
        });

        assert_eq!(
            app.errors.back().map(String::as_str),
            Some("tool error: file not found")
        );
        assert!(
            !app.transcript
                .iter()
                .any(|entry| matches!(entry, TranscriptEntry::Error { .. }))
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
                false,
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
                false,
            ),
            InputAction::Submit("hello".to_owned())
        );
    }

    #[test]
    fn unbracketed_multiline_paste_keeps_newlines_without_submitting_first_line() {
        let mut app = app();
        let mut burst = KeyBurst::default();
        let started = std::time::Instant::now();
        let events = [
            crossterm::event::KeyCode::Char('f'),
            crossterm::event::KeyCode::Char('i'),
            crossterm::event::KeyCode::Char('r'),
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyCode::Char('t'),
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyCode::Char('s'),
            crossterm::event::KeyCode::Char('e'),
            crossterm::event::KeyCode::Char('c'),
            crossterm::event::KeyCode::Char('o'),
            crossterm::event::KeyCode::Char('n'),
            crossterm::event::KeyCode::Char('d'),
        ];

        for (index, code) in events.into_iter().enumerate() {
            let action = handle_terminal_event(
                crossterm::event::Event::Key(key(code, crossterm::event::KeyModifiers::NONE)),
                &mut app,
                false,
                &mut burst,
                started + Duration::from_millis(index as u64),
            );
            assert_eq!(action, InputAction::None);
        }

        assert_eq!(app.input.content, "first\nsecond");
    }

    #[test]
    fn deliberate_enter_after_paste_burst_submits_all_lines() {
        let mut app = app();
        let mut burst = KeyBurst::default();
        let started = std::time::Instant::now();
        for (index, code) in [
            crossterm::event::KeyCode::Char('a'),
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyCode::Char('b'),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                handle_terminal_event(
                    crossterm::event::Event::Key(key(code, crossterm::event::KeyModifiers::NONE,)),
                    &mut app,
                    false,
                    &mut burst,
                    started + Duration::from_millis(index as u64),
                ),
                InputAction::None
            );
        }

        assert_eq!(
            handle_terminal_event(
                crossterm::event::Event::Key(key(
                    crossterm::event::KeyCode::Enter,
                    crossterm::event::KeyModifiers::NONE,
                )),
                &mut app,
                false,
                &mut burst,
                started + Duration::from_millis(100),
            ),
            InputAction::Submit("a\nb".to_owned())
        );
    }

    #[test]
    fn tool_selection_and_expansion_are_explicit() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            arguments: "{}".to_owned(),
            timeout: Duration::from_secs(60),
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
    fn slash_completion_filters_selects_and_accepts_session_command() {
        let mut app = app();
        app.input.insert_str("/se");
        app.refresh_completion();

        let matches = app.completion_matches();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "/sessions");
        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Tab,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::None
        );
        assert_eq!(app.input.content, "/sessions");
        assert!(!app.completion_open());
    }

    #[test]
    fn exact_resume_completion_executes_instead_of_inserting_argument_space() {
        let mut app = app();
        app.input.insert_str("/resume");
        app.refresh_completion();

        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Enter,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::Submit("/resume".to_owned())
        );
        assert!(app.input.is_empty());
    }

    #[test]
    fn resume_picker_selects_with_arrows_and_returns_the_chosen_session() {
        let mut app = app();
        app.push_command_output(CommandOutput::ResumePicker(vec![
            crate::session::SessionSummary {
                id: "20260812-121500-cafebabe".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH + Duration::from_secs(60),
                message_count: 3,
                preview: "newer task".to_owned(),
            },
            crate::session::SessionSummary {
                id: "20260812-120000-deadbeef".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 2,
                preview: "older task".to_owned(),
            },
        ]));

        assert!(app.session_picker_open());
        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Down,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::None
        );
        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Enter,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::Resume("20260812-120000-deadbeef".to_owned())
        );
        assert!(!app.session_picker_open());
        assert_eq!(app.input.content, "");
    }

    #[test]
    fn resume_picker_enter_returns_the_first_recent_session_by_default() {
        let mut app = app();
        app.push_command_output(CommandOutput::ResumePicker(vec![
            crate::session::SessionSummary {
                id: "20260812-121500-cafebabe".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH + Duration::from_secs(60),
                message_count: 3,
                preview: "newer task".to_owned(),
            },
            crate::session::SessionSummary {
                id: "20260812-120000-deadbeef".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 2,
                preview: "older task".to_owned(),
            },
        ]));

        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Enter,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::Resume("20260812-121500-cafebabe".to_owned())
        );
    }

    #[test]
    fn resume_picker_escape_cancels_without_touching_the_transcript() {
        let mut app = app();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Keep this conversation visible.".to_owned(),
        });
        let transcript_before = app.transcript.clone();
        app.push_command_output(CommandOutput::ResumePicker(vec![
            crate::session::SessionSummary {
                id: "20260812-121500-cafebabe".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 1,
                preview: "saved task".to_owned(),
            },
        ]));

        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Esc,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::None
        );
        assert!(!app.session_picker_open());
        assert_eq!(app.transcript, transcript_before);
    }

    #[test]
    fn resume_picker_renders_short_id_time_preview_count_and_empty_state() {
        let mut app = app();
        app.push_command_output(CommandOutput::ResumePicker(vec![
            crate::session::SessionSummary {
                id: "20260812-121500-cafebabe".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 2,
                preview: "first saved task".to_owned(),
            },
        ]));
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let populated = format!("{}", terminal.backend());
        assert!(populated.contains("Resume session"));
        assert!(populated.contains("cafebabe"));
        assert!(!populated.contains("20260812-121500-cafebabe"));
        assert!(populated.contains("1970-01-01"));
        assert!(populated.contains("2 messages"));
        assert!(populated.contains("first saved task"));
        assert!(populated.contains("Enter resume"));

        app.push_command_output(CommandOutput::ResumePicker(Vec::new()));
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let empty = format!("{}", terminal.backend());
        assert!(empty.contains("No saved sessions"));
        assert!(empty.contains("Esc close"));
    }

    #[test]
    fn resumed_session_id_is_visible_in_the_footer() {
        let mut app = app();
        app.session_id = Some("20260812-121500-cafebabe".to_owned());
        let backend = TestBackend::new(120, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("session cafebabe"));
    }

    #[test]
    fn replacing_transcript_after_resume_loads_saved_messages_and_session_status() {
        let mut app = app();
        app.model = "saved-model".to_owned();
        app.session_id = Some("20260812-121500-cafebabe".to_owned());

        app.replace_transcript(&[
            Message::User {
                content: "saved context".to_owned(),
            },
            Message::Assistant {
                content: "saved answer".to_owned(),
                tool_calls: Vec::new(),
                provider_state: None,
            },
        ]);

        assert!(app.transcript.iter().any(|entry| matches!(
            entry,
            TranscriptEntry::Message {
                role: MessageRole::User,
                content,
            } if content == "saved context"
        )));
        assert!(app.transcript.iter().any(|entry| matches!(
            entry,
            TranscriptEntry::Message {
                role: MessageRole::Assistant,
                content,
            } if content == "saved answer"
        )));
        assert_eq!(app.session_id.as_deref(), Some("20260812-121500-cafebabe"));
        assert_eq!(app.model, "saved-model");
    }

    #[test]
    fn history_navigation_restores_the_unsent_draft() {
        let mut app = app();
        app.remember_submission("first task");
        app.remember_submission("second task");
        app.input.insert_str("draft");

        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Up,
                    crossterm::event::KeyModifiers::NONE,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::None
        );
        assert_eq!(app.input.content, "second task");
        handle_key_event(
            key(
                crossterm::event::KeyCode::Up,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );
        assert_eq!(app.input.content, "first task");
        handle_key_event(
            key(
                crossterm::event::KeyCode::Down,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );
        assert_eq!(app.input.content, "second task");
        handle_key_event(
            key(
                crossterm::event::KeyCode::Down,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );
        assert_eq!(app.input.content, "draft");
    }

    #[test]
    fn completion_arrows_do_not_browse_input_history() {
        let mut app = app();
        app.remember_submission("older task");
        app.input.insert_str("/");
        app.refresh_completion();

        handle_key_event(
            key(
                crossterm::event::KeyCode::Down,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );

        assert_eq!(app.input.content, "/");
        assert!(app.history_cursor.is_none());
        assert_eq!(app.completion.selected, 1);
    }

    #[test]
    fn transient_command_status_does_not_enter_the_feed() {
        let mut app = app();
        let before = app.transcript.len();

        assert!(!app.push_command_output(CommandOutput::Status("Thinking · high".to_owned())));
        assert!(!app.push_command_output(CommandOutput::Status(
            "Compacted context: freed approximately 4000 chars".to_owned(),
        )));

        assert_eq!(app.transcript.len(), before);
        assert_eq!(
            app.toast.as_ref().map(|toast| toast.message.as_str()),
            Some("Compacted context: freed approximately 4000 chars")
        );
    }

    #[test]
    fn auto_compaction_and_interruption_use_toasts_not_feed_rows() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ContextCompacted {
            stats: crate::agent::CompactStats {
                before_chars: 10_000,
                after_chars: 6_000,
                freed_chars: 4_000,
                kept_turns: 6,
                summarized_turns: 2,
                summarized_tool_outputs: 3,
            },
        });

        assert!(app.transcript.is_empty());
        assert!(
            app.toast
                .as_ref()
                .is_some_and(|toast| toast.message.contains("Context compacted"))
        );

        app.apply_agent_event(AgentEvent::TurnCancelled);
        assert!(app.transcript.is_empty());
        assert!(
            app.toast
                .as_ref()
                .is_some_and(|toast| toast.message == "Turn interrupted")
        );
    }

    #[test]
    fn errors_are_short_until_explicitly_expanded() {
        let mut app = app();
        app.record_error("provider failed\ncaused by: socket closed\nstack frame".to_owned());
        let backend = TestBackend::new(80, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let folded = format!("{}", terminal.backend());
        assert!(folded.contains("provider failed"));
        assert!(folded.contains("Ctrl+E details"));
        assert!(!folded.contains("socket closed"));

        app.toggle_latest_error();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let expanded = format!("{}", terminal.backend());
        assert!(expanded.contains("socket closed"));
        assert!(expanded.contains("Ctrl+E hide"));
    }

    #[test]
    fn mouse_scroll_moves_in_small_steps_and_returns_to_follow_mode() {
        let mut app = app();
        app.max_scroll = 40;
        app.scroll_top = 40;
        app.follow_output = true;

        app.scroll_lines_up(SCROLL_STEP);
        assert_eq!(app.scroll_top, 37);
        assert!(!app.follow_output);

        app.scroll_lines_down(SCROLL_STEP);
        assert_eq!(app.scroll_top, 40);
        assert!(app.follow_output);
    }

    #[test]
    fn long_feed_scrolls_without_moving_the_fixed_input() {
        let mut app = app();
        for index in 0..80 {
            app.transcript.push(TranscriptEntry::Message {
                role: if index % 2 == 0 {
                    MessageRole::User
                } else {
                    MessageRole::Assistant
                },
                content: format!("feed entry {index}\nsecond line"),
            });
        }
        app.input.insert_str("fixed draft");
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert!(app.max_scroll > 0);
        assert_eq!(app.scroll_top, app.max_scroll);
        let bottom = format!("{}", terminal.backend());
        assert!(bottom.contains("feed entry 79"));
        assert!(bottom.contains("fixed draft"));

        app.scroll_page_up();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let scrolled = format!("{}", terminal.backend());
        assert!(!app.follow_output);
        assert!(app.scroll_top < app.max_scroll);
        assert!(scrolled.contains("fixed draft"));
        assert!(scrolled.contains(" / "));
    }

    #[test]
    fn streamed_render_keeps_one_assistant_block_and_follows_output() {
        let mut app = app();
        let backend = TestBackend::new(90, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        app.start_turn();
        for delta in [
            "# Result\n",
            "- first\n",
            "- second\n",
            "```rust\n",
            "fn main() {}\n",
            "```",
        ] {
            app.apply_agent_event(AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: delta.to_owned(),
            });
            terminal.draw(|frame| render(frame, &mut app)).unwrap();
            assert!(app.follow_output);
            assert_eq!(
                app.transcript
                    .iter()
                    .filter(|entry| matches!(
                        entry,
                        TranscriptEntry::Message {
                            role: MessageRole::Assistant,
                            ..
                        }
                    ))
                    .count(),
                1
            );
        }

        let screen = format!("{}", terminal.backend());
        assert!(!screen.lines().any(|row| row.trim() == "you"));
        assert!(!screen.lines().any(|row| row.trim() == "assistant"));
        assert!(screen.contains("Result"));
        assert!(screen.contains("first"));
        assert!(screen.contains("fn main() {}"));
    }

    #[test]
    fn repeated_setting_toasts_replace_each_other_without_feed_noise() {
        let mut app = app();
        for level in ["low", "medium", "high", "off"] {
            assert!(!app.push_command_output(CommandOutput::Status(format!("Thinking · {level}"))));
        }

        assert!(app.transcript.is_empty());
        assert_eq!(
            app.toast.as_ref().map(|toast| toast.message.as_str()),
            Some("Thinking · off")
        );
    }

    #[test]
    fn think_shortcut_submits_shared_slash_command() {
        let mut app = app();

        assert_eq!(
            handle_key_event(
                key(
                    crossterm::event::KeyCode::Char('t'),
                    crossterm::event::KeyModifiers::CONTROL,
                ),
                &mut app,
                false,
                false,
            ),
            InputAction::Submit("/think".to_owned())
        );
    }

    #[test]
    fn ctrl_o_expands_selected_tool() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"git status"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });

        handle_key_event(
            key(
                crossterm::event::KeyCode::Char('o'),
                crossterm::event::KeyModifiers::CONTROL,
            ),
            &mut app,
            false,
            false,
        );

        let TranscriptEntry::Tool(tool) = &app.transcript[0] else {
            panic!("expected tool entry");
        };
        assert!(tool.expanded);
    }

    #[test]
    fn renders_multiple_shell_cards_and_completion_between_status_and_input() {
        let mut app = app();
        app.thinking_level = Some(ThinkingLevel::High);
        for (index, command) in ["git status", "git rev-parse --short HEAD"]
            .into_iter()
            .enumerate()
        {
            let call_id = format!("call-{index}");
            app.apply_agent_event(AgentEvent::ToolStart {
                call_id: call_id.clone(),
                name: "bash".to_owned(),
                arguments: format!(r#"{{"command":"{command}"}}"#),
                timeout: Duration::from_secs(30),
            });
            app.apply_agent_event(AgentEvent::ToolEnd {
                call_id,
                name: "bash".to_owned(),
                output: "ok".to_owned(),
                is_error: false,
                elapsed: Duration::from_millis(20 + index as u64),
            });
        }
        app.input.insert_str("/se");
        app.refresh_completion();
        let backend = TestBackend::new(120, 28);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("$ git status"));
        assert!(screen.contains("$ git rev-parse --short HEAD"));
        assert!(screen.contains("done  20ms"));
        assert!(!screen.contains("timeout 30.0s"));
        assert!(!screen.contains("Ctrl+O"));
        assert!(screen.contains("/sessions"));
        assert!(screen.contains("View saved sessions"));
        assert!(screen.contains("think high"));
        assert!(screen.contains("› /se"));

        let rows = screen.lines().collect::<Vec<_>>();
        let completion_row = rows
            .iter()
            .position(|row| row.contains("/sessions"))
            .expect("missing completion row");
        let status_row = rows
            .iter()
            .position(|row| row.contains("think high"))
            .expect("missing status row");
        let keymap_row = rows
            .iter()
            .position(|row| row.contains("↑↓ select"))
            .expect("missing keymap row");
        let input_row = rows
            .iter()
            .rposition(|row| row.contains("› /se"))
            .expect("missing input row");
        assert!(completion_row < status_row);
        assert!(status_row < keymap_row);
        assert!(keymap_row < input_row);
    }

    #[test]
    fn long_tool_details_are_truncated() {
        assert_eq!(truncate_chars("abcdef", 4), "abcd\n… truncated");
    }

    #[test]
    fn help_renders_one_registered_command_per_row_on_wide_terminals() {
        let mut app = App::new(
            &[],
            "gpt-test".to_owned(),
            None,
            AppContext {
                working_dir: PathBuf::from("."),
                thinking_level: None,
                thinking_preference: ThinkingLevel::Medium,
                context_chars: 0,
                max_context_chars: 120_000,
                default_tool_timeout: Duration::from_secs(60),
            },
        );
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Keep this conversation visible.".to_owned(),
        });
        let transcript_before = app.transcript.clone();
        app.push_command_output(CommandOutput::Help);
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        for command in command_specs() {
            assert!(
                screen.lines().any(|row| {
                    let command_column = row.find('/').unwrap_or(usize::MAX);
                    let description_column = row.find(command.description).unwrap_or(0);
                    row.contains(command.usage)
                        && description_column > command_column
                        && command_specs()
                            .iter()
                            .filter(|candidate| {
                                row.get(command_column..)
                                    .is_some_and(|content| content.starts_with(candidate.usage))
                            })
                            .max_by_key(|candidate| candidate.usage.len())
                            == Some(command)
                }),
                "missing row for {}\nscreen:\n{screen}",
                command.usage
            );
        }
        assert!(screen.contains("Esc close"));
        assert_eq!(app.transcript, transcript_before);
        assert!(app.help_open);

        app.cancel_ui_layer();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let closed = format!("{}", terminal.backend());
        assert!(closed.contains("Keep this conversation visible."));
        assert!(!closed.contains("Esc close"));
    }

    #[test]
    fn help_stays_compact_on_narrow_terminals() {
        let mut app = app();
        app.push_command_output(CommandOutput::Help);
        let backend = TestBackend::new(38, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let rows = screen.lines().collect::<Vec<_>>();

        for command in command_specs() {
            assert!(
                rows.iter().any(|row| row.contains(command.usage)),
                "missing {} in narrow help",
                command.usage
            );
        }
        assert!(!screen.contains("List slash commands"));
        assert!(
            !rows.iter().any(|row| {
                command_specs()
                    .iter()
                    .filter(|command| row.trim_start().starts_with(command.usage))
                    .count()
                    > 1
            }),
            "multiple commands merged onto one row"
        );
    }

    #[test]
    fn completion_uses_the_registered_usage_and_description() {
        let mut app = app();
        app.input.insert_str("/");
        app.refresh_completion();
        let backend = TestBackend::new(120, 56);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        for command in command_specs() {
            assert!(
                screen.contains(command.usage),
                "missing {}\nscreen:\n{screen}",
                command.usage
            );
            assert!(
                screen.contains(command.description),
                "missing description for {}",
                command.usage
            );
        }
    }

    #[test]
    fn session_records_and_multiline_feedback_keep_explicit_boundaries() {
        let mut app = app();
        app.push_command_output(CommandOutput::Sessions(vec![
            crate::session::SessionSummary {
                id: "20260812-120000-deadbeef".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 2,
                preview: "first task".to_owned(),
            },
            crate::session::SessionSummary {
                id: "20260812-121500-cafebabe".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 1,
                preview: "second task with a narrow layout".to_owned(),
            },
        ]));
        app.record_error("First error line\nSecond error line".to_owned());
        app.toggle_latest_error();
        let backend = TestBackend::new(38, 28);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let rows = screen.lines().collect::<Vec<_>>();

        for value in [
            "20260812-120000-deadbeef",
            "20260812-121500-cafebabe",
            "first task",
            "second task",
            "First error line",
            "Second error line",
        ] {
            assert!(screen.contains(value), "missing {value}");
        }
        assert!(
            !rows.iter().any(|row| {
                row.contains("20260812-120000-deadbeef") && row.contains("20260812-121500-cafebabe")
            }),
            "session records merged"
        );
        assert_eq!(screen.matches("Ctrl+E hide").count(), 1);
    }

    #[test]
    fn renders_status_conversation_and_folded_tool_regions() {
        let mut app = App::new(
            &[],
            "gpt-test".to_owned(),
            None,
            AppContext {
                working_dir: PathBuf::from("."),
                thinking_level: None,
                thinking_preference: ThinkingLevel::Medium,
                context_chars: 0,
                max_context_chars: 120_000,
                default_tool_timeout: Duration::from_secs(60),
            },
        );
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
            timeout: Duration::from_secs(60),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            output: "package zex".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(14),
        });
        app.input.insert_str("next prompt");
        let backend = TestBackend::new(120, 28);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("gpt-test"));
        assert!(!screen.contains("YOU"));
        assert!(!screen.contains("ASSISTANT"));
        assert!(!screen.lines().any(|row| row.trim() == "you"));
        assert!(screen.contains("read"));
        assert!(screen.contains("done"));
        assert!(screen.contains("package zex"));
        assert!(screen.contains("done  14ms"));
        assert!(!screen.contains("Ctrl+O expand"));
        assert!(!screen.contains("timeout 60.0s"));
        assert!(!screen.contains("\"path\": \"Cargo.toml\""));
        assert!(screen.contains("next prompt"));
        assert!(screen.contains("Enter send"));
    }

    #[test]
    fn git_status_tool_is_short_by_default_and_expands_in_place() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "Run git status".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-git".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"git status --short --branch","timeout_seconds":30}"#
                .to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-git".to_owned(),
            name: "bash".to_owned(),
            output:
                "exit_code: 0\nstdout:\n## main...origin/main [ahead 1]\n M src/tui.rs\n\nstderr:\n"
                    .to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(18),
        });
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "Branch `main` is ahead by one commit with one modified file.".to_owned(),
        });
        let backend = TestBackend::new(110, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let folded = format!("{}", terminal.backend());
        assert!(folded.contains("$ git status --short --branch"));
        assert!(folded.contains("done  18ms"));
        assert_eq!(folded.matches("## main...origin/main [ahead 1]").count(), 1);
        assert_eq!(
            folded
                .matches("Branch `main` is ahead by one commit with one modified file.")
                .count(),
            1
        );
        assert!(!folded.contains("exit_code: 0"));
        assert!(!folded.contains("timeout 30.0s"));
        assert!(!folded.contains("\"timeout_seconds\": 30"));

        app.toggle_selected_tool();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let expanded = format!("{}", terminal.backend());
        assert!(expanded.contains("input"));
        assert!(expanded.contains("output"));
        assert!(expanded.contains("exit_code: 0"));
        assert!(expanded.contains("M src/tui.rs"));
        assert!(expanded.contains("timeout 30.0s"));
        assert!(expanded.contains("Branch `main` is ahead by one commit"));
    }

    #[test]
    fn quiet_shell_command_uses_completed_summary_without_metadata() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-quiet".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"git status --porcelain"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-quiet".to_owned(),
            name: "bash".to_owned(),
            output: "exit_code: 0\nstdout:\n\nstderr:\n".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(9),
        });
        let backend = TestBackend::new(90, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("Completed"));
        assert!(!screen.contains("exit_code: 0"));
        assert!(!screen.contains("stdout:"));
        assert!(!screen.contains("stderr:"));
    }

    #[test]
    fn busy_state_lives_in_the_footer_without_feed_noise() {
        let mut app = app();
        app.start_turn();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-running".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"git status"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        let backend = TestBackend::new(100, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert_eq!(screen.matches("Agent is working").count(), 1);
        assert!(screen.contains("$ git status"));
        assert!(screen.contains("running"));
    }
}
