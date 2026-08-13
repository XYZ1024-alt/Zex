use std::{
    collections::{BTreeMap, VecDeque},
    io::{self, IsTerminal},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, EventStream, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent,
        MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures_util::StreamExt;
use ratatui::{
    DefaultTerminal, Frame, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{
    agent::{Agent, AgentEvent, Message, MessageRole, PromptOutcome},
    command::{
        CommandEffect, CommandOutput, CommandSpec, SlashCommand, command_specs, execute, parse,
    },
    config::{
        ModelChoice, ModelConfig, ModelRef, ProviderCatalog, ProviderConfig, SecretValue,
        persist_active_model, persist_provider_catalog, persist_show_thinking,
    },
    provider::{OpenAiApi, Provider, ProviderRegistry, ThinkingLevel},
    session::{SessionStore, SessionSummary},
};

const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const TOAST_DURATION: Duration = Duration::from_secs(4);
const MAX_ERRORS: usize = 3;
const MAX_ERROR_DETAIL_CHARS: usize = 4_000;
const MAX_THINKING_DETAIL_CHARS: usize = 16_000;
const MAX_TOOL_ARGUMENT_CHARS: usize = 2_000;
const TOOL_OUTPUT_PREVIEW_LINES: usize = 12;
const MAX_INPUT_HISTORY: usize = 100;
const MIN_TRANSCRIPT_HEIGHT: u16 = 2;
const HORIZONTAL_GUTTER: u16 = 2;
const MAX_INPUT_ROWS: usize = 5;
const INPUT_HORIZONTAL_PADDING: u16 = 3;
const INPUT_VERTICAL_PADDING: u16 = 1;
const SCROLL_STEP: usize = 3;
const PASTE_BURST_WINDOW: Duration = Duration::from_millis(12);
const MODEL_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(15);

const BACKGROUND: Color = Color::Rgb(20, 20, 20); // #141414
const SURFACE: Color = Color::Rgb(27, 27, 27);
const SURFACE_HOVER: Color = Color::Rgb(34, 34, 34);
const SURFACE_RAISED: Color = Color::Rgb(41, 41, 41);
const TEXT: Color = Color::Rgb(243, 243, 243); // #F3F3F3
const TEXT_STRONG: Color = TEXT;
const TEXT_DIM: Color = Color::Rgb(160, 160, 160); // #A0A0A0
const TEXT_FAINT: Color = Color::Rgb(120, 120, 120); // #787878
const ACCENT_PRIMARY: Color = Color::Rgb(122, 162, 247); // #7AA2F7
const ACCENT_SECONDARY: Color = Color::Rgb(187, 154, 247); // #BB9AF7
const OK: Color = Color::Rgb(158, 206, 106); // #9ECE6A
const BAD: Color = Color::Rgb(219, 75, 75); // #DB4B4B
const LANDING_LOGO_ROWS: [&str; 5] = [
    "█████ █████ ██ ██",
    "   ██ ██     ███ ",
    "  ██  ████    █  ",
    " ██   ██     ███ ",
    "█████ █████ ██ ██",
];
const LANDING_LOGO_DARK: Color = Color::Rgb(82, 82, 82);

pub fn is_available() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub async fn run(
    agent: &mut Agent<ProviderRegistry>,
    event_receiver: mpsc::UnboundedReceiver<AgentEvent>,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
    context: TuiContext<'_>,
) -> Result<()> {
    let mut terminal = TerminalSession::start()?;
    let result = run_loop(
        terminal.terminal_mut(),
        agent,
        event_receiver,
        EventStream::new(),
        session_store,
        session_id,
        RunContext {
            working_dir: context.working_dir,
            hide_thinking_block: context.hide_thinking_block,
            providers: context.providers,
            provider_registry: context.provider_registry,
        },
    )
    .await;
    let restore_result = terminal.restore();

    match (result, restore_result) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

pub struct TuiContext<'a> {
    pub working_dir: &'a Path,
    pub hide_thinking_block: bool,
    pub providers: ProviderCatalog,
    pub provider_registry: ProviderRegistry,
}

#[derive(Clone)]
struct RunContext<'a> {
    working_dir: &'a Path,
    hide_thinking_block: bool,
    providers: ProviderCatalog,
    provider_registry: ProviderRegistry,
}

async fn run_loop(
    terminal: &mut DefaultTerminal,
    agent: &mut Agent<ProviderRegistry>,
    mut event_receiver: mpsc::UnboundedReceiver<AgentEvent>,
    mut terminal_events: EventStream,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
    context: RunContext<'_>,
) -> Result<()> {
    let mut app = App::new(
        agent.messages(),
        agent.model().to_owned(),
        session_id.clone(),
        AppContext {
            working_dir: context.working_dir.to_path_buf(),
            thinking_level: Some(agent.thinking_level()),
            thinking_preference: agent.thinking_preference(),
            context_chars: agent.context_chars(),
            max_context_chars: agent.max_context_chars(),
            default_tool_timeout: agent.default_tool_timeout(),
            show_thinking: !context.hide_thinking_block,
            providers: context.providers.clone(),
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
                        InputAction::SwitchModel(target) => {
                            if let Err(error) =
                                switch_model(
                                    agent,
                                    &mut app,
                                    context.working_dir,
                                    session_id.as_deref(),
                                    target,
                                )
                                .await
                            {
                                app.record_error(format!("{error:#}"));
                            }
                        }
                        InputAction::SaveProviders(catalog) => {
                            if let Err(error) = save_provider_changes(
                                agent,
                                &mut app,
                                context.working_dir,
                                &context.provider_registry,
                                catalog,
                            )
                            .await
                            {
                                app.record_error(format!("{error:#}"));
                            }
                        }
                        InputAction::FetchProviderModels(provider) => {
                            match crate::provider::OpenAiProvider::list_models(
                                &provider.base_url,
                                provider.api_key.expose(),
                                MODEL_DISCOVERY_TIMEOUT,
                            )
                            .await
                            {
                                Ok(models) => app.merge_discovered_models(&provider.id, models),
                                Err(error) => app.record_error(format!("{error:#}")),
                            }
                        }
                        InputAction::Resume(requested_id) => {
                            match execute(
                                crate::command::SlashCommand::Resume(Some(requested_id)),
                                agent,
                                session_store,
                                session_id,
                                context.working_dir,
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
                                Ok(Some(SlashCommand::Model)) => {
                                    app.open_model_picker();
                                }
                                Ok(Some(SlashCommand::Provider)) => {
                                    app.open_provider_editor();
                                }
                                Ok(Some(SlashCommand::Thinking(requested))) => {
                                    if let Some(show_thinking) = requested {
                                        match persist_show_thinking(
                                            context.working_dir,
                                            show_thinking,
                                        )
                                        .await
                                        {
                                            Ok(()) => app.set_show_thinking(show_thinking),
                                            Err(error) => app.record_error(format!("{error:#}")),
                                        }
                                    } else {
                                        app.show_toast(
                                            format!(
                                                "Thinking cards · {}",
                                                if app.show_thinking { "shown" } else { "hidden" }
                                            ),
                                            ToastTone::Success,
                                        );
                                    }
                                }
                                Ok(Some(command)) => {
                                    match execute(
                                        command,
                                        agent,
                                        session_store,
                                        session_id,
                                        context.working_dir,
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

async fn switch_model<P>(
    agent: &mut Agent<P>,
    app: &mut App,
    working_dir: &Path,
    session_id: Option<&str>,
    target: ModelRef,
) -> Result<()>
where
    P: Provider,
{
    if !app.providers.contains(&target) {
        bail!("model {} is no longer configured", target.key());
    }
    persist_active_model(working_dir, &target).await?;
    app.providers.active_model = Some(target.clone());
    agent.set_model(target.key());
    app.sync_agent_status(agent, session_id);
    app.dismiss_model_picker();
    let label = app
        .providers
        .model(&target)
        .map(|(provider, model)| format!("{} / {}", provider.display_name, model.display_name))
        .unwrap_or_else(|| target.key());
    app.show_toast(format!("Model switched · {label}"), ToastTone::Success);
    Ok(())
}

fn checked_provider_catalog(
    agent: &Agent<ProviderRegistry>,
    editor: Option<&ProviderEditor>,
    mut catalog: ProviderCatalog,
) -> Result<(ProviderCatalog, Option<ModelRef>)> {
    let Some(active) = ModelRef::from_key(agent.model()) else {
        catalog.validate()?;
        return Ok((catalog, None));
    };
    if catalog.contains(&active) {
        catalog.validate()?;
        return Ok((catalog, None));
    }
    if let Some(editor) = editor
        && let Some(remapped) = remap_active_model(editor, &active)
        && editor.draft.contains(&remapped)
    {
        catalog.active_model = Some(remapped.clone());
        catalog.validate()?;
        return Ok((catalog, Some(remapped)));
    }
    bail!("switch away from the active model before removing it")
}

async fn save_provider_changes(
    agent: &mut Agent<ProviderRegistry>,
    app: &mut App,
    working_dir: &Path,
    provider_registry: &ProviderRegistry,
    catalog: ProviderCatalog,
) -> Result<()> {
    let (catalog, renamed_active) =
        checked_provider_catalog(agent, app.provider_editor.as_ref(), catalog)?;
    let registry_update = provider_registry.prepare_update(&catalog)?;
    persist_provider_catalog(working_dir, &catalog).await?;
    provider_registry.apply_update(registry_update)?;
    if let Some(active) = renamed_active {
        agent.set_model(active.key());
    }
    app.providers = catalog.clone();
    app.finish_provider_save(catalog);
    let session_id = app.session_id.clone();
    app.sync_agent_status(agent, session_id.as_deref());
    Ok(())
}

fn remap_active_model(editor: &ProviderEditor, active: &ModelRef) -> Option<ModelRef> {
    let provider_index = editor
        .original
        .providers
        .iter()
        .position(|provider| provider.id == active.provider_id)?;
    let model_index = editor.original.providers[provider_index]
        .models
        .iter()
        .position(|model| model.id == active.model_id)?;
    let provider = editor.draft.providers.get(provider_index)?;
    let model = provider.models.get(model_index)?;
    Some(ModelRef {
        provider_id: provider.id.clone(),
        model_id: model.id.clone(),
    })
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
                dirty |= app.busy;
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
                        InputAction::Quit
                        | InputAction::Resume(_)
                        | InputAction::SwitchModel(_)
                        | InputAction::SaveProviders(_)
                        | InputAction::FetchProviderModels(_)
                        | InputAction::Submit(_) => {}
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
    SwitchModel(ModelRef),
    SaveProviders(ProviderCatalog),
    FetchProviderModels(ProviderConfig),
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
            if app.provider_editor_is_editing() {
                for character in content.chars() {
                    app.provider_input_insert(character);
                }
            } else if !app.model_picker_open() && !app.provider_editor_open() {
                app.prepare_input_edit();
                app.input.insert_str(&content);
                app.refresh_completion();
            }
            InputAction::None
        }
        Event::Mouse(mouse) => handle_mouse_event(mouse, app, turn_active),
        Event::Key(key) if key.kind == crossterm::event::KeyEventKind::Press => {
            let in_paste_burst = burst.observe(key, now);
            handle_key_event(key, app, turn_active, in_paste_burst)
        }
        _ => InputAction::None,
    }
}

fn handle_mouse_event(mouse: MouseEvent, app: &mut App, turn_active: bool) -> InputAction {
    let target = app.hit_target_at(mouse.column, mouse.row);
    match mouse.kind {
        MouseEventKind::Moved | MouseEventKind::Drag(_) => {
            app.hovered = target;
            InputAction::None
        }
        MouseEventKind::ScrollUp if !app.page_open() => {
            app.scroll_lines_up(SCROLL_STEP);
            InputAction::None
        }
        MouseEventKind::ScrollDown if !app.page_open() => {
            app.scroll_lines_down(SCROLL_STEP);
            InputAction::None
        }
        MouseEventKind::Down(MouseButton::Left) => {
            app.hovered.clone_from(&target);
            let repeated = target
                .as_ref()
                .is_some_and(|target| app.armed_click.as_ref() == Some(target));
            app.armed_click = target
                .as_ref()
                .filter(|target| target.requires_confirmation())
                .cloned();

            match target {
                Some(HitTarget::Transcript) => {
                    app.input_focused = false;
                    InputAction::None
                }
                Some(HitTarget::Card(index)) => {
                    app.armed_click = None;
                    app.input_focused = false;
                    app.toggle_card(index);
                    InputAction::None
                }
                Some(HitTarget::ToolOutput(index)) => {
                    app.armed_click = None;
                    app.input_focused = false;
                    app.selected_card = Some(index);
                    app.toggle_selected_tool_output();
                    InputAction::None
                }
                Some(HitTarget::Completion(index)) => {
                    app.input_focused = false;
                    app.select_completion_at(index);
                    if repeated {
                        app.take_selected_completion()
                            .map(InputAction::Submit)
                            .unwrap_or(InputAction::None)
                    } else {
                        InputAction::None
                    }
                }
                Some(HitTarget::Session(index)) => {
                    app.input_focused = false;
                    app.select_session_at(index);
                    if repeated {
                        app.take_selected_session()
                            .map(InputAction::Resume)
                            .unwrap_or(InputAction::None)
                    } else {
                        InputAction::None
                    }
                }
                Some(HitTarget::Model(index)) => {
                    app.input_focused = false;
                    app.select_model_at(index);
                    if repeated {
                        app.take_selected_model()
                            .map(InputAction::SwitchModel)
                            .unwrap_or(InputAction::None)
                    } else {
                        InputAction::None
                    }
                }
                Some(HitTarget::Help(index)) => {
                    app.input_focused = false;
                    app.select_help_at(index);
                    if repeated {
                        app.selected_help_command()
                            .map(|command| InputAction::Submit(command.name.to_owned()))
                            .unwrap_or(InputAction::None)
                    } else {
                        InputAction::None
                    }
                }
                Some(HitTarget::Provider { pane, index }) => {
                    app.input_focused = false;
                    let selected = app.provider_item_selected(pane, index);
                    app.select_provider_at(pane, index);
                    if repeated && selected {
                        app.edit_provider_item();
                    }
                    InputAction::None
                }
                Some(HitTarget::Input) if !turn_active && !app.page_open() => {
                    app.armed_click = None;
                    app.focus_input();
                    InputAction::None
                }
                Some(HitTarget::StatusModel) if !turn_active => {
                    app.armed_click = None;
                    app.input_focused = false;
                    app.open_model_picker();
                    InputAction::None
                }
                Some(HitTarget::StatusThinking) if !turn_active => {
                    app.armed_click = None;
                    app.input_focused = false;
                    next_thinking_action(app)
                }
                _ => {
                    app.armed_click = None;
                    app.input_focused = false;
                    InputAction::None
                }
            }
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
    app.armed_click = None;

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return if turn_active {
            InputAction::Interrupt
        } else {
            InputAction::Quit
        };
    }

    if app.model_picker_open() && !turn_active {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => app.select_model(true),
            KeyCode::Down | KeyCode::Char('j') => app.select_model(false),
            KeyCode::Home | KeyCode::Char('g') => app.select_first_model(),
            KeyCode::End | KeyCode::Char('G') => app.select_last_model(),
            KeyCode::Enter | KeyCode::Char(' ') => {
                if let Some(target) = app.take_selected_model() {
                    return InputAction::SwitchModel(target);
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => app.dismiss_model_picker(),
            _ => {}
        }
        return InputAction::None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL)
        && key.code == KeyCode::Char('s')
        && app.provider_editor_open()
        && !turn_active
    {
        return app
            .provider_catalog_to_save()
            .map(InputAction::SaveProviders)
            .unwrap_or(InputAction::None);
    }

    if app.provider_editor_open() && !turn_active {
        if app.provider_editor_is_confirming() {
            match key.code {
                KeyCode::Enter | KeyCode::Char('y') => app.confirm_provider_action(),
                KeyCode::Esc | KeyCode::Char('n') => app.cancel_provider_action(),
                _ => {}
            }
            return InputAction::None;
        }
        if app.provider_editor_is_editing() {
            match key.code {
                KeyCode::Enter => app.commit_provider_field(),
                KeyCode::Esc => app.cancel_provider_field(),
                KeyCode::Backspace => app.provider_input_backspace(),
                KeyCode::Delete => app.provider_input_delete(),
                KeyCode::Left => app.provider_input_left(),
                KeyCode::Right => app.provider_input_right(),
                KeyCode::Char(character)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    app.provider_input_insert(character);
                }
                _ => {}
            }
            return InputAction::None;
        }
        match key.code {
            KeyCode::Tab | KeyCode::Char('l') => app.next_provider_pane(false),
            KeyCode::BackTab | KeyCode::Char('h') => app.next_provider_pane(true),
            KeyCode::Up | KeyCode::Char('k') => app.select_provider_item(true),
            KeyCode::Down | KeyCode::Char('j') => app.select_provider_item(false),
            KeyCode::Enter | KeyCode::Char('e') => app.edit_provider_item(),
            KeyCode::Char('i') => app.edit_selected_model_id(),
            KeyCode::Char('t') => app.cycle_selected_model_thinking_level(),
            KeyCode::Char('m') => app.cycle_selected_model_thinking_min_level(),
            KeyCode::Char('r') => app.cycle_selected_model_reasoning_map(),
            KeyCode::Char('f') => {
                if let Some(provider) = app.selected_provider_to_fetch() {
                    return InputAction::FetchProviderModels(provider);
                }
            }
            KeyCode::Char('n') => app.new_provider_item(),
            KeyCode::Char('d') => app.request_provider_delete(),
            KeyCode::Esc | KeyCode::Char('q') => app.request_provider_exit(),
            KeyCode::Char(' ') => app.toggle_provider_value(),
            _ => {}
        }
        return InputAction::None;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
        let can_toggle_output = app
            .selected_card
            .and_then(|index| app.transcript.get(index))
            .is_some_and(|entry| {
                matches!(
                    entry,
                    TranscriptEntry::Tool(tool)
                        if tool.expanded
                            && tool_detail_line_count(tool) > TOOL_OUTPUT_PREVIEW_LINES
                )
            });
        if can_toggle_output {
            app.toggle_selected_tool_output();
        } else {
            app.toggle_all_cards();
        }
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
            next_thinking_action(app)
        };
    }

    if app.session_picker_open() && !turn_active {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => app.select_session(true),
            KeyCode::Down | KeyCode::Char('j') => app.select_session(false),
            KeyCode::Enter | KeyCode::Char(' ') => {
                if let Some(session_id) = app.take_selected_session() {
                    return InputAction::Resume(session_id);
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => app.dismiss_session_picker(),
            _ => {}
        }
        return InputAction::None;
    }

    if app.help_open && !turn_active {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => app.select_help(true),
            KeyCode::Down | KeyCode::Char('j') => app.select_help(false),
            KeyCode::Enter | KeyCode::Char(' ') => {
                if let Some(command) = app.selected_help_command() {
                    return InputAction::Submit(command.name.to_owned());
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => app.dismiss_help(),
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
                return app
                    .take_selected_completion()
                    .map(InputAction::Submit)
                    .unwrap_or(InputAction::None);
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
        KeyCode::Char(' ') if app.selected_card.is_some() && app.input.is_empty() => {
            app.toggle_selected_tool();
            return InputAction::None;
        }
        KeyCode::Esc => {
            return if app.cancel_ui_layer() {
                InputAction::None
            } else if turn_active {
                InputAction::Interrupt
            } else {
                InputAction::Quit
            };
        }
        _ => {}
    }

    if turn_active {
        return InputAction::None;
    }

    if !app.input_focused && app.selected_card.is_some() && key.code == KeyCode::Enter {
        app.toggle_selected_tool();
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

fn next_thinking_action(app: &App) -> InputAction {
    let levels = app
        .providers
        .active_model
        .as_ref()
        .map(|model| {
            app.providers
                .thinking_capabilities(model)
                .available_levels()
        })
        .unwrap_or_else(|| crate::provider::ThinkingCapabilities::default().available_levels());
    let current = levels
        .iter()
        .position(|level| *level == app.thinking_preference)
        .unwrap_or(0);
    InputAction::Submit(format!("/think {}", levels[(current + 1) % levels.len()]))
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
    fn color(self) -> Color {
        match self {
            Self::Idle => OK,
            Self::Thinking | Self::RunningTool | Self::Cancelling => ACCENT_PRIMARY,
            Self::Error => BAD,
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
            Self::Done => "ok",
            Self::Failed => "failed",
            Self::Cancelled => "stopped",
        }
    }

    fn color(self) -> Color {
        match self {
            Self::Running | Self::Cancelled => ACCENT_PRIMARY,
            Self::Done => OK,
            Self::Failed => BAD,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkingStatus {
    Active,
    Done,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HitTarget {
    Transcript,
    Card(usize),
    ToolOutput(usize),
    Completion(usize),
    Session(usize),
    Model(usize),
    Help(usize),
    Provider { pane: ProviderPane, index: usize },
    Input,
    StatusModel,
    StatusThinking,
}

impl HitTarget {
    fn requires_confirmation(&self) -> bool {
        matches!(
            self,
            Self::Completion(_)
                | Self::Session(_)
                | Self::Model(_)
                | Self::Help(_)
                | Self::Provider { .. }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HitRegion {
    area: Rect,
    target: HitTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolEntry {
    call_id: String,
    name: String,
    arguments: String,
    output: String,
    status: ToolStatus,
    expanded: bool,
    show_full_output: bool,
    started_at: Option<Instant>,
    elapsed: Option<Duration>,
    timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ThinkingEntry {
    content: String,
    expanded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOutcome {
    Done,
    Failed,
    Stopped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TurnEntry {
    outcome: TurnOutcome,
    model: String,
    thinking: ThinkingLevel,
    tool_count: usize,
    elapsed: Option<Duration>,
    output_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TranscriptEntry {
    Message {
        role: MessageRole,
        content: String,
    },
    Thinking(ThinkingEntry),
    Tool(ToolEntry),
    Error {
        summary: String,
        detail: String,
        expanded: bool,
    },
    Turn(TurnEntry),
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
            ToastTone::Neutral => ACCENT_PRIMARY,
            ToastTone::Success => OK,
        }
    }
}

#[derive(Debug, Default, Clone)]
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
    dirty_count: usize,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelPicker {
    choices: Vec<ModelChoice>,
    selected: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderPane {
    Providers,
    Details,
    Models,
}

impl ProviderPane {
    fn next(self, reverse: bool) -> Self {
        match (self, reverse) {
            (Self::Providers, false) | (Self::Details, true) => Self::Details,
            (Self::Details, false) | (Self::Models, true) => Self::Models,
            (Self::Models, false) | (Self::Providers, true) => Self::Providers,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderField {
    Id,
    DisplayName,
    BaseUrl,
    ApiKey,
}

impl ProviderField {
    const COUNT: usize = 5;

    fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Id),
            1 => Some(Self::DisplayName),
            2 => Some(Self::BaseUrl),
            3 => Some(Self::ApiKey),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderEditTarget {
    Provider(ProviderField),
    ModelId { model_index: usize },
    ModelName { model_index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeleteTarget {
    Provider(usize),
    Model {
        provider_index: usize,
        model_index: usize,
    },
}

#[derive(Debug, Clone)]
struct FieldEditor {
    target: ProviderEditTarget,
    input: InputBuffer,
}

#[derive(Debug, Clone)]
enum ProviderDialog {
    Delete(DeleteTarget),
    Discard,
}

#[derive(Debug, Clone)]
struct ProviderEditor {
    original: ProviderCatalog,
    draft: ProviderCatalog,
    pane: ProviderPane,
    provider_selected: usize,
    detail_selected: usize,
    model_selected: usize,
    field_editor: Option<FieldEditor>,
    dialog: Option<ProviderDialog>,
}

impl ProviderEditor {
    fn new(catalog: ProviderCatalog) -> Self {
        Self {
            original: catalog.clone(),
            draft: catalog,
            pane: ProviderPane::Providers,
            provider_selected: 0,
            detail_selected: 0,
            model_selected: 0,
            field_editor: None,
            dialog: None,
        }
    }

    fn dirty(&self) -> bool {
        self.original != self.draft
    }

    fn selected_provider(&self) -> Option<&ProviderConfig> {
        self.draft.providers.get(self.provider_selected)
    }

    fn selected_provider_mut(&mut self) -> Option<&mut ProviderConfig> {
        self.draft.providers.get_mut(self.provider_selected)
    }
}

#[derive(Debug, Clone)]
struct AppContext {
    working_dir: PathBuf,
    thinking_level: Option<ThinkingLevel>,
    thinking_preference: ThinkingLevel,
    context_chars: usize,
    max_context_chars: usize,
    default_tool_timeout: Duration,
    show_thinking: bool,
    providers: ProviderCatalog,
}

#[derive(Debug)]
struct App {
    model: String,
    session_id: Option<String>,
    transcript: Vec<TranscriptEntry>,
    input: InputBuffer,
    active_tools: BTreeMap<String, String>,
    errors: VecDeque<String>,
    selected_card: Option<usize>,
    hovered: Option<HitTarget>,
    armed_click: Option<HitTarget>,
    hit_regions: Vec<HitRegion>,
    input_focused: bool,
    help_selected: usize,
    status: Status,
    busy: bool,
    turn_started: Option<Instant>,
    turn_tool_count: usize,
    turn_output_tokens: u64,
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
    tokens_per_second: Option<f64>,
    default_tool_timeout: Duration,
    show_thinking: bool,
    completion: CompletionState,
    input_history: VecDeque<String>,
    history_cursor: Option<usize>,
    history_draft: String,
    toast: Option<Toast>,
    help_open: bool,
    session_picker: Option<SessionPicker>,
    providers: ProviderCatalog,
    model_picker: Option<ModelPicker>,
    provider_editor: Option<ProviderEditor>,
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
            selected_card: None,
            hovered: None,
            armed_click: None,
            hit_regions: Vec::new(),
            input_focused: true,
            help_selected: 0,
            status: Status::Idle,
            busy: false,
            turn_started: None,
            turn_tool_count: 0,
            turn_output_tokens: 0,
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
            tokens_per_second: None,
            default_tool_timeout: context.default_tool_timeout,
            show_thinking: context.show_thinking,
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
            providers: context.providers,
            model_picker: None,
            provider_editor: None,
        };

        let mut restored_turn_open = false;
        let mut restored_tool_count = 0;
        for message in messages {
            match message {
                Message::System { .. } => {}
                Message::User { content } => {
                    restored_turn_open = true;
                    restored_tool_count = 0;
                    app.transcript.push(TranscriptEntry::Message {
                        role: MessageRole::User,
                        content: content.clone(),
                    });
                }
                Message::Assistant {
                    content,
                    thinking,
                    tool_calls,
                    ..
                } => {
                    if let Some(thinking) =
                        thinking.as_deref().filter(|thinking| !thinking.is_empty())
                    {
                        app.transcript
                            .push(TranscriptEntry::Thinking(ThinkingEntry {
                                content: truncate_chars(thinking, MAX_THINKING_DETAIL_CHARS),
                                expanded: false,
                            }));
                    }
                    let completes_turn = restored_turn_open && tool_calls.is_empty();
                    if completes_turn {
                        app.transcript.push(TranscriptEntry::Turn(TurnEntry {
                            outcome: TurnOutcome::Done,
                            model: app.model.clone(),
                            thinking: app.thinking_level.unwrap_or(app.thinking_preference),
                            tool_count: restored_tool_count,
                            elapsed: None,
                            output_tokens: None,
                        }));
                    }
                    if !content.is_empty() {
                        app.transcript.push(TranscriptEntry::Message {
                            role: MessageRole::Assistant,
                            content: content.clone(),
                        });
                    }
                    restored_tool_count = restored_tool_count.saturating_add(tool_calls.len());
                    app.transcript.extend(tool_calls.iter().map(|call| {
                        TranscriptEntry::Tool(ToolEntry {
                            call_id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: truncate_chars(
                                &sanitize_terminal_text(&format_json(&call.arguments)),
                                MAX_TOOL_ARGUMENT_CHARS,
                            ),
                            output: String::new(),
                            status: ToolStatus::Done,
                            expanded: false,
                            show_full_output: false,
                            started_at: None,
                            elapsed: None,
                            timeout: app.default_tool_timeout,
                        })
                    }));
                    if completes_turn {
                        restored_turn_open = false;
                    }
                }
                Message::Tool {
                    tool_call_id,
                    content,
                } => {
                    if let Some(tool) = app.find_tool_mut(tool_call_id) {
                        tool.output = sanitize_terminal_text(content);
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

    fn begin_render(&mut self) {
        self.hit_regions.clear();
    }

    fn register_hit(&mut self, area: Rect, target: HitTarget) {
        if !area.is_empty() {
            self.hit_regions.push(HitRegion { area, target });
        }
    }

    fn hit_target_at(&self, x: u16, y: u16) -> Option<HitTarget> {
        self.hit_regions
            .iter()
            .rev()
            .find(|region| region.area.contains((x, y).into()))
            .map(|region| region.target.clone())
    }

    fn hovered(&self, target: &HitTarget) -> bool {
        self.hovered.as_ref() == Some(target)
    }

    fn armed(&self, target: &HitTarget) -> bool {
        self.armed_click.as_ref() == Some(target)
    }

    fn start_turn(&mut self) {
        self.help_open = false;
        self.input_focused = false;
        self.busy = true;
        self.turn_started = Some(Instant::now());
        self.turn_tool_count = 0;
        self.turn_output_tokens = 0;
        self.status = Status::Thinking;
        self.scroll_to_bottom();
    }

    fn landing_visible(&self) -> bool {
        self.transcript.is_empty() && !self.busy && self.status == Status::Idle
    }

    fn finish_turn(&mut self, status: Status) {
        let outcome = if status == Status::Error {
            TurnOutcome::Failed
        } else {
            TurnOutcome::Done
        };
        self.finish_turn_with_outcome(status, outcome);
    }

    fn finish_turn_with_outcome(&mut self, status: Status, outcome: TurnOutcome) {
        let elapsed = self.turn_started.take().map(|started| started.elapsed());
        if elapsed.is_some() {
            let turn = TranscriptEntry::Turn(TurnEntry {
                outcome,
                model: self.model.clone(),
                thinking: self.thinking_level.unwrap_or(self.thinking_preference),
                tool_count: self.turn_tool_count,
                elapsed,
                output_tokens: (self.turn_output_tokens > 0).then_some(self.turn_output_tokens),
            });
            let insert_before_answer = self.transcript.last().is_some_and(|entry| {
                matches!(
                    entry,
                    TranscriptEntry::Message {
                        role: MessageRole::Assistant,
                        ..
                    }
                )
            });
            let insert_at = if insert_before_answer {
                self.transcript.len().saturating_sub(1)
            } else {
                self.transcript.len()
            };
            self.transcript.insert(insert_at, turn);
        }
        self.busy = false;
        self.turn_tool_count = 0;
        self.turn_output_tokens = 0;
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
            AgentEvent::ThinkingDelta { delta } => {
                self.append_thinking(delta);
                if self.status != Status::Cancelling {
                    self.status = Status::Thinking;
                }
            }
            AgentEvent::ThinkingNormalized {
                requested,
                effective,
                ..
            } => {
                self.thinking_preference = requested;
                self.thinking_level = Some(effective);
            }
            AgentEvent::ToolStart {
                call_id,
                name,
                arguments,
                timeout,
            } => {
                self.active_tools.insert(call_id.clone(), name.clone());
                self.turn_tool_count = self.turn_tool_count.saturating_add(1);
                self.status = Status::RunningTool;
                self.transcript.push(TranscriptEntry::Tool(ToolEntry {
                    call_id,
                    name,
                    arguments: truncate_chars(
                        &sanitize_terminal_text(&format_json(&arguments)),
                        MAX_TOOL_ARGUMENT_CHARS,
                    ),
                    output: String::new(),
                    status: ToolStatus::Running,
                    expanded: false,
                    show_full_output: false,
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
                let output = sanitize_terminal_text(&output);
                let status = if is_error {
                    ToolStatus::Failed
                } else {
                    ToolStatus::Done
                };
                if is_error {
                    self.remember_error(&output);
                }
                if let Some(tool) = self.find_tool_mut(&call_id) {
                    tool.name = name;
                    tool.output = output;
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
            AgentEvent::ProviderUsage {
                output_tokens,
                elapsed,
            } => {
                self.turn_output_tokens = self.turn_output_tokens.saturating_add(output_tokens);
                self.tokens_per_second = (elapsed > Duration::ZERO)
                    .then(|| output_tokens as f64 / elapsed.as_secs_f64())
                    .filter(|rate| rate.is_finite());
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
                self.finish_turn_with_outcome(Status::Idle, TurnOutcome::Stopped);
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

    fn append_thinking(&mut self, delta: String) {
        if let Some(TranscriptEntry::Thinking(thinking)) = self.transcript.last_mut() {
            thinking.content.push_str(&delta);
            if thinking.content.chars().count() > MAX_THINKING_DETAIL_CHARS {
                thinking.content = truncate_chars(&thinking.content, MAX_THINKING_DETAIL_CHARS);
            }
            return;
        }

        self.transcript
            .push(TranscriptEntry::Thinking(ThinkingEntry {
                content: truncate_chars(&delta, MAX_THINKING_DETAIL_CHARS),
                expanded: false,
            }));
    }

    fn reset_transcript(&mut self) {
        self.transcript.clear();
        self.active_tools.clear();
        self.errors.clear();
        self.selected_card = None;
        self.hovered = None;
        self.armed_click = None;
        self.input_focused = true;
        self.help_selected = 0;
        self.status = Status::Idle;
        self.busy = false;
        self.turn_started = None;
        self.turn_tool_count = 0;
        self.turn_output_tokens = 0;
        self.toast = None;
        self.help_open = false;
        self.session_picker = None;
        self.model_picker = None;
        self.provider_editor = None;
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
                show_thinking: self.show_thinking,
                providers: self.providers.clone(),
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
        let message = sanitize_terminal_text(&message);
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

    fn set_show_thinking(&mut self, show_thinking: bool) {
        self.show_thinking = show_thinking;
        self.selected_card = None;
        self.show_toast(
            format!(
                "Thinking cards · {}",
                if show_thinking { "shown" } else { "hidden" }
            ),
            ToastTone::Success,
        );
        self.scroll_to_bottom();
    }

    fn record_error_if_new(&mut self, message: String) {
        self.record_error(message);
    }

    fn push_command_output(&mut self, output: CommandOutput) -> bool {
        match output {
            CommandOutput::Help => {
                self.help_open = true;
                self.help_selected = 0;
                self.input_focused = false;
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

    fn card_indices(&self) -> Vec<usize> {
        self.transcript
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| match entry {
                TranscriptEntry::Thinking(_) if self.show_thinking => Some(index),
                TranscriptEntry::Tool(_) => Some(index),
                _ => None,
            })
            .collect()
    }

    fn select_tool(&mut self, reverse: bool) {
        let indices = self.card_indices();
        if indices.is_empty() {
            self.selected_card = None;
            return;
        }

        let index = self
            .selected_card
            .and_then(|selected| indices.iter().position(|index| *index == selected));
        let next = match (index, reverse) {
            (Some(0), true) | (None, true) => indices.len() - 1,
            (Some(index), true) => index - 1,
            (Some(index), false) => (index + 1) % indices.len(),
            (None, false) => 0,
        };
        self.selected_card = Some(indices[next]);
        self.input_focused = false;
    }

    fn toggle_selected_tool(&mut self) {
        if self.selected_card.is_none() {
            self.selected_card = self.card_indices().last().copied();
        }
        if let Some(selected) = self.selected_card {
            self.toggle_card(selected);
        }
    }

    fn toggle_card(&mut self, index: usize) {
        self.selected_card = Some(index);
        self.input_focused = false;
        match self.transcript.get_mut(index) {
            Some(TranscriptEntry::Thinking(thinking)) => {
                thinking.expanded = !thinking.expanded;
            }
            Some(TranscriptEntry::Tool(tool)) => {
                tool.expanded = !tool.expanded;
                if !tool.expanded {
                    tool.show_full_output = false;
                }
            }
            _ => {}
        }
    }

    fn toggle_all_cards(&mut self) {
        let expand = self.card_indices().into_iter().any(|index| {
            matches!(
                self.transcript.get(index),
                Some(TranscriptEntry::Thinking(ThinkingEntry {
                    expanded: false,
                    ..
                })) | Some(TranscriptEntry::Tool(ToolEntry {
                    expanded: false,
                    ..
                }))
            )
        });
        for entry in &mut self.transcript {
            match entry {
                TranscriptEntry::Thinking(thinking) if self.show_thinking => {
                    thinking.expanded = expand;
                }
                TranscriptEntry::Tool(tool) => {
                    tool.expanded = expand;
                    if !expand {
                        tool.show_full_output = false;
                    }
                }
                _ => {}
            }
        }
    }

    fn toggle_selected_tool_output(&mut self) {
        let Some(selected) = self.selected_card else {
            return;
        };
        let Some(TranscriptEntry::Tool(tool)) = self.transcript.get_mut(selected) else {
            return;
        };
        if tool.expanded && tool_detail_line_count(tool) > TOOL_OUTPUT_PREVIEW_LINES {
            tool.show_full_output = !tool.show_full_output;
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
        if self.provider_editor_open() {
            self.request_provider_exit();
            return true;
        }
        if self.model_picker_open() {
            self.dismiss_model_picker();
            return true;
        }
        if self.session_picker_open() {
            self.dismiss_session_picker();
            return true;
        }
        if self.completion_open() {
            self.dismiss_completion();
            return true;
        }
        if self.help_open {
            self.dismiss_help();
            return true;
        }
        if let Some(selected) = self.selected_card {
            match self.transcript.get_mut(selected) {
                Some(TranscriptEntry::Thinking(thinking)) if thinking.expanded => {
                    thinking.expanded = false;
                    return true;
                }
                Some(TranscriptEntry::Tool(tool)) if tool.expanded => {
                    tool.expanded = false;
                    tool.show_full_output = false;
                    return true;
                }
                _ => {}
            }
            self.selected_card = None;
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
        let model = agent.model();
        if self.model != model {
            self.tokens_per_second = None;
        }
        self.model = model.to_owned();
        self.session_id = session_id.map(str::to_owned);
        self.thinking_level = Some(agent.thinking_level());
        self.thinking_preference = agent.thinking_preference();
        self.context_chars = agent.context_chars();
        self.max_context_chars = agent.max_context_chars();
    }

    fn open_session_picker(&mut self, sessions: Vec<SessionSummary>) {
        self.help_open = false;
        self.input_focused = false;
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

    fn page_open(&self) -> bool {
        self.help_open
            || self.session_picker_open()
            || self.model_picker_open()
            || self.provider_editor_open()
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

    fn select_session_at(&mut self, index: usize) {
        if let Some(picker) = &mut self.session_picker
            && index < picker.sessions.len()
        {
            picker.selected = index;
        }
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

    fn select_help(&mut self, reverse: bool) {
        let count = command_specs().len();
        if count == 0 {
            return;
        }
        self.help_selected = if reverse {
            self.help_selected.checked_sub(1).unwrap_or(count - 1)
        } else {
            (self.help_selected + 1) % count
        };
    }

    fn select_help_at(&mut self, index: usize) {
        if index < command_specs().len() {
            self.help_selected = index;
        }
    }

    fn selected_help_command(&self) -> Option<&'static CommandSpec> {
        command_specs().get(self.help_selected)
    }

    fn dismiss_help(&mut self) {
        self.help_open = false;
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

    fn select_completion_at(&mut self, index: usize) {
        if index < self.completion_matches().len() {
            self.completion.selected = index;
        }
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

    fn take_selected_completion(&mut self) -> Option<String> {
        let matches = self.completion_matches();
        let command = matches.get(self.completion.selected)?;
        let command = command.name.to_owned();
        self.input.clear();
        self.reset_history_navigation();
        self.completion.dismissed = true;
        Some(command)
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
        self.dismiss_help();
        self.input_focused = true;
        self.selected_card = None;
        self.reset_history_navigation();
    }

    fn focus_input(&mut self) {
        if self.provider_editor_open() {
            self.request_provider_exit();
            if self.provider_editor_open() {
                self.input_focused = false;
                return;
            }
        }
        self.dismiss_model_picker();
        self.dismiss_session_picker();
        self.dismiss_help();
        self.input_focused = true;
        self.selected_card = None;
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

    fn open_model_picker(&mut self) {
        self.help_open = false;
        self.input_focused = false;
        self.session_picker = None;
        self.provider_editor = None;
        self.input.clear();
        self.reset_history_navigation();
        let choices = self.providers.choices();
        let selected = self
            .providers
            .active_model
            .as_ref()
            .and_then(|active| choices.iter().position(|choice| choice.target == *active))
            .unwrap_or(0);
        self.model_picker = Some(ModelPicker { choices, selected });
    }

    fn model_picker_open(&self) -> bool {
        self.model_picker.is_some()
    }

    fn dismiss_model_picker(&mut self) {
        self.model_picker = None;
    }

    fn select_model(&mut self, reverse: bool) {
        let Some(picker) = &mut self.model_picker else {
            return;
        };
        if picker.choices.is_empty() {
            return;
        }
        picker.selected = if reverse {
            picker
                .selected
                .checked_sub(1)
                .unwrap_or(picker.choices.len() - 1)
        } else {
            (picker.selected + 1) % picker.choices.len()
        };
    }

    fn select_model_at(&mut self, index: usize) {
        if let Some(picker) = &mut self.model_picker
            && index < picker.choices.len()
        {
            picker.selected = index;
        }
    }

    fn select_first_model(&mut self) {
        if let Some(picker) = &mut self.model_picker {
            picker.selected = 0;
        }
    }

    fn select_last_model(&mut self) {
        if let Some(picker) = &mut self.model_picker
            && !picker.choices.is_empty()
        {
            picker.selected = picker.choices.len() - 1;
        }
    }

    fn take_selected_model(&self) -> Option<ModelRef> {
        let picker = self.model_picker.as_ref()?;
        picker
            .choices
            .get(picker.selected)
            .map(|choice| choice.target.clone())
    }

    fn open_provider_editor(&mut self) {
        self.help_open = false;
        self.input_focused = false;
        self.session_picker = None;
        self.model_picker = None;
        self.input.clear();
        self.reset_history_navigation();
        self.provider_editor = Some(ProviderEditor::new(self.providers.clone()));
    }

    fn provider_editor_open(&self) -> bool {
        self.provider_editor.is_some()
    }

    fn provider_editor_is_editing(&self) -> bool {
        self.provider_editor
            .as_ref()
            .is_some_and(|editor| editor.field_editor.is_some())
    }

    fn provider_editor_is_confirming(&self) -> bool {
        self.provider_editor
            .as_ref()
            .is_some_and(|editor| editor.dialog.is_some())
    }

    fn next_provider_pane(&mut self, reverse: bool) {
        if let Some(editor) = &mut self.provider_editor {
            editor.pane = editor.pane.next(reverse);
        }
    }

    fn select_provider_item(&mut self, reverse: bool) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        match editor.pane {
            ProviderPane::Providers => {
                let count = editor.draft.providers.len();
                if count == 0 {
                    return;
                }
                editor.provider_selected = if reverse {
                    editor.provider_selected.checked_sub(1).unwrap_or(count - 1)
                } else {
                    (editor.provider_selected + 1) % count
                };
                editor.detail_selected = 0;
                editor.model_selected = 0;
            }
            ProviderPane::Details => {
                editor.detail_selected = if reverse {
                    editor
                        .detail_selected
                        .checked_sub(1)
                        .unwrap_or(ProviderField::COUNT - 1)
                } else {
                    (editor.detail_selected + 1) % ProviderField::COUNT
                };
            }
            ProviderPane::Models => {
                let count = editor
                    .selected_provider()
                    .map_or(0, |provider| provider.models.len());
                if count == 0 {
                    return;
                }
                editor.model_selected = if reverse {
                    editor.model_selected.checked_sub(1).unwrap_or(count - 1)
                } else {
                    (editor.model_selected + 1) % count
                };
            }
        }
    }

    fn provider_item_selected(&self, pane: ProviderPane, index: usize) -> bool {
        self.provider_editor.as_ref().is_some_and(|editor| {
            editor.pane == pane
                && match pane {
                    ProviderPane::Providers => editor.provider_selected == index,
                    ProviderPane::Details => editor.detail_selected == index,
                    ProviderPane::Models => editor.model_selected == index,
                }
        })
    }

    fn select_provider_at(&mut self, pane: ProviderPane, index: usize) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        let count = match pane {
            ProviderPane::Providers => editor.draft.providers.len(),
            ProviderPane::Details => ProviderField::COUNT,
            ProviderPane::Models => editor
                .selected_provider()
                .map_or(0, |provider| provider.models.len()),
        };
        if index >= count {
            return;
        }
        editor.pane = pane;
        match pane {
            ProviderPane::Providers => {
                if editor.provider_selected != index {
                    editor.provider_selected = index;
                    editor.detail_selected = 0;
                    editor.model_selected = 0;
                }
            }
            ProviderPane::Details => editor.detail_selected = index,
            ProviderPane::Models => editor.model_selected = index,
        }
    }

    fn edit_provider_item(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        let target = match editor.pane {
            ProviderPane::Providers => {
                if editor.selected_provider().is_some() {
                    editor.pane = ProviderPane::Details;
                }
                return;
            }
            ProviderPane::Details => match ProviderField::from_index(editor.detail_selected) {
                Some(field) => {
                    if editor.detail_selected == 4 {
                        if let Some(provider) = editor.selected_provider_mut() {
                            provider.openai_api = match provider.openai_api {
                                OpenAiApi::ChatCompletions => OpenAiApi::Responses,
                                OpenAiApi::Responses => OpenAiApi::ChatCompletions,
                            };
                        }
                        return;
                    }
                    ProviderEditTarget::Provider(field)
                }
                None => return,
            },
            ProviderPane::Models => {
                if editor
                    .selected_provider()
                    .and_then(|provider| provider.models.get(editor.model_selected))
                    .is_none()
                {
                    return;
                }
                ProviderEditTarget::ModelName {
                    model_index: editor.model_selected,
                }
            }
        };
        let value = provider_field_value(editor, target);
        let mut input = InputBuffer::default();
        if !matches!(target, ProviderEditTarget::Provider(ProviderField::ApiKey)) {
            input.replace(&value);
        }
        editor.field_editor = Some(FieldEditor { target, input });
    }

    fn edit_selected_model_id(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        if editor.pane != ProviderPane::Models {
            return;
        }
        let model_index = editor.model_selected;
        let Some(model) = editor
            .selected_provider()
            .and_then(|provider| provider.models.get(model_index))
        else {
            return;
        };
        let mut input = InputBuffer::default();
        input.replace(&model.id);
        editor.field_editor = Some(FieldEditor {
            target: ProviderEditTarget::ModelId { model_index },
            input,
        });
    }

    fn cycle_selected_model_thinking_level(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        if editor.pane != ProviderPane::Models {
            return;
        }
        let model_selected = editor.model_selected;
        if let Some(model) = editor
            .selected_provider_mut()
            .and_then(|provider| provider.models.get_mut(model_selected))
        {
            let thinking = model
                .thinking
                .get_or_insert_with(crate::provider::ThinkingConfig::default);
            thinking.max_level = thinking.max_level.next();
            if thinking.max_level == ThinkingLevel::Off {
                thinking.max_level = thinking.min_level;
            }
            if thinking.max_level < thinking.min_level {
                thinking.max_level = thinking.min_level;
            }
            thinking.supported = None;
        }
    }

    fn cycle_selected_model_thinking_min_level(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        if editor.pane != ProviderPane::Models {
            return;
        }
        let model_selected = editor.model_selected;
        if let Some(model) = editor
            .selected_provider_mut()
            .and_then(|provider| provider.models.get_mut(model_selected))
        {
            let thinking = model
                .thinking
                .get_or_insert_with(crate::provider::ThinkingConfig::default);
            let mut next = thinking.min_level.next();
            if next == ThinkingLevel::Off || next > thinking.max_level {
                next = ThinkingLevel::Minimal;
            }
            thinking.min_level = next;
            thinking.supported = None;
        }
    }

    fn cycle_selected_model_reasoning_map(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        if editor.pane != ProviderPane::Models {
            return;
        }
        let model_selected = editor.model_selected;
        let Some(model) = editor
            .selected_provider_mut()
            .and_then(|provider| provider.models.get_mut(model_selected))
        else {
            return;
        };
        let thinking = model
            .thinking
            .get_or_insert_with(crate::provider::ThinkingConfig::default)
            .clone();
        let compat = model.compat.get_or_insert_default();
        compat.supports_reasoning_effort = Some(true);
        compat.reasoning_effort_map.clear();
        for level in ThinkingLevel::ALL
            .into_iter()
            .filter(|level| *level >= thinking.min_level && *level <= thinking.max_level)
        {
            compat
                .reasoning_effort_map
                .entry(level)
                .or_insert_with(|| level.to_string());
        }
    }

    fn selected_provider_to_fetch(&mut self) -> Option<ProviderConfig> {
        self.commit_provider_field();
        let editor = self.provider_editor.as_ref()?;
        editor.selected_provider().cloned()
    }

    fn merge_discovered_models(&mut self, provider_id: &str, models: Vec<String>) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        let Some(provider_index) = editor
            .draft
            .providers
            .iter()
            .position(|provider| provider.id == provider_id)
        else {
            self.show_toast(
                "Provider changed before model discovery completed".to_owned(),
                ToastTone::Neutral,
            );
            return;
        };
        let provider = &mut editor.draft.providers[provider_index];
        let existing = provider
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let imported = models
            .into_iter()
            .filter(|model_id| !existing.contains(model_id.as_str()))
            .map(|model_id| ModelConfig {
                display_name: model_id.clone(),
                id: model_id,
                thinking: None,
                compat: None,
            })
            .collect::<Vec<_>>();
        let imported_count = imported.len();
        provider.models.extend(imported);
        provider
            .models
            .sort_by(|left, right| left.id.cmp(&right.id));
        editor.provider_selected = provider_index;
        editor.pane = ProviderPane::Models;
        editor.model_selected = provider.models.len().saturating_sub(imported_count.max(1));
        self.show_toast(
            if imported_count == 0 {
                "Models refreshed · no new models".to_owned()
            } else {
                format!("Models fetched · imported {imported_count}")
            },
            ToastTone::Success,
        );
    }

    fn new_provider_item(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        match editor.pane {
            ProviderPane::Providers | ProviderPane::Details => {
                let number = editor.draft.providers.len() + 1;
                editor.draft.providers.push(ProviderConfig {
                    id: format!("provider-{number}"),
                    display_name: format!("Provider {number}"),
                    base_url: "https://api.openai.com/v1".to_owned(),
                    api_key: SecretValue::new(String::new()),
                    openai_api: OpenAiApi::Responses,
                    thinking: None,
                    compat: None,
                    models: Vec::new(),
                });
                editor.provider_selected = editor.draft.providers.len() - 1;
                editor.detail_selected = 0;
                editor.model_selected = 0;
                editor.pane = ProviderPane::Details;
                self.edit_provider_item();
            }
            ProviderPane::Models => {
                let provider_index = editor.provider_selected;
                let Some(provider) = editor.draft.providers.get_mut(provider_index) else {
                    return;
                };
                let number = provider.models.len() + 1;
                provider.models.push(ModelConfig {
                    id: format!("model-{number}"),
                    display_name: format!("Model {number}"),
                    thinking: None,
                    compat: None,
                });
                let model_selected = provider.models.len() - 1;
                let model_id = provider.models[model_selected].id.clone();
                editor.model_selected = model_selected;
                let target = ProviderEditTarget::ModelId {
                    model_index: model_selected,
                };
                let mut input = InputBuffer::default();
                input.replace(&model_id);
                editor.field_editor = Some(FieldEditor { target, input });
            }
        }
    }

    fn request_provider_delete(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        let target = match editor.pane {
            ProviderPane::Providers | ProviderPane::Details => {
                if editor.selected_provider().is_none() {
                    return;
                }
                DeleteTarget::Provider(editor.provider_selected)
            }
            ProviderPane::Models => {
                if editor
                    .selected_provider()
                    .and_then(|provider| provider.models.get(editor.model_selected))
                    .is_none()
                {
                    return;
                }
                DeleteTarget::Model {
                    provider_index: editor.provider_selected,
                    model_index: editor.model_selected,
                }
            }
        };
        if delete_target_contains_active(&editor.draft, target) {
            self.show_toast(
                "Switch away from the active model before deleting it".to_owned(),
                ToastTone::Neutral,
            );
            return;
        }
        editor.dialog = Some(ProviderDialog::Delete(target));
    }

    fn request_provider_exit(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        if editor.dirty() {
            editor.dialog = Some(ProviderDialog::Discard);
        } else {
            self.provider_editor = None;
        }
    }

    fn confirm_provider_action(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        match editor.dialog.take() {
            Some(ProviderDialog::Discard) => {
                self.provider_editor = None;
            }
            Some(ProviderDialog::Delete(DeleteTarget::Provider(index))) => {
                if index < editor.draft.providers.len() {
                    editor.draft.providers.remove(index);
                    editor.provider_selected = editor
                        .provider_selected
                        .min(editor.draft.providers.len().saturating_sub(1));
                    editor.model_selected = 0;
                }
            }
            Some(ProviderDialog::Delete(DeleteTarget::Model {
                provider_index,
                model_index,
            })) => {
                if let Some(provider) = editor.draft.providers.get_mut(provider_index)
                    && model_index < provider.models.len()
                {
                    provider.models.remove(model_index);
                    editor.model_selected = editor
                        .model_selected
                        .min(provider.models.len().saturating_sub(1));
                }
            }
            None => {}
        }
    }

    fn cancel_provider_action(&mut self) {
        if let Some(editor) = &mut self.provider_editor {
            editor.dialog = None;
        }
    }

    fn toggle_provider_value(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        match editor.pane {
            ProviderPane::Details if editor.detail_selected == 4 => {
                if let Some(provider) = editor.selected_provider_mut() {
                    provider.openai_api = match provider.openai_api {
                        OpenAiApi::ChatCompletions => OpenAiApi::Responses,
                        OpenAiApi::Responses => OpenAiApi::ChatCompletions,
                    };
                }
            }
            ProviderPane::Models => {
                let model_selected = editor.model_selected;
                if let Some(model) = editor
                    .selected_provider_mut()
                    .and_then(|provider| provider.models.get_mut(model_selected))
                {
                    model.thinking = if model.thinking.is_some() {
                        None
                    } else {
                        Some(crate::provider::ThinkingConfig::default())
                    };
                }
            }
            _ => {}
        }
    }

    fn provider_input_insert(&mut self, character: char) {
        if let Some(input) = self
            .provider_editor
            .as_mut()
            .and_then(|editor| editor.field_editor.as_mut())
            .map(|field| &mut field.input)
        {
            input.insert_char(character);
        }
    }

    fn provider_input_backspace(&mut self) {
        if let Some(input) = self
            .provider_editor
            .as_mut()
            .and_then(|editor| editor.field_editor.as_mut())
            .map(|field| &mut field.input)
        {
            input.backspace();
        }
    }

    fn provider_input_delete(&mut self) {
        if let Some(input) = self
            .provider_editor
            .as_mut()
            .and_then(|editor| editor.field_editor.as_mut())
            .map(|field| &mut field.input)
        {
            input.delete();
        }
    }

    fn provider_input_left(&mut self) {
        if let Some(input) = self
            .provider_editor
            .as_mut()
            .and_then(|editor| editor.field_editor.as_mut())
            .map(|field| &mut field.input)
        {
            input.move_left();
        }
    }

    fn provider_input_right(&mut self) {
        if let Some(input) = self
            .provider_editor
            .as_mut()
            .and_then(|editor| editor.field_editor.as_mut())
            .map(|field| &mut field.input)
        {
            input.move_right();
        }
    }

    fn commit_provider_field(&mut self) {
        let Some(editor) = &mut self.provider_editor else {
            return;
        };
        let Some(field_editor) = editor.field_editor.take() else {
            return;
        };
        let value = field_editor.input.content.trim().to_owned();
        match field_editor.target {
            ProviderEditTarget::Provider(field) => {
                let Some(provider) = editor.selected_provider_mut() else {
                    return;
                };
                match field {
                    ProviderField::Id => provider.id = value,
                    ProviderField::DisplayName => provider.display_name = value,
                    ProviderField::BaseUrl => provider.base_url = value,
                    ProviderField::ApiKey if value.is_empty() => {}
                    ProviderField::ApiKey => provider.api_key = SecretValue::new(value),
                }
            }
            ProviderEditTarget::ModelName { model_index } => {
                let Some(model) = editor
                    .selected_provider_mut()
                    .and_then(|provider| provider.models.get_mut(model_index))
                else {
                    return;
                };
                model.display_name = value;
            }
            ProviderEditTarget::ModelId { model_index } => {
                let Some(model) = editor
                    .selected_provider_mut()
                    .and_then(|provider| provider.models.get_mut(model_index))
                else {
                    return;
                };
                model.id = value;
            }
        }
    }

    fn cancel_provider_field(&mut self) {
        if let Some(editor) = &mut self.provider_editor {
            editor.field_editor = None;
        }
    }

    fn provider_catalog_to_save(&mut self) -> Option<ProviderCatalog> {
        self.commit_provider_field();
        let editor = self.provider_editor.as_ref()?;
        Some(editor.draft.clone())
    }

    fn finish_provider_save(&mut self, catalog: ProviderCatalog) {
        if let Some(editor) = &mut self.provider_editor {
            editor.original = catalog.clone();
            editor.draft = catalog;
        }
        self.show_toast(
            "Provider configuration saved".to_owned(),
            ToastTone::Success,
        );
    }
}

fn provider_field_value(editor: &ProviderEditor, target: ProviderEditTarget) -> String {
    match target {
        ProviderEditTarget::Provider(field) => {
            let Some(provider) = editor.selected_provider() else {
                return String::new();
            };
            match field {
                ProviderField::Id => provider.id.clone(),
                ProviderField::DisplayName => provider.display_name.clone(),
                ProviderField::BaseUrl => provider.base_url.clone(),
                ProviderField::ApiKey => provider.api_key.expose().to_owned(),
            }
        }
        ProviderEditTarget::ModelId { model_index }
        | ProviderEditTarget::ModelName { model_index } => {
            let Some(model) = editor
                .selected_provider()
                .and_then(|provider| provider.models.get(model_index))
            else {
                return String::new();
            };
            match target {
                ProviderEditTarget::ModelId { .. } => model.id.clone(),
                ProviderEditTarget::ModelName { .. } => model.display_name.clone(),
                ProviderEditTarget::Provider(_) => unreachable!(),
            }
        }
    }
}

fn delete_target_contains_active(catalog: &ProviderCatalog, target: DeleteTarget) -> bool {
    let Some(active) = &catalog.active_model else {
        return false;
    };
    match target {
        DeleteTarget::Provider(provider_index) => catalog
            .providers
            .get(provider_index)
            .is_some_and(|provider| provider.id == active.provider_id),
        DeleteTarget::Model {
            provider_index,
            model_index,
        } => catalog
            .providers
            .get(provider_index)
            .and_then(|provider| {
                provider
                    .models
                    .get(model_index)
                    .map(|model| (provider, model))
            })
            .is_some_and(|(provider, model)| {
                provider.id == active.provider_id && model.id == active.model_id
            }),
    }
}

fn load_git_status(working_dir: &Path) -> Option<GitStatus> {
    let branch = git_output(working_dir, &["branch", "--show-current"]).or_else(|| {
        git_output(working_dir, &["rev-parse", "--short", "HEAD"])
            .map(|commit| format!("detached:{commit}"))
    })?;
    let commit = git_output(working_dir, &["rev-parse", "--short", "HEAD"])?;
    let dirty_count = git_output(working_dir, &["status", "--porcelain"])
        .map(|status| status.lines().count())
        .unwrap_or(0);
    Some(GitStatus {
        branch,
        commit,
        dirty_count,
    })
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

mod render;

use render::{
    error_summary, format_json, render, sanitize_terminal_text, single_line,
    tool_detail_line_count, truncate_chars,
};

#[cfg(test)]
use render::{
    align_with_footer_input, input_metrics, landing_regions, ui_regions, working_shimmer_line,
};

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
mod tests;
