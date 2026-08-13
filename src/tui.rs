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
const HORIZONTAL_GUTTER: u16 = 1;
const MAX_INPUT_ROWS: usize = 5;
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

fn render(frame: &mut Frame<'_>, app: &mut App) {
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

fn working_shimmer_line(text: &str, elapsed_secs: f32) -> Line<'static> {
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
struct UiRegions {
    transcript: Rect,
    completion: Rect,
    working: Rect,
    footer: Rect,
    keymap: Rect,
}

fn ui_regions(area: Rect, app: &App) -> UiRegions {
    let keymap_height = u16::from(area.height >= 3);
    let working_height = u16::from(app.busy && area.height >= 8);
    let requested_input_rows = if app.busy || app.page_open() {
        1
    } else {
        input_metrics(
            &app.input.content,
            app.input.cursor,
            footer_input_width(area.width).saturating_sub(4).max(1) as usize,
        )
        .total_rows
        .clamp(1, MAX_INPUT_ROWS)
    };
    let preferred_footer_height = (requested_input_rows as u16).saturating_add(1);
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

fn align_with_footer_input(area: Rect, footer: Rect) -> Rect {
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
struct LandingRegions {
    brand: Rect,
    card: Rect,
    hint: Rect,
    status: Rect,
}

fn landing_regions(area: Rect, app: &App) -> LandingRegions {
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
        card_width.saturating_sub(4).max(1) as usize,
    )
    .total_rows
    .clamp(1, MAX_INPUT_ROWS) as u16;
    let card_height = match stage.height {
        9.. => input_rows.saturating_add(1).max(2),
        5..=8 => 3,
        1..=4 => 1,
        _ => 0,
    };
    let brand_height = u16::from(stage.height >= 7);
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
        area.x,
        card.bottom().saturating_add(hint_gap),
        area.width,
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
        let brand = Line::from(Span::styled(
            "ZEX",
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        ));
        frame.render_widget(
            Paragraph::new(brand).alignment(Alignment::Center),
            regions.brand,
        );
    }

    render_landing_card(frame, regions.card, app);

    if !regions.hint.is_empty() {
        let hint = if regions.hint.width >= 72 {
            "Enter send  ·  Shift+Enter newline  ·  / commands"
        } else if regions.hint.width >= 42 {
            "Enter send  ·  Shift+Enter newline  ·  /"
        } else {
            "Enter send  ·  /"
        };
        frame.render_widget(
            Paragraph::new(Span::styled(hint, Style::default().fg(TEXT_FAINT)))
                .alignment(Alignment::Center),
            regions.hint,
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

fn landing_card_width(width: u16) -> u16 {
    if width == 0 {
        return 0;
    }
    let max_width = width.saturating_sub(2).max(1);
    let target = (u32::from(width) * 3 / 5) as u16;
    target.clamp(max_width.min(24), max_width.min(72))
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
    let model_row = u16::from(inner_rows >= 2);
    let editor_area = Rect::new(
        area.x + 3,
        area.y,
        area.width.saturating_sub(4),
        inner_rows - model_row,
    );
    if model_row == 1 {
        let model = model_short_name(&app.model);
        let thinking = thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference));
        frame.render_widget(
            Paragraph::new(Span::styled(
                format!("{model}  ·  {thinking}"),
                Style::default().fg(TEXT_DIM).bg(SURFACE),
            ))
            .style(Style::default().bg(SURFACE)),
            Rect::new(editor_area.x, editor_area.bottom(), editor_area.width, 1),
        );
    }
    if !editor_area.is_empty() {
        render_input_buffer(
            frame,
            editor_area,
            app,
            "",
            Some(Line::from(Span::styled(
                "ask anything…",
                Style::default().fg(TEXT_FAINT).bg(SURFACE),
            ))),
            SURFACE,
        );
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
    let label = truncate_inline(
        &format!(
            "{}  ·  Zex v{}",
            app.working_dir.display(),
            env!("CARGO_PKG_VERSION")
        ),
        area.width as usize,
    );
    frame.render_widget(
        Paragraph::new(Span::styled(label, Style::default().fg(TEXT_FAINT)))
            .style(Style::default().bg(BACKGROUND)),
        area,
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
        match entry {
            TranscriptEntry::Message { role, content } => {
                let final_answer = *role == MessageRole::Assistant
                    && index > 0
                    && matches!(
                        app.transcript.get(index - 1),
                        Some(TranscriptEntry::Turn(_))
                    );
                append_markdown_lines(&mut lines, content, *role, final_answer, width);
                lines.push(Line::default());
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
                lines.push(Line::default());
            }
            TranscriptEntry::Sessions(sessions) => {
                append_session_lines(&mut lines, sessions);
                lines.push(Line::default());
            }
        }
    }
    if app.busy {
        append_running_turn_line(&mut lines, app, width);
        lines.push(Line::default());
    }

    TranscriptRender {
        text: Text::from(lines),
        card_lines,
        output_lines,
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
    let mut details = vec![
        model_short_name(&turn.model),
        thinking_short_name(turn.thinking).to_owned(),
        format!(
            "{} tool{}",
            turn.tool_count,
            if turn.tool_count == 1 { "" } else { "s" }
        ),
    ];
    if let Some(elapsed) = turn.elapsed {
        details.push(format_turn_duration(elapsed));
    }
    if let Some(tokens) = turn.output_tokens {
        details.push(format_compact_count(tokens));
    }
    let prefix = format!("{marker} {label}   ");
    let details = details.join("  ·  ");
    let details_width = width.saturating_sub(UnicodeWidthStr::width(prefix.as_str()));
    lines.push(Line::from(vec![
        Span::styled(
            prefix,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            truncate_display(&details, details_width),
            Style::default().fg(TEXT_DIM),
        ),
    ]));
}

fn append_running_turn_line(lines: &mut Vec<Line<'static>>, app: &App, width: usize) {
    let elapsed = app
        .turn_started
        .map(|started| format_turn_duration(started.elapsed()))
        .unwrap_or_else(|| "0s".to_owned());
    let details = [
        model_short_name(&app.model),
        thinking_short_name(app.thinking_level.unwrap_or(app.thinking_preference)).to_owned(),
        format!(
            "{} tool{}",
            app.turn_tool_count,
            if app.turn_tool_count == 1 { "" } else { "s" }
        ),
        elapsed,
    ]
    .join("  ·  ");
    let prefix = "… running     ";
    let details_width = width.saturating_sub(UnicodeWidthStr::width(prefix));
    lines.push(Line::from(vec![
        Span::styled(
            prefix,
            Style::default()
                .fg(ACCENT_PRIMARY)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            truncate_display(&details, details_width),
            Style::default().fg(TEXT_DIM),
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
        MessageRole::Assistant => TEXT,
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
        (MessageRole::User, _, true) => "│ ",
        (MessageRole::User, _, false) => "│ ",
        (MessageRole::Assistant, false, true) => "• ",
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
    for segment in wrap_display_words(content, width.saturating_sub(2).max(1)) {
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
    let content_width = width.saturating_sub(2 + marker_width).max(1);
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

fn append_tool_lines(
    lines: &mut Vec<Line<'static>>,
    tool: &ToolEntry,
    selected: bool,
    hovered: bool,
    width: usize,
) {
    let marker = tool_marker(tool);
    let subject = tool_subject(tool);
    let elapsed = tool_elapsed(tool);
    let marker_color = if selected {
        ACCENT_PRIMARY
    } else {
        tool_result_color(tool)
    };
    let header_style = if selected {
        Style::default().bg(SURFACE_RAISED)
    } else if hovered {
        Style::default().bg(SURFACE_HOVER)
    } else {
        Style::default().bg(SURFACE)
    };
    let result = tool_result(tool);
    let duration = format_duration(elapsed);
    lines.push(
        tool_header_line(
            marker,
            &tool.name,
            &subject,
            &result,
            &duration,
            marker_color,
            tool_result_color(tool),
            width,
        )
        .style(header_style),
    );

    if tool.expanded {
        let rail_style = Style::default().fg(marker_color);
        let label_style = Style::default().fg(TEXT_DIM);
        let body_style = Style::default().fg(TEXT);
        let added_style = Style::default().fg(OK);
        let removed_style = Style::default().fg(BAD);
        let card_bg = Style::default().bg(SURFACE);
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
        let mut push = |spans: Vec<Span<'static>>| {
            let mut row = vec![Span::styled("  │ ", rail_style)];
            row.extend(spans);
            lines.push(Line::from(row).style(card_bg));
        };

        if edit_diff.is_some() {
            push(vec![
                Span::styled("diff  ", label_style),
                Span::styled(param.to_owned(), body_style),
            ]);
        } else if tool.name == "bash" && !failed {
            for (index, line) in param.lines().enumerate() {
                push(vec![
                    Span::styled(if index == 0 { "$ " } else { "  " }, label_style),
                    Span::styled(line.to_owned(), body_style),
                ]);
            }
        } else if param.is_empty() {
            push(vec![Span::styled(
                single_line(&tool.arguments, 200),
                label_style,
            )]);
        } else if tool.name == "bash" {
            push(vec![Span::styled("command", label_style)]);
            for line in param.lines() {
                push(vec![Span::styled(line.to_owned(), body_style)]);
            }
            push(vec![]);
            push(vec![Span::styled("stderr", label_style)]);
        } else {
            let indent = " ".repeat(param_key.len() + 2);
            for (index, line) in param.lines().enumerate() {
                push(vec![
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
                let style = if line.starts_with("+ ") {
                    added_style
                } else if line.starts_with("- ") {
                    removed_style
                } else {
                    body_style
                };
                push(vec![Span::styled(line.clone(), style)]);
            }
        } else if tool.output.is_empty() && matches!(tool.status, ToolStatus::Running) {
            push(vec![Span::styled("running…", label_style)]);
        } else if body_lines.is_empty() {
            push(vec![Span::styled("(no output)", label_style)]);
        } else {
            for line in body_lines.iter().take(visible_lines) {
                push(vec![Span::styled(line.clone(), body_style)]);
            }
        }

        let total = edit_diff
            .as_ref()
            .map_or_else(|| body_lines.len(), Vec::len);
        let unit = if total == 1 { "line" } else { "lines" };
        let footer = if !tool.show_full_output && total > TOOL_OUTPUT_PREVIEW_LINES {
            format!(
                "{total} {unit} · {} more · click or Ctrl+O expand",
                total - TOOL_OUTPUT_PREVIEW_LINES
            )
        } else if tool.show_full_output && total > TOOL_OUTPUT_PREVIEW_LINES {
            format!("{total} {unit} · click or Ctrl+O collapse")
        } else if total > 0 {
            format!("{total} {unit}")
        } else {
            format!("timeout {}", format_duration(tool.timeout))
        };
        lines.push(
            Line::from(vec![
                Span::styled("  └ ", rail_style),
                Span::styled(footer, label_style),
            ])
            .style(card_bg),
        );
    }
    lines.push(Line::default());
}

#[allow(clippy::too_many_arguments)]
fn tool_header_line(
    marker: &str,
    name: &str,
    subject: &str,
    result: &str,
    duration: &str,
    marker_color: Color,
    result_color: Color,
    width: usize,
) -> Line<'static> {
    const NAME_WIDTH: usize = 8;
    const RESULT_WIDTH: usize = 12;
    const DURATION_WIDTH: usize = 7;
    const MARKER_WIDTH: usize = 2;
    const GAP_WIDTH: usize = 2;

    let wide_fixed_width =
        MARKER_WIDTH + NAME_WIDTH + GAP_WIDTH * 3 + RESULT_WIDTH + DURATION_WIDTH;
    if width >= wide_fixed_width + 4 {
        let subject_width = width - wide_fixed_width;
        return Line::from(vec![
            Span::styled(format!("{marker} "), Style::default().fg(marker_color)),
            Span::styled(
                pad_display(&truncate_display(name, NAME_WIDTH), NAME_WIDTH),
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                pad_display(&truncate_display(subject, subject_width), subject_width),
                Style::default().fg(TEXT),
            ),
            Span::raw("  "),
            Span::styled(
                pad_display_left(&truncate_display(result, RESULT_WIDTH), RESULT_WIDTH),
                Style::default().fg(result_color),
            ),
            Span::raw("  "),
            Span::styled(
                pad_display_left(&truncate_display(duration, DURATION_WIDTH), DURATION_WIDTH),
                Style::default().fg(TEXT_DIM),
            ),
        ]);
    }

    let duration_width = UnicodeWidthStr::width(duration).min(DURATION_WIDTH);
    let result_width = UnicodeWidthStr::width(result).clamp(4, RESULT_WIDTH);
    let fixed_width = MARKER_WIDTH + NAME_WIDTH + GAP_WIDTH * 3 + result_width + duration_width;
    let subject_width = width.saturating_sub(fixed_width);
    if subject_width > 0 {
        return Line::from(vec![
            Span::styled(format!("{marker} "), Style::default().fg(marker_color)),
            Span::styled(
                pad_display(&truncate_display(name, NAME_WIDTH), NAME_WIDTH),
                Style::default()
                    .fg(TEXT_STRONG)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                pad_display(&truncate_display(subject, subject_width), subject_width),
                Style::default().fg(TEXT),
            ),
            Span::raw("  "),
            Span::styled(
                pad_display_left(&truncate_display(result, result_width), result_width),
                Style::default().fg(result_color),
            ),
            Span::raw("  "),
            Span::styled(
                truncate_display(duration, duration_width),
                Style::default().fg(TEXT_DIM),
            ),
        ]);
    }

    let available_name =
        width.saturating_sub(MARKER_WIDTH + GAP_WIDTH * 2 + result_width + duration_width);
    let name_width = available_name.clamp(1, NAME_WIDTH);
    Line::from(vec![
        Span::styled(format!("{marker} "), Style::default().fg(marker_color)),
        Span::styled(
            pad_display(&truncate_display(name, name_width), name_width),
            Style::default()
                .fg(TEXT_STRONG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            pad_display_left(&truncate_display(result, result_width), result_width),
            Style::default().fg(result_color),
        ),
        Span::raw("  "),
        Span::styled(
            truncate_display(duration, duration_width),
            Style::default().fg(TEXT_DIM),
        ),
    ])
}

fn tool_output_line_count(output: &str) -> usize {
    output.split('\n').count().max(1)
}

fn tool_detail_line_count(tool: &ToolEntry) -> usize {
    file_change_body(tool)
        .map_or_else(|| tool_output_line_count(&tool.output), |lines| lines.len())
        .max(1)
}

fn tool_marker(tool: &ToolEntry) -> &'static str {
    if tool.expanded {
        return "◆";
    }
    match tool.status {
        ToolStatus::Running => "◌",
        ToolStatus::Done => "●",
        ToolStatus::Failed => "✗",
        ToolStatus::Cancelled => "×",
    }
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
        area.x.saturating_add(2),
        area.y,
        area.width.saturating_sub(4),
        area.height,
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

/// Strip ANSI escape sequences, carriage returns and control characters so
/// captured tool output never renders as garbled terminal bytes. Runs of
/// replacement characters (from lossy decoding) collapse to a single marker.
fn sanitize_terminal_text(text: &str) -> String {
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
    use ratatui::{Terminal, backend::TestBackend, style::Color};

    use super::{
        ACCENT_PRIMARY, ACCENT_SECONDARY, App, AppContext, BACKGROUND, BAD, CommandOutput,
        HitTarget, InputAction, InputBuffer, KeyBurst, OK, ProviderPane, SCROLL_STEP, SURFACE,
        SURFACE_RAISED, Status, TEXT_DIM, ThinkingEntry, ToolStatus, TranscriptEntry,
        command_specs, handle_key_event, handle_mouse_event, handle_terminal_event, input_metrics,
        landing_regions, render, sanitize_terminal_text, truncate_chars, ui_regions,
        working_shimmer_line,
    };
    use crate::agent::{AgentEvent, Message, MessageRole};
    use crate::config::{ModelConfig, ModelRef, ProviderCatalog, ProviderConfig, SecretValue};
    use crate::provider::{OpenAiApi, ThinkingLevel};

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
                show_thinking: true,
                providers: ProviderCatalog::default(),
            },
        )
    }

    fn configured_app() -> App {
        let active_model = ModelRef {
            provider_id: "openai".to_owned(),
            model_id: "gpt-5".to_owned(),
        };
        let providers = ProviderCatalog {
            active_model: Some(active_model.clone()),
            models_dev: Default::default(),
            models_dev_aliases: Vec::new(),
            providers: vec![ProviderConfig {
                id: "openai".to_owned(),
                display_name: "OpenAI".to_owned(),
                base_url: "https://api.openai.com/v1".to_owned(),
                api_key: SecretValue::new("secret".to_owned()),
                openai_api: OpenAiApi::Responses,
                thinking: None,
                compat: None,
                models: vec![
                    ModelConfig {
                        id: "gpt-5".to_owned(),
                        display_name: "GPT-5".to_owned(),
                        thinking: Some(crate::provider::ThinkingConfig {
                            min_level: ThinkingLevel::Low,
                            max_level: ThinkingLevel::Max,
                            supported: None,
                            mode: crate::provider::ThinkingMode::Effort,
                        }),
                        compat: None,
                    },
                    ModelConfig {
                        id: "gpt-4.1-mini".to_owned(),
                        display_name: "GPT-4.1 Mini".to_owned(),
                        thinking: None,
                        compat: Some(crate::provider::ThinkingCompat {
                            supports_reasoning_effort: Some(false),
                            reasoning_effort_map: Default::default(),
                            supports_interleaved_thinking: Some(false),
                        }),
                    },
                ],
            }],
        };
        App::new(
            &[],
            active_model.key(),
            None,
            AppContext {
                working_dir: PathBuf::from("."),
                thinking_level: Some(ThinkingLevel::High),
                thinking_preference: ThinkingLevel::High,
                context_chars: 0,
                max_context_chars: 120_000,
                default_tool_timeout: Duration::from_secs(60),
                show_thinking: true,
                providers,
            },
        )
    }

    fn registry_agent(
        catalog: &ProviderCatalog,
        active_model: &ModelRef,
    ) -> crate::agent::Agent<crate::provider::ProviderRegistry> {
        let (events, _) = tokio::sync::mpsc::unbounded_channel();
        crate::agent::Agent::new(
            crate::provider::ProviderRegistry::new(catalog, Duration::from_secs(1)).unwrap(),
            crate::tools::ToolRegistry::new(Duration::from_secs(1), 32_000),
            events,
            crate::agent::AgentOptions {
                model: active_model.key(),
                turn_timeout: Duration::from_secs(1),
                max_turns: 1,
                max_context_chars: 120_000,
                compact_keep_turns: 6,
                thinking_level: ThinkingLevel::High,
            },
            None,
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

    fn mouse(
        kind: crossterm::event::MouseEventKind,
        column: u16,
        row: u16,
    ) -> crossterm::event::MouseEvent {
        crossterm::event::MouseEvent {
            kind,
            column,
            row,
            modifiers: crossterm::event::KeyModifiers::NONE,
        }
    }

    fn style_at(terminal: &Terminal<TestBackend>, x: u16, y: u16) -> ratatui::style::Style {
        terminal.backend().buffer()[(x, y)].style()
    }

    #[test]
    fn model_picker_selects_configured_models_without_editing_the_catalog() {
        let mut app = configured_app();
        app.open_model_picker();

        assert_eq!(app.model_picker.as_ref().unwrap().selected, 0);
        let action = handle_key_event(
            key(
                crossterm::event::KeyCode::Char('j'),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );
        assert_eq!(action, InputAction::None);
        let action = handle_key_event(
            key(
                crossterm::event::KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );
        assert!(matches!(
            action,
            InputAction::SwitchModel(ModelRef {
                provider_id,
                model_id
            }) if provider_id == "openai" && model_id == "gpt-4.1-mini"
        ));
        assert_eq!(
            app.providers.active_model.as_ref().unwrap().model_id,
            "gpt-5"
        );
    }

    #[test]
    fn thinking_normalization_updates_effective_status_without_changing_visibility() {
        let mut app = configured_app();
        app.show_thinking = false;

        app.apply_agent_event(AgentEvent::ThinkingNormalized {
            requested: ThinkingLevel::Max,
            clamped: ThinkingLevel::Max,
            effective: ThinkingLevel::High,
            provider_value: Some("high".to_owned()),
        });

        assert_eq!(app.thinking_preference, ThinkingLevel::Max);
        assert_eq!(app.thinking_level, Some(ThinkingLevel::High));
        assert!(!app.show_thinking);
    }

    #[test]
    fn provider_usage_updates_statusline_rate_without_feed_rows() {
        let mut app = configured_app();
        let transcript = app.transcript.clone();

        app.apply_agent_event(AgentEvent::ProviderUsage {
            output_tokens: 128,
            elapsed: Duration::from_secs(2),
        });

        assert_eq!(app.tokens_per_second, Some(64.0));
        assert_eq!(app.transcript, transcript);

        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Ready.".to_owned(),
        });
        let backend = TestBackend::new(120, 18);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        assert!(format!("{}", terminal.backend()).contains("64.0 tok/s"));
    }

    #[test]
    fn statusline_prefers_model_think_and_context_as_width_shrinks() {
        let mut app = configured_app();
        app.model = "openai/gpt-5.6-sol".to_owned();
        app.working_dir = PathBuf::from("D:/code/Zex");
        app.git_status = Some(super::GitStatus {
            branch: "feature/statusline-polish".to_owned(),
            commit: "019ff991".to_owned(),
            dirty_count: 3,
        });
        app.session_id = Some("20260813-120000-cafebabe".to_owned());
        app.context_chars = 58_920;
        app.tokens_per_second = Some(42.7);
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Ready.".to_owned(),
        });

        let backend = TestBackend::new(120, 18);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let wide = format!("{}", terminal.backend());
        assert!(wide.contains("gpt-5.6-sol"));
        assert!(wide.contains("high"));
        assert!(wide.contains("Zex"));
        assert!(wide.contains("*3"));
        assert!(wide.contains("42.7 tok/s"));
        assert!(wide.contains("49.1%/120Kc"));

        let backend = TestBackend::new(42, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let narrow = format!("{}", terminal.backend());
        assert!(narrow.contains("gpt-5.6-sol"));
        assert!(narrow.contains("high"));
        assert!(narrow.contains("49.1%/120K"));
        assert!(!narrow.contains("feature/statusline-polish"));
        assert!(!narrow.contains("cafebabe"));
    }

    #[test]
    fn thinking_statusline_hides_stale_rate_and_keeps_input_frame_empty() {
        let mut app = configured_app();
        app.tokens_per_second = Some(42.7);
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::User,
            content: "Inspect.".to_owned(),
        });
        app.start_turn();
        let backend = TestBackend::new(100, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let regions = ui_regions(ratatui::layout::Rect::new(0, 0, 100, 18), &app);
        assert!(screen.contains("thinking"));
        assert!(!screen.contains("42.7 tok/s"));
        assert!(!screen.contains("processing turn"));
        assert_eq!(
            terminal.backend().buffer()[(regions.footer.x + 3, regions.footer.y + 1)].symbol(),
            " "
        );
    }

    #[test]
    fn model_picker_and_provider_editor_replace_the_main_area() {
        let mut app = configured_app();
        app.open_model_picker();
        let backend = TestBackend::new(110, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("Models"));
        assert!(screen.contains("Current: OpenAI / GPT-5"));
        assert!(!screen.contains("Ask anything…"));

        app.dismiss_model_picker();
        app.open_provider_editor();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("Providers"));
        assert!(screen.contains("Provider details"));
        assert!(screen.contains("API key"));
        assert!(!screen.contains("secret"));
        assert!(!screen.contains("Ask anything…"));
    }

    #[test]
    fn provider_editor_protects_the_active_model_from_deletion() {
        let mut app = configured_app();
        app.open_provider_editor();
        app.provider_editor.as_mut().unwrap().pane = ProviderPane::Models;
        app.provider_editor.as_mut().unwrap().model_selected = 0;

        app.request_provider_delete();

        assert!(app.provider_editor.as_ref().unwrap().dialog.is_none());
        assert!(app.toast.is_some());
        assert_eq!(
            app.provider_editor.as_ref().unwrap().draft.providers[0]
                .models
                .len(),
            2
        );
    }

    #[test]
    fn provider_editor_can_add_and_edit_a_model_draft() {
        let mut app = configured_app();
        app.open_provider_editor();
        app.provider_editor.as_mut().unwrap().pane = ProviderPane::Models;

        app.new_provider_item();
        let editor = app.provider_editor.as_ref().unwrap();
        assert_eq!(editor.draft.providers[0].models.len(), 3);
        assert!(editor.field_editor.is_some());

        app.provider_editor
            .as_mut()
            .unwrap()
            .field_editor
            .as_mut()
            .unwrap()
            .input
            .replace("custom-model");
        app.commit_provider_field();

        assert_eq!(
            app.provider_editor.as_ref().unwrap().draft.providers[0].models[2].id,
            "custom-model"
        );
        assert_eq!(
            app.providers.providers[0].models.len(),
            2,
            "editing remains isolated until save"
        );
    }

    #[test]
    fn provider_editor_fetch_action_uses_current_draft_credentials() {
        let mut app = configured_app();
        app.open_provider_editor();

        let action = handle_key_event(
            key(
                crossterm::event::KeyCode::Char('f'),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );

        assert!(matches!(
            action,
            InputAction::FetchProviderModels(provider)
                if provider.id == "openai"
                    && provider.base_url == "https://api.openai.com/v1"
                    && provider.api_key.expose() == "secret"
        ));
    }

    #[test]
    fn discovered_models_merge_without_overwriting_existing_configuration() {
        let mut app = configured_app();
        app.open_provider_editor();

        app.merge_discovered_models(
            "openai",
            vec![
                "gpt-5".to_owned(),
                "gpt-new".to_owned(),
                "gpt-4.1-mini".to_owned(),
            ],
        );

        let provider = &app.provider_editor.as_ref().unwrap().draft.providers[0];
        assert_eq!(
            provider
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["gpt-4.1-mini", "gpt-5", "gpt-new"]
        );
        let existing = provider
            .models
            .iter()
            .find(|model| model.id == "gpt-5")
            .unwrap();
        assert_eq!(
            existing
                .thinking
                .as_ref()
                .map(|thinking| thinking.max_level),
            Some(ThinkingLevel::Max)
        );
        let imported = provider
            .models
            .iter()
            .find(|model| model.id == "gpt-new")
            .unwrap();
        assert_eq!(imported.display_name, "gpt-new");
        assert!(imported.thinking.is_none());
        assert!(imported.compat.is_none());
        assert_eq!(
            app.toast.as_ref().map(|toast| toast.message.as_str()),
            Some("Models fetched · imported 1")
        );
    }

    #[test]
    fn model_picker_renders_models_dev_namespaced_thinking_levels() {
        let mut app = configured_app();
        app.providers.models_dev = crate::provider::ModelsDevCatalog::from_json(
            br#"{
                "gateway-one": {
                    "models": {
                        "openai/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh"]}
                            ]
                        }
                    }
                },
                "gateway-two": {
                    "models": {
                        "openai/gpt-5.4-mini": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh"]}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();
        app.providers.providers[0].models.push(ModelConfig {
            id: "gpt-5.4-mini".to_owned(),
            display_name: "GPT-5.4 mini".to_owned(),
            thinking: None,
            compat: None,
        });
        app.open_model_picker();

        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("GPT-5.4 mini"));
        assert!(screen.contains("think off/low/medium/high/xhigh"));
    }

    #[test]
    fn model_picker_renders_merged_xhigh_and_max_levels() {
        let mut app = configured_app();
        app.providers.models_dev = crate::provider::ModelsDevCatalog::from_json(
            br#"{
                "limited": {
                    "models": {
                        "gpt-5.6-sol": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["low", "medium", "high"]}
                            ]
                        }
                    }
                },
                "extended": {
                    "models": {
                        "openai/gpt-5.6-sol": {
                            "reasoning": true,
                            "reasoning_options": [
                                {"type": "effort", "values": ["none", "low", "medium", "high", "xhigh", "max"]}
                            ]
                        }
                    }
                }
            }"#,
        )
        .unwrap();
        app.providers.providers[0].models.push(ModelConfig {
            id: "gpt-5.6-sol".to_owned(),
            display_name: "GPT-5.6 Sol".to_owned(),
            thinking: None,
            compat: None,
        });
        app.open_model_picker();

        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("GPT-5.6 Sol"));
        assert!(screen.contains("think off/low/medium/high/xhigh/max"));
    }

    #[tokio::test]
    async fn model_switch_persists_and_updates_agent_status_without_touching_transcript() {
        let root = std::env::temp_dir().join(format!(
            "zex-model-switch-{}-{}",
            std::process::id(),
            time::OffsetDateTime::now_utc().unix_timestamp_nanos()
        ));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let mut app = configured_app();
        let original_transcript = app.transcript.clone();
        let active = app.providers.active_model.clone().unwrap();
        let mut agent = registry_agent(&app.providers, &active);
        let target = ModelRef {
            provider_id: "openai".to_owned(),
            model_id: "gpt-4.1-mini".to_owned(),
        };

        super::switch_model(&mut agent, &mut app, &root, None, target.clone())
            .await
            .unwrap();

        assert_eq!(agent.model(), target.key());
        assert_eq!(app.model, target.key());
        assert_eq!(app.transcript, original_transcript);
        let config = tokio::fs::read_to_string(root.join(".zex/config.toml"))
            .await
            .unwrap();
        assert!(config.contains("provider_id = \"openai\""));
        assert!(config.contains("model_id = \"gpt-4.1-mini\""));
        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn provider_save_refreshes_runtime_registry_and_model_picker_catalog() {
        let root = std::env::temp_dir().join(format!(
            "zex-provider-save-{}-{}",
            std::process::id(),
            time::OffsetDateTime::now_utc().unix_timestamp_nanos()
        ));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let mut app = configured_app();
        let active = app.providers.active_model.clone().unwrap();
        let registry =
            crate::provider::ProviderRegistry::new(&app.providers, Duration::from_secs(1)).unwrap();
        let mut agent = crate::agent::Agent::new(
            registry.clone(),
            crate::tools::ToolRegistry::new(Duration::from_secs(1), 32_000),
            tokio::sync::mpsc::unbounded_channel().0,
            crate::agent::AgentOptions {
                model: active.key(),
                turn_timeout: Duration::from_secs(1),
                max_turns: 1,
                max_context_chars: 120_000,
                compact_keep_turns: 6,
                thinking_level: ThinkingLevel::High,
            },
            None,
        );
        app.open_provider_editor();
        let mut draft = app.providers.clone();
        draft.providers[0].models.push(ModelConfig {
            id: "new-model".to_owned(),
            display_name: "New Model".to_owned(),
            thinking: Some(crate::provider::ThinkingConfig {
                min_level: ThinkingLevel::Minimal,
                max_level: ThinkingLevel::Medium,
                supported: None,
                mode: crate::provider::ThinkingMode::Effort,
            }),
            compat: None,
        });

        super::save_provider_changes(&mut agent, &mut app, &root, &registry, draft)
            .await
            .unwrap();
        app.open_model_picker();

        assert!(
            app.model_picker
                .as_ref()
                .unwrap()
                .choices
                .iter()
                .any(|choice| {
                    choice.target.model_id == "new-model"
                        && choice.thinking.max_level == ThinkingLevel::Medium
                })
        );
        let config = tokio::fs::read_to_string(root.join(".zex/config.toml"))
            .await
            .unwrap();
        assert!(config.contains("id = \"new-model\""));
        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn provider_save_remaps_the_active_target_when_ids_are_renamed() {
        let root = std::env::temp_dir().join(format!(
            "zex-provider-rename-{}-{}",
            std::process::id(),
            time::OffsetDateTime::now_utc().unix_timestamp_nanos()
        ));
        tokio::fs::create_dir_all(&root).await.unwrap();
        let mut app = configured_app();
        let active = app.providers.active_model.clone().unwrap();
        let registry =
            crate::provider::ProviderRegistry::new(&app.providers, Duration::from_secs(1)).unwrap();
        let mut agent = crate::agent::Agent::new(
            registry.clone(),
            crate::tools::ToolRegistry::new(Duration::from_secs(1), 32_000),
            tokio::sync::mpsc::unbounded_channel().0,
            crate::agent::AgentOptions {
                model: active.key(),
                turn_timeout: Duration::from_secs(1),
                max_turns: 1,
                max_context_chars: 120_000,
                compact_keep_turns: 6,
                thinking_level: ThinkingLevel::High,
            },
            None,
        );
        app.open_provider_editor();
        app.provider_editor.as_mut().unwrap().draft.providers[0].id = "renamed".to_owned();
        app.provider_editor.as_mut().unwrap().draft.providers[0].models[0].id =
            "renamed-model".to_owned();
        app.provider_editor.as_mut().unwrap().draft.active_model = Some(ModelRef {
            provider_id: "openai".to_owned(),
            model_id: "gpt-5".to_owned(),
        });
        let draft = app.provider_editor.as_ref().unwrap().draft.clone();

        super::save_provider_changes(&mut agent, &mut app, &root, &registry, draft)
            .await
            .unwrap();

        assert_eq!(agent.model(), "renamed/renamed-model");
        assert_eq!(
            app.providers.active_model.as_ref().unwrap().key(),
            "renamed/renamed-model"
        );
        let config = tokio::fs::read_to_string(root.join(".zex/config.toml"))
            .await
            .unwrap();
        assert!(config.contains("provider_id = \"renamed\""));
        assert!(config.contains("model_id = \"renamed-model\""));
        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[test]
    fn empty_state_centers_the_zex_card_and_keeps_status_quiet() {
        let mut app = configured_app();
        app.working_dir = PathBuf::from("D:/workspaces/zex");
        app.git_status = Some(super::GitStatus {
            branch: "main".to_owned(),
            commit: "a1b2c3d".to_owned(),
            dirty_count: 0,
        });
        app.thinking_level = Some(ThinkingLevel::High);
        app.context_chars = 30_000;
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 100, 24), &app);

        assert!(screen.contains("ZEX"));
        assert!(!screen.contains("Z  E  X"));
        assert!(screen.contains("ask anything…"));
        assert!(screen.contains("Enter send"));
        assert!(screen.contains("gpt-5"));
        assert!(screen.contains("high"));
        assert!(!screen.contains("25.0%/120Kc"));
        assert!(screen.contains("D:/workspaces/zex"));
        assert!(screen.contains(env!("CARGO_PKG_VERSION")));
        assert!(!screen.contains("● idle"));
        assert_eq!(style_at(&terminal, 0, 0).bg, Some(BACKGROUND));
        assert!(regions.card.y > 4);
        assert!(regions.card.bottom() < regions.status.y);
        assert!(regions.card.width < 100);
        assert_eq!(
            terminal.backend().buffer()[(regions.card.x, regions.card.y)].symbol(),
            "▌"
        );
        assert_eq!(
            style_at(&terminal, regions.card.x + 1, regions.card.y).bg,
            Some(SURFACE)
        );
    }

    #[test]
    fn empty_input_keeps_the_cursor_cell_clear_for_ime_preedit() {
        let mut app = app();
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 80, 16), &app);
        let editor_x = regions.card.x + 3;
        let editor_y = regions.card.y;

        terminal
            .backend_mut()
            .assert_cursor_position((editor_x, editor_y));
        assert_eq!(
            terminal.backend().buffer()[(editor_x, editor_y)].symbol(),
            " ",
            "the IME preedit must not overlap application-rendered text"
        );
    }

    #[test]
    fn typed_input_starts_at_the_empty_editor_cursor_origin() {
        let mut app = app();
        app.input.insert_str("hello");
        let backend = TestBackend::new(80, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 80, 16), &app);
        let editor_x = regions.card.x + 3;
        let editor_y = regions.card.y;

        assert!(screen.contains("hello"));
        assert!(!screen.contains("ask anything…"));
        terminal
            .backend_mut()
            .assert_cursor_position((editor_x + 5, editor_y));
    }

    #[test]
    fn empty_layout_remains_composed_across_terminal_sizes() {
        for (width, height) in [(120, 32), (70, 18), (38, 12), (16, 6), (6, 3)] {
            let mut app = app();
            app.input
                .insert_str("first line\nsecond line that wraps on narrow terminals");
            app.refresh_completion();
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();

            terminal.draw(|frame| render(frame, &mut app)).unwrap();
            let area = ratatui::layout::Rect::new(0, 0, width, height);
            let regions = landing_regions(area, &app);
            assert!(regions.brand.bottom() <= area.bottom());
            assert!(regions.card.bottom() <= area.bottom());
            assert!(regions.hint.bottom() <= area.bottom());
            assert!(regions.status.bottom() <= area.bottom());
            assert_eq!(style_at(&terminal, 0, 0).bg, Some(BACKGROUND));
        }
    }

    #[test]
    fn every_draw_paints_the_full_terminal_background() {
        for (width, height) in [(100, 24), (47, 13), (9, 4)] {
            let mut app = app();
            app.apply_agent_event(AgentEvent::MessageDelta {
                role: MessageRole::User,
                delta: "Inspect the project".to_owned(),
            });
            app.apply_agent_event(AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: "Ready.\n- one\n- two\n```text\noutput\n```".to_owned(),
            });
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();

            terminal.draw(|frame| render(frame, &mut app)).unwrap();

            for cell in terminal.backend().buffer().content() {
                assert!(
                    matches!(cell.style().bg, Some(BACKGROUND | SURFACE | SURFACE_RAISED)),
                    "unpainted cell at {width}x{height}: {:?}",
                    cell.style()
                );
            }
            for &(x, y) in &[
                (0, 0),
                (width.saturating_sub(1), 0),
                (0, height.saturating_sub(1)),
                (width.saturating_sub(1), height.saturating_sub(1)),
            ] {
                assert!(matches!(
                    style_at(&terminal, x, y).bg,
                    Some(BACKGROUND | SURFACE | SURFACE_RAISED)
                ));
            }
        }
    }

    #[test]
    fn grok_night_palette_matches_the_ui_plan() {
        assert_eq!(BACKGROUND, Color::Rgb(20, 20, 20));
        assert_eq!(super::TEXT, Color::Rgb(243, 243, 243));
        assert_eq!(TEXT_DIM, Color::Rgb(160, 160, 160));
        assert_eq!(super::TEXT_FAINT, Color::Rgb(120, 120, 120));
        assert_eq!(ACCENT_PRIMARY, Color::Rgb(122, 162, 247));
        assert_eq!(ACCENT_SECONDARY, Color::Rgb(187, 154, 247));
        assert_eq!(OK, Color::Rgb(158, 206, 106));
        assert_eq!(BAD, Color::Rgb(219, 75, 75));
    }

    #[test]
    fn first_turn_switches_to_a_top_anchored_work_timeline() {
        let mut app = app();
        assert!(app.landing_visible());
        app.start_turn();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::User,
            content: "Inspect the project".to_owned(),
        });
        let area = ratatui::layout::Rect::new(0, 0, 100, 24);
        let backend = TestBackend::new(area.width, area.height);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();

        let regions = ui_regions(area, &app);
        let screen = format!("{}", terminal.backend());
        let message_row = screen
            .lines()
            .position(|row| row.contains("Inspect the project"))
            .expect("message should be visible") as u16;
        assert!(!app.landing_visible());
        assert!(!screen.contains("ask anything…"));
        assert!(screen.contains("thinking"));
        assert_eq!(message_row, regions.transcript.y);
    }

    #[test]
    fn clearing_the_timeline_restores_the_landing_layout() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "Inspect the project".to_owned(),
        });
        assert!(!app.landing_visible());

        app.reset_transcript();
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(app.landing_visible());
        assert!(screen.contains("ZEX"));
        assert!(!screen.contains("Z  E  X"));
        assert!(screen.contains("ask anything…"));
        assert!(!screen.contains("Inspect the project"));
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
        for cell in buffer.content() {
            saw_surface |= cell.style().bg == Some(SURFACE);
            assert!(matches!(
                cell.style().bg,
                Some(BACKGROUND | SURFACE | SURFACE_RAISED)
            ));
        }

        assert!(saw_surface);
        assert!(format!("{}", terminal.backend()).contains("fn main() {}"));
        assert!(
            format!("{}", terminal.backend())
                .lines()
                .any(|row| row.contains("bash") && row.contains("cargo check"))
        );
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
            dirty_count: 0,
        });
        app.thinking_level = Some(ThinkingLevel::Medium);
        app.context_chars = 60_000;
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::User,
            content: "statusline stays readable when narrow".to_owned(),
        });
        let backend = TestBackend::new(38, 12);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("ZEX"), "screen:\n{screen}");
        assert!(screen.contains("med"));
        assert!(screen.contains("50.0%/120K"));
        assert!(!screen.contains("feature/very-long-branch-name"));
    }

    #[test]
    fn idle_thinking_and_running_states_are_clear_in_the_footer() {
        let mut app = app();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Ready for the next step.".to_owned(),
        });
        let backend = TestBackend::new(100, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let idle = format!("{}", terminal.backend());
        assert!(idle.contains("idle"));
        assert!(idle.contains("Enter send"));

        app.start_turn();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let thinking = format!("{}", terminal.backend());
        assert!(thinking.contains("thinking"));
        assert!(thinking.contains("Working..."));
        assert!(thinking.contains("Esc stop"));

        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-running".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let running = format!("{}", terminal.backend());
        assert!(running.contains("running"));
        assert!(
            running
                .lines()
                .any(|row| row.contains("read") && row.contains("Cargo.toml"))
        );
        assert!(running.contains("Esc stop"));
    }

    #[test]
    fn multiline_input_scrolls_inside_the_stable_footer() {
        let mut app = app();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Ready.".to_owned(),
        });
        let area = ratatui::layout::Rect::new(0, 0, 72, 18);
        let single = ui_regions(area, &app);
        app.input
            .insert_str("first line\nsecond line\nthird line\nfourth line");
        let multiline = ui_regions(area, &app);
        let backend = TestBackend::new(area.width, area.height);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();

        assert!(multiline.footer.height > single.footer.height);
        for y in [
            multiline.footer.y + 1,
            multiline.footer.y + 2,
            multiline.footer.bottom() - 1,
        ] {
            assert_eq!(
                terminal.backend().buffer()[(multiline.footer.x + 1, y)].symbol(),
                "│"
            );
        }
    }

    #[test]
    fn completion_panel_aligns_with_footer_and_highlights_selection() {
        let mut app = app();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Ready.".to_owned(),
        });
        app.input.insert_str("/");
        app.refresh_completion();
        let backend = TestBackend::new(100, 28);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let regions = ui_regions(ratatui::layout::Rect::new(0, 0, 100, 28), &app);
        let completion = super::align_with_footer_input(regions.completion, regions.footer);
        assert!(regions.completion.width > 0);
        assert_eq!(completion.x, regions.footer.x + super::HORIZONTAL_GUTTER);
        assert_eq!(
            completion.width,
            regions.footer.width - super::HORIZONTAL_GUTTER.saturating_mul(2)
        );
        assert_eq!(
            terminal.backend().buffer()[(
                regions.footer.x + super::HORIZONTAL_GUTTER,
                regions.footer.y + 1,
            )]
                .symbol(),
            "│"
        );
        let selected_row = regions.completion.y + 1;
        assert!(
            (regions.completion.x..regions.completion.right()).any(|x| style_at(
                &terminal,
                x,
                selected_row
            )
            .fg == Some(ACCENT_PRIMARY))
        );
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
    fn thinking_then_answer_remain_separate_timeline_entries() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ThinkingDelta {
            delta: "Reason first.".to_owned(),
        });
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "Final answer.".to_owned(),
        });

        assert!(matches!(
            &app.transcript[..],
            [
                TranscriptEntry::Thinking(ThinkingEntry { content: thinking, .. }),
                TranscriptEntry::Message {
                    role: MessageRole::Assistant,
                    content: answer,
                },
            ] if thinking == "Reason first." && answer == "Final answer."
        ));
    }

    #[test]
    fn thinking_is_a_folded_card_in_the_single_timeline() {
        let mut app = app();
        app.start_turn();
        app.apply_agent_event(AgentEvent::ThinkingDelta {
            delta: "Inspect constraints first.".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
            timeout: Duration::from_secs(60),
        });
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "Final answer.".to_owned(),
        });

        assert!(matches!(
            &app.transcript[..],
            [
                TranscriptEntry::Thinking(ThinkingEntry {
                    content,
                    expanded: false,
                }),
                TranscriptEntry::Tool(_),
                TranscriptEntry::Message {
                    role: MessageRole::Assistant,
                    content: answer,
                },
            ] if content == "Inspect constraints first." && answer == "Final answer."
        ));

        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let folded = format!("{}", terminal.backend());
        assert!(folded.contains("Inspect constraints first."));
        let thinking_row = folded
            .lines()
            .position(|row| row.contains("think") && row.contains("medium") && row.contains("done"))
            .expect("thinking row should be visible") as u16;
        assert!((0..100).any(|x| style_at(&terminal, x, thinking_row).bg == Some(SURFACE)));

        app.select_tool(false);
        app.toggle_selected_tool();
        let TranscriptEntry::Thinking(thinking) = &app.transcript[0] else {
            panic!("expected thinking entry");
        };
        assert!(thinking.expanded);
    }

    #[test]
    fn folded_trace_and_tool_cards_use_one_summary_row_each() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ThinkingDelta {
            delta: "Inspect constraints first.".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-summary".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-summary".to_owned(),
            name: "read".to_owned(),
            output: "package zex".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(12),
        });
        let backend = TestBackend::new(100, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let rows = screen.lines().collect::<Vec<_>>();

        assert_eq!(
            rows.iter()
                .filter(|row| row.contains("Inspect constraints first."))
                .count(),
            1
        );
        assert_eq!(
            rows.iter()
                .filter(|row| row.contains("read") && row.contains("Cargo.toml"))
                .count(),
            1
        );
    }

    #[test]
    fn completed_turn_status_precedes_the_final_answer() {
        let mut app = app();
        app.start_turn();
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "Inspect the project".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-read".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-read".to_owned(),
            name: "read".to_owned(),
            output: "line one\nline two".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(12),
        });
        app.apply_agent_event(AgentEvent::ProviderUsage {
            output_tokens: 1_234,
            elapsed: Duration::from_secs(2),
        });
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "Final answer.".to_owned(),
        });
        app.apply_agent_event(AgentEvent::TurnEnd);
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let status_row = screen
            .lines()
            .position(|row| row.contains("turn done"))
            .expect("missing completed turn row");
        let answer_row = screen
            .lines()
            .position(|row| row.contains("Final answer."))
            .expect("missing final answer");
        assert!(status_row < answer_row);
        assert!(screen.contains("1 tool"));
        assert!(screen.contains("1.2k"));
    }

    #[test]
    fn active_turn_renders_running_status_without_system_feed_rows() {
        let mut app = app();
        app.start_turn();
        app.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "Inspect".to_owned(),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-read".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        let backend = TestBackend::new(100, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert!(screen.lines().any(|row| {
            row.contains("running") && row.contains("test-model") && row.contains("1 tool")
        }));
        assert_eq!(screen.matches("Working...").count(), 1);
    }

    #[test]
    fn tool_cards_use_zex_subject_result_and_duration_summaries() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-bash".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"git status"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-bash".to_owned(),
            name: "bash".to_owned(),
            output: "exit_code: 0\nstdout:\nclean\nstderr:\n".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(8),
        });
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-grep".to_owned(),
            name: "grep".to_owned(),
            arguments: r#"{"pattern":"render_footer"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-grep".to_owned(),
            name: "grep".to_owned(),
            output: "src/tui.rs:1:render_footer\n\n11 matching line(s) in 1 file(s)".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(14),
        });
        let backend = TestBackend::new(110, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.lines().any(|row| {
            row.contains("bash")
                && row.contains("git status")
                && row.contains("exit 0")
                && row.contains("8ms")
        }));
        assert!(screen.lines().any(|row| {
            row.contains("grep")
                && row.contains("render_footer")
                && row.contains("11 matches")
                && row.contains("14ms")
        }));
        let tool_row = screen
            .lines()
            .position(|row| row.contains("bash") && row.contains("git status"))
            .expect("tool row should be visible") as u16;
        assert!((0..110).any(|x| style_at(&terminal, x, tool_row).bg == Some(SURFACE)));
    }

    #[test]
    fn tool_cards_align_status_and_duration_columns() {
        let mut app = app();
        let glob_output = format!("{}\n115 matching path(s)", "path\n".repeat(115));
        for (call_id, name, arguments, output, elapsed) in [
            (
                "call-bash",
                "bash",
                r#"{"command":"pwd"}"#,
                "exit_code: 1\nstdout:\n\nstderr:\nfailed".to_owned(),
                24,
            ),
            ("call-glob", "glob", r#"{"pattern":"*"}"#, glob_output, 23),
        ] {
            app.apply_agent_event(AgentEvent::ToolStart {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                arguments: arguments.to_owned(),
                timeout: Duration::from_secs(30),
            });
            app.apply_agent_event(AgentEvent::ToolEnd {
                call_id: call_id.to_owned(),
                name: name.to_owned(),
                output,
                is_error: false,
                elapsed: Duration::from_millis(elapsed),
            });
        }
        let backend = TestBackend::new(90, 20);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let rows = format!("{}", terminal.backend())
            .lines()
            .filter(|row| row.contains("pwd") || row.contains("115 paths"))
            .map(str::to_owned)
            .collect::<Vec<_>>();

        assert_eq!(rows.len(), 2);
        let status_columns = rows
            .iter()
            .map(|row| {
                let (status, byte) = ["exit 1", "115 paths"]
                    .into_iter()
                    .find_map(|status| row.find(status).map(|byte| (status, byte)))
                    .expect("missing tool status");
                unicode_width::UnicodeWidthStr::width(&row[..byte])
                    + unicode_width::UnicodeWidthStr::width(status)
            })
            .collect::<Vec<_>>();
        let duration_columns = rows
            .iter()
            .map(|row| {
                let byte = row.rfind("2").expect("missing duration");
                unicode_width::UnicodeWidthStr::width(&row[..byte])
            })
            .collect::<Vec<_>>();
        assert_eq!(status_columns[0], status_columns[1]);
        assert_eq!(duration_columns[0], duration_columns[1]);
    }

    #[test]
    fn failed_tool_colors_only_the_status_field_as_error() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-failed".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"cargo check"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-failed".to_owned(),
            name: "bash".to_owned(),
            output: "exit_code: 1\nstdout:\n\nstderr:\nfailed".to_owned(),
            is_error: true,
            elapsed: Duration::from_millis(418),
        });
        let backend = TestBackend::new(90, 16);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        let row = screen
            .lines()
            .position(|row| row.contains("cargo check"))
            .expect("missing failed tool row") as u16;
        let row_text = screen.lines().nth(row as usize).unwrap();
        let status_byte = row_text.find("exit 1").unwrap();
        let name_byte = row_text.find("bash").unwrap();
        let status_x = unicode_width::UnicodeWidthStr::width(&row_text[..status_byte]) as u16;
        let name_x = unicode_width::UnicodeWidthStr::width(&row_text[..name_byte]) as u16;

        assert_eq!(style_at(&terminal, status_x, row).fg, Some(BAD));
        assert_ne!(style_at(&terminal, name_x, row).fg, Some(BAD));
    }

    #[test]
    fn long_tool_output_previews_then_expands_fully() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-long".to_owned(),
            name: "read".to_owned(),
            arguments: r#"{"path":"long.txt"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-long".to_owned(),
            name: "read".to_owned(),
            output: (1..=20)
                .map(|line| format!("line {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
            is_error: false,
            elapsed: Duration::from_millis(12),
        });
        let backend = TestBackend::new(90, 40);
        let mut terminal = Terminal::new(backend).unwrap();

        app.toggle_selected_tool();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let preview = format!("{}", terminal.backend());
        assert!(preview.contains("path  long.txt"));
        assert!(!preview.contains("\"path\""));
        assert!(preview.contains("line 12"));
        assert!(!preview.contains("line 13"));
        assert!(preview.contains("20 lines · 8 more · click or Ctrl+O expand"));

        app.toggle_selected_tool_output();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let full = format!("{}", terminal.backend());
        assert!(full.contains("line 20"));
        assert!(full.contains("20 lines · click or Ctrl+O collapse"));
    }

    #[test]
    fn edit_card_renders_a_human_readable_diff_without_json() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-edit".to_owned(),
            name: "edit".to_owned(),
            arguments: serde_json::json!({
                "path": "src/tui.rs",
                "old_text": "fn render() {\n    old();\n}",
                "new_text": "fn render() {\n    new();\n    finish();\n}"
            })
            .to_string(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-edit".to_owned(),
            name: "edit".to_owned(),
            output: "edited src/tui.rs".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(8),
        });
        app.toggle_selected_tool();
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("edit"));
        assert!(screen.contains("src/tui.rs"));
        assert!(screen.contains("+2 −1"));
        assert!(screen.contains("-     old();"));
        assert!(screen.contains("+     new();"));
        assert!(screen.contains("+     finish();"));
        assert!(!screen.contains("\"old_text\""));
        assert!(!screen.contains("\"new_text\""));
    }

    #[test]
    fn failed_bash_card_shows_command_and_stderr_sections() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-fail".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"pwd && find . -name '*.rs'","timeout_seconds":30}"#
                .to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-fail".to_owned(),
            name: "bash".to_owned(),
            output: "exit_code: 1\nstdout:\n\nstderr:\n'pwd' is not recognized\n".to_owned(),
            is_error: true,
            elapsed: Duration::from_millis(20),
        });
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        app.toggle_selected_tool();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let expanded = format!("{}", terminal.backend());
        assert!(expanded.contains("exit 1"));
        assert!(expanded.contains("command"));
        assert!(expanded.contains("pwd && find . -name '*.rs'"));
        assert!(expanded.contains("stderr"));
        assert!(expanded.contains("'pwd' is not recognized"));
        assert!(!expanded.contains("exit_code:"));
        assert!(!expanded.contains("stdout:"));
        assert!(!expanded.contains("timeout_seconds"));
        assert!(!expanded.contains('{'));
    }

    #[test]
    fn working_shimmer_sweeps_primary_with_a_secondary_tail() {
        let line = working_shimmer_line("Working... (Esc)", 0.6);
        let colors: Vec<_> = line.spans.iter().map(|span| span.style.fg).collect();
        assert!(colors.contains(&Some(ACCENT_PRIMARY)));
        assert!(colors.contains(&Some(TEXT_DIM)));
        assert!(
            colors.iter().any(|color| {
                matches!(
                    color,
                    Some(Color::Rgb(r, g, b))
                        if *r > 122 && *r < 187 && *g >= 154 && *g < 162 && *b == 247
                )
            }),
            "tail should blend toward the secondary accent: {colors:?}"
        );

        let later = working_shimmer_line("Working... (Esc)", 0.7);
        let later_colors: Vec<_> = later.spans.iter().map(|span| span.style.fg).collect();
        assert_ne!(colors, later_colors, "the band must move over time");
    }

    #[test]
    fn sanitize_terminal_text_strips_escape_sequences_and_control_bytes() {
        assert_eq!(
            sanitize_terminal_text("ok\u{1b}[31mred\u{1b}[0m\r\nplain"),
            "okred\nplain"
        );
        assert_eq!(sanitize_terminal_text("a\u{7}b\u{0}c"), "abc");
        assert_eq!(
            sanitize_terminal_text("x\u{fffd}\u{fffd}\u{fffd}y"),
            "x\u{fffd}y"
        );
        assert_eq!(sanitize_terminal_text("tab\there"), "tab\there");
    }

    #[test]
    fn clicking_the_landing_card_focuses_the_input() {
        let mut app = app();
        app.input_focused = false;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let card = app
            .hit_regions
            .iter()
            .find(|region| region.target == HitTarget::Input)
            .expect("landing card registers an input hit area")
            .area;

        handle_mouse_event(
            mouse(
                crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
                card.x + 2,
                card.y + 1,
            ),
            &mut app,
            false,
        );
        assert!(app.input_focused);
    }

    #[test]
    #[ignore = "visual smoke dump; run with --ignored --nocapture"]
    fn visual_dump() {
        let dump = |app: &mut App, w: u16, h: u16| {
            let backend = TestBackend::new(w, h);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal.draw(|frame| render(frame, app)).unwrap();
            format!("{}", terminal.backend())
        };

        let mut landing = configured_app();
        landing.working_dir = PathBuf::from("D:/code/Zex");
        println!("=== LANDING 80x24 ===\n{}", dump(&mut landing, 80, 24));

        let mut work = configured_app();
        work.working_dir = PathBuf::from("D:/code/Zex");
        work.git_status = Some(super::GitStatus {
            branch: "main".to_owned(),
            commit: "9b2995d".to_owned(),
            dirty_count: 1,
        });
        work.tokens_per_second = Some(21.5);
        work.context_chars = 1800;
        work.start_turn();
        work.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "当前目录有哪些文件".to_owned(),
        });
        work.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "我会先查看当前工作目录中的条目\n- 先列目录\n- 再总结".to_owned(),
        });
        work.apply_agent_event(AgentEvent::ToolStart {
            call_id: "c1".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"dir /b /a"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        work.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "c1".to_owned(),
            name: "bash".to_owned(),
            output: "exit_code: 0\nstdout:\n.git/\nsrc/\nCargo.toml\nstderr:\n".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(19),
        });
        work.apply_agent_event(AgentEvent::ToolStart {
            call_id: "c2".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"pwd && find . -name '*.rs'"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        work.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "c2".to_owned(),
            name: "bash".to_owned(),
            output: "exit_code: 1\nstdout:\n\nstderr:\n'pwd' 不是内部或外部命令\n".to_owned(),
            is_error: true,
            elapsed: Duration::from_millis(20),
        });
        work.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "当前目录是 D:\\code\\Zex".to_owned(),
        });
        work.apply_agent_event(AgentEvent::TurnEnd);
        println!("=== WORK 100x26 ===\n{}", dump(&mut work, 100, 26));

        if let Some(super::TranscriptEntry::Tool(tool)) = work.transcript.get_mut(2) {
            tool.expanded = true;
        }
        if let Some(super::TranscriptEntry::Tool(tool)) = work.transcript.get_mut(3) {
            tool.expanded = true;
        }
        println!("=== EXPANDED 100x30 ===\n{}", dump(&mut work, 100, 30));

        let mut busy = configured_app();
        busy.apply_agent_event(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "go".to_owned(),
        });
        busy.start_turn();
        println!("=== BUSY 90x18 ===\n{}", dump(&mut busy, 90, 18));

        println!("=== NARROW 38x12 ===\n{}", dump(&mut work, 38, 12));
    }

    #[test]
    fn garbled_tool_output_is_cleaned_before_it_reaches_the_screen() {
        let mut app = app();
        app.apply_agent_event(AgentEvent::ToolStart {
            call_id: "call-garbled".to_owned(),
            name: "bash".to_owned(),
            arguments: r#"{"command":"dir"}"#.to_owned(),
            timeout: Duration::from_secs(30),
        });
        app.apply_agent_event(AgentEvent::ToolEnd {
            call_id: "call-garbled".to_owned(),
            name: "bash".to_owned(),
            output: "exit_code: 0\nstdout:\n\u{1b}[32mfile.txt\u{1b}[0m\r\nstderr:\n".to_owned(),
            is_error: false,
            elapsed: Duration::from_millis(5),
        });
        app.toggle_selected_tool();
        let backend = TestBackend::new(90, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());
        assert!(screen.contains("file.txt"));
        assert!(!screen.contains("[32m"));
        assert!(!screen.contains('\r'));
    }

    #[test]
    fn thinking_visibility_hides_live_and_restored_cards() {
        let messages = vec![Message::Assistant {
            content: "Answer".to_owned(),
            thinking: Some("Saved thinking".to_owned()),
            tool_calls: Vec::new(),
            provider_state: None,
        }];
        let mut app = App::new(
            &messages,
            "test-model".to_owned(),
            None,
            AppContext {
                working_dir: PathBuf::from("."),
                thinking_level: Some(ThinkingLevel::Medium),
                thinking_preference: ThinkingLevel::Medium,
                context_chars: 0,
                max_context_chars: 120_000,
                default_tool_timeout: Duration::from_secs(60),
                show_thinking: false,
                providers: ProviderCatalog::default(),
            },
        );

        assert!(matches!(
            app.transcript.first(),
            Some(TranscriptEntry::Thinking(ThinkingEntry { content, .. }))
                if content == "Saved thinking"
        ));
        app.apply_agent_event(AgentEvent::ThinkingDelta {
            delta: "Live thinking".to_owned(),
        });
        assert!(matches!(
            app.transcript.last(),
            Some(TranscriptEntry::Thinking(ThinkingEntry { content, .. }))
                if content == "Live thinking"
        ));

        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let hidden = format!("{}", terminal.backend());
        assert!(!hidden.contains("Saved thinking"));
        assert!(!hidden.contains("Live thinking"));

        app.set_show_thinking(true);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let shown = format!("{}", terminal.backend());
        assert!(shown.contains("Saved thinking"));
        assert!(shown.contains("Live thinking"));

        app.set_show_thinking(false);
        assert!(matches!(
            app.transcript.first(),
            Some(TranscriptEntry::Thinking(ThinkingEntry { content, .. }))
                if content == "Saved thinking"
        ));
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

        assert_eq!(app.selected_card, Some(0));
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
        assert!(populated.contains("Session index"));
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
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::User,
            content: "resumed session".to_owned(),
        });
        let backend = TestBackend::new(120, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("cafebabe"));
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
                thinking: None,
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
    fn main_area_pages_restore_the_exact_timeline_scroll_position() {
        let mut app = configured_app();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Keep this timeline position.".to_owned(),
        });
        app.scroll_top = 7;
        app.max_scroll = 30;
        app.follow_output = false;

        app.push_command_output(CommandOutput::Help);
        app.cancel_ui_layer();
        assert_eq!(app.scroll_top, 7);
        assert!(!app.follow_output);

        app.open_model_picker();
        app.dismiss_model_picker();
        assert_eq!(app.scroll_top, 7);
        assert!(!app.follow_output);

        app.open_session_picker(Vec::new());
        app.dismiss_session_picker();
        assert_eq!(app.scroll_top, 7);
        assert!(!app.follow_output);
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
    fn dismissed_completion_returns_arrows_to_input_history() {
        let mut app = app();
        app.remember_submission("older task");
        app.input.insert_str("/");
        app.refresh_completion();

        handle_key_event(
            key(
                crossterm::event::KeyCode::Esc,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );
        handle_key_event(
            key(
                crossterm::event::KeyCode::Up,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );

        assert_eq!(app.input.content, "older task");
        assert!(app.history_cursor.is_some());
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
    fn mouse_click_toggles_tool_and_thinking_card_headers() {
        let mut app = app();
        app.transcript.extend([
            TranscriptEntry::Thinking(ThinkingEntry {
                content: "inspect state".to_owned(),
                expanded: false,
            }),
            TranscriptEntry::Tool(super::ToolEntry {
                call_id: "tool-1".to_owned(),
                name: "read".to_owned(),
                arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
                output: "contents".to_owned(),
                status: ToolStatus::Done,
                expanded: false,
                show_full_output: false,
                started_at: None,
                elapsed: Some(Duration::from_millis(10)),
                timeout: Duration::from_secs(30),
            }),
        ]);
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();

        let card_hits = app
            .hit_regions
            .iter()
            .filter_map(|region| match region.target {
                HitTarget::Card(index) => Some((index, region.area)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(card_hits.len(), 2);
        for (index, area) in card_hits {
            assert_eq!(
                handle_mouse_event(
                    mouse(
                        crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left,),
                        area.x,
                        area.y,
                    ),
                    &mut app,
                    false,
                ),
                InputAction::None
            );
            match &app.transcript[index] {
                TranscriptEntry::Thinking(thinking) => assert!(thinking.expanded),
                TranscriptEntry::Tool(tool) => assert!(tool.expanded),
                _ => panic!("expected clickable card"),
            }
        }
    }

    #[test]
    fn mouse_click_selects_then_confirms_completion() {
        let mut app = app();
        app.input.insert_str("/");
        app.refresh_completion();
        let backend = TestBackend::new(100, 28);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let help_index = command_specs()
            .iter()
            .position(|command| command.name == "/help")
            .unwrap();
        let area = app
            .hit_regions
            .iter()
            .find(|region| region.target == HitTarget::Completion(help_index))
            .unwrap()
            .area;

        let click = mouse(
            crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
            area.x,
            area.y,
        );
        assert_eq!(
            handle_mouse_event(click, &mut app, false),
            InputAction::None
        );
        assert_eq!(app.completion.selected, help_index);
        assert_eq!(
            handle_mouse_event(click, &mut app, false),
            InputAction::Submit("/help".to_owned())
        );
    }

    #[test]
    fn mouse_click_selects_then_confirms_model_and_session_rows() {
        let mut app = configured_app();
        app.open_model_picker();
        let backend = TestBackend::new(100, 28);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let area = app
            .hit_regions
            .iter()
            .find(|region| region.target == HitTarget::Model(1))
            .unwrap()
            .area;
        let click = mouse(
            crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
            area.x,
            area.y,
        );
        assert_eq!(
            handle_mouse_event(click, &mut app, false),
            InputAction::None
        );
        assert_eq!(app.model_picker.as_ref().unwrap().selected, 1);
        assert!(matches!(
            handle_mouse_event(click, &mut app, false),
            InputAction::SwitchModel(ModelRef { model_id, .. }) if model_id == "gpt-4.1-mini"
        ));

        app.dismiss_model_picker();
        app.open_session_picker(vec![
            crate::session::SessionSummary {
                id: "20260812-121500-cafebabe".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 1,
                preview: "first".to_owned(),
            },
            crate::session::SessionSummary {
                id: "20260812-131500-deadbeef".to_owned(),
                updated_at: time::OffsetDateTime::UNIX_EPOCH,
                message_count: 2,
                preview: "second".to_owned(),
            },
        ]);
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let area = app
            .hit_regions
            .iter()
            .find(|region| region.target == HitTarget::Session(1))
            .unwrap()
            .area;
        let click = mouse(
            crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
            area.x,
            area.y,
        );
        assert_eq!(
            handle_mouse_event(click, &mut app, false),
            InputAction::None
        );
        assert_eq!(app.session_picker.as_ref().unwrap().selected, 1);
        assert_eq!(
            handle_mouse_event(click, &mut app, false),
            InputAction::Resume("20260812-131500-deadbeef".to_owned())
        );
    }

    #[test]
    fn ctrl_o_batches_card_expansion_and_collapse() {
        let mut app = app();
        app.transcript.extend([
            TranscriptEntry::Thinking(ThinkingEntry {
                content: "inspect state".to_owned(),
                expanded: false,
            }),
            TranscriptEntry::Tool(super::ToolEntry {
                call_id: "tool-1".to_owned(),
                name: "read".to_owned(),
                arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
                output: "contents".to_owned(),
                status: ToolStatus::Done,
                expanded: false,
                show_full_output: false,
                started_at: None,
                elapsed: Some(Duration::from_millis(10)),
                timeout: Duration::from_secs(30),
            }),
        ]);

        let ctrl_o = key(
            crossterm::event::KeyCode::Char('o'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        assert_eq!(
            handle_key_event(ctrl_o, &mut app, false, false),
            InputAction::None
        );
        assert!(matches!(
            &app.transcript[0],
            TranscriptEntry::Thinking(ThinkingEntry { expanded: true, .. })
        ));
        assert!(matches!(
            &app.transcript[1],
            TranscriptEntry::Tool(super::ToolEntry { expanded: true, .. })
        ));

        assert_eq!(
            handle_key_event(ctrl_o, &mut app, false, false),
            InputAction::None
        );
        assert!(matches!(
            &app.transcript[0],
            TranscriptEntry::Thinking(ThinkingEntry {
                expanded: false,
                ..
            })
        ));
        assert!(matches!(
            &app.transcript[1],
            TranscriptEntry::Tool(super::ToolEntry {
                expanded: false,
                ..
            })
        ));
    }

    #[test]
    fn mouse_click_focuses_input_and_statusline_fields_share_actions() {
        let mut app = configured_app();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::User,
            content: "keep the working chrome visible".to_owned(),
        });
        app.input_focused = false;
        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let input = app
            .hit_regions
            .iter()
            .find(|region| region.target == HitTarget::Input)
            .unwrap()
            .area;
        let think = app
            .hit_regions
            .iter()
            .find(|region| region.target == HitTarget::StatusThinking)
            .unwrap()
            .area;

        assert_eq!(
            handle_mouse_event(
                mouse(
                    crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left,),
                    input.x,
                    input.y,
                ),
                &mut app,
                false,
            ),
            InputAction::None
        );
        assert!(app.input_focused);
        assert_eq!(
            handle_mouse_event(
                mouse(
                    crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left,),
                    think.x,
                    think.y,
                ),
                &mut app,
                false,
            ),
            InputAction::Submit("/think off".to_owned())
        );
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let model = app
            .hit_regions
            .iter()
            .find(|region| region.target == HitTarget::StatusModel)
            .unwrap()
            .area;
        assert_eq!(
            handle_mouse_event(
                mouse(
                    crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left,),
                    model.x,
                    model.y,
                ),
                &mut app,
                false,
            ),
            InputAction::None
        );
        assert!(app.model_picker_open());
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
            InputAction::Submit("/think high".to_owned())
        );
    }

    #[test]
    fn footer_updates_model_thinking_session_and_status_without_feed_rows() {
        let mut app = configured_app();
        app.session_id = Some("20260812-121500-cafebabe".to_owned());
        app.thinking_level = Some(ThinkingLevel::Max);
        app.status = Status::RunningTool;
        let transcript_len = app.transcript.len();
        let backend = TestBackend::new(120, 18);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let screen = format!("{}", terminal.backend());

        assert!(screen.contains("gpt-5"));
        assert!(screen.contains("max"));
        assert!(screen.contains("cafebabe"));
        assert!(screen.contains("tool"));
        assert_eq!(app.transcript.len(), transcript_len);
    }

    #[test]
    fn model_picker_navigation_does_not_mutate_the_timeline_scroll() {
        let mut app = configured_app();
        app.transcript.push(TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Keep the selected viewport.".to_owned(),
        });
        app.scroll_top = 11;
        app.max_scroll = 40;
        app.follow_output = false;
        app.open_model_picker();

        handle_key_event(
            key(
                crossterm::event::KeyCode::Down,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );
        handle_key_event(
            key(
                crossterm::event::KeyCode::Esc,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        );

        assert_eq!(app.scroll_top, 11);
        assert!(!app.follow_output);
        assert!(!app.model_picker_open());
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

        assert!(screen.lines().any(|row| {
            row.contains("bash")
                && row.contains("git status")
                && row.contains("ok")
                && row.contains("20ms")
        }));
        assert!(screen.lines().any(|row| {
            row.contains("bash")
                && row.contains("git rev-parse --short HEAD")
                && row.contains("ok")
                && row.contains("21ms")
        }));
        assert!(!screen.contains("timeout 30.0s"));
        assert!(!screen.contains("Ctrl+O"));
        assert!(screen.contains("/sessions"));
        assert!(screen.contains("List saved sessions"));
        assert!(screen.contains("high"));
        assert!(screen.contains("/se"));

        let rows = screen.lines().collect::<Vec<_>>();
        let completion_row = rows
            .iter()
            .position(|row| row.contains("/sessions"))
            .expect("missing completion row");
        let status_row = rows
            .iter()
            .position(|row| row.contains("ZEX") && row.contains("high"))
            .expect("missing status row");
        let keymap_row = rows
            .iter()
            .position(|row| row.contains("↑↓ select"))
            .expect("missing keymap row");
        let input_row = rows
            .iter()
            .rposition(|row| row.contains("/se"))
            .expect("missing input row");
        assert!(completion_row < status_row);
        assert!(completion_row < input_row);
        assert!(input_row < keymap_row);
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
                show_thinking: true,
                providers: ProviderCatalog::default(),
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
                show_thinking: true,
                providers: ProviderCatalog::default(),
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
        assert!(screen.contains("1 lines"));
        assert!(!screen.contains("package zex"));
        assert!(screen.lines().any(|row| {
            row.contains("read")
                && row.contains("Cargo.toml")
                && row.contains("1 lines")
                && row.contains("14ms")
        }));
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
        assert!(folded.lines().any(|row| {
            row.contains("bash")
                && row.contains("git status --short --branch")
                && row.contains("exit 0")
                && row.contains("18ms")
        }));
        assert_eq!(folded.matches("## main...origin/main [ahead 1]").count(), 0);
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
        assert!(expanded.contains("$ git status --short --branch"));
        assert!(expanded.contains("M src/tui.rs"));
        assert!(expanded.contains("2 lines"));
        assert!(!expanded.contains("input"));
        assert!(!expanded.contains("output {"));
        assert!(!expanded.contains("exit_code:"));
        assert!(!expanded.contains("stdout:"));
        assert!(!expanded.contains("stderr:"));
        assert!(!expanded.contains("timeout_seconds"));
        assert!(!expanded.contains("timeout 30.0s"));
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
        assert!(screen.lines().any(|row| {
            row.contains("bash")
                && row.contains("git status --porcelain")
                && row.contains("exit 0")
                && row.contains("9ms")
        }));
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
        assert!(screen.contains("Working..."));
        assert!(screen.contains("Esc stop"));
        assert!(screen.lines().any(|row| {
            row.contains("bash") && row.contains("git status") && row.contains("running")
        }));
        assert!(screen.contains("running"));
    }
}
