use std::{path::PathBuf, time::Duration};

use crossterm::event::{KeyEvent, KeyEventKind, KeyEventState};
use ratatui::{Terminal, backend::TestBackend, style::Color};

use super::glyphs::{HERO_BOX_MIN_WIDTH, LogoTier, logo_tier};
use super::{
    App, AppContext, CommandOutput, HitTarget, InputAction, InputBuffer, KeyBurst, PRODUCT_NAME,
    ProviderPane, SCROLL_STEP, Status, ThinkingEntry, ToolStatus, TranscriptEntry, command_specs,
    composer_editor_origin, handle_key_event, handle_mouse_event, handle_terminal_event,
    input_metrics, landing_regions, regions_inside_frame, regions_non_overlapping, render,
    sanitize_terminal_text, theme::theme, truncate_chars, ui_regions,
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
    assert!(format!("{}", terminal.backend()).contains("64.0/s"));
    assert_no_banned_branding(&format!("{}", terminal.backend()));
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
    assert!(wide.contains("feature/statusline-polish") || wide.contains("Zex"));
    assert!(wide.contains("*3"));
    assert!(wide.contains("42.7/s"));
    assert!(wide.contains("49%"));
    assert!(wide.contains(" · "));
    assert_no_banned_branding(&wide);

    let backend = TestBackend::new(42, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let narrow = format!("{}", terminal.backend());
    assert!(narrow.contains("high"));
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
    let (text_x, text_y) = composer_editor_origin(content_inset(regions.footer));
    let prefix = terminal.backend().buffer()[(text_x.saturating_sub(2), text_y)].symbol();
    assert!(
        prefix == "\u{276F}" || prefix == "❯" || prefix == ">",
        "expected prompt arrow, got {prefix:?}"
    );
    assert!(screen.contains(" · ") || screen.contains("working"));
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
    assert!(!screen.contains("Ask anything"));
    assert!(!screen.contains("Type a message..."));
    assert_no_banned_branding(&screen);

    app.dismiss_model_picker();
    app.open_provider_editor();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    assert!(screen.contains("Providers"));
    assert!(screen.contains("Provider details"));
    assert!(screen.contains("API key"));
    assert!(!screen.contains("secret"));
    assert!(!screen.contains("Ask anything"));
    assert!(!screen.contains("Type a message..."));
    assert_no_banned_branding(&screen);
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
    assert!(screen.contains("think low–xhigh"));
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
    assert!(screen.contains("think low–max"));
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
fn empty_state_centers_a_large_zex_wordmark_and_focused_prompt_surface() {
    let mut app = configured_app();
    app.working_dir = PathBuf::from("D:/workspaces/zex");
    app.git_status = Some(super::GitStatus {
        branch: "main".to_owned(),
        commit: "a1b2c3d".to_owned(),
        dirty_count: 0,
    });
    app.thinking_level = Some(ThinkingLevel::High);
    app.context_chars = 30_000;
    let backend = TestBackend::new(104, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    write_screen_dump("landing", &screen);
    let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 104, 24), &app);

    assert_zex_welcome_identity(&screen);
    assert!(!screen.contains("Ask anything"));
    assert!(!screen.contains("Type a message..."));
    assert!(screen.contains("D:/workspaces/zex") || screen.contains("workspaces/zex"));
    assert!(!screen.contains("● idle"));
    assert_eq!(style_at(&terminal, 0, 0).bg, Some(theme().background));
    assert_eq!(theme().background, Color::Reset);
    assert!(
        !regions.hero.is_empty(),
        "width 104 should use the hero box"
    );
    assert!(regions.menu.height >= 2);
    assert!(regions.card.bottom() <= regions.status.y || regions.status.is_empty());
    assert_eq!(
        terminal.backend().buffer()[(regions.card.x, regions.card.y)].symbol(),
        "╭"
    );
    assert_eq!(
        terminal.backend().buffer()[(regions.card.x, regions.card.y + 1)].symbol(),
        "│"
    );
    let prefix = terminal.backend().buffer()[(regions.card.x + 2, regions.card.y + 1)].symbol();
    assert!(
        prefix == "\u{276F}" || prefix == "❯" || prefix == ">",
        "expected prompt arrow, got {prefix:?}"
    );
    assert!(
        screen.contains(env!("CARGO_PKG_VERSION")),
        "missing version on welcome badge"
    );
}

#[test]
fn focused_empty_input_keeps_the_ime_preedit_region_clear() {
    let mut app = app();
    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 80, 16), &app);
    let (editor_x, editor_y) = composer_editor_origin(regions.card);

    assert!(!screen.contains("Ask anything"));
    assert!(!screen.contains("Type a message..."));
    terminal
        .backend_mut()
        .assert_cursor_position((editor_x, editor_y));
    assert_eq!(
        terminal.backend().buffer()[(editor_x, editor_y)].symbol(),
        " ",
        "IME preedit text needs an empty row so no placeholder suffix remains after it"
    );
}

#[test]
fn unfocused_empty_input_displays_the_placeholder() {
    let mut app = app();
    app.input_focused = false;
    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();

    assert!(format!("{}", terminal.backend()).contains("Type a message..."));
    assert!(!format!("{}", terminal.backend()).contains("Ask anything"));
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
    let (editor_x, editor_y) = composer_editor_origin(regions.card);

    assert!(screen.contains("hello"));
    assert!(!screen.contains("Ask anything"));
    assert!(!screen.contains("Type a message..."));
    terminal
        .backend_mut()
        .assert_cursor_position((editor_x + 5, editor_y));
}

#[test]
fn unfocused_landing_keeps_the_cursor_anchored_in_the_composer() {
    // With the landing visible the ambient animations repaint every frame, so
    // a hidden cursor would be left parked at the last-diffed cell (the hero
    // box corner), where Windows IME then anchors its preedit composition.
    let mut app = app();
    app.input_focused = false;
    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 80, 16), &app);
    let (editor_x, editor_y) = composer_editor_origin(regions.card);

    terminal
        .backend_mut()
        .assert_cursor_position((editor_x, editor_y));
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
        assert!(regions.top.bottom() <= area.bottom());
        assert!(regions.logo.bottom() <= area.bottom());
        assert!(regions.hero.bottom() <= area.bottom());
        assert!(regions.card.bottom() <= area.bottom());
        assert!(regions.menu.bottom() <= area.bottom());
        assert!(regions.status.bottom() <= area.bottom());
        assert!(regions.completion.bottom() <= area.bottom());
        assert_eq!(style_at(&terminal, 0, 0).bg, Some(theme().background));
    }
}

#[test]
fn work_content_uses_contiguous_panel_backgrounds() {
    let mut app = app();
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::User,
        delta: "Inspect this repository".to_owned(),
    });
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::Assistant,
        delta: "I will check the project.\n- read the manifest\n- summarize it".to_owned(),
    });
    app.apply_agent_event(AgentEvent::ToolStart {
        call_id: "call-panel".to_owned(),
        name: "read".to_owned(),
        arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
        timeout: Duration::from_secs(30),
    });
    app.apply_agent_event(AgentEvent::ToolEnd {
        call_id: "call-panel".to_owned(),
        name: "read".to_owned(),
        output: "line one\nline two".to_owned(),
        is_error: false,
        elapsed: Duration::from_millis(8),
    });
    let TranscriptEntry::Tool(tool) = &mut app.transcript[2] else {
        panic!("expected tool");
    };
    tool.expanded = true;

    let backend = TestBackend::new(96, 28);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());

    for needle in [
        "Inspect this repository",
        "I will check the project.",
        "line one",
    ] {
        let row = screen
            .lines()
            .position(|line| line.contains(needle))
            .expect("content row should be visible") as u16;
        let background = style_at(&terminal, super::HORIZONTAL_GUTTER + 4, row).bg;
        assert!(matches!(
            background,
            Some(color) if [theme().background, theme().surface, theme().surface_raised].contains(&color)
        ));
        let _ = background;
        for x in super::HORIZONTAL_GUTTER..96 - super::HORIZONTAL_GUTTER {
            assert_eq!(
                style_at(&terminal, x, row).bg,
                background,
                "panel background broke at x={x}, row={row}"
            );
        }
    }
}

#[test]
fn footer_is_a_full_input_band_in_idle_and_busy_states() {
    let mut app = app();
    app.transcript.push(TranscriptEntry::Message {
        role: MessageRole::Assistant,
        content: "Ready.".to_owned(),
    });
    let area = ratatui::layout::Rect::new(0, 0, 100, 20);
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let idle_regions = ui_regions(area, &app);
    let footer = content_inset(idle_regions.footer);
    assert_eq!(
        terminal.backend().buffer()[(footer.x, footer.y)].symbol(),
        "╭"
    );
    assert_eq!(
        terminal.backend().buffer()[(footer.x, footer.y + 1)].symbol(),
        "│"
    );
    let idle = format!("{}", terminal.backend());
    assert!(idle.contains("test-model"));
    assert!(idle.contains(" · "));
    assert!(idle.contains("enter") && idle.contains("send"));
    assert!(idle.contains("│"));

    app.start_turn();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let busy = format!("{}", terminal.backend());
    let busy_regions = ui_regions(area, &app);
    let busy_footer = content_inset(busy_regions.footer);
    assert_eq!(
        terminal.backend().buffer()[(busy_footer.x, busy_footer.y)].symbol(),
        "╭"
    );
    assert!(busy.contains("interrupt"));
    assert!(busy.contains("thinking"));
}

#[test]
fn busy_input_band_preserves_metadata_and_editor_context() {
    let mut app = app();
    app.transcript.push(TranscriptEntry::Message {
        role: MessageRole::User,
        content: "Inspect the project".to_owned(),
    });
    app.start_turn();
    let area = ratatui::layout::Rect::new(0, 0, 90, 18);
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());

    assert!(screen.contains("test-model"));
    assert!(screen.contains("thinking"));
    assert!(screen.contains("interrupt"));
}

#[test]
fn browse_mode_uses_the_complete_navigation_footer() {
    let mut app = app();
    app.transcript.push(TranscriptEntry::Tool(super::ToolEntry {
        call_id: "tool-1".to_owned(),
        name: "read".to_owned(),
        arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
        output: "manifest".to_owned(),
        status: ToolStatus::Done,
        expanded: false,
        show_full_output: false,
        started_at: None,
        elapsed: Some(Duration::from_millis(10)),
        timeout: Duration::from_secs(30),
    }));
    app.selected_entry = Some(0);
    app.input_focused = false;
    let backend = TestBackend::new(100, 18);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());

    assert!(screen.contains("browse"));
    assert!(screen.contains("open"));
    assert!(screen.contains("compose"));
    assert!(screen.contains("clear"));
    assert!(screen.contains("│"));
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
                matches!(
                    cell.style().bg,
                    Some(color) if [theme().background, theme().surface, theme().surface_raised, theme().code_bg].contains(&color)
                ),
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
                Some(color) if [theme().background, theme().surface, theme().surface_raised, theme().code_bg].contains(&color)
            ));
        }
    }
}

#[test]
fn default_palette_uses_shipped_rgb_values() {
    assert_eq!(theme().background, Color::Reset);
    assert_eq!(theme().text, Color::Rgb(212, 215, 221));
    assert_eq!(theme().surface, Color::Rgb(29, 33, 41));
    assert_eq!(theme().surface_raised, Color::Rgb(49, 54, 63));
    assert_eq!(theme().accent_primary, Color::Rgb(103, 232, 249));
    assert_eq!(theme().accent_secondary, Color::Rgb(178, 129, 214));
    assert_eq!(theme().command, Color::Rgb(254, 188, 56));
    assert_eq!(theme().bad, Color::Rgb(252, 58, 75));
    assert_ne!(theme().accent_primary, Color::Rgb(56, 189, 248));
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
    assert!(!screen.contains("Ask anything"));
    assert!(screen.contains("thinking"));
    assert!(!screen.contains("precision, without noise"));
    // The user band leads with a blank padding row, so the message itself
    // sits one row below the transcript top.
    assert_eq!(message_row, regions.transcript.y + 1);
    assert_no_banned_branding(&screen);
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
    assert!(screen.contains(PRODUCT_NAME));
    assert!(!screen.contains("precision, without noise"));
    assert!(!screen.contains("Ask anything"));
    assert!(!screen.contains("Inspect the project"));
    assert_no_banned_branding(&screen);
}

#[test]
fn code_blocks_use_a_raised_band_and_tools_stay_on_the_base_background() {
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
    for cell in buffer.content() {
        assert!(
            matches!(
                cell.style().bg,
                Some(color) if [theme().background, theme().surface, theme().surface_raised, theme().code_bg].contains(&color)
            ),
            "unexpected background {:?}",
            cell.style().bg
        );
    }

    let screen = format!("{}", terminal.backend());
    assert!(screen.contains("fn main() {}"));
    let code_row = screen
        .lines()
        .position(|row| row.contains("fn main() {}"))
        .expect("code block should be visible") as u16;
    assert_eq!(style_at(&terminal, 40, code_row).bg, Some(theme().code_bg));
    assert!(
        format!("{}", terminal.backend())
            .lines()
            .any(|row| row.contains("Bash") && row.contains("cargo check"))
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
    assert!(
        screen.contains("high") || screen.contains("very-long-model") || screen.contains("enter"),
        "screen:\n{screen}"
    );
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
    assert!(idle.contains("test-model"));
    assert!(idle.contains("enter") && idle.contains("send"));

    app.start_turn();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let thinking = format!("{}", terminal.backend());
    assert!(thinking.contains("thinking"));
    assert!(thinking.contains("interrupt"));

    app.apply_agent_event(AgentEvent::ToolStart {
        call_id: "call-running".to_owned(),
        name: "read".to_owned(),
        arguments: r#"{"path":"Cargo.toml"}"#.to_owned(),
        timeout: Duration::from_secs(30),
    });
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let running = format!("{}", terminal.backend());
    let tool_row = running
        .lines()
        .find(|row| (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml"))
        .expect("missing running tool row");
    // The spinner marks liveness; the meta tail shows live elapsed time.
    assert!(tool_row.contains("ms"), "running tool shows live elapsed");
    assert!(running.contains("interrupt"));
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
    let box_area = content_inset(multiline.footer);
    assert_eq!(
        terminal.backend().buffer()[(box_area.x, box_area.y)].symbol(),
        "╭"
    );
    assert_eq!(
        terminal.backend().buffer()[(box_area.x, box_area.y + 1)].symbol(),
        "│"
    );
}

#[test]
fn typed_text_and_cursor_stay_inside_the_work_input_frame() {
    let mut app = app();
    app.transcript.push(TranscriptEntry::Message {
        role: MessageRole::Assistant,
        content: "Ready.".to_owned(),
    });
    app.input.insert_str("fixed draft");
    let area = ratatui::layout::Rect::new(0, 0, 80, 16);
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();

    let regions = ui_regions(area, &app);
    let (input_x, input_y) = composer_editor_origin(content_inset(regions.footer));
    assert_eq!(
        terminal.backend().buffer()[(input_x, input_y)].symbol(),
        "f"
    );
    terminal
        .backend_mut()
        .assert_cursor_position((input_x + "fixed draft".len() as u16, input_y));
    assert!(input_y < regions.footer.bottom());
}

#[test]
fn work_input_background_and_text_share_the_block_inner_rect() {
    let mut app = app();
    app.transcript.push(TranscriptEntry::Message {
        role: MessageRole::Assistant,
        content: "Ready.".to_owned(),
    });
    app.input.insert_str("align");
    let area = ratatui::layout::Rect::new(0, 0, 80, 16);
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();

    let footer = ui_regions(area, &app).footer;
    let frame_area = content_inset(footer);
    assert_eq!(
        terminal.backend().buffer()[(frame_area.x, frame_area.y)].symbol(),
        "╭"
    );
    assert_eq!(
        terminal.backend().buffer()[(frame_area.right() - 1, frame_area.y)].symbol(),
        "╮"
    );
    let (text_x, text_y) = composer_editor_origin(frame_area);
    assert_eq!(terminal.backend().buffer()[(text_x, text_y)].symbol(), "a");
    assert_eq!(
        style_at(&terminal, text_x, text_y).bg,
        Some(theme().background),
        "text cell must sit inside the prompt box, not a slab"
    );
}

#[test]
fn narrow_metadata_never_overflows_the_input_surface() {
    let mut app = configured_app();
    app.model = "provider/very-long-model-name-that-needs-truncation".to_owned();
    app.transcript.push(TranscriptEntry::Message {
        role: MessageRole::Assistant,
        content: "Ready.".to_owned(),
    });
    let area = ratatui::layout::Rect::new(0, 0, 30, 12);
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();

    let footer = ui_regions(area, &app).footer;
    let frame_area = content_inset(footer);
    assert!(
        matches!(
            terminal.backend().buffer()[(frame_area.x, frame_area.y)].symbol(),
            "╭" | "│" | "╰"
        ),
        "narrow composer lost its box chrome"
    );
    assert!(
        matches!(
            terminal.backend().buffer()[(frame_area.right() - 1, frame_area.y)].symbol(),
            "╮" | "│" | "╯"
        ),
        "narrow composer overflowed its right edge"
    );
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
            regions.footer.y,
        )]
            .symbol(),
        "╭"
    );
    let selected_row = regions.completion.y;
    assert!((regions.completion.x..regions.completion.right()).any(|x| {
        let style = style_at(&terminal, x, selected_row);
        style.fg == Some(theme().accent_primary) || style.bg == Some(theme().surface_raised)
    }));
    let screen = format!("{}", terminal.backend());
    assert!(screen.contains("/help") || screen.contains("/"));
    assert_no_banned_branding(&screen);
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
                ..
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
    assert!(folded.contains("Thought") || folded.contains("Thinking"));
    assert!(!folded.contains("Inspect constraints first."));
    let thinking_row = folded
        .lines()
        .position(|row| row.contains("Thought") || row.contains("Thinking"))
        .expect("thinking row should be visible") as u16;
    assert!((0..100).any(|x| {
        let bg = style_at(&terminal, x, thinking_row).bg;
        matches!(bg, Some(color) if [theme().background, theme().surface, theme().surface_raised].contains(&color))
    }));

    app.select_timeline_entry(false);
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
            .filter(|row| row.contains("Thought") || row.contains("Thinking"))
            .count(),
        1
    );
    assert_eq!(
        rows.iter()
            .filter(
                |row| (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml")
            )
            .count(),
        1
    );
}

#[test]
fn completed_turn_renders_in_order_without_status_noise() {
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
    let user_row = screen
        .lines()
        .position(|row| row.contains("Inspect the project"))
        .expect("missing user message");
    let tool_row = screen
        .lines()
        .position(|row| {
            (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml")
        })
        .expect("missing tool summary");
    let answer_row = screen
        .lines()
        .position(|row| row.contains("Final answer."))
        .expect("missing final answer");
    let rail = terminal.backend().buffer()[(super::HORIZONTAL_GUTTER, user_row as u16)].symbol();
    assert!(
        rail == "\u{276F}" || rail == "❯" || rail == ">",
        "expected user prompt arrow, got {rail:?}"
    );
    assert!(user_row < tool_row);
    assert!(tool_row < answer_row);
    // Turn bookkeeping stays out of the transcript: no status noise rows.
    assert!(!screen.contains("turn done"));
    assert!(!screen.contains("1.2k"));
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
    assert!(
        screen
            .lines()
            .any(|row| row.contains("working") && row.contains("1 tool"))
    );
    assert!(screen.contains("working"));
}

#[test]
fn active_turn_status_surface_ends_after_its_content() {
    let mut app = app();
    app.start_turn();
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::User,
        delta: "Inspect".to_owned(),
    });
    let area = ratatui::layout::Rect::new(0, 0, 120, 32);
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    let detail_row = screen
        .lines()
        .position(|row| row.contains("thinking"))
        .expect("running detail should be visible") as u16;
    let regions = ui_regions(area, &app);
    assert!(
        detail_row >= regions.turn_status.y || detail_row < regions.transcript.bottom(),
        "turn status should sit in the turn-status row or transcript"
    );
    assert!(screen.contains("thinking"));
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

    assert!(
        screen
            .lines()
            .any(|row| { row.contains("Bash") && row.contains("git status") })
    );
    assert!(
        screen
            .lines()
            .any(|row| { row.contains("Grep") && row.contains("render_footer") })
    );
    let tool_row = screen
        .lines()
        .position(|row| row.contains("Bash") && row.contains("git status"))
        .expect("tool row should be visible") as u16;
    assert!((0..110).any(|x| {
        let bg = style_at(&terminal, x, tool_row).bg;
        matches!(bg, Some(color) if [theme().background, theme().surface, theme().surface_raised].contains(&color))
    }));
}

#[test]
fn tool_cards_keep_verb_and_subject_on_one_line() {
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
    assert!(
        rows.iter()
            .any(|row| row.contains("Bash") && row.contains("pwd"))
    );
    assert!(rows.iter().any(|row| row.contains("Glob")));
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
    let name_byte = row_text.find("Bash").unwrap();
    let status_x = unicode_width::UnicodeWidthStr::width(&row_text[..status_byte]) as u16;
    let name_x = unicode_width::UnicodeWidthStr::width(&row_text[..name_byte]) as u16;

    assert_eq!(style_at(&terminal, status_x, row).fg, Some(theme().bad));
    assert_ne!(style_at(&terminal, name_x, row).fg, Some(theme().bad));
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
    assert!(preview.contains("long.txt"));
    assert!(preview.contains("20 lines"));
    assert!(!preview.contains("\"path\""));
    assert!(preview.contains("line 12"));
    assert!(!preview.contains("line 13"));
    assert!(preview.contains("… 8 more · ctrl+o"));

    app.toggle_selected_tool_output();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let full = format!("{}", terminal.backend());
    assert!(full.contains("line 20"));
    assert!(full.contains("ctrl+o to collapse"));
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
    assert!(screen.contains("Edit"));
    assert!(screen.contains("src/tui.rs"));
    assert!(screen.contains("+2 −1"));
    assert!(screen.contains("-     old();"));
    assert!(screen.contains("+     new();"));
    assert!(screen.contains("+     finish();"));
    assert!(!screen.contains("\"old_text\""));
    assert!(!screen.contains("\"new_text\""));
}

#[test]
fn failed_bash_card_shows_exit_code_and_output() {
    let mut app = app();
    app.apply_agent_event(AgentEvent::ToolStart {
        call_id: "call-fail".to_owned(),
        name: "bash".to_owned(),
        arguments: r#"{"command":"pwd && find . -name '*.rs'","timeout_seconds":30}"#.to_owned(),
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
    // The header carries the command; the expanded body shows raw output
    // without labeled command/stderr sections.
    assert!(expanded.contains("pwd && find . -name '*.rs'"));
    assert!(expanded.contains("'pwd' is not recognized"));
    assert!(!expanded.contains("exit_code:"));
    assert!(!expanded.contains("stdout:"));
    assert!(!expanded.contains("timeout_seconds"));
    assert!(!expanded.contains('{'));
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

fn content_inset(area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let gutter = super::HORIZONTAL_GUTTER.min(area.width / 2);
    ratatui::layout::Rect::new(
        area.x + gutter,
        area.y,
        area.width.saturating_sub(gutter.saturating_mul(2)),
        area.height,
    )
}

fn draw(app: &mut App, width: u16, height: u16) -> (Terminal<TestBackend>, String) {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, app)).unwrap();
    let screen = format!("{}", terminal.backend());
    (terminal, screen)
}

fn assert_no_banned_branding(screen: &str) {
    let lower = screen.to_ascii_lowercase();
    for banned in ["grok", "x.ai", "supergrok"] {
        assert!(
            !lower.contains(banned),
            "painted copy leaked banned branding {banned:?}:\n{screen}"
        );
    }
}

fn assert_zex_welcome_identity(screen: &str) {
    let lower = screen.to_ascii_lowercase();
    assert!(
        screen.contains(PRODUCT_NAME),
        "missing ZEX product name:\n{screen}"
    );
    assert!(
        screen.contains('\u{276F}') || screen.contains('❯'),
        "missing composer prefix:\n{screen}"
    );
    assert!(
        !screen.contains('›'),
        "old composer prefix still painted:\n{screen}"
    );
    assert!(
        screen.contains(env!("CARGO_PKG_VERSION")),
        "missing package version:\n{screen}"
    );
    assert!(
        screen.contains("Resume sessions"),
        "missing resume-sessions menu row:\n{screen}"
    );
    assert!(screen.contains("Quit"), "missing quit menu row:\n{screen}");
    assert_no_banned_branding(screen);
    for banned in [
        "worktree",
        "import claude",
        "precision, without noise",
        "upgrade subscription",
        "login with",
        "logged in with api key",
        "ask anything",
    ] {
        assert!(
            !lower.contains(banned),
            "welcome leaked banned copy {banned:?}:\n{screen}"
        );
    }
}

fn screen_has_glyph(terminal: &Terminal<TestBackend>, glyph: &str) -> bool {
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .any(|cell| cell.symbol() == glyph)
}

fn screen_has_rgb(terminal: &Terminal<TestBackend>, color: Color) -> bool {
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .any(|cell| cell.style().bg == Some(color) || cell.style().fg == Some(color))
}

fn write_screen_dump(name: &str, screen: &str) {
    let Ok(dir) = std::env::var("ZEX_SCREEN_DUMP_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{name}.txt")), screen).unwrap();
}

fn assert_no_role_labels(screen: &str) {
    assert!(!screen.contains("YOU"));
    assert!(!screen.contains("ASSISTANT"));
    assert!(!screen.lines().any(|row| row.trim() == "you"));
    assert!(!screen.lines().any(|row| row.trim() == "assistant"));
}

fn assert_buffer_in_bounds(terminal: &Terminal<TestBackend>, width: u16, height: u16) {
    let buffer = terminal.backend().buffer();
    for y in 0..height {
        for x in 0..width {
            let symbol = buffer[(x, y)].symbol();
            let glyph_width = unicode_width::UnicodeWidthStr::width(symbol) as u16;
            assert!(
                x.saturating_add(glyph_width) <= width || glyph_width == 0,
                "glyph {:?} at ({x},{y}) paints past width {width}",
                symbol
            );
        }
    }
}

fn assert_work_regions(app: &App, width: u16, height: u16) {
    let area = ratatui::layout::Rect::new(0, 0, width, height);
    let regions = ui_regions(area, app);
    assert!(
        regions_inside_frame(area, &regions),
        "a workspace region left the {width}x{height} frame"
    );
    assert!(
        regions_non_overlapping(&regions),
        "workspace regions overlap at {width}x{height}"
    );
}

fn two_turn_conversation() -> App {
    let mut app = configured_app();
    app.working_dir = PathBuf::from("D:/code/Zex");
    app.git_status = Some(super::GitStatus {
        branch: "main".to_owned(),
        commit: "9b2995d".to_owned(),
        dirty_count: 1,
    });
    app.tokens_per_second = Some(21.5);
    app.context_chars = 1_800;
    app.start_turn();
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::User,
        delta: "当前目录有哪些文件".to_owned(),
    });
    app.apply_agent_event(AgentEvent::ThinkingDelta {
        delta: "先列目录再总结当前工作区。".to_owned(),
    });
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::Assistant,
        delta: "我会先查看当前工作目录中的条目".to_owned(),
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
        elapsed: Duration::from_millis(14),
    });
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::Assistant,
        delta: "当前目录是 D:\\code\\Zex".to_owned(),
    });
    app.apply_agent_event(AgentEvent::TurnEnd);
    app
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

    let mut sessions = configured_app();
    sessions.open_session_picker(vec![
        crate::session::SessionSummary {
            id: "20260813-120000-cafebabe".to_owned(),
            updated_at: time::OffsetDateTime::UNIX_EPOCH + Duration::from_secs(3_600),
            message_count: 12,
            preview: "Polish the TUI timeline and input band".to_owned(),
        },
        crate::session::SessionSummary {
            id: "20260812-121500-deadbeef".to_owned(),
            updated_at: time::OffsetDateTime::UNIX_EPOCH,
            message_count: 4,
            preview: "Review provider configuration".to_owned(),
        },
    ]);
    println!("=== SESSIONS 100x24 ===\n{}", dump(&mut sessions, 100, 24));

    sessions.open_session_picker(Vec::new());
    println!(
        "=== SESSIONS EMPTY 80x20 ===\n{}",
        dump(&mut sessions, 80, 20)
    );
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
    assert!(!hidden.contains("Thought"));

    app.set_show_thinking(true);
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let shown = format!("{}", terminal.backend());
    assert!(shown.contains("Thought") || shown.contains("Thinking"));
    assert!(!shown.contains("Saved thinking"));
    assert!(!shown.contains("Live thinking"));

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
fn final_answer_selection_opens_and_closes_the_output_panel() {
    let mut app = app();
    app.transcript.extend([
        TranscriptEntry::Turn(super::TurnEntry {
            outcome: super::TurnOutcome::Done,
            model: "test-model".to_owned(),
            thinking: ThinkingLevel::Medium,
            tool_count: 0,
            elapsed: Some(Duration::from_secs(1)),
            output_tokens: Some(12),
        }),
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Complete answer.\n```rust\nfn main() {}\n```".to_owned(),
        },
    ]);

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
    assert_eq!(app.selected_entry, Some(1));
    assert!(!app.input_focused);
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
        InputAction::None
    );
    assert!(app.output_panel_open());
    assert_eq!(app.output_panel.as_ref().unwrap().entry_index, 1);

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
    assert!(!app.output_panel_open());
    assert_eq!(app.selected_entry, Some(1));
    assert!(!app.input_focused);
}

#[test]
fn only_final_assistant_answers_are_openable() {
    let mut app = app();
    app.transcript.extend([
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Intermediate note.".to_owned(),
        },
        TranscriptEntry::Tool(super::ToolEntry {
            call_id: "tool-1".to_owned(),
            name: "read".to_owned(),
            arguments: "{}".to_owned(),
            output: "contents".to_owned(),
            status: ToolStatus::Done,
            expanded: false,
            show_full_output: false,
            started_at: None,
            elapsed: Some(Duration::from_millis(10)),
            timeout: Duration::from_secs(30),
        }),
        TranscriptEntry::Turn(super::TurnEntry {
            outcome: super::TurnOutcome::Done,
            model: "test-model".to_owned(),
            thinking: ThinkingLevel::Medium,
            tool_count: 1,
            elapsed: Some(Duration::from_secs(1)),
            output_tokens: Some(6),
        }),
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Final answer.".to_owned(),
        },
    ]);

    assert!(!app.is_final_answer(0));
    assert!(app.is_final_answer(3));
    assert_eq!(app.selectable_entry_indices(), vec![1, 3]);
}

#[test]
fn space_focuses_input_from_browse_mode_without_inserting_text() {
    let mut app = app();
    app.transcript.extend([
        TranscriptEntry::Turn(super::TurnEntry {
            outcome: super::TurnOutcome::Done,
            model: "test-model".to_owned(),
            thinking: ThinkingLevel::Medium,
            tool_count: 0,
            elapsed: None,
            output_tokens: None,
        }),
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Final answer.".to_owned(),
        },
    ]);
    app.selected_entry = Some(1);
    app.input_focused = false;

    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Char(' '),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        ),
        InputAction::None
    );
    assert!(app.input_focused);
    assert_eq!(app.selected_entry, None);
    assert!(app.input.is_empty());

    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Char(' '),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        ),
        InputAction::None
    );
    assert_eq!(app.input.content, " ");
}

#[test]
fn space_does_not_steal_focus_from_busy_or_page_layers() {
    let mut busy = app();
    busy.input_focused = false;
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Char(' '),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut busy,
            true,
            false,
        ),
        InputAction::None
    );
    assert!(!busy.input_focused);
    assert!(busy.input.is_empty());

    let mut picker = configured_app();
    picker.open_model_picker();
    assert!(matches!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Char(' '),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut picker,
            false,
            false,
        ),
        InputAction::SwitchModel(_)
    ));
    assert!(!picker.input_focused);
    assert!(picker.input.is_empty());
}

#[test]
fn output_panel_owns_space_and_scroll_until_escape() {
    let mut app = app();
    app.transcript.extend([
        TranscriptEntry::Turn(super::TurnEntry {
            outcome: super::TurnOutcome::Done,
            model: "test-model".to_owned(),
            thinking: ThinkingLevel::Medium,
            tool_count: 0,
            elapsed: None,
            output_tokens: None,
        }),
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: (0..80)
                .map(|index| format!("line {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
        },
    ]);
    app.selected_entry = Some(1);
    app.open_selected_output();

    let backend = TestBackend::new(90, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let initial = format!("{}", terminal.backend());
    assert!(initial.contains("Response"));
    assert!(initial.contains("esc"));
    assert!(initial.contains("line 0"));
    assert!(app.output_panel.as_ref().unwrap().max_scroll > 0);

    handle_key_event(
        key(
            crossterm::event::KeyCode::Char(' '),
            crossterm::event::KeyModifiers::NONE,
        ),
        &mut app,
        false,
        false,
    );
    assert!(app.output_panel.as_ref().unwrap().scroll_top > 0);
    assert!(!app.input_focused);
    assert!(app.input.is_empty());

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let scrolled = format!("{}", terminal.backend());
    assert!(!scrolled.contains("line 0"));
}

#[test]
fn output_panel_mouse_wheel_scrolls_and_close_hit_restores_browse_mode() {
    let mut app = app();
    app.transcript.extend([
        TranscriptEntry::Turn(super::TurnEntry {
            outcome: super::TurnOutcome::Done,
            model: "test-model".to_owned(),
            thinking: ThinkingLevel::Medium,
            tool_count: 0,
            elapsed: None,
            output_tokens: None,
        }),
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: (0..80)
                .map(|index| format!("line {index}"))
                .collect::<Vec<_>>()
                .join("\n"),
        },
    ]);
    app.selected_entry = Some(1);
    app.open_selected_output();
    let backend = TestBackend::new(90, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();

    assert_eq!(
        handle_mouse_event(
            mouse(crossterm::event::MouseEventKind::ScrollDown, 20, 10),
            &mut app,
            false,
        ),
        InputAction::None
    );
    assert_eq!(app.output_panel.as_ref().unwrap().scroll_top, SCROLL_STEP);

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let close = app
        .hit_regions
        .iter()
        .find(|region| region.target == HitTarget::OutputClose)
        .expect("output panel registers a close hit area")
        .area;
    assert_eq!(
        handle_mouse_event(
            mouse(
                crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
                close.x,
                close.y,
            ),
            &mut app,
            false,
        ),
        InputAction::None
    );
    assert!(!app.output_panel_open());
    assert_eq!(app.selected_entry, Some(1));
    assert!(!app.input_focused);
}

#[test]
fn clicking_a_final_answer_selects_then_opens_it() {
    let mut app = app();
    app.transcript.extend([
        TranscriptEntry::Turn(super::TurnEntry {
            outcome: super::TurnOutcome::Done,
            model: "test-model".to_owned(),
            thinking: ThinkingLevel::Medium,
            tool_count: 0,
            elapsed: None,
            output_tokens: None,
        }),
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "Open this answer.".to_owned(),
        },
    ]);
    let backend = TestBackend::new(90, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let response = app
        .hit_regions
        .iter()
        .find(|region| region.target == HitTarget::Response(1))
        .expect("final answer registers a hit area")
        .area;

    for expected_open in [false, true] {
        assert_eq!(
            handle_mouse_event(
                mouse(
                    crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left,),
                    response.x,
                    response.y,
                ),
                &mut app,
                false,
            ),
            InputAction::None
        );
        assert_eq!(app.output_panel_open(), expected_open);
    }
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
    app.select_timeline_entry(false);
    app.toggle_selected_tool();

    assert_eq!(app.selected_entry, Some(0));
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
    assert!(populated.contains("Session index") || populated.contains("cafebabe"));
    assert!(populated.contains("cafebabe"));
    assert!(!populated.contains("20260812-121500-cafebabe"));
    assert!(populated.contains("1970-01-01"));
    assert!(populated.contains("2 messages"));
    assert!(populated.contains("first saved task"));
    assert!(populated.contains("resume") || populated.contains("enter"));

    app.push_command_output(CommandOutput::ResumePicker(Vec::new()));
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let empty = format!("{}", terminal.backend());
    assert!(empty.contains("No saved sessions"));
    assert!(empty.contains("close"));
}

#[test]
fn resume_picker_paints_complete_two_line_rows_with_a_selected_accent() {
    let mut app = app();
    app.push_command_output(CommandOutput::ResumePicker(vec![
        crate::session::SessionSummary {
            id: "20260812-121500-cafebabe".to_owned(),
            updated_at: time::OffsetDateTime::UNIX_EPOCH,
            message_count: 2,
            preview: "first saved task".to_owned(),
        },
        crate::session::SessionSummary {
            id: "20260812-120000-deadbeef".to_owned(),
            updated_at: time::OffsetDateTime::UNIX_EPOCH,
            message_count: 1,
            preview: "second saved task".to_owned(),
        },
    ]));
    let backend = TestBackend::new(100, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();

    let selected = app
        .hit_regions
        .iter()
        .find(|region| region.target == HitTarget::Session(0))
        .expect("selected session row should be clickable")
        .area;
    assert_eq!(selected.height, 1);
    assert!(
        (selected.x..selected.right()).any(|x| style_at(&terminal, x, selected.y).bg
            == Some(theme().surface_raised)
            || style_at(&terminal, x, selected.y).bg == Some(theme().surface))
    );

    let unselected = app
        .hit_regions
        .iter()
        .find(|region| region.target == HitTarget::Session(1))
        .expect("unselected session row should be clickable")
        .area;
    assert_eq!(unselected.height, 1);
}

#[test]
fn resumed_session_id_does_not_compete_with_the_input_band() {
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

    assert!(!screen.contains("cafebabe"));
    assert!(screen.contains("test-model"));
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
    assert!(folded.contains("ctrl+e"));
    assert!(!folded.contains("socket closed"));

    app.toggle_latest_error();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let expanded = format!("{}", terminal.backend());
    assert!(expanded.contains("socket closed"));
    assert!(expanded.contains("ctrl+e"));
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
            elapsed: None,
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
            elapsed: None,
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
fn inline_markdown_is_rendered_instead_of_raw_markers() {
    let mut app = app();
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::Assistant,
        delta: "**文件**\n- `.gitignore`\n- `Cargo.toml`".to_owned(),
    });
    let (_, screen) = draw(&mut app, 80, 20);
    assert!(screen.contains("文件"));
    assert!(screen.contains(".gitignore"));
    assert!(screen.contains("Cargo.toml"));
    assert!(!screen.contains("**文件**"));
    assert!(!screen.contains("`.gitignore`"));
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
    assert!(!screen.contains("cafebabe"));
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

    assert!(
        screen
            .lines()
            .any(|row| { row.contains("Bash") && row.contains("git status") })
    );
    assert!(
        screen
            .lines()
            .any(|row| { row.contains("Bash") && row.contains("git rev-parse --short HEAD") })
    );
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
        .position(|row| row.contains("high"))
        .expect("missing status row");
    let keymap_row = rows
        .iter()
        .position(|row| row.contains("↑↓") && row.contains("select"))
        .expect("missing keymap row");
    let input_row = rows
        .iter()
        .rposition(|row| row.contains("/se"))
        .expect("missing input row");
    assert!(completion_row < status_row);
    assert!(completion_row <= input_row);
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
    assert!(screen.contains("close"));
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
    assert!(app.session_picker_open());
    let backend = TestBackend::new(100, 24);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    let rows = screen.lines().collect::<Vec<_>>();

    for value in ["deadbeef", "cafebabe", "first task", "second task"] {
        assert!(screen.contains(value), "missing {value}\n{screen}");
    }
    assert!(
        !rows
            .iter()
            .any(|row| row.contains("deadbeef") && row.contains("cafebabe")),
        "session records merged"
    );

    app.dismiss_session_picker();
    app.record_error("First error line\nSecond error line".to_owned());
    app.toggle_latest_error();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let error_screen = format!("{}", terminal.backend());
    assert!(error_screen.contains("First error line"));
    assert!(error_screen.contains("Second error line"));
    assert_eq!(error_screen.matches("ctrl+e").count(), 1);
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
    assert!(screen.contains("Read") || screen.contains("read"));
    assert!(!screen.contains("package zex"));
    assert!(screen.lines().any(|row| {
        (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml")
    }));
    assert!(!screen.contains("Ctrl+O expand"));
    assert!(!screen.contains("timeout 60.0s"));
    assert!(!screen.contains("\"path\": \"Cargo.toml\""));
    assert!(screen.contains("next prompt"));
    assert!(screen.contains("enter") && screen.contains("send"));
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
        arguments: r#"{"command":"git status --short --branch","timeout_seconds":30}"#.to_owned(),
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
    assert!(
        folded
            .lines()
            .any(|row| { row.contains("Bash") && row.contains("git status --short --branch") })
    );
    assert_eq!(folded.matches("## main...origin/main [ahead 1]").count(), 0);
    assert_eq!(
        folded
            .matches("Branch main is ahead by one commit with one modified file.")
            .count(),
        1
    );
    assert!(!folded.contains("exit_code: 0"));
    assert!(!folded.contains("timeout 30.0s"));
    assert!(!folded.contains("\"timeout_seconds\": 30"));

    app.toggle_selected_tool();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let expanded = format!("{}", terminal.backend());
    assert!(expanded.contains("git status --short --branch"));
    assert!(expanded.contains("M src/tui.rs"));
    assert!(!expanded.contains("input"));
    assert!(!expanded.contains("output {"));
    assert!(!expanded.contains("exit_code:"));
    assert!(!expanded.contains("stdout:"));
    assert!(!expanded.contains("stderr:"));
    assert!(!expanded.contains("timeout_seconds"));
    assert!(!expanded.contains("timeout 30.0s"));
    assert!(expanded.contains("Branch main is ahead by one commit"));
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
    assert!(
        screen
            .lines()
            .any(|row| { row.contains("Bash") && row.contains("git status --porcelain") })
    );
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
    assert!(screen.contains("working"));
    assert!(screen.contains("interrupt"));
    // The running tool row leads with a spinner; the word "running" would be
    // redundant next to it and the footer's activity label.
    assert!(
        screen
            .lines()
            .any(|row| { row.contains("Bash") && row.contains("git status") })
    );
    assert!(!screen.contains("running"));
}

#[test]
fn landing_and_workspace_regions_stay_inside_representative_frames() {
    for (width, height) in [(120, 32), (80, 24), (38, 12), (16, 6)] {
        let mut landing = configured_app();
        landing.working_dir = PathBuf::from("D:/code/Zex");
        let (terminal, screen) = draw(&mut landing, width, height);
        write_screen_dump(&format!("landing-{width}x{height}"), &screen);
        let area = ratatui::layout::Rect::new(0, 0, width, height);
        let regions = landing_regions(area, &landing);
        assert!(regions.top.bottom() <= area.bottom());
        assert!(regions.card.bottom() <= area.bottom());
        assert!(regions.menu.bottom() <= area.bottom());
        assert!(regions.status.bottom() <= area.bottom());
        assert!(regions.completion.bottom() <= area.bottom());
        assert!(regions.hero.bottom() <= area.bottom());
        assert!(regions.logo.bottom() <= area.bottom());
        assert!(
            regions.menu.is_empty()
                || regions.card.y >= regions.menu.bottom()
                || !regions.hero.is_empty()
        );
        assert!(regions.completion.is_empty() || regions.completion.bottom() <= regions.card.y);
        assert_buffer_in_bounds(&terminal, width, height);
        assert_no_role_labels(&screen);

        let mut work = two_turn_conversation();
        let (terminal, screen) = draw(&mut work, width, height);
        write_screen_dump(&format!("work-{width}x{height}"), &screen);
        assert_work_regions(&work, width, height);
        assert_buffer_in_bounds(&terminal, width, height);
        assert_no_role_labels(&screen);
        if width >= 80 {
            assert!(screen.contains("当前目录有哪些文件") || screen.contains("Cargo.toml"));
            assert!(screen.contains("gpt-5"));
            assert!(screen.contains("high"));
        }
    }
}

#[test]
fn two_turn_transcript_folds_then_expands_the_tool_card() {
    let mut app = two_turn_conversation();
    let (terminal, folded) = draw(&mut app, 120, 32);
    write_screen_dump("two-turn-folded-120x32", &folded);
    assert_no_role_labels(&folded);
    assert!(folded.contains("Thought") || folded.contains("Thinking"));
    assert!(folded.lines().any(|row| {
        (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml")
    }));
    assert!(!folded.contains("line one"));
    assert_eq!(
        folded
            .lines()
            .filter(
                |row| (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml")
            )
            .count(),
        1
    );
    assert!(
        !folded.contains("先列目录再总结当前工作区。"),
        "collapsed thought should hide the body"
    );
    assert_buffer_in_bounds(&terminal, 120, 32);

    let tool_index = app
        .transcript
        .iter()
        .position(|entry| matches!(entry, TranscriptEntry::Tool(_)))
        .unwrap();
    handle_key_event(
        key(
            crossterm::event::KeyCode::Tab,
            crossterm::event::KeyModifiers::NONE,
        ),
        &mut app,
        false,
        false,
    );
    app.selected_entry = Some(tool_index);
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
        InputAction::None
    );
    let TranscriptEntry::Tool(tool) = &app.transcript[tool_index] else {
        panic!("expected tool");
    };
    assert!(tool.expanded);

    let (_, expanded) = draw(&mut app, 120, 32);
    write_screen_dump("two-turn-expanded-120x32", &expanded);
    assert!(expanded.contains("line one"));
    assert!(expanded.contains("line two"));
    assert_no_role_labels(&expanded);

    let (_, expanded_narrow) = draw(&mut app, 80, 24);
    write_screen_dump("work-expanded-80x24", &expanded_narrow);
    assert!(expanded_narrow.contains("line one") || expanded_narrow.contains("Cargo.toml"));
}

#[test]
fn slash_completion_opens_above_the_input_and_lists_sessions() {
    let mut app = two_turn_conversation();
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Char('/'),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        ),
        InputAction::None
    );
    assert!(app.completion_open());
    let area = ratatui::layout::Rect::new(0, 0, 80, 24);
    let (terminal, screen) = draw(&mut app, 80, 24);
    write_screen_dump("completion-80x24", &screen);
    let regions = ui_regions(area, &app);
    assert!(regions.completion.height > 0);
    assert!(regions.completion.bottom() <= regions.footer.y);
    assert!(regions_non_overlapping(&regions));
    assert!(screen.contains("/sessions"));
    assert!(screen.contains("List saved sessions"));
    assert!(screen.contains("/"));
    assert_buffer_in_bounds(&terminal, 80, 24);

    let (narrow_terminal, narrow) = draw(&mut app, 38, 12);
    write_screen_dump("completion-38x12", &narrow);
    assert_work_regions(&app, 38, 12);
    assert_buffer_in_bounds(&narrow_terminal, 38, 12);
    assert!(narrow.contains("/sessions") || narrow.contains("/help") || narrow.contains("/"));
}

#[test]
fn overlay_page_replaces_the_transcript_and_keeps_the_input_band() {
    let mut app = two_turn_conversation();
    app.open_model_picker();
    let area = ratatui::layout::Rect::new(0, 0, 80, 24);
    let (terminal, screen) = draw(&mut app, 80, 24);
    write_screen_dump("overlay-model-80x24", &screen);
    let regions = ui_regions(area, &app);
    assert!(regions_inside_frame(area, &regions));
    assert!(regions_non_overlapping(&regions));
    assert!(screen.contains("Models"));
    assert!(screen.contains("Current: OpenAI / GPT-5"));
    assert!(!screen.contains("Ask anything"));
    assert!(screen.contains("Models"));
    assert!(screen.contains("model switcher") || screen.contains("gpt-5"));
    assert_buffer_in_bounds(&terminal, 80, 24);

    let (narrow_terminal, narrow) = draw(&mut app, 38, 12);
    write_screen_dump("overlay-model-38x12", &narrow);
    assert_work_regions(&app, 38, 12);
    assert_buffer_in_bounds(&narrow_terminal, 38, 12);
    assert!(narrow.contains("Models") || narrow.contains("GPT-5") || narrow.contains(PRODUCT_NAME));
}

#[test]
fn idle_and_busy_footers_stay_identifiable() {
    let mut app = two_turn_conversation();
    let (_, idle) = draw(&mut app, 80, 24);
    write_screen_dump("idle-80x24", &idle);
    assert!(idle.contains("gpt-5"));
    assert!(idle.contains("high"));
    assert!(idle.contains("enter") && idle.contains("send"));
    assert!(!idle.contains("thinking"));

    app.start_turn();
    let (terminal, busy) = draw(&mut app, 80, 24);
    write_screen_dump("busy-80x24", &busy);
    assert!(busy.contains("thinking"));
    assert!(busy.contains("interrupt"));
    assert!(busy.contains("gpt-5"));
    assert!(busy.contains("high"));
    assert_work_regions(&app, 80, 24);
    assert_buffer_in_bounds(&terminal, 80, 24);

    let (tiny_terminal, tiny_busy) = draw(&mut app, 16, 6);
    write_screen_dump("busy-16x6", &tiny_busy);
    assert_work_regions(&app, 16, 6);
    assert_buffer_in_bounds(&tiny_terminal, 16, 6);
    assert!(!tiny_busy.is_empty());
}

#[test]
fn cjk_and_wide_text_truncate_instead_of_spilling() {
    let mut app = two_turn_conversation();
    app.model = "provider/非常长的模型名称-that-cannot-fit-in-a-narrow-band".to_owned();
    app.working_dir = PathBuf::from("D:/非常长的工作区路径/需要被截断/Zex");
    let (terminal, screen) = draw(&mut app, 38, 12);
    write_screen_dump("cjk-38x12", &screen);
    assert_work_regions(&app, 38, 12);
    assert_buffer_in_bounds(&terminal, 38, 12);
    assert!(
        screen.contains("当前")
            || screen.contains("Read")
            || screen.contains("read")
            || screen.contains("enter"),
        "narrow CJK screen lost its primary chrome:\n{screen}"
    );
    assert!(!screen.contains("YOU"));
}

#[test]
fn conversation_fixture_renders_identically_twice() {
    let mut app = two_turn_conversation();
    let (_, first) = draw(&mut app, 120, 32);
    let (_, second) = draw(&mut app, 120, 32);
    assert_eq!(first, second);
}

#[test]
fn tiny_frame_does_not_panic_or_leave_the_buffer() {
    let mut app = two_turn_conversation();
    let (terminal, screen) = draw(&mut app, 16, 6);
    write_screen_dump("tiny-16x6", &screen);
    assert_work_regions(&app, 16, 6);
    assert_buffer_in_bounds(&terminal, 16, 6);
    assert!(!screen.is_empty());
}

#[test]
fn prompt_flag_is_a_headless_path_and_tui_requires_a_tty() {
    let main = include_str!("../main.rs");
    let prompt_branch = main
        .find("if let Some(prompt) = cli.run_prompt()")
        .expect("main must branch on -p / --prompt");
    let headless_call = main
        .find("headless::run_prompt(")
        .expect("prompt path must call headless::run_prompt");
    let tui_gate = main
        .find("} else if tui::is_available()")
        .expect("interactive TUI must be gated on is_available");
    let tui_run = main.find("tui::run(").expect("TTY path must call tui::run");
    assert!(prompt_branch < headless_call);
    assert!(headless_call < tui_gate);
    assert!(tui_gate < tui_run);

    let tui = include_str!("../tui.rs");
    let available = tui
        .split("pub fn is_available() -> bool")
        .nth(1)
        .expect("is_available must exist");
    assert!(available.contains("io::stdin().is_terminal()"));
    assert!(available.contains("io::stdout().is_terminal()"));
}

#[test]
fn conversation_paint_shows_typed_scrollback_and_composer_metadata() {
    let mut app = two_turn_conversation();
    let (_, first) = draw(&mut app, 100, 24);
    write_screen_dump("conversation", &first);
    assert!(
        first.lines().any(|row| row.contains("当前目录有哪些文件")),
        "user message missing:\n{first}"
    );
    assert!(first.contains("我会先查看当前工作目录中的条目"));
    assert!(
        first.contains("Thought") || first.contains("Thinking"),
        "thinking header missing:\n{first}"
    );
    assert!(
        first
            .lines()
            .any(|row| (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml")),
        "tool header missing:\n{first}"
    );
    assert_no_banned_branding(&first);
    assert!(first.contains("gpt-5"));
    assert!(first.contains("high"));
    let (_, second) = draw(&mut app, 100, 24);
    assert_eq!(first, second);
}

#[test]
fn welcome_and_agent_enter_send_non_slash_drafts() {
    let mut welcome = configured_app();
    welcome.input.insert_str("inspect the workspace");
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut welcome,
            false,
            false,
        ),
        InputAction::Submit("inspect the workspace".to_owned())
    );

    let mut agent = two_turn_conversation();
    agent.input.insert_str("continue from here");
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut agent,
            false,
            false,
        ),
        InputAction::Submit("continue from here".to_owned())
    );
}

#[test]
fn slash_from_welcome_lists_the_shipped_zex_commands() {
    let mut app = configured_app();
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Char('/'),
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            false,
            false,
        ),
        InputAction::None
    );
    assert!(app.completion_open());
    let names: Vec<_> = app
        .completion_matches()
        .into_iter()
        .map(|command| command.name)
        .collect();
    for command in [
        "/help",
        "/model",
        "/provider",
        "/new",
        "/clear",
        "/sessions",
        "/resume",
        "/compact",
        "/think",
        "/thinking",
    ] {
        assert!(names.contains(&command), "missing {command} in {names:?}");
    }
}

#[test]
fn welcome_resume_and_session_commands_open_the_session_index() {
    let mut welcome = configured_app();
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Char('s'),
                crossterm::event::KeyModifiers::CONTROL,
            ),
            &mut welcome,
            false,
            false,
        ),
        InputAction::Submit("/resume".to_owned())
    );

    welcome.input.replace("/sessions");
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut welcome,
            false,
            false,
        ),
        InputAction::Submit("/sessions".to_owned())
    );

    let mut resume = configured_app();
    resume.input.replace("/resume");
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut resume,
            false,
            false,
        ),
        InputAction::Submit("/resume".to_owned())
    );

    let sessions = vec![crate::session::SessionSummary {
        id: "20260813-120000-cafebabe".to_owned(),
        updated_at: time::OffsetDateTime::UNIX_EPOCH,
        message_count: 3,
        preview: "resume me".to_owned(),
    }];
    for output in [
        CommandOutput::Sessions(sessions.clone()),
        CommandOutput::ResumePicker(sessions),
    ] {
        let mut app = configured_app();
        app.push_command_output(output);
        assert!(app.session_picker_open());
        let (_, screen) = draw(&mut app, 100, 24);
        assert!(screen.contains("Session index") || screen.contains("cafebabe"));
        assert!(screen.contains("resume me"));
    }

    let mut menu = configured_app();
    let (_, _) = draw(&mut menu, 100, 24);
    handle_key_event(
        key(
            crossterm::event::KeyCode::Down,
            crossterm::event::KeyModifiers::NONE,
        ),
        &mut menu,
        false,
        false,
    );
    assert_eq!(menu.welcome_menu_selected, Some(0));
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut menu,
            false,
            false,
        ),
        InputAction::Submit("/resume".to_owned())
    );
}

#[test]
fn esc_while_busy_requests_interrupt() {
    let mut app = two_turn_conversation();
    app.start_turn();
    assert!(app.busy);
    assert_eq!(
        handle_key_event(
            key(
                crossterm::event::KeyCode::Esc,
                crossterm::event::KeyModifiers::NONE,
            ),
            &mut app,
            true,
            false,
        ),
        InputAction::Interrupt
    );
}

#[test]
fn welcome_wide_uses_hero_box_and_night_palette() {
    let mut app = configured_app();
    app.working_dir = PathBuf::from("D:/workspaces/zex");
    app.git_status = Some(super::GitStatus {
        branch: "main".to_owned(),
        commit: "abc1234".to_owned(),
        dirty_count: 0,
    });
    app.input_focused = false;
    let backend = TestBackend::new(104, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    write_screen_dump("welcome-wide", &screen);
    let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 104, 24), &app);

    assert!(104 >= HERO_BOX_MIN_WIDTH);
    assert!(
        !regions.hero.is_empty(),
        "wide welcome must use the hero box"
    );
    assert_zex_welcome_identity(&screen);
    assert!(screen.contains("Type a message..."));
    assert!(screen.contains("main"));
    assert!(screen_has_glyph(&terminal, "╭"));
    assert!(
        screen_has_glyph(&terminal, "\u{276F}") || screen.contains('❯') || screen.contains('>')
    );
    assert_eq!(style_at(&terminal, 0, 0).bg, Some(theme().background));
    assert!(!screen_has_rgb(&terminal, Color::Rgb(120, 158, 166)));
    assert_no_banned_branding(&screen);
}

#[test]
fn welcome_narrow_stacks_without_hero_box() {
    let mut app = configured_app();
    app.working_dir = PathBuf::from("D:/workspaces/zex");
    app.input_focused = false;
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    write_screen_dump("welcome-narrow", &screen);
    let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 80, 24), &app);

    assert!(regions.hero.is_empty(), "narrow welcome must stack");
    assert!(
        !regions.logo.is_empty(),
        "24-row stacked welcome shows the wordmark"
    );
    assert_eq!(logo_tier(24), LogoTier::Full);
    assert!(screen.contains(PRODUCT_NAME));
    assert!(screen.contains("Type a message..."));
    assert!(screen.contains("Resume sessions"));
    assert!(screen_has_glyph(&terminal, "╭"));
    assert_no_banned_branding(&screen);
}

#[test]
fn welcome_short_height_hides_the_logo() {
    let mut app = configured_app();
    let backend = TestBackend::new(80, 16);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let regions = landing_regions(ratatui::layout::Rect::new(0, 0, 80, 16), &app);
    assert_eq!(logo_tier(16), LogoTier::Hidden);
    assert!(regions.logo.is_empty());
    assert!(regions.hero.is_empty());
    let screen = format!("{}", terminal.backend());
    assert!(screen.contains(PRODUCT_NAME) || screen.contains("╭"));
    assert_no_banned_branding(&screen);
}

#[test]
fn conversation_paints_rail_thought_tool_info_and_shortcuts() {
    let mut app = two_turn_conversation();
    let backend = TestBackend::new(100, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    write_screen_dump("conversation", &screen);

    assert!(screen.contains("当前目录有哪些文件"));
    assert!(screen.contains("Thought") || screen.contains("Thinking"));
    assert!(screen.lines().any(|row| {
        (row.contains("Read") || row.contains("read")) && row.contains("Cargo.toml")
    }));
    assert!(screen.contains("gpt-5") && screen.contains(" · ") && screen.contains("high"));
    assert!(screen.contains("│"));
    assert!(screen_has_glyph(&terminal, "╭"));
    assert!(
        screen_has_glyph(&terminal, "\u{2503}")
            || screen_has_glyph(&terminal, "┃")
            || screen_has_glyph(&terminal, "\u{2502}")
            || screen_has_glyph(&terminal, "│")
    );
    assert_eq!(style_at(&terminal, 0, 0).bg, Some(theme().background));
    assert!(!screen.contains("Ask anything"));
    assert!(!screen.contains('›'));
    assert_no_banned_branding(&screen);
}

#[test]
fn slash_completion_is_a_highlighted_label_description_dropdown() {
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
    let screen = format!("{}", terminal.backend());
    write_screen_dump("slash-open", &screen);
    assert!(screen.contains("/help"));
    assert!(
        screen.contains("Open command index")
            || screen.contains("List saved sessions")
            || screen.contains("/sessions")
    );
    assert!(
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .any(|cell| cell.style().bg == Some(theme().surface_raised)
                || cell.style().fg == Some(theme().accent_primary))
    );
    assert_no_banned_branding(&screen);
}

#[test]
fn expanded_tool_and_answer_keep_base_background_and_ascii_glyphs() {
    let mut app = app();
    app.apply_agent_event(AgentEvent::ToolStart {
        call_id: "call-dir".to_owned(),
        name: "glob".to_owned(),
        arguments: r#"{"pattern":"*"}"#.to_owned(),
        timeout: Duration::from_secs(30),
    });
    app.apply_agent_event(AgentEvent::ToolEnd {
        call_id: "call-dir".to_owned(),
        name: "glob".to_owned(),
        output: ".git\n.gitignore\nCargo.lock\nCargo.toml\nsrc\ntarget\n6 matching path(s)"
            .to_owned(),
        is_error: false,
        elapsed: Duration::from_millis(20),
    });
    app.apply_agent_event(AgentEvent::MessageDelta {
        role: MessageRole::Assistant,
        delta: "当前目录包含：\n- Cargo.toml\n- README.md\n- target".to_owned(),
    });
    let tool_index = app
        .transcript
        .iter()
        .position(|entry| matches!(entry, TranscriptEntry::Tool(_)))
        .unwrap();
    app.toggle_card(tool_index);
    app.selected_entry = Some(tool_index);

    let backend = TestBackend::new(100, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());

    assert!(screen.contains("Glob"), "verb smashed:\n{screen}");
    assert!(screen.contains("Cargo.toml"), "path smashed:\n{screen}");
    assert!(screen.contains("Cargo.lock"));
    assert!(screen.contains("当前目录包含"));
    assert!(!screen.contains("g1ob"));
    assert!(!screen.contains("tom1"));

    let output_row = screen
        .lines()
        .position(|row| row.contains("Cargo.toml") && row.contains("path") == false)
        .or_else(|| screen.lines().position(|row| row.contains("Cargo.toml")))
        .expect("tool body row") as u16;
    assert_eq!(
        style_at(&terminal, super::HORIZONTAL_GUTTER + 4, output_row).bg,
        Some(theme().background),
        "expanded tool body must not sit on a raised slab"
    );

    let answer_row = screen
        .lines()
        .position(|row| row.contains("README.md"))
        .expect("answer row") as u16;
    assert_eq!(
        style_at(&terminal, super::HORIZONTAL_GUTTER + 4, answer_row).bg,
        Some(theme().background),
        "assistant body must not sit on a raised slab"
    );
}

#[test]
fn collapsed_thinking_uses_thought_for_when_elapsed_is_known() {
    let mut app = app();
    app.transcript
        .push(TranscriptEntry::Thinking(ThinkingEntry {
            content: "hidden body".to_owned(),
            expanded: false,
            elapsed: Some(Duration::from_secs(3)),
        }));
    let backend = TestBackend::new(80, 18);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    assert!(screen.contains("Thought for"));
    assert!(screen.contains("3s"));
    assert!(!screen.contains("hidden body"));
    assert_no_banned_branding(&screen);
}

#[test]
fn session_picker_is_a_highlight_list_not_raised_cards() {
    let mut app = app();
    app.open_session_picker(vec![crate::session::SessionSummary {
        id: "20260813-120000-cafebabe".to_owned(),
        updated_at: time::OffsetDateTime::UNIX_EPOCH,
        message_count: 3,
        preview: "resume me".to_owned(),
    }]);
    let backend = TestBackend::new(100, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    assert!(screen.contains("cafebabe"));
    assert!(screen.contains("resume me"));
    let selected = app
        .hit_regions
        .iter()
        .find(|region| region.target == HitTarget::Session(0))
        .expect("session row")
        .area;
    assert_eq!(selected.height, 1);
    assert_no_banned_branding(&screen);
}

#[test]
fn grok_style_visual_regression() {
    let mut app = app();
    app.transcript.extend([
        TranscriptEntry::Message {
            role: MessageRole::User,
            content: "Refactor the renderer to match grok.".to_owned(),
        },
        TranscriptEntry::Thinking(ThinkingEntry {
            content: "compare themes".to_owned(),
            expanded: false,
            elapsed: Some(Duration::from_millis(2400)),
        }),
        TranscriptEntry::Tool(super::ToolEntry {
            call_id: "tool-edit".to_owned(),
            name: "edit".to_owned(),
            arguments: r#"{"path":"src/tui.rs","old_text":"let a = 1;","new_text":"let a = 2;"}"#
                .to_owned(),
            output: String::new(),
            status: ToolStatus::Done,
            expanded: true,
            show_full_output: false,
            started_at: None,
            elapsed: Some(Duration::from_millis(12)),
            timeout: Duration::from_secs(30),
        }),
        TranscriptEntry::Message {
            role: MessageRole::Assistant,
            content: "## Plan\n- one\n- two\n```rust\nfn main() {}\n```\nDone with `cargo test`."
                .to_owned(),
        },
    ]);
    app.start_turn();
    let backend = TestBackend::new(100, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());
    write_screen_dump("grok-style-regression-100x30", &screen);

    // User turn: ❯ prompt plus a full-width raised band.
    let user_row = screen
        .lines()
        .position(|row| row.contains("Refactor the renderer"))
        .expect("user message visible") as u16;
    let buffer = terminal.backend().buffer();
    let transcript_x = super::HORIZONTAL_GUTTER;
    assert_eq!(buffer[(transcript_x, user_row)].symbol(), "❯");
    let regions = ui_regions(ratatui::layout::Rect::new(0, 0, 100, 30), &app);
    let band_right = regions.transcript.right().saturating_sub(3);
    assert_eq!(
        style_at(&terminal, band_right, user_row).bg,
        Some(theme().surface),
        "user band must span the full transcript width"
    );

    // Diff rows: full-width delete/insert bands.
    let del_row = screen
        .lines()
        .position(|row| row.contains("- let a = 1;"))
        .expect("delete row visible") as u16;
    let add_row = screen
        .lines()
        .position(|row| row.contains("+ let a = 2;"))
        .expect("insert row visible") as u16;
    assert_eq!(
        style_at(&terminal, band_right, del_row).bg,
        Some(super::theme().diff_del_bg)
    );
    assert_eq!(
        style_at(&terminal, band_right, add_row).bg,
        Some(super::theme().diff_add_bg)
    );

    // Code block rows sit on the raised code surface.
    let code_row = screen
        .lines()
        .position(|row| row.contains("fn main() {}"))
        .expect("code row visible") as u16;
    assert_eq!(
        style_at(&terminal, band_right, code_row).bg,
        Some(theme().code_bg)
    );

    // Tool header: path subject stays on the quiet gray ramp.
    let tool_row = screen
        .lines()
        .position(|row| row.contains("Edit") && row.contains("src/tui.rs"))
        .expect("tool header visible") as u16;
    let path_col = screen
        .lines()
        .nth(tool_row as usize)
        .unwrap()
        .find("src/tui.rs")
        .unwrap() as u16;
    assert_eq!(
        style_at(&terminal, path_col, tool_row).fg,
        Some(super::theme().text_dim)
    );

    // Composer chrome: teal model name.
    assert!(screen.contains("test-model"));
    let model_row = screen
        .lines()
        .position(|row| row.contains("test-model"))
        .expect("model caption visible") as u16;
    let model_col = screen
        .lines()
        .nth(model_row as usize)
        .unwrap()
        .find("test-model")
        .unwrap() as u16;
    assert_eq!(
        style_at(&terminal, model_col, model_row).fg,
        Some(super::theme().model_accent)
    );

    // Thinking status label replaces the generic "working".
    assert!(screen.contains("thinking"));
    assert!(screen.contains("Thought for 2s"));
}

#[test]
fn wrapped_message_rows_keep_inline_markdown_styling() {
    let mut app = app();
    app.transcript.push(TranscriptEntry::Message {
        role: MessageRole::Assistant,
        content: format!("{} **marker** {}", "plain ".repeat(20), "tail ".repeat(20)),
    });
    let backend = TestBackend::new(60, 20);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let screen = format!("{}", terminal.backend());

    let row = screen
        .lines()
        .position(|line| line.contains("marker"))
        .expect("bold word should render") as u16;
    let column = screen
        .lines()
        .nth(row as usize)
        .unwrap()
        .find("marker")
        .unwrap() as u16;
    let style = style_at(&terminal, column, row);
    assert_eq!(style.fg, Some(super::theme().text_strong));
    assert!(style.add_modifier.contains(ratatui::style::Modifier::BOLD));
}
