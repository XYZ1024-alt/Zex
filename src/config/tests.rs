use std::{
    ffi::OsString,
    process,
    sync::{Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

use super::Config;

static ENV_LOCK: Mutex<()> = Mutex::new(());

const TEST_ENV: &[&str] = &[
    "ZEX_CONFIG_DIR",
    "ZEX_API_KEY",
    "OPENAI_API_KEY",
    "ZEX_MODEL",
    "OPENAI_MODEL",
    "ZEX_BASE_URL",
    "OPENAI_BASE_URL",
    "ZEX_OPENAI_API",
    "ZEX_MAX_TURNS",
    "ZEX_TOOL_TIMEOUT_SECONDS",
    "ZEX_AGENT_TIMEOUT_SECONDS",
    "ZEX_MAX_TOOL_OUTPUT_CHARS",
    "ZEX_MAX_CONTEXT_CHARS",
    "ZEX_MAX_CONTEXT_TOKENS",
    "ZEX_COMPACT_KEEP_TURNS",
    "ZEX_MEMORY_ENABLED",
    "ZEX_MEMORY_MODE",
    "ZEX_MEMORY_RECALL_RATE_LIMIT",
    "ZEX_MEMORY_MAX_RECALL_TOKENS",
    "ZEX_MEMORY_HOT_CACHE_SIZE",
    "ZEX_MEMORY_AUTO_PIN_IMPORTANT",
    "ZEX_DEFAULT_THINKING_LEVEL",
    "ZEX_HIDE_THINKING_BLOCK",
    "ZEX_SESSION_DIR",
];

struct EnvGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvGuard {
    fn clear() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|error| error.into_inner());
        let saved = TEST_ENV
            .iter()
            .map(|name| (*name, std::env::var_os(name)))
            .collect();
        for name in TEST_ENV {
            unsafe { std::env::remove_var(name) };
        }
        Self { _lock: lock, saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in self.saved.drain(..) {
            match value {
                Some(value) => unsafe { std::env::set_var(name, value) },
                None => unsafe { std::env::remove_var(name) },
            }
        }
    }
}

fn temp_directory(label: &str) -> std::path::PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("zex-{label}-{}-{unique}", process::id()))
}

async fn write_config(path: &std::path::Path, content: &str) {
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .unwrap();
    tokio::fs::write(path, content).await.unwrap();
}

#[tokio::test]
async fn provider_catalog_overrides_legacy_values_and_environment_overrides_runtime_settings() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("config");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &global.join("config.toml"),
        r#"
api_key = "global-secret"
model = "global-model"
base_url = "https://global.example/v1"
max_turns = 4
"#,
    )
    .await;
    write_config(
        &project.join(".zex/config.toml"),
        r#"
active_model = { provider_id = "project", model_id = "project-model" }

[[providers]]
id = "project"
display_name = "Project"
base_url = "https://project.example/v1"
api_key = "project-secret"
openai_api = "responses"

[[providers.models]]
id = "project-model"
display_name = "Project Model"

max_turns = 8
"#,
    )
    .await;
    unsafe {
        std::env::set_var("ZEX_API_KEY", "environment-secret");
        std::env::set_var("ZEX_MAX_TURNS", "10");
    }

    let config = Config::load_from(&project, &global).await.unwrap();

    let active = config.active_model.as_ref().unwrap();
    let (provider, model) = config.providers.model(active).unwrap();
    assert_eq!(provider.api_key.expose(), "project-secret");
    assert_eq!(model.id, "project-model");
    assert_eq!(provider.base_url, "https://project.example/v1");
    assert_eq!(config.max_turns, 10);
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn persists_thinking_level_in_project_config() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("thinking");
    let project = root.join("project");
    tokio::fs::create_dir_all(&project).await.unwrap();

    super::persist_thinking_level(&project, crate::provider::ThinkingLevel::High)
        .await
        .unwrap();

    let content = tokio::fs::read_to_string(project.join(".zex/config.toml"))
        .await
        .unwrap();
    assert!(content.contains("default_thinking_level = \"high\""));
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn loads_and_persists_thinking_visibility() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("show-thinking");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &project.join(".zex/config.toml"),
        r#"
api_key = "secret"
model = "model"
hide_thinking_block = true
"#,
    )
    .await;

    let config = Config::load_from(&project, &global).await.unwrap();
    assert!(config.hide_thinking_block);

    super::persist_show_thinking(&project, true).await.unwrap();
    let content = tokio::fs::read_to_string(project.join(".zex/config.toml"))
        .await
        .unwrap();
    assert!(content.contains("hide_thinking_block = false"));
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn persists_provider_catalog_and_active_model_without_legacy_fields() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("provider-catalog");
    let project = root.join("project");
    tokio::fs::create_dir_all(&project).await.unwrap();
    write_config(
        &project.join(".zex/config.toml"),
        "model = \"legacy\"\napi_key = \"legacy-secret\"\n",
    )
    .await;
    let active_model = super::ModelRef {
        provider_id: "openai".to_owned(),
        model_id: "gpt-5".to_owned(),
    };
    let catalog = super::ProviderCatalog {
        active_model: Some(active_model.clone()),
        models_dev: Default::default(),
        models_dev_aliases: Vec::new(),
        providers: vec![super::ProviderConfig {
            id: "openai".to_owned(),
            display_name: "OpenAI".to_owned(),
            base_url: "https://api.openai.com/v1".to_owned(),
            api_key: super::SecretValue::new("secret".to_owned()),
            openai_api: crate::provider::OpenAiApi::Responses,
            thinking: None,
            compat: None,
            models: vec![super::ModelConfig {
                id: "gpt-5".to_owned(),
                display_name: "GPT-5".to_owned(),
                thinking: Some(crate::provider::ThinkingConfig {
                    min_level: crate::provider::ThinkingLevel::Low,
                    max_level: crate::provider::ThinkingLevel::Max,
                    supported: None,
                    mode: crate::provider::ThinkingMode::Effort,
                }),
                compat: None,
                context_window: None,
            }],
        }],
    };

    super::persist_provider_catalog(&project, &catalog)
        .await
        .unwrap();
    let content = tokio::fs::read_to_string(project.join(".zex/config.toml"))
        .await
        .unwrap();

    assert!(content.contains("[[providers]]"));
    assert!(content.contains("provider_id = \"openai\""));
    assert!(!content.contains("model = \"legacy\""));
    assert!(!content.contains("api_key = \"legacy-secret\""));
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn loads_provider_and_model_thinking_capabilities() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("thinking-capabilities");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &project.join(".zex/config.toml"),
        r#"
active_model = { provider_id = "openai", model_id = "codex-ultra" }
default_thinking_level = "max"
hide_thinking_block = true

[[providers]]
id = "openai"
display_name = "OpenAI"
base_url = "https://api.openai.com/v1"
api_key = "secret"
openai_api = "responses"

[providers.thinking]
min_level = "low"
max_level = "xhigh"
mode = "effort"

[providers.compat]
supports_reasoning_effort = true
supports_interleaved_thinking = true

[providers.compat.reasoning_effort_map]
xhigh = "xhigh"

[[providers.models]]
id = "codex-ultra"
display_name = "Codex Ultra"

[providers.models.thinking]
min_level = "minimal"
max_level = "max"
mode = "effort"

[providers.models.compat.reasoning_effort_map]
max = "max"
"#,
    )
    .await;

    let config = Config::load_from(&project, &global).await.unwrap();
    let capabilities = config
        .providers
        .thinking_capabilities(config.active_model.as_ref().unwrap());

    assert_eq!(
        config.default_thinking_level,
        crate::provider::ThinkingLevel::Max
    );
    assert!(config.hide_thinking_block);
    assert_eq!(
        capabilities.min_level,
        crate::provider::ThinkingLevel::Minimal
    );
    assert_eq!(capabilities.max_level, crate::provider::ThinkingLevel::Max);
    assert!(capabilities.supports_interleaved_thinking);
    assert_eq!(
        capabilities
            .reasoning_effort_map
            .get(&crate::provider::ThinkingLevel::Max)
            .map(String::as_str),
        Some("max")
    );
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[test]
fn provider_base_url_can_match_models_dev_provider() {
    let mut catalog = super::ProviderCatalog {
        models_dev: crate::provider::ModelsDevCatalog::default(),
        models_dev_aliases: vec![crate::provider::ModelsDevProviderAlias {
            id: "openai".to_owned(),
            api: Some("https://api.openai.com/v1".to_owned()),
        }],
        ..Default::default()
    };
    catalog.providers.push(super::ProviderConfig {
        id: "gateway".to_owned(),
        display_name: "Gateway".to_owned(),
        base_url: "https://api.openai.com/v1/".to_owned(),
        api_key: super::SecretValue::new("secret".to_owned()),
        openai_api: crate::provider::OpenAiApi::Responses,
        thinking: None,
        compat: None,
        models: Vec::new(),
    });

    assert_eq!(
        catalog.matched_models_dev_provider(&catalog.providers[0]),
        Some("openai")
    );
}

#[test]
fn custom_provider_uses_namespaced_models_dev_capabilities() {
    let mut catalog = super::ProviderCatalog {
            models_dev: crate::provider::ModelsDevCatalog::from_json(
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
            .unwrap(),
            ..Default::default()
        };
    catalog.providers.push(super::ProviderConfig {
        id: "custom".to_owned(),
        display_name: "Custom".to_owned(),
        base_url: "https://example.com/v1".to_owned(),
        api_key: super::SecretValue::new("secret".to_owned()),
        openai_api: crate::provider::OpenAiApi::Responses,
        thinking: None,
        compat: None,
        models: vec![super::ModelConfig {
            id: "gpt-5.4-mini".to_owned(),
            display_name: "GPT-5.4 mini".to_owned(),
            thinking: None,
            compat: None,
            context_window: None,
        }],
    });

    assert_eq!(
        catalog
            .thinking_capabilities(&super::ModelRef {
                provider_id: "custom".to_owned(),
                model_id: "gpt-5.4-mini".to_owned(),
            })
            .available_levels(),
        vec![
            crate::provider::ThinkingLevel::Off,
            crate::provider::ThinkingLevel::Low,
            crate::provider::ThinkingLevel::Medium,
            crate::provider::ThinkingLevel::High,
            crate::provider::ThinkingLevel::XHigh
        ]
    );
}

#[tokio::test]
async fn relative_session_directory_is_resolved_from_project() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("session-dir");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &project.join(".zex/config.toml"),
        r#"
api_key = "secret"
model = "model"
session_dir = ".zex/sessions"
"#,
    )
    .await;

    Config::load_from(&project, &global).await.unwrap();
    assert_eq!(
        Config::session_dir_from(&project, &global).await.unwrap(),
        project.join(".zex/sessions")
    );
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn session_directory_can_load_without_provider_credentials() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("sessions-only");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &project.join(".zex/config.toml"),
        "session_dir = \".zex/history\"\n",
    )
    .await;

    let directory = Config::session_dir_from(&project, &global).await.unwrap();

    assert_eq!(directory, project.join(".zex/history"));
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[test]
fn explicit_global_config_directory_has_priority() {
    let _environment = EnvGuard::clear();
    let directory = temp_directory("config-dir");
    unsafe { std::env::set_var("ZEX_CONFIG_DIR", &directory) };

    assert_eq!(super::global_config_dir().unwrap(), directory);
}

#[test]
fn theme_color_parses_hex_and_terminal_keywords() {
    use super::ThemeColor;

    assert_eq!(
        ThemeColor::parse("#7df").unwrap(),
        ThemeColor::Rgb(0x77, 0xdd, 0xff)
    );
    assert_eq!(
        ThemeColor::parse("#7DCFFF").unwrap(),
        ThemeColor::Rgb(0x7d, 0xcf, 0xff)
    );
    assert_eq!(
        ThemeColor::parse(" default ").unwrap(),
        ThemeColor::Terminal
    );
    assert_eq!(ThemeColor::parse("RESET").unwrap(), ThemeColor::Terminal);

    for invalid in ["", "red", "7dcfff", "#12", "#12345", "#1234567", "#zzzzzz"] {
        assert!(
            ThemeColor::parse(invalid).is_err(),
            "{invalid:?} should fail"
        );
    }
}

#[tokio::test]
async fn theme_overrides_merge_project_over_global() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("theme");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &global.join("config.toml"),
        r##"
[theme]
accent_primary = "#111111"
accent_secondary = "#222222"
"##,
    )
    .await;
    write_config(
        &project.join(".zex/config.toml"),
        r##"
[theme]
accent_primary = "#333333"
"##,
    )
    .await;

    let config = Config::load_from(&project, &global).await.unwrap();

    assert_eq!(
        config.theme.accent_primary,
        Some(super::ThemeColor::Rgb(0x33, 0x33, 0x33))
    );
    assert_eq!(
        config.theme.accent_secondary,
        Some(super::ThemeColor::Rgb(0x22, 0x22, 0x22))
    );
    assert_eq!(config.theme.background, None);
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn invalid_theme_color_fails_config_load() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("theme-invalid");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &project.join(".zex/config.toml"),
        r#"
[theme]
accent_primary = "not-a-color"
"#,
    )
    .await;

    assert!(Config::load_from(&project, &global).await.is_err());
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn legacy_max_context_chars_key_and_environment_variable_still_work() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("legacy-context");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &project.join(".zex/config.toml"),
        "max_context_chars = 64_000\n",
    )
    .await;

    let config = Config::load_from(&project, &global).await.unwrap();
    assert_eq!(config.max_context_tokens, 64_000);

    unsafe {
        std::env::set_var("ZEX_MAX_CONTEXT_CHARS", "80000");
    }
    let config = Config::load_from(&project, &global).await.unwrap();
    assert_eq!(config.max_context_tokens, 80_000);

    unsafe {
        std::env::set_var("ZEX_MAX_CONTEXT_TOKENS", "96000");
    }
    let config = Config::load_from(&project, &global).await.unwrap();
    assert_eq!(config.max_context_tokens, 96_000);

    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn memory_config_merges_and_environment_overrides_runtime_limits() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("memory-config");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &global.join("config.toml"),
        r#"
[memory]
enabled = true
mode = "summary"
recall_rate_limit = 4
max_recall_tokens = 1024
hot_cache_size = 8
auto_pin_important = false
"#,
    )
    .await;
    write_config(
        &project.join(".zex/config.toml"),
        r#"
[memory]
enabled = false
mode = "pointer_priority"
max_recall_tokens = 2048
"#,
    )
    .await;
    unsafe {
        std::env::set_var("ZEX_MEMORY_ENABLED", "true");
        std::env::set_var("ZEX_MEMORY_RECALL_RATE_LIMIT", "7");
    }

    let config = Config::load_from(&project, &global).await.unwrap();

    assert!(config.memory.enabled);
    assert_eq!(
        config.memory.mode,
        crate::memory::MemoryMode::PointerPriority
    );
    assert_eq!(config.memory.recall_rate_limit, 7);
    assert_eq!(config.memory.max_recall_tokens, 2_048);
    assert_eq!(config.memory.hot_cache_size, 8);
    assert!(!config.memory.auto_pin_important);
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[tokio::test]
async fn memory_can_be_disabled_for_traditional_behavior() {
    let _environment = EnvGuard::clear();
    let root = temp_directory("memory-disabled");
    let project = root.join("project");
    let global = root.join("global");
    write_config(
        &project.join(".zex/config.toml"),
        "[memory]\nenabled = false\n",
    )
    .await;

    let config = Config::load_from(&project, &global).await.unwrap();

    assert!(!config.memory.enabled);
    tokio::fs::remove_dir_all(root).await.unwrap();
}

#[test]
fn model_context_window_overrides_models_dev_limit() {
    let mut catalog = super::ProviderCatalog {
        models_dev: crate::provider::ModelsDevCatalog::from_json(
            br#"{
                "one": {
                    "models": {
                        "m": {"limit": {"context": 100000, "output": 32000}}
                    }
                }
            }"#,
        )
        .unwrap(),
        ..Default::default()
    };
    catalog.providers.push(super::ProviderConfig {
        id: "one".to_owned(),
        display_name: "One".to_owned(),
        base_url: "https://one.example/v1".to_owned(),
        api_key: super::SecretValue::new("secret".to_owned()),
        openai_api: crate::provider::OpenAiApi::Responses,
        thinking: None,
        compat: None,
        models: vec![
            super::ModelConfig {
                id: "m".to_owned(),
                display_name: "M".to_owned(),
                thinking: None,
                compat: None,
                context_window: None,
            },
            super::ModelConfig {
                id: "big".to_owned(),
                display_name: "Big".to_owned(),
                thinking: None,
                compat: None,
                context_window: Some(500_000),
            },
        ],
    });

    let discovered = catalog
        .context_limit(&super::ModelRef {
            provider_id: "one".to_owned(),
            model_id: "m".to_owned(),
        })
        .unwrap();
    assert_eq!(discovered.context, 100_000);
    assert_eq!(discovered.output, Some(32_000));

    let overridden = catalog
        .context_limit(&super::ModelRef {
            provider_id: "one".to_owned(),
            model_id: "big".to_owned(),
        })
        .unwrap();
    assert_eq!(overridden.context, 500_000);
    assert_eq!(overridden.output, None);
}
