use std::{
    env,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::{
    memory::{MemoryConfig, MemoryMode},
    provider::{
        ModelLimit, ModelsDevCatalog, ModelsDevLoad, ModelsDevProviderAlias, OpenAiApi,
        ThinkingCapabilities, ThinkingCompat, ThinkingConfig, ThinkingLevel,
    },
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_TOOL_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_AGENT_TIMEOUT_SECONDS: u64 = 600;
const DEFAULT_MAX_TURNS: usize = 12;
const DEFAULT_MAX_TOOL_OUTPUT_CHARS: usize = 32_000;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 128_000;
const DEFAULT_COMPACT_KEEP_TURNS: usize = 6;
const PROJECT_CONFIG_PATH: &str = ".zex/config.toml";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider_id: String,
    pub model_id: String,
}

impl ModelRef {
    pub fn key(&self) -> String {
        format!("{}/{}", self.provider_id, self.model_id)
    }

    pub fn from_key(value: &str) -> Option<Self> {
        let (provider_id, model_id) = value.split_once('/')?;
        (!provider_id.is_empty() && !model_id.is_empty()).then(|| Self {
            provider_id: provider_id.to_owned(),
            model_id: model_id.to_owned(),
        })
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SecretValue(String);

impl SecretValue {
    pub fn new(value: String) -> Self {
        Self(value)
    }

    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl std::fmt::Debug for SecretValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("<redacted>")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ThinkingCompat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub display_name: String,
    pub base_url: String,
    pub api_key: SecretValue,
    #[serde(default)]
    pub openai_api: OpenAiApi,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ThinkingCompat>,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCatalog {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_model: Option<ModelRef>,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(skip)]
    pub models_dev: ModelsDevCatalog,
    #[serde(skip)]
    pub models_dev_aliases: Vec<ModelsDevProviderAlias>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelChoice {
    pub target: ModelRef,
    pub provider_name: String,
    pub model_name: String,
    pub thinking: ThinkingCapabilities,
}

impl ProviderCatalog {
    pub fn choices(&self) -> Vec<ModelChoice> {
        self.providers
            .iter()
            .flat_map(|provider| {
                provider.models.iter().map(|model| ModelChoice {
                    target: ModelRef {
                        provider_id: provider.id.clone(),
                        model_id: model.id.clone(),
                    },
                    provider_name: provider.display_name.clone(),
                    model_name: model.display_name.clone(),
                    thinking: self.resolved_thinking(provider, model),
                })
            })
            .collect()
    }

    pub fn provider(&self, provider_id: &str) -> Option<&ProviderConfig> {
        self.providers
            .iter()
            .find(|provider| provider.id == provider_id)
    }

    pub fn model(&self, target: &ModelRef) -> Option<(&ProviderConfig, &ModelConfig)> {
        let provider = self.provider(&target.provider_id)?;
        let model = provider
            .models
            .iter()
            .find(|model| model.id == target.model_id)?;
        Some((provider, model))
    }

    pub fn contains(&self, target: &ModelRef) -> bool {
        self.model(target).is_some()
    }

    pub fn thinking_capabilities(&self, target: &ModelRef) -> ThinkingCapabilities {
        self.model(target)
            .map(|(provider, model)| self.resolved_thinking(provider, model))
            .unwrap_or_default()
    }

    pub fn context_limit(&self, target: &ModelRef) -> Option<ModelLimit> {
        let (provider, model) = self.model(target)?;
        if let Some(context) = model.context_window {
            return Some(ModelLimit {
                context,
                output: None,
            });
        }
        self.models_dev.limits(&provider.id, &model.id).or_else(|| {
            self.matched_models_dev_provider(provider)
                .and_then(|provider_id| self.models_dev.limits(provider_id, &model.id))
        })
    }

    fn resolved_thinking(
        &self,
        provider: &ProviderConfig,
        model: &ModelConfig,
    ) -> ThinkingCapabilities {
        let discovered = self
            .models_dev
            .capabilities(&provider.id, &model.id)
            .or_else(|| {
                self.matched_models_dev_provider(provider)
                    .and_then(|provider_id| self.models_dev.capabilities(provider_id, &model.id))
            });
        let mut capabilities = discovered.unwrap_or_default();
        capabilities.apply(provider.thinking.as_ref(), provider.compat.as_ref());
        capabilities.apply(model.thinking.as_ref(), model.compat.as_ref());
        capabilities
    }

    fn matched_models_dev_provider<'a>(&'a self, provider: &ProviderConfig) -> Option<&'a str> {
        let configured_api = normalize_api_url(&provider.base_url)?;
        self.models_dev_aliases
            .iter()
            .find(|candidate| {
                candidate
                    .api
                    .as_deref()
                    .and_then(normalize_api_url)
                    .is_some_and(|api| api == configured_api)
            })
            .map(|candidate| candidate.id.as_str())
    }

    pub fn validate(&self) -> Result<()> {
        let mut provider_ids = std::collections::BTreeSet::new();
        for provider in &self.providers {
            if provider.id.trim().is_empty() {
                bail!("provider ID must not be empty");
            }
            if provider.id.contains('/') {
                bail!("provider ID {:?} must not contain '/'", provider.id);
            }
            if provider.display_name.trim().is_empty() {
                bail!("provider display name must not be empty");
            }
            if provider.base_url.trim().is_empty() {
                bail!("provider base URL must not be empty");
            }
            if provider.api_key.is_empty() {
                bail!("provider {} API key must not be empty", provider.id);
            }
            if !provider_ids.insert(provider.id.as_str()) {
                bail!("provider ID {:?} is duplicated", provider.id);
            }

            let mut model_ids = std::collections::BTreeSet::new();
            for model in &provider.models {
                if model.id.trim().is_empty() {
                    bail!("model ID must not be empty for provider {}", provider.id);
                }
                if model.display_name.trim().is_empty() {
                    bail!(
                        "model display name must not be empty for {}/{}",
                        provider.id,
                        model.id
                    );
                }
                if !model_ids.insert(model.id.as_str()) {
                    bail!(
                        "model ID {:?} is duplicated for provider {}",
                        model.id,
                        provider.id
                    );
                }
                validate_thinking(
                    &format!("model {}/{}", provider.id, model.id),
                    model.thinking.as_ref(),
                    model.compat.as_ref(),
                )?;
            }
            validate_thinking(
                &format!("provider {}", provider.id),
                provider.thinking.as_ref(),
                provider.compat.as_ref(),
            )?;
        }

        if let Some(active_model) = &self.active_model
            && !self.contains(active_model)
        {
            bail!("active model {} is not configured", active_model.key());
        }
        Ok(())
    }

    pub fn has_ready_model(&self) -> bool {
        self.active_model
            .as_ref()
            .is_some_and(|active| self.contains(active))
    }
}

fn validate_thinking(
    owner: &str,
    thinking: Option<&ThinkingConfig>,
    compat: Option<&ThinkingCompat>,
) -> Result<()> {
    if let Some(thinking) = thinking {
        if thinking.min_level == ThinkingLevel::Off {
            bail!("{owner} thinking.min_level must not be off");
        }
        if thinking.min_level > thinking.max_level {
            bail!(
                "{owner} thinking.min_level {} exceeds max_level {}",
                thinking.min_level,
                thinking.max_level
            );
        }
        if let Some(supported) = &thinking.supported {
            if supported.is_empty() {
                bail!("{owner} thinking.supported must not be empty");
            }
            if supported.iter().any(|level| {
                *level != ThinkingLevel::Off
                    && (*level < thinking.min_level || *level > thinking.max_level)
            }) {
                bail!(
                    "{owner} thinking.supported levels must be between {} and {}",
                    thinking.min_level,
                    thinking.max_level
                );
            }
            let unique = supported
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            if unique.len() != supported.len() {
                bail!("{owner} thinking.supported must not contain duplicates");
            }
            if !supported.contains(&thinking.min_level) || !supported.contains(&thinking.max_level)
            {
                bail!("{owner} thinking.supported must contain min_level and max_level");
            }
        }
    }
    if let Some(compat) = compat {
        for (level, value) in &compat.reasoning_effort_map {
            if *level == ThinkingLevel::Off {
                bail!("{owner} reasoning_effort_map must not map off");
            }
            if value.trim().is_empty() {
                bail!("{owner} reasoning_effort_map value for {level} must not be empty");
            }
            if let Some(thinking) = thinking
                && (level < &thinking.min_level
                    || level > &thinking.max_level
                    || thinking
                        .supported
                        .as_ref()
                        .is_some_and(|supported| !supported.contains(level)))
            {
                bail!("{owner} reasoning_effort_map contains unsupported level {level}");
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Config {
    pub providers: ProviderCatalog,
    pub active_model: Option<ModelRef>,
    pub working_dir: PathBuf,
    pub configured: bool,
    pub tool_timeout: Duration,
    pub agent_timeout: Duration,
    pub max_turns: usize,
    pub max_tool_output_chars: usize,
    pub max_context_tokens: usize,
    pub compact_keep_turns: usize,
    pub memory: MemoryConfig,
    pub default_thinking_level: ThinkingLevel,
    pub hide_thinking_block: bool,
    pub theme: ThemeConfig,
}

/// A `[theme]` color value: `"#rgb"`, `"#rrggbb"`, or `"default"`/`"reset"`
/// to follow the terminal's own color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeColor {
    Terminal,
    Rgb(u8, u8, u8),
}

impl ThemeColor {
    pub fn parse(value: &str) -> Result<Self, String> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("default") || value.eq_ignore_ascii_case("reset") {
            return Ok(Self::Terminal);
        }
        let Some(hex) = value.strip_prefix('#') else {
            return Err(format!("{value:?} is not a hex color like #7dcfff"));
        };
        let expanded;
        let hex = match hex.len() {
            3 => {
                expanded = hex
                    .chars()
                    .flat_map(|digit| [digit, digit])
                    .collect::<String>();
                &expanded
            }
            6 => hex,
            _ => {
                return Err(format!(
                    "#{hex} must have 3 or 6 hex digits, like #7df or #7dcfff"
                ));
            }
        };
        let channel = |pair: &str| {
            u8::from_str_radix(pair, 16).map_err(|_| format!("#{hex} contains a non-hex digit"))
        };
        Ok(Self::Rgb(
            channel(&hex[0..2])?,
            channel(&hex[2..4])?,
            channel(&hex[4..6])?,
        ))
    }
}

impl<'de> Deserialize<'de> for ThemeColor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        ThemeColor::parse(&value).map_err(serde::de::Error::custom)
    }
}

/// `[theme]` palette overrides for the TUI. Every key is optional; unset keys
/// keep the built-in defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThemeConfig {
    pub background: Option<ThemeColor>,
    pub surface: Option<ThemeColor>,
    pub surface_hover: Option<ThemeColor>,
    pub surface_raised: Option<ThemeColor>,
    pub text: Option<ThemeColor>,
    pub text_strong: Option<ThemeColor>,
    pub text_dim: Option<ThemeColor>,
    pub text_faint: Option<ThemeColor>,
    pub gray_dim: Option<ThemeColor>,
    pub accent_primary: Option<ThemeColor>,
    pub accent_secondary: Option<ThemeColor>,
    pub accent_user: Option<ThemeColor>,
    pub accent_thinking: Option<ThemeColor>,
    pub accent_tool: Option<ThemeColor>,
    pub border: Option<ThemeColor>,
    pub border_active: Option<ThemeColor>,
    pub ok: Option<ThemeColor>,
    pub bad: Option<ThemeColor>,
    pub command: Option<ThemeColor>,
    pub running: Option<ThemeColor>,
    pub model_accent: Option<ThemeColor>,
    pub md_code: Option<ThemeColor>,
    pub code_bg: Option<ThemeColor>,
    pub diff_add_bg: Option<ThemeColor>,
    pub diff_del_bg: Option<ThemeColor>,
    pub wordmark_ink: Option<ThemeColor>,
}

impl ThemeConfig {
    /// Project config wins key by key over the global config.
    fn merge(self, project: Self) -> Self {
        Self {
            background: project.background.or(self.background),
            surface: project.surface.or(self.surface),
            surface_hover: project.surface_hover.or(self.surface_hover),
            surface_raised: project.surface_raised.or(self.surface_raised),
            text: project.text.or(self.text),
            text_strong: project.text_strong.or(self.text_strong),
            text_dim: project.text_dim.or(self.text_dim),
            text_faint: project.text_faint.or(self.text_faint),
            gray_dim: project.gray_dim.or(self.gray_dim),
            accent_primary: project.accent_primary.or(self.accent_primary),
            accent_secondary: project.accent_secondary.or(self.accent_secondary),
            accent_user: project.accent_user.or(self.accent_user),
            accent_thinking: project.accent_thinking.or(self.accent_thinking),
            accent_tool: project.accent_tool.or(self.accent_tool),
            border: project.border.or(self.border),
            border_active: project.border_active.or(self.border_active),
            ok: project.ok.or(self.ok),
            bad: project.bad.or(self.bad),
            command: project.command.or(self.command),
            running: project.running.or(self.running),
            model_accent: project.model_accent.or(self.model_accent),
            md_code: project.md_code.or(self.md_code),
            code_bg: project.code_bg.or(self.code_bg),
            diff_add_bg: project.diff_add_bg.or(self.diff_add_bg),
            diff_del_bg: project.diff_del_bg.or(self.diff_del_bg),
            wordmark_ink: project.wordmark_ink.or(self.wordmark_ink),
        }
    }
}

impl Config {
    pub async fn load() -> Result<Self> {
        let working_dir =
            env::current_dir().context("failed to determine the working directory")?;
        Self::load_from(&working_dir, &global_config_dir()?).await
    }

    pub async fn session_dir() -> Result<PathBuf> {
        let working_dir =
            env::current_dir().context("failed to determine the working directory")?;
        Self::session_dir_from(&working_dir, &global_config_dir()?).await
    }

    async fn load_from(working_dir: &Path, global_config_dir: &Path) -> Result<Self> {
        let global_path = global_config_dir.join("config.toml");
        let project_path = working_dir.join(PROJECT_CONFIG_PATH);
        let global = read_config_file(&global_path).await?;
        let project = read_config_file(&project_path).await?;
        let file = global.merge(project);

        let legacy_api_key = preferred_env("ZEX_API_KEY", "OPENAI_API_KEY")
            .or_else(|| non_empty_file_value(file.api_key.clone()));
        let legacy_model = preferred_env("ZEX_MODEL", "OPENAI_MODEL")
            .or_else(|| non_empty_file_value(file.model.clone()));
        let legacy_base_url = preferred_env("ZEX_BASE_URL", "OPENAI_BASE_URL")
            .or_else(|| non_empty_file_value(file.base_url.clone()))
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        let legacy_openai_api = env::var("ZEX_OPENAI_API")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.parse())
            .transpose()?
            .or(file.openai_api)
            .unwrap_or(OpenAiApi::ChatCompletions);
        let mut providers = ProviderCatalog {
            active_model: file.active_model,
            providers: file.providers,
            models_dev: ModelsDevCatalog::default(),
            models_dev_aliases: Vec::new(),
        };
        if providers.providers.is_empty()
            && let (Some(api_key), Some(model)) = (legacy_api_key, legacy_model)
        {
            providers.providers.push(ProviderConfig {
                id: "default".to_owned(),
                display_name: "Default".to_owned(),
                base_url: legacy_base_url,
                api_key: SecretValue::new(api_key),
                openai_api: legacy_openai_api,
                thinking: None,
                compat: None,
                models: vec![ModelConfig {
                    display_name: model.clone(),
                    id: model.clone(),
                    thinking: None,
                    compat: None,
                    context_window: None,
                }],
            });
            providers.active_model = Some(ModelRef {
                provider_id: "default".to_owned(),
                model_id: model,
            });
        }
        let (models_dev, models_dev_aliases, models_dev_load) =
            ModelsDevCatalog::load(global_config_dir).await;
        providers.models_dev = models_dev;
        providers.models_dev_aliases = models_dev_aliases;
        match models_dev_load {
            ModelsDevLoad::Cached => {
                eprintln!("Zex: models.dev refresh failed; using cached model capabilities");
            }
            ModelsDevLoad::Unavailable => {
                eprintln!("Zex: models.dev capabilities unavailable; using safe defaults");
            }
            ModelsDevLoad::Refreshed => {}
        }
        providers.validate()?;
        let active_model = providers.active_model.clone();
        let configured = providers.has_ready_model();

        Ok(Self {
            providers,
            active_model,
            working_dir: working_dir.to_path_buf(),
            configured,
            tool_timeout: Duration::from_secs(positive_u64(
                "tool_timeout_seconds",
                env_or_file(
                    "ZEX_TOOL_TIMEOUT_SECONDS",
                    file.tool_timeout_seconds,
                    DEFAULT_TOOL_TIMEOUT_SECONDS,
                )?,
            )?),
            agent_timeout: Duration::from_secs(positive_u64(
                "agent_timeout_seconds",
                env_or_file(
                    "ZEX_AGENT_TIMEOUT_SECONDS",
                    file.agent_timeout_seconds,
                    DEFAULT_AGENT_TIMEOUT_SECONDS,
                )?,
            )?),
            max_turns: positive(
                "max_turns",
                env_or_file("ZEX_MAX_TURNS", file.max_turns, DEFAULT_MAX_TURNS)?,
            )?,
            max_tool_output_chars: positive(
                "max_tool_output_chars",
                env_or_file(
                    "ZEX_MAX_TOOL_OUTPUT_CHARS",
                    file.max_tool_output_chars,
                    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
                )?,
            )?,
            max_context_tokens: positive(
                "max_context_tokens",
                env_or_file_legacy(
                    "ZEX_MAX_CONTEXT_TOKENS",
                    "ZEX_MAX_CONTEXT_CHARS",
                    file.max_context_tokens,
                    DEFAULT_MAX_CONTEXT_TOKENS,
                )?,
            )?,
            compact_keep_turns: positive(
                "compact_keep_turns",
                env_or_file(
                    "ZEX_COMPACT_KEEP_TURNS",
                    file.compact_keep_turns,
                    DEFAULT_COMPACT_KEEP_TURNS,
                )?,
            )?,
            memory: {
                let defaults = MemoryConfig::default();
                MemoryConfig {
                    enabled: env_or_file(
                        "ZEX_MEMORY_ENABLED",
                        file.memory.enabled,
                        defaults.enabled,
                    )?,
                    mode: env_or_file("ZEX_MEMORY_MODE", file.memory.mode, MemoryMode::default())?,
                    recall_rate_limit: positive(
                        "memory.recall_rate_limit",
                        env_or_file(
                            "ZEX_MEMORY_RECALL_RATE_LIMIT",
                            file.memory.recall_rate_limit,
                            defaults.recall_rate_limit,
                        )?,
                    )?,
                    max_recall_tokens: positive(
                        "memory.max_recall_tokens",
                        env_or_file(
                            "ZEX_MEMORY_MAX_RECALL_TOKENS",
                            file.memory.max_recall_tokens,
                            defaults.max_recall_tokens,
                        )?,
                    )?,
                    hot_cache_size: positive(
                        "memory.hot_cache_size",
                        env_or_file(
                            "ZEX_MEMORY_HOT_CACHE_SIZE",
                            file.memory.hot_cache_size,
                            defaults.hot_cache_size,
                        )?,
                    )?,
                    auto_pin_important: env_or_file(
                        "ZEX_MEMORY_AUTO_PIN_IMPORTANT",
                        file.memory.auto_pin_important,
                        defaults.auto_pin_important,
                    )?,
                }
            },
            default_thinking_level: env_or_file(
                "ZEX_DEFAULT_THINKING_LEVEL",
                file.default_thinking_level,
                ThinkingLevel::default(),
            )?,
            hide_thinking_block: env_or_file(
                "ZEX_HIDE_THINKING_BLOCK",
                file.hide_thinking_block,
                false,
            )?,
            theme: file.theme,
        })
    }

    async fn session_dir_from(working_dir: &Path, global_config_dir: &Path) -> Result<PathBuf> {
        let global = read_config_file(&global_config_dir.join("config.toml")).await?;
        let project = read_config_file(&working_dir.join(PROJECT_CONFIG_PATH)).await?;
        Ok(resolve_session_dir(
            working_dir,
            global_config_dir,
            global.merge(project).session_dir,
        ))
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    openai_api: Option<OpenAiApi>,
    active_model: Option<ModelRef>,
    #[serde(default)]
    providers: Vec<ProviderConfig>,
    max_turns: Option<usize>,
    tool_timeout_seconds: Option<u64>,
    agent_timeout_seconds: Option<u64>,
    max_tool_output_chars: Option<usize>,
    #[serde(alias = "max_context_chars")]
    max_context_tokens: Option<usize>,
    compact_keep_turns: Option<usize>,
    default_thinking_level: Option<ThinkingLevel>,
    hide_thinking_block: Option<bool>,
    session_dir: Option<String>,
    #[serde(default)]
    memory: FileMemoryConfig,
    #[serde(default)]
    theme: ThemeConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileMemoryConfig {
    enabled: Option<bool>,
    mode: Option<MemoryMode>,
    recall_rate_limit: Option<usize>,
    max_recall_tokens: Option<usize>,
    hot_cache_size: Option<usize>,
    auto_pin_important: Option<bool>,
}

impl FileMemoryConfig {
    fn merge(self, project: Self) -> Self {
        Self {
            enabled: project.enabled.or(self.enabled),
            mode: project.mode.or(self.mode),
            recall_rate_limit: project.recall_rate_limit.or(self.recall_rate_limit),
            max_recall_tokens: project.max_recall_tokens.or(self.max_recall_tokens),
            hot_cache_size: project.hot_cache_size.or(self.hot_cache_size),
            auto_pin_important: project.auto_pin_important.or(self.auto_pin_important),
        }
    }
}

impl FileConfig {
    fn merge(self, project: Self) -> Self {
        Self {
            api_key: project.api_key.or(self.api_key),
            model: project.model.or(self.model),
            base_url: project.base_url.or(self.base_url),
            openai_api: project.openai_api.or(self.openai_api),
            active_model: project.active_model.or(self.active_model),
            providers: if project.providers.is_empty() {
                self.providers
            } else {
                project.providers
            },
            max_turns: project.max_turns.or(self.max_turns),
            tool_timeout_seconds: project.tool_timeout_seconds.or(self.tool_timeout_seconds),
            agent_timeout_seconds: project.agent_timeout_seconds.or(self.agent_timeout_seconds),
            max_tool_output_chars: project.max_tool_output_chars.or(self.max_tool_output_chars),
            max_context_tokens: project.max_context_tokens.or(self.max_context_tokens),
            compact_keep_turns: project.compact_keep_turns.or(self.compact_keep_turns),
            default_thinking_level: project
                .default_thinking_level
                .or(self.default_thinking_level),
            hide_thinking_block: project.hide_thinking_block.or(self.hide_thinking_block),
            session_dir: project.session_dir.or(self.session_dir),
            memory: self.memory.merge(project.memory),
            theme: self.theme.merge(project.theme),
        }
    }
}

pub async fn persist_active_model(working_dir: &Path, active_model: &ModelRef) -> Result<()> {
    update_project_config(working_dir, |table| {
        table.insert(
            "active_model".to_owned(),
            toml::Value::try_from(active_model).expect("ModelRef is TOML serializable"),
        );
    })
    .await
}

pub async fn persist_provider_catalog(working_dir: &Path, catalog: &ProviderCatalog) -> Result<()> {
    catalog.validate()?;
    update_project_config(working_dir, |table| {
        table.insert(
            "providers".to_owned(),
            toml::Value::try_from(&catalog.providers).expect("providers are TOML serializable"),
        );
        match &catalog.active_model {
            Some(active_model) => {
                table.insert(
                    "active_model".to_owned(),
                    toml::Value::try_from(active_model).expect("ModelRef is TOML serializable"),
                );
            }
            None => {
                table.remove("active_model");
            }
        }
        table.remove("api_key");
        table.remove("base_url");
        table.remove("model");
        table.remove("openai_api");
    })
    .await
}

pub async fn persist_thinking_level(
    working_dir: &Path,
    thinking_level: ThinkingLevel,
) -> Result<()> {
    update_project_config(working_dir, |table| {
        table.insert(
            "default_thinking_level".to_owned(),
            toml::Value::String(thinking_level.to_string()),
        );
    })
    .await
}

pub async fn persist_show_thinking(working_dir: &Path, show_thinking: bool) -> Result<()> {
    update_project_config(working_dir, |table| {
        table.insert(
            "hide_thinking_block".to_owned(),
            toml::Value::Boolean(!show_thinking),
        );
    })
    .await
}

async fn update_project_config(
    working_dir: &Path,
    update: impl FnOnce(&mut toml::Table),
) -> Result<()> {
    let path = working_dir.join(PROJECT_CONFIG_PATH);
    let content = match tokio::fs::read_to_string(&path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read config file {}", path.display()));
        }
    };
    let mut table = if content.trim().is_empty() {
        toml::Table::new()
    } else {
        toml::from_str::<toml::Table>(&content)
            .with_context(|| format!("failed to parse config file {}", path.display()))?
    };
    update(&mut table);
    let serialized =
        toml::to_string_pretty(&table).context("failed to serialize project config")?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temporary_path = path.with_extension("toml.tmp");
    tokio::fs::write(&temporary_path, serialized)
        .await
        .with_context(|| format!("failed to write config file {}", temporary_path.display()))?;
    match tokio::fs::rename(&temporary_path, &path).await {
        Ok(()) => Ok(()),
        Err(_error) if cfg!(windows) && path.exists() => {
            tokio::fs::remove_file(&path)
                .await
                .with_context(|| format!("failed to replace config file {}", path.display()))?;
            tokio::fs::rename(&temporary_path, &path)
                .await
                .with_context(|| format!("failed to replace config file {}", path.display()))
        }
        Err(error) => {
            Err(error).with_context(|| format!("failed to replace config file {}", path.display()))
        }
    }
}

fn global_config_dir() -> Result<PathBuf> {
    if let Some(directory) = env::var_os("ZEX_CONFIG_DIR").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(directory));
    }

    let base = if cfg!(target_os = "windows") {
        ProjectDirs::from("", "", "zex").map(|directories| directories.config_dir().to_path_buf())
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME").map(|home| {
            PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("zex")
        })
    } else {
        env::var_os("XDG_CONFIG_HOME")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("HOME")
                    .filter(|value| !value.is_empty())
                    .map(|home| PathBuf::from(home).join(".config"))
            })
            .map(|directory| directory.join("zex"))
    };
    base.context("failed to determine the platform config directory")
}

async fn read_config_file(path: &Path) -> Result<FileConfig> {
    let content = match tokio::fs::read_to_string(path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(FileConfig::default());
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read config file {}", path.display()));
        }
    };

    toml::from_str(&content)
        .with_context(|| format!("failed to parse config file {}", path.display()))
}

fn resolve_path(working_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        working_dir.join(path)
    }
}

fn resolve_session_dir(
    working_dir: &Path,
    global_config_dir: &Path,
    file_value: Option<String>,
) -> PathBuf {
    env::var_os("ZEX_SESSION_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| non_empty_file_value(file_value).map(PathBuf::from))
        .map(|path| resolve_path(working_dir, path))
        .unwrap_or_else(|| global_config_dir.join("sessions"))
}

fn preferred_env(primary: &str, fallback: &str) -> Option<String> {
    env::var(primary)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var(fallback)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
}

fn non_empty_file_value(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

fn normalize_api_url(value: &str) -> Option<String> {
    let value = value
        .trim()
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .trim_end_matches('/');
    (!value.is_empty()).then(|| value.to_ascii_lowercase())
}

fn positive(name: &str, value: usize) -> Result<usize> {
    if value == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(value)
}

fn positive_u64(name: &str, value: u64) -> Result<u64> {
    if value == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(value)
}

fn env_or_file_legacy<T>(
    name: &str,
    legacy_name: &str,
    file_value: Option<T>,
    default: T,
) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if matches!(env::var(name), Err(env::VarError::NotPresent)) {
        env_or_file(legacy_name, file_value, default)
    } else {
        env_or_file(name, file_value, default)
    }
}

fn env_or_file<T>(name: &str, file_value: Option<T>, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(value) => {
            if value.trim().is_empty() {
                bail!("{name} cannot be empty");
            }
            value
                .parse()
                .map_err(|error| anyhow::anyhow!("{name} contains an invalid value: {error}"))
        }
        Err(env::VarError::NotPresent) => Ok(file_value.unwrap_or(default)),
        Err(error) => Err(error).with_context(|| format!("failed to read {name}")),
    }
}

#[cfg(test)]
mod tests;
