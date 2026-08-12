use std::{
    env,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::provider::{OpenAiApi, ThinkingLevel};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_TOOL_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_AGENT_TIMEOUT_SECONDS: u64 = 600;
const DEFAULT_MAX_TURNS: usize = 12;
const DEFAULT_MAX_TOOL_OUTPUT_CHARS: usize = 32_000;
const DEFAULT_MAX_CONTEXT_CHARS: usize = 120_000;
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
    #[serde(default)]
    pub supports_thinking: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_thinking_level: Option<ThinkingLevel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub display_name: String,
    pub base_url: String,
    pub api_key: SecretValue,
    #[serde(default)]
    pub openai_api: OpenAiApi,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCatalog {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_model: Option<ModelRef>,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelChoice {
    pub target: ModelRef,
    pub provider_name: String,
    pub model_name: String,
    pub supports_thinking: bool,
    pub default_thinking_level: Option<ThinkingLevel>,
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
                    supports_thinking: model.supports_thinking,
                    default_thinking_level: model.default_thinking_level,
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
                if !model.supports_thinking && model.default_thinking_level.is_some() {
                    bail!(
                        "model {}/{} cannot define a thinking level when thinking is disabled",
                        provider.id,
                        model.id
                    );
                }
            }
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
    pub max_context_chars: usize,
    pub compact_keep_turns: usize,
    pub thinking_level: ThinkingLevel,
    pub show_thinking: bool,
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
        };
        if providers.providers.is_empty()
            && let (Some(api_key), Some(model)) = (legacy_api_key, legacy_model)
        {
            let supports_thinking = legacy_model_supports_thinking(&model);
            providers.providers.push(ProviderConfig {
                id: "default".to_owned(),
                display_name: "Default".to_owned(),
                base_url: legacy_base_url,
                api_key: SecretValue::new(api_key),
                openai_api: legacy_openai_api,
                models: vec![ModelConfig {
                    display_name: model.clone(),
                    id: model.clone(),
                    supports_thinking,
                    default_thinking_level: supports_thinking.then_some(ThinkingLevel::default()),
                }],
            });
            providers.active_model = Some(ModelRef {
                provider_id: "default".to_owned(),
                model_id: model,
            });
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
            max_context_chars: positive(
                "max_context_chars",
                env_or_file(
                    "ZEX_MAX_CONTEXT_CHARS",
                    file.max_context_chars,
                    DEFAULT_MAX_CONTEXT_CHARS,
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
            thinking_level: env_or_file(
                "ZEX_THINKING_LEVEL",
                file.thinking_level,
                ThinkingLevel::default(),
            )?,
            show_thinking: env_or_file("ZEX_SHOW_THINKING", file.show_thinking, true)?,
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
    max_context_chars: Option<usize>,
    compact_keep_turns: Option<usize>,
    thinking_level: Option<ThinkingLevel>,
    show_thinking: Option<bool>,
    session_dir: Option<String>,
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
            max_context_chars: project.max_context_chars.or(self.max_context_chars),
            compact_keep_turns: project.compact_keep_turns.or(self.compact_keep_turns),
            thinking_level: project.thinking_level.or(self.thinking_level),
            show_thinking: project.show_thinking.or(self.show_thinking),
            session_dir: project.session_dir.or(self.session_dir),
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
            "thinking_level".to_owned(),
            toml::Value::String(thinking_level.to_string()),
        );
    })
    .await
}

pub async fn persist_show_thinking(working_dir: &Path, show_thinking: bool) -> Result<()> {
    update_project_config(working_dir, |table| {
        table.insert(
            "show_thinking".to_owned(),
            toml::Value::Boolean(show_thinking),
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

fn legacy_model_supports_thinking(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("gpt-5")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.contains("reasoner")
        || model.contains("thinking")
        || model.contains("claude")
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
mod tests {
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
        "ZEX_COMPACT_KEEP_TURNS",
        "ZEX_THINKING_LEVEL",
        "ZEX_SHOW_THINKING",
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
supports_thinking = true

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
        assert!(content.contains("thinking_level = \"high\""));
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
show_thinking = false
"#,
        )
        .await;

        let config = Config::load_from(&project, &global).await.unwrap();
        assert!(!config.show_thinking);

        super::persist_show_thinking(&project, true).await.unwrap();
        let content = tokio::fs::read_to_string(project.join(".zex/config.toml"))
            .await
            .unwrap();
        assert!(content.contains("show_thinking = true"));
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
            providers: vec![super::ProviderConfig {
                id: "openai".to_owned(),
                display_name: "OpenAI".to_owned(),
                base_url: "https://api.openai.com/v1".to_owned(),
                api_key: super::SecretValue::new("secret".to_owned()),
                openai_api: crate::provider::OpenAiApi::Responses,
                models: vec![super::ModelConfig {
                    id: "gpt-5".to_owned(),
                    display_name: "GPT-5".to_owned(),
                    supports_thinking: true,
                    default_thinking_level: Some(crate::provider::ThinkingLevel::High),
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
}
