mod openai;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fmt,
    str::FromStr,
    sync::{Arc, RwLock},
    time::Duration,
};

use crate::agent::{AssistantMessage, EventSender, Message};
use crate::config::{ModelConfig, ModelRef, ProviderCatalog};

pub use openai::OpenAiProvider;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Low,
    #[default]
    Medium,
    High,
}

impl ThinkingLevel {
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::Off,
        }
    }

    pub fn as_provider_value(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
        }
    }
}

impl FromStr for ThinkingLevel {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            _ => anyhow::bail!(
                "thinking level must be one of off, low, medium, or high; got {value:?}"
            ),
        }
    }
}

impl fmt::Display for ThinkingLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        })
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAiApi {
    #[default]
    ChatCompletions,
    Responses,
}

impl OpenAiApi {
    pub fn endpoint(self) -> &'static str {
        match self {
            Self::ChatCompletions => "chat/completions",
            Self::Responses => "responses",
        }
    }
}

impl FromStr for OpenAiApi {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "chat-completions" | "chat_completions" | "chat" => Ok(Self::ChatCompletions),
            "responses" | "response" => Ok(Self::Responses),
            _ => anyhow::bail!(
                "ZEX_OPENAI_API must be 'chat-completions' or 'responses', got {value:?}"
            ),
        }
    }
}

impl fmt::Display for OpenAiApi {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatCompletions => formatter.write_str("chat-completions"),
            Self::Responses => formatter.write_str("responses"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

pub trait Provider: Send + Sync {
    fn supports_thinking(&self, _model: &str) -> bool {
        false
    }

    async fn complete(
        &self,
        model: &str,
        thinking_level: Option<ThinkingLevel>,
        messages: &[Message],
        tools: &[ToolDefinition],
        events: &EventSender,
    ) -> Result<AssistantMessage>;
}

#[derive(Clone)]
pub struct ProviderRegistry {
    inner: Arc<RwLock<ProviderRegistryState>>,
    request_timeout: Duration,
}

struct ProviderRegistryState {
    providers: BTreeMap<String, OpenAiProvider>,
    models: BTreeMap<String, ModelConfig>,
}

pub struct ProviderRegistryUpdate(ProviderRegistryState);

impl ProviderRegistry {
    pub fn new(catalog: &ProviderCatalog, request_timeout: Duration) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(Self::build_state(catalog, request_timeout)?)),
            request_timeout,
        })
    }

    pub fn prepare_update(&self, catalog: &ProviderCatalog) -> Result<ProviderRegistryUpdate> {
        Ok(ProviderRegistryUpdate(Self::build_state(
            catalog,
            self.request_timeout,
        )?))
    }

    pub fn apply_update(&self, update: ProviderRegistryUpdate) -> Result<()> {
        *self
            .inner
            .write()
            .map_err(|_| anyhow::anyhow!("provider registry lock is poisoned"))? = update.0;
        Ok(())
    }

    fn build_state(
        catalog: &ProviderCatalog,
        request_timeout: Duration,
    ) -> Result<ProviderRegistryState> {
        let mut providers = BTreeMap::new();
        let mut models = BTreeMap::new();
        for provider in &catalog.providers {
            providers.insert(
                provider.id.clone(),
                OpenAiProvider::new(
                    &provider.base_url,
                    provider.api_key.expose().to_owned(),
                    provider.openai_api,
                    request_timeout,
                )?,
            );
            for model in &provider.models {
                models.insert(
                    ModelRef {
                        provider_id: provider.id.clone(),
                        model_id: model.id.clone(),
                    }
                    .key(),
                    model.clone(),
                );
            }
        }
        Ok(ProviderRegistryState { providers, models })
    }
}

impl Provider for ProviderRegistry {
    fn supports_thinking(&self, model: &str) -> bool {
        self.inner
            .read()
            .ok()
            .and_then(|state| state.models.get(model).map(|model| model.supports_thinking))
            .unwrap_or(false)
    }

    async fn complete(
        &self,
        model: &str,
        thinking_level: Option<ThinkingLevel>,
        messages: &[Message],
        tools: &[ToolDefinition],
        events: &EventSender,
    ) -> Result<AssistantMessage> {
        if model.is_empty() {
            anyhow::bail!("no active model configured; use /provider, then /model");
        }
        let (provider_id, model_id) = model
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("invalid model target {model:?}"))?;
        let (provider, configured) =
            {
                let state = self
                    .inner
                    .read()
                    .map_err(|_| anyhow::anyhow!("provider registry lock is poisoned"))?;
                (
                    state.providers.get(provider_id).cloned().ok_or_else(|| {
                        anyhow::anyhow!("provider {provider_id:?} is not configured")
                    })?,
                    state.models.get(model).cloned().ok_or_else(|| {
                        anyhow::anyhow!("model target {model:?} is not configured")
                    })?,
                )
            };
        provider
            .complete(
                model_id,
                configured.supports_thinking.then(|| {
                    thinking_level
                        .or(configured.default_thinking_level)
                        .unwrap_or_default()
                }),
                messages,
                tools,
                events,
            )
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{OpenAiApi, Provider, ProviderRegistry, ThinkingLevel};
    use crate::config::{ModelConfig, ModelRef, ProviderCatalog, ProviderConfig, SecretValue};

    fn catalog() -> ProviderCatalog {
        let active_model = ModelRef {
            provider_id: "one".to_owned(),
            model_id: "reasoning".to_owned(),
        };
        ProviderCatalog {
            active_model: Some(active_model),
            providers: vec![
                ProviderConfig {
                    id: "one".to_owned(),
                    display_name: "One".to_owned(),
                    base_url: "https://one.example/v1".to_owned(),
                    api_key: SecretValue::new("one-secret".to_owned()),
                    openai_api: OpenAiApi::Responses,
                    models: vec![ModelConfig {
                        id: "reasoning".to_owned(),
                        display_name: "Reasoning".to_owned(),
                        supports_thinking: true,
                        default_thinking_level: Some(ThinkingLevel::High),
                    }],
                },
                ProviderConfig {
                    id: "two".to_owned(),
                    display_name: "Two".to_owned(),
                    base_url: "https://two.example/v1".to_owned(),
                    api_key: SecretValue::new("two-secret".to_owned()),
                    openai_api: OpenAiApi::ChatCompletions,
                    models: vec![ModelConfig {
                        id: "fast".to_owned(),
                        display_name: "Fast".to_owned(),
                        supports_thinking: false,
                        default_thinking_level: None,
                    }],
                },
            ],
        }
    }

    #[test]
    fn registry_resolves_model_capabilities_by_provider_and_model() {
        let registry = ProviderRegistry::new(&catalog(), Duration::from_secs(1)).unwrap();

        assert!(registry.supports_thinking("one/reasoning"));
        assert!(!registry.supports_thinking("two/fast"));
        assert!(!registry.supports_thinking("one/fast"));
    }

    #[test]
    fn registry_replacement_is_visible_to_existing_clones() {
        let registry = ProviderRegistry::new(&catalog(), Duration::from_secs(1)).unwrap();
        let shared = registry.clone();
        let mut updated = catalog();
        updated.providers[1].models[0].supports_thinking = true;
        updated.providers[1].models[0].default_thinking_level = Some(ThinkingLevel::Low);

        let update = registry.prepare_update(&updated).unwrap();
        assert!(!shared.supports_thinking("two/fast"));

        registry.apply_update(update).unwrap();

        assert!(shared.supports_thinking("two/fast"));
    }
}
