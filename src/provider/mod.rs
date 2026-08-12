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

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    XHigh,
    Max,
}

impl ThinkingLevel {
    pub const ALL: [Self; 7] = [
        Self::Off,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::XHigh,
        Self::Max,
    ];

    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Minimal,
            Self::Minimal => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::XHigh,
            Self::XHigh => Self::Max,
            Self::Max => Self::Off,
        }
    }

    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}

impl FromStr for ThinkingLevel {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "minimal" => Ok(Self::Minimal),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            _ => anyhow::bail!(
                "thinking level must be one of off, minimal, low, medium, high, xhigh, or max; got {value:?}"
            ),
        }
    }
}

impl fmt::Display for ThinkingLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        })
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    #[default]
    Effort,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThinkingConfig {
    pub min_level: ThinkingLevel,
    pub max_level: ThinkingLevel,
    #[serde(default)]
    pub mode: ThinkingMode,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            min_level: ThinkingLevel::Low,
            max_level: ThinkingLevel::High,
            mode: ThinkingMode::Effort,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThinkingCompat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub reasoning_effort_map: BTreeMap<ThinkingLevel, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_interleaved_thinking: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingCapabilities {
    pub min_level: ThinkingLevel,
    pub max_level: ThinkingLevel,
    pub mode: ThinkingMode,
    pub supports_reasoning_effort: bool,
    pub reasoning_effort_map: BTreeMap<ThinkingLevel, String>,
    pub supports_interleaved_thinking: bool,
}

impl Default for ThinkingCapabilities {
    fn default() -> Self {
        Self {
            min_level: ThinkingLevel::Low,
            max_level: ThinkingLevel::High,
            mode: ThinkingMode::Effort,
            supports_reasoning_effort: true,
            reasoning_effort_map: [
                (ThinkingLevel::Low, "low".to_owned()),
                (ThinkingLevel::Medium, "medium".to_owned()),
                (ThinkingLevel::High, "high".to_owned()),
            ]
            .into_iter()
            .collect(),
            supports_interleaved_thinking: false,
        }
    }
}

impl ThinkingCapabilities {
    pub fn available_levels(&self) -> Vec<ThinkingLevel> {
        std::iter::once(ThinkingLevel::Off)
            .chain(
                ThinkingLevel::ALL
                    .into_iter()
                    .filter(|level| *level >= self.min_level && *level <= self.max_level),
            )
            .collect()
    }

    pub(crate) fn apply(
        &mut self,
        thinking: Option<&ThinkingConfig>,
        compat: Option<&ThinkingCompat>,
    ) {
        if let Some(thinking) = thinking {
            self.min_level = thinking.min_level;
            self.max_level = thinking.max_level;
            self.mode = thinking.mode;
        }
        if let Some(compat) = compat {
            if let Some(supports_reasoning_effort) = compat.supports_reasoning_effort {
                self.supports_reasoning_effort = supports_reasoning_effort;
            }
            self.reasoning_effort_map
                .extend(compat.reasoning_effort_map.clone());
            if let Some(supports_interleaved_thinking) = compat.supports_interleaved_thinking {
                self.supports_interleaved_thinking = supports_interleaved_thinking;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedThinking {
    pub requested: ThinkingLevel,
    pub clamped: ThinkingLevel,
    pub provider_value: Option<String>,
}

pub fn normalize_thinking_level(
    capabilities: &ThinkingCapabilities,
    requested: ThinkingLevel,
) -> NormalizedThinking {
    if requested == ThinkingLevel::Off || !capabilities.supports_reasoning_effort {
        return NormalizedThinking {
            requested,
            clamped: ThinkingLevel::Off,
            provider_value: None,
        };
    }

    let clamped = requested.clamp(capabilities.min_level, capabilities.max_level);
    let provider_value = capabilities
        .reasoning_effort_map
        .get(&clamped)
        .cloned()
        .or_else(|| match clamped {
            ThinkingLevel::Max | ThinkingLevel::XHigh => fallback_levels(clamped)
                .into_iter()
                .skip(1)
                .find_map(|level| capabilities.reasoning_effort_map.get(&level).cloned()),
            _ => None,
        });
    NormalizedThinking {
        requested,
        clamped,
        provider_value,
    }
}

fn fallback_levels(level: ThinkingLevel) -> Vec<ThinkingLevel> {
    let mut levels = vec![level];
    match level {
        ThinkingLevel::Max => {
            levels.extend([
                ThinkingLevel::XHigh,
                ThinkingLevel::High,
                ThinkingLevel::Medium,
                ThinkingLevel::Low,
            ]);
        }
        ThinkingLevel::XHigh => levels.push(ThinkingLevel::High),
        ThinkingLevel::Off
        | ThinkingLevel::Minimal
        | ThinkingLevel::Low
        | ThinkingLevel::Medium
        | ThinkingLevel::High => {}
    }
    levels
}

#[cfg(test)]
mod thinking_tests {
    use super::{
        ThinkingCapabilities, ThinkingCompat, ThinkingConfig, ThinkingLevel, ThinkingMode,
        normalize_thinking_level, sanitize_messages,
    };
    use crate::agent::{Message, ToolCall};

    #[test]
    fn parses_and_orders_the_fixed_thinking_ladder() {
        let parsed = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
            .map(|level| level.parse::<ThinkingLevel>().unwrap());

        assert_eq!(parsed, ThinkingLevel::ALL);
        assert!(ThinkingLevel::Minimal < ThinkingLevel::Max);
        assert_eq!(ThinkingLevel::Max.to_string(), "max");
    }

    #[test]
    fn safe_default_clamps_max_to_high() {
        let normalized =
            normalize_thinking_level(&ThinkingCapabilities::default(), ThinkingLevel::Max);

        assert_eq!(normalized.requested, ThinkingLevel::Max);
        assert_eq!(normalized.clamped, ThinkingLevel::High);
        assert_eq!(normalized.provider_value.as_deref(), Some("high"));
    }

    #[test]
    fn custom_map_supports_max_and_falls_back_without_passthrough() {
        let mut capabilities = ThinkingCapabilities::default();
        capabilities.apply(
            Some(&ThinkingConfig {
                min_level: ThinkingLevel::Minimal,
                max_level: ThinkingLevel::Max,
                mode: ThinkingMode::Effort,
            }),
            Some(&ThinkingCompat {
                supports_reasoning_effort: Some(true),
                reasoning_effort_map: [
                    (ThinkingLevel::XHigh, "extra".to_owned()),
                    (ThinkingLevel::Max, "max".to_owned()),
                ]
                .into_iter()
                .collect(),
                supports_interleaved_thinking: None,
            }),
        );
        let max = normalize_thinking_level(&capabilities, ThinkingLevel::Max);
        assert_eq!(max.provider_value.as_deref(), Some("max"));

        capabilities
            .reasoning_effort_map
            .remove(&ThinkingLevel::Max);
        let fallback = normalize_thinking_level(&capabilities, ThinkingLevel::Max);
        assert_eq!(fallback.clamped, ThinkingLevel::Max);
        assert_eq!(fallback.provider_value.as_deref(), Some("extra"));

        capabilities
            .reasoning_effort_map
            .remove(&ThinkingLevel::XHigh);
        let high_fallback = normalize_thinking_level(&capabilities, ThinkingLevel::Max);
        assert_eq!(high_fallback.provider_value.as_deref(), Some("high"));

        capabilities.reasoning_effort_map.clear();
        let omitted = normalize_thinking_level(&capabilities, ThinkingLevel::Max);
        assert_eq!(omitted.provider_value, None);
    }

    #[test]
    fn provider_defaults_merge_with_model_overrides() {
        let mut capabilities = ThinkingCapabilities::default();
        capabilities.apply(
            Some(&ThinkingConfig {
                min_level: ThinkingLevel::Low,
                max_level: ThinkingLevel::XHigh,
                mode: ThinkingMode::Effort,
            }),
            Some(&ThinkingCompat {
                supports_reasoning_effort: Some(true),
                reasoning_effort_map: [(ThinkingLevel::XHigh, "extra-high".to_owned())]
                    .into_iter()
                    .collect(),
                supports_interleaved_thinking: Some(true),
            }),
        );
        capabilities.apply(
            Some(&ThinkingConfig {
                min_level: ThinkingLevel::Minimal,
                max_level: ThinkingLevel::Max,
                mode: ThinkingMode::Effort,
            }),
            Some(&ThinkingCompat {
                supports_reasoning_effort: None,
                reasoning_effort_map: [(ThinkingLevel::Max, "maximum".to_owned())]
                    .into_iter()
                    .collect(),
                supports_interleaved_thinking: None,
            }),
        );

        assert_eq!(capabilities.min_level, ThinkingLevel::Minimal);
        assert_eq!(capabilities.max_level, ThinkingLevel::Max);
        assert!(capabilities.supports_interleaved_thinking);
        assert_eq!(
            capabilities
                .reasoning_effort_map
                .get(&ThinkingLevel::XHigh)
                .map(String::as_str),
            Some("extra-high")
        );
        assert_eq!(
            capabilities
                .reasoning_effort_map
                .get(&ThinkingLevel::Max)
                .map(String::as_str),
            Some("maximum")
        );
    }

    #[test]
    fn off_and_explicitly_disabled_effort_are_omitted() {
        let off = normalize_thinking_level(&ThinkingCapabilities::default(), ThinkingLevel::Off);
        assert_eq!(off.provider_value, None);

        let disabled = ThinkingCapabilities {
            supports_reasoning_effort: false,
            ..ThinkingCapabilities::default()
        };
        let normalized = normalize_thinking_level(&disabled, ThinkingLevel::High);
        assert_eq!(normalized.clamped, ThinkingLevel::Off);
        assert_eq!(normalized.provider_value, None);
    }

    #[test]
    fn unmapped_non_extended_levels_are_never_substituted() {
        let mut capabilities = ThinkingCapabilities::default();
        capabilities
            .reasoning_effort_map
            .remove(&ThinkingLevel::Medium);

        let normalized = normalize_thinking_level(&capabilities, ThinkingLevel::Medium);

        assert_eq!(normalized.clamped, ThinkingLevel::Medium);
        assert_eq!(normalized.provider_value, None);
    }

    #[test]
    fn request_history_keeps_reasoning_only_for_interleaved_tool_turns() {
        let tool_turn = Message::Assistant {
            content: String::new(),
            thinking: Some("Need the tool.".to_owned()),
            tool_calls: vec![ToolCall {
                id: "call".to_owned(),
                name: "read".to_owned(),
                arguments: "{}".to_owned(),
            }],
            provider_state: Some(serde_json::json!({"reasoning_content": "Need the tool."})),
        };
        let final_turn = Message::Assistant {
            content: "Done".to_owned(),
            thinking: Some("Finished.".to_owned()),
            tool_calls: Vec::new(),
            provider_state: Some(serde_json::json!({"reasoning_content": "Finished."})),
        };

        let interleaved = sanitize_messages(&[tool_turn.clone(), final_turn.clone()], true);
        assert!(matches!(
            &interleaved[0],
            Message::Assistant {
                thinking: Some(_),
                provider_state: Some(_),
                ..
            }
        ));
        assert!(matches!(
            &interleaved[1],
            Message::Assistant {
                thinking: None,
                provider_state: None,
                ..
            }
        ));

        let stripped = sanitize_messages(&[tool_turn, final_turn], false);
        assert!(stripped.iter().all(|message| matches!(
            message,
            Message::Assistant {
                thinking: None,
                provider_state: None,
                ..
            }
        )));
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
    fn thinking_capabilities(&self, _model: &str) -> ThinkingCapabilities {
        ThinkingCapabilities::default()
    }

    async fn complete(
        &self,
        model: &str,
        thinking_level: ThinkingLevel,
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
    models: BTreeMap<String, (ModelConfig, ThinkingCapabilities)>,
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
                let capabilities = catalog.thinking_capabilities(&ModelRef {
                    provider_id: provider.id.clone(),
                    model_id: model.id.clone(),
                });
                models.insert(
                    ModelRef {
                        provider_id: provider.id.clone(),
                        model_id: model.id.clone(),
                    }
                    .key(),
                    (model.clone(), capabilities),
                );
            }
        }
        Ok(ProviderRegistryState { providers, models })
    }
}

impl Provider for ProviderRegistry {
    fn thinking_capabilities(&self, model: &str) -> ThinkingCapabilities {
        self.inner
            .read()
            .ok()
            .and_then(|state| {
                state
                    .models
                    .get(model)
                    .map(|(_, capabilities)| capabilities.clone())
            })
            .unwrap_or_default()
    }

    async fn complete(
        &self,
        model: &str,
        thinking_level: ThinkingLevel,
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
        let (_, capabilities) = configured;
        let normalized = normalize_thinking_level(&capabilities, thinking_level);
        let _ = events.send(crate::agent::AgentEvent::ThinkingNormalized {
            requested: normalized.requested,
            clamped: normalized.clamped,
            provider_value: normalized.provider_value.clone(),
        });
        let request_messages =
            sanitize_messages(messages, capabilities.supports_interleaved_thinking);
        provider
            .complete_normalized(model_id, &normalized, &request_messages, tools, events)
            .await
    }
}

fn sanitize_messages(messages: &[Message], supports_interleaved_thinking: bool) -> Vec<Message> {
    messages
        .iter()
        .cloned()
        .map(|message| match message {
            Message::Assistant {
                content,
                thinking,
                tool_calls,
                provider_state,
            } => {
                let keep_reasoning = supports_interleaved_thinking && !tool_calls.is_empty();
                Message::Assistant {
                    content,
                    thinking: keep_reasoning.then_some(thinking).flatten(),
                    tool_calls,
                    provider_state: keep_reasoning.then_some(provider_state).flatten(),
                }
            }
            message => message,
        })
        .collect()
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
                    thinking: None,
                    compat: Some(super::ThinkingCompat {
                        supports_reasoning_effort: Some(true),
                        reasoning_effort_map: Default::default(),
                        supports_interleaved_thinking: Some(true),
                    }),
                    models: vec![ModelConfig {
                        id: "reasoning".to_owned(),
                        display_name: "Reasoning".to_owned(),
                        thinking: Some(super::ThinkingConfig {
                            min_level: ThinkingLevel::Low,
                            max_level: ThinkingLevel::Max,
                            mode: super::ThinkingMode::Effort,
                        }),
                        compat: None,
                    }],
                },
                ProviderConfig {
                    id: "two".to_owned(),
                    display_name: "Two".to_owned(),
                    base_url: "https://two.example/v1".to_owned(),
                    api_key: SecretValue::new("two-secret".to_owned()),
                    openai_api: OpenAiApi::ChatCompletions,
                    thinking: None,
                    compat: Some(super::ThinkingCompat {
                        supports_reasoning_effort: Some(false),
                        reasoning_effort_map: Default::default(),
                        supports_interleaved_thinking: Some(false),
                    }),
                    models: vec![ModelConfig {
                        id: "fast".to_owned(),
                        display_name: "Fast".to_owned(),
                        thinking: None,
                        compat: None,
                    }],
                },
            ],
        }
    }

    #[test]
    fn registry_resolves_model_capabilities_by_provider_and_model() {
        let registry = ProviderRegistry::new(&catalog(), Duration::from_secs(1)).unwrap();

        assert_eq!(
            registry.thinking_capabilities("one/reasoning").max_level,
            ThinkingLevel::Max
        );
        assert!(
            !registry
                .thinking_capabilities("two/fast")
                .supports_reasoning_effort
        );
        assert_eq!(
            registry.thinking_capabilities("one/fast").max_level,
            ThinkingLevel::High
        );
    }

    #[test]
    fn registry_replacement_is_visible_to_existing_clones() {
        let registry = ProviderRegistry::new(&catalog(), Duration::from_secs(1)).unwrap();
        let shared = registry.clone();
        let mut updated = catalog();
        updated.providers[1].compat = Some(super::ThinkingCompat {
            supports_reasoning_effort: Some(true),
            reasoning_effort_map: Default::default(),
            supports_interleaved_thinking: Some(false),
        });
        updated.providers[1].models[0].thinking = Some(super::ThinkingConfig {
            min_level: ThinkingLevel::Minimal,
            max_level: ThinkingLevel::XHigh,
            mode: super::ThinkingMode::Effort,
        });

        let update = registry.prepare_update(&updated).unwrap();
        assert!(
            !shared
                .thinking_capabilities("two/fast")
                .supports_reasoning_effort
        );

        registry.apply_update(update).unwrap();

        assert_eq!(
            shared.thinking_capabilities("two/fast").max_level,
            ThinkingLevel::XHigh
        );
    }
}
