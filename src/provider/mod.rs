mod models_dev;
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

pub use models_dev::{ModelsDevCatalog, ModelsDevLoad, ModelsDevProviderAlias};
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported: Option<Vec<ThinkingLevel>>,
    #[serde(default)]
    pub mode: ThinkingMode,
}

impl Default for ThinkingConfig {
    fn default() -> Self {
        Self {
            min_level: ThinkingLevel::Low,
            max_level: ThinkingLevel::High,
            supported: None,
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
    pub supported: Vec<ThinkingLevel>,
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
            supported: vec![
                ThinkingLevel::Off,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ],
            mode: ThinkingMode::Effort,
            supports_reasoning_effort: true,
            reasoning_effort_map: default_reasoning_effort_map(),
            supports_interleaved_thinking: false,
        }
    }
}

impl ThinkingCapabilities {
    pub fn available_levels(&self) -> Vec<ThinkingLevel> {
        if !self.supports_reasoning_effort {
            return vec![ThinkingLevel::Off];
        }
        std::iter::once(ThinkingLevel::Off)
            .chain(self.supported.iter().copied().filter(|level| {
                *level != ThinkingLevel::Off && self.reasoning_effort_map.contains_key(level)
            }))
            .collect()
    }

    pub fn summary(&self) -> String {
        self.available_levels()
            .into_iter()
            .map(|level| level.to_string())
            .collect::<Vec<_>>()
            .join("/")
    }

    pub(crate) fn from_discovered(
        supported: Vec<ThinkingLevel>,
        reasoning_effort_map: BTreeMap<ThinkingLevel, String>,
        supports_interleaved_thinking: bool,
    ) -> Self {
        let supported = normalize_supported_levels(supported);
        let mut capabilities = Self {
            min_level: first_enabled_level(&supported).unwrap_or(ThinkingLevel::Off),
            max_level: last_enabled_level(&supported).unwrap_or(ThinkingLevel::Off),
            supported,
            mode: ThinkingMode::Effort,
            supports_reasoning_effort: true,
            reasoning_effort_map,
            supports_interleaved_thinking,
        };
        if capabilities.min_level == ThinkingLevel::Off {
            capabilities.supports_reasoning_effort = false;
        }
        capabilities
    }

    pub(crate) fn disabled(supports_interleaved_thinking: bool) -> Self {
        Self {
            min_level: ThinkingLevel::Off,
            max_level: ThinkingLevel::Off,
            supported: vec![ThinkingLevel::Off],
            mode: ThinkingMode::Effort,
            supports_reasoning_effort: false,
            reasoning_effort_map: BTreeMap::new(),
            supports_interleaved_thinking,
        }
    }

    pub(crate) fn apply(
        &mut self,
        thinking: Option<&ThinkingConfig>,
        compat: Option<&ThinkingCompat>,
    ) {
        if let Some(thinking) = thinking {
            self.supports_reasoning_effort = true;
            if self.reasoning_effort_map.is_empty() {
                self.reasoning_effort_map = default_reasoning_effort_map();
            }
            self.min_level = thinking.min_level;
            self.max_level = thinking.max_level;
            self.supported = thinking.supported.clone().unwrap_or_else(|| {
                std::iter::once(ThinkingLevel::Off)
                    .chain(ThinkingLevel::ALL.into_iter().filter(|level| {
                        *level != ThinkingLevel::Off
                            && *level >= thinking.min_level
                            && *level <= thinking.max_level
                    }))
                    .collect()
            });
            self.supported = normalize_supported_levels(std::mem::take(&mut self.supported));
            self.mode = thinking.mode;
        }
        if let Some(compat) = compat {
            if let Some(supports_reasoning_effort) = compat.supports_reasoning_effort {
                if supports_reasoning_effort && !self.supports_reasoning_effort {
                    if compat.reasoning_effort_map.is_empty() {
                        let interleaved = self.supports_interleaved_thinking;
                        *self = Self::default();
                        self.supports_interleaved_thinking = interleaved;
                    } else {
                        self.supported = normalize_supported_levels(
                            compat.reasoning_effort_map.keys().copied().collect(),
                        );
                        self.min_level =
                            first_enabled_level(&self.supported).unwrap_or(ThinkingLevel::Off);
                        self.max_level =
                            last_enabled_level(&self.supported).unwrap_or(ThinkingLevel::Off);
                        self.reasoning_effort_map.clear();
                    }
                }
                self.supports_reasoning_effort = supports_reasoning_effort;
            }
            self.reasoning_effort_map
                .extend(compat.reasoning_effort_map.clone());
            if let Some(supports_interleaved_thinking) = compat.supports_interleaved_thinking {
                self.supports_interleaved_thinking = supports_interleaved_thinking;
            }
        }
        if !self.supports_reasoning_effort {
            self.min_level = ThinkingLevel::Off;
            self.max_level = ThinkingLevel::Off;
            self.supported = vec![ThinkingLevel::Off];
            self.reasoning_effort_map.clear();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedThinking {
    pub requested: ThinkingLevel,
    pub clamped: ThinkingLevel,
    pub effective: ThinkingLevel,
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
            effective: ThinkingLevel::Off,
            provider_value: None,
        };
    }

    let supported = capabilities
        .supported
        .iter()
        .copied()
        .filter(|level| *level != ThinkingLevel::Off)
        .collect::<Vec<_>>();
    let Some(first_supported) = supported.first().copied() else {
        return NormalizedThinking {
            requested,
            clamped: ThinkingLevel::Off,
            effective: ThinkingLevel::Off,
            provider_value: None,
        };
    };
    let bounded = requested.clamp(capabilities.min_level, capabilities.max_level);
    let clamped = supported
        .iter()
        .copied()
        .rev()
        .find(|level| *level <= bounded)
        .unwrap_or(first_supported);
    for effective in fallback_levels(clamped) {
        if !supported.contains(&effective) {
            continue;
        }
        if let Some(provider_value) = capabilities.reasoning_effort_map.get(&effective) {
            return NormalizedThinking {
                requested,
                clamped,
                effective,
                provider_value: Some(provider_value.clone()),
            };
        }
    }
    NormalizedThinking {
        requested,
        clamped,
        effective: ThinkingLevel::Off,
        provider_value: None,
    }
}

fn fallback_levels(level: ThinkingLevel) -> Vec<ThinkingLevel> {
    match level {
        ThinkingLevel::Max => vec![
            ThinkingLevel::Max,
            ThinkingLevel::XHigh,
            ThinkingLevel::High,
        ],
        ThinkingLevel::XHigh => vec![ThinkingLevel::XHigh, ThinkingLevel::High],
        ThinkingLevel::Off
        | ThinkingLevel::Minimal
        | ThinkingLevel::Low
        | ThinkingLevel::Medium
        | ThinkingLevel::High => vec![level],
    }
}

fn default_reasoning_effort_map() -> BTreeMap<ThinkingLevel, String> {
    [
        (ThinkingLevel::Low, "low".to_owned()),
        (ThinkingLevel::Medium, "medium".to_owned()),
        (ThinkingLevel::High, "high".to_owned()),
    ]
    .into_iter()
    .collect()
}

fn normalize_supported_levels(levels: Vec<ThinkingLevel>) -> Vec<ThinkingLevel> {
    ThinkingLevel::ALL
        .into_iter()
        .filter(|level| *level == ThinkingLevel::Off || levels.contains(level))
        .collect()
}

fn first_enabled_level(levels: &[ThinkingLevel]) -> Option<ThinkingLevel> {
    levels
        .iter()
        .copied()
        .find(|level| *level != ThinkingLevel::Off)
}

fn last_enabled_level(levels: &[ThinkingLevel]) -> Option<ThinkingLevel> {
    levels
        .iter()
        .copied()
        .rev()
        .find(|level| *level != ThinkingLevel::Off)
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
    fn non_contiguous_capabilities_clamp_without_increasing_depth() {
        let capabilities = ThinkingCapabilities::from_discovered(
            vec![ThinkingLevel::Off, ThinkingLevel::High, ThinkingLevel::Max],
            [
                (ThinkingLevel::High, "high".to_owned()),
                (ThinkingLevel::Max, "max".to_owned()),
            ]
            .into_iter()
            .collect(),
            false,
        );

        let below_min = normalize_thinking_level(&capabilities, ThinkingLevel::Medium);
        assert_eq!(below_min.clamped, ThinkingLevel::High);
        assert_eq!(below_min.effective, ThinkingLevel::High);

        let gap = normalize_thinking_level(&capabilities, ThinkingLevel::XHigh);
        assert_eq!(gap.clamped, ThinkingLevel::High);
        assert_eq!(gap.effective, ThinkingLevel::High);
        assert_eq!(
            capabilities.available_levels(),
            vec![ThinkingLevel::Off, ThinkingLevel::High, ThinkingLevel::Max]
        );
    }

    #[test]
    fn custom_map_supports_max_and_falls_back_without_passthrough() {
        let mut capabilities = ThinkingCapabilities::default();
        capabilities.apply(
            Some(&ThinkingConfig {
                min_level: ThinkingLevel::Minimal,
                max_level: ThinkingLevel::Max,
                supported: None,
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
    fn manual_thinking_reenables_effort_after_discovered_disable() {
        let mut capabilities = ThinkingCapabilities::disabled(true);
        capabilities.apply(
            Some(&ThinkingConfig {
                min_level: ThinkingLevel::High,
                max_level: ThinkingLevel::Max,
                supported: Some(vec![
                    ThinkingLevel::Off,
                    ThinkingLevel::High,
                    ThinkingLevel::Max,
                ]),
                mode: ThinkingMode::Effort,
            }),
            Some(&ThinkingCompat {
                supports_reasoning_effort: Some(true),
                reasoning_effort_map: [
                    (ThinkingLevel::High, "high".to_owned()),
                    (ThinkingLevel::Max, "max".to_owned()),
                ]
                .into_iter()
                .collect(),
                supports_interleaved_thinking: None,
            }),
        );

        assert_eq!(
            capabilities.available_levels(),
            vec![ThinkingLevel::Off, ThinkingLevel::High, ThinkingLevel::Max]
        );
        assert_eq!(
            normalize_thinking_level(&capabilities, ThinkingLevel::Max).effective,
            ThinkingLevel::Max
        );
    }

    #[test]
    fn provider_defaults_merge_with_model_overrides() {
        let mut capabilities = ThinkingCapabilities::default();
        capabilities.apply(
            Some(&ThinkingConfig {
                min_level: ThinkingLevel::Low,
                max_level: ThinkingLevel::XHigh,
                supported: None,
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
                supported: None,
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
            effective: normalized.effective,
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
            models_dev: Default::default(),
            models_dev_aliases: Vec::new(),
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
                            supported: None,
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
            supported: None,
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
