mod openai;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt, str::FromStr};

use crate::agent::{AssistantMessage, EventSender, Message};

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAiApi {
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
