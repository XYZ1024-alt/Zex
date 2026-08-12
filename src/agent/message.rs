use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thinking: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_state: Option<Value>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantMessage {
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub provider_state: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptOutcome {
    Completed(AssistantMessage),
    Cancelled,
}

impl Message {
    pub fn character_count(&self) -> usize {
        match self {
            Self::System { content } | Self::User { content } | Self::Tool { content, .. } => {
                content.chars().count()
            }
            Self::Assistant {
                content,
                thinking,
                tool_calls,
                provider_state,
            } => {
                content.chars().count()
                    + thinking
                        .as_ref()
                        .map_or(0, |thinking| thinking.chars().count())
                    + tool_calls
                        .iter()
                        .map(|call| {
                            call.id.chars().count()
                                + call.name.chars().count()
                                + call.arguments.chars().count()
                        })
                        .sum::<usize>()
                    + provider_state
                        .as_ref()
                        .map_or(0, |state| state.to_string().chars().count())
            }
        }
    }
}
