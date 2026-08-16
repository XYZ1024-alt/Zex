use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::OnceLock;

/// Shared o200k_base tokenizer used for context budgeting. It approximates
/// the vocab of the models Zex talks to closely enough for threshold
/// decisions; if construction ever fails we fall back to a character
/// heuristic instead of breaking the agent.
fn tokenizer() -> Option<&'static tiktoken_rs::CoreBPE> {
    static TOKENIZER: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();
    TOKENIZER
        .get_or_init(|| tiktoken_rs::o200k_base().ok())
        .as_ref()
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantMessage {
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub provider_state: Option<Value>,
    pub usage: Option<CompletionUsage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptOutcome {
    Completed(AssistantMessage),
    Cancelled,
}

impl Message {
    /// Lightweight per-message estimate used for memory metadata and as a
    /// last-resort fallback when a provider cannot prepare its exact request.
    /// Runtime budgeting uses `PreparedRequest`, which measures the sanitized
    /// protocol payload including tools and retained reasoning state.
    pub fn token_estimate(&self) -> usize {
        match self {
            Self::System { content } | Self::User { content } | Self::Tool { content, .. } => {
                estimate_tokens(content)
            }
            Self::Assistant {
                content,
                tool_calls,
                ..
            } => {
                estimate_tokens(content)
                    + tool_calls
                        .iter()
                        .map(|call| {
                            estimate_tokens(&call.id)
                                + estimate_tokens(&call.name)
                                + estimate_tokens(&call.arguments)
                        })
                        .sum::<usize>()
            }
        }
    }
}

pub(crate) fn estimate_tokens(content: &str) -> usize {
    if let Some(bpe) = tokenizer() {
        // `encode_ordinary` treats special-token-looking text as plain text,
        // so untrusted file contents can never panic the counter.
        return bpe.encode_ordinary(content).len();
    }
    heuristic_estimate(content)
}

pub(crate) fn truncate_to_token_budget(content: &str, max_tokens: usize) -> (String, bool, usize) {
    let original_tokens = estimate_tokens(content);
    if original_tokens <= max_tokens {
        return (content.to_owned(), false, original_tokens);
    }
    if max_tokens == 0 {
        return (String::new(), true, 0);
    }
    if let Some(bpe) = tokenizer() {
        let tokens = bpe.encode_ordinary(content);
        let mut end = max_tokens.min(tokens.len());
        while end > 0 {
            if let Ok(excerpt) = bpe.decode(&tokens[..end]) {
                let returned_tokens = estimate_tokens(&excerpt);
                return (excerpt, true, returned_tokens);
            }
            end -= 1;
        }
        return (String::new(), true, 0);
    }

    let mut excerpt = String::new();
    for character in content.chars() {
        excerpt.push(character);
        if heuristic_estimate(&excerpt) > max_tokens {
            excerpt.pop();
            break;
        }
    }
    let returned_tokens = heuristic_estimate(&excerpt);
    (excerpt, true, returned_tokens)
}

/// Last-resort estimate when the tokenizer is unavailable: ASCII text
/// averages ~4 characters per token while CJK and other non-ASCII scripts
/// are closer to 1 token per character.
fn heuristic_estimate(content: &str) -> usize {
    let mut ascii = 0usize;
    let mut non_ascii = 0usize;
    for character in content.chars() {
        if character.is_ascii() {
            ascii += 1;
        } else {
            non_ascii += 1;
        }
    }
    ascii / 4 + non_ascii
}
