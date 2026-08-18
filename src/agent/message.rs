use serde::{Deserialize, Serialize};
use serde_json::Value;

const ASCII_CHARS_PER_TOKEN: usize = 4;

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
    let mut tokens = 0usize;
    let mut ascii_run = 0usize;
    for character in content.chars() {
        if character.is_ascii() {
            ascii_run += 1;
            if ascii_run == ASCII_CHARS_PER_TOKEN {
                tokens = tokens.saturating_add(1);
                ascii_run = 0;
            }
            continue;
        }
        if ascii_run > 0 {
            tokens = tokens.saturating_add(1);
            ascii_run = 0;
        }
        tokens = tokens.saturating_add(1);
    }
    tokens.saturating_add(usize::from(ascii_run > 0))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenWindow {
    pub content: String,
    pub start: usize,
    pub end: usize,
    pub total: usize,
}

pub(crate) fn token_window(content: &str, offset: usize, max_tokens: usize) -> TokenWindow {
    let total = estimate_tokens(content);
    let start = offset.min(total);
    let end = offset.saturating_add(max_tokens).min(total);
    let start_byte = estimated_token_byte_offset(content, start);
    let end_byte = estimated_token_byte_offset(content, end);
    TokenWindow {
        content: content[start_byte..end_byte].to_owned(),
        start,
        end,
        total,
    }
}

/// Maps the lightweight estimate back to a UTF-8 byte boundary without a
/// tokenizer. Consecutive ASCII is grouped four characters at a time; a
/// partial run and every non-ASCII character each form one estimated token.
fn estimated_token_byte_offset(content: &str, token_offset: usize) -> usize {
    if token_offset == 0 {
        return 0;
    }
    let mut tokens = 0usize;
    let mut ascii_run = 0usize;
    for (byte, character) in content.char_indices() {
        if character.is_ascii() {
            ascii_run += 1;
            if ascii_run == ASCII_CHARS_PER_TOKEN {
                tokens += 1;
                ascii_run = 0;
                if tokens == token_offset {
                    return byte + character.len_utf8();
                }
            }
            continue;
        }
        if ascii_run > 0 {
            tokens += 1;
            ascii_run = 0;
            if tokens == token_offset {
                return byte;
            }
        }
        tokens += 1;
        if tokens == token_offset {
            return byte + character.len_utf8();
        }
    }
    content.len()
}

#[cfg(test)]
mod tests {
    use super::{estimate_tokens, token_window};

    #[test]
    fn estimates_ascii_runs_and_non_ascii_without_a_model_tokenizer() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("你好"), 2);
        assert_eq!(estimate_tokens("ab你cd"), 3);
    }

    #[test]
    fn token_windows_page_without_splitting_utf8_or_losing_content() {
        let content = "abcdefgh你好ijkl";
        let first = token_window(content, 0, 2);
        let middle = token_window(content, first.end, 2);
        let last = token_window(content, middle.end, 1);

        assert_eq!(first.content, "abcdefgh");
        assert_eq!(middle.content, "你好");
        assert_eq!(last.content, "ijkl");
        assert_eq!(
            format!("{}{}{}", first.content, middle.content, last.content),
            content
        );
        assert_eq!(last.end, last.total);
    }
}
