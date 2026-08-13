use std::time::Duration;

use tokio::sync::mpsc;

use crate::agent::CompactStats;
use crate::provider::ThinkingLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    MessageDelta {
        role: MessageRole,
        delta: String,
    },
    ThinkingDelta {
        delta: String,
    },
    ThinkingNormalized {
        requested: ThinkingLevel,
        clamped: ThinkingLevel,
        effective: ThinkingLevel,
        provider_value: Option<String>,
    },
    ToolStart {
        call_id: String,
        name: String,
        arguments: String,
        timeout: Duration,
    },
    ToolEnd {
        call_id: String,
        name: String,
        output: String,
        is_error: bool,
        elapsed: Duration,
    },
    Error {
        message: String,
    },
    ContextCompacted {
        stats: CompactStats,
    },
    ProviderUsage {
        output_tokens: u64,
        elapsed: Duration,
    },
    TurnCancelled,
    TurnEnd,
}

pub type EventSender = mpsc::UnboundedSender<AgentEvent>;
