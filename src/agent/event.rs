use std::time::Duration;

use tokio::sync::mpsc;

use crate::agent::CompactStats;

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
    TurnCancelled,
    TurnEnd,
}

pub type EventSender = mpsc::UnboundedSender<AgentEvent>;
