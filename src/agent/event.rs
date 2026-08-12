use tokio::sync::mpsc;

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
    },
    ToolEnd {
        call_id: String,
        name: String,
        output: String,
        is_error: bool,
    },
    Error {
        message: String,
    },
    TurnEnd,
}

pub type EventSender = mpsc::UnboundedSender<AgentEvent>;
