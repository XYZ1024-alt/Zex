use tokio::sync::mpsc;
use tokio::sync::oneshot;

#[derive(Debug)]
pub enum AgentEvent {
    TextDelta(String),
    TurnFinished {
        acknowledged: oneshot::Sender<()>,
    },
    ToolStarted {
        call_id: String,
        name: String,
    },
    ToolFinished {
        call_id: String,
        name: String,
        output: String,
        is_error: bool,
    },
    Error(String),
}

pub type EventSender = mpsc::UnboundedSender<AgentEvent>;
