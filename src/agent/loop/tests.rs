use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Result, bail};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::{
    agent::{AgentEvent, AssistantMessage, MessageRole, ToolCall},
    provider::{Provider, ToolDefinition},
    tools::{Tool, ToolFuture, ToolRegistry},
};

use super::{AUTO_COMPACT_PERCENT, Agent, AgentOptions};

struct SequenceProvider {
    messages: Mutex<VecDeque<AssistantMessage>>,
    requests: Arc<Mutex<Vec<Vec<crate::agent::Message>>>>,
}

impl Provider for SequenceProvider {
    async fn complete(
        &self,
        _model: &str,
        _thinking_level: crate::provider::ThinkingLevel,
        messages: &[crate::agent::Message],
        _tools: &[ToolDefinition],
        events: &crate::agent::EventSender,
    ) -> Result<AssistantMessage> {
        self.requests
            .lock()
            .expect("sequence provider requests mutex poisoned")
            .push(messages.to_vec());
        let message = self
            .messages
            .lock()
            .expect("sequence provider mutex poisoned")
            .pop_front()
            .expect("sequence provider exhausted");
        if let Some(thinking) = &message.thinking {
            let _ = events.send(AgentEvent::ThinkingDelta {
                delta: thinking.clone(),
            });
        }
        if !message.content.is_empty() {
            let _ = events.send(AgentEvent::MessageDelta {
                role: MessageRole::Assistant,
                delta: message.content.clone(),
            });
        }
        Ok(message)
    }
}

struct FailingProvider;

impl Provider for FailingProvider {
    async fn complete(
        &self,
        _model: &str,
        _thinking_level: crate::provider::ThinkingLevel,
        _messages: &[crate::agent::Message],
        _tools: &[ToolDefinition],
        _events: &crate::agent::EventSender,
    ) -> Result<AssistantMessage> {
        bail!("provider unavailable")
    }
}

struct PendingProvider;

impl Provider for PendingProvider {
    async fn complete(
        &self,
        _model: &str,
        _thinking_level: crate::provider::ThinkingLevel,
        _messages: &[crate::agent::Message],
        _tools: &[ToolDefinition],
        _events: &crate::agent::EventSender,
    ) -> Result<AssistantMessage> {
        std::future::pending().await
    }
}

#[test]
fn system_only_agent_has_no_conversation() {
    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::new()),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let (events, _) = mpsc::unbounded_channel();
    let agent = Agent::new(
        provider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 1,
            max_context_chars: 120_000,
            compact_keep_turns: 6,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );

    assert!(!agent.has_conversation());
}

struct EchoTool;

impl Tool for EchoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "echo".to_owned(),
            description: "Returns the supplied value.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }),
        }
    }

    fn execute(&self, arguments: Value, _timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            Ok(arguments
                .get("value")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned())
        })
    }
}

#[tokio::test]
async fn emits_message_tool_and_turn_events_in_order() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::from([
            AssistantMessage {
                content: String::new(),
                thinking: Some("Need the echo result.".to_owned()),
                tool_calls: vec![ToolCall {
                    id: "call-1".to_owned(),
                    name: "echo".to_owned(),
                    arguments: r#"{"value":"observed"}"#.to_owned(),
                }],
                provider_state: None,
            },
            AssistantMessage {
                content: "done".to_owned(),
                thinking: Some("The tool returned the requested value.".to_owned()),
                tool_calls: Vec::new(),
                provider_state: None,
            },
        ])),
        requests: Arc::clone(&requests),
    };
    let mut tools = ToolRegistry::new(Duration::from_secs(1), 32_000);
    tools.register(EchoTool);
    let (events, mut receiver) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        provider,
        tools,
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 3,
            max_context_chars: 120_000,
            compact_keep_turns: 6,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );

    agent.prompt("inspect").await.unwrap();

    let captured_requests = requests
        .lock()
        .expect("sequence provider requests mutex poisoned");
    assert_eq!(captured_requests.len(), 2);
    assert!(matches!(
        captured_requests[1].as_slice(),
        [
            crate::agent::Message::System { .. },
            crate::agent::Message::User { content },
            crate::agent::Message::Assistant {
                thinking: Some(thinking),
                tool_calls,
                ..
            },
            crate::agent::Message::Tool {
                tool_call_id,
                content: output,
            },
        ] if content == "inspect"
            && thinking == "Need the echo result."
            && tool_calls[0].id == "call-1"
            && tool_call_id == "call-1"
            && output == "observed"
    ));
    drop(captured_requests);

    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "inspect".to_owned(),
        }
    );
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::ThinkingDelta {
            delta: "Need the echo result.".to_owned(),
        }
    );
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::ToolStart {
            call_id: "call-1".to_owned(),
            name: "echo".to_owned(),
            arguments: r#"{"value":"observed"}"#.to_owned(),
            timeout: Duration::from_secs(1),
        }
    );
    assert!(matches!(
        receiver.try_recv().unwrap(),
        AgentEvent::ToolEnd {
            call_id,
            name,
            output,
            is_error: false,
            elapsed,
        } if call_id == "call-1"
            && name == "echo"
            && output == "observed"
            && elapsed > Duration::ZERO
    ));
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::ThinkingDelta {
            delta: "The tool returned the requested value.".to_owned(),
        }
    );
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::MessageDelta {
            role: MessageRole::Assistant,
            delta: "done".to_owned(),
        }
    );
    assert_eq!(receiver.try_recv().unwrap(), AgentEvent::TurnEnd);
}

#[tokio::test]
async fn emits_provider_errors() {
    let (events, mut receiver) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        FailingProvider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 1,
            max_context_chars: 120_000,
            compact_keep_turns: 6,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );

    let error = agent.prompt("fail").await.unwrap_err();

    assert_eq!(error.to_string(), "provider unavailable");
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "fail".to_owned(),
        }
    );
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::Error {
            message: "provider unavailable".to_owned(),
        }
    );
}

#[tokio::test]
async fn cancellation_keeps_user_prompt_and_discards_partial_turn_state() {
    let (events, mut receiver) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        PendingProvider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(60),
            max_turns: 1,
            max_context_chars: 120_000,
            compact_keep_turns: 6,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );
    let (cancel_sender, cancel_receiver) = tokio::sync::watch::channel(false);

    let outcome = tokio::time::timeout(Duration::from_secs(1), async {
        let prompt = agent.prompt_cancellable("stop this", cancel_receiver);
        tokio::pin!(prompt);
        tokio::task::yield_now().await;
        cancel_sender.send(true).unwrap();
        prompt.await
    })
    .await
    .expect("cancellation timed out")
    .unwrap();

    assert_eq!(outcome, crate::agent::PromptOutcome::Cancelled);
    assert_eq!(agent.messages().len(), 2);
    assert!(matches!(
        &agent.messages()[1],
        crate::agent::Message::User { content } if content == "stop this"
    ));
    assert_eq!(
        receiver.try_recv().unwrap(),
        AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: "stop this".to_owned(),
        }
    );
    assert_eq!(receiver.try_recv().unwrap(), AgentEvent::TurnCancelled);
}

#[test]
fn compact_summarizes_old_tool_output_and_keeps_recent_turns() {
    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::new()),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let (events, _) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        provider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 1,
            max_context_chars: 120_000,
            compact_keep_turns: 2,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        Some(vec![
            crate::agent::Message::User {
                content: "old task".to_owned(),
            },
            crate::agent::Message::Assistant {
                content: String::new(),
                thinking: Some("Need to inspect the large file.".to_owned()),
                tool_calls: vec![ToolCall {
                    id: "old-call".to_owned(),
                    name: "read".to_owned(),
                    arguments: r#"{"path":"large.txt"}"#.to_owned(),
                }],
                provider_state: None,
            },
            crate::agent::Message::Tool {
                tool_call_id: "old-call".to_owned(),
                content: "x".repeat(4_000),
            },
            crate::agent::Message::User {
                content: "recent one".to_owned(),
            },
            crate::agent::Message::Assistant {
                content: "answer one".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
            },
            crate::agent::Message::User {
                content: "recent two".to_owned(),
            },
            crate::agent::Message::Assistant {
                content: "answer two".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
            },
        ]),
    );

    let before = agent.context_chars();
    let stats = agent.compact();

    assert_eq!(stats.kept_turns, 2);
    assert_eq!(stats.summarized_turns, 1);
    assert_eq!(stats.summarized_tool_outputs, 1);
    assert!(stats.freed_chars > 3_000);
    assert!(agent.context_chars() < before);
    println!(
        "compact verification: before={} after={} freed={} kept={} summarized_turns={} summarized_tools={}",
        stats.before_chars,
        stats.after_chars,
        stats.freed_chars,
        stats.kept_turns,
        stats.summarized_turns,
        stats.summarized_tool_outputs
    );
    assert!(matches!(
        &agent.messages()[1],
        crate::agent::Message::System { content }
            if content.contains("Compacted earlier conversation")
                && content.contains("Assistant thinking")
                && content.contains("Tool result read")
    ));
    assert!(agent.messages().iter().any(
            |message| matches!(message, crate::agent::Message::User { content } if content == "recent one")
        ));
    assert!(agent.messages().iter().any(
            |message| matches!(message, crate::agent::Message::User { content } if content == "recent two")
        ));
}

#[test]
fn automatic_compact_emits_feedback_when_context_crosses_threshold() {
    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::new()),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let (events, mut receiver) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        provider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 1,
            max_context_chars: 1_000,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        Some(vec![
            crate::agent::Message::User {
                content: "old".repeat(AUTO_COMPACT_PERCENT * 5),
            },
            crate::agent::Message::Assistant {
                content: "old answer".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
            },
            crate::agent::Message::User {
                content: "recent".to_owned(),
            },
        ]),
    );

    let stats = agent.compact_if_needed().expect("context should compact");

    assert!(stats.freed_chars > 0);
    assert!(matches!(
        receiver.try_recv().unwrap(),
        AgentEvent::ContextCompacted { stats: emitted } if emitted == stats
    ));
}
