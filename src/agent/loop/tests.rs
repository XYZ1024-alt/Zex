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
    memory::{MemoryConfig, MemoryRuntime, extract_memory_ids},
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
            max_context_tokens: 120_000,
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
            Ok(crate::tools::ToolOutcome::output_only(
                arguments
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            ))
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
                usage: None,
            },
            AssistantMessage {
                content: "done".to_owned(),
                thinking: Some("The tool returned the requested value.".to_owned()),
                tool_calls: Vec::new(),
                provider_state: None,
                usage: None,
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
            max_context_tokens: 120_000,
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
            change: None,
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
async fn large_tool_result_compacts_to_a_recallable_citation() {
    let directory = std::env::temp_dir().join(format!(
        "zex-agent-memory-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let memory = Arc::new(MemoryRuntime::new(MemoryConfig {
        max_recall_tokens: 8_192,
        ..MemoryConfig::default()
    }));
    memory
        .activate("20260815-120000-cafebabe", directory.clone())
        .await
        .unwrap();
    let original = "precise-large-observation\n".repeat(600);
    let old_prompt = "old task context ".repeat(2_000);
    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::from([
            AssistantMessage {
                content: String::new(),
                thinking: Some("Need the large observation.".to_owned()),
                tool_calls: vec![ToolCall {
                    id: "call-big".to_owned(),
                    name: "echo".to_owned(),
                    arguments: serde_json::to_string(&json!({"value": original.clone()})).unwrap(),
                }],
                provider_state: None,
                usage: None,
            },
            AssistantMessage {
                content: "old turn complete".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
                usage: None,
            },
            AssistantMessage {
                content: "recent turn complete".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
                usage: None,
            },
        ])),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let mut tools = ToolRegistry::new(Duration::from_secs(1), 32_000);
    tools.set_memory(Arc::clone(&memory));
    tools.register(EchoTool);
    let (events, _) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        provider,
        tools,
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(5),
            max_turns: 3,
            max_context_tokens: 120_000,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );
    agent.initialize_memory().await.unwrap();

    agent.prompt(old_prompt.clone()).await.unwrap();
    agent.prompt("recent task").await.unwrap();

    let tool_content = agent
        .messages()
        .iter()
        .find_map(|message| match message {
            crate::agent::Message::Tool { content, .. } => Some(content),
            _ => None,
        })
        .unwrap();
    let id = extract_memory_ids(tool_content).pop().unwrap();
    assert!(tool_content.contains("recall available"));
    assert!(!tool_content.contains("precise-large-observation"));
    assert!(matches!(
        agent.messages().first(),
        Some(crate::agent::Message::System { content })
            if content.contains("[Current valid addressable pointers]")
                && content.contains(&id)
                && content.contains("Never invent")
                && content.contains("A recall failure means the content is unavailable")
    ));
    let stored_history = memory.list_pointers(Some("old task")).unwrap();
    let turn_id = extract_memory_ids(&stored_history).pop().unwrap();
    assert!(turn_id.starts_with("§turn_"));

    let before = agent.context_tokens();
    let stats = agent.compact().await.unwrap();
    assert!(stats.freed_tokens > 0);
    assert!(agent.context_tokens() * 2 < before);
    assert!(matches!(
        agent.messages().get(1),
        Some(crate::agent::Message::System { content })
            if content.contains("[Available addressable pointers]")
                && content.contains(&id)
    ));
    let recalled_turn = memory
        .recall(&turn_id, Some("recover compacted request".to_owned()))
        .await
        .unwrap();
    assert!(recalled_turn.ends_with(&old_prompt));
    let recalled = memory
        .recall(&id, Some("continue from exact evidence".to_owned()))
        .await
        .unwrap();
    assert!(recalled.ends_with(&original));
    tokio::fs::remove_dir_all(directory).await.unwrap();
}

#[tokio::test]
async fn disabled_memory_preserves_the_original_tool_and_prompt_contract() {
    let original = "legacy-full-output".repeat(200);
    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::from([
            AssistantMessage {
                content: String::new(),
                thinking: None,
                tool_calls: vec![ToolCall {
                    id: "legacy-call".to_owned(),
                    name: "echo".to_owned(),
                    arguments: serde_json::to_string(&json!({"value": original.clone()})).unwrap(),
                }],
                provider_state: None,
                usage: None,
            },
            AssistantMessage {
                content: "done".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
                usage: None,
            },
        ])),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let mut tools = ToolRegistry::new(Duration::from_secs(1), 32_000);
    tools.register(EchoTool);
    let (events, _) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        provider,
        tools,
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(5),
            max_turns: 2,
            max_context_tokens: 120_000,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );

    agent.prompt("legacy behavior").await.unwrap();

    assert!(matches!(
        agent.messages().first(),
        Some(crate::agent::Message::System { content })
            if !content.contains("[Addressable memory policy]")
    ));
    assert!(agent.messages().iter().any(|message| matches!(
        message,
        crate::agent::Message::Tool { content, .. } if content == &original
    )));
    assert!(
        !agent
            .messages()
            .iter()
            .any(|message| !extract_memory_ids(super::message_content(message)).is_empty())
    );
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
            max_context_tokens: 120_000,
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
            max_context_tokens: 120_000,
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

#[tokio::test]
async fn compact_summarizes_old_tool_output_and_keeps_recent_turns() {
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
            max_context_tokens: 120_000,
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
                content: "x".repeat(16_000),
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

    let before = agent.context_tokens();
    let stats = agent.compact().await.unwrap();

    assert_eq!(stats.kept_turns, 2);
    assert_eq!(stats.summarized_turns, 1);
    assert_eq!(stats.summarized_tool_outputs, 1);
    assert!(stats.freed_tokens > 0);
    assert!(agent.context_tokens() * 2 < before);
    println!(
        "compact verification: before={} after={} freed={} kept={} summarized_turns={} summarized_tools={}",
        stats.before_tokens,
        stats.after_tokens,
        stats.freed_tokens,
        stats.kept_turns,
        stats.summarized_turns,
        stats.summarized_tool_outputs
    );
    assert!(matches!(
        &agent.messages()[1],
        crate::agent::Message::System { content }
            if content.contains("Compacted earlier conversation")
                && content.contains("Original request:\nold task")
                && !content.contains("User: old task")
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

#[tokio::test]
async fn automatic_compact_emits_feedback_when_context_crosses_threshold() {
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
            max_context_tokens: 1_000,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        Some(vec![
            crate::agent::Message::User {
                content: "old".repeat(AUTO_COMPACT_PERCENT * 50),
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

    let stats = agent
        .compact_if_needed()
        .await
        .unwrap()
        .expect("context should compact");

    assert!(stats.freed_tokens > 0);
    assert!(matches!(
        receiver.try_recv().unwrap(),
        AgentEvent::ContextCompacted { stats: emitted } if emitted == stats
    ));
}

#[tokio::test]
async fn tool_end_event_carries_the_file_change_for_mutations() {
    let working_dir = std::env::temp_dir().join(format!(
        "zex-change-event-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    tokio::fs::create_dir_all(&working_dir).await.unwrap();

    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::from([
            AssistantMessage {
                content: String::new(),
                thinking: None,
                tool_calls: vec![
                    ToolCall {
                        id: "call-write".to_owned(),
                        name: "write".to_owned(),
                        arguments: r#"{"path":"note.txt","content":"alpha\n"}"#.to_owned(),
                    },
                    ToolCall {
                        id: "call-edit".to_owned(),
                        name: "edit".to_owned(),
                        arguments: r#"{"path":"missing.txt","old_text":"a","new_text":"b"}"#
                            .to_owned(),
                    },
                ],
                provider_state: None,
                usage: None,
            },
            AssistantMessage {
                content: "done".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
                usage: None,
            },
        ])),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let (events, mut receiver) = mpsc::unbounded_channel();
    let mut tools = ToolRegistry::new(Duration::from_secs(5), 32_000);
    tools.register(crate::tools::WriteTool::new(working_dir.clone()));
    tools.register(crate::tools::EditTool::new(working_dir.clone()));
    let mut agent = Agent::new(
        provider,
        tools,
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(5),
            max_turns: 2,
            max_context_tokens: 120_000,
            compact_keep_turns: 6,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );

    agent.prompt("write then fail an edit").await.unwrap();

    let mut changes = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        if let AgentEvent::ToolEnd {
            call_id, change, ..
        } = event
        {
            changes.push((call_id, change));
        }
    }

    assert_eq!(changes.len(), 2);
    let (call_id, change) = &changes[0];
    assert_eq!(call_id, "call-write");
    let change = change.as_ref().expect("successful write carries a change");
    assert_eq!(change.before, None);
    assert_eq!(change.after, "alpha\n");
    assert!(change.path.ends_with("note.txt"));
    let (call_id, change) = &changes[1];
    assert_eq!(call_id, "call-edit");
    assert!(change.is_none(), "failed tools carry no change");

    tokio::fs::remove_dir_all(working_dir).await.unwrap();
}

struct LimitedProvider;

impl Provider for LimitedProvider {
    fn context_limit(&self, _model: &str) -> Option<crate::provider::ModelLimit> {
        Some(crate::provider::ModelLimit {
            context: 100_000,
            output: Some(4_000),
        })
    }

    async fn complete(
        &self,
        _model: &str,
        _thinking_level: crate::provider::ThinkingLevel,
        _messages: &[crate::agent::Message],
        _tools: &[ToolDefinition],
        _events: &crate::agent::EventSender,
    ) -> Result<AssistantMessage> {
        bail!("not used")
    }
}

#[test]
fn token_estimate_uses_bpe_and_skips_thinking() {
    // o200k_base merges common ASCII runs into single tokens.
    let ascii = crate::agent::Message::User {
        content: "abcdefgh".repeat(8),
    };
    assert_eq!(ascii.token_estimate(), 8);

    // CJK text tokenizes at roughly one token per two characters, far below
    // the one-token-per-character worst case of the character heuristic.
    let cjk = crate::agent::Message::User {
        content: "你好".repeat(32),
    };
    assert_eq!(cjk.token_estimate(), 32);

    let thinking = crate::agent::Message::Assistant {
        content: "answer".to_owned(),
        thinking: Some("long internal monologue ".repeat(1_000)),
        tool_calls: Vec::new(),
        provider_state: Some(json!({"reasoning_content": "stuffed"})),
    };
    let plain = crate::agent::Message::Assistant {
        content: "answer".to_owned(),
        thinking: None,
        tool_calls: Vec::new(),
        provider_state: None,
    };
    assert_eq!(thinking.token_estimate(), plain.token_estimate());
}

#[test]
fn context_budget_uses_model_limit_minus_output_reserve() {
    let (events, _) = mpsc::unbounded_channel();
    let agent = Agent::new(
        LimitedProvider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "limited".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 1,
            max_context_tokens: 120_000,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );

    assert_eq!(agent.context_budget(), 96_000);
}

#[test]
fn context_budget_falls_back_when_no_model_limit_is_known() {
    let (events, _) = mpsc::unbounded_channel();
    let agent = Agent::new(
        FailingProvider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "unknown".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 1,
            max_context_tokens: 42_000,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        None,
    );

    assert_eq!(agent.context_budget(), 42_000);
}

#[tokio::test]
async fn prune_clears_old_tool_outputs_before_full_compaction() {
    let (events, _) = mpsc::unbounded_channel();
    let mut messages = vec![crate::agent::Message::User {
        content: "task".to_owned(),
    }];
    for index in 0..6 {
        messages.push(crate::agent::Message::Tool {
            tool_call_id: format!("call-{index}"),
            content: "x".repeat(4_000),
        });
    }
    // Derive the budget from the measured size so the test does not depend on
    // tokenizer internals: trigger above the threshold, but close enough that
    // pruning two old outputs alone brings us back under it.
    let total: usize = std::iter::once(crate::agent::Message::System {
        content: super::SYSTEM_PROMPT.to_owned(),
    })
    .chain(messages.iter().cloned())
    .map(|message| message.token_estimate())
    .sum();
    let per_tool = crate::agent::Message::Tool {
        tool_call_id: "probe".to_owned(),
        content: "x".repeat(4_000),
    }
    .token_estimate();
    let budget = (total - per_tool / 2) * 100 / AUTO_COMPACT_PERCENT;
    let mut agent = Agent::new(
        FailingProvider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(1),
            max_turns: 1,
            max_context_tokens: budget,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        Some(messages),
    );

    let stats = agent
        .compact_if_needed()
        .await
        .unwrap()
        .expect("context should cross the threshold");

    assert_eq!(stats.pruned_tool_outputs, 2);
    assert_eq!(stats.summarized_turns, 0, "pruning alone was enough");
    let tools = agent
        .messages()
        .iter()
        .filter_map(|message| match message {
            crate::agent::Message::Tool { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tools.len(), 6);
    assert!(tools[..2].iter().all(|content| content
        .starts_with("[tool output cleared to free context: 4000 chars]")));
    assert!(tools[2..].iter().all(|content| content.len() == 4_000));
    assert!(!agent.messages().iter().any(|message| matches!(
        message,
        crate::agent::Message::System { content }
            if content.starts_with("[Compacted earlier conversation:")
    )));
}

#[tokio::test]
async fn server_usage_calibrates_context_and_compaction_resets_it() {
    let provider = SequenceProvider {
        messages: Mutex::new(VecDeque::from([AssistantMessage {
            content: "ok".to_owned(),
            thinking: None,
            tool_calls: Vec::new(),
            provider_state: None,
            usage: Some(crate::agent::CompletionUsage {
                input_tokens: Some(5_000),
                output_tokens: Some(2),
            }),
        }])),
        requests: Arc::new(Mutex::new(Vec::new())),
    };
    let (events, _) = mpsc::unbounded_channel();
    let mut agent = Agent::new(
        provider,
        ToolRegistry::new(Duration::from_secs(1), 32_000),
        events,
        AgentOptions {
            model: "test-model".to_owned(),
            turn_timeout: Duration::from_secs(5),
            max_turns: 1,
            max_context_tokens: 120_000,
            compact_keep_turns: 1,
            thinking_level: crate::provider::ThinkingLevel::Medium,
        },
        Some(vec![
            crate::agent::Message::User {
                content: "old task".to_owned(),
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

    let local_sum = |agent: &Agent<SequenceProvider>| {
        agent
            .messages()
            .iter()
            .map(crate::agent::Message::token_estimate)
            .sum::<usize>()
    };
    assert_eq!(agent.context_tokens(), local_sum(&agent));

    agent.prompt("hello").await.unwrap();

    // After a completion, context = server-reported input tokens + local
    // estimate of the messages appended since (the assistant reply).
    let assistant_tail = crate::agent::Message::Assistant {
        content: "ok".to_owned(),
        thinking: None,
        tool_calls: Vec::new(),
        provider_state: None,
    }
    .token_estimate();
    assert_eq!(agent.context_tokens(), 5_000 + assistant_tail);

    // Compaction rewrites the history, so the server baseline is dropped and
    // the estimate becomes fully local again.
    agent.compact().await.unwrap();
    assert_eq!(agent.context_tokens(), local_sum(&agent));
}
