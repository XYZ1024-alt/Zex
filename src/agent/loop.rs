use std::{
    future::Future,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use serde_json::Value;

use crate::{
    agent::{AgentEvent, AssistantMessage, EventSender, Message, MessageRole, PromptOutcome},
    provider::{Provider, ThinkingLevel},
    tools::ToolRegistry,
};

const SYSTEM_PROMPT: &str = "You are Zex, a minimal AI agent core. Be concise and accurate. Use grep to search file contents, glob to find files, and bash only for other system commands. Use read, write, and edit for file operations. Use tool results to finish the task.";
const AUTO_COMPACT_PERCENT: usize = 85;
const SUMMARY_ITEM_CHARS: usize = 480;
const TOOL_SUMMARY_EDGE_CHARS: usize = 180;

pub struct Agent<P> {
    provider: P,
    tools: ToolRegistry,
    model: String,
    messages: Vec<Message>,
    events: EventSender,
    turn_timeout: Duration,
    max_turns: usize,
    max_context_chars: usize,
    compact_keep_turns: usize,
    thinking_level: ThinkingLevel,
}

pub struct AgentOptions {
    pub model: String,
    pub turn_timeout: Duration,
    pub max_turns: usize,
    pub max_context_chars: usize,
    pub compact_keep_turns: usize,
    pub thinking_level: ThinkingLevel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactStats {
    pub before_chars: usize,
    pub after_chars: usize,
    pub freed_chars: usize,
    pub kept_turns: usize,
    pub summarized_turns: usize,
    pub summarized_tool_outputs: usize,
}

impl<P> Agent<P>
where
    P: Provider,
{
    pub fn new(
        provider: P,
        tools: ToolRegistry,
        events: EventSender,
        options: AgentOptions,
        messages: Option<Vec<Message>>,
    ) -> Self {
        Self {
            provider,
            tools,
            model: options.model,
            messages: normalize_messages(messages.unwrap_or_default()),
            events,
            turn_timeout: options.turn_timeout,
            max_turns: options.max_turns,
            max_context_chars: options.max_context_chars,
            compact_keep_turns: options.compact_keep_turns,
            thinking_level: options.thinking_level,
        }
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub fn thinking_level(&self) -> ThinkingLevel {
        crate::provider::normalize_thinking_level(
            &self.provider.thinking_capabilities(&self.model),
            self.thinking_level,
        )
        .clamped
    }

    pub fn thinking_preference(&self) -> ThinkingLevel {
        self.thinking_level
    }

    pub fn thinking_capabilities(&self) -> crate::provider::ThinkingCapabilities {
        self.provider.thinking_capabilities(&self.model)
    }

    pub fn set_thinking_level(&mut self, thinking_level: ThinkingLevel) {
        self.thinking_level = thinking_level;
    }

    pub fn max_context_chars(&self) -> usize {
        self.max_context_chars
    }

    pub fn default_tool_timeout(&self) -> Duration {
        self.tools.default_timeout()
    }

    pub fn clear(&mut self) {
        self.messages = fresh_messages();
    }

    pub fn replace_messages(&mut self, messages: Vec<Message>) {
        self.messages = normalize_messages(messages);
        self.compact_if_needed();
    }

    pub fn context_chars(&self) -> usize {
        context_chars(&self.messages)
    }

    pub fn compact(&mut self) -> CompactStats {
        let mut stats = compact_messages(&mut self.messages, self.compact_keep_turns);
        while stats.after_chars > self.max_context_chars && stats.kept_turns > 1 {
            let next = compact_messages(&mut self.messages, stats.kept_turns - 1);
            stats.after_chars = next.after_chars;
            stats.freed_chars = stats.before_chars.saturating_sub(next.after_chars);
            stats.kept_turns = next.kept_turns;
            stats.summarized_turns += 1;
            stats.summarized_tool_outputs += next.summarized_tool_outputs;
        }
        stats
    }

    pub fn has_conversation(&self) -> bool {
        self.messages
            .iter()
            .any(|message| !matches!(message, Message::System { .. }))
    }

    pub async fn prompt(&mut self, prompt: impl Into<String>) -> Result<AssistantMessage> {
        match self
            .prompt_with_cancellation(prompt, std::future::pending())
            .await?
        {
            PromptOutcome::Completed(message) => Ok(message),
            PromptOutcome::Cancelled => {
                unreachable!("a pending cancellation future cannot resolve")
            }
        }
    }

    pub async fn prompt_cancellable(
        &mut self,
        prompt: impl Into<String>,
        mut cancellation: tokio::sync::watch::Receiver<bool>,
    ) -> Result<PromptOutcome> {
        self.prompt_with_cancellation(prompt, async move {
            while !*cancellation.borrow() {
                if cancellation.changed().await.is_err() {
                    std::future::pending::<()>().await;
                }
            }
        })
        .await
    }

    async fn prompt_with_cancellation<F>(
        &mut self,
        prompt: impl Into<String>,
        cancellation: F,
    ) -> Result<PromptOutcome>
    where
        F: Future<Output = ()>,
    {
        self.compact_if_needed();
        let checkpoint = self.messages.len();
        let prompt = prompt.into();
        self.messages.push(Message::User {
            content: prompt.clone(),
        });
        let _ = self.events.send(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: prompt,
        });

        let resolution = {
            let turn = tokio::time::timeout(self.turn_timeout, self.run_loop());
            tokio::pin!(turn);
            tokio::pin!(cancellation);

            tokio::select! {
                biased;
                _ = &mut cancellation => None,
                result = &mut turn => Some(result),
            }
        };

        match resolution {
            None => {
                retain_user_prompt(&mut self.messages, checkpoint);
                let _ = self.events.send(AgentEvent::TurnCancelled);
                Ok(PromptOutcome::Cancelled)
            }
            Some(result) => match result {
                Ok(Ok(message)) => Ok(PromptOutcome::Completed(message)),
                Ok(Err(error)) => {
                    retain_user_prompt(&mut self.messages, checkpoint);
                    Err(error)
                }
                Err(_) => {
                    retain_user_prompt(&mut self.messages, checkpoint);
                    let message = format!(
                        "agent turn exceeded its {} second timeout",
                        self.turn_timeout.as_secs()
                    );
                    let _ = self.events.send(AgentEvent::Error {
                        message: message.clone(),
                    });
                    bail!(message);
                }
            },
        }
    }

    /// Repeats provider completion and tool execution until the model returns text only.
    async fn run_loop(&mut self) -> Result<AssistantMessage> {
        let definitions = self.tools.definitions();

        for _ in 0..self.max_turns {
            self.compact_if_needed();
            let assistant = match self
                .provider
                .complete(
                    &self.model,
                    self.thinking_level,
                    &self.messages,
                    &definitions,
                    &self.events,
                )
                .await
            {
                Ok(assistant) => assistant,
                Err(error) => {
                    let _ = self.events.send(AgentEvent::Error {
                        message: format!("{error:#}"),
                    });
                    return Err(error);
                }
            };

            self.messages.push(Message::Assistant {
                content: assistant.content.clone(),
                thinking: assistant.thinking.clone(),
                tool_calls: assistant.tool_calls.clone(),
                provider_state: assistant.provider_state.clone(),
            });

            if assistant.tool_calls.is_empty() {
                let _ = self.events.send(AgentEvent::TurnEnd);
                return Ok(assistant);
            }

            for tool_call in assistant.tool_calls {
                let name = tool_call.name;
                let call_id = tool_call.id;
                let arguments = tool_call.arguments;
                let parsed_arguments = serde_json::from_str::<Value>(&arguments);
                let timeout = parsed_arguments
                    .as_ref()
                    .ok()
                    .and_then(|arguments| self.tools.execution_timeout(arguments).ok())
                    .unwrap_or_else(|| self.tools.default_timeout());
                let _ = self.events.send(AgentEvent::ToolStart {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                    timeout,
                });

                let started = Instant::now();
                let result = match parsed_arguments {
                    Ok(arguments) => self.tools.execute(&name, arguments).await,
                    Err(error) => Err(anyhow::Error::from(error)),
                };
                let elapsed = started.elapsed();
                let (content, is_error) = match result {
                    Ok(output) => (output, false),
                    Err(error) => (format!("tool error: {error:#}"), true),
                };

                let _ = self.events.send(AgentEvent::ToolEnd {
                    call_id: call_id.clone(),
                    name,
                    output: content.clone(),
                    is_error,
                    elapsed,
                });
                self.messages.push(Message::Tool {
                    tool_call_id: call_id,
                    content,
                });
                self.compact_if_needed();
            }
        }

        let message = format!(
            "agent reached the configured limit of {} provider turns",
            self.max_turns
        );
        let _ = self.events.send(AgentEvent::Error {
            message: message.clone(),
        });
        bail!(message)
    }

    fn compact_if_needed(&mut self) -> Option<CompactStats> {
        let threshold = self.max_context_chars.saturating_mul(AUTO_COMPACT_PERCENT) / 100;
        if self.context_chars() < threshold {
            return None;
        }
        let stats = self.compact();
        if stats.freed_chars > 0 {
            let _ = self.events.send(AgentEvent::ContextCompacted {
                stats: stats.clone(),
            });
        }
        Some(stats)
    }
}

fn retain_user_prompt(messages: &mut Vec<Message>, checkpoint: usize) {
    let user_prompt = messages.get(checkpoint).cloned();
    messages.truncate(checkpoint);
    if let Some(user_prompt) = user_prompt {
        messages.push(user_prompt);
    }
}

fn fresh_messages() -> Vec<Message> {
    vec![Message::System {
        content: SYSTEM_PROMPT.to_owned(),
    }]
}

fn normalize_messages(messages: Vec<Message>) -> Vec<Message> {
    if messages.is_empty() {
        return fresh_messages();
    }
    if matches!(messages.first(), Some(Message::System { .. })) {
        messages
    } else {
        let mut normalized = fresh_messages();
        normalized.extend(messages);
        normalized
    }
}

fn context_chars(messages: &[Message]) -> usize {
    messages.iter().map(Message::character_count).sum()
}

fn compact_messages(messages: &mut Vec<Message>, keep_turns: usize) -> CompactStats {
    let before_chars = context_chars(messages);
    let user_indices = messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| matches!(message, Message::User { .. }).then_some(index))
        .collect::<Vec<_>>();
    let existing_summary = matches!(
        messages.get(1),
        Some(Message::System { content })
            if content.starts_with("[Compacted earlier conversation:")
    );
    let kept_turns = user_indices.len().min(keep_turns);
    let summarized_turns = user_indices.len().saturating_sub(kept_turns);
    if summarized_turns == 0 {
        return CompactStats {
            before_chars,
            after_chars: before_chars,
            freed_chars: 0,
            kept_turns,
            summarized_turns: 0,
            summarized_tool_outputs: 0,
        };
    }

    let keep_start = user_indices[summarized_turns];
    let mut tool_names = std::collections::HashMap::new();
    let mut summary_lines = Vec::new();
    if existing_summary && let Message::System { content } = &messages[1] {
        let prior_body = content
            .split_once('\n')
            .map(|(_, body)| body)
            .unwrap_or(content);
        summary_lines.push(prior_body.to_owned());
    }
    let mut summarized_tool_outputs = 0usize;
    for message in &messages[1..keep_start] {
        match message {
            Message::System { content }
                if !content.starts_with("[Compacted earlier conversation:") =>
            {
                summary_lines.push(format!("Prior context: {}", summarize_text(content)));
            }
            Message::System { .. } => {}
            Message::User { content } => {
                summary_lines.push(format!("User: {}", summarize_text(content)));
            }
            Message::Assistant {
                content,
                thinking,
                tool_calls,
                ..
            } => {
                if let Some(thinking) = thinking
                    .as_deref()
                    .filter(|thinking| !thinking.trim().is_empty())
                {
                    summary_lines.push(format!("Assistant thinking: {}", summarize_text(thinking)));
                }
                if !content.trim().is_empty() {
                    summary_lines.push(format!("Assistant: {}", summarize_text(content)));
                }
                for call in tool_calls {
                    tool_names.insert(call.id.clone(), call.name.clone());
                    summary_lines.push(format!(
                        "Tool call {}: {}",
                        call.name,
                        summarize_text(&call.arguments)
                    ));
                }
            }
            Message::Tool {
                tool_call_id,
                content,
            } => {
                summarized_tool_outputs += 1;
                summary_lines.push(format!(
                    "Tool result {}: {}",
                    tool_names
                        .get(tool_call_id)
                        .map(String::as_str)
                        .unwrap_or("unknown"),
                    summarize_tool_output(content)
                ));
            }
        }
    }

    let total_summarized_turns = prior_summary_turns(messages).saturating_add(summarized_turns);
    let summary = Message::System {
        content: format!(
            "[Compacted earlier conversation: {} turn(s)]\n{}",
            total_summarized_turns,
            summary_lines.join("\n")
        ),
    };
    let mut compacted = Vec::with_capacity(messages.len() - keep_start + 2);
    compacted.push(messages[0].clone());
    compacted.push(summary);
    compacted.extend(messages[keep_start..].iter().cloned());
    *messages = compacted;

    let after_chars = context_chars(messages);
    CompactStats {
        before_chars,
        after_chars,
        freed_chars: before_chars.saturating_sub(after_chars),
        kept_turns,
        summarized_turns,
        summarized_tool_outputs,
    }
}

fn prior_summary_turns(messages: &[Message]) -> usize {
    let Some(Message::System { content }) = messages.get(1) else {
        return 0;
    };
    content
        .strip_prefix("[Compacted earlier conversation: ")
        .and_then(|content| content.split_once(" turn(s)]"))
        .and_then(|(turns, _)| turns.parse().ok())
        .unwrap_or(0)
}

fn summarize_text(content: &str) -> String {
    truncate_middle(&content.replace(['\r', '\n'], " "), SUMMARY_ITEM_CHARS)
}

fn summarize_tool_output(content: &str) -> String {
    let count = content.chars().count();
    if count <= TOOL_SUMMARY_EDGE_CHARS * 2 {
        return content.replace(['\r', '\n'], " ");
    }
    let head = content
        .chars()
        .take(TOOL_SUMMARY_EDGE_CHARS)
        .collect::<String>();
    let tail = content
        .chars()
        .rev()
        .take(TOOL_SUMMARY_EDGE_CHARS)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!(
        "{} … [{} chars omitted] … {}",
        head.replace(['\r', '\n'], " "),
        count.saturating_sub(TOOL_SUMMARY_EDGE_CHARS * 2),
        tail.replace(['\r', '\n'], " ")
    )
}

fn truncate_middle(content: &str, max_chars: usize) -> String {
    let count = content.chars().count();
    if count <= max_chars {
        return content.to_owned();
    }
    let edge = max_chars / 2;
    let head = content.chars().take(edge).collect::<String>();
    let tail = content
        .chars()
        .rev()
        .take(edge)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{head} … {tail}")
}

#[cfg(test)]
mod tests {
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
}
