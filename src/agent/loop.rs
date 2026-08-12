use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::{
    agent::{AgentEvent, AssistantMessage, EventSender, Message, MessageRole},
    provider::Provider,
    tools::ToolRegistry,
};

const SYSTEM_PROMPT: &str = "You are Zex, a minimal AI agent core. Be concise and accurate. Use the available tools when they are needed, then use their results to finish the task.";

pub struct Agent<P> {
    provider: P,
    tools: ToolRegistry,
    messages: Vec<Message>,
    events: EventSender,
    turn_timeout: Duration,
    max_turns: usize,
}

impl<P> Agent<P>
where
    P: Provider,
{
    pub fn new(
        provider: P,
        tools: ToolRegistry,
        events: EventSender,
        turn_timeout: Duration,
        max_turns: usize,
        messages: Option<Vec<Message>>,
    ) -> Self {
        Self {
            provider,
            tools,
            messages: messages
                .filter(|messages| !messages.is_empty())
                .unwrap_or_else(|| {
                    vec![Message::System {
                        content: SYSTEM_PROMPT.to_owned(),
                    }]
                }),
            events,
            turn_timeout,
            max_turns,
        }
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn has_conversation(&self) -> bool {
        self.messages
            .iter()
            .any(|message| !matches!(message, Message::System { .. }))
    }

    pub async fn prompt(&mut self, prompt: impl Into<String>) -> Result<AssistantMessage> {
        let prompt = prompt.into();
        self.messages.push(Message::User {
            content: prompt.clone(),
        });
        let _ = self.events.send(AgentEvent::MessageDelta {
            role: MessageRole::User,
            delta: prompt,
        });

        match tokio::time::timeout(self.turn_timeout, self.run_loop()).await {
            Ok(result) => result,
            Err(_) => {
                let message = format!(
                    "agent turn exceeded its {} second timeout",
                    self.turn_timeout.as_secs()
                );
                let _ = self.events.send(AgentEvent::Error {
                    message: message.clone(),
                });
                bail!(message);
            }
        }
    }

    /// Repeats provider completion and tool execution until the model returns text only.
    async fn run_loop(&mut self) -> Result<AssistantMessage> {
        let definitions = self.tools.definitions();

        for _ in 0..self.max_turns {
            let assistant = match self
                .provider
                .complete(&self.messages, &definitions, &self.events)
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
                let _ = self.events.send(AgentEvent::ToolStart {
                    call_id: call_id.clone(),
                    name: name.clone(),
                });

                let result = match serde_json::from_str::<Value>(&tool_call.arguments) {
                    Ok(arguments) => self.tools.execute(&name, arguments).await,
                    Err(error) => Err(anyhow::Error::from(error)),
                };
                let (content, is_error) = match result {
                    Ok(output) => (output, false),
                    Err(error) => (format!("tool error: {error:#}"), true),
                };

                let _ = self.events.send(AgentEvent::ToolEnd {
                    call_id: call_id.clone(),
                    name,
                    output: content.clone(),
                    is_error,
                });
                self.messages.push(Message::Tool {
                    tool_call_id: call_id,
                    content,
                });
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
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, sync::Mutex, time::Duration};

    use anyhow::{Result, bail};
    use serde_json::{Value, json};
    use tokio::sync::mpsc;

    use crate::{
        agent::{AgentEvent, AssistantMessage, MessageRole, ToolCall},
        provider::{Provider, ToolDefinition},
        tools::{Tool, ToolFuture, ToolRegistry},
    };

    use super::Agent;

    struct SequenceProvider {
        messages: Mutex<VecDeque<AssistantMessage>>,
    }

    impl Provider for SequenceProvider {
        async fn complete(
            &self,
            _messages: &[crate::agent::Message],
            _tools: &[ToolDefinition],
            events: &crate::agent::EventSender,
        ) -> Result<AssistantMessage> {
            let message = self
                .messages
                .lock()
                .expect("sequence provider mutex poisoned")
                .pop_front()
                .expect("sequence provider exhausted");
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
            _messages: &[crate::agent::Message],
            _tools: &[ToolDefinition],
            _events: &crate::agent::EventSender,
        ) -> Result<AssistantMessage> {
            bail!("provider unavailable")
        }
    }

    #[test]
    fn system_only_agent_has_no_conversation() {
        let provider = SequenceProvider {
            messages: Mutex::new(VecDeque::new()),
        };
        let (events, _) = mpsc::unbounded_channel();
        let agent = Agent::new(
            provider,
            ToolRegistry::new(),
            events,
            Duration::from_secs(1),
            1,
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

        fn execute(&self, arguments: Value) -> ToolFuture<'_> {
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
        let provider = SequenceProvider {
            messages: Mutex::new(VecDeque::from([
                AssistantMessage {
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "call-1".to_owned(),
                        name: "echo".to_owned(),
                        arguments: r#"{"value":"observed"}"#.to_owned(),
                    }],
                    provider_state: None,
                },
                AssistantMessage {
                    content: "done".to_owned(),
                    tool_calls: Vec::new(),
                    provider_state: None,
                },
            ])),
        };
        let mut tools = ToolRegistry::new();
        tools.register(EchoTool);
        let (events, mut receiver) = mpsc::unbounded_channel();
        let mut agent = Agent::new(provider, tools, events, Duration::from_secs(1), 3, None);

        agent.prompt("inspect").await.unwrap();

        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::MessageDelta {
                role: MessageRole::User,
                delta: "inspect".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ToolStart {
                call_id: "call-1".to_owned(),
                name: "echo".to_owned(),
            }
        );
        assert_eq!(
            receiver.try_recv().unwrap(),
            AgentEvent::ToolEnd {
                call_id: "call-1".to_owned(),
                name: "echo".to_owned(),
                output: "observed".to_owned(),
                is_error: false,
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
            ToolRegistry::new(),
            events,
            Duration::from_secs(1),
            1,
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
}
