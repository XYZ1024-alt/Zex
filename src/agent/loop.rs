use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::{
    agent::{AgentEvent, AssistantMessage, EventSender, Message},
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
    max_steps: usize,
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
        max_steps: usize,
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
            max_steps,
        }
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub async fn prompt(&mut self, prompt: impl Into<String>) -> Result<AssistantMessage> {
        self.messages.push(Message::User {
            content: prompt.into(),
        });

        match tokio::time::timeout(self.turn_timeout, self.run_loop()).await {
            Ok(result) => result,
            Err(_) => {
                let message = format!(
                    "agent turn exceeded its {} second timeout",
                    self.turn_timeout.as_secs()
                );
                let _ = self.events.send(AgentEvent::Error(message.clone()));
                bail!(message);
            }
        }
    }

    /// Repeats provider completion and tool execution until the model returns text only.
    async fn run_loop(&mut self) -> Result<AssistantMessage> {
        let definitions = self.tools.definitions();

        for _ in 0..self.max_steps {
            let assistant = match self
                .provider
                .complete(&self.messages, &definitions, &self.events)
                .await
            {
                Ok(assistant) => assistant,
                Err(error) => {
                    let _ = self.events.send(AgentEvent::Error(format!("{error:#}")));
                    return Err(error);
                }
            };

            self.messages.push(Message::Assistant {
                content: assistant.content.clone(),
                tool_calls: assistant.tool_calls.clone(),
                provider_state: assistant.provider_state.clone(),
            });

            if assistant.tool_calls.is_empty() {
                self.finish_turn().await;
                return Ok(assistant);
            }

            for tool_call in assistant.tool_calls {
                let name = tool_call.name;
                let call_id = tool_call.id;
                let _ = self.events.send(AgentEvent::ToolStarted {
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

                let _ = self.events.send(AgentEvent::ToolFinished {
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
            "agent reached the configured limit of {} provider steps",
            self.max_steps
        );
        let _ = self.events.send(AgentEvent::Error(message.clone()));
        bail!(message)
    }

    async fn finish_turn(&self) {
        let (acknowledged, acknowledgement) = oneshot::channel();
        if self
            .events
            .send(AgentEvent::TurnFinished { acknowledged })
            .is_ok()
        {
            let _ = acknowledgement.await;
        }
    }
}
