use anyhow::{Context, Result, bail};

use crate::{
    agent::{Agent, CompactStats},
    provider::Provider,
    session::{SessionStore, format_session_summaries},
};

const HELP: &str = "\
/help                 List slash commands
/model                Show the active model
/model <name>         Switch the active model for this session
/clear                Clear this view and start a fresh context
/sessions             List saved sessions
/resume [id]          Resume the named session, or latest non-current session
/compact              Compact older context and report reclaimed characters";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    Help,
    Model(Option<String>),
    Clear,
    Sessions,
    Resume(Option<String>),
    Compact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandEffect {
    None,
    ClearView,
    ReplaceView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub message: String,
    pub effect: CommandEffect,
}

pub fn parse(input: &str) -> Result<Option<SlashCommand>> {
    let input = input.trim();
    if !input.starts_with('/') {
        return Ok(None);
    }
    let mut parts = input.split_whitespace();
    let name = parts.next().unwrap_or_default();
    let arguments = parts.collect::<Vec<_>>();
    let command = match name {
        "/help" if arguments.is_empty() => SlashCommand::Help,
        "/model" => SlashCommand::Model((!arguments.is_empty()).then(|| arguments.join(" "))),
        "/clear" if arguments.is_empty() => SlashCommand::Clear,
        "/sessions" if arguments.is_empty() => SlashCommand::Sessions,
        "/resume" if arguments.len() <= 1 => {
            SlashCommand::Resume(arguments.first().map(|value| (*value).to_owned()))
        }
        "/compact" if arguments.is_empty() => SlashCommand::Compact,
        "/help" | "/clear" | "/sessions" | "/compact" => {
            bail!("{name} does not accept arguments")
        }
        "/resume" => bail!("/resume accepts at most one session ID"),
        _ => bail!("unknown slash command {name:?}; use /help"),
    };
    Ok(Some(command))
}

pub async fn execute<P>(
    command: SlashCommand,
    agent: &mut Agent<P>,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
) -> Result<CommandResult>
where
    P: Provider,
{
    match command {
        SlashCommand::Help => Ok(CommandResult {
            message: HELP.to_owned(),
            effect: CommandEffect::None,
        }),
        SlashCommand::Model(None) => Ok(CommandResult {
            message: format!("Active model: {}", agent.model()),
            effect: CommandEffect::None,
        }),
        SlashCommand::Model(Some(model)) => {
            let model = model.trim();
            if model.is_empty() {
                bail!("model name must not be empty");
            }
            agent.set_model(model.to_owned());
            Ok(CommandResult {
                message: format!("Active model switched to {model}."),
                effect: CommandEffect::None,
            })
        }
        SlashCommand::Clear => {
            agent.clear();
            *session_id = None;
            Ok(CommandResult {
                message: "Cleared the conversation view and model context. Saved sessions were not deleted; the next prompt starts a new session.".to_owned(),
                effect: CommandEffect::ClearView,
            })
        }
        SlashCommand::Sessions => {
            let sessions = session_store.list().await?;
            Ok(CommandResult {
                message: format_session_summaries(&sessions)?,
                effect: CommandEffect::None,
            })
        }
        SlashCommand::Resume(requested_id) => {
            let loaded = session_store
                .load_excluding(requested_id.as_deref(), session_id.as_deref())
                .await?
                .with_context(|| match requested_id {
                    Some(id) => format!("session {id:?} was not found"),
                    None => "no other saved sessions found".to_owned(),
                })?;
            let resumed_id = loaded.id;
            if let Some(model) = loaded.model {
                agent.set_model(model);
            }
            agent.replace_messages(loaded.messages);
            *session_id = Some(resumed_id.clone());
            Ok(CommandResult {
                message: format!(
                    "Resumed session {resumed_id} with {} message(s) using model {}.",
                    agent
                        .messages()
                        .iter()
                        .filter(|message| !matches!(message, crate::agent::Message::System { .. }))
                        .count(),
                    agent.model()
                ),
                effect: CommandEffect::ReplaceView,
            })
        }
        SlashCommand::Compact => {
            let stats = agent.compact();
            Ok(CommandResult {
                message: compact_feedback(&stats),
                effect: CommandEffect::ReplaceView,
            })
        }
    }
}

#[cfg(test)]
mod execution_tests {
    use std::{
        collections::VecDeque,
        sync::Mutex,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use anyhow::Result;

    use super::{CommandEffect, SlashCommand, execute};
    use crate::{
        agent::{Agent, AgentOptions, AssistantMessage, Message},
        provider::{Provider, ToolDefinition},
        session::SessionStore,
        tools::ToolRegistry,
    };

    struct IdleProvider {
        responses: Mutex<VecDeque<AssistantMessage>>,
    }

    impl Provider for IdleProvider {
        async fn complete(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _events: &crate::agent::EventSender,
        ) -> Result<AssistantMessage> {
            Ok(self
                .responses
                .lock()
                .expect("provider mutex poisoned")
                .pop_front()
                .unwrap_or(AssistantMessage {
                    content: String::new(),
                    tool_calls: Vec::new(),
                    provider_state: None,
                }))
        }
    }

    fn temporary_directory() -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("zex-command-{}-{unique}", std::process::id()))
    }

    fn agent(messages: Vec<Message>) -> Agent<IdleProvider> {
        let (events, _) = tokio::sync::mpsc::unbounded_channel();
        Agent::new(
            IdleProvider {
                responses: Mutex::new(VecDeque::new()),
            },
            ToolRegistry::new(Duration::from_secs(1), 32_000),
            events,
            AgentOptions {
                model: "model-a".to_owned(),
                turn_timeout: Duration::from_secs(1),
                max_turns: 1,
                max_context_chars: 120_000,
                compact_keep_turns: 1,
            },
            Some(messages),
        )
    }

    #[tokio::test]
    async fn clear_starts_a_new_session_without_deleting_saved_sessions() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let saved_id = store
            .save(
                None,
                "saved-model",
                &[Message::User {
                    content: "saved".to_owned(),
                }],
            )
            .await
            .unwrap();
        let mut agent = agent(vec![Message::User {
            content: "current".to_owned(),
        }]);
        let mut session_id = Some(saved_id.clone());

        let result = execute(SlashCommand::Clear, &mut agent, &store, &mut session_id)
            .await
            .unwrap();

        assert_eq!(result.effect, CommandEffect::ClearView);
        assert!(session_id.is_none());
        assert!(!agent.has_conversation());
        assert!(store.load(Some(&saved_id)).await.unwrap().is_some());
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn model_and_resume_update_agent_session_state() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let resumed_id = store
            .save(
                None,
                "saved-model",
                &[Message::User {
                    content: "saved context".to_owned(),
                }],
            )
            .await
            .unwrap();
        let mut agent = agent(Vec::new());
        let mut session_id = None;

        execute(
            SlashCommand::Model(Some("model-b".to_owned())),
            &mut agent,
            &store,
            &mut session_id,
        )
        .await
        .unwrap();
        assert_eq!(agent.model(), "model-b");

        let result = execute(
            SlashCommand::Resume(Some(resumed_id.clone())),
            &mut agent,
            &store,
            &mut session_id,
        )
        .await
        .unwrap();
        assert_eq!(result.effect, CommandEffect::ReplaceView);
        assert_eq!(session_id.as_deref(), Some(resumed_id.as_str()));
        assert_eq!(agent.model(), "saved-model");
        assert!(agent.messages().iter().any(
            |message| matches!(message, Message::User { content } if content == "saved context")
        ));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn compact_command_reports_before_and_after_context_size() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let mut agent = agent(vec![
            Message::User {
                content: "old task".to_owned(),
            },
            Message::Assistant {
                content: String::new(),
                tool_calls: vec![crate::agent::ToolCall {
                    id: "call".to_owned(),
                    name: "read".to_owned(),
                    arguments: r#"{"path":"large.txt"}"#.to_owned(),
                }],
                provider_state: None,
            },
            Message::Tool {
                tool_call_id: "call".to_owned(),
                content: "x".repeat(4_000),
            },
            Message::User {
                content: "recent task".to_owned(),
            },
        ]);
        let before = agent.context_chars();
        let mut session_id = None;

        let result = execute(SlashCommand::Compact, &mut agent, &store, &mut session_id)
            .await
            .unwrap();

        assert_eq!(result.effect, CommandEffect::ReplaceView);
        assert!(result.message.contains("freed approximately"));
        assert!(result.message.contains(&before.to_string()));
        assert!(result.message.contains(&agent.context_chars().to_string()));
        assert!(agent.context_chars() < before);
        let _ = tokio::fs::remove_dir_all(directory).await;
    }
}

fn compact_feedback(stats: &CompactStats) -> String {
    if stats.freed_chars == 0 {
        return format!(
            "Context already compact: {} chars, {} recent turn(s) kept.",
            stats.after_chars, stats.kept_turns
        );
    }
    format!(
        "Compacted context: freed approximately {} chars ({} → {}); kept {} recent turn(s), summarized {} older turn(s) and {} tool output(s).",
        stats.freed_chars,
        stats.before_chars,
        stats.after_chars,
        stats.kept_turns,
        stats.summarized_turns,
        stats.summarized_tool_outputs
    )
}

#[cfg(test)]
mod tests {
    use super::{SlashCommand, parse};

    #[test]
    fn parses_commands_without_treating_messages_as_commands() {
        assert_eq!(parse("hello").unwrap(), None);
        assert_eq!(parse("/help").unwrap(), Some(SlashCommand::Help));
        assert_eq!(
            parse("/model gpt-5").unwrap(),
            Some(SlashCommand::Model(Some("gpt-5".to_owned())))
        );
        assert!(parse("/unknown").is_err());
    }
}
