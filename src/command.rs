use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::{
    agent::{Agent, CompactStats},
    config::persist_thinking_level,
    provider::Provider,
    provider::ThinkingLevel,
    session::{SessionStore, SessionSummary},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub usage: &'static str,
    pub description: &'static str,
    pub accepts_arguments: bool,
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        name: "/help",
        usage: "/help",
        description: "List slash commands",
        accepts_arguments: false,
    },
    CommandSpec {
        name: "/model",
        usage: "/model",
        description: "Choose the active configured model",
        accepts_arguments: false,
    },
    CommandSpec {
        name: "/provider",
        usage: "/provider",
        description: "Configure Providers and their models",
        accepts_arguments: false,
    },
    CommandSpec {
        name: "/clear",
        usage: "/clear",
        description: "Start a new session (alias of /new)",
        accepts_arguments: false,
    },
    CommandSpec {
        name: "/new",
        usage: "/new",
        description: "Start a new session (alias of /clear)",
        accepts_arguments: false,
    },
    CommandSpec {
        name: "/sessions",
        usage: "/sessions",
        description: "View saved sessions",
        accepts_arguments: false,
    },
    CommandSpec {
        name: "/resume",
        usage: "/resume [id]",
        description: "Choose a saved session, or resume ID directly",
        accepts_arguments: true,
    },
    CommandSpec {
        name: "/compact",
        usage: "/compact",
        description: "Compact older context",
        accepts_arguments: false,
    },
    CommandSpec {
        name: "/think",
        usage: "/think [level]",
        description: "Show, set, or cycle thinking level",
        accepts_arguments: true,
    },
    CommandSpec {
        name: "/thinking",
        usage: "/thinking [show|hide]",
        description: "Show or set thinking-card visibility",
        accepts_arguments: true,
    },
];

pub fn command_specs() -> &'static [CommandSpec] {
    COMMANDS
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    Help,
    Model,
    Provider,
    NewSession,
    Sessions,
    Resume(Option<String>),
    Compact,
    Think(Option<ThinkingLevel>),
    Thinking(Option<bool>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandEffect {
    None,
    ClearView,
    ReplaceView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandOutput {
    Help,
    Sessions(Vec<SessionSummary>),
    ResumePicker(Vec<SessionSummary>),
    Status(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub output: CommandOutput,
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
        "/model" if arguments.is_empty() => SlashCommand::Model,
        "/provider" if arguments.is_empty() => SlashCommand::Provider,
        "/clear" | "/new" if arguments.is_empty() => SlashCommand::NewSession,
        "/sessions" if arguments.is_empty() => SlashCommand::Sessions,
        "/resume" if arguments.len() <= 1 => {
            SlashCommand::Resume(arguments.first().map(|value| (*value).to_owned()))
        }
        "/compact" if arguments.is_empty() => SlashCommand::Compact,
        "/think" if arguments.len() <= 1 => {
            SlashCommand::Think(arguments.first().map(|value| value.parse()).transpose()?)
        }
        "/thinking" if arguments.len() <= 1 => SlashCommand::Thinking(
            arguments
                .first()
                .map(|value| match value.to_ascii_lowercase().as_str() {
                    "show" | "on" => Ok(true),
                    "hide" | "off" => Ok(false),
                    _ => bail!("/thinking expects show or hide"),
                })
                .transpose()?,
        ),
        "/help" | "/model" | "/provider" | "/clear" | "/new" | "/sessions" | "/compact" => {
            bail!("{name} does not accept arguments")
        }
        "/resume" => bail!("/resume accepts at most one session ID"),
        "/think" => bail!("/think accepts at most one level"),
        "/thinking" => bail!("/thinking accepts at most one value"),
        _ => bail!("unknown slash command {name:?}; use /help"),
    };
    Ok(Some(command))
}

pub async fn execute<P>(
    command: SlashCommand,
    agent: &mut Agent<P>,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
    working_dir: &Path,
) -> Result<CommandResult>
where
    P: Provider,
{
    match command {
        SlashCommand::Help => Ok(CommandResult {
            output: CommandOutput::Help,
            effect: CommandEffect::None,
        }),
        SlashCommand::Model | SlashCommand::Provider => {
            unreachable!("configuration pages are handled by the TUI")
        }
        SlashCommand::NewSession => {
            let saved_id = if agent.has_conversation() {
                Some(
                    session_store
                        .save(session_id.as_deref(), agent.model(), agent.messages())
                        .await?,
                )
            } else {
                None
            };
            agent.clear();
            *session_id = None;
            Ok(CommandResult {
                output: CommandOutput::Status(match saved_id {
                    Some(id) => format!("New session started. Saved previous session {id}."),
                    None => "New session started.".to_owned(),
                }),
                effect: CommandEffect::ClearView,
            })
        }
        SlashCommand::Sessions => {
            let sessions = session_store.list().await?;
            Ok(CommandResult {
                output: CommandOutput::Sessions(sessions),
                effect: CommandEffect::None,
            })
        }
        SlashCommand::Resume(requested_id) => {
            let Some(requested_id) = requested_id else {
                return Ok(CommandResult {
                    output: CommandOutput::ResumePicker(session_store.list().await?),
                    effect: CommandEffect::None,
                });
            };
            let loaded = session_store
                .load(Some(&requested_id))
                .await?
                .with_context(|| {
                    format!("Session {requested_id:?} not found. Use /resume to choose a session.")
                })?;
            let resumed_id = loaded.id;
            agent.replace_messages(loaded.messages);
            *session_id = Some(resumed_id.clone());
            Ok(CommandResult {
                output: CommandOutput::Status(format!(
                    "Resumed {resumed_id} · {} messages · model {}",
                    agent
                        .messages()
                        .iter()
                        .filter(|message| !matches!(message, crate::agent::Message::System { .. }))
                        .count(),
                    agent.model()
                )),
                effect: CommandEffect::ReplaceView,
            })
        }
        SlashCommand::Compact => {
            let stats = agent.compact();
            Ok(CommandResult {
                output: CommandOutput::Status(compact_feedback(&stats)),
                effect: CommandEffect::ReplaceView,
            })
        }
        SlashCommand::Think(requested) => {
            let thinking_level = requested.unwrap_or_else(|| agent.cycle_thinking_level());
            agent.set_thinking_level(thinking_level);
            persist_thinking_level(working_dir, thinking_level).await?;
            let message = match agent.thinking_level() {
                Some(level) => format!("Thinking · {level}"),
                None => format!(
                    "Thinking · n/a for {} · preference {thinking_level}",
                    agent.model()
                ),
            };
            Ok(CommandResult {
                output: CommandOutput::Status(message),
                effect: CommandEffect::None,
            })
        }
        SlashCommand::Thinking(_) => {
            unreachable!("thinking visibility is handled by the TUI")
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

    use super::{CommandEffect, CommandOutput, SlashCommand, execute};
    use crate::{
        agent::{Agent, AgentOptions, AssistantMessage, Message},
        provider::{Provider, ThinkingLevel, ToolDefinition},
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
            _thinking_level: Option<crate::provider::ThinkingLevel>,
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
                    thinking: None,
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
                thinking_level: crate::provider::ThinkingLevel::Medium,
            },
            Some(messages),
        )
    }

    #[tokio::test]
    async fn clear_saves_the_current_conversation_and_starts_a_new_session() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let mut agent = agent(vec![Message::User {
            content: "current".to_owned(),
        }]);
        let mut session_id = None;

        let result = execute(
            SlashCommand::NewSession,
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();

        assert_eq!(result.effect, CommandEffect::ClearView);
        assert!(session_id.is_none());
        assert!(!agent.has_conversation());
        let saved = store.list().await.unwrap();
        assert_eq!(saved.len(), 1);
        let loaded = store.load(Some(&saved[0].id)).await.unwrap().unwrap();
        assert!(
            loaded.messages.iter().any(
                |message| matches!(message, Message::User { content } if content == "current")
            )
        );
        assert!(matches!(
            result.output,
            CommandOutput::Status(message) if message.contains(&saved[0].id)
        ));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn resume_updates_session_state_without_changing_the_active_model() {
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

        let result = execute(
            SlashCommand::Resume(Some(resumed_id.clone())),
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();
        assert_eq!(result.effect, CommandEffect::ReplaceView);
        assert_eq!(session_id.as_deref(), Some(resumed_id.as_str()));
        assert_eq!(agent.model(), "model-a");
        assert!(agent.messages().iter().any(
            |message| matches!(message, Message::User { content } if content == "saved context")
        ));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn resume_without_id_returns_recent_session_choices_without_switching() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let older_id = store
            .save(
                None,
                "older-model",
                &[Message::User {
                    content: "older context".to_owned(),
                }],
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(2)).await;
        let newer_id = store
            .save(
                None,
                "newer-model",
                &[Message::User {
                    content: "newer context".to_owned(),
                }],
            )
            .await
            .unwrap();
        let original_messages = vec![Message::User {
            content: "current context".to_owned(),
        }];
        let mut agent = agent(original_messages.clone());
        let mut session_id = Some("current-session".to_owned());

        let result = execute(
            SlashCommand::Resume(None),
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();

        assert_eq!(result.effect, CommandEffect::None);
        let CommandOutput::ResumePicker(sessions) = result.output else {
            panic!("expected resume picker");
        };
        assert_eq!(
            sessions
                .iter()
                .map(|session| session.id.as_str())
                .collect::<Vec<_>>(),
            vec![newer_id.as_str(), older_id.as_str()]
        );
        assert_eq!(session_id.as_deref(), Some("current-session"));
        assert_eq!(
            agent
                .messages()
                .iter()
                .filter(|message| !matches!(message, Message::System { .. }))
                .cloned()
                .collect::<Vec<_>>(),
            original_messages
        );
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn sessions_lists_without_switching_the_active_conversation() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let saved_id = store
            .save(
                None,
                "saved-model",
                &[Message::User {
                    content: "saved context".to_owned(),
                }],
            )
            .await
            .unwrap();
        let original_messages = vec![Message::User {
            content: "current context".to_owned(),
        }];
        let mut agent = agent(original_messages.clone());
        let mut session_id = Some("current-session".to_owned());

        let result = execute(
            SlashCommand::Sessions,
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();

        assert_eq!(result.effect, CommandEffect::None);
        assert!(matches!(
            result.output,
            CommandOutput::Sessions(sessions)
                if sessions.len() == 1 && sessions[0].id == saved_id
        ));
        assert_eq!(session_id.as_deref(), Some("current-session"));
        assert_eq!(
            agent
                .messages()
                .iter()
                .filter(|message| !matches!(message, Message::System { .. }))
                .cloned()
                .collect::<Vec<_>>(),
            original_messages
        );
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn invalid_resume_id_returns_a_short_error_without_switching() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let original_messages = vec![Message::User {
            content: "keep this context".to_owned(),
        }];
        let mut agent = agent(original_messages.clone());
        let mut session_id = Some("current-session".to_owned());

        let error = execute(
            SlashCommand::Resume(Some("missing-session".to_owned())),
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Session \"missing-session\" not found. Use /resume to choose a session."
        );
        assert_eq!(session_id.as_deref(), Some("current-session"));
        assert_eq!(
            agent
                .messages()
                .iter()
                .filter(|message| !matches!(message, Message::System { .. }))
                .cloned()
                .collect::<Vec<_>>(),
            original_messages
        );
    }

    #[tokio::test]
    async fn continued_chat_saves_back_to_the_resumed_session() {
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
            SlashCommand::Resume(Some(resumed_id.clone())),
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();
        agent.prompt("continued question").await.unwrap();
        let saved_id = store
            .save(session_id.as_deref(), agent.model(), agent.messages())
            .await
            .unwrap();
        let reloaded = store.load(Some(&resumed_id)).await.unwrap().unwrap();

        assert_eq!(saved_id, resumed_id);
        assert!(reloaded.messages.iter().any(
            |message| matches!(message, Message::User { content } if content == "continued question")
        ));
        assert_eq!(store.list().await.unwrap().len(), 1);
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
                thinking: None,
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

        let result = execute(
            SlashCommand::Compact,
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();

        assert_eq!(result.effect, CommandEffect::ReplaceView);
        let CommandOutput::Status(message) = result.output else {
            panic!("expected compact status");
        };
        assert!(message.contains("freed approximately"));
        assert!(message.contains(&before.to_string()));
        assert!(message.contains(&agent.context_chars().to_string()));
        assert!(agent.context_chars() < before);
        let _ = tokio::fs::remove_dir_all(directory).await;
    }

    #[tokio::test]
    async fn think_command_cycles_and_persists_preference() {
        let directory = temporary_directory();
        tokio::fs::create_dir_all(&directory).await.unwrap();
        let store = SessionStore::new(directory.join("sessions"));
        let mut agent = agent(Vec::new());
        let mut session_id = None;

        let result = execute(
            SlashCommand::Think(Some(ThinkingLevel::High)),
            &mut agent,
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();

        assert_eq!(agent.thinking_preference(), ThinkingLevel::High);
        assert!(matches!(
            result.output,
            CommandOutput::Status(message) if message == "Thinking · n/a for model-a · preference high"
        ));
        let config = tokio::fs::read_to_string(directory.join(".zex/config.toml"))
            .await
            .unwrap();
        assert!(config.contains("thinking_level = \"high\""));
        tokio::fs::remove_dir_all(directory).await.unwrap();
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
    use super::{COMMANDS, SlashCommand, command_specs, parse};
    use crate::provider::ThinkingLevel;

    #[test]
    fn parses_commands_without_treating_messages_as_commands() {
        assert_eq!(parse("hello").unwrap(), None);
        assert_eq!(parse("/help").unwrap(), Some(SlashCommand::Help));
        assert_eq!(parse("/clear").unwrap(), Some(SlashCommand::NewSession));
        assert_eq!(parse("/new").unwrap(), Some(SlashCommand::NewSession));
        assert_eq!(parse("/model").unwrap(), Some(SlashCommand::Model));
        assert_eq!(parse("/provider").unwrap(), Some(SlashCommand::Provider));
        assert!(parse("/model gpt-5").is_err());
        assert!(parse("/unknown").is_err());
        assert_eq!(
            parse("/think high").unwrap(),
            Some(SlashCommand::Think(Some(ThinkingLevel::High)))
        );
        assert_eq!(
            parse("/thinking hide").unwrap(),
            Some(SlashCommand::Thinking(Some(false)))
        );
        assert_eq!(
            parse("/thinking").unwrap(),
            Some(SlashCommand::Thinking(None))
        );
        assert!(parse("/thinking maybe").is_err());
        assert!(
            command_specs()
                .iter()
                .any(|command| command.name == "/think")
        );
        assert!(
            command_specs()
                .iter()
                .any(|command| command.name == "/thinking")
        );
        assert_eq!(command_specs(), COMMANDS);
        assert_eq!(
            command_specs()
                .iter()
                .find(|command| command.name == "/clear")
                .map(|command| command.description),
            Some("Start a new session (alias of /new)")
        );
        assert_eq!(
            command_specs()
                .iter()
                .find(|command| command.name == "/new")
                .map(|command| command.description),
            Some("Start a new session (alias of /clear)")
        );
        assert_eq!(
            command_specs()
                .iter()
                .find(|command| command.name == "/sessions")
                .map(|command| command.description),
            Some("View saved sessions")
        );
        assert_eq!(
            command_specs()
                .iter()
                .find(|command| command.name == "/resume")
                .map(|command| (command.usage, command.description)),
            Some((
                "/resume [id]",
                "Choose a saved session, or resume ID directly"
            ))
        );
    }
}
