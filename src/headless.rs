use std::{
    io::{self, Write},
    path::Path,
};

use anyhow::{Context, Result, bail};
use tokio::sync::mpsc;

use crate::{
    agent::{Agent, AgentEvent, MessageRole},
    command::{CommandEffect, CommandOutput, command_specs, execute, parse},
    provider::Provider,
    session::{SessionStore, format_session_summaries},
};

pub async fn run_prompt<P>(
    agent: &mut Agent<P>,
    prompt: String,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
    working_dir: &Path,
) -> Result<()>
where
    P: Provider,
{
    match parse(&prompt)? {
        Some(
            crate::command::SlashCommand::Thinking(_)
            | crate::command::SlashCommand::Model
            | crate::command::SlashCommand::Provider,
        ) => {
            bail!("this command is available only in the TUI")
        }
        Some(command) => {
            let result = execute(command, agent, session_store, session_id, working_dir).await?;
            if result.effect == CommandEffect::ReplaceView {
                println!("[context] {} tokens", agent.context_tokens());
            }
            print_command_output(&result.output)?;
            Ok(())
        }
        None => {
            agent.prompt(prompt).await?;
            checkpoint_session(agent, session_store, session_id).await
        }
    }
}

pub async fn run_repl<P>(
    agent: &mut Agent<P>,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
    working_dir: &Path,
) -> Result<()>
where
    P: Provider,
{
    println!("Zex interactive session. Press Ctrl-D or Ctrl-Z then Enter to exit.");
    let stdin = io::stdin();

    loop {
        print!("zex> ");
        io::stdout().flush().context("failed to flush the prompt")?;

        let mut input = String::new();
        if stdin
            .read_line(&mut input)
            .context("failed to read terminal input")?
            == 0
        {
            println!();
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match parse(input) {
            Ok(Some(
                crate::command::SlashCommand::Thinking(_)
                | crate::command::SlashCommand::Model
                | crate::command::SlashCommand::Provider,
            )) => {
                eprintln!("Zex command error: this command is available only in the TUI");
            }
            Ok(Some(command)) => {
                match execute(command, agent, session_store, session_id, working_dir).await {
                    Ok(result) => {
                        if result.effect == CommandEffect::ReplaceView {
                            println!("[context] {} tokens", agent.context_tokens());
                        }
                        print_command_output(&result.output)?;
                    }
                    Err(error) => eprintln!("Zex command error: {error:#}"),
                }
            }
            Ok(None) => match agent.prompt(input).await {
                Ok(_) => checkpoint_session(agent, session_store, session_id).await?,
                Err(_) => continue,
            },
            Err(error) => eprintln!("Zex command error: {error:#}"),
        }
    }

    Ok(())
}

async fn checkpoint_session<P>(
    agent: &Agent<P>,
    session_store: &SessionStore,
    session_id: &mut Option<String>,
) -> Result<()>
where
    P: Provider,
{
    *session_id = Some(
        session_store
            .save(
                session_id.as_deref(),
                agent.model(),
                agent.thinking_preference(),
                agent.messages(),
            )
            .await?,
    );
    Ok(())
}

fn print_command_output(output: &CommandOutput) -> Result<()> {
    match output {
        CommandOutput::Help => {
            let usage_width = command_specs()
                .iter()
                .map(|command| command.usage.len())
                .max()
                .unwrap_or(0);
            for command in command_specs() {
                println!("{:<usage_width$}  {}", command.usage, command.description);
            }
        }
        CommandOutput::Sessions(sessions) => println!("{}", format_session_summaries(sessions)?),
        CommandOutput::ResumePicker(sessions) => {
            println!("{}", format_session_summaries(sessions)?)
        }
        CommandOutput::Status(message) => println!("{message}"),
    }
    Ok(())
}

pub fn spawn_event_printer(
    mut event_receiver: mpsc::UnboundedReceiver<AgentEvent>,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            print_event(event)?;
        }
        Ok(())
    })
}

pub async fn finish_event_printer(printer: tokio::task::JoinHandle<Result<()>>) -> Result<()> {
    printer.await.context("event printer task failed")?
}

fn print_event(event: AgentEvent) -> Result<()> {
    match event {
        AgentEvent::MessageDelta { role, delta } => match role {
            MessageRole::User => {}
            MessageRole::Assistant => {
                print!("{delta}");
                io::stdout()
                    .flush()
                    .context("failed to flush assistant output")?;
            }
        },
        AgentEvent::ThinkingDelta { .. } => {}
        AgentEvent::ThinkingNormalized {
            requested,
            clamped,
            effective,
            provider_value,
        } => eprintln!(
            "[thinking] requested={requested} clamped={clamped} effective={effective} upstream={}",
            provider_value.as_deref().unwrap_or("<omitted>")
        ),
        AgentEvent::ToolStart { name, .. } => println!("\n[tool] {name}"),
        AgentEvent::ToolEnd {
            name,
            is_error,
            change,
            ..
        } => {
            let status = if is_error { "failed" } else { "done" };
            println!("\n[tool] {name}: {status}");
            if let Some(change) = change {
                let (added, removed) = crate::agent::change_counts(&change);
                let counts = match (added, removed) {
                    (added, 0) => format!("+{added}"),
                    (0, removed) => format!("−{removed}"),
                    (added, removed) => format!("+{added} −{removed}"),
                };
                let suffix = if change.before.is_none() {
                    " (new file)"
                } else {
                    ""
                };
                println!("[change] {}: {counts}{suffix}", change.path.display());
            }
        }
        AgentEvent::Error { message } => eprintln!("\nZex error: {message}"),
        AgentEvent::ContextCompacted { stats } => eprintln!(
            "\n[compact] freed approximately {} tokens; kept {} recent turn(s)",
            stats.freed_tokens, stats.kept_turns
        ),
        AgentEvent::ProviderUsage { .. } => {}
        AgentEvent::TurnCancelled => eprintln!("\n[interrupted]"),
        AgentEvent::TurnEnd => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::{
        agent::{AgentOptions, AssistantMessage, Message},
        provider::{PreparedRequest, ThinkingLevel, ToolDefinition},
        tools::ToolRegistry,
    };

    struct ReplyProvider;

    impl Provider for ReplyProvider {
        type Request = ();

        fn prepare_request(
            &self,
            _model: &str,
            _thinking_level: ThinkingLevel,
            messages: &[Message],
            _tools: &[ToolDefinition],
            max_output_tokens: usize,
        ) -> Result<PreparedRequest<Self::Request>> {
            Ok(PreparedRequest::new(
                messages.iter().map(Message::token_estimate).sum(),
                max_output_tokens,
                (),
            ))
        }

        async fn complete(
            &self,
            _request: Self::Request,
            _events: &crate::agent::EventSender,
        ) -> Result<AssistantMessage> {
            Ok(AssistantMessage {
                content: "saved answer".to_owned(),
                thinking: None,
                tool_calls: Vec::new(),
                provider_state: None,
                usage: None,
            })
        }
    }

    fn temporary_directory() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "zex-headless-checkpoint-{}-{unique}",
            std::process::id()
        ))
    }

    #[tokio::test]
    async fn successful_prompt_is_checkpointed_before_returning() {
        let directory = temporary_directory();
        let store = SessionStore::new(directory.clone());
        let (events, _) = mpsc::unbounded_channel();
        let mut agent = Agent::new(
            ReplyProvider,
            ToolRegistry::new(Duration::from_secs(1), 32_000),
            events,
            AgentOptions {
                model: "test-model".to_owned(),
                turn_timeout: Duration::from_secs(1),
                max_turns: 1,
                max_context_tokens: 8_192,
                compact_keep_turns: 2,
                thinking_level: ThinkingLevel::Medium,
            },
            None,
        );
        let mut session_id = None;

        run_prompt(
            &mut agent,
            "persist this turn".to_owned(),
            &store,
            &mut session_id,
            &directory,
        )
        .await
        .unwrap();

        let loaded = store.load(session_id.as_deref()).await.unwrap().unwrap();
        assert!(loaded.messages.iter().any(
            |message| matches!(message, Message::User { content } if content == "persist this turn")
        ));
        assert!(loaded.messages.iter().any(
            |message| matches!(message, Message::Assistant { content, .. } if content == "saved answer")
        ));
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }
}
