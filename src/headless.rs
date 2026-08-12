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
        Some(crate::command::SlashCommand::Thinking(_)) => {
            bail!("/thinking is available only in the TUI")
        }
        Some(command) => {
            let result = execute(command, agent, session_store, session_id, working_dir).await?;
            if result.effect == CommandEffect::ReplaceView {
                println!("[context] {} chars", agent.context_chars());
            }
            print_command_output(&result.output)?;
            Ok(())
        }
        None => agent.prompt(prompt).await.map(|_| ()),
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
            Ok(Some(crate::command::SlashCommand::Thinking(_))) => {
                eprintln!("Zex command error: /thinking is available only in the TUI");
            }
            Ok(Some(command)) => {
                match execute(command, agent, session_store, session_id, working_dir).await {
                    Ok(result) => {
                        if result.effect == CommandEffect::ReplaceView {
                            println!("[context] {} chars", agent.context_chars());
                        }
                        print_command_output(&result.output)?;
                    }
                    Err(error) => eprintln!("Zex command error: {error:#}"),
                }
            }
            Ok(None) => {
                if agent.prompt(input).await.is_err() {
                    continue;
                }
            }
            Err(error) => eprintln!("Zex command error: {error:#}"),
        }
    }

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
        AgentEvent::ToolStart { name, .. } => println!("\n[tool] {name}"),
        AgentEvent::ToolEnd { name, is_error, .. } => {
            let status = if is_error { "failed" } else { "done" };
            println!("\n[tool] {name}: {status}");
        }
        AgentEvent::Error { message } => eprintln!("\nZex error: {message}"),
        AgentEvent::ContextCompacted { stats } => eprintln!(
            "\n[compact] freed approximately {} chars; kept {} recent turn(s)",
            stats.freed_chars, stats.kept_turns
        ),
        AgentEvent::TurnCancelled => eprintln!("\nZex turn interrupted."),
        AgentEvent::TurnEnd => println!(),
    }
    Ok(())
}
