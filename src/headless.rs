use std::io::{self, Write};

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::{
    agent::{Agent, AgentEvent, MessageRole},
    provider::Provider,
};

pub async fn run_prompt<P>(agent: &mut Agent<P>, prompt: String) -> Result<()>
where
    P: Provider,
{
    agent.prompt(prompt).await.map(|_| ())
}

pub async fn run_repl<P>(agent: &mut Agent<P>) -> Result<()>
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

        if agent.prompt(input).await.is_err() {
            continue;
        }
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
        AgentEvent::ToolStart { name, .. } => println!("\n[tool] {name}"),
        AgentEvent::ToolEnd { name, is_error, .. } => {
            let status = if is_error { "failed" } else { "done" };
            println!("\n[tool] {name}: {status}");
        }
        AgentEvent::Error { message } => eprintln!("\nZex error: {message}"),
        AgentEvent::TurnCancelled => eprintln!("\nZex turn interrupted."),
        AgentEvent::TurnEnd => println!(),
    }
    Ok(())
}
