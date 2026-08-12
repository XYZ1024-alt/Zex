mod agent;
mod cli;
mod config;
mod provider;
mod session;
mod tools;

use std::io::{self, Write};

use agent::{Agent, AgentEvent};
use anyhow::{Context, Result};
use clap::Parser;
use cli::Cli;
use config::Config;
use provider::OpenAiProvider;
use session::SessionStore;
use tokio::sync::mpsc;
use tools::{BashTool, EditTool, ReadTool, ToolRegistry, WriteTool};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::from_env()?;
    let session_store = SessionStore::new(config.session_dir);
    let resumed_messages = if cli.continue_session {
        session_store.load_latest().await?
    } else {
        None
    };
    let provider = OpenAiProvider::new(
        &config.base_url,
        config.api_key,
        config.model,
        config.openai_api,
        config.agent_timeout,
    )?;
    let (events, mut event_receiver) = mpsc::unbounded_channel();
    let printer = tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            print_event(event);
        }
    });
    let mut tools = ToolRegistry::new();
    tools.register(ReadTool::new(
        config.working_dir.clone(),
        config.max_tool_output_chars,
    ));
    tools.register(BashTool::new(
        config.working_dir.clone(),
        config.bash_timeout,
        config.max_tool_output_chars,
    ));
    tools.register(WriteTool::new(config.working_dir.clone()));
    tools.register(EditTool::new(config.working_dir));
    let mut agent = Agent::new(
        provider,
        tools,
        events,
        config.agent_timeout,
        config.max_steps,
        resumed_messages,
    );

    let result = if let Some(prompt) = cli.prompt {
        agent.prompt(prompt).await.map(|_| ())
    } else {
        run_repl(&mut agent).await
    };

    let save_result = session_store.save(agent.messages()).await;
    drop(agent);
    printer.await.context("event printer task failed")?;
    result?;
    save_result.map(|_| ())
}

async fn run_repl<P>(agent: &mut Agent<P>) -> Result<()>
where
    P: provider::Provider,
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
            return Ok(());
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if agent.prompt(input).await.is_err() {
            continue;
        }
    }
}

fn print_event(event: AgentEvent) {
    match event {
        AgentEvent::TextDelta(delta) => {
            print!("{delta}");
            let _ = io::stdout().flush();
        }
        AgentEvent::TurnFinished { acknowledged } => {
            println!();
            let _ = acknowledged.send(());
        }
        AgentEvent::ToolStarted { call_id, name } => {
            let _ = call_id;
            println!("\n[tool] {name}");
        }
        AgentEvent::ToolFinished {
            call_id,
            name,
            output,
            is_error,
        } => {
            let _ = (call_id, output);
            let status = if is_error { "failed" } else { "done" };
            println!("\n[tool] {name}: {status}");
        }
        AgentEvent::Error(error) => eprintln!("\nZex error: {error}"),
    }
}
