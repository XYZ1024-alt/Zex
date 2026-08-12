mod agent;
mod cli;
mod config;
mod headless;
mod provider;
mod session;
mod tools;
mod tui;

use agent::Agent;
use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command};
use config::Config;
use provider::OpenAiProvider;
use session::SessionStore;
use tokio::sync::mpsc;
use tools::{BashTool, EditTool, ReadTool, ToolRegistry, WriteTool};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let session_store = SessionStore::new(Config::session_dir().await?);

    if matches!(&cli.command, Some(Command::Sessions)) {
        print_sessions(&session_store).await?;
        return Ok(());
    }
    let loaded_session = match &cli.command {
        Some(Command::Resume { id, .. }) => Some(
            session_store
                .load(id.as_deref())
                .await?
                .with_context(|| match id {
                    Some(id) => format!("session {id:?} was not found"),
                    None => "no saved sessions found".to_owned(),
                })?,
        ),
        _ => None,
    };
    let config = Config::load().await?;
    let session_id = loaded_session.as_ref().map(|session| session.id.clone());
    let resumed_messages = loaded_session.map(|session| session.messages);
    let Config {
        api_key,
        base_url,
        model,
        openai_api,
        working_dir,
        bash_timeout,
        agent_timeout,
        max_turns,
        max_tool_output_chars,
    } = config;
    let provider = OpenAiProvider::new(&base_url, api_key, model, openai_api, agent_timeout)?;
    let (events, event_receiver) = mpsc::unbounded_channel();
    let mut tools = ToolRegistry::new();
    tools.register(ReadTool::new(working_dir.clone(), max_tool_output_chars));
    tools.register(BashTool::new(
        working_dir.clone(),
        bash_timeout,
        max_tool_output_chars,
    ));
    tools.register(WriteTool::new(working_dir.clone()));
    tools.register(EditTool::new(working_dir));
    let mut agent = Agent::new(
        provider,
        tools,
        events,
        agent_timeout,
        max_turns,
        resumed_messages,
    );

    let (result, event_printer) = if let Some(prompt) = cli.run_prompt() {
        let printer = headless::spawn_event_printer(event_receiver);
        (
            headless::run_prompt(&mut agent, prompt.to_owned()).await,
            Some(printer),
        )
    } else if tui::is_available() {
        (tui::run(&mut agent, event_receiver).await, None)
    } else {
        let printer = headless::spawn_event_printer(event_receiver);
        (headless::run_repl(&mut agent).await, Some(printer))
    };

    let save_result = if agent.has_conversation() {
        Some(
            session_store
                .save(session_id.as_deref(), agent.messages())
                .await,
        )
    } else {
        None
    };
    drop(agent);
    if let Some(printer) = event_printer {
        headless::finish_event_printer(printer).await?;
    }
    result?;
    match save_result {
        Some(result) => result.map(|_| ()),
        None => Ok(()),
    }
}

async fn print_sessions(session_store: &SessionStore) -> Result<()> {
    let sessions = session_store.list().await?;
    if sessions.is_empty() {
        println!("No saved sessions.");
        return Ok(());
    }

    println!(
        "{:<28}  {:<25}  {:>8}  Preview",
        "ID", "Updated", "Messages"
    );
    for session in sessions {
        let updated_at = session
            .updated_at
            .to_offset(time::UtcOffset::current_local_offset().unwrap_or(time::UtcOffset::UTC))
            .format(&time::format_description::well_known::Rfc3339)
            .context("failed to format session timestamp")?;
        println!(
            "{:<28}  {:<25}  {:>8}  {}",
            session.id, updated_at, session.message_count, session.preview
        );
    }
    Ok(())
}
