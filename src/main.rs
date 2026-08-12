mod agent;
mod cli;
mod command;
mod config;
mod headless;
mod provider;
mod session;
mod tools;
mod tui;

use agent::{Agent, AgentOptions};
use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command};
use config::Config;
use provider::OpenAiProvider;
use session::{SessionStore, format_session_summaries};
use tokio::sync::mpsc;
use tools::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, ToolRegistry, WriteTool};

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
    let mut session_id = loaded_session.as_ref().map(|session| session.id.clone());
    let session_model = loaded_session
        .as_ref()
        .and_then(|session| session.model.clone());
    let resumed_messages = loaded_session.map(|session| session.messages);
    let Config {
        api_key,
        base_url,
        model,
        openai_api,
        working_dir,
        tool_timeout,
        agent_timeout,
        max_turns,
        max_tool_output_chars,
        max_context_chars,
        compact_keep_turns,
        thinking_level,
        show_thinking,
    } = config;
    let model = session_model.unwrap_or(model);
    let provider = OpenAiProvider::new(&base_url, api_key, openai_api, agent_timeout)?;
    let (events, event_receiver) = mpsc::unbounded_channel();
    let mut tools = ToolRegistry::new(tool_timeout, max_tool_output_chars);
    tools.register(ReadTool::new(working_dir.clone()));
    tools.register(BashTool::new(working_dir.clone()));
    tools.register(WriteTool::new(working_dir.clone()));
    tools.register(EditTool::new(working_dir.clone()));
    tools.register(GrepTool::new(working_dir.clone()));
    tools.register(GlobTool::new(working_dir.clone()));
    let mut agent = Agent::new(
        provider,
        tools,
        events,
        AgentOptions {
            model,
            turn_timeout: agent_timeout,
            max_turns,
            max_context_chars,
            compact_keep_turns,
            thinking_level,
        },
        resumed_messages,
    );

    let (result, event_printer) = if let Some(prompt) = cli.run_prompt() {
        let printer = headless::spawn_event_printer(event_receiver);
        (
            headless::run_prompt(
                &mut agent,
                prompt.to_owned(),
                &session_store,
                &mut session_id,
                &working_dir,
            )
            .await,
            Some(printer),
        )
    } else if tui::is_available() {
        (
            tui::run(
                &mut agent,
                event_receiver,
                &session_store,
                &mut session_id,
                &working_dir,
                show_thinking,
            )
            .await,
            None,
        )
    } else {
        let printer = headless::spawn_event_printer(event_receiver);
        (
            headless::run_repl(&mut agent, &session_store, &mut session_id, &working_dir).await,
            Some(printer),
        )
    };

    let save_result = match &result {
        Err(error) => session_store
            .save(session_id.as_deref(), agent.model(), agent.messages())
            .await
            .map_err(|save_error| {
                anyhow::anyhow!("failed to save the session after {error:#}: {save_error:#}")
            })
            .map(Some),
        Ok(()) if agent.has_conversation() => session_store
            .save(session_id.as_deref(), agent.model(), agent.messages())
            .await
            .map(Some),
        Ok(()) => Ok(None),
    };
    drop(agent);
    if let Some(printer) = event_printer {
        headless::finish_event_printer(printer).await?;
    }
    match (result, save_result) {
        (_, Err(error)) => Err(error),
        (Err(error), Ok(_)) => Err(error),
        (Ok(()), Ok(_)) => Ok(()),
    }
}

async fn print_sessions(session_store: &SessionStore) -> Result<()> {
    let sessions = session_store.list().await?;
    println!("{}", format_session_summaries(&sessions)?);
    Ok(())
}
