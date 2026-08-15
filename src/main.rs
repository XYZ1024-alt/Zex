use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::sync::mpsc;
use zex::{
    agent::{Agent, AgentOptions},
    cli::{Cli, Command},
    config::Config,
    headless,
    memory::MemoryRuntime,
    provider::ProviderRegistry,
    session::{SessionStore, format_session_summaries},
    tools::{
        BashTool, EditTool, GlobTool, GrepTool, ListPointersTool, PinTool, ReadTool, RecallTool,
        ToolRegistry, UnpinTool, WriteTool,
    },
    tui,
};

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
    let Config {
        providers,
        active_model,
        working_dir,
        configured,
        tool_timeout,
        agent_timeout,
        max_turns,
        max_tool_output_chars,
        max_context_tokens,
        compact_keep_turns,
        memory,
        default_thinking_level,
        hide_thinking_block,
        theme,
    } = config;
    tui::install_theme(&theme);
    let thinking_level = loaded_session
        .as_ref()
        .and_then(|session| session.thinking_level)
        .unwrap_or(default_thinking_level);
    let resumed_messages = loaded_session.map(|session| session.messages);
    if !configured && !tui::is_available() {
        anyhow::bail!("no Provider configured; start Zex in a TTY and use /provider");
    }
    let model = active_model.map_or_else(String::new, |active_model| active_model.key());
    let provider = ProviderRegistry::new(&providers, agent_timeout)?;
    let provider_registry = provider.clone();
    let (events, event_receiver) = mpsc::unbounded_channel();
    let mut tools = ToolRegistry::new(tool_timeout, max_tool_output_chars);
    if memory.enabled {
        let active_id = match session_id.clone() {
            Some(id) => id,
            None => {
                let id = session_store.allocate_id()?;
                session_id = Some(id.clone());
                id
            }
        };
        let runtime = Arc::new(MemoryRuntime::new(memory));
        runtime
            .activate(&active_id, session_store.memory_directory(&active_id)?)
            .await?;
        tools.set_memory(Arc::clone(&runtime));
        tools.register(RecallTool::new(Arc::clone(&runtime)));
        tools.register(PinTool::new(Arc::clone(&runtime)));
        tools.register(UnpinTool::new(Arc::clone(&runtime)));
        tools.register(ListPointersTool::new(runtime));
    }
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
            max_context_tokens,
            compact_keep_turns,
            thinking_level,
        },
        resumed_messages,
    );
    agent.initialize_memory().await?;

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
                tui::TuiContext {
                    working_dir: &working_dir,
                    hide_thinking_block,
                    providers,
                    provider_registry,
                },
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
            .save(
                session_id.as_deref(),
                agent.model(),
                agent.thinking_preference(),
                agent.messages(),
            )
            .await
            .map_err(|save_error| {
                anyhow::anyhow!("failed to save the session after {error:#}: {save_error:#}")
            })
            .map(Some),
        Ok(()) if agent.has_conversation() => session_store
            .save(
                session_id.as_deref(),
                agent.model(),
                agent.thinking_preference(),
                agent.messages(),
            )
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
