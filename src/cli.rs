use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "zex", version, about = "Zex — a minimal AI agent harness core")]
pub struct Cli {
    /// Run one task non-interactively.
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    /// Continue the most recently saved session.
    #[arg(short = 'c', long)]
    pub continue_session: bool,
}
