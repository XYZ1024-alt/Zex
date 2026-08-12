use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "zex", version, about = "Zex — a minimal AI agent harness core")]
pub struct Cli {
    /// Run one task non-interactively.
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// List saved sessions for the current project.
    Sessions,
    /// Continue a saved session, or the latest session when ID is omitted.
    Resume {
        /// Session ID shown by `zex sessions`.
        id: Option<String>,

        /// Run one task non-interactively after loading the session.
        #[arg(short = 'p', long)]
        prompt: Option<String>,
    },
}

impl Cli {
    pub fn run_prompt(&self) -> Option<&str> {
        match &self.command {
            Some(Command::Resume { prompt, .. }) => prompt.as_deref(),
            _ => self.prompt.as_deref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, Command};

    #[test]
    fn parses_supported_cli_forms() {
        let prompt = Cli::try_parse_from(["zex", "-p", "hello"]).unwrap();
        assert_eq!(prompt.run_prompt(), Some("hello"));

        let sessions = Cli::try_parse_from(["zex", "sessions"]).unwrap();
        assert!(matches!(sessions.command, Some(Command::Sessions)));

        let resume =
            Cli::try_parse_from(["zex", "resume", "session-id", "-p", "continue"]).unwrap();
        assert!(matches!(
            resume.command,
            Some(Command::Resume {
                id: Some(ref id),
                ..
            }) if id == "session-id"
        ));
        assert_eq!(resume.run_prompt(), Some("continue"));
    }
}
