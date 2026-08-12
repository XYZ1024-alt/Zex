use std::{env, path::PathBuf, time::Duration};

use anyhow::{Context, Result};

use crate::provider::OpenAiApi;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_BASH_TIMEOUT_SECONDS: u64 = 60;
const DEFAULT_AGENT_TIMEOUT_SECONDS: u64 = 600;
const DEFAULT_MAX_STEPS: usize = 12;
const DEFAULT_MAX_TOOL_OUTPUT_CHARS: usize = 32_000;
const DEFAULT_SESSION_DIR: &str = ".zex/sessions";

#[derive(Debug, Clone)]
pub struct Config {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub openai_api: OpenAiApi,
    pub working_dir: PathBuf,
    pub bash_timeout: Duration,
    pub agent_timeout: Duration,
    pub max_steps: usize,
    pub max_tool_output_chars: usize,
    pub session_dir: PathBuf,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let api_key = required_env("ZEX_API_KEY", "OPENAI_API_KEY")?;
        let model = required_env("ZEX_MODEL", "OPENAI_MODEL")?;
        let base_url = preferred_env("ZEX_BASE_URL", "OPENAI_BASE_URL")
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        let working_dir =
            env::current_dir().context("failed to determine the working directory")?;
        let session_dir = env::var_os("ZEX_SESSION_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| working_dir.join(DEFAULT_SESSION_DIR));
        let openai_api = env::var("ZEX_OPENAI_API")
            .unwrap_or_else(|_| "chat-completions".to_owned())
            .parse()?;

        Ok(Self {
            api_key,
            base_url,
            model,
            openai_api,
            working_dir,
            bash_timeout: Duration::from_secs(parsed_env(
                "ZEX_BASH_TIMEOUT_SECONDS",
                DEFAULT_BASH_TIMEOUT_SECONDS,
            )?),
            agent_timeout: Duration::from_secs(parsed_env(
                "ZEX_AGENT_TIMEOUT_SECONDS",
                DEFAULT_AGENT_TIMEOUT_SECONDS,
            )?),
            max_steps: parsed_env("ZEX_MAX_STEPS", DEFAULT_MAX_STEPS)?,
            max_tool_output_chars: parsed_env(
                "ZEX_MAX_TOOL_OUTPUT_CHARS",
                DEFAULT_MAX_TOOL_OUTPUT_CHARS,
            )?,
            session_dir,
        })
    }
}

fn preferred_env(primary: &str, fallback: &str) -> Option<String> {
    env::var(primary)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var(fallback)
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
}

fn required_env(primary: &str, fallback: &str) -> Result<String> {
    preferred_env(primary, fallback)
        .with_context(|| format!("set {primary} (preferred) or {fallback} before running Zex"))
}

fn parsed_env<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(value) => value
            .parse()
            .map_err(|error| anyhow::anyhow!("{name} must contain a valid number: {error}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error).with_context(|| format!("failed to read {name}")),
    }
}

#[cfg(test)]
mod tests {
    use crate::provider::OpenAiApi;

    #[test]
    fn parses_supported_openai_api_modes() {
        assert_eq!(
            "chat-completions".parse::<OpenAiApi>().unwrap(),
            OpenAiApi::ChatCompletions
        );
        assert_eq!(
            "responses".parse::<OpenAiApi>().unwrap(),
            OpenAiApi::Responses
        );
        assert!("completions".parse::<OpenAiApi>().is_err());
    }
}
