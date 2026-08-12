mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod write;

use std::{
    collections::HashMap,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::provider::ToolDefinition;

pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use write::WriteTool;

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, arguments: Value, timeout: Duration) -> ToolFuture<'_>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    default_timeout: Duration,
    max_output_chars: usize,
}

impl ToolRegistry {
    pub fn new(default_timeout: Duration, max_output_chars: usize) -> Self {
        Self {
            tools: HashMap::new(),
            default_timeout,
            max_output_chars,
        }
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: Tool + 'static,
    {
        let name = tool.definition().name;
        self.tools.insert(name, Arc::new(tool));
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions = self
            .tools
            .values()
            .map(|tool| {
                let mut definition = tool.definition();
                definition.description = format!(
                    "{} Defaults to a {} second timeout; output is truncated to {} characters.",
                    definition.description,
                    self.default_timeout.as_secs(),
                    self.max_output_chars
                );
                if let Some(properties) = definition
                    .parameters
                    .get_mut("properties")
                    .and_then(Value::as_object_mut)
                {
                    properties.insert(
                        "timeout_seconds".to_owned(),
                        json!({
                            "type": "integer",
                            "minimum": 1,
                            "description": format!(
                                "Optional timeout override in seconds; defaults to {}",
                                self.default_timeout.as_secs()
                            )
                        }),
                    );
                }
                definition
            })
            .collect::<Vec<_>>();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    pub async fn execute(&self, name: &str, arguments: Value) -> Result<String> {
        let Some(tool) = self.tools.get(name) else {
            bail!("unknown tool '{name}'");
        };

        let timeout = timeout_from_arguments(&arguments, self.default_timeout)
            .with_context(|| format!("tool '{name}' received invalid arguments"))?;
        let result = tool
            .execute(arguments, timeout)
            .await
            .with_context(|| format!("tool '{name}' failed"))?;
        Ok(truncate_output(result, self.max_output_chars))
    }

    pub fn default_timeout(&self) -> Duration {
        self.default_timeout
    }

    pub fn execution_timeout(&self, arguments: &Value) -> Result<Duration> {
        timeout_from_arguments(arguments, self.default_timeout)
    }
}

fn timeout_from_arguments(arguments: &Value, default: Duration) -> Result<Duration> {
    let Some(value) = arguments.get("timeout_seconds") else {
        return Ok(default);
    };
    let seconds = value
        .as_u64()
        .context("timeout_seconds must be a positive integer")?;
    if seconds == 0 {
        bail!("timeout_seconds must be greater than zero");
    }
    Ok(Duration::from_secs(seconds))
}

pub(crate) fn truncate_output(output: String, max_chars: usize) -> String {
    let character_count = output.chars().count();
    if character_count <= max_chars {
        return output;
    }

    let truncated = output.chars().take(max_chars).collect::<String>();
    format!("{truncated}\n\n[truncated: {character_count} characters total]")
}

pub(crate) fn resolve_path(working_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        working_dir.join(path)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::PathBuf,
        process,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::{BashTool, EditTool, GlobTool, GrepTool, ReadTool, ToolRegistry, WriteTool};

    #[tokio::test]
    async fn registry_executes_all_builtin_tools() {
        let working_dir = temporary_directory("builtins");
        tokio::fs::create_dir_all(&working_dir).await.unwrap();
        let mut tools = ToolRegistry::new(Duration::from_secs(5), 32_000);
        tools.register(ReadTool::new(working_dir.clone()));
        tools.register(BashTool::new(working_dir.clone()));
        tools.register(WriteTool::new(working_dir.clone()));
        tools.register(EditTool::new(working_dir.clone()));
        tools.register(GrepTool::new(working_dir.clone()));
        tools.register(GlobTool::new(working_dir.clone()));

        let names = tools
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["bash", "edit", "glob", "grep", "read", "write"]);
        let definitions = tools.definitions();
        for definition in &definitions {
            assert!(
                definition
                    .parameters
                    .pointer("/properties/timeout_seconds")
                    .is_some(),
                "{} is missing timeout_seconds",
                definition.name
            );
        }
        let grep = definitions
            .iter()
            .find(|definition| definition.name == "grep")
            .unwrap();
        assert!(grep.description.contains("content"));
        let glob = definitions
            .iter()
            .find(|definition| definition.name == "glob")
            .unwrap();
        assert!(glob.description.contains("locating files"));
        let bash = definitions
            .iter()
            .find(|definition| definition.name == "bash")
            .unwrap();
        assert!(bash.description.contains("other system commands"));

        tools
            .execute("write", json!({"path": "sample.txt", "content": "alpha"}))
            .await
            .unwrap();
        tools
            .execute(
                "edit",
                json!({
                    "path": "sample.txt",
                    "old_text": "alpha",
                    "new_text": "beta"
                }),
            )
            .await
            .unwrap();
        let content = tools
            .execute("read", json!({"path": "sample.txt"}))
            .await
            .unwrap();
        assert_eq!(content, "beta");

        let grep = tools
            .execute("grep", json!({"pattern": "beta"}))
            .await
            .unwrap();
        assert!(grep.contains("sample.txt:1:beta"));
        let glob = tools
            .execute("glob", json!({"pattern": "*.txt"}))
            .await
            .unwrap();
        assert!(glob.contains("sample.txt"));

        let output = tools
            .execute("bash", json!({"command": "echo zex"}))
            .await
            .unwrap();
        assert!(output.contains("exit_code: 0"));
        assert!(output.contains("zex"));

        tokio::fs::remove_dir_all(working_dir).await.unwrap();
    }

    #[tokio::test]
    async fn bash_enforces_its_timeout() {
        let working_dir = temporary_directory("timeout");
        tokio::fs::create_dir_all(&working_dir).await.unwrap();
        let mut tools = ToolRegistry::new(Duration::from_millis(20), 32_000);
        tools.register(BashTool::new(working_dir.clone()));

        #[cfg(windows)]
        let command = "ping -n 3 127.0.0.1 > NUL";
        #[cfg(not(windows))]
        let command = "sleep 1";

        let error = tools
            .execute("bash", json!({"command": command}))
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("timeout"));

        for _ in 0..20 {
            match tokio::fs::remove_dir_all(&working_dir).await {
                Ok(()) => return,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
                Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
            }
        }
        panic!(
            "timed out waiting to remove temporary directory {}",
            working_dir.display()
        );
    }

    #[tokio::test]
    async fn search_tools_respect_gitignore_and_result_limits() {
        let working_dir = temporary_directory("search");
        tokio::fs::create_dir_all(working_dir.join("ignored"))
            .await
            .unwrap();
        tokio::fs::write(working_dir.join(".gitignore"), "ignored/\n")
            .await
            .unwrap();
        tokio::fs::write(working_dir.join("visible.rs"), "needle\nneedle\n")
            .await
            .unwrap();
        tokio::fs::write(working_dir.join("ignored/hidden.rs"), "needle\n")
            .await
            .unwrap();
        let mut tools = ToolRegistry::new(Duration::from_secs(5), 32_000);
        tools.register(GrepTool::new(working_dir.clone()));
        tools.register(GlobTool::new(working_dir.clone()));

        let grep = tools
            .execute("grep", json!({"pattern": "needle", "max_results": 1}))
            .await
            .unwrap();
        assert!(grep.contains("visible.rs:1:needle"));
        assert!(grep.contains("stopped at max_results=1"));
        assert!(!grep.contains("ignored"));

        let glob = tools
            .execute("glob", json!({"pattern": "**/*.rs"}))
            .await
            .unwrap();
        assert!(glob.contains("visible.rs"));
        assert!(!glob.contains("ignored"));

        tokio::fs::remove_dir_all(working_dir).await.unwrap();
    }

    #[tokio::test]
    async fn registry_adds_uniform_timeout_and_truncates_all_outputs() {
        let working_dir = temporary_directory("uniform-contract");
        tokio::fs::create_dir_all(&working_dir).await.unwrap();
        tokio::fs::write(working_dir.join("long.txt"), "abcdef")
            .await
            .unwrap();
        let mut tools = ToolRegistry::new(Duration::from_secs(5), 3);
        tools.register(ReadTool::new(working_dir.clone()));

        let definition = tools.definitions().pop().unwrap();
        assert!(
            definition
                .parameters
                .pointer("/properties/timeout_seconds")
                .is_some()
        );
        let output = tools
            .execute("read", json!({"path": "long.txt"}))
            .await
            .unwrap();
        assert_eq!(output, "abc\n\n[truncated: 6 characters total]");
        let error = tools
            .execute("read", json!({"path": "long.txt", "timeout_seconds": 0}))
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("timeout_seconds must be greater than zero"));

        tokio::fs::remove_dir_all(working_dir).await.unwrap();
    }

    #[tokio::test]
    async fn write_and_edit_outputs_use_the_uniform_truncation_limit() {
        let working_dir = temporary_directory("mutation-output");
        tokio::fs::create_dir_all(&working_dir).await.unwrap();
        tokio::fs::write(working_dir.join("sample.txt"), "before")
            .await
            .unwrap();
        let mut tools = ToolRegistry::new(Duration::from_secs(5), 8);
        tools.register(WriteTool::new(working_dir.clone()));
        tools.register(EditTool::new(working_dir.clone()));

        let write = tools
            .execute(
                "write",
                json!({"path": "nested/file.txt", "content": "content"}),
            )
            .await
            .unwrap();
        assert!(write.contains("[truncated:"));

        let edit = tools
            .execute(
                "edit",
                json!({
                    "path": "sample.txt",
                    "old_text": "before",
                    "new_text": "after"
                }),
            )
            .await
            .unwrap();
        assert!(edit.contains("[truncated:"));

        tokio::fs::remove_dir_all(working_dir).await.unwrap();
    }

    fn temporary_directory(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("zex-{label}-{}-{unique}", process::id()))
    }
}
