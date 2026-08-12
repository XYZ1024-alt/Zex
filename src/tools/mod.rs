mod bash;
mod edit;
mod read;
mod write;

use std::{
    collections::HashMap,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use anyhow::{Result, bail};
use serde_json::Value;

use crate::provider::ToolDefinition;

pub use bash::BashTool;
pub use edit::EditTool;
pub use read::ReadTool;
pub use write::WriteTool;

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, arguments: Value) -> ToolFuture<'_>;
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
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
            .map(|tool| tool.definition())
            .collect::<Vec<_>>();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    pub async fn execute(&self, name: &str, arguments: Value) -> Result<String> {
        let Some(tool) = self.tools.get(name) else {
            bail!("unknown tool '{name}'");
        };

        tool.execute(arguments).await
    }
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

    use super::{BashTool, EditTool, ReadTool, ToolRegistry, WriteTool};

    #[tokio::test]
    async fn registry_executes_all_builtin_tools() {
        let working_dir = temporary_directory("builtins");
        tokio::fs::create_dir_all(&working_dir).await.unwrap();
        let mut tools = ToolRegistry::new();
        tools.register(ReadTool::new(working_dir.clone(), 32_000));
        tools.register(BashTool::new(
            working_dir.clone(),
            Duration::from_secs(5),
            32_000,
        ));
        tools.register(WriteTool::new(working_dir.clone()));
        tools.register(EditTool::new(working_dir.clone()));

        let names = tools
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["bash", "edit", "read", "write"]);

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
        let mut tools = ToolRegistry::new();
        tools.register(BashTool::new(
            working_dir.clone(),
            Duration::from_millis(20),
            32_000,
        ));

        #[cfg(windows)]
        let command = "ping -n 3 127.0.0.1 > NUL";
        #[cfg(not(windows))]
        let command = "sleep 1";

        let error = tools
            .execute("bash", json!({"command": command}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timeout"));

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

    fn temporary_directory(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("zex-{label}-{}-{unique}", process::id()))
    }
}
