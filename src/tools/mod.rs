mod bash;
mod edit;
mod glob;
mod grep;
mod read;
mod recall;
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

use crate::{
    agent::FileChange,
    memory::{MemoryPointer, MemoryRuntime},
    provider::ToolDefinition,
};

pub use bash::BashTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use recall::{ListPointersTool, PinTool, RecallTool, UnpinTool};
pub use write::WriteTool;

/// Tool execution result. `output` is fed back to the model; `change`
/// carries the file mutation captured by `write`/`edit` for consumers that
/// render diffs, and never enters the model context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutcome {
    pub output: String,
    pub change: Option<FileChange>,
    pub memory: Option<MemoryPointer>,
}

impl ToolOutcome {
    pub fn output_only(output: String) -> Self {
        Self {
            output,
            change: None,
            memory: None,
        }
    }
}

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutcome>> + Send + 'a>>;

pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    fn execute(&self, arguments: Value, timeout: Duration) -> ToolFuture<'_>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    default_timeout: Duration,
    max_output_chars: usize,
    memory: Option<Arc<MemoryRuntime>>,
}

impl ToolRegistry {
    pub fn new(default_timeout: Duration, max_output_chars: usize) -> Self {
        Self {
            tools: HashMap::new(),
            default_timeout,
            max_output_chars,
            memory: None,
        }
    }

    pub fn set_memory(&mut self, memory: Arc<MemoryRuntime>) {
        self.memory = Some(memory);
    }

    pub fn memory(&self) -> Option<&Arc<MemoryRuntime>> {
        self.memory.as_ref()
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

    pub async fn execute(&self, name: &str, arguments: Value) -> Result<ToolOutcome> {
        let Some(tool) = self.tools.get(name) else {
            bail!("unknown tool '{name}'");
        };

        let memory_arguments = arguments.clone();
        let timeout = timeout_from_arguments(&arguments, self.default_timeout)
            .with_context(|| format!("tool '{name}' received invalid arguments"))?;
        let mut outcome = tool
            .execute(arguments, timeout)
            .await
            .with_context(|| format!("tool '{name}' failed"))?;
        if let Some(memory) = &self.memory
            && !MemoryRuntime::is_control_tool(name)
        {
            let pointer = memory
                .store_tool_result(name, &memory_arguments, outcome.output.clone())
                .await?;
            outcome.memory = Some(pointer);
        }
        outcome.output = truncate_output(outcome.output, self.max_output_chars);
        Ok(outcome)
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

    // Keep both edges: errors and summaries usually live at the tail, while
    // headers and the start of listings live at the head.
    let head_chars = max_chars * 7 / 10;
    let tail_chars = max_chars - head_chars;
    let head = output.chars().take(head_chars).collect::<String>();
    let tail = output
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    let omitted = character_count - head_chars - tail_chars;
    format!("{head}\n\n[truncated: {omitted} of {character_count} characters omitted]\n\n{tail}")
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
        sync::Arc,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::{
        BashTool, EditTool, GlobTool, GrepTool, ListPointersTool, PinTool, ReadTool, RecallTool,
        ToolRegistry, UnpinTool, WriteTool,
    };
    use crate::memory::{MemoryConfig, MemoryRuntime};

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
            .unwrap()
            .output;
        assert_eq!(content, "beta");

        let grep = tools
            .execute("grep", json!({"pattern": "beta"}))
            .await
            .unwrap()
            .output;
        assert!(grep.contains("sample.txt:1:beta"));
        let glob = tools
            .execute("glob", json!({"pattern": "*.txt"}))
            .await
            .unwrap()
            .output;
        assert!(glob.contains("sample.txt"));

        let output = tools
            .execute("bash", json!({"command": "echo zex"}))
            .await
            .unwrap()
            .output;
        assert!(output.contains("exit_code: 0"));
        assert!(output.contains("zex"));

        tokio::fs::remove_dir_all(working_dir).await.unwrap();
    }

    #[tokio::test]
    async fn registry_stores_full_output_before_active_view_truncation() {
        let working_dir = temporary_directory("memory-before-truncation");
        tokio::fs::create_dir_all(&working_dir).await.unwrap();
        let original = "addressable-source-".repeat(100);
        tokio::fs::write(working_dir.join("long.txt"), &original)
            .await
            .unwrap();
        let memory = Arc::new(MemoryRuntime::new(MemoryConfig {
            max_recall_tokens: 4_096,
            ..MemoryConfig::default()
        }));
        memory
            .activate(
                "20260815-120000-aabbccdd",
                working_dir.join("session-memory"),
            )
            .await
            .unwrap();
        let mut tools = ToolRegistry::new(Duration::from_secs(5), 32);
        tools.set_memory(Arc::clone(&memory));
        tools.register(ReadTool::new(working_dir.clone()));

        let outcome = tools
            .execute("read", json!({"path": "long.txt"}))
            .await
            .unwrap();
        let pointer = outcome.memory.expect("read result should be addressable");

        assert!(outcome.output.contains("[truncated:"));
        let recalled = memory.recall(&pointer.id, None, None, None).await.unwrap();
        assert!(recalled.ends_with(&original));
        tokio::fs::remove_dir_all(working_dir).await.unwrap();
    }

    #[tokio::test]
    async fn memory_control_tools_validate_and_return_stored_content() {
        let working_dir = temporary_directory("memory-tools");
        let memory = Arc::new(MemoryRuntime::new(MemoryConfig::default()));
        memory
            .activate(
                "20260815-120000-11223344",
                working_dir.join("session-memory"),
            )
            .await
            .unwrap();
        let pointer = memory
            .store_tool_result(
                "read",
                &json!({"path": "small.txt"}),
                "exact body".to_owned(),
            )
            .await
            .unwrap();
        memory.set_active_pointers([pointer.id.clone()]);
        let mut tools = ToolRegistry::new(Duration::from_secs(5), 32_000);
        tools.set_memory(Arc::clone(&memory));
        tools.register(RecallTool::new(Arc::clone(&memory)));
        tools.register(PinTool::new(Arc::clone(&memory)));
        tools.register(UnpinTool::new(Arc::clone(&memory)));
        tools.register(ListPointersTool::new(Arc::clone(&memory)));

        let names = tools
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["list_pointers", "pin", "recall", "unpin"]);
        let recalled = tools
            .execute(
                "recall",
                json!({"id": pointer.id, "reason": "exact verification"}),
            )
            .await
            .unwrap()
            .output;
        assert!(recalled.ends_with("exact body"));
        let missing = tools
            .execute("recall", json!({"id": "§obs_000000000000000000000000"}))
            .await
            .unwrap_err();
        assert!(format!("{missing:#}").contains("does not exist"));
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
            .unwrap()
            .output;
        assert!(grep.contains("visible.rs:1:needle"));
        assert!(grep.contains("stopped at max_results=1"));
        assert!(!grep.contains("ignored"));

        let glob = tools
            .execute("glob", json!({"pattern": "**/*.rs"}))
            .await
            .unwrap()
            .output;
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
            .unwrap()
            .output;
        assert_eq!(output, "ab\n\n[truncated: 3 of 6 characters omitted]\n\nf");
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
            .unwrap()
            .output;
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
            .unwrap()
            .output;
        assert!(edit.contains("[truncated:"));

        tokio::fs::remove_dir_all(working_dir).await.unwrap();
    }

    #[tokio::test]
    async fn write_and_edit_capture_file_changes() {
        let working_dir = temporary_directory("change-capture");
        tokio::fs::create_dir_all(&working_dir).await.unwrap();
        tokio::fs::write(working_dir.join("existing.txt"), "old\nkeep\n")
            .await
            .unwrap();
        let mut tools = ToolRegistry::new(Duration::from_secs(5), 32_000);
        tools.register(WriteTool::new(working_dir.clone()));
        tools.register(EditTool::new(working_dir.clone()));

        let created = tools
            .execute(
                "write",
                json!({"path": "fresh.txt", "content": "new file\n"}),
            )
            .await
            .unwrap()
            .change
            .expect("write should capture a change");
        assert_eq!(created.before, None);
        assert_eq!(created.after, "new file\n");
        assert!(created.path.ends_with("fresh.txt"));

        let overwritten = tools
            .execute(
                "write",
                json!({"path": "existing.txt", "content": "replaced\nkeep\n"}),
            )
            .await
            .unwrap()
            .change
            .expect("overwrite should capture a change");
        assert_eq!(overwritten.before.as_deref(), Some("old\nkeep\n"));
        assert_eq!(overwritten.after, "replaced\nkeep\n");

        let edited = tools
            .execute(
                "edit",
                json!({
                    "path": "existing.txt",
                    "old_text": "replaced",
                    "new_text": "edited"
                }),
            )
            .await
            .unwrap()
            .change
            .expect("edit should capture a change");
        assert_eq!(edited.before.as_deref(), Some("replaced\nkeep\n"));
        assert_eq!(edited.after, "edited\nkeep\n");

        let oversized = "x".repeat(crate::agent::CHANGE_MAX_BYTES + 1);
        let skipped = tools
            .execute("write", json!({"path": "large.txt", "content": oversized}))
            .await
            .unwrap()
            .change;
        assert!(skipped.is_none(), "oversized files skip the change record");

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
