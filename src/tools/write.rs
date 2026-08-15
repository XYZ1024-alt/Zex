use std::{path::PathBuf, time::Duration};

use anyhow::Context;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    agent::FileChange,
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, ToolOutcome, resolve_path},
};

pub struct WriteTool {
    working_dir: PathBuf,
}

impl WriteTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for WriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write".to_owned(),
            description: "Create or replace a UTF-8 file. Parent directories are created."
                .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }
    }

    fn execute(&self, arguments: Value, timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: WriteArguments =
                serde_json::from_value(arguments).context("invalid write arguments")?;
            let path = resolve_path(&self.working_dir, &arguments.path);

            tokio::time::timeout(timeout, async {
                let before = tokio::fs::read_to_string(&path).await.ok();
                if let Some(parent) = path.parent() {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .with_context(|| format!("failed to create {}", parent.display()))?;
                }
                tokio::fs::write(&path, arguments.content.as_bytes())
                    .await
                    .with_context(|| format!("failed to write {}", path.display()))?;

                Ok(ToolOutcome {
                    output: format!(
                        "wrote {} bytes to {}",
                        arguments.content.len(),
                        path.display()
                    ),
                    change: FileChange::capture(path.clone(), before, arguments.content),
                    memory: None,
                })
            })
            .await
            .with_context(|| {
                format!(
                    "write exceeded its {} second timeout for {}",
                    timeout.as_secs(),
                    path.display()
                )
            })?
        })
    }
}

#[derive(Deserialize)]
struct WriteArguments {
    path: PathBuf,
    content: String,
}
