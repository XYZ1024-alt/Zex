use std::{path::PathBuf, time::Duration};

use anyhow::Context;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, ToolOutcome, resolve_path},
};

pub struct ReadTool {
    working_dir: PathBuf,
}

impl ReadTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for ReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read".to_owned(),
            description: "Read a UTF-8 file. Relative paths resolve from the working directory."
                .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    fn execute(&self, arguments: Value, _timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: ReadArguments =
                serde_json::from_value(arguments).context("invalid read arguments")?;
            let path = resolve_path(&self.working_dir, &arguments.path);
            let content = tokio::time::timeout(_timeout, tokio::fs::read_to_string(&path))
                .await
                .with_context(|| {
                    format!(
                        "read exceeded its {} second timeout for {}",
                        _timeout.as_secs(),
                        path.display()
                    )
                })?
                .with_context(|| format!("failed to read {}", path.display()))?;
            Ok(ToolOutcome::output_only(content))
        })
    }
}

#[derive(Deserialize)]
struct ReadArguments {
    path: PathBuf,
}
