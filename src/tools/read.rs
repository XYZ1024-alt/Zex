use std::path::PathBuf;

use anyhow::Context;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, resolve_path, truncate_output},
};

pub struct ReadTool {
    working_dir: PathBuf,
    max_output_chars: usize,
}

impl ReadTool {
    pub fn new(working_dir: PathBuf, max_output_chars: usize) -> Self {
        Self {
            working_dir,
            max_output_chars,
        }
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

    fn execute(&self, arguments: Value) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: ReadArguments =
                serde_json::from_value(arguments).context("invalid read arguments")?;
            let path = resolve_path(&self.working_dir, &arguments.path);
            let content = tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("failed to read {}", path.display()))?;
            Ok(truncate_output(content, self.max_output_chars))
        })
    }
}

#[derive(Deserialize)]
struct ReadArguments {
    path: PathBuf,
}
