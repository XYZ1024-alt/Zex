use std::path::PathBuf;

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, resolve_path},
};

pub struct EditTool {
    working_dir: PathBuf,
}

impl EditTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for EditTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit".to_owned(),
            description:
                "Replace one exact text occurrence in a UTF-8 file. Fails if the text is absent or occurs more than once."
                    .to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to edit"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text"
                    }
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        }
    }

    fn execute(&self, arguments: Value) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: EditArguments =
                serde_json::from_value(arguments).context("invalid edit arguments")?;
            if arguments.old_text.is_empty() {
                bail!("old_text must not be empty");
            }

            let path = resolve_path(&self.working_dir, &arguments.path);
            let content = tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("failed to read {}", path.display()))?;
            let match_count = content.matches(&arguments.old_text).count();

            match match_count {
                0 => bail!("old_text was not found in {}", path.display()),
                1 => {}
                count => bail!(
                    "old_text occurs {count} times in {}; make the edit more specific",
                    path.display()
                ),
            }

            let edited = content.replacen(&arguments.old_text, &arguments.new_text, 1);
            tokio::fs::write(&path, edited.as_bytes())
                .await
                .with_context(|| format!("failed to write {}", path.display()))?;
            Ok(format!("edited {}", path.display()))
        })
    }
}

#[derive(Deserialize)]
struct EditArguments {
    path: PathBuf,
    old_text: String,
    new_text: String,
}
