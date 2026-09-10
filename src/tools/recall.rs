use std::{sync::Arc, time::Duration};

use anyhow::Context;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    memory::MemoryRuntime,
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, ToolOutcome},
};

pub struct RecallTool {
    memory: Arc<MemoryRuntime>,
}

impl RecallTool {
    pub fn new(memory: Arc<MemoryRuntime>) -> Self {
        Self { memory }
    }
}

#[derive(Debug, Deserialize)]
struct RecallArguments {
    id: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    offset_tokens: Option<usize>,
    #[serde(default)]
    max_tokens: Option<usize>,
}

impl Tool for RecallTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall".to_owned(),
            description: "Retrieve exact content for an addressable §id that is already visible or returned by list_pointers. Use only when missing details directly affect the current decision. Never invent IDs, never guess after an error, and do not call repeatedly for information already present in the active context.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "An exact existing §obs_..., §turn_..., §decision_..., or §summary_... ID"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional short explanation of the decision that requires exact content"
                    },
                    "offset_tokens": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional zero-based token offset for paginating a large record; use next_offset from the previous recall"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional requested window size, capped by memory.max_recall_tokens"
                    }
                },
                "required": ["id"],
                "additionalProperties": false
            }),
        }
    }

    fn execute(&self, arguments: Value, _timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: RecallArguments =
                serde_json::from_value(arguments).context("invalid recall arguments")?;
            Ok(ToolOutcome::output_only(
                self.memory
                    .recall(
                        &arguments.id,
                        arguments.reason,
                        arguments.offset_tokens,
                        arguments.max_tokens,
                    )
                    .await?,
            ))
        })
    }
}

pub struct PinTool {
    memory: Arc<MemoryRuntime>,
}

impl PinTool {
    pub fn new(memory: Arc<MemoryRuntime>) -> Self {
        Self { memory }
    }
}

pub struct UnpinTool {
    memory: Arc<MemoryRuntime>,
}

impl UnpinTool {
    pub fn new(memory: Arc<MemoryRuntime>) -> Self {
        Self { memory }
    }
}

#[derive(Debug, Deserialize)]
struct PointerArguments {
    id: String,
}

impl Tool for PinTool {
    fn definition(&self) -> ToolDefinition {
        pointer_definition(
            "pin",
            "Mark an existing addressable §id as high priority. Pin only evidence or decisions that must remain easy to discover across compaction; never pin an invented or unverified ID.",
        )
    }

    fn execute(&self, arguments: Value, _timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: PointerArguments =
                serde_json::from_value(arguments).context("invalid pin arguments")?;
            Ok(ToolOutcome::output_only(
                self.memory.pin(&arguments.id).await?,
            ))
        })
    }
}

impl Tool for UnpinTool {
    fn definition(&self) -> ToolDefinition {
        pointer_definition(
            "unpin",
            "Remove high-priority status from an existing addressable §id when it is no longer central. This does not delete content and must not be used for unknown IDs.",
        )
    }

    fn execute(&self, arguments: Value, _timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: PointerArguments =
                serde_json::from_value(arguments).context("invalid unpin arguments")?;
            Ok(ToolOutcome::output_only(
                self.memory.unpin(&arguments.id).await?,
            ))
        })
    }
}

pub struct ListPointersTool {
    memory: Arc<MemoryRuntime>,
}

impl ListPointersTool {
    pub fn new(memory: Arc<MemoryRuntime>) -> Self {
        Self { memory }
    }
}

#[derive(Debug, Deserialize)]
struct ListArguments {
    #[serde(default)]
    filter: Option<String>,
}

impl Tool for ListPointersTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_pointers".to_owned(),
            description: "List valid addressable memory IDs with bounded content previews. Without a filter it shows active and pinned pointers. Plain-text filters search metadata and full stored content; structured terms such as tool=read, path=src/main.rs, kind=file_snapshot, role=user, or pinned=true may be AND-combined. Use this before recall when the exact ID is not visible. This is substring search, not semantic search.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional narrow text or structured filter. Supported fields: id, kind, tool, role, path, pattern, file_glob, command, pinned"
                    }
                },
                "additionalProperties": false
            }),
        }
    }

    fn execute(&self, arguments: Value, _timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: ListArguments =
                serde_json::from_value(arguments).context("invalid list_pointers arguments")?;
            Ok(ToolOutcome::output_only(
                self.memory
                    .list_pointers(arguments.filter.as_deref())
                    .await?,
            ))
        })
    }
}

fn pointer_definition(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_owned(),
        description: description.to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "An exact existing addressable memory ID"
                }
            },
            "required": ["id"],
            "additionalProperties": false
        }),
    }
}
