use crate::{
    instructions::read_skill,
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, ToolOutcome},
};
use anyhow::Context;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Duration};

pub struct ReadSkillTool {
    entries: Arc<BTreeMap<String, PathBuf>>,
}
impl ReadSkillTool {
    pub fn new(entries: BTreeMap<String, PathBuf>) -> Self {
        Self {
            entries: Arc::new(entries),
        }
    }
}
impl Tool for ReadSkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_skill".into(),
            description: "Read a discovered SKILL.md by skill name.".into(),
            parameters: json!({"type":"object","properties":{"name":{"type":"string","description":"Skill name"}},"required":["name"],"additionalProperties":false}),
        }
    }
    fn execute(&self, arguments: Value, timeout: Duration) -> ToolFuture<'_> {
        Box::pin(async move {
            let args: Args =
                serde_json::from_value(arguments).context("invalid read_skill arguments")?;
            let path = self
                .entries
                .get(&args.name)
                .with_context(|| format!("unknown skill {:?}", args.name))?
                .clone();
            let content = tokio::time::timeout(timeout, read_skill(&path))
                .await
                .context("read_skill timed out")??;
            Ok(ToolOutcome::output_only(content))
        })
    }
}
#[derive(Deserialize)]
struct Args {
    name: String,
}
