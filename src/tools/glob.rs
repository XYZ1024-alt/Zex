use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use globset::{GlobBuilder, GlobMatcher};
use ignore::WalkBuilder;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, resolve_path},
};

const DEFAULT_MAX_RESULTS: usize = 200;
const MAX_RESULT_LIMIT: usize = 10_000;

pub struct GlobTool {
    working_dir: PathBuf,
}

impl GlobTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for GlobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob".to_owned(),
            description: "Find files and directories by path glob using a pure Rust walker. Use this instead of bash for locating files. Git ignore files are respected when possible.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Path glob such as **/*.rs or Cargo.*; a pattern without a slash matches at any depth"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search; defaults to the working directory"
                    },
                    "hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories; defaults to false"
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RESULT_LIMIT,
                        "description": "Maximum paths to return; defaults to 200"
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        }
    }

    fn execute(&self, arguments: Value, timeout: Duration) -> ToolFuture<'_> {
        let working_dir = self.working_dir.clone();
        Box::pin(async move {
            let arguments: GlobArguments =
                serde_json::from_value(arguments).context("invalid glob arguments")?;
            validate_limit(arguments.max_results)?;
            let root = resolve_path(
                &working_dir,
                arguments.path.as_deref().unwrap_or_else(|| Path::new(".")),
            );
            let output = tokio::task::spawn_blocking(move || search(root, arguments, timeout))
                .await
                .context("glob worker task failed")??;
            Ok(output)
        })
    }
}

#[derive(Deserialize)]
struct GlobArguments {
    pattern: String,
    path: Option<PathBuf>,
    #[serde(default)]
    hidden: bool,
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn search(root: PathBuf, arguments: GlobArguments, timeout: Duration) -> Result<String> {
    if !root.is_dir() {
        bail!("glob path is not a directory: {}", root.display());
    }
    let matcher = build_matcher(&arguments.pattern)?;
    let deadline = Instant::now() + timeout;
    let mut matches = Vec::new();
    let mut limited = false;
    let mut builder = WalkBuilder::new(&root);
    builder
        .hidden(!arguments.hidden)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true);

    for entry in builder.build() {
        if Instant::now() >= deadline {
            bail!("glob exceeded its {} second timeout", timeout.as_secs());
        }
        let entry = entry.with_context(|| format!("failed to walk {}", root.display()))?;
        if entry.path() == root {
            continue;
        }
        let relative = display_path(&root, entry.path());
        if matcher.is_match(&relative) {
            let suffix = entry.file_type().is_some_and(|kind| kind.is_dir());
            matches.push(format!(
                "{}{}",
                relative.display(),
                if suffix { "/" } else { "" }
            ));
            if matches.len() == arguments.max_results {
                limited = true;
                break;
            }
        }
    }

    matches.sort();
    if matches.is_empty() {
        return Ok("No matching paths found.".to_owned());
    }
    let mut output = matches.join("\n");
    output.push_str(&format!("\n\n{} matching path(s)", matches.len()));
    if limited {
        output.push_str(&format!(
            "; stopped at max_results={}",
            arguments.max_results
        ));
    }
    Ok(output)
}

fn build_matcher(pattern: &str) -> Result<GlobMatcher> {
    let normalized = pattern.replace('\\', "/");
    let pattern = if normalized.contains('/') {
        normalized
    } else {
        format!("{{{normalized},**/{normalized}}}")
    };
    GlobBuilder::new(&pattern)
        .literal_separator(true)
        .build()
        .with_context(|| format!("invalid glob pattern {pattern:?}"))
        .map(|glob| glob.compile_matcher())
}

fn display_path(root: &Path, path: &Path) -> PathBuf {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .collect()
}

fn validate_limit(limit: usize) -> Result<()> {
    if !(1..=MAX_RESULT_LIMIT).contains(&limit) {
        bail!("max_results must be between 1 and {MAX_RESULT_LIMIT}");
    }
    Ok(())
}

const fn default_max_results() -> usize {
    DEFAULT_MAX_RESULTS
}
