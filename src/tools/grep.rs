use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, resolve_path},
};

const DEFAULT_MAX_RESULTS: usize = 100;
const MAX_RESULT_LIMIT: usize = 10_000;

pub struct GrepTool {
    working_dir: PathBuf,
}

impl GrepTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep".to_owned(),
            description: "Search UTF-8 file contents recursively with a Rust regex. Use this instead of bash for content search. Git ignore files are respected when possible.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Rust regular expression to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search; defaults to the working directory"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether matching is case-sensitive; defaults to true"
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "Optional glob applied to relative file paths, for example **/*.rs"
                    },
                    "hidden": {
                        "type": "boolean",
                        "description": "Include hidden files and directories; defaults to false"
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RESULT_LIMIT,
                        "description": "Maximum matching lines to return; defaults to 100"
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
            let arguments: GrepArguments =
                serde_json::from_value(arguments).context("invalid grep arguments")?;
            validate_limit(arguments.max_results)?;
            let root = resolve_path(
                &working_dir,
                arguments.path.as_deref().unwrap_or_else(|| Path::new(".")),
            );
            let output = tokio::task::spawn_blocking(move || search(root, arguments, timeout))
                .await
                .context("grep worker task failed")??;
            Ok(output)
        })
    }
}

#[derive(Deserialize)]
struct GrepArguments {
    pattern: String,
    path: Option<PathBuf>,
    #[serde(default = "default_true")]
    case_sensitive: bool,
    file_glob: Option<String>,
    #[serde(default)]
    hidden: bool,
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn search(root: PathBuf, arguments: GrepArguments, timeout: Duration) -> Result<String> {
    let regex = RegexBuilder::new(&arguments.pattern)
        .case_insensitive(!arguments.case_sensitive)
        .build()
        .with_context(|| format!("invalid grep pattern {:?}", arguments.pattern))?;
    let path_matcher = arguments
        .file_glob
        .as_deref()
        .map(build_path_matcher)
        .transpose()?;
    let deadline = Instant::now() + timeout;
    let root_is_file = root.is_file();
    let display_root = if root.is_dir() {
        root.clone()
    } else {
        root.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| root.clone())
    };
    let mut results = Vec::new();
    let mut matched_files = 0usize;
    let mut matching = false;

    let mut builder = WalkBuilder::new(&root);
    builder
        .hidden(!arguments.hidden)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .parents(true);

    for entry in builder.build() {
        if Instant::now() >= deadline {
            bail!("grep exceeded its {} second timeout", timeout.as_secs());
        }
        let entry = entry.with_context(|| format!("failed to walk {}", root.display()))?;
        let Some(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let path = entry.path();
        let relative = if root_is_file {
            path.file_name()
                .map(PathBuf::from)
                .unwrap_or_else(|| path.to_path_buf())
        } else {
            display_path(&display_root, path)
        };
        if path_matcher
            .as_ref()
            .is_some_and(|matcher| !matcher.is_match(&relative))
        {
            continue;
        }
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => continue,
            Err(error) => {
                return Err(error).with_context(|| format!("failed to read {}", path.display()));
            }
        };
        let mut file_matched = false;
        for (index, line) in content.lines().enumerate() {
            if regex.is_match(line) {
                if !file_matched {
                    matched_files += 1;
                    file_matched = true;
                }
                results.push(format!("{}:{}:{}", relative.display(), index + 1, line));
                if results.len() == arguments.max_results {
                    matching = true;
                    break;
                }
            }
        }
        if matching {
            break;
        }
    }

    if results.is_empty() {
        return Ok("No matches found.".to_owned());
    }
    let mut output = results.join("\n");
    output.push_str(&format!(
        "\n\n{} matching line(s) in {} file(s)",
        results.len(),
        matched_files
    ));
    if matching {
        output.push_str(&format!(
            "; stopped at max_results={}",
            arguments.max_results
        ));
    }
    Ok(output)
}

fn build_path_matcher(pattern: &str) -> Result<globset::GlobMatcher> {
    globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .with_context(|| format!("invalid file_glob {pattern:?}"))
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

const fn default_true() -> bool {
    true
}

const fn default_max_results() -> usize {
    DEFAULT_MAX_RESULTS
}
