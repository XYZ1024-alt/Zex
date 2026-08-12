use std::{path::PathBuf, process::Stdio, time::Duration};

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
};

use crate::{
    provider::ToolDefinition,
    tools::{Tool, ToolFuture, truncate_output},
};

pub struct BashTool {
    working_dir: PathBuf,
    timeout: Duration,
    max_output_chars: usize,
}

impl BashTool {
    pub fn new(working_dir: PathBuf, timeout: Duration, max_output_chars: usize) -> Self {
        Self {
            working_dir,
            timeout,
            max_output_chars,
        }
    }
}

impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_owned(),
            description: format!(
                "Run a shell command in the working directory. The command times out after {} seconds.",
                self.timeout.as_secs()
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    fn execute(&self, arguments: Value) -> ToolFuture<'_> {
        Box::pin(async move {
            let arguments: BashArguments =
                serde_json::from_value(arguments).context("invalid bash arguments")?;
            let mut command = shell_command(&arguments.command);
            command
                .current_dir(&self.working_dir)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            #[cfg(unix)]
            command.process_group(0);

            let mut child = command.spawn().context("failed to execute shell command")?;
            let process_id = child.id();
            let stdout = child.stdout.take().context("failed to capture stdout")?;
            let stderr = child.stderr.take().context("failed to capture stderr")?;
            let stdout_reader = tokio::spawn(read_output(stdout));
            let stderr_reader = tokio::spawn(read_output(stderr));

            let status = match tokio::time::timeout(self.timeout, child.wait()).await {
                Ok(status) => status.context("failed to wait for shell command")?,
                Err(_) => {
                    terminate_process_tree(&mut child, process_id).await;
                    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
                    bail!(
                        "shell command exceeded its {} second timeout",
                        self.timeout.as_secs()
                    );
                }
            };
            let stdout = stdout_reader
                .await
                .context("stdout reader task failed")?
                .context("failed to read stdout")?;
            let stderr = stderr_reader
                .await
                .context("stderr reader task failed")?
                .context("failed to read stderr")?;

            let stdout = String::from_utf8_lossy(&stdout);
            let stderr = String::from_utf8_lossy(&stderr);
            let result = format!(
                "exit_code: {}\nstdout:\n{}\nstderr:\n{}",
                status.code().map_or_else(
                    || "terminated by signal".to_owned(),
                    |code| code.to_string()
                ),
                stdout,
                stderr
            );
            Ok(truncate_output(result, self.max_output_chars))
        })
    }
}

#[derive(Deserialize)]
struct BashArguments {
    command: String,
}

async fn read_output<R>(mut reader: R) -> std::io::Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut output = Vec::new();
    reader.read_to_end(&mut output).await?;
    Ok(output)
}

#[cfg(windows)]
async fn terminate_process_tree(child: &mut Child, process_id: Option<u32>) {
    if let Some(process_id) = process_id {
        let _ = Command::new("taskkill")
            .args(["/PID", &process_id.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
    let _ = child.kill().await;
}

#[cfg(unix)]
async fn terminate_process_tree(child: &mut Child, process_id: Option<u32>) {
    if let Some(process_id) = process_id {
        let _ = Command::new("kill")
            .args(["-KILL", "--", &format!("-{process_id}")])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
    let _ = child.kill().await;
}

#[cfg(windows)]
fn shell_command(command: &str) -> Command {
    let mut process = Command::new("cmd");
    process.args(["/D", "/S", "/C", command]);
    process
}

#[cfg(not(windows))]
fn shell_command(command: &str) -> Command {
    let mut process = Command::new("sh");
    process.args(["-c", command]);
    process
}
