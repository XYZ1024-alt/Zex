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
    tools::{Tool, ToolFuture, ToolOutcome},
};

pub struct BashTool {
    working_dir: PathBuf,
}

impl BashTool {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

impl Tool for BashTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_owned(),
            description: "Run other system commands in the working directory. Use grep to search file contents and glob to find files instead of shell search commands.".to_owned(),
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

    fn execute(&self, arguments: Value, timeout: Duration) -> ToolFuture<'_> {
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

            let status = match tokio::time::timeout(timeout, child.wait()).await {
                Ok(status) => status.context("failed to wait for shell command")?,
                Err(_) => {
                    terminate_process_tree(&mut child, process_id).await;
                    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
                    bail!(
                        "shell command exceeded its {} second timeout",
                        timeout.as_secs()
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

            let stdout = decode_shell_output(&stdout);
            let stderr = decode_shell_output(&stderr);
            let result = format!(
                "exit_code: {}\nstdout:\n{}\nstderr:\n{}",
                status.code().map_or_else(
                    || "terminated by signal".to_owned(),
                    |code| code.to_string()
                ),
                stdout,
                stderr
            );
            Ok(ToolOutcome::output_only(result))
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

/// Decode child-process output. Fast path: strict UTF-8 (unix shells,
/// `chcp 65001`, cross-platform tools). Anything else falls back to the
/// legacy console code page.
fn decode_shell_output(bytes: &[u8]) -> String {
    if let Ok(text) = std::str::from_utf8(bytes) {
        return text.to_owned();
    }
    decode_legacy_console(bytes)
}

/// Windows consoles write the active ANSI code page (GBK on zh-CN, CP932 on
/// ja-JP, …); transcode through Win32 instead of emitting lossy mojibake.
#[cfg(windows)]
fn decode_legacy_console(bytes: &[u8]) -> String {
    use windows_sys::Win32::Globalization::{CP_ACP, MultiByteToWideChar};
    if bytes.is_empty() {
        return String::new();
    }
    let length = bytes.len().min(i32::MAX as usize) as i32;
    let lossy = || String::from_utf8_lossy(bytes).into_owned();
    unsafe {
        let needed =
            MultiByteToWideChar(CP_ACP, 0, bytes.as_ptr(), length, std::ptr::null_mut(), 0);
        if needed <= 0 {
            return lossy();
        }
        let mut wide = vec![0u16; needed as usize];
        let written =
            MultiByteToWideChar(CP_ACP, 0, bytes.as_ptr(), length, wide.as_mut_ptr(), needed);
        if written <= 0 {
            return lossy();
        }
        String::from_utf16_lossy(&wide[..written as usize])
    }
}

#[cfg(not(windows))]
fn decode_legacy_console(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(test)]
mod tests {
    #[test]
    fn utf8_output_passes_through_unchanged() {
        assert_eq!(
            super::decode_shell_output("héllo 中文\n".as_bytes()),
            "héllo 中文\n"
        );
        assert_eq!(super::decode_shell_output(b"plain ascii"), "plain ascii");
    }

    /// `'pwd' 不是内部或外部命令…` arrives as GBK bytes on zh-CN Windows
    /// (code page 936); it must decode to the original text, not mojibake.
    #[cfg(windows)]
    #[test]
    fn gbk_console_output_decodes_on_chinese_windows() {
        if unsafe { windows_sys::Win32::Globalization::GetACP() } != 936 {
            return; // other locales decode via their own ANSI code page
        }
        let gbk = [
            0x27, 0x70, 0x77, 0x64, 0x27, 0x20, // 'pwd'␠
            0xB2, 0xBB, 0xCA, 0xC7, 0xC4, 0xDA, 0xB2, 0xBF, 0xBB, 0xF2, 0xCD, 0xE2, 0xB2, 0xBF,
            0xC3, 0xFC, 0xC1, 0xEE, // 不是内部或外部命令
        ];
        assert_eq!(super::decode_shell_output(&gbk), "'pwd' 不是内部或外部命令");
    }
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
