use std::{path::PathBuf, process::Stdio, time::Duration};

use anyhow::{Context, bail};
use process_wrap::tokio::{ChildWrapper, CommandWrap, KillOnDrop};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{io::AsyncReadExt, process::Command};

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
                .stderr(Stdio::piped());

            let mut command = CommandWrap::from(command);
            command.wrap(KillOnDrop);
            #[cfg(unix)]
            command.wrap(process_wrap::tokio::ProcessGroup::leader());
            #[cfg(windows)]
            command.wrap(process_wrap::tokio::JobObject);

            let mut process = RunningShell {
                child: command.spawn().context("failed to execute shell command")?,
                completed: false,
            };
            let stdout = process
                .child
                .stdout()
                .take()
                .context("failed to capture stdout")?;
            let stderr = process
                .child
                .stderr()
                .take()
                .context("failed to capture stderr")?;
            let execution = async {
                tokio::try_join!(
                    process.child.wait(),
                    read_output(stdout),
                    read_output(stderr)
                )
            };
            let (status, stdout, stderr) = match tokio::time::timeout(timeout, execution).await {
                Ok(result) => {
                    result.context("failed to wait for shell command or read its output")?
                }
                Err(_) => {
                    process
                        .child
                        .start_kill()
                        .context("shell timed out; failed to terminate its process tree")?;
                    tokio::time::timeout(Duration::from_secs(5), process.child.wait())
                        .await
                        .context("shell timed out; process tree cleanup timed out")?
                        .context("shell timed out; failed to reap its process tree")?;
                    process.completed = true;
                    bail!(
                        "shell command exceeded its {} second timeout",
                        timeout.as_secs()
                    );
                }
            };
            process.completed = true;

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

struct RunningShell {
    child: Box<dyn ChildWrapper>,
    completed: bool,
}

impl Drop for RunningShell {
    fn drop(&mut self) {
        if !self.completed
            && let Err(error) = self.child.start_kill()
        {
            // On Unix the entire group may already have exited before cancellation.
            #[cfg(unix)]
            if error.raw_os_error() == Some(3) {
                return;
            }
            eprintln!("failed to terminate interrupted shell process tree: {error}");
        }
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
    use crate::tools::Tool;
    use std::{path::PathBuf, time::Duration};

    #[test]
    fn process_tree_fixture() {
        let Some(directory) = std::env::var_os("ZEX_PROCESS_FIXTURE") else {
            return;
        };
        let directory = PathBuf::from(directory);
        if std::env::var_os("ZEX_PROCESS_DESCENDANT").is_some() {
            std::fs::write(directory.join("ready"), std::process::id().to_string()).unwrap();
            std::thread::sleep(Duration::from_secs(15));
            return;
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tools::bash::tests::process_tree_fixture",
                "--nocapture",
            ])
            .env("ZEX_PROCESS_DESCENDANT", "1")
            .spawn()
            .unwrap();
        if std::env::var_os("ZEX_PROCESS_PARENT_EXITS").is_none() {
            child.wait().unwrap();
        }
    }

    #[cfg(windows)]
    fn process_running(pid: u32) -> bool {
        use windows_sys::Win32::{
            Foundation::{CloseHandle, WAIT_TIMEOUT},
            System::Threading::{OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject},
        };
        unsafe {
            let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
            if handle == 0 {
                return false;
            }
            let status = WaitForSingleObject(handle, 0);
            CloseHandle(handle);
            status == WAIT_TIMEOUT
        }
    }

    #[cfg(unix)]
    fn process_running(pid: u32) -> bool {
        let output = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .unwrap();
        let state = String::from_utf8(output.stdout).unwrap();
        !state.trim().is_empty() && !state.trim().starts_with('Z')
    }

    async fn check_process_tree_cleanup(cancel: bool, parent_exits: bool) {
        let directory = std::env::temp_dir().join(format!(
            "zex-tree-{}-{cancel}-{parent_exits}",
            std::process::id()
        ));
        tokio::fs::create_dir_all(&directory).await.unwrap();
        let exe = std::env::current_exe().unwrap();
        #[cfg(windows)]
        let command = format!(
            "set \"ZEX_PROCESS_FIXTURE={}\"&&set \"ZEX_PROCESS_PARENT_EXITS={}\"&&\"{}\" --exact tools::bash::tests::process_tree_fixture --nocapture",
            directory.display(),
            if parent_exits { "1" } else { "" },
            exe.display()
        );
        #[cfg(unix)]
        let command = format!(
            "ZEX_PROCESS_FIXTURE='{}' {} '{}' --exact tools::bash::tests::process_tree_fixture --nocapture",
            directory.display(),
            if parent_exits {
                "ZEX_PROCESS_PARENT_EXITS=1"
            } else {
                ""
            },
            exe.display()
        );
        let tool = super::BashTool::new(directory.clone());
        let execution = tool.execute(
            serde_json::json!({"command":command}),
            Duration::from_secs(2),
        );
        let mut execution = Some(execution);
        let started = tokio::time::Instant::now();
        let pid = loop {
            tokio::select! {
                result = execution.as_mut().unwrap() => panic!("shell ended before descendant started: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    if let Ok(text) = tokio::fs::read_to_string(directory.join("ready")).await
                        && let Ok(pid) = text.parse::<u32>() { break pid; }
                    assert!(started.elapsed() < Duration::from_secs(5), "descendant did not start");
                }
            }
        };
        assert!(process_running(pid));
        if cancel {
            drop(execution.take());
        } else {
            let error = execution.take().unwrap().await.unwrap_err();
            assert!(format!("{error:#}").contains("timeout"));
            assert!(started.elapsed() < Duration::from_secs(5));
        }
        for _ in 0..100 {
            if !process_running(pid) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            !process_running(pid),
            "descendant survived shell cancellation/timeout"
        );
        tokio::fs::remove_dir_all(directory).await.unwrap();
    }

    #[tokio::test]
    async fn dropping_execution_kills_descendants() {
        check_process_tree_cleanup(true, false).await;
    }

    #[tokio::test]
    async fn timeout_covers_descendant_output_after_parent_exits() {
        check_process_tree_cleanup(false, true).await;
    }

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
fn shell_command(command: &str) -> Command {
    let mut process = Command::new("cmd");
    process.args(["/D", "/S", "/C"]);
    process.raw_arg(format!("\"{command}\""));
    process
}

#[cfg(not(windows))]
fn shell_command(command: &str) -> Command {
    let mut process = Command::new("sh");
    process.args(["-c", command]);
    process
}
