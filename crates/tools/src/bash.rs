use async_trait::async_trait;
use graphirm_graph::edges::EdgeType;
use graphirm_graph::nodes::{ContentData, GraphNode, NodeType};
use serde_json::json;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};

use crate::{Tool, ToolContext, ToolError, ToolOutput};

pub struct BashTool;

impl BashTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command. Captures stdout and stderr. Non-zero exit codes are reported as errors."
    }

    fn is_destructive(&self) -> bool {
        true
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "working_directory": {
                    "type": "string",
                    "description": "Working directory for the command (optional)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        if ctx.disable_bash {
            return Err(ToolError::ExecutionFailed(
                "bash is disabled on this server".into(),
            ));
        }

        let command = args["command"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("missing 'command' field".into()))?;

        let timeout_secs = args["timeout"].as_u64().unwrap_or(120);

        let working_dir = if let Some(wd) = args["working_directory"].as_str() {
            std::path::PathBuf::from(wd)
        } else {
            ctx.working_dir.clone()
        };

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(command)
            .current_dir(&working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // Safety net: if this future is dropped without reaching `kill_child`
            // (e.g. the whole agent task is aborted), tokio SIGKILLs the shell.
            .kill_on_drop(true);
        // Run the shell as leader of its own process group so `kill_child` can
        // take down the shell *and* everything it launched (`bash -c 'sleep 30'`
        // forks `sleep`; killing only the shell would orphan it).
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to spawn bash: {e}")))?;
        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();

        let signal = ctx.signal.clone();

        // Hold the child directly (not in a spawned task) so that on timeout or
        // cancel we can kill the OS process instead of merely dropping a future.
        let outcome = tokio::select! {
            result = wait_with_output(&mut child, stdout_pipe, stderr_pipe) => Ok(result),
            _ = tokio::time::sleep(Duration::from_secs(timeout_secs)) => {
                Err(ToolError::Timeout(timeout_secs))
            }
            _ = signal.cancelled() => Err(ToolError::Cancelled),
        };
        let output = match outcome {
            Ok(result) => {
                result.map_err(|e| ToolError::ExecutionFailed(format!("command error: {e}")))?
            }
            Err(err) => {
                kill_child(&mut child).await;
                return Err(err);
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        let combined = if stderr.is_empty() {
            stdout.clone()
        } else if stdout.is_empty() {
            format!("stderr:\n{stderr}")
        } else {
            format!("{stdout}\nstderr:\n{stderr}")
        };

        let node = GraphNode::new(NodeType::Content(ContentData {
            content_type: "command_output".to_string(),
            path: None,
            body: combined.clone(),
            language: None,
        }));
        let content_node = ctx.record_content_node(node, EdgeType::Produces).await?;

        let is_error = !output.status.success();

        let output_text = if is_error {
            format!("Exit {exit_code}\n{combined}")
        } else {
            combined
        };

        if is_error {
            Ok(ToolOutput {
                content: output_text,
                is_error: true,
                node_id: Some(content_node),
            })
        } else {
            Ok(ToolOutput::success_with_node(output_text, content_node))
        }
    }
}

/// Drain a piped stream to completion; `None` (stream not captured) yields empty output.
async fn read_all<R: AsyncRead + Unpin>(stream: Option<R>) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    if let Some(mut stream) = stream {
        stream.read_to_end(&mut buf).await?;
    }
    Ok(buf)
}

/// Equivalent of `Child::wait_with_output`, but borrows the child instead of consuming
/// it so the caller can still kill the process if this future is abandoned.
async fn wait_with_output(
    child: &mut Child,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
) -> std::io::Result<std::process::Output> {
    let (status, stdout, stderr) =
        tokio::try_join!(child.wait(), read_all(stdout), read_all(stderr))?;
    Ok(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

/// Kill the shell and, on unix, its whole process group, then reap it.
///
/// The child was spawned with `process_group(0)`, so its pgid equals its pid and
/// `kill -9 -- -<pid>` reaches every descendant the shell forked. We shell out to
/// `kill(1)` rather than calling `kill(2)` to avoid a `libc` dependency here.
/// `start_kill` is the portable fallback and is harmless if the group kill already
/// landed. Errors are ignored: the process may already have exited.
async fn kill_child(child: &mut Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let _ = Command::new("kill")
            .arg("-9")
            .arg("--")
            .arg(format!("-{pid}"))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;
    }
    let _ = child.start_kill();
    // Reap so the shell does not linger as a zombie. SIGKILL cannot be blocked, so
    // this returns promptly; the timeout is only a guard against a pathological
    // uninterruptible-sleep state.
    let _ = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use serde_json::json;
    use tempfile::TempDir;

    fn make_ctx_with_dir(dir: &TempDir) -> ToolContext {
        let mut ctx = make_test_context();
        ctx.working_dir = dir.path().to_path_buf();
        ctx
    }

    #[tokio::test]
    async fn bash_disabled_returns_execution_failed() {
        let tool = BashTool::new();
        let mut ctx = make_test_context();
        ctx.disable_bash = true;
        let err = tool
            .execute(json!({"command": "echo hi"}), &ctx)
            .await
            .unwrap_err();
        assert!(
            matches!(err, ToolError::ExecutionFailed(ref s) if s == "bash is disabled on this server")
        );
    }

    #[tokio::test]
    async fn bash_echo() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "echo hello"}), &ctx)
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello"));
    }

    #[tokio::test]
    async fn bash_captures_stderr() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "echo error >&2"}), &ctx)
            .await
            .unwrap();
        assert!(out.content.contains("error"));
    }

    #[tokio::test]
    async fn bash_exit_code_nonzero() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "exit 1"}), &ctx)
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(out.content.contains("Exit 1"));
    }

    #[tokio::test]
    async fn bash_working_directory() {
        let dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let ctx = make_ctx_with_dir(&dir);
        let out = tool
            .execute(
                json!({"command": "pwd", "working_directory": dir.path().to_str().unwrap()}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.trim().contains(dir.path().to_str().unwrap()));
    }

    #[tokio::test]
    async fn bash_missing_command() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let result = tool.execute(json!({}), &ctx).await;
        assert!(matches!(result, Err(ToolError::InvalidArguments(_))));
    }

    #[tokio::test]
    async fn bash_creates_graph_node() {
        let tool = BashTool::new();
        let ctx = make_test_context();
        let out = tool
            .execute(json!({"command": "echo tracked"}), &ctx)
            .await
            .unwrap();
        let node_id = out.node_id.expect("should create a graph node");
        let node = ctx.graph.get_node(&node_id).unwrap();
        assert_eq!(node.label(), Some("content_1_1_1"));
        assert_eq!(
            node.metadata.get("session_id"),
            Some(&serde_json::json!(ctx.agent_id.to_string()))
        );
    }

    #[tokio::test]
    async fn bash_cancellation() {
        use tokio_util::sync::CancellationToken;

        let mut ctx = make_test_context();
        let token = CancellationToken::new();
        ctx.signal = token.clone();

        let tool = BashTool::new();

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            token.cancel();
        });

        let result = tool.execute(json!({"command": "sleep 10"}), &ctx).await;
        assert!(matches!(result, Err(ToolError::Cancelled)));
    }

    /// `kill -0 <pid>` via std::process (avoids a libc dep in tools). True if alive.
    fn pid_is_alive(pid: i32) -> bool {
        std::process::Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap()
            .success()
    }

    /// Wait up to ~2s for the pid to disappear; returns whether it is still alive afterwards.
    async fn wait_for_exit(pid: i32) -> bool {
        for _ in 0..20 {
            if !pid_is_alive(pid) {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        pid_is_alive(pid)
    }

    /// Read the pid written by `echo $$ > <pidfile>` (the `bash -c` shell pid).
    fn read_pidfile(pidfile: &std::path::Path) -> i32 {
        std::fs::read_to_string(pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    #[tokio::test]
    async fn cancel_kills_the_child_process() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx_with_dir(&dir);
        let signal = ctx.signal.clone();
        let pidfile = dir.path().join("pid");
        let cmd = format!("echo $$ > {} && sleep 30", pidfile.display());
        let tool = BashTool::new();
        let fut = tool.execute(json!({"command": cmd}), &ctx);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            signal.cancel();
        });
        let err = fut.await.unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
        let pid = read_pidfile(&pidfile);
        assert!(
            !wait_for_exit(pid).await,
            "bash child {pid} still alive after cancel"
        );
    }

    /// Process-group kill: the `sleep` forked by the shell must die too, not be orphaned.
    #[cfg(unix)]
    #[tokio::test]
    async fn cancel_kills_the_shells_descendants() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx_with_dir(&dir);
        let signal = ctx.signal.clone();
        let sleep_pidfile = dir.path().join("sleep_pid");
        let cmd = format!("sleep 30 & echo $! > {}; wait", sleep_pidfile.display());
        let tool = BashTool::new();
        let fut = tool.execute(json!({"command": cmd}), &ctx);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            signal.cancel();
        });
        let err = fut.await.unwrap_err();
        assert!(matches!(err, ToolError::Cancelled));
        let sleep_pid = read_pidfile(&sleep_pidfile);
        assert!(
            !wait_for_exit(sleep_pid).await,
            "orphaned `sleep` {sleep_pid} still alive after cancel"
        );
    }

    #[tokio::test]
    async fn timeout_kills_the_child_process() {
        let dir = TempDir::new().unwrap();
        let ctx = make_ctx_with_dir(&dir);
        let pidfile = dir.path().join("pid");
        let cmd = format!("echo $$ > {} && sleep 30", pidfile.display());
        let tool = BashTool::new();
        let err = tool
            .execute(json!({"command": cmd, "timeout": 1}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(1)));
        let pid = read_pidfile(&pidfile);
        assert!(
            !wait_for_exit(pid).await,
            "bash child {pid} still alive after timeout"
        );
    }
}
