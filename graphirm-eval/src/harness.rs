//! Test harness — spawns a Graphirm server, runs tasks, collects results.

use std::path::PathBuf;
use std::time::Instant;

use crate::client::{GraphirmClient, InfrastructureError};
use crate::task::{EvalTask, TaskOutcome, TaskResult, Verifier};

/// How many times to run a task whose prompt wait ends before any assistant
/// message. Each try pays the full task timeout, so this stays small.
const EMPTY_TIMEOUT_ATTEMPTS: u32 = 3;

const EMPTY_TIMEOUT_REASON: &str = "prompt timed out before a model response";

/// Assistant replies already stored on the session.
pub fn assistant_message_count(messages: &[serde_json::Value]) -> u32 {
    messages
        .iter()
        .filter(|message| message["node_type"]["role"].as_str() == Some("assistant"))
        .count() as u32
}

/// True when the prompt wait ended and the model had not answered yet.
pub fn is_empty_prompt_timeout(result: &TaskResult) -> bool {
    result.outcome == TaskOutcome::Error
        && result
            .failure_reason
            .as_deref()
            .is_some_and(|reason| reason.contains(EMPTY_TIMEOUT_REASON))
}

/// Must match `GRAPHIRM_API_KEY` on the spawned `graphirm serve` process.
const EVAL_HARNESS_API_KEY: &str = "graphirm-eval-harness-key";

/// This harness always spawns its own server here. It does not send tasks at
/// `app.graphirm.ai` (that limiter is shared with users) or at the dogfood host.
const EVAL_HOST: &str = "127.0.0.1";

/// Burst handed to the spawned process via `GRAPHIRM_RATE_LIMIT_BURST`.
/// The production default stays 60; only this local process is raised.
const EVAL_RATE_LIMIT_BURST: &str = "1000";

/// True for the loopback address this harness binds. A public hostname is refused.
pub fn eval_host_allowed(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

pub struct TestHarness {
    pub client: GraphirmClient,
    /// Temp directory for the SQLite DB — kept alive for the harness lifetime.
    _db_dir: tempfile::TempDir,
    /// Handle to the spawned server process.
    _server: std::process::Child,
}

impl TestHarness {
    /// Start a Graphirm server on an available port and return a ready harness.
    /// `binary_path` is the path to the compiled `graphirm` binary.
    pub async fn start(binary_path: PathBuf) -> anyhow::Result<Self> {
        if !eval_host_allowed(EVAL_HOST) {
            anyhow::bail!("refusing to run evals against {EVAL_HOST}");
        }
        let db_dir = tempfile::TempDir::new()?;
        let db_path = db_dir.path().join("eval.db");
        // Default stays 19555. A second suite sets GRAPHIRM_EVAL_PORT so it
        // does not share that socket with a run already in progress.
        let port = std::env::var("GRAPHIRM_EVAL_PORT")
            .ok()
            .and_then(|raw| raw.parse::<u16>().ok())
            .filter(|port| *port > 0)
            .unwrap_or(19555);
        tracing::info!(host = EVAL_HOST, port, "eval server is local");

        // Use a fast model for eval — prefer EVAL_MODEL env var, then GRAPHIRM_MODEL,
        // defaulting to DeepSeek Chat. Anthropic is no longer preferred by default
        // because it hits rate limits and has stricter message ordering requirements.
        let eval_model = std::env::var("EVAL_MODEL")
            .or_else(|_| std::env::var("GRAPHIRM_MODEL"))
            .unwrap_or_else(|_| "deepseek/deepseek-chat".to_string());

        let mut cmd = std::process::Command::new(&binary_path);
        cmd.args([
            "--db",
            db_path.to_str().expect("tempdir path is not valid UTF-8"),
            "serve",
            "--port",
            &port.to_string(),
        ])
        .env("GRAPHIRM_MODEL", &eval_model)
        .env("GRAPHIRM_API_KEY", EVAL_HARNESS_API_KEY)
        .env("GRAPHIRM_RATE_LIMIT_BURST", EVAL_RATE_LIMIT_BURST);
        // Forward API keys and model config from environment
        for key in &[
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "MISTRAL_API_KEY",
            "OPENROUTER_API_KEY",
            "GLINER2_MODEL_DIR",
            "GRAPHIRM_CONTEXT_MAX_TOKENS",
        ] {
            if let Ok(val) = std::env::var(key) {
                cmd.env(key, val);
            }
        }
        let mut server = cmd.spawn()?;

        let client = GraphirmClient::new(format!("http://{EVAL_HOST}:{port}"))
            .with_api_key(EVAL_HARNESS_API_KEY);

        // Wait up to 10s for the server to become healthy
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if Instant::now() > deadline {
                let _ = server.kill();
                let _ = server.wait();
                anyhow::bail!("Server did not become healthy within 10s");
            }
            if client.health().await.unwrap_or(false) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        let report = client.health_report().await?;
        if !crate::client::extraction_is_ready(&report.extraction)
            || report.memory != "on"
            || !crate::client::compaction_is_ready(&report.compaction)
        {
            let _ = server.kill();
            let _ = server.wait();
            anyhow::bail!(
                "eval preflight failed: extraction={} memory={} compaction={}. Refusing to score a run that skips extraction or embeddings, or compacts with no model.",
                report.extraction,
                report.memory,
                report.compaction
            );
        }
        tracing::info!(
            extraction = %report.extraction,
            memory = %report.memory,
            "eval preflight passed"
        );

        Ok(Self {
            client,
            _db_dir: db_dir,
            _server: server,
        })
    }

    /// Run a single task and return its result.
    pub async fn run_task(&self, task: &EvalTask) -> TaskResult {
        let start = Instant::now();
        let mut result = match self.run_task_inner(task).await {
            Ok(r) => r,
            Err(e) => result_from_harness_err(&task.id, e),
        };
        result.elapsed_secs = start.elapsed().as_secs_f64();
        result
    }

    async fn run_task_inner(&self, task: &EvalTask) -> anyhow::Result<TaskResult> {
        for attempt in 1..=EMPTY_TIMEOUT_ATTEMPTS {
            let result = self.run_task_once(task).await?;
            if is_empty_prompt_timeout(&result) && attempt < EMPTY_TIMEOUT_ATTEMPTS {
                tracing::warn!(
                    task = %task.id,
                    attempt,
                    "retrying a prompt that timed out before a model response"
                );
                continue;
            }
            return Ok(result);
        }
        anyhow::bail!("empty timeout retry loop exited without a result")
    }

    async fn run_task_once(&self, task: &EvalTask) -> anyhow::Result<TaskResult> {
        crate::workspace::clear_shared_eval_files();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        let workspace = crate::workspace::eval_workspace_name(&task.id, nonce);
        let session = self
            .client
            .create_session(
                task.enable_segments,
                task.segment_filter.as_deref(),
                &workspace,
            )
            .await?;
        let session_id = session.id.clone();
        let workspace_path = session.workspace_path.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "session {session_id} has no workspace; refusing to run tools in the repo checkout"
            )
        })?;
        crate::workspace::copy_repo_slice(
            std::path::Path::new("."),
            std::path::Path::new(&workspace_path),
        )?;
        if task.tags.iter().any(|tag| tag == "selection") {
            crate::workspace::write_selection_fixtures(std::path::Path::new(&workspace_path))?;
        }
        crate::workspace::init_workspace_repo(std::path::Path::new(&workspace_path))?;
        let mut last_response = String::new();
        let mut turns_used = 0u32;

        let prompts_result = self
            .run_task_prompts(
                task,
                &session_id,
                &mut last_response,
                &mut turns_used,
                &workspace_path,
            )
            .await;

        // Run the verifier BEFORE deleting the session so graph endpoints still work.
        let mut r = match prompts_result {
            Err(e) => result_from_harness_err(&task.id, e),
            Ok(inner) if !inner.passed => inner,
            Ok(_) => {
                match self
                    .check_verifier(&task.verifier, &session_id, &last_response, &workspace_path)
                    .await
                {
                    Ok(true) => TaskResult::pass(&task.id, turns_used, 0.0),
                    Ok(false) => {
                        let mut r = TaskResult::fail(&task.id, "verifier returned false");
                        r.turns_used = turns_used;
                        r
                    }
                    Err(e) => result_from_harness_err(&task.id, e),
                }
            }
        };

        // Delete session after verifier so agent loop stops and DB connections are released.
        if !r.passed
            && let Ok(messages) = self.client.get_messages(&session_id).await
        {
            let (answer, tools) = crate::task::failure_transcript(&messages);
            if !answer.is_empty() {
                r.final_answer = Some(answer);
            }
            r.tool_trace = tools;
        }
        if !r.passed
            && let Ok(knowledge) = self.client.get_knowledge(&session_id).await
        {
            r.compaction_summary = crate::task::latest_compaction_summary(&knowledge);
        }

        let _ = self.client.delete_session(&session_id).await;

        r.session_id = Some(session_id);
        Ok(r)
    }

    async fn run_task_prompts(
        &self,
        task: &EvalTask,
        session_id: &str,
        last_response: &mut String,
        turns_used: &mut u32,
        workspace_path: &str,
    ) -> anyhow::Result<TaskResult> {
        for prompt in &task.prompts {
            let status = self
                .client
                .prompt_and_wait(session_id, prompt, task.timeout_secs)
                .await?;

            match status.as_str() {
                "timeout" => {
                    let messages = self.client.get_messages(session_id).await?;
                    let assistants = assistant_message_count(&messages);
                    if assistants == 0 {
                        return Ok(TaskResult::error(&task.id, EMPTY_TIMEOUT_REASON));
                    }
                    let last_response = messages
                        .iter()
                        .rev()
                        .find(|message| message["node_type"]["role"].as_str() == Some("assistant"))
                        .and_then(|message| message["node_type"]["content"].as_str())
                        .unwrap_or("");
                    let verifier_passed = self
                        .check_verifier(&task.verifier, session_id, last_response, workspace_path)
                        .await
                        .unwrap_or(false);
                    let reason = crate::task::timeout_failure_reason(assistants, verifier_passed);
                    let mut timed_out = if verifier_passed {
                        TaskResult::ran_over(&task.id, reason)
                    } else {
                        TaskResult::fail(&task.id, reason)
                    };
                    timed_out.turns_used = assistants;
                    return Ok(timed_out);
                }
                "failed" => return Ok(TaskResult::fail(&task.id, "session failed")),
                "cancelled" => return Ok(TaskResult::fail(&task.id, "session cancelled")),
                "token_cap_exceeded" => {
                    return Ok(TaskResult::fail(&task.id, "session token cap exceeded"));
                }
                "idle" | "completed" => {}
                other => {
                    return Ok(TaskResult::fail(
                        &task.id,
                        format!("session ended in {other}"),
                    ));
                }
            }

            // Grab last assistant message.
            // GraphNode serialises as: {"node_type": {"type": "Interaction", "role": "...", "content": "..."}, ...}
            let messages = self.client.get_messages(session_id).await?;
            *last_response = messages
                .iter()
                .rev()
                .find(|m| m["node_type"]["role"].as_str() == Some("assistant"))
                .and_then(|m| m["node_type"]["content"].as_str())
                .unwrap_or("")
                .to_string();

            *turns_used += 1;
        }
        Ok(TaskResult::pass(&task.id, *turns_used, 0.0))
    }

    async fn check_verifier(
        &self,
        verifier: &Verifier,
        session_id: &str,
        last_response: &str,
        workspace_path: &str,
    ) -> anyhow::Result<bool> {
        match verifier {
            Verifier::ResponseContains { substring } => Ok(last_response
                .to_lowercase()
                .contains(&substring.to_lowercase())),
            Verifier::ResponseContainsAny { substrings } => {
                let lower = last_response.to_lowercase();
                Ok(substrings.iter().any(|s| lower.contains(&s.to_lowercase())))
            }
            Verifier::ResponseNotContains { substring } => Ok(!last_response
                .to_lowercase()
                .contains(&substring.to_lowercase())),
            Verifier::FileContains { path, substring } => {
                let file = std::path::Path::new(path);
                let full = if file.is_absolute() {
                    file.to_path_buf()
                } else {
                    std::path::Path::new(workspace_path).join(file)
                };
                let contents = std::fs::read_to_string(full).unwrap_or_default();
                Ok(contents.contains(substring.as_str()))
            }
            Verifier::CommandSucceeds { command, args } => {
                let status = std::process::Command::new(command).args(args).status()?;
                Ok(status.success())
            }
            Verifier::ResponseContainsCommandOutput { command, args } => {
                let out = std::process::Command::new(command).args(args).output()?;
                let expected = String::from_utf8_lossy(&out.stdout).trim().to_string();
                Ok(crate::task::answer_contains_command_output(
                    last_response,
                    &expected,
                ))
            }
            Verifier::KnowledgeNodeCount { min_count } => {
                let nodes = self.client.get_knowledge(session_id).await?;
                Ok(nodes.len() >= *min_count)
            }
            Verifier::GraphContains {
                min_nodes,
                type_name,
            } => {
                let graph = self.client.get_graph(session_id).await?;
                if graph.nodes.len() < *min_nodes {
                    return Ok(false);
                }
                Ok(graph
                    .nodes
                    .iter()
                    .any(|n| n["node_type"]["type"].as_str() == Some(type_name.as_str())))
            }
            Verifier::GraphContainsContentType { content_type } => {
                let graph = self.client.get_graph(session_id).await?;
                Ok(graph.nodes.iter().any(|n| {
                    n["node_type"]["content_type"].as_str() == Some(content_type.as_str())
                }))
            }
            Verifier::All(_) => {
                for v in verifier.decisive_checks() {
                    if !Box::pin(self.check_verifier(v, session_id, last_response, workspace_path))
                        .await?
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }
}

fn result_from_harness_err(task_id: &str, err: anyhow::Error) -> TaskResult {
    if err.downcast_ref::<InfrastructureError>().is_some() {
        TaskResult::error(task_id, format!("infrastructure error: {err}"))
    } else {
        TaskResult::fail(task_id, format!("harness error: {err}"))
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        let _ = self._server.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::{assistant_message_count, eval_host_allowed, is_empty_prompt_timeout};
    use crate::task::TaskResult;

    #[test]
    fn eval_stays_off_the_public_host() {
        assert!(eval_host_allowed("127.0.0.1"));
        assert!(eval_host_allowed("localhost"));
        assert!(eval_host_allowed("::1"));
        assert!(!eval_host_allowed("app.graphirm.ai"));
        assert!(!eval_host_allowed("91.98.94.217"));
    }

    #[test]
    fn a_timeout_with_no_assistant_message_is_infrastructure() {
        let messages = vec![
            serde_json::json!({"node_type": {"role": "user", "content": "count"}}),
            serde_json::json!({"node_type": {"role": "assistant", "content": "12"}}),
            serde_json::json!({"node_type": {"role": "tool", "content": "12"}}),
        ];
        assert_eq!(assistant_message_count(&messages), 1);
        assert!(is_empty_prompt_timeout(&TaskResult::error(
            "cascading-pipeline",
            "prompt timed out before a model response",
        )));
        let mut worked = TaskResult::fail(
            "cascading-pipeline",
            "session timed out after 6 model responses",
        );
        worked.turns_used = 6;
        assert!(!is_empty_prompt_timeout(&worked));
    }
}
