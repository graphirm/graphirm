//! Test harness — spawns a Graphirm server, runs tasks, collects results.

use std::path::PathBuf;
use std::time::Instant;

use crate::client::GraphirmClient;
use crate::task::{EvalTask, TaskResult, Verifier};

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
        let db_dir = tempfile::TempDir::new()?;
        let db_path = db_dir.path().join("eval.db");
        let port = 19555u16; // Fixed eval port — don't run alongside the real server

        // Use a fast model for eval — prefer EVAL_MODEL env var, then GRAPHIRM_MODEL,
        // defaulting to DeepSeek Chat. Anthropic is no longer preferred by default
        // because it hits rate limits and has stricter message ordering requirements.
        let eval_model = std::env::var("EVAL_MODEL")
            .or_else(|_| std::env::var("GRAPHIRM_MODEL"))
            .unwrap_or_else(|_| "deepseek/deepseek-chat".to_string());

        let mut cmd = std::process::Command::new(&binary_path);
        cmd.args(["--db", db_path.to_str().expect("tempdir path is not valid UTF-8"), "serve", "--port", &port.to_string()])
            .env("EMBEDDING_BACKEND", "") // disable memory for most tasks
            .env("GRAPHIRM_MODEL", &eval_model);
        // Forward API keys and model config from environment
        for key in &["ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY", "GLINER2_MODEL_DIR"] {
            if let Ok(val) = std::env::var(key) {
                cmd.env(key, val);
            }
        }
        let server = cmd.spawn()?;

        let client = GraphirmClient::new(format!("http://127.0.0.1:{port}"));

        // Wait up to 10s for the server to become healthy
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if Instant::now() > deadline {
                anyhow::bail!("Server did not become healthy within 10s");
            }
            if client.health().await.unwrap_or(false) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        Ok(Self { client, _db_dir: db_dir, _server: server })
    }

    /// Run a single task and return its result.
    pub async fn run_task(&self, task: &EvalTask) -> TaskResult {
        let start = Instant::now();
        let mut result = match self.run_task_inner(task).await {
            Ok(r) => r,
            Err(e) => TaskResult::fail(&task.id, format!("harness error: {e}")),
        };
        result.elapsed_secs = start.elapsed().as_secs_f64();
        result
    }

    async fn run_task_inner(&self, task: &EvalTask) -> anyhow::Result<TaskResult> {
        let session = self.client.create_session(task.enable_segments, task.segment_filter.as_deref()).await?;
        let session_id = session.id.clone();
        let mut last_response = String::new();
        let mut turns_used = 0u32;

        let prompts_result = self.run_task_prompts(task, &session_id, &mut last_response, &mut turns_used).await;

        // Run the verifier BEFORE deleting the session so graph endpoints still work.
        let mut r = match prompts_result {
            Err(e) => TaskResult::fail(&task.id, format!("harness error: {e}")),
            Ok(inner) if !inner.passed => inner,
            Ok(_) => {
                match self.check_verifier(&task.verifier, &session_id, &last_response).await {
                    Ok(true) => TaskResult::pass(&task.id, turns_used, 0.0),
                    Ok(false) => {
                        let mut r = TaskResult::fail(&task.id, "verifier returned false");
                        r.turns_used = turns_used;
                        r
                    }
                    Err(e) => TaskResult::fail(&task.id, format!("harness error: {e}")),
                }
            }
        };

        // Delete session after verifier so agent loop stops and DB connections are released.
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
    ) -> anyhow::Result<TaskResult> {
        for prompt in &task.prompts {
            self.client.prompt(session_id, prompt).await?;
            let status = self.client.wait_for_idle(session_id, task.timeout_secs).await?;

            if status == "timeout" {
                return Ok(TaskResult::fail(&task.id, "session timed out"));
            }
            if status == "failed" {
                return Ok(TaskResult::fail(&task.id, "session failed"));
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
    ) -> anyhow::Result<bool> {
        match verifier {
            Verifier::ResponseContains { substring } => {
                Ok(last_response.to_lowercase().contains(&substring.to_lowercase()))
            }
            Verifier::ResponseContainsAny { substrings } => {
                let lower = last_response.to_lowercase();
                Ok(substrings.iter().any(|s| lower.contains(&s.to_lowercase())))
            }
            Verifier::ResponseNotContains { substring } => {
                Ok(!last_response.to_lowercase().contains(&substring.to_lowercase()))
            }
            Verifier::FileContains { path, substring } => {
                let contents = std::fs::read_to_string(path).unwrap_or_default();
                Ok(contents.contains(substring.as_str()))
            }
            Verifier::CommandSucceeds { command, args } => {
                let status = std::process::Command::new(command).args(args).status()?;
                Ok(status.success())
            }
            Verifier::ResponseContainsCommandOutput { command, args } => {
                let out = std::process::Command::new(command).args(args).output()?;
                let expected = String::from_utf8_lossy(&out.stdout).trim().to_string();
                Ok(last_response.to_lowercase().contains(&expected.to_lowercase()))
            }
            Verifier::KnowledgeNodeCount { min_count } => {
                let nodes = self.client.get_knowledge(session_id).await?;
                Ok(nodes.len() >= *min_count)
            }
            Verifier::GraphContains { min_nodes, type_name } => {
                let graph = self.client.get_graph(session_id).await?;
                if graph.nodes.len() < *min_nodes {
                    return Ok(false);
                }
                Ok(graph.nodes.iter().any(|n| {
                    n["node_type"]["type"].as_str() == Some(type_name.as_str())
                }))
            }
            Verifier::GraphContainsContentType { content_type } => {
                let graph = self.client.get_graph(session_id).await?;
                Ok(graph.nodes.iter().any(|n| {
                    n["node_type"]["content_type"].as_str() == Some(content_type.as_str())
                }))
            }
            Verifier::All(verifiers) => {
                for v in verifiers {
                    if !Box::pin(self.check_verifier(v, session_id, last_response)).await? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        let _ = self._server.kill();
    }
}
