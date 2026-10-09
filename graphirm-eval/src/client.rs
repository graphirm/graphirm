//! Thin async HTTP client for the Graphirm REST API.
//!
//! A 429 is retried. The wait is the larger of one second, the server's
//! `Retry-After`, and an exponential backoff. When the retries run out the
//! error is [`InfrastructureError`], which the harness records apart from a
//! task failure. Waiting for a turn uses the session SSE stream rather than
//! polling `GET /api/sessions/{id}`.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How many 429 responses one call will absorb before giving up.
const RATE_LIMIT_ATTEMPTS: u32 = 5;

/// The server was unreachable in a way that says nothing about the task.
/// Retries are already exhausted when this is returned.
#[derive(Debug)]
pub struct InfrastructureError {
    pub message: String,
}

impl std::fmt::Display for InfrastructureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for InfrastructureError {}

/// Delay before the next attempt.
///
/// `attempt` is how many 429s have already been seen, starting at 0.
/// A `Retry-After` of 0 (or a missing header) still waits at least one second,
/// and a later attempt waits at least `2^attempt` seconds.
pub(crate) fn retry_delay(retry_after: Option<&str>, attempt: u32) -> std::time::Duration {
    let header_secs = retry_after
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let backoff = 1u64 << attempt.min(4);
    std::time::Duration::from_secs(header_secs.max(1).max(backoff))
}

/// Pull complete SSE blocks out of `buffer`, leaving a partial tail in place.
/// Comment keepalives (`: heartbeat`) are skipped. Returned names are the
/// `event:` field of each block.
pub(crate) fn drain_sse_events(buffer: &mut String) -> Vec<String> {
    let mut names = Vec::new();
    while let Some(split_at) = buffer.find("\n\n") {
        let block: String = buffer.drain(..split_at + 2).collect();
        if let Some(name) = sse_event_name(&block) {
            names.push(name);
        }
    }
    names
}

fn sse_event_name(block: &str) -> Option<String> {
    let mut name = None;
    for line in block.split('\n') {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            name = Some(rest.trim().to_string());
        }
    }
    name
}

/// `GET /api/health` fields the harness reads before scoring.
#[derive(Debug, Clone, Deserialize)]
pub struct HealthReport {
    pub status: String,
    #[serde(default)]
    pub extraction: String,
    #[serde(default)]
    pub memory: String,
    /// `off`, `ready`, or `unconfigured`. Missing on older servers.
    #[serde(default)]
    pub compaction: String,
}

/// Compaction may be off. It may not be enabled with no model.
pub fn compaction_is_ready(status: &str) -> bool {
    status != "unconfigured"
}

/// True when the server selected a backend it can run.
pub fn extraction_is_ready(extraction: &str) -> bool {
    matches!(extraction, "llm" | "local" | "hybrid")
}

/// Statuses that mean the agent loop is no longer running.
pub(crate) fn is_terminal_session_status(status: &str) -> bool {
    matches!(
        status,
        "idle" | "completed" | "failed" | "cancelled" | "token_cap_exceeded"
    )
}

#[derive(Debug, Clone)]
pub struct GraphirmClient {
    base: String,
    http: reqwest::Client,
    /// When set, adds `Authorization: Bearer` to non-health API calls.
    api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SessionResponse {
    pub id: String,
    pub status: String,
    #[serde(default)]
    pub workspace_path: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    pub node_type: Value,
}

#[derive(Debug, Deserialize)]
pub struct GraphResponse {
    pub nodes: Vec<Value>,
    pub edges: Vec<Value>,
}

impl GraphirmClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base: base_url.into(),
            http: reqwest::Client::new(),
            api_key: None,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    fn authorized(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(k) => req.bearer_auth(k),
            None => req,
        }
    }

    async fn send(
        &self,
        mut build: impl FnMut() -> reqwest::RequestBuilder,
    ) -> anyhow::Result<reqwest::Response> {
        let mut attempt = 0u32;
        loop {
            let response = self.authorized(build()).send().await?;
            if response.status() != reqwest::StatusCode::TOO_MANY_REQUESTS {
                return Ok(response);
            }
            if attempt + 1 >= RATE_LIMIT_ATTEMPTS {
                return Err(InfrastructureError {
                    message: format!("rate limit exhausted after {RATE_LIMIT_ATTEMPTS} attempts"),
                }
                .into());
            }
            let header = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string);
            let delay = retry_delay(header.as_deref(), attempt);
            attempt += 1;
            tokio::time::sleep(delay).await;
        }
    }

    async fn json<T: serde::de::DeserializeOwned>(
        &self,
        build: impl FnMut() -> reqwest::RequestBuilder,
    ) -> anyhow::Result<T> {
        let response = self.send(build).await?;
        let status = response.status();
        let bytes = response.bytes().await?;
        serde_json::from_slice(&bytes).map_err(|err| {
            let preview: String = String::from_utf8_lossy(&bytes).chars().take(200).collect();
            anyhow::anyhow!("decode http={status} err={err} body={preview}")
        })
    }

    pub async fn health_report(&self) -> anyhow::Result<HealthReport> {
        let http = self.http.clone();
        let url = format!("{}/api/health", self.base);
        self.json(|| http.get(&url)).await
    }

    pub async fn health(&self) -> reqwest::Result<bool> {
        let r = self
            .http
            .get(format!("{}/api/health", self.base))
            .send()
            .await?;
        Ok(r.status().is_success())
    }

    pub async fn create_session(
        &self,
        enable_segments: bool,
        segment_filter: Option<&[String]>,
        workspace: &str,
    ) -> anyhow::Result<SessionResponse> {
        // auto_approve: true bypasses the HITL gate so bash/write/edit run
        // without human confirmation — required for programmatic eval runs.
        let mut body = serde_json::json!({
            "auto_approve": true,
            "enable_segments": enable_segments,
            "workspace": workspace,
        });
        if let Some(filter) = segment_filter {
            body["segment_filter"] = serde_json::json!(filter);
        }
        let http = self.http.clone();
        let url = format!("{}/api/sessions", self.base);
        self.json(|| http.post(&url).json(&body)).await
    }

    pub async fn prompt(&self, session_id: &str, content: &str) -> anyhow::Result<()> {
        #[derive(Serialize)]
        struct Prompt<'a> {
            content: &'a str,
        }
        let http = self.http.clone();
        let url = format!("{}/api/sessions/{session_id}/prompt", self.base);
        let response = self
            .send(|| http.post(&url).json(&Prompt { content }))
            .await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("prompt http={status} body={text}");
        }
        Ok(())
    }

    /// Open the session event stream, send the prompt, then wait until
    /// `agent_end` or the timeout. The stream is open before the prompt so a
    /// fast turn cannot finish before we subscribe.
    pub async fn prompt_and_wait(
        &self,
        session_id: &str,
        content: &str,
        timeout_secs: u64,
    ) -> anyhow::Result<String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        let response = self.open_events(session_id).await?;
        self.prompt(session_id, content).await?;
        self.read_until_settled(session_id, response, deadline)
            .await
    }

    async fn open_events(&self, session_id: &str) -> anyhow::Result<reqwest::Response> {
        let http = self.http.clone();
        let url = format!("{}/api/events/{session_id}", self.base);
        let response = self.send(|| http.get(&url)).await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("sse connect http={status} body={text}");
        }
        Ok(response)
    }

    async fn read_until_settled(
        &self,
        session_id: &str,
        mut response: reqwest::Response,
        deadline: std::time::Instant,
    ) -> anyhow::Result<String> {
        let mut buffer = String::new();
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok("timeout".to_string());
            }
            let chunk = tokio::time::timeout(remaining, response.chunk()).await;
            let dropped = match chunk {
                Err(_) => return Ok("timeout".to_string()),
                Ok(Err(_)) => true,
                Ok(Ok(None)) => true,
                Ok(Ok(Some(bytes))) => {
                    buffer.push_str(&String::from_utf8_lossy(&bytes));
                    let names = drain_sse_events(&mut buffer);
                    if names.iter().any(|name| name == "agent_end") {
                        // AgentEnd is broadcast before the route handler stores
                        // the terminal status, so the first read can still say running.
                        // A nested loop can also emit agent_end while this session
                        // is still running; keep reading in that case.
                        let status = self.wait_for_terminal_status(session_id).await?;
                        if is_terminal_session_status(&status) {
                            return Ok(status);
                        }
                    }
                    false
                }
            };
            if dropped {
                if let Some(status) = self.settled_status(session_id).await? {
                    return Ok(status);
                }
                if std::time::Instant::now() >= deadline {
                    return Ok("timeout".to_string());
                }
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                response = self.open_events(session_id).await?;
                buffer.clear();
            }
        }
    }

    async fn session_status(&self, session_id: &str) -> anyhow::Result<String> {
        let session = self.get_session(session_id).await?;
        Ok(session.status)
    }

    async fn wait_for_terminal_status(&self, session_id: &str) -> anyhow::Result<String> {
        for _ in 0..20 {
            let status = self.session_status(session_id).await?;
            if is_terminal_session_status(&status) {
                return Ok(status);
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        self.session_status(session_id).await
    }

    async fn settled_status(&self, session_id: &str) -> anyhow::Result<Option<String>> {
        let status = self.session_status(session_id).await?;
        if is_terminal_session_status(&status) {
            Ok(Some(status))
        } else {
            Ok(None)
        }
    }

    async fn get_session(&self, session_id: &str) -> anyhow::Result<SessionResponse> {
        let http = self.http.clone();
        let url = format!("{}/api/sessions/{session_id}", self.base);
        self.json(|| http.get(&url)).await
    }

    pub async fn get_messages(&self, session_id: &str) -> anyhow::Result<Vec<Value>> {
        let http = self.http.clone();
        let url = format!("{}/api/sessions/{session_id}/messages", self.base);
        self.json(|| http.get(&url)).await
    }

    pub async fn get_knowledge(&self, session_id: &str) -> anyhow::Result<Vec<Value>> {
        let http = self.http.clone();
        let url = format!("{}/api/graph/{session_id}/knowledge", self.base);
        self.json(|| http.get(&url)).await
    }

    pub async fn get_graph(&self, session_id: &str) -> anyhow::Result<GraphResponse> {
        let http = self.http.clone();
        let url = format!("{}/api/graph/{session_id}", self.base);
        self.json(|| http.get(&url)).await
    }

    /// Cancel and remove a session. Fires the CancellationToken, stopping any in-flight agent loop.
    pub async fn delete_session(&self, session_id: &str) -> anyhow::Result<()> {
        let http = self.http.clone();
        let url = format!("{}/api/sessions/{session_id}", self.base);
        let _ = self.send(|| http.delete(&url)).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_preflight_rejects_a_disabled_backend() {
        assert!(extraction_is_ready("llm"));
        assert!(extraction_is_ready("local"));
        assert!(!extraction_is_ready("disabled"));
        assert!(!extraction_is_ready(""));
    }

    #[test]
    fn compaction_preflight_rejects_an_unconfigured_model() {
        assert!(compaction_is_ready("ready"));
        assert!(compaction_is_ready("off"));
        assert!(compaction_is_ready(""));
        assert!(!compaction_is_ready("unconfigured"));
    }

    #[test]
    fn retry_delay_floors_a_zero_retry_after_and_honors_a_longer_one() {
        assert_eq!(retry_delay(Some("0"), 0), std::time::Duration::from_secs(1));
        assert_eq!(retry_delay(None, 0), std::time::Duration::from_secs(1));
        assert_eq!(retry_delay(Some("8"), 0), std::time::Duration::from_secs(8));
        assert_eq!(retry_delay(Some("1"), 3), std::time::Duration::from_secs(8));
    }

    #[test]
    fn drain_sse_events_reads_agent_end_and_keeps_a_partial_block() {
        let mut buf = "event: heartbeat\ndata: ping\n\nevent: agent_end\ndata: {}\n\nevent: turn_"
            .to_string();
        let names = drain_sse_events(&mut buf);
        assert_eq!(
            names,
            vec!["heartbeat".to_string(), "agent_end".to_string()]
        );
        assert_eq!(buf, "event: turn_");
    }

    #[test]
    fn drain_sse_events_ignores_comment_heartbeats() {
        let mut buf = ": heartbeat\n\nevent: agent_end\ndata: {}\n\n".to_string();
        let names = drain_sse_events(&mut buf);
        assert_eq!(names, vec!["agent_end".to_string()]);
        assert!(buf.is_empty());
    }

    #[test]
    fn terminal_statuses_include_cap_and_cancel() {
        assert!(is_terminal_session_status("completed"));
        assert!(is_terminal_session_status("token_cap_exceeded"));
        assert!(is_terminal_session_status("cancelled"));
        assert!(!is_terminal_session_status("running"));
    }
}
