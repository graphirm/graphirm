//! Thin async HTTP client for the Graphirm REST API.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct GraphirmClient {
    base: String,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct SessionResponse {
    pub id: String,
    pub status: String,
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
        }
    }

    pub async fn health(&self) -> reqwest::Result<bool> {
        let r = self.http.get(format!("{}/api/health", self.base)).send().await?;
        Ok(r.status().is_success())
    }

    pub async fn create_session(&self) -> reqwest::Result<SessionResponse> {
        // The endpoint requires Content-Type: application/json even with an empty body.
        self.http
            .post(format!("{}/api/sessions", self.base))
            .json(&serde_json::json!({}))
            .send()
            .await?
            .json()
            .await
    }

    pub async fn prompt(&self, session_id: &str, content: &str) -> reqwest::Result<()> {
        #[derive(Serialize)]
        struct Prompt<'a> { content: &'a str }
        self.http
            .post(format!("{}/api/sessions/{}/prompt", self.base, session_id))
            .json(&Prompt { content })
            .send()
            .await?;
        Ok(())
    }

    /// Poll until session status is "idle" or "completed" or "failed".
    /// Returns the final status string.
    pub async fn wait_for_idle(
        &self,
        session_id: &str,
        timeout_secs: u64,
    ) -> anyhow::Result<String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        loop {
            if std::time::Instant::now() > deadline {
                return Ok("timeout".to_string());
            }
            let resp: SessionResponse = self
                .http
                .get(format!("{}/api/sessions/{}", self.base, session_id))
                .send()
                .await?
                .json()
                .await?;
            match resp.status.as_str() {
                "idle" | "completed" | "failed" => return Ok(resp.status),
                _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
            }
        }
    }

    pub async fn get_messages(&self, session_id: &str) -> reqwest::Result<Vec<Value>> {
        self.http
            .get(format!("{}/api/sessions/{}/messages", self.base, session_id))
            .send()
            .await?
            .json()
            .await
    }

    pub async fn get_knowledge(&self, session_id: &str) -> reqwest::Result<Vec<Value>> {
        self.http
            .get(format!("{}/api/graph/{}/knowledge", self.base, session_id))
            .send()
            .await?
            .json()
            .await
    }

    pub async fn get_graph(&self, session_id: &str) -> reqwest::Result<GraphResponse> {
        self.http
            .get(format!("{}/api/graph/{}", self.base, session_id))
            .send()
            .await?
            .json()
            .await
    }

    /// Cancel and remove a session. Fires the CancellationToken, stopping any in-flight agent loop.
    pub async fn delete_session(&self, session_id: &str) -> reqwest::Result<()> {
        let _ = self.http
            .delete(format!("{}/api/sessions/{}", self.base, session_id))
            .send()
            .await?;
        Ok(())
    }
}
