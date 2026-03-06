//! Structured request logging for API usage analysis.
//!
//! Captures method, path, status, duration, and session context for every
//! HTTP request. Logged as JSONL (one JSON object per line) for easy
//! post-hoc analysis of endpoint frequency, call sequences, and timing.

use std::io::Write;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// A single logged HTTP request with timing and context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLogEntry {
    /// UTC timestamp when the request was received.
    pub timestamp: DateTime<Utc>,
    /// HTTP method (GET, POST, DELETE, etc.).
    pub method: String,
    /// Request path (e.g. `/api/sessions/abc-123/prompt`).
    pub path: String,
    /// HTTP response status code.
    pub status: u16,
    /// Request-to-response duration in milliseconds.
    pub duration_ms: f64,
    /// Session ID extracted from the path, if present.
    pub session_id: Option<String>,
    /// Endpoint group for aggregation: "health", "sessions", "graph", "events".
    pub endpoint_group: String,
}

/// Non-blocking request logger that writes JSONL entries via a background blocking task.
///
/// Entries are sent through an unbounded mpsc channel to a `spawn_blocking` task that
/// appends them to a file using synchronous I/O (safe on a dedicated blocking thread).
/// Dropping all `RequestLogger` clones closes the channel; the background task then
/// drains remaining entries and exits.
///
/// To ensure all buffered entries are flushed on shutdown, `await` the `JoinHandle`
/// returned from [`RequestLogger::new`] after dropping all logger instances.
#[derive(Clone)]
pub struct RequestLogger {
    tx: mpsc::UnboundedSender<RequestLogEntry>,
}

impl RequestLogger {
    /// Create a new logger that writes JSONL entries to `path`.
    ///
    /// Returns the logger and a `JoinHandle` for the background writer task. The handle
    /// should be awaited (with an optional timeout) during graceful shutdown to ensure all
    /// buffered entries reach disk before the process exits.
    pub fn new(path: PathBuf) -> (Self, tokio::task::JoinHandle<()>) {
        let (tx, mut rx) = mpsc::unbounded_channel::<RequestLogEntry>();

        let handle = tokio::task::spawn_blocking(move || {
            let mut file = match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(f) => f,
                Err(e) => {
                    tracing::error!("Failed to open request log file {}: {e}", path.display());
                    return;
                }
            };

            while let Some(entry) = rx.blocking_recv() {
                if let Ok(json) = serde_json::to_string(&entry) {
                    if let Err(e) = writeln!(file, "{json}") {
                        tracing::warn!("request log write failed: {e}");
                    }
                }
            }
        });

        (Self { tx }, handle)
    }

    /// Log a request entry (non-blocking, fire-and-forget).
    pub fn log(&self, entry: RequestLogEntry) {
        let _ = self.tx.send(entry);
    }
}

/// Extract a session ID from a request path, if present.
///
/// Recognizes these patterns:
/// - `/api/sessions/{id}[/...]`
/// - `/api/graph/{id}[/...]`
/// - `/api/events/{id}`
pub fn extract_session_id(path: &str) -> Option<String> {
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();

    if segments.len() < 3 || segments[0] != "api" {
        return None;
    }

    match segments[1] {
        "sessions" | "graph" | "events" if segments.len() >= 3 && !segments[2].is_empty() => {
            Some(segments[2].to_string())
        }
        _ => None,
    }
}

/// Classify a request path into an endpoint group for aggregation.
pub fn classify_endpoint(path: &str) -> &'static str {
    if path.starts_with("/api/health") {
        "health"
    } else if path.starts_with("/api/sessions") {
        "sessions"
    } else if path.starts_with("/api/graph") {
        "graph"
    } else if path.starts_with("/api/events") {
        "events"
    } else {
        "other"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_log_entry_serializes_to_json() {
        let entry = RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "GET".to_string(),
            path: "/api/sessions".to_string(),
            status: 200,
            duration_ms: 1.23,
            session_id: Some("abc-123".to_string()),
            endpoint_group: "sessions".to_string(),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"method\":\"GET\""));
        assert!(json.contains("\"status\":200"));
        assert!(json.contains("\"session_id\":\"abc-123\""));
        assert!(json.contains("\"endpoint_group\":\"sessions\""));
    }

    #[test]
    fn request_log_entry_without_session_id() {
        let entry = RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "GET".to_string(),
            path: "/api/health".to_string(),
            status: 200,
            duration_ms: 0.5,
            session_id: None,
            endpoint_group: "health".to_string(),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"session_id\":null"));
    }

    #[tokio::test]
    async fn request_logger_writes_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let log_path = dir.path().join("test_request.jsonl");

        let (logger, _handle) = RequestLogger::new(log_path.clone());

        logger.log(RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "POST".to_string(),
            path: "/api/sessions".to_string(),
            status: 201,
            duration_ms: 5.0,
            session_id: None,
            endpoint_group: "sessions".to_string(),
        });

        logger.log(RequestLogEntry {
            timestamp: chrono::Utc::now(),
            method: "GET".to_string(),
            path: "/api/sessions/abc/messages".to_string(),
            status: 200,
            duration_ms: 2.1,
            session_id: Some("abc".to_string()),
            endpoint_group: "sessions".to_string(),
        });

        // Flush by dropping the logger (closes the channel)
        drop(logger);
        // Give the background task time to write
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let contents = std::fs::read_to_string(&log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "Expected 2 log lines, got {}", lines.len());

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["method"], "POST");
        assert_eq!(first["status"], 201);

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["method"], "GET");
        assert_eq!(second["session_id"], "abc");
    }

    #[test]
    fn extract_session_id_from_session_path() {
        assert_eq!(
            extract_session_id("/api/sessions/abc-123"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/sessions/abc-123/prompt"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/sessions/abc-123/messages"),
            Some("abc-123".to_string())
        );
    }

    #[test]
    fn extract_session_id_from_graph_path() {
        assert_eq!(
            extract_session_id("/api/graph/abc-123"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/graph/abc-123/node/xyz"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_session_id("/api/graph/abc-123/subgraph/xyz"),
            Some("abc-123".to_string())
        );
    }

    #[test]
    fn extract_session_id_from_events_path() {
        assert_eq!(
            extract_session_id("/api/events/abc-123"),
            Some("abc-123".to_string())
        );
    }

    #[test]
    fn extract_session_id_returns_none_for_paths_without_id() {
        assert_eq!(extract_session_id("/api/health"), None);
        assert_eq!(extract_session_id("/api/sessions"), None);
        assert_eq!(extract_session_id("/api/events"), None);
    }

    #[test]
    fn classify_endpoint_group_for_known_paths() {
        assert_eq!(classify_endpoint("/api/health"), "health");
        assert_eq!(classify_endpoint("/api/sessions"), "sessions");
        assert_eq!(classify_endpoint("/api/sessions/abc/prompt"), "sessions");
        assert_eq!(classify_endpoint("/api/graph/abc"), "graph");
        assert_eq!(classify_endpoint("/api/graph/abc/node/xyz"), "graph");
        assert_eq!(classify_endpoint("/api/events"), "events");
        assert_eq!(classify_endpoint("/api/events/abc"), "events");
        assert_eq!(classify_endpoint("/unknown"), "other");
    }
}
