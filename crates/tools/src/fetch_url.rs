//! Non-destructive HTTP(S) GET for public documentation and API responses.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Value, json};

use crate::{Tool, ToolContext, ToolError, ToolOutput};

const DEFAULT_MAX_BYTES: usize = 256 * 1024;
const MAX_MAX_BYTES: usize = 2 * 1024 * 1024;

pub struct FetchUrlTool;

impl FetchUrlTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FetchUrlTool {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_http_url(raw: &str) -> Result<reqwest::Url, ToolError> {
    let url = reqwest::Url::parse(raw.trim())
        .map_err(|e| ToolError::InvalidArguments(format!("invalid URL: {e}")))?;
    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(ToolError::InvalidArguments(format!(
                "only http and https URLs are allowed, got scheme '{other}'"
            )));
        }
    }
    if url.host_str().is_none() {
        return Err(ToolError::InvalidArguments(
            "URL must include a host".into(),
        ));
    }
    Ok(url)
}

#[async_trait]
impl Tool for FetchUrlTool {
    fn name(&self) -> &str {
        "fetch_url"
    }

    fn description(&self) -> &str {
        "Fetch a public HTTP or HTTPS URL and return the response status, \
         content-type, and body as UTF-8 text (lossy if needed). Read-only; \
         does not write the graph. Use for documentation, release notes, or \
         small API JSON responses. Large bodies are truncated."
    }

    fn parameters(&self) -> Value {
        let max_desc = format!(
            "Max response body bytes to return (default {}, cap {})",
            DEFAULT_MAX_BYTES, MAX_MAX_BYTES
        );
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to GET"
                },
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1024,
                    "description": max_desc
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let url_str = args["url"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidArguments("'url' is required".into()))?;
        let url = parse_http_url(url_str)?;

        let max_bytes = args
            .get("max_bytes")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_MAX_BYTES)
            .clamp(1024, MAX_MAX_BYTES);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(45))
            .redirect(reqwest::redirect::Policy::limited(8))
            .user_agent(concat!("graphirm-fetch_url/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let req = client.get(url.clone());

        let response = tokio::select! {
            biased;
            _ = ctx.signal.cancelled() => return Err(ToolError::Cancelled),
            r = req.send() => r.map_err(|e| ToolError::ExecutionFailed(e.to_string()))?,
        };

        let status = response.status();
        let ct = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("(unknown)")
            .to_string();

        let body = tokio::select! {
            biased;
            _ = ctx.signal.cancelled() => return Err(ToolError::Cancelled),
            b = response.bytes() => b.map_err(|e| ToolError::ExecutionFailed(e.to_string()))?,
        };

        let truncated = body.len() > max_bytes;
        let slice = if truncated {
            &body[..max_bytes]
        } else {
            body.as_ref()
        };

        let text = String::from_utf8_lossy(slice);
        let mut out = format!(
            "HTTP {}\ncontent-type: {}\nbytes: {}{}\n\n",
            status,
            ct.as_str(),
            body.len(),
            if truncated {
                format!(" (truncated to {max_bytes} bytes)")
            } else {
                String::new()
            }
        );
        out.push_str(&text);

        Ok(ToolOutput::success(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_file_scheme() {
        let e = parse_http_url("file:///etc/passwd").unwrap_err();
        assert!(matches!(e, ToolError::InvalidArguments(_)));
    }

    #[test]
    fn accepts_https() {
        parse_http_url("https://example.com/path?q=1").unwrap();
    }
}
