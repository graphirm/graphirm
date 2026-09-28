//! Typed decision calls to Jev (TypeSafe "System One") over the OpenRouter
//! Decisions API.
//!
//! Jev answers bounded questions about a `state` with calibrated probabilities
//! instead of generating text. Three question types exist:
//!
//! - `choice` — pick one option from `criteria: {option: description}`; the
//!   reply carries `choice`, `probabilities`, `confidence`.
//! - `score` — a value in `[0, 1]` against `criteria: [legend...]`; the reply
//!   carries `score`, `probabilities`, `confidence`.
//! - `noul` — a yes/no; the reply carries a single `noul` field which *is*
//!   P(yes). There is no separate confidence.
//!
//! Every call is fail-soft: transport errors, timeouts, non-2xx replies and
//! malformed bodies all surface as `Err(LlmError)` and never panic. Callers
//! are expected to fall back to a deterministic path on `Err`.
//!
//! The client is provider-agnostic: the request/reply shape is the same
//! whether served by OpenRouter (`/api/alpha/decisions`, bearer key) or a
//! local System One server such as Laya (`/v1/systemone`, typically no key).
//! [`DecisionsConfig`] carries endpoint, optional key, and model string;
//! OpenRouter + [`JEV_MODEL`] are the defaults.
//!
//! Companion study: `~/codeporate-connect/docs/evaluations/2026-09-27-jev-where-in-graphirm.md`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::error::LlmError;

/// OpenRouter's Decisions endpoint — the default provider.
pub const OPENROUTER_DECISIONS_URL: &str = "https://openrouter.ai/api/alpha/decisions";

/// Pinned patch version. `jev-latest` moves; stored answers must stay comparable.
pub const JEV_MODEL: &str = "typesafe/jev-1.13";

/// Where and how to reach a Decisions-shaped endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionsConfig {
    /// Full URL to POST to (e.g. OpenRouter's, or `http://localhost:PORT/v1/systemone`).
    pub endpoint: String,
    /// Bearer token. `None` sends no `Authorization` header (local servers).
    pub api_key: Option<String>,
    /// Model string sent in every request body.
    pub model: String,
}

impl Default for DecisionsConfig {
    fn default() -> Self {
        Self {
            endpoint: OPENROUTER_DECISIONS_URL.to_string(),
            api_key: None,
            model: JEV_MODEL.to_string(),
        }
    }
}

impl DecisionsConfig {
    /// OpenRouter with the given key and the pinned model.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            ..Self::default()
        }
    }
}

/// One typed answer from Jev.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// Picked one option key out of the question's `criteria` map.
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    /// A value in `[0, 1]` against the question's legend.
    Score {
        score: f64,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    /// Yes/no. The value itself is P(yes); Jev returns no separate confidence.
    Noul { p_yes: f64 },
}

impl Answer {
    /// Model-reported confidence. `None` for `noul` answers, which carry none.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => {
                Some(*confidence)
            }
            Answer::Noul { .. } => None,
        }
    }

    /// Probability assigned to `key` (option key for `choice`, legend key for `score`).
    pub fn probability_of(&self, key: &str) -> Option<f64> {
        match self {
            Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
                probabilities.get(key).copied()
            }
            Answer::Noul { .. } => None,
        }
    }
}

/// A whole Decisions reply plus the telemetry needed to audit it.
#[derive(Debug, Clone)]
pub struct Decision {
    /// Model string as reported by the server (may carry a date suffix).
    pub model: String,
    /// Answers keyed by question id.
    pub answers: HashMap<String, Answer>,
    /// Wall-clock round trip as measured by the client.
    pub latency: Duration,
    /// Cost in USD as reported in `usage.cost`; `0.0` when absent.
    pub cost_usd: f64,
}

impl Decision {
    /// Answer for `question_id`, if Jev returned one.
    pub fn answer(&self, question_id: &str) -> Option<&Answer> {
        self.answers.get(question_id)
    }
}

/// Transport boundary. Production uses [`HttpTransport`]; tests substitute a fake.
#[async_trait]
pub trait DecisionsTransport: Send + Sync {
    /// POST `body` to the Decisions endpoint and return the parsed JSON reply.
    async fn post(&self, body: &Value) -> Result<Value, LlmError>;
}

/// `reqwest`-backed transport. Sends `Authorization: Bearer` only when a key is set.
pub struct HttpTransport {
    http: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
}

impl HttpTransport {
    pub fn new(endpoint: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: endpoint.into(),
            api_key,
        }
    }
}

#[async_trait]
impl DecisionsTransport for HttpTransport {
    async fn post(&self, body: &Value) -> Result<Value, LlmError> {
        let mut request = self
            .http
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://graphirm.ai")
            .header("X-OpenRouter-Title", "graphirm")
            .json(body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .map_err(|e| LlmError::Request(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| LlmError::Request(e.to_string()))?;
        if !status.is_success() {
            return Err(LlmError::provider(format!("Decisions {status}: {text}")));
        }
        serde_json::from_str(&text).map_err(LlmError::Serde)
    }
}

/// Client for a Decisions-shaped API.
pub struct DecisionsClient {
    transport: Arc<dyn DecisionsTransport>,
    model: String,
}

impl DecisionsClient {
    /// HTTP client for `config` (endpoint, optional key, model).
    pub fn from_config(config: DecisionsConfig) -> Self {
        Self::with_transport(Arc::new(HttpTransport::new(
            config.endpoint,
            config.api_key,
        )))
        .with_model(config.model)
    }

    /// OpenRouter with the given key and the pinned [`JEV_MODEL`].
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::from_config(DecisionsConfig::openrouter(api_key))
    }

    /// OpenRouter client reading `OPENROUTER_API_KEY`. `Err` when unset or empty.
    pub fn from_env() -> Result<Self, LlmError> {
        match std::env::var("OPENROUTER_API_KEY") {
            Ok(key) if !key.trim().is_empty() => Ok(Self::openrouter(key)),
            _ => Err(LlmError::config("OPENROUTER_API_KEY not set")),
        }
    }

    /// Client over an arbitrary transport (tests, alternative gateways), pinned model.
    pub fn with_transport(transport: Arc<dyn DecisionsTransport>) -> Self {
        Self {
            transport,
            model: JEV_MODEL.to_string(),
        }
    }

    /// Override the model string sent in every request.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// The model string sent with every request.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Ask Jev `questions` about `state`. Fails on transport error, timeout,
    /// non-2xx reply, or a reply without at least one parseable answer.
    pub async fn decide(
        &self,
        state: Value,
        questions: Value,
        timeout: Duration,
    ) -> Result<Decision, LlmError> {
        let body = serde_json::json!({
            "model": self.model,
            "state": state,
            "questions": questions,
        });
        let started = Instant::now();
        let raw = match tokio::time::timeout(timeout, self.transport.post(&body)).await {
            Ok(result) => result?,
            Err(_) => {
                return Err(LlmError::Request(format!(
                    "Decisions call timed out after {} ms",
                    timeout.as_millis()
                )));
            }
        };
        let mut decision = parse_decision(&raw, started.elapsed())?;
        if decision.model.is_empty() {
            decision.model = self.model.clone();
        }
        Ok(decision)
    }
}

// ----------------------------------------------------------------
// Reply parsing
// ----------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RawReply {
    model: Option<String>,
    answers: Option<HashMap<String, Value>>,
    usage: Option<RawUsage>,
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RawUsage {
    cost: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawChoice {
    choice: String,
    #[serde(default)]
    probabilities: BTreeMap<String, f64>,
    #[serde(default)]
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct RawScore {
    score: f64,
    #[serde(default)]
    probabilities: BTreeMap<String, f64>,
    #[serde(default)]
    confidence: f64,
}

#[derive(Debug, Deserialize)]
struct RawNoul {
    noul: f64,
}

/// Parse one answer object. `Ok(None)` for unknown types (skipped), `Err` for a
/// known type whose required fields are missing.
fn parse_answer(raw: &Value) -> Result<Option<Answer>, LlmError> {
    let kind = raw.get("type").and_then(Value::as_str).unwrap_or("");
    let answer = match kind {
        "choice" => {
            let a: RawChoice = serde_json::from_value(raw.clone())?;
            Answer::Choice {
                choice: a.choice,
                probabilities: a.probabilities,
                confidence: a.confidence,
            }
        }
        "score" => {
            let a: RawScore = serde_json::from_value(raw.clone())?;
            Answer::Score {
                score: a.score,
                probabilities: a.probabilities,
                confidence: a.confidence,
            }
        }
        "noul" => {
            let a: RawNoul = serde_json::from_value(raw.clone())?;
            Answer::Noul { p_yes: a.noul }
        }
        _ => return Ok(None),
    };
    Ok(Some(answer))
}

/// Convert a raw Decisions reply into a [`Decision`]. Public within the crate
/// so the shape can be unit-tested without a transport.
pub fn parse_decision(raw: &Value, latency: Duration) -> Result<Decision, LlmError> {
    let reply: RawReply = serde_json::from_value(raw.clone())
        .map_err(|e| LlmError::provider(format!("Decisions reply malformed: {e}")))?;

    if let Some(err) = reply.error {
        let message = err
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| err.to_string());
        return Err(LlmError::provider(format!("Decisions error: {message}")));
    }

    let raw_answers = reply
        .answers
        .ok_or_else(|| LlmError::provider("Decisions reply has no answers"))?;

    let mut answers = HashMap::with_capacity(raw_answers.len());
    for (id, value) in raw_answers {
        if let Some(answer) = parse_answer(&value)
            .map_err(|e| LlmError::provider(format!("Decisions answer {id:?} malformed: {e}")))?
        {
            answers.insert(id, answer);
        }
    }
    if answers.is_empty() {
        return Err(LlmError::provider(
            "Decisions reply carried no parseable answers",
        ));
    }

    Ok(Decision {
        model: reply.model.unwrap_or_default(),
        answers,
        latency,
        cost_usd: reply.usage.and_then(|u| u.cost).unwrap_or(0.0),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use serde_json::{Value, json};

    use super::*;
    use crate::error::LlmError;

    /// The exact reply shape observed from the live endpoint on 2026-09-28.
    fn live_reply() -> Value {
        json!({
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "tier": {
                    "type": "choice",
                    "choice": "cheap",
                    "probabilities": {"cheap": 1, "smart": 0},
                    "confidence": 1
                },
                "is_error": {"type": "noul", "noul": 0.02},
                "complexity": {
                    "type": "score",
                    "score": 0.11,
                    "legend": {"0": "0 = empty message", "1": "1 = very long message (500+ tokens)"},
                    "probabilities": {"0": 0.89, "1": 0.11},
                    "confidence": 0.78
                }
            },
            "usage": {"input_tokens": 435, "output_tokens": 64, "cost": 0.00001827},
            "id": "gen-dec-1790577060-TTVKwPciis7L1ZrPh2Tf",
            "provider": "TypeSafe"
        })
    }

    #[test]
    fn parses_choice_answer() {
        let d = parse_decision(&live_reply(), Duration::from_millis(300)).expect("parse");
        match d.answer("tier") {
            Some(Answer::Choice {
                choice,
                probabilities,
                confidence,
            }) => {
                assert_eq!(choice, "cheap");
                assert_eq!(probabilities.get("cheap"), Some(&1.0));
                assert_eq!(probabilities.get("smart"), Some(&0.0));
                assert_eq!(*confidence, 1.0);
            }
            other => panic!("expected choice answer, got {other:?}"),
        }
        assert_eq!(d.answer("tier").and_then(Answer::confidence), Some(1.0));
    }

    #[test]
    fn parses_score_answer() {
        let d = parse_decision(&live_reply(), Duration::from_millis(300)).expect("parse");
        match d.answer("complexity") {
            Some(Answer::Score {
                score,
                probabilities,
                confidence,
            }) => {
                assert!((score - 0.11).abs() < 1e-9);
                assert_eq!(probabilities.get("0"), Some(&0.89));
                assert!((confidence - 0.78).abs() < 1e-9);
            }
            other => panic!("expected score answer, got {other:?}"),
        }
    }

    #[test]
    fn parses_noul_answer_value_is_p_yes_and_has_no_confidence() {
        let d = parse_decision(&live_reply(), Duration::from_millis(300)).expect("parse");
        match d.answer("is_error") {
            Some(Answer::Noul { p_yes }) => assert!((p_yes - 0.02).abs() < 1e-9),
            other => panic!("expected noul answer, got {other:?}"),
        }
        assert_eq!(d.answer("is_error").and_then(Answer::confidence), None);
    }

    #[test]
    fn parses_model_cost_and_latency() {
        let d = parse_decision(&live_reply(), Duration::from_millis(300)).expect("parse");
        assert_eq!(d.model, "typesafe/jev-1.13-20260917");
        assert!((d.cost_usd - 0.00001827).abs() < 1e-12);
        assert_eq!(d.latency, Duration::from_millis(300));
        assert_eq!(d.answers.len(), 3);
    }

    #[test]
    fn choice_probability_lookup() {
        let d = parse_decision(&live_reply(), Duration::from_millis(1)).expect("parse");
        let a = d.answer("tier").expect("tier");
        assert_eq!(a.probability_of("cheap"), Some(1.0));
        assert_eq!(a.probability_of("nope"), None);
    }

    #[test]
    fn malformed_reply_without_answers_is_error() {
        let err = parse_decision(&json!({"foo": 1}), Duration::ZERO).unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)), "got {err:?}");
    }

    #[test]
    fn reply_with_empty_answers_is_error() {
        let err = parse_decision(&json!({"answers": {}}), Duration::ZERO).unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)), "got {err:?}");
    }

    #[test]
    fn reply_with_only_unknown_answer_types_is_error() {
        let raw = json!({"answers": {"q": {"type": "banana", "banana": 1}}});
        let err = parse_decision(&raw, Duration::ZERO).unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)), "got {err:?}");
    }

    #[test]
    fn choice_missing_choice_field_is_error() {
        let raw = json!({"answers": {"q": {"type": "choice", "probabilities": {"a": 1}}}});
        assert!(parse_decision(&raw, Duration::ZERO).is_err());
    }

    #[test]
    fn api_error_envelope_is_error() {
        let raw = json!({"error": {"message": "Invalid input", "code": 400}});
        let err = parse_decision(&raw, Duration::ZERO).unwrap_err();
        assert!(err.to_string().contains("Invalid input"), "got {err}");
    }

    // ---- transport-level behaviour, no network ----

    struct CapturingTransport {
        reply: Value,
        seen: Mutex<Option<Value>>,
    }

    #[async_trait]
    impl DecisionsTransport for CapturingTransport {
        async fn post(&self, body: &Value) -> Result<Value, LlmError> {
            *self.seen.lock().expect("lock") = Some(body.clone());
            Ok(self.reply.clone())
        }
    }

    struct HangingTransport;

    #[async_trait]
    impl DecisionsTransport for HangingTransport {
        async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok(json!({}))
        }
    }

    struct FailingTransport;

    #[async_trait]
    impl DecisionsTransport for FailingTransport {
        async fn post(&self, _body: &Value) -> Result<Value, LlmError> {
            Err(LlmError::Request("connection refused".into()))
        }
    }

    #[tokio::test]
    async fn decide_sends_pinned_model_state_and_questions() {
        let transport = Arc::new(CapturingTransport {
            reply: live_reply(),
            seen: Mutex::new(None),
        });
        let client = DecisionsClient::with_transport(transport.clone());
        let state = json!({"turn": 1});
        let questions = json!({"tier": {"type": "choice", "instructions": "x", "criteria": {}}});
        let d = client
            .decide(state.clone(), questions.clone(), Duration::from_secs(5))
            .await
            .expect("decide");
        assert_eq!(d.answers.len(), 3);

        let body = transport
            .seen
            .lock()
            .expect("lock")
            .clone()
            .expect("body sent");
        assert_eq!(body["model"], JEV_MODEL);
        assert_eq!(body["state"], state);
        assert_eq!(body["questions"], questions);
    }

    #[tokio::test(start_paused = true)]
    async fn decide_times_out() {
        let client = DecisionsClient::with_transport(Arc::new(HangingTransport));
        let fut = client.decide(json!({}), json!({}), Duration::from_millis(500));
        let err = fut.await.unwrap_err();
        assert!(matches!(err, LlmError::Request(_)), "got {err:?}");
        assert!(err.to_string().contains("timed out"), "got {err}");
    }

    #[tokio::test]
    async fn decide_propagates_transport_error() {
        let client = DecisionsClient::with_transport(Arc::new(FailingTransport));
        let err = client
            .decide(json!({}), json!({}), Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::Request(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn decide_rejects_malformed_reply() {
        let transport = Arc::new(CapturingTransport {
            reply: json!({"answers": "not an object"}),
            seen: Mutex::new(None),
        });
        let client = DecisionsClient::with_transport(transport);
        let err = client
            .decide(json!({}), json!({}), Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)), "got {err:?}");
    }

    #[test]
    fn model_constant_is_pinned_patch_version() {
        assert_eq!(JEV_MODEL, "typesafe/jev-1.13");
        assert_eq!(
            OPENROUTER_DECISIONS_URL,
            "https://openrouter.ai/api/alpha/decisions"
        );
    }

    // ---- provider-agnostic configuration ----

    #[test]
    fn config_defaults_to_openrouter_without_key() {
        let cfg = DecisionsConfig::default();
        assert_eq!(cfg.endpoint, OPENROUTER_DECISIONS_URL);
        assert_eq!(cfg.api_key, None);
        assert_eq!(cfg.model, JEV_MODEL);
    }

    #[test]
    fn openrouter_constructor_sets_key_and_default_endpoint() {
        let cfg = DecisionsConfig::openrouter("sk-test");
        assert_eq!(cfg.endpoint, OPENROUTER_DECISIONS_URL);
        assert_eq!(cfg.api_key.as_deref(), Some("sk-test"));
        assert_eq!(cfg.model, JEV_MODEL);
    }

    #[tokio::test]
    async fn decide_sends_configured_model_override() {
        let transport = Arc::new(CapturingTransport {
            reply: live_reply(),
            seen: Mutex::new(None),
        });
        let client =
            DecisionsClient::with_transport(transport.clone()).with_model("laya/systemone-1");
        client
            .decide(json!({}), json!({}), Duration::from_secs(1))
            .await
            .expect("decide");
        let body = transport.seen.lock().expect("lock").clone().expect("sent");
        assert_eq!(body["model"], "laya/systemone-1");
        assert_eq!(client.model(), "laya/systemone-1");
    }

    /// Minimal one-shot HTTP/1.1 stub: captures the raw request, replies with `body`.
    async fn stub_server(body: &'static str) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.expect("accept");
            let mut buf = vec![0u8; 16 * 1024];
            let mut raw = Vec::new();
            loop {
                let n = sock.read(&mut buf).await.expect("read");
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw);
                if let Some(idx) = text.find("\r\n\r\n") {
                    let head = &text[..idx];
                    let len = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length: "))
                        .or_else(|| {
                            head.lines()
                                .find_map(|l| l.strip_prefix("Content-Length: "))
                        })
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if raw.len() >= idx + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).await.expect("write");
            sock.shutdown().await.ok();
            String::from_utf8_lossy(&raw).to_string()
        });
        (format!("http://{addr}/v1/systemone"), handle)
    }

    const STUB_REPLY: &str = r#"{"answers":{"q":{"type":"noul","noul":0.4}}}"#;

    #[tokio::test]
    async fn http_transport_posts_to_configured_endpoint_with_bearer() {
        let (url, handle) = stub_server(STUB_REPLY).await;
        let client = DecisionsClient::from_config(DecisionsConfig {
            endpoint: url,
            api_key: Some("local-key".into()),
            model: "laya/systemone-1".into(),
        });
        let d = client
            .decide(json!({"a": 1}), json!({}), Duration::from_secs(5))
            .await
            .expect("decide");
        assert_eq!(d.answer("q"), Some(&Answer::Noul { p_yes: 0.4 }));

        let raw = handle.await.expect("join");
        assert!(raw.starts_with("POST /v1/systemone HTTP/1.1"), "{raw}");
        assert!(raw.contains("authorization: Bearer local-key"), "{raw}");
        assert!(raw.contains(r#""model":"laya/systemone-1""#), "{raw}");
    }

    #[tokio::test]
    async fn http_transport_omits_authorization_without_key() {
        let (url, handle) = stub_server(STUB_REPLY).await;
        let client = DecisionsClient::from_config(DecisionsConfig {
            endpoint: url,
            api_key: None,
            model: JEV_MODEL.into(),
        });
        client
            .decide(json!({}), json!({}), Duration::from_secs(5))
            .await
            .expect("decide");
        let raw = handle.await.expect("join").to_lowercase();
        assert!(!raw.contains("authorization:"), "{raw}");
    }
}
