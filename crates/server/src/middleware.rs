//! Axum middleware for request logging and API key authentication.

use axum::Json;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::http::header;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use chrono::Utc;

use crate::request_log::{RequestLogEntry, RequestLogger, classify_endpoint, extract_session_id};
use crate::state::AppState;

/// Axum middleware function that logs every request to the [`RequestLogger`].
///
/// Must be used with `axum::middleware::from_fn` and requires a
/// `RequestLogger` in an axum `Extension`.
pub async fn request_logging(
    logger: Option<axum::Extension<RequestLogger>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let Some(axum::Extension(logger)) = logger else {
        return next.run(request).await;
    };

    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let start = std::time::Instant::now();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status().as_u16();

    let session_id = extract_session_id(&path);
    let endpoint_group = classify_endpoint(&path).to_string();

    logger.log(RequestLogEntry {
        timestamp: Utc::now(),
        method,
        path,
        status,
        duration_ms: duration.as_secs_f64() * 1000.0,
        session_id,
        endpoint_group,
    });

    response
}

/// Bearer token auth. Checks `Authorization: Bearer <key>` first, then `?token=` (for EventSource).
///
/// When [`AppState::api_key`] is empty, all requests pass (tests / local in-process).
/// `OPTIONS` is always allowed so CORS preflight succeeds. `/api/health` is exempt (load balancers).
pub async fn api_key_auth(State(state): State<AppState>, request: Request, next: Next) -> Response {
    if state.api_key.is_empty() {
        return next.run(request).await;
    }

    if request.uri().path() == "/api/health" {
        return next.run(request).await;
    }

    if request.method() == axum::http::Method::OPTIONS {
        return next.run(request).await;
    }

    let header_ok = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| t == state.api_key.as_str());

    let query_ok = request
        .uri()
        .query()
        .and_then(|q| {
            q.split('&').find_map(|pair| {
                let (k, v) = pair.split_once('=')?;
                (k == "token").then_some(v)
            })
        })
        .is_some_and(|t| t == state.api_key.as_str());

    if header_ok || query_ok {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Unauthorized" })),
        )
            .into_response()
    }
}
