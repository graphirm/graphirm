//! Axum middleware for request logging.
//!
//! Wraps every request to capture method, path, status, and duration, then
//! logs the entry through the [`RequestLogger`] attached as an axum Extension.

use axum::body::Body;
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use chrono::Utc;

use crate::request_log::{RequestLogEntry, RequestLogger, classify_endpoint, extract_session_id};

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
