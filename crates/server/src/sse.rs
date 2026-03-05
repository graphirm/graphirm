//! Server-Sent Events (SSE) handlers for streaming real-time agent events.
//!
//! Two endpoints are provided:
//! - [`sse_handler`] — global stream; emits all events from all sessions.
//! - [`sse_session_handler`] — session-scoped stream; filtered to one session.
//!
//! Both endpoints keep the HTTP connection alive with 15-second heartbeat
//! pings so proxies and browsers don't time out idle connections.

use std::convert::Infallible;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::state::AppState;
use crate::types::{SessionId, SseEvent};

/// `GET /api/events` — global SSE stream; emits all agent events from every session.
pub async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();

    let stream = BroadcastStream::new(rx)
        .filter_map(|result| result.ok())
        .map(|event: SseEvent| -> Result<Event, Infallible> {
            let data = serde_json::to_string(&event).unwrap_or_default();
            Ok(Event::default()
                .event(event.event_type.to_string())
                .data(data))
        });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("heartbeat"),
    )
}

/// `GET /api/events/{session_id}` — session-scoped SSE stream.
///
/// Subscribes to the global broadcast channel and filters to events whose
/// `session_id` matches the path parameter. Clients should use this endpoint
/// when they are interested in a single session's lifecycle.
pub async fn sse_session_handler(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let rx = state.event_tx.subscribe();
    let target = SessionId::from(session_id.as_str());

    let stream = BroadcastStream::new(rx)
        .filter_map(|result| result.ok())
        .filter(move |event: &SseEvent| event.session_id == target)
        .map(|event: SseEvent| -> Result<Event, Infallible> {
            let data = serde_json::to_string(&event).unwrap_or_default();
            Ok(Event::default()
                .event(event.event_type.to_string())
                .data(data))
        });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("heartbeat"),
    )
}

#[cfg(test)]
mod tests {
    use tokio::sync::broadcast;
    use tokio_stream::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    use crate::types::{SessionId, SseEvent, SseEventType};

    fn make_event(session: &str, event_type: SseEventType) -> SseEvent {
        SseEvent {
            session_id: SessionId::from(session),
            event_type,
            data: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn test_broadcast_channel_delivers_events() {
        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let mut rx = tx.subscribe();

        let event = make_event("s1", SseEventType::TurnStart);
        tx.send(event).unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.session_id, SessionId::from("s1"));
        assert!(matches!(received.event_type, SseEventType::TurnStart));
    }

    #[tokio::test]
    async fn test_broadcast_multiple_subscribers() {
        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let mut rx1 = tx.subscribe();
        let mut rx2 = tx.subscribe();

        tx.send(make_event("s1", SseEventType::AgentStart)).unwrap();

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert!(matches!(e1.event_type, SseEventType::AgentStart));
        assert!(matches!(e2.event_type, SseEventType::AgentStart));
    }

    #[tokio::test]
    async fn test_broadcast_stream_maps_to_sse_events() {
        use std::convert::Infallible;
        use axum::response::sse::Event;

        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let rx = tx.subscribe();

        tx.send(make_event("s1", SseEventType::TurnStart)).unwrap();
        tx.send(make_event("s2", SseEventType::ToolStart)).unwrap();
        drop(tx);

        let stream = BroadcastStream::new(rx)
            .filter_map(|r| r.ok())
            .map(|event: SseEvent| -> Result<Event, Infallible> {
                let data = serde_json::to_string(&event).unwrap_or_default();
                Ok(Event::default()
                    .event(event.event_type.to_string())
                    .data(data))
            });

        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 2);
        assert!(events[0].is_ok());
        assert!(events[1].is_ok());
    }

    #[tokio::test]
    async fn test_session_filtered_stream() {
        let (tx, _) = broadcast::channel::<SseEvent>(256);
        let rx = tx.subscribe();
        let target = SessionId::from("s1");

        tx.send(make_event("s1", SseEventType::TurnStart)).unwrap();
        tx.send(make_event("s2", SseEventType::TurnStart)).unwrap();
        tx.send(make_event("s1", SseEventType::AgentEnd)).unwrap();
        drop(tx);

        let stream = BroadcastStream::new(rx)
            .filter_map(|r| r.ok())
            .filter(move |e: &SseEvent| e.session_id == target);

        let events: Vec<SseEvent> = stream.collect().await;
        assert_eq!(events.len(), 2);
        assert!(events
            .iter()
            .all(|e| e.session_id == SessionId::from("s1")));
    }
}
