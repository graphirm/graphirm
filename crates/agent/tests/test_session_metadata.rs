use graphirm_agent::session::{SessionMetadata, SessionStatus};
use chrono::Utc;

#[test]
fn test_session_metadata_creation() {
    let now = Utc::now();
    let metadata = SessionMetadata::from_agent_node_id(
        "session-001".to_string(),
        "auth-refactor".to_string(),
        "claude-sonnet-4".to_string(),
        now,
        SessionStatus::Completed,
    );

    assert_eq!(metadata.session_id, "session-001");
    assert_eq!(metadata.name, "auth-refactor");
    assert_eq!(metadata.model, "claude-sonnet-4");
    assert_eq!(metadata.status, SessionStatus::Completed);
    assert_eq!(metadata.created_at, now);
}

#[test]
fn test_session_status_enum() {
    // Test all status variants exist and are comparable
    let statuses = vec![
        SessionStatus::Running,
        SessionStatus::Idle,
        SessionStatus::Completed,
        SessionStatus::Failed,
    ];
    
    assert_eq!(statuses.len(), 4);
    
    // Verify completed status
    let completed = SessionStatus::Completed;
    assert_eq!(completed, SessionStatus::Completed);
}
