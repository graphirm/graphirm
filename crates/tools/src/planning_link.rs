//! Planning Knowledge ↔ artifact linkage (shared by `graph_query`, `write`/`edit`, and delegate).
//!
//! # Content artifacts (`link_content`, auto-link on file write)
//!
//! File [`Content`](graphirm_graph::nodes::NodeType::Content) nodes use **`RelatesTo`** from
//! planning Knowledge → content with edge metadata **`artifact_link`**: `implements` | `documents`.
//! Session ownership is enforced via **`metadata.session_id`** on the Content node matching the
//! session **Agent** id (see [`link_planning_content_edge`]).
//!
//! # Task artifacts (`link_task`, optional auto-link on delegate)
//!
//! Delegated work uses [`NodeType::Task`](graphirm_graph::nodes::NodeType::Task). The same edge
//! shape applies: **`RelatesTo`** planning → task with **`artifact_link`**. Task nodes do not
//! require `session_id` metadata; instead we validate **delegation structure**:
//!
//! - **Primary (parent) session:** `EdgeType::DelegatesTo` from the parent **Agent** → **Task**
//!   (`spawn_subagent` in `graphirm-agent` creates this before the subagent Agent exists).
//! - **Subagent session:** `EdgeType::SpawnedBy` from **Task** → child **Agent** (the subagent’s
//!   session id is the child agent node id).
//!
//! Either relationship is sufficient to prove the Task belongs to the caller’s session when
//! invoking [`link_planning_task_edge`].

use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{NodeId, NodeType};
use graphirm_graph::{GraphError, GraphStore};
use serde_json::json;

/// Outcome of attempting a planning → artifact `relates_to` edge (content or task).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningArtifactLink {
    Inserted,
    AlreadyLinked,
    /// Wrong node types, session mismatch, unknown relationship, etc. (message for `graph_query` errors).
    NotApplicable(&'static str),
}

/// Back-compat alias (older name).
pub type PlanningContentLink = PlanningArtifactLink;

/// Resolve the planning Knowledge node for this session from **`link_session`** edges:
/// outgoing **`DerivedFrom`** from **`agent_id`** to a Knowledge node with **`metadata.planning == true`**.
///
/// When several such targets exist, returns the one with the **latest `created_at`** on the
/// planning node (deterministic tie-break).
pub fn resolve_session_planning_node(
    graph: &GraphStore,
    agent_id: &NodeId,
) -> Result<Option<NodeId>, GraphError> {
    let edges = graph.edges_for_node(agent_id)?;
    let mut candidates: Vec<NodeId> = Vec::new();
    for e in edges {
        if e.edge_type != EdgeType::DerivedFrom || e.source != *agent_id {
            continue;
        }
        let node = graph.get_node(&e.target)?;
        if !matches!(node.node_type, NodeType::Knowledge(_)) {
            continue;
        }
        if node.metadata.get("planning").and_then(|v| v.as_bool()) != Some(true) {
            continue;
        }
        candidates.push(e.target);
    }
    match candidates.len() {
        0 => Ok(None),
        1 => Ok(Some(candidates[0].clone())),
        _ => {
            let mut best: Option<(NodeId, _)> = None;
            for id in candidates {
                let node = graph.get_node(&id)?;
                let replace = match &best {
                    None => true,
                    Some((_, ts)) => node.created_at > *ts,
                };
                if replace {
                    best = Some((id, node.created_at));
                }
            }
            Ok(best.map(|(id, _)| id))
        }
    }
}

/// Returns true if **`task_id`** is the delegation node for **`agent_id`** (parent or subagent).
pub fn task_in_scope_for_agent(
    graph: &GraphStore,
    agent_id: &NodeId,
    task_id: &NodeId,
) -> Result<bool, GraphError> {
    for e in graph.edges_for_node(agent_id)? {
        if e.edge_type == EdgeType::DelegatesTo && e.source == *agent_id && e.target == *task_id {
            return Ok(true);
        }
    }
    for e in graph.edges_for_node(task_id)? {
        if e.edge_type == EdgeType::SpawnedBy && e.source == *task_id && e.target == *agent_id {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Add **`RelatesTo`** from planning Knowledge → Content with **`artifact_link`** metadata,
/// matching `graph_query` `project` **`link_content`** rules.
///
/// Returns [`PlanningArtifactLink::NotApplicable`] when validation fails (caller maps to tool error
/// or silent skip). Graph errors propagate for I/O failures.
pub fn link_planning_content_edge(
    graph: &GraphStore,
    agent_id: &NodeId,
    planning_id: &NodeId,
    content_id: &NodeId,
    relationship: &str,
) -> Result<PlanningArtifactLink, GraphError> {
    if relationship != "implements" && relationship != "documents" {
        return Ok(PlanningArtifactLink::NotApplicable(
            "unknown relationship; must be one of: implements, documents",
        ));
    }

    let content_node = graph.get_node(content_id)?;
    if !matches!(content_node.node_type, NodeType::Content(_)) {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_content: content_id must refer to a Content node",
        ));
    }
    let sid = content_node
        .metadata
        .get("session_id")
        .and_then(|v| v.as_str());
    if sid != Some(agent_id.0.as_str()) {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_content: Content node session_id must match this session's agent id",
        ));
    }

    let plan_node = graph.get_node(planning_id)?;
    if plan_node.metadata.get("planning").and_then(|v| v.as_bool()) != Some(true) {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_content: planning_node_id must be a planning Knowledge node",
        ));
    }
    if !matches!(plan_node.node_type, NodeType::Knowledge(_)) {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_content: planning_node_id must be a Knowledge node",
        ));
    }

    let edges = graph.edges_for_node(planning_id)?;
    if edges.iter().any(|e| {
        e.edge_type == EdgeType::RelatesTo && e.source == *planning_id && e.target == *content_id
    }) {
        return Ok(PlanningArtifactLink::AlreadyLinked);
    }

    let meta = json!({ "artifact_link": relationship });
    graph.add_edge(
        GraphEdge::new(EdgeType::RelatesTo, planning_id.clone(), content_id.clone())
            .with_metadata(meta),
    )?;
    Ok(PlanningArtifactLink::Inserted)
}

/// Add **`RelatesTo`** from planning Knowledge → Task with **`artifact_link`** metadata
/// (`graph_query` `project` **`link_task`**).
///
/// The Task must be in scope for **`agent_id`** (`DelegatesTo` from parent agent, or
/// `SpawnedBy` from task to subagent agent). See module docs.
pub fn link_planning_task_edge(
    graph: &GraphStore,
    agent_id: &NodeId,
    planning_id: &NodeId,
    task_id: &NodeId,
    relationship: &str,
) -> Result<PlanningArtifactLink, GraphError> {
    if relationship != "implements" && relationship != "documents" {
        return Ok(PlanningArtifactLink::NotApplicable(
            "unknown relationship; must be one of: implements, documents",
        ));
    }

    let task_node = graph.get_node(task_id)?;
    if !matches!(task_node.node_type, NodeType::Task(_)) {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_task: task_id must refer to a Task node",
        ));
    }

    if !task_in_scope_for_agent(graph, agent_id, task_id)? {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_task: Task is not delegated to or spawned by this session's agent",
        ));
    }

    let plan_node = graph.get_node(planning_id)?;
    if plan_node.metadata.get("planning").and_then(|v| v.as_bool()) != Some(true) {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_task: planning_node_id must be a planning Knowledge node",
        ));
    }
    if !matches!(plan_node.node_type, NodeType::Knowledge(_)) {
        return Ok(PlanningArtifactLink::NotApplicable(
            "link_task: planning_node_id must be a Knowledge node",
        ));
    }

    let edges = graph.edges_for_node(planning_id)?;
    if edges.iter().any(|e| {
        e.edge_type == EdgeType::RelatesTo && e.source == *planning_id && e.target == *task_id
    }) {
        return Ok(PlanningArtifactLink::AlreadyLinked);
    }

    let meta = json!({ "artifact_link": relationship });
    graph.add_edge(
        GraphEdge::new(EdgeType::RelatesTo, planning_id.clone(), task_id.clone())
            .with_metadata(meta),
    )?;
    Ok(PlanningArtifactLink::Inserted)
}

/// After persisting a **`file`** Content node, link it to the session’s planning node when
/// [`crate::ToolContext::auto_link_write_to_planning`] is true. Never fails the caller’s tool.
pub async fn try_auto_link_written_file_content(ctx: &crate::ToolContext, content_id: &NodeId) {
    if !ctx.auto_link_write_to_planning {
        return;
    }
    let graph = ctx.graph.clone();
    let agent = ctx.agent_id.clone();
    let cid = content_id.clone();
    let join = tokio::task::spawn_blocking(move || -> Result<(), GraphError> {
        let n = graph.get_node(&cid)?;
        let NodeType::Content(data) = &n.node_type else {
            return Ok(());
        };
        if data.content_type != "file" {
            return Ok(());
        }
        let Some(pid) = resolve_session_planning_node(&graph, &agent)? else {
            return Ok(());
        };
        match link_planning_content_edge(&graph, &agent, &pid, &cid, "implements")? {
            PlanningArtifactLink::Inserted => tracing::debug!(
                content_id = %cid,
                planning_id = %pid,
                "auto-linked file content to planning node"
            ),
            PlanningArtifactLink::AlreadyLinked => tracing::debug!(
                content_id = %cid,
                "file content already linked to planning node"
            ),
            PlanningArtifactLink::NotApplicable(_) => tracing::debug!(
                content_id = %cid,
                "auto-link skipped (planning↔content validation)"
            ),
        }
        Ok(())
    })
    .await;
    match join {
        Ok(Ok(())) => {}
        Ok(Err(e)) => tracing::warn!(error = %e, "auto-link planning↔content graph error"),
        Err(e) => tracing::warn!(error = %e, "auto-link planning↔content task join error"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::nodes::TaskStatus;
    use graphirm_graph::nodes::{AgentData, ContentData, GraphNode, KnowledgeData, TaskData};

    fn planning_knowledge(entity: &str, summary: &str) -> GraphNode {
        let mut n = GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: entity.to_string(),
            entity_type: "story".to_string(),
            summary: summary.to_string(),
            confidence: 1.0,
        }));
        n.metadata["planning"] = json!(true);
        n
    }

    #[test]
    fn resolve_none_without_derived_from() {
        let g = GraphStore::open_memory().unwrap();
        let agent = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "a".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        assert_eq!(resolve_session_planning_node(&g, &agent).unwrap(), None);
    }

    #[test]
    fn resolve_single_planning_target() {
        let g = GraphStore::open_memory().unwrap();
        let agent = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "a".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        let story = g.add_node(planning_knowledge("S1", "slice")).unwrap();
        g.add_edge(GraphEdge::new(
            EdgeType::DerivedFrom,
            agent.clone(),
            story.clone(),
        ))
        .unwrap();
        assert_eq!(
            resolve_session_planning_node(&g, &agent).unwrap(),
            Some(story)
        );
    }

    #[test]
    fn resolve_picks_latest_created_planning_node() {
        let g = GraphStore::open_memory().unwrap();
        let agent = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "a".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        let older = g.add_node(planning_knowledge("old", "first")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let newer = g.add_node(planning_knowledge("new", "second")).unwrap();
        g.add_edge(GraphEdge::new(EdgeType::DerivedFrom, agent.clone(), older))
            .unwrap();
        g.add_edge(GraphEdge::new(
            EdgeType::DerivedFrom,
            agent.clone(),
            newer.clone(),
        ))
        .unwrap();
        assert_eq!(
            resolve_session_planning_node(&g, &agent).unwrap(),
            Some(newer)
        );
    }

    #[test]
    fn link_edge_idempotent() {
        let g = GraphStore::open_memory().unwrap();
        let agent = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "a".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        let plan = g.add_node(planning_knowledge("S", "s")).unwrap();
        let mut file = GraphNode::new(NodeType::Content(ContentData {
            content_type: "file".to_string(),
            path: Some("/x.rs".into()),
            body: "".into(),
            language: None,
        }));
        file.metadata["session_id"] = json!(agent.0.as_str());
        let file_id = g.add_node(file).unwrap();

        assert_eq!(
            link_planning_content_edge(&g, &agent, &plan, &file_id, "implements").unwrap(),
            PlanningArtifactLink::Inserted
        );
        assert_eq!(
            link_planning_content_edge(&g, &agent, &plan, &file_id, "implements").unwrap(),
            PlanningArtifactLink::AlreadyLinked
        );
    }

    #[test]
    fn link_task_via_delegates_to_parent() {
        let g = GraphStore::open_memory().unwrap();
        let parent = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "p".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        let plan = g.add_node(planning_knowledge("S", "s")).unwrap();
        let task = g
            .add_node(GraphNode::new(NodeType::Task(TaskData {
                title: "t".into(),
                description: "".into(),
                status: TaskStatus::Pending,
                priority: None,
            })))
            .unwrap();
        g.add_edge(GraphEdge::new(
            EdgeType::DelegatesTo,
            parent.clone(),
            task.clone(),
        ))
        .unwrap();

        assert_eq!(
            link_planning_task_edge(&g, &parent, &plan, &task, "implements").unwrap(),
            PlanningArtifactLink::Inserted
        );
        assert_eq!(
            link_planning_task_edge(&g, &parent, &plan, &task, "implements").unwrap(),
            PlanningArtifactLink::AlreadyLinked
        );
    }

    #[test]
    fn link_task_via_spawned_by_subagent() {
        let g = GraphStore::open_memory().unwrap();
        let parent = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "p".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        let child = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "c".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        let plan = g.add_node(planning_knowledge("S", "s")).unwrap();
        let task = g
            .add_node(GraphNode::new(NodeType::Task(TaskData {
                title: "t".into(),
                description: "".into(),
                status: TaskStatus::Pending,
                priority: None,
            })))
            .unwrap();
        g.add_edge(GraphEdge::new(EdgeType::DelegatesTo, parent, task.clone()))
            .unwrap();
        g.add_edge(GraphEdge::new(
            EdgeType::SpawnedBy,
            task.clone(),
            child.clone(),
        ))
        .unwrap();

        assert_eq!(
            link_planning_task_edge(&g, &child, &plan, &task, "documents").unwrap(),
            PlanningArtifactLink::Inserted
        );
    }

    #[test]
    fn link_task_rejects_unrelated_agent() {
        let g = GraphStore::open_memory().unwrap();
        let stranger = g
            .add_node(GraphNode::new(NodeType::Agent(AgentData {
                name: "x".into(),
                model: "m".into(),
                system_prompt: None,
                status: "active".into(),
            })))
            .unwrap();
        let plan = g.add_node(planning_knowledge("S", "s")).unwrap();
        let task = g
            .add_node(GraphNode::new(NodeType::Task(TaskData {
                title: "t".into(),
                description: "".into(),
                status: TaskStatus::Pending,
                priority: None,
            })))
            .unwrap();

        assert!(matches!(
            link_planning_task_edge(&g, &stranger, &plan, &task, "implements").unwrap(),
            PlanningArtifactLink::NotApplicable(_)
        ));
    }
}
