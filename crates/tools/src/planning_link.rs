//! Planning Knowledge ↔ file Content linkage (shared by `graph_query` and `write`/`edit`).

use graphirm_graph::edges::{EdgeType, GraphEdge};
use graphirm_graph::nodes::{NodeId, NodeType};
use graphirm_graph::{GraphError, GraphStore};
use serde_json::json;

/// Outcome of attempting a planning → content `relates_to` edge (see `link_planning_content_edge`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanningContentLink {
    Inserted,
    AlreadyLinked,
    /// Wrong node types, session mismatch, unknown relationship, etc. (message for `graph_query` errors).
    NotApplicable(&'static str),
}

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

/// Add **`RelatesTo`** from planning Knowledge → Content with **`artifact_link`** metadata,
/// matching `graph_query` `project` **`link_content`** rules.
///
/// Returns [`PlanningContentLink::NotApplicable`] when validation fails (caller maps to tool error
/// or silent skip). Graph errors propagate for I/O failures.
pub fn link_planning_content_edge(
    graph: &GraphStore,
    agent_id: &NodeId,
    planning_id: &NodeId,
    content_id: &NodeId,
    relationship: &str,
) -> Result<PlanningContentLink, GraphError> {
    if relationship != "implements" && relationship != "documents" {
        return Ok(PlanningContentLink::NotApplicable(
            "unknown relationship; must be one of: implements, documents",
        ));
    }

    let content_node = graph.get_node(content_id)?;
    if !matches!(content_node.node_type, NodeType::Content(_)) {
        return Ok(PlanningContentLink::NotApplicable(
            "link_content: content_id must refer to a Content node",
        ));
    }
    let sid = content_node
        .metadata
        .get("session_id")
        .and_then(|v| v.as_str());
    if sid != Some(agent_id.0.as_str()) {
        return Ok(PlanningContentLink::NotApplicable(
            "link_content: Content node session_id must match this session's agent id",
        ));
    }

    let plan_node = graph.get_node(planning_id)?;
    if plan_node.metadata.get("planning").and_then(|v| v.as_bool()) != Some(true) {
        return Ok(PlanningContentLink::NotApplicable(
            "link_content: planning_node_id must be a planning Knowledge node",
        ));
    }
    if !matches!(plan_node.node_type, NodeType::Knowledge(_)) {
        return Ok(PlanningContentLink::NotApplicable(
            "link_content: planning_node_id must be a Knowledge node",
        ));
    }

    let edges = graph.edges_for_node(planning_id)?;
    if edges.iter().any(|e| {
        e.edge_type == EdgeType::RelatesTo && e.source == *planning_id && e.target == *content_id
    }) {
        return Ok(PlanningContentLink::AlreadyLinked);
    }

    let meta = json!({ "artifact_link": relationship });
    graph.add_edge(
        GraphEdge::new(EdgeType::RelatesTo, planning_id.clone(), content_id.clone())
            .with_metadata(meta),
    )?;
    Ok(PlanningContentLink::Inserted)
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
            PlanningContentLink::Inserted => tracing::debug!(
                content_id = %cid,
                planning_id = %pid,
                "auto-linked file content to planning node"
            ),
            PlanningContentLink::AlreadyLinked => tracing::debug!(
                content_id = %cid,
                "file content already linked to planning node"
            ),
            PlanningContentLink::NotApplicable(_) => tracing::debug!(
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
    use graphirm_graph::nodes::{AgentData, ContentData, GraphNode, KnowledgeData};

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
            PlanningContentLink::Inserted
        );
        assert_eq!(
            link_planning_content_edge(&g, &agent, &plan, &file_id, "implements").unwrap(),
            PlanningContentLink::AlreadyLinked
        );
    }
}
