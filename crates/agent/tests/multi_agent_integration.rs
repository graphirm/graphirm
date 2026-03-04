use std::collections::HashMap;
use std::sync::Arc;

use graphirm_agent::{
    AgentConfig, AgentMode, AgentRegistry, EventBus, LlmFactory, Session, SubagentTool,
    run_agent_loop, spawn_subagent, wait_for_subagents,
};
use graphirm_graph::edges::EdgeType;
use graphirm_graph::nodes::NodeType;
use graphirm_graph::{Direction, GraphStore};
use graphirm_llm::{LlmProvider, MockProvider, MockResponse};
use graphirm_tools::registry::ToolRegistry;
use tokio_util::sync::CancellationToken;

fn primary_config() -> AgentConfig {
    AgentConfig {
        name: "build".to_string(),
        mode: AgentMode::Primary,
        model: "test-primary".to_string(),
        description: "Primary coding agent".to_string(),
        system_prompt: "You are the primary coding agent.".to_string(),
        max_turns: 10,
        tools: vec!["delegate".to_string()],
        ..AgentConfig::default()
    }
}

fn explore_config() -> AgentConfig {
    AgentConfig {
        name: "explore".to_string(),
        mode: AgentMode::Subagent,
        model: "test-explore".to_string(),
        description: "Read-only exploration agent".to_string(),
        system_prompt: "You are an exploration agent. Read code and report findings.".to_string(),
        max_turns: 5,
        tools: vec!["read".to_string()],
        ..AgentConfig::default()
    }
}

#[tokio::test]
async fn test_full_multi_agent_flow() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());

    let mut configs = HashMap::new();
    configs.insert("build".to_string(), primary_config());
    configs.insert("explore".to_string(), explore_config());
    let registry = Arc::new(AgentRegistry::from_configs(configs));

    // The primary agent:
    // - Turn 1: calls delegate tool
    // - Turn 2: provides final answer (after tool result comes back)
    //
    // The explore subagent:
    // - Turn 1: returns findings
    let factory: LlmFactory = Arc::new(|model: &str| -> Box<dyn LlmProvider> {
        if model == "test-primary" {
            Box::new(MockProvider::new(vec![
                MockResponse::tool_call(
                    "tc_delegate_1",
                    "delegate",
                    serde_json::json!({
                        "agent": "explore",
                        "task": "Analyze the auth module patterns"
                    }),
                ),
                MockResponse::text(
                    "Based on the exploration results, the auth module uses JWT tokens \
                     with refresh token rotation. The implementation is solid.",
                ),
            ]))
        } else {
            Box::new(MockProvider::new(vec![MockResponse::text(
                "Found: auth uses JWT with 24h access tokens and 7d refresh tokens. \
                 Refresh rotation is implemented in src/auth/refresh.rs.",
            )]))
        }
    });

    let base_tools = Arc::new(ToolRegistry::new());
    let events = Arc::new(EventBus::new());

    // Set up primary session
    let primary = primary_config();
    let session = Session::new(graph.clone(), primary.clone()).unwrap();
    session
        .add_user_message("Review the auth module and give me a summary.")
        .unwrap();

    // Build tool registry with the delegate tool
    let delegate_tool = SubagentTool::new(
        graph.clone(),
        registry.clone(),
        factory.clone(),
        base_tools.clone(),
        events.clone(),
        session.id.clone(),
    );

    let mut tool_registry = ToolRegistry::new();
    tool_registry.register(Arc::new(delegate_tool));

    let llm = (factory)("test-primary");
    let cancel = CancellationToken::new();

    run_agent_loop(&session, llm.as_ref(), &tool_registry, &events, &cancel)
        .await
        .unwrap();

    // ── Graph structure verification ────────────────────────────────────────

    // 1. Primary agent node exists
    let agent_node = graph.get_node(&session.id).unwrap();
    assert!(matches!(agent_node.node_type, NodeType::Agent(_)));

    // 2. Primary agent delegated a task
    let delegated = graph
        .neighbors(&session.id, Some(EdgeType::DelegatesTo), Direction::Outgoing)
        .unwrap();
    assert_eq!(delegated.len(), 1, "Primary should have delegated 1 task");

    let task_node = &delegated[0];
    assert!(matches!(task_node.node_type, NodeType::Task(_)));
    if let NodeType::Task(data) = &task_node.node_type {
        assert_eq!(data.status, "completed");
    }

    // 3. Task spawned a subagent
    let spawned = graph
        .neighbors(&task_node.id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
        .unwrap();
    assert_eq!(spawned.len(), 1, "Task should have spawned 1 subagent");

    let subagent = &spawned[0];
    assert!(matches!(subagent.node_type, NodeType::Agent(_)));

    // 4. Subagent produced output (assistant responses)
    let subagent_outputs = graph
        .neighbors(&subagent.id, Some(EdgeType::Produces), Direction::Outgoing)
        .unwrap();
    let assistant_responses: Vec<_> = subagent_outputs
        .iter()
        .filter(|n| {
            matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant")
        })
        .collect();
    assert!(
        !assistant_responses.is_empty(),
        "Subagent should have produced at least one assistant response"
    );

    // 5. Primary agent produced responses (at least tool call + final answer)
    let primary_outputs = graph
        .neighbors(&session.id, Some(EdgeType::Produces), Direction::Outgoing)
        .unwrap();
    let primary_assistant_responses: Vec<_> = primary_outputs
        .iter()
        .filter(|n| {
            matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant")
        })
        .collect();
    assert!(
        primary_assistant_responses.len() >= 2,
        "Primary should have at least 2 assistant responses (delegate + final), got {}",
        primary_assistant_responses.len()
    );

    // 6. Delegation chain is traversable
    let chain = graph
        .traverse(
            &session.id,
            &[EdgeType::DelegatesTo, EdgeType::SpawnedBy],
            3,
        )
        .unwrap();
    assert!(
        chain.len() >= 2,
        "Delegation chain should include task + subagent"
    );
}

#[tokio::test]
async fn test_multi_agent_graph_isolation() {
    let graph = Arc::new(GraphStore::open_memory().unwrap());

    let mut configs = HashMap::new();
    configs.insert("build".to_string(), primary_config());
    configs.insert("explore".to_string(), explore_config());
    let registry = Arc::new(AgentRegistry::from_configs(configs));

    let factory: LlmFactory = Arc::new(|_model: &str| -> Box<dyn LlmProvider> {
        Box::new(MockProvider::new(vec![MockResponse::text("done")]))
    });

    let base_tools = Arc::new(ToolRegistry::new());
    let events = Arc::new(EventBus::new());

    // Create primary session with 3 user messages
    let primary = primary_config();
    let session = Session::new(graph.clone(), primary.clone()).unwrap();
    session.add_user_message("Message 1").unwrap();
    session.add_user_message("Message 2").unwrap();
    session.add_user_message("Message 3").unwrap();

    // Spawn a subagent
    let cancel = CancellationToken::new();
    let handle = spawn_subagent(
        &graph,
        &registry,
        &factory,
        &base_tools,
        &events,
        &session.id,
        "explore",
        "Explore something",
        vec![],
        cancel,
    )
    .await
    .unwrap();

    let subagent_id = handle.agent_id.clone();
    wait_for_subagents(vec![handle]).await.unwrap();

    // The subagent's produced messages should only contain its task description,
    // NOT the parent's 3 conversation messages
    let subagent_messages = graph
        .neighbors(&subagent_id, Some(EdgeType::Produces), Direction::Outgoing)
        .unwrap();

    let user_messages: Vec<_> = subagent_messages
        .iter()
        .filter(|n| {
            matches!(&n.node_type, NodeType::Interaction(d) if d.role == "user")
        })
        .collect();

    // The subagent should have 1 user message (the task description),
    // NOT 3 (the parent's messages)
    assert_eq!(
        user_messages.len(),
        1,
        "Subagent should only see its task, not parent conversation (got {} user messages)",
        user_messages.len()
    );
}
