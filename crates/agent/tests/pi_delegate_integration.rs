//! End-to-end: the agent loop calls `delegate_pi`, the fake `pi` replays the
//! recorded fixture, and the graph ends up in the `spawn_subagent` shape with
//! `executor: "pi"` nodes under the delegation Task.

#![cfg(unix)]

use std::sync::Arc;

use graphirm_agent::config::PiConfig;
use graphirm_agent::{
    AgentConfig, EventBus, HitlGate, Session, register_pi_delegate, run_agent_loop,
};
use graphirm_graph::edges::EdgeType;
use graphirm_graph::nodes::{GraphNode, NodeType, TaskStatus};
use graphirm_graph::{Direction, GraphStore};
use graphirm_llm::{LlmProvider, MockProvider, MockResponse};
use graphirm_tools::registry::ToolRegistry;
use tokio_util::sync::CancellationToken;

const FAKE_PI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pi/fake_pi.sh");

fn pi_config() -> PiConfig {
    PiConfig {
        enabled: true,
        binary: FAKE_PI.to_string(),
        extra_args: vec!["--fake-knob".to_string(), "FAKE_PI_DELAY_MS=1".to_string()],
        ..PiConfig::default()
    }
}

fn interactions<'a>(nodes: &'a [GraphNode], role: &str) -> Vec<&'a GraphNode> {
    nodes
        .iter()
        .filter(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == role))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn agent_loop_delegates_to_pi_and_records_the_run() {
    let workspace = tempfile::TempDir::new().expect("tempdir");
    let graph = Arc::new(GraphStore::open_memory().expect("graph"));
    let config = AgentConfig {
        working_dir: workspace.path().to_path_buf(),
        pi: Some(pi_config()),
        tool_gate_enabled: false,
        pre_completion_verify: false,
        max_turns: 5,
        ..AgentConfig::default()
    };

    let mut tools = ToolRegistry::new();
    register_pi_delegate(&mut tools, &config).await;
    assert!(
        tools.get("delegate_pi").is_ok(),
        "tool registered when enabled"
    );

    // `delegate_pi` is destructive: with a HITL gate in auto-approve mode it
    // runs without a confirm card, like `bash` would.
    let gate = Arc::new(HitlGate::new());
    gate.set_auto_approve(true);
    let session = Session::new(graph.clone(), config)
        .expect("session")
        .with_hitl(gate);
    assert!(
        session.agent_config.system_prompt.contains("delegate_pi"),
        "system-prompt notice applied when pi.enabled"
    );
    session
        .add_user_message("Create hello.txt containing hi")
        .await
        .expect("user message");

    let provider: Arc<dyn LlmProvider> = Arc::new(MockProvider::new(vec![
        MockResponse::tool_call(
            "tc_pi_1",
            "delegate_pi",
            serde_json::json!({ "task": "Create hello.txt containing exactly 'hi'" }),
        ),
        MockResponse::text("Pi created hello.txt; I verified it contains `hi`."),
    ]));
    let events = EventBus::new();
    let cancel = CancellationToken::new();

    run_agent_loop(&session, provider, &tools, &events, &cancel)
        .await
        .expect("agent loop");

    // Director --DelegatesTo--> Task(Completed, executor pi)
    let tasks = graph
        .neighbors(
            &session.id,
            Some(EdgeType::DelegatesTo),
            Direction::Outgoing,
        )
        .expect("delegated");
    assert_eq!(tasks.len(), 1, "one delegation Task");
    let task = &tasks[0];
    match &task.node_type {
        NodeType::Task(data) => {
            assert_eq!(data.status, TaskStatus::Completed);
            assert_eq!(data.title, "Delegated to pi");
        }
        other => panic!("expected Task, got {}", other.type_name()),
    }
    assert_eq!(task.metadata["executor"], "pi");
    assert_eq!(task.metadata["result"], "The file content is: `hi`");
    assert_eq!(task.metadata["exit_code"], 0);
    assert_eq!(task.metadata["tool_calls"], 4);
    assert!(task.metadata.get("binary").is_none());

    // Parent assistant Interaction --Produces--> Task
    let producers = graph
        .neighbors(&task.id, Some(EdgeType::Produces), Direction::Incoming)
        .expect("producers");
    assert!(
        producers
            .iter()
            .any(|n| matches!(&n.node_type, NodeType::Interaction(d) if d.role == "assistant")),
        "the assistant turn that called delegate_pi produces the Task: {producers:?}"
    );

    // Task --SpawnedBy--> Pi Agent --Produces--> tool / assistant nodes
    let agents = graph
        .neighbors(&task.id, Some(EdgeType::SpawnedBy), Direction::Outgoing)
        .expect("spawned");
    assert_eq!(agents.len(), 1);
    match &agents[0].node_type {
        NodeType::Agent(a) => {
            assert_eq!(a.name, "pi");
            assert_eq!(a.status, "completed");
        }
        other => panic!("expected Agent, got {}", other.type_name()),
    }
    assert_eq!(agents[0].metadata["pi_version"], "0.85.1-fake");
    let pi_nodes = graph
        .neighbors(&agents[0].id, Some(EdgeType::Produces), Direction::Outgoing)
        .expect("pi nodes");
    let tool_nodes = interactions(&pi_nodes, "tool");
    assert_eq!(tool_nodes.len(), 4, "{pi_nodes:?}");
    assert!(tool_nodes.iter().all(|n| n.metadata["executor"] == "pi"));
    assert!(
        tool_nodes
            .iter()
            .all(|n| n.metadata["session_id"] == agents[0].id.to_string())
    );
    assert!(!interactions(&pi_nodes, "assistant").is_empty());

    // The director's own turn: a `delegate_pi` tool-result node carrying the
    // summary, then the final assistant message.
    let recent = graph.list_recent_nodes(200).expect("recent");
    let director_tool = recent
        .iter()
        .find(|n| n.metadata["tool_name"] == "delegate_pi")
        .expect("director's delegate_pi tool node");
    if let NodeType::Interaction(d) = &director_tool.node_type {
        assert_eq!(d.role, "tool");
        assert!(
            d.content.starts_with("Pi completed (exit 0, "),
            "{}",
            d.content
        );
        assert!(
            d.content.contains("Tool calls: 4 (1 errors)"),
            "{}",
            d.content
        );
    } else {
        panic!("expected Interaction");
    }
    assert_eq!(director_tool.metadata["is_error"], false);
    assert!(
        recent.iter().any(|n| matches!(
            &n.node_type,
            NodeType::Interaction(d)
                if d.role == "assistant" && d.content.contains("I verified it contains")
        )),
        "final assistant message recorded"
    );
}
