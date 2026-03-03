use std::sync::Arc;

use graphirm_graph::GraphStore;
use graphirm_graph::nodes::{AgentData, GraphNode, InteractionData, NodeType};
use graphirm_tools::{
    ToolCall, ToolContext,
    bash::BashTool,
    edit::EditTool,
    executor::{execute_parallel, execute_single},
    find::FindTool,
    grep::GrepTool,
    ls::LsTool,
    read::ReadTool,
    registry::ToolRegistry,
    write::WriteTool,
};
use serde_json::json;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

fn setup() -> (TempDir, ToolRegistry, ToolContext) {
    let dir = TempDir::new().unwrap();
    let graph = Arc::new(GraphStore::open_memory().expect("memory graph"));

    let agent_id = graph
        .add_node(GraphNode::new(NodeType::Agent(AgentData {
            name: "test-agent".to_string(),
            model: "test".to_string(),
            system_prompt: None,
            status: "active".to_string(),
        })))
        .unwrap();

    let interaction_id = graph
        .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
            role: "user".to_string(),
            content: "test".to_string(),
            token_count: None,
        })))
        .unwrap();

    let ctx = ToolContext {
        graph,
        agent_id,
        interaction_id,
        working_dir: dir.path().to_path_buf(),
        signal: CancellationToken::new(),
    };

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(ReadTool::new()));
    registry.register(Arc::new(WriteTool::new()));
    registry.register(Arc::new(EditTool::new()));
    registry.register(Arc::new(BashTool::new()));
    registry.register(Arc::new(GrepTool::new()));
    registry.register(Arc::new(FindTool::new()));
    registry.register(Arc::new(LsTool::new()));

    (dir, registry, ctx)
}

#[tokio::test]
async fn full_workflow_sequence() {
    let (dir, registry, ctx) = setup();

    // write
    let write_call = ToolCall {
        id: "w1".into(),
        name: "write".into(),
        arguments: json!({"path": "workflow.txt", "content": "initial content here"}),
    };
    let written = execute_single(&registry, &write_call, &ctx).await.unwrap();
    assert!(!written.is_error);
    assert!(written.content.contains("created"));

    // read
    let read_call = ToolCall {
        id: "r1".into(),
        name: "read".into(),
        arguments: json!({"path": "workflow.txt"}),
    };
    let read_out = execute_single(&registry, &read_call, &ctx).await.unwrap();
    assert!(!read_out.is_error);
    assert!(read_out.content.contains("initial content here"));

    // edit
    let edit_call = ToolCall {
        id: "e1".into(),
        name: "edit".into(),
        arguments: json!({
            "path": "workflow.txt",
            "old_string": "initial content here",
            "new_string": "updated content here"
        }),
    };
    let edited = execute_single(&registry, &edit_call, &ctx).await.unwrap();
    assert!(!edited.is_error);

    // bash
    let bash_call = ToolCall {
        id: "b1".into(),
        name: "bash".into(),
        arguments: json!({"command": "cat workflow.txt"}),
    };
    let bash_out = execute_single(&registry, &bash_call, &ctx).await.unwrap();
    assert!(!bash_out.is_error);
    assert!(bash_out.content.contains("updated content here"));

    // ls
    let ls_call = ToolCall {
        id: "l1".into(),
        name: "ls".into(),
        arguments: json!({}),
    };
    let ls_out = execute_single(&registry, &ls_call, &ctx).await.unwrap();
    assert!(!ls_out.is_error);
    assert!(ls_out.content.contains("workflow.txt"));

    // find
    let find_call = ToolCall {
        id: "f1".into(),
        name: "find".into(),
        arguments: json!({"pattern": "*.txt"}),
    };
    let find_out = execute_single(&registry, &find_call, &ctx).await.unwrap();
    assert!(!find_out.is_error);
    assert!(find_out.content.contains("workflow.txt"));

    // grep
    let grep_call = ToolCall {
        id: "g1".into(),
        name: "grep".into(),
        arguments: json!({"pattern": "updated"}),
    };
    let grep_out = execute_single(&registry, &grep_call, &ctx).await.unwrap();
    assert!(!grep_out.is_error);
    assert!(grep_out.content.contains("updated"));

    // Verify file on disk
    let disk_content = std::fs::read_to_string(dir.path().join("workflow.txt")).unwrap();
    assert_eq!(disk_content, "updated content here");
}

#[tokio::test]
async fn parallel_reads() {
    let (dir, registry, ctx) = setup();

    // Create 3 files
    for i in 1..=3 {
        std::fs::write(
            dir.path().join(format!("file{i}.txt")),
            format!("content {i}"),
        )
        .unwrap();
    }

    let calls: Vec<ToolCall> = (1..=3)
        .map(|i| ToolCall {
            id: format!("r{i}"),
            name: "read".into(),
            arguments: json!({"path": format!("file{i}.txt")}),
        })
        .collect();

    let results = execute_parallel(&registry, calls, &ctx).await;
    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(r.result.is_ok(), "read failed: {:?}", r.call_id);
    }
}

#[tokio::test]
async fn registry_lists_all_tools() {
    let (_dir, registry, _ctx) = setup();
    let names = registry.list();
    assert_eq!(names.len(), 7, "expected 7 tools, got: {:?}", names);
    assert!(names.contains(&"read"));
    assert!(names.contains(&"write"));
    assert!(names.contains(&"edit"));
    assert!(names.contains(&"bash"));
    assert!(names.contains(&"grep"));
    assert!(names.contains(&"find"));
    assert!(names.contains(&"ls"));
}

#[tokio::test]
async fn definitions_for_llm() {
    let (_dir, registry, _ctx) = setup();
    let defs = registry.definitions();
    assert_eq!(defs.len(), 7);
    for def in &defs {
        assert!(!def.name.is_empty(), "tool name should not be empty");
        assert!(
            !def.description.is_empty(),
            "tool description should not be empty"
        );
        assert!(
            def.parameters.is_object(),
            "tool parameters should be a JSON object for tool: {}",
            def.name
        );
    }
}

#[tokio::test]
async fn graph_trail_after_workflow() {
    let (dir, registry, ctx) = setup();

    let initial_count = {
        // Count how many content nodes exist (2 from setup: agent + interaction)
        // We can't easily count all nodes, so we track how many we add
        0usize
    };

    std::fs::write(dir.path().join("tracked.txt"), "track me").unwrap();

    let write_call = ToolCall {
        id: "w".into(),
        name: "write".into(),
        arguments: json!({"path": "new.txt", "content": "data"}),
    };
    let write_out = execute_single(&registry, &write_call, &ctx).await.unwrap();
    assert!(
        write_out.node_id.is_some(),
        "write should produce a graph node"
    );

    let read_call = ToolCall {
        id: "r".into(),
        name: "read".into(),
        arguments: json!({"path": "tracked.txt"}),
    };
    let read_out = execute_single(&registry, &read_call, &ctx).await.unwrap();
    assert!(
        read_out.node_id.is_some(),
        "read should produce a graph node"
    );

    // Verify the nodes exist in the graph
    let write_node_id = write_out.node_id.unwrap();
    let read_node_id = read_out.node_id.unwrap();

    let write_node = ctx.graph.get_node(&write_node_id).unwrap();
    let read_node = ctx.graph.get_node(&read_node_id).unwrap();

    use graphirm_graph::nodes::NodeType;
    assert!(matches!(write_node.node_type, NodeType::Content(_)));
    assert!(matches!(read_node.node_type, NodeType::Content(_)));

    let _ = initial_count;
}
