//! Re-pack a saved graph turn by turn. No model call.

use std::path::PathBuf;

use graphirm_agent::context::{ContextConfig, build_context};
use graphirm_graph::nodes::{GraphNode, NodeId, NodeType};
use graphirm_graph::{GraphEdge, GraphStore};
use graphirm_llm::ContentPart;

use crate::error::GraphirmError;

/// Schema characters measured on the eval tool list. They are outside the
/// packed nodes and are the same for every config.
const SCHEMA_CHARS: usize = 14_237;

/// Phase C actual input tokens were about 0.53 of the old characters/2
/// estimate, which is characters/3.8.
const CHARS_PER_TOKEN: f64 = 3.8;

/// `deepseek/deepseek-v4.1-flash` on OpenRouter, read 2026-10-11 from
/// `https://openrouter.ai/api/v1/models`.
const FLASH_IN: f64 = 0.0000003;
const FLASH_CACHE: f64 = 0.000000006;
const FLASH_OUT: f64 = 0.0000012;

pub struct ReplayArgs {
    pub graph: PathBuf,
    pub config: PathBuf,
    pub out: PathBuf,
    pub packing: String,
    pub no_dedup: bool,
    pub dedup: bool,
    pub excerpt_after: Option<usize>,
    pub context_max_tokens: Option<usize>,
    /// When false, each call's kept ids become the next call's `pack_prefix_ids`.
    pub stateless: bool,
}

pub fn run(args: ReplayArgs) -> Result<(), GraphirmError> {
    if args.dedup && args.no_dedup {
        return Err(GraphirmError::Config(
            "replay-pack accepts only one of --dedup and --no-dedup".into(),
        ));
    }
    let agent_config = graphirm_agent::AgentConfig::from_file(&args.config)
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let mut dedup = args.packing != "legacy";
    if args.dedup {
        dedup = true;
    }
    if args.no_dedup {
        dedup = false;
    }
    let mut context = ContextConfig {
        system_prompt: agent_config.system_prompt.clone(),
        max_tokens: args.context_max_tokens.unwrap_or(100_000),
        prefix_stable: args.packing != "legacy",
        dedup_derived: dedup,
        excerpt_after_turns: args.excerpt_after,
        recency_only: agent_config.recency_only,
        ..ContextConfig::default()
    };
    let mut pack_prefix: Vec<NodeId> = Vec::new();
    let dump_windows = std::env::var_os("GRAPHIRM_REPLAY_DUMP").is_some();
    let mut windows: Vec<String> = Vec::new();
    let source = GraphStore::open(args.graph.to_str().unwrap_or("graph.db"))
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let nodes = source
        .all_nodes()
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let edges = source
        .all_edges()
        .map_err(|e| GraphirmError::Config(e.to_string()))?;
    let agent_id = nodes
        .iter()
        .find(|node| matches!(node.node_type, NodeType::Agent(_)))
        .map(|node| node.id.clone())
        .ok_or_else(|| GraphirmError::Config("graph has no agent node".into()))?;
    let mut calls: Vec<&GraphNode> = nodes
        .iter()
        .filter(|node| {
            matches!(&node.node_type, NodeType::Interaction(data) if data.role == "assistant")
        })
        .collect();
    calls.sort_by_key(|node| node.created_at);

    let mut lines = vec![
        "call,input_tokens,prefix_share,duplicates,output_tokens,projected_cost_usd,chars_per_token"
            .to_string(),
    ];
    let mut previous = String::new();
    for (index, call) in calls.iter().enumerate() {
        let snapshot = snapshot_before(&nodes, &edges, call)?;
        if args.stateless {
            context.pack_prefix_ids.clear();
        } else {
            context.pack_prefix_ids = pack_prefix.clone();
        }
        let window = build_context(&snapshot, &agent_id, &context)
            .map_err(|e| GraphirmError::Config(e.to_string()))?;
        if !args.stateless {
            pack_prefix = window.kept_node_ids.clone();
        }
        let text = render_window(&window);
        let chars = text.chars().count() + SCHEMA_CHARS;
        let input_tokens = estimate_input_tokens(chars);
        let prefix_share = if previous.is_empty() {
            0.0
        } else {
            common_prefix(&previous, &text) as f64 / previous.chars().count() as f64
        };
        let output_tokens = call
            .metadata
            .get("usage_output")
            .and_then(|value| value.as_u64())
            .unwrap_or(0);
        let cached = if previous.is_empty() {
            0
        } else {
            ((prefix_share * input_tokens as f64).round() as u64).min(input_tokens)
        };
        let uncached = input_tokens.saturating_sub(cached);
        let cost = uncached as f64 * FLASH_IN
            + cached as f64 * FLASH_CACHE
            + output_tokens as f64 * FLASH_OUT;
        lines.push(format!(
            "{},{},{:.4},{},{},{:.6},{:.1}",
            index + 1,
            input_tokens,
            prefix_share,
            window.duplicate_derived,
            output_tokens,
            cost,
            CHARS_PER_TOKEN
        ));
        if dump_windows {
            windows.push(serde_json::json!({ "call": index + 1, "text": text }).to_string());
        }
        previous = text;
    }
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| GraphirmError::Config(format!("create {}: {e}", parent.display())))?;
    }
    if dump_windows {
        let dump_path = args.out.with_extension("windows.jsonl");
        std::fs::write(&dump_path, windows.join("\n") + "\n")
            .map_err(|e| GraphirmError::Config(format!("write {}: {e}", dump_path.display())))?;
    }
    std::fs::write(&args.out, lines.join("\n") + "\n")
        .map_err(|e| GraphirmError::Config(format!("write {}: {e}", args.out.display())))?;
    Ok(())
}

fn snapshot_before(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    before: &GraphNode,
) -> Result<GraphStore, GraphirmError> {
    let store = GraphStore::open_memory().map_err(|e| GraphirmError::Config(e.to_string()))?;
    let mut kept = std::collections::HashSet::<NodeId>::new();
    for node in nodes {
        if node.created_at < before.created_at {
            kept.insert(node.id.clone());
            store
                .add_node(node.clone())
                .map_err(|e| GraphirmError::Config(e.to_string()))?;
        }
    }
    for edge in edges {
        if kept.contains(&edge.source) && kept.contains(&edge.target) {
            store
                .add_edge(edge.clone())
                .map_err(|e| GraphirmError::Config(e.to_string()))?;
        }
    }
    Ok(store)
}

fn render_window(window: &graphirm_agent::context::ContextWindow) -> String {
    let mut parts = vec![part_text(&window.system)];
    for message in &window.messages {
        parts.push(part_text(message));
    }
    parts.join("\n")
}

fn part_text(message: &graphirm_llm::LlmMessage) -> String {
    message
        .content
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => text.clone(),
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => format!("{id} {name} {arguments}"),
            ContentPart::ToolResult { id, content, .. } => format!("{id}\n{content}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn estimate_input_tokens(chars: usize) -> u64 {
    (chars as f64 / CHARS_PER_TOKEN).round() as u64
}

fn common_prefix(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(a, b)| a == b)
        .count()
}

#[cfg(test)]
mod tests {
    use super::estimate_input_tokens;

    #[test]
    fn phase_c_divisor_turns_3800_chars_into_1000_tokens() {
        assert_eq!(estimate_input_tokens(3800), 1000);
    }
}
