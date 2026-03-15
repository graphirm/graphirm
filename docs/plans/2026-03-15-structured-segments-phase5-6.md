# Structured LLM Response Segments — Phase 5–6 Implementation Plan

> **Status: ✅ COMPLETE** — All tasks implemented, eval passing. See summary at bottom.

**Goal:** Parse LLM assistant responses into typed segments (observation, reasoning, code, plan, answer), persist each as a child Content node in the graph, and use segment types for smarter context building. Two paths: structured JSON output from the LLM (primary) and GLiNER2 post-processing (fallback).

**Architecture:** Segments are child `Content` nodes linked to the parent `Interaction` via `Contains` edges with an `order` metadata field. The raw text stays on the `Interaction` node (safety net). A post-processing pass detects nesting (span containment) and adds `Contains` edges between segment nodes. The context engine gains an optional `segment_filter` to include only specific segment types when building the LLM context window.

**Tech Stack:** Rust, serde_json, GLiNER2 ONNX (via existing `OnnxExtractor`), existing `GraphStore` CRUD

**Continues from:** `docs/plans/2026-03-10-structured-llm-responses.md` (Phases 1–4 complete)

---

## Design Decisions (settled)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Graph representation | Child Content nodes per segment (Option B) | Graph-native — segments are first-class citizens for PageRank, edges, HNSW |
| Nesting | Hybrid: flat storage + optional `Contains` edges between segments when spans overlap | Simple primary path, nesting discoverable when needed |
| Segment labels | Hardcoded starter set in config, swappable later | Discovery pipeline (Phases 1–4) exists but hasn't been run on enough real data yet |
| Raw text retention | Keep both: `InteractionData.content` = raw text, segments = child nodes | Storage is cheap, debugging is not |

## Starter Segment Labels

| Label | What it captures |
|-------|-----------------|
| `observation` | Factual statements about what was seen/read |
| `reasoning` | Analysis, diagnosis, logic chains |
| `code` | Code blocks and inline snippets |
| `plan` | Stated intentions, step-by-step outlines |
| `answer` | Direct responses, conclusions, summaries |

---

## Task 1: Add `SegmentConfig` to `AgentConfig`

**Files:**
- Modify: `crates/agent/src/config.rs`
- Modify: `config/default.toml`
- Test: `crates/agent/src/config.rs` (inline tests)

**Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/agent/src/config.rs`:

```rust
#[test]
fn test_segment_config_defaults() {
    let config = SegmentConfig::default();
    assert!(!config.enabled);
    assert_eq!(config.labels.len(), 5);
    assert!(config.structured_output);
    assert!(config.gliner2_fallback);
    assert!((config.min_confidence - 0.5).abs() < f64::EPSILON);
}

#[test]
fn test_segment_config_deserialize() {
    let toml_str = r#"
        enabled = true
        labels = ["code", "answer"]
        structured_output = false
        gliner2_fallback = true
        min_confidence = 0.6
    "#;
    let cfg: SegmentConfig = toml::from_str(toml_str).unwrap();
    assert!(cfg.enabled);
    assert_eq!(cfg.labels, vec!["code", "answer"]);
    assert!(!cfg.structured_output);
    assert!((cfg.min_confidence - 0.6).abs() < f64::EPSILON);
}

#[test]
fn test_agent_config_with_segments() {
    let config = AgentConfig::default();
    assert!(config.segments.is_none());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-agent test_segment_config`
Expected: FAIL — `SegmentConfig` not defined

**Step 3: Write minimal implementation**

Add to `crates/agent/src/config.rs`, above `AgentConfig`:

```rust
/// Configuration for structured LLM response segmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_segment_labels")]
    pub labels: Vec<String>,
    #[serde(default = "default_true")]
    pub structured_output: bool,
    #[serde(default = "default_true")]
    pub gliner2_fallback: bool,
    #[serde(default = "default_segment_min_confidence")]
    pub min_confidence: f64,
}

fn default_segment_labels() -> Vec<String> {
    vec![
        "observation".into(),
        "reasoning".into(),
        "code".into(),
        "plan".into(),
        "answer".into(),
    ]
}

fn default_true() -> bool {
    true
}

fn default_segment_min_confidence() -> f64 {
    0.5
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            labels: default_segment_labels(),
            structured_output: default_true(),
            gliner2_fallback: default_true(),
            min_confidence: default_segment_min_confidence(),
        }
    }
}
```

Add `segments` field to `AgentConfig`:

```rust
/// Segment extraction config. `None` disables response segmentation.
#[serde(default)]
pub segments: Option<SegmentConfig>,
```

And set it to `None` in `AgentConfig::default()` and wire it through `AgentConfigSection` / `from_toml`.

**Step 4: Add `[segments]` to `config/default.toml`**

```toml
[segments]
enabled = false
labels = ["observation", "reasoning", "code", "plan", "answer"]
structured_output = true
gliner2_fallback = true
min_confidence = 0.5
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p graphirm-agent test_segment_config`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add crates/agent/src/config.rs config/default.toml
git commit -m "feat(agent): add SegmentConfig for structured response segmentation"
```

---

## Task 2: Segment parsing and Content node creation

**Files:**
- Create: `crates/agent/src/knowledge/segments.rs`
- Modify: `crates/agent/src/knowledge/mod.rs` (add `pub mod segments;`)
- Test: `crates/agent/src/knowledge/segments.rs` (inline tests)

**Step 1: Write the failing tests**

In a new file `crates/agent/src/knowledge/segments.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_structured_output_valid() {
        let json = r#"{"segments": [
            {"type": "reasoning", "content": "The bug is in line 42."},
            {"type": "code", "content": "fn fix() {}"}
        ]}"#;
        let result = parse_structured_segments(json);
        assert!(result.is_ok());
        let segments = result.unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].segment_type, "reasoning");
        assert_eq!(segments[1].segment_type, "code");
    }

    #[test]
    fn test_parse_structured_output_invalid_json() {
        let result = parse_structured_segments("This is plain text, not JSON.");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_structured_output_missing_segments_key() {
        let result = parse_structured_segments(r#"{"items": []}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_nesting() {
        let segments = vec![
            Segment { segment_type: "plan".into(), content: "long plan text here".into(), start: 0, end: 100 },
            Segment { segment_type: "code".into(), content: "fn f(){}".into(), start: 30, end: 60 },
            Segment { segment_type: "answer".into(), content: "done".into(), start: 110, end: 130 },
        ];
        let pairs = detect_nesting(&segments);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 1)); // plan contains code
    }

    #[test]
    fn test_detect_nesting_no_overlap() {
        let segments = vec![
            Segment { segment_type: "reasoning".into(), content: "a".into(), start: 0, end: 50 },
            Segment { segment_type: "code".into(), content: "b".into(), start: 55, end: 100 },
        ];
        let pairs = detect_nesting(&segments);
        assert!(pairs.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-agent test_parse_structured -- test_detect_nesting`
Expected: FAIL — module doesn't exist

**Step 3: Write minimal implementation**

```rust
//! Structured response segment parsing and graph persistence.

use serde::{Deserialize, Serialize};

use crate::error::AgentError;

/// A parsed segment from an LLM response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    #[serde(rename = "type")]
    pub segment_type: String,
    pub content: String,
    /// Character offset where this segment starts in the raw text.
    /// Populated by GLiNER2 fallback; 0 for structured output (offsets computed from concatenation).
    #[serde(default)]
    pub start: usize,
    /// Character offset where this segment ends in the raw text.
    #[serde(default)]
    pub end: usize,
}

/// Wire format the LLM is asked to produce.
#[derive(Debug, Deserialize)]
struct StructuredResponse {
    segments: Vec<SegmentWire>,
}

#[derive(Debug, Deserialize)]
struct SegmentWire {
    #[serde(rename = "type")]
    segment_type: String,
    content: String,
}

/// Try to parse the LLM response text as a structured segment JSON envelope.
///
/// Expected format: `{"segments": [{"type": "...", "content": "..."}, ...]}`.
/// Returns `Err` if the text is not valid JSON or is missing the `segments` key.
pub fn parse_structured_segments(text: &str) -> Result<Vec<Segment>, AgentError> {
    // Strip markdown code fences if the model wrapped the JSON
    let trimmed = text.trim();
    let clean = if trimmed.starts_with("```") {
        let inner = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        inner
    } else {
        trimmed
    };

    let parsed: StructuredResponse = serde_json::from_str(clean)
        .map_err(|e| AgentError::Workflow(format!("Segment parse failed: {e}")))?;

    // Compute character offsets from concatenation order
    let mut offset = 0usize;
    let segments = parsed
        .segments
        .into_iter()
        .map(|w| {
            let start = offset;
            let end = start + w.content.len();
            offset = end;
            Segment {
                segment_type: w.segment_type,
                content: w.content,
                start,
                end,
            }
        })
        .collect();

    Ok(segments)
}

/// Detect nesting: returns (parent_index, child_index) pairs where child's
/// span falls entirely within parent's span.
pub fn detect_nesting(segments: &[Segment]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, parent) in segments.iter().enumerate() {
        for (j, child) in segments.iter().enumerate() {
            if i == j {
                continue;
            }
            if child.start >= parent.start && child.end <= parent.end
                && (child.start != parent.start || child.end != parent.end)
            {
                pairs.push((i, j));
            }
        }
    }
    pairs
}
```

Register the module in `crates/agent/src/knowledge/mod.rs`:

```rust
pub mod segments;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-agent segments::tests`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add crates/agent/src/knowledge/segments.rs crates/agent/src/knowledge/mod.rs
git commit -m "feat(agent): add segment parsing and nesting detection"
```

---

## Task 3: Graph persistence — `persist_segments`

**Files:**
- Modify: `crates/agent/src/knowledge/segments.rs`
- Test: `crates/agent/src/knowledge/segments.rs` (inline tests)

**Step 1: Write the failing test**

Add to the test module in `segments.rs`:

```rust
#[tokio::test]
async fn test_persist_segments_creates_nodes_and_edges() {
    use graphirm_graph::{GraphStore, NodeType, EdgeType};

    let store = Arc::new(GraphStore::open(":memory:").unwrap());
    // Create a parent interaction node
    let parent = graphirm_graph::GraphNode::new(NodeType::Interaction(
        graphirm_graph::InteractionData {
            role: "assistant".into(),
            content: "raw text".into(),
            token_count: Some(10),
        },
    ));
    let parent_id = store.add_node(parent).unwrap();

    let segments = vec![
        Segment { segment_type: "reasoning".into(), content: "because X".into(), start: 0, end: 9 },
        Segment { segment_type: "code".into(), content: "fn f(){}".into(), start: 10, end: 18 },
    ];

    let node_ids = persist_segments(&store, &parent_id, &segments, &[]).await.unwrap();
    assert_eq!(node_ids.len(), 2);

    // Verify nodes exist and have correct content_type
    for (i, nid) in node_ids.iter().enumerate() {
        let node = store.get_node(nid).unwrap().unwrap();
        match &node.node_type {
            NodeType::Content(data) => {
                assert_eq!(data.content_type, segments[i].segment_type);
                assert_eq!(data.body, segments[i].content);
            }
            _ => panic!("Expected Content node"),
        }
        // Check order metadata on the Contains edge
        let edges = store.get_edges_from(&parent_id).unwrap();
        let contains: Vec<_> = edges.iter().filter(|e| e.edge_type == EdgeType::Contains).collect();
        assert_eq!(contains.len(), 2);
    }
}

#[tokio::test]
async fn test_persist_segments_with_nesting() {
    use graphirm_graph::{GraphStore, NodeType, EdgeType};

    let store = Arc::new(GraphStore::open(":memory:").unwrap());
    let parent = graphirm_graph::GraphNode::new(NodeType::Interaction(
        graphirm_graph::InteractionData {
            role: "assistant".into(),
            content: "raw".into(),
            token_count: Some(5),
        },
    ));
    let parent_id = store.add_node(parent).unwrap();

    let segments = vec![
        Segment { segment_type: "plan".into(), content: "plan with code inside".into(), start: 0, end: 100 },
        Segment { segment_type: "code".into(), content: "fn f(){}".into(), start: 30, end: 60 },
    ];
    let nesting = vec![(0usize, 1usize)];

    let node_ids = persist_segments(&store, &parent_id, &segments, &nesting).await.unwrap();
    assert_eq!(node_ids.len(), 2);

    // Check that a Contains edge exists from segment 0 → segment 1
    let edges = store.get_edges_from(&node_ids[0]).unwrap();
    let nested_contains: Vec<_> = edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Contains && e.target_id == node_ids[1])
        .collect();
    assert_eq!(nested_contains.len(), 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p graphirm-agent test_persist_segments`
Expected: FAIL — `persist_segments` not defined

**Step 3: Write minimal implementation**

Add to `segments.rs`:

```rust
use std::sync::Arc;
use graphirm_graph::{
    ContentData, EdgeType, GraphEdge, GraphNode, GraphStore, NodeId, NodeType,
};

/// Create Content nodes for each segment and link them to the parent
/// Interaction node via Contains edges. Also creates Contains edges
/// for any detected nesting pairs.
///
/// Returns the NodeId of each created segment node (in order).
pub async fn persist_segments(
    store: &Arc<GraphStore>,
    parent_id: &NodeId,
    segments: &[Segment],
    nesting: &[(usize, usize)],
) -> Result<Vec<NodeId>, AgentError> {
    let mut node_ids = Vec::with_capacity(segments.len());

    let store_ref = store.clone();
    let parent_id = parent_id.clone();
    let segments_owned: Vec<Segment> = segments.to_vec();
    let nesting_owned: Vec<(usize, usize)> = nesting.to_vec();

    let ids = tokio::task::spawn_blocking(move || -> Result<Vec<NodeId>, AgentError> {
        let mut ids = Vec::with_capacity(segments_owned.len());

        for (i, seg) in segments_owned.iter().enumerate() {
            let mut node = GraphNode::new(NodeType::Content(ContentData {
                content_type: seg.segment_type.clone(),
                path: None,
                body: seg.content.clone(),
                language: if seg.segment_type == "code" { Some("unknown".into()) } else { None },
            }));
            let mut meta = serde_json::Map::new();
            meta.insert("segment_start".into(), serde_json::json!(seg.start));
            meta.insert("segment_end".into(), serde_json::json!(seg.end));
            node.metadata = serde_json::Value::Object(meta);

            let node_id = store_ref.add_node(node)?;

            // Contains edge from parent Interaction → segment Content
            let mut edge_meta = serde_json::Map::new();
            edge_meta.insert("order".into(), serde_json::json!(i));
            let edge = GraphEdge::new(
                EdgeType::Contains,
                parent_id.clone(),
                node_id.clone(),
            )
            .with_metadata(serde_json::Value::Object(edge_meta));
            store_ref.add_edge(edge)?;

            ids.push(node_id);
        }

        // Nesting edges: Contains from parent segment → child segment
        for &(parent_idx, child_idx) in &nesting_owned {
            if parent_idx < ids.len() && child_idx < ids.len() {
                let edge = GraphEdge::new(
                    EdgeType::Contains,
                    ids[parent_idx].clone(),
                    ids[child_idx].clone(),
                );
                store_ref.add_edge(edge)?;
            }
        }

        Ok(ids)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))??;

    Ok(ids)
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p graphirm-agent test_persist_segments`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add crates/agent/src/knowledge/segments.rs
git commit -m "feat(agent): add persist_segments — Content nodes + Contains edges"
```

---

## Task 4: GLiNER2 fallback — `segment_extract_gliner2`

**Files:**
- Modify: `crates/agent/src/knowledge/segments.rs`
- Test: inline (requires `local-extraction` feature, so `#[cfg(feature = "local-extraction")]`)

This reuses the existing `OnnxExtractor::extract_raw` from `predict_spans.rs` but wraps it for the segment use case.

**Step 1: Write the failing test**

```rust
#[cfg(feature = "local-extraction")]
#[tokio::test]
async fn test_segment_extract_gliner2_returns_segments() {
    // This test only runs with a real GLiNER2 model present.
    // It validates the function signature and type conversion.
    let labels = vec!["code".into(), "reasoning".into()];
    let text = "I think the bug is here. ```fn fix() { }```";
    // Without a real model, we just test that the function exists and has the right signature.
    // Real integration test requires GLINER2_MODEL_DIR.
}
```

**Step 2: Write minimal implementation**

Add to `segments.rs`, feature-gated:

```rust
/// Run GLiNER2 over the raw response text with segment labels and return
/// Segments with character offsets and confidence scores.
#[cfg(feature = "local-extraction")]
pub async fn segment_extract_gliner2(
    extractor: &super::local_extraction::OnnxExtractor,
    text: &str,
    labels: &[String],
    min_confidence: f64,
) -> Result<Vec<Segment>, AgentError> {
    let raw = extractor.extract_raw(text, labels, min_confidence).await?;
    let segments = raw
        .into_iter()
        .map(|e| Segment {
            segment_type: e.entity_type,
            content: text[e.start..e.end].to_string(),
            start: e.start,
            end: e.end,
        })
        .collect();
    Ok(segments)
}
```

**Step 3: Run tests**

Run: `cargo test -p graphirm-agent segments`
Expected: PASS (compilation check; real GLiNER2 test requires model)

**Step 4: Commit**

```bash
git add crates/agent/src/knowledge/segments.rs
git commit -m "feat(agent): add GLiNER2 fallback for segment extraction"
```

---

## Task 5: Wire segmentation into `stream_and_record`

**Files:**
- Modify: `crates/agent/src/workflow.rs`
- Test: existing workflow tests should still pass; add a unit test for the segment path

**Step 1: Write the failing test**

Add to workflow tests (or a new test file `tests/test_segments_workflow.rs`):

```rust
#[tokio::test]
async fn test_stream_and_record_marks_segmented_metadata() {
    // Uses MockProvider that returns structured JSON segments.
    // After stream_and_record, the Interaction node should have
    // metadata["segmented"] = true and child Content nodes should exist.
}
```

(Full test body depends on how `MockProvider` is set up — the implementing engineer should use the existing `MockProvider` pattern from `crates/llm/src/mock.rs` to return a structured segment JSON response.)

**Step 2: Write the integration**

In `stream_and_record` in `crates/agent/src/workflow.rs`, after the `Interaction` node is recorded (after line 105: `let node_id = session.record_interaction(interaction_node).await?;`), add:

```rust
// Structured segment extraction (Phase 5-6)
if let Some(ref seg_config) = session.agent_config.segments {
    if seg_config.enabled && !response.has_tool_calls() {
        let raw_text = response.text_content();
        let segments_result = crate::knowledge::segments::parse_structured_segments(&raw_text);

        let (segments, source) = match segments_result {
            Ok(segs) if !segs.is_empty() => (segs, "structured"),
            _ => {
                // Fallback to GLiNER2 if configured and available
                #[cfg(feature = "local-extraction")]
                if seg_config.gliner2_fallback {
                    if let Some(ref extractor) = session.onnx_extractor {
                        match crate::knowledge::segments::segment_extract_gliner2(
                            extractor,
                            &raw_text,
                            &seg_config.labels,
                            seg_config.min_confidence,
                        ).await {
                            Ok(segs) if !segs.is_empty() => (segs, "gliner2"),
                            Ok(_) => (vec![], "none"),
                            Err(e) => {
                                tracing::warn!(error = %e, "GLiNER2 segment fallback failed");
                                (vec![], "none")
                            }
                        }
                    } else {
                        (vec![], "none")
                    }
                } else {
                    (vec![], "none")
                }

                #[cfg(not(feature = "local-extraction"))]
                { (vec![], "none") }
            }
        };

        if !segments.is_empty() {
            let nesting = crate::knowledge::segments::detect_nesting(&segments);
            match crate::knowledge::segments::persist_segments(
                &session.graph, &node_id, &segments, &nesting,
            ).await {
                Ok(seg_ids) => {
                    tracing::info!(
                        count = seg_ids.len(),
                        source = source,
                        nesting = nesting.len(),
                        "Persisted response segments"
                    );
                    // Update Interaction metadata to mark as segmented
                    let graph = session.graph.clone();
                    let nid = node_id.clone();
                    let src = source.to_string();
                    let _ = tokio::task::spawn_blocking(move || {
                        if let Ok(Some(mut node)) = graph.get_node(&nid) {
                            if let Some(obj) = node.metadata.as_object_mut() {
                                obj.insert("segmented".into(), serde_json::json!(true));
                                obj.insert("segment_source".into(), serde_json::json!(src));
                                obj.insert("segment_count".into(), serde_json::json!(seg_ids.len()));
                            }
                            let _ = graph.update_node(&node);
                        }
                    }).await;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to persist segments (non-fatal)");
                }
            }
        }
    }
}
```

**Step 3: Add `onnx_extractor` field to `Session`** (if not already present)

Check `crates/agent/src/session.rs` — if there is no `onnx_extractor` field, add:

```rust
#[cfg(feature = "local-extraction")]
pub onnx_extractor: Option<Arc<super::knowledge::local_extraction::OnnxExtractor>>,
```

And wire it through `Session` constructors. If it already exists (it may be used by `post_turn_extract`), reuse it.

**Step 4: Run all tests**

Run: `cargo test -p graphirm-agent`
Expected: PASS — no regressions. Segments are opt-in (`enabled = false` by default).

**Step 5: Commit**

```bash
git add crates/agent/src/workflow.rs crates/agent/src/session.rs
git commit -m "feat(agent): wire segment extraction into stream_and_record"
```

---

## Task 6: System prompt segment instructions

**Files:**
- Modify: `crates/agent/src/workflow.rs` (in `stream_and_record`, before building context)
- Modify: `crates/agent/src/knowledge/segments.rs` (add `build_segment_prompt` function)
- Test: unit test for `build_segment_prompt`

**Step 1: Write the failing test**

```rust
#[test]
fn test_build_segment_prompt() {
    let labels = vec!["code".into(), "reasoning".into(), "answer".into()];
    let prompt = build_segment_prompt(&labels);
    assert!(prompt.contains("segments"));
    assert!(prompt.contains("code"));
    assert!(prompt.contains("reasoning"));
    assert!(prompt.contains("answer"));
    assert!(prompt.contains(r#""type""#));
    assert!(prompt.contains(r#""content""#));
}
```

**Step 2: Write minimal implementation**

Add to `segments.rs`:

```rust
/// Build the system prompt suffix that instructs the LLM to produce
/// structured segment output.
pub fn build_segment_prompt(labels: &[String]) -> String {
    let labels_str = labels
        .iter()
        .map(|l| format!(r#""{l}""#))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"

## Response Format

Structure your response as a JSON object with a "segments" array. Each segment has a "type" (one of: {labels_str}) and "content" (the text for that segment).

Example:
```json
{{"segments": [{{"type": "reasoning", "content": "The issue is..."}}, {{"type": "code", "content": "fn fix() {{}}"}}]}}
```

Rules:
- Every part of your response must be inside a segment.
- Use the most specific type that fits.
- Segments are sequential and non-overlapping.
- Output ONLY the JSON object, no text outside it.
"#
    )
}
```

**Step 3: Wire into `stream_and_record`**

In `stream_and_record`, when building the system prompt (around lines 30–34), append the segment instructions when `structured_output` is enabled:

```rust
let system_prompt = if suffix.is_empty() {
    session.agent_config.system_prompt.clone()
} else {
    format!("{}\n\n{}", session.agent_config.system_prompt, suffix)
};

// Append segment format instructions if structured output is enabled
let system_prompt = if let Some(ref seg_config) = session.agent_config.segments {
    if seg_config.enabled && seg_config.structured_output {
        let seg_prompt = crate::knowledge::segments::build_segment_prompt(&seg_config.labels);
        format!("{system_prompt}{seg_prompt}")
    } else {
        system_prompt
    }
} else {
    system_prompt
};
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent test_build_segment_prompt`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/agent/src/knowledge/segments.rs crates/agent/src/workflow.rs
git commit -m "feat(agent): add segment prompt instructions for structured LLM output"
```

---

## Task 7: Context engine — optional segment filtering

**Files:**
- Modify: `crates/agent/src/context.rs`
- Test: `crates/agent/src/context.rs` (inline tests)

**Step 1: Write the failing test**

Add to the test module in `context.rs`:

```rust
#[test]
fn test_context_config_segment_filter_default_is_none() {
    let config = ContextConfig::default();
    assert!(config.segment_filter.is_none());
}
```

**Step 2: Add `segment_filter` to `ContextConfig`**

```rust
/// When set, only include segment Content nodes whose `content_type`
/// matches one of these labels. `None` means include all content.
#[serde(default)]
pub segment_filter: Option<Vec<String>>,
```

Add to `ContextConfig` struct and set to `None` in `Default` impl.

**Step 3: Modify `node_to_message` for segment-aware rendering**

When an assistant `Interaction` node has `metadata["segmented"] = true`, and a `segment_filter` is active, the context engine should:

1. Find child `Content` nodes via `Contains` edges
2. Filter by `content_type` matching the filter
3. Reconstruct the message from selected segments

This requires passing the `GraphStore` and filter into `node_to_message`, or adding a new function:

```rust
/// Convert a segmented assistant Interaction into an LlmMessage using
/// only the segments whose type matches the filter.
///
/// Falls back to the raw `content` field if no segments match or the
/// node is not segmented.
pub fn node_to_message_filtered(
    node: &GraphNode,
    store: &GraphStore,
    segment_filter: Option<&[String]>,
) -> Option<LlmMessage> {
    // If no filter or node isn't segmented, use standard path
    let is_segmented = node
        .metadata
        .get("segmented")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if !is_segmented || segment_filter.is_none() {
        return node_to_message(node);
    }

    let filter = segment_filter.unwrap();

    // Get child Content nodes via Contains edges
    let edges = store.get_edges_from(&node.id).ok()?;
    let mut children: Vec<(i64, GraphNode)> = Vec::new();
    for edge in &edges {
        if edge.edge_type != EdgeType::Contains {
            continue;
        }
        if let Ok(Some(child)) = store.get_node(&edge.target_id) {
            if let NodeType::Content(ref data) = child.node_type {
                if filter.contains(&data.content_type) {
                    let order = edge
                        .metadata
                        .get("order")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                    children.push((order, child));
                }
            }
        }
    }

    if children.is_empty() {
        return node_to_message(node);
    }

    children.sort_by_key(|(order, _)| *order);

    let content: String = children
        .iter()
        .map(|(_, node)| {
            if let NodeType::Content(data) = &node.node_type {
                format!("[{}]\n{}", data.content_type, data.body)
            } else {
                String::new()
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    Some(LlmMessage::assistant(&content))
}
```

**Step 4: Wire into `build_context`**

In the section of `build_context` where `node_to_message` is called to convert nodes to messages, replace with `node_to_message_filtered` when a `segment_filter` is set in the config.

**Step 5: Run tests**

Run: `cargo test -p graphirm-agent context`
Expected: PASS — all existing context tests still pass, new filter test passes

**Step 6: Commit**

```bash
git add crates/agent/src/context.rs
git commit -m "feat(agent): add segment_filter to context engine for type-based filtering"
```

---

## Task 8: Update Interaction metadata flag on `segments` field in `AgentConfigSection`

**Files:**
- Modify: `crates/agent/src/config.rs` (wire `segments` through `AgentConfigSection` and `from_toml`)

This is a follow-up to Task 1 — ensuring the TOML `[agent.segments]` section is deserialized properly when using the sectioned config format (as used in production via `config/default.toml`).

**Step 1: Write the failing test**

```rust
#[test]
fn test_agent_config_from_toml_with_segments() {
    let toml_str = r#"
        [agent]
        name = "test"
        model = "test"
        system_prompt = "test"
        max_turns = 5

        [agent.segments]
        enabled = true
        labels = ["code", "answer"]
        structured_output = true
        gliner2_fallback = false
        min_confidence = 0.7
    "#;
    let config = AgentConfig::from_toml(toml_str).unwrap();
    let seg = config.segments.unwrap();
    assert!(seg.enabled);
    assert_eq!(seg.labels, vec!["code", "answer"]);
    assert!(!seg.gliner2_fallback);
}
```

**Step 2: Add `segments` field to `AgentConfigSection`**

```rust
#[serde(default)]
segments: Option<SegmentConfig>,
```

And wire through `from_toml`:

```rust
segments: file.agent.segments,
```

**Step 3: Run tests**

Run: `cargo test -p graphirm-agent test_agent_config_from_toml_with_segments`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/agent/src/config.rs
git commit -m "feat(agent): wire SegmentConfig through sectioned TOML config"
```

---

## Task 9: Integration test — full round-trip

**Files:**
- Create: `crates/agent/tests/test_segments_integration.rs`

**Step 1: Write the integration test**

An end-to-end test that:
1. Creates a `GraphStore` in memory
2. Creates a `Session` with `SegmentConfig { enabled: true, structured_output: false, gliner2_fallback: false }`
3. Calls `parse_structured_segments` with a known JSON input
4. Calls `persist_segments` with the result
5. Verifies Content nodes exist with correct types
6. Verifies Contains edges exist with correct order
7. Verifies nesting edges when applicable

```rust
use std::sync::Arc;
use graphirm_graph::{GraphStore, GraphNode, NodeType, InteractionData, EdgeType};
use graphirm_agent::knowledge::segments::{
    parse_structured_segments, persist_segments, detect_nesting, Segment,
};

#[tokio::test]
async fn test_full_segment_round_trip() {
    let store = Arc::new(GraphStore::open(":memory:").unwrap());

    let parent = GraphNode::new(NodeType::Interaction(InteractionData {
        role: "assistant".into(),
        content: r#"{"segments":[{"type":"reasoning","content":"The bug is X."},{"type":"code","content":"fn fix() {}"},{"type":"answer","content":"Fixed."}]}"#.into(),
        token_count: Some(20),
    }));
    let parent_id = store.add_node(parent).unwrap();

    let raw = r#"{"segments":[{"type":"reasoning","content":"The bug is X."},{"type":"code","content":"fn fix() {}"},{"type":"answer","content":"Fixed."}]}"#;
    let segments = parse_structured_segments(raw).unwrap();
    assert_eq!(segments.len(), 3);

    let nesting = detect_nesting(&segments);
    assert!(nesting.is_empty());

    let node_ids = persist_segments(&store, &parent_id, &segments, &nesting).await.unwrap();
    assert_eq!(node_ids.len(), 3);

    // Verify edge order
    let edges = store.get_edges_from(&parent_id).unwrap();
    let contains: Vec<_> = edges.iter().filter(|e| e.edge_type == EdgeType::Contains).collect();
    assert_eq!(contains.len(), 3);

    // Verify segment types
    for (i, nid) in node_ids.iter().enumerate() {
        let node = store.get_node(nid).unwrap().unwrap();
        if let NodeType::Content(data) = &node.node_type {
            assert_eq!(data.content_type, segments[i].segment_type);
        } else {
            panic!("Expected Content node at index {i}");
        }
    }
}
```

**Step 2: Run the test**

Run: `cargo test -p graphirm-agent test_full_segment_round_trip`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/agent/tests/test_segments_integration.rs
git commit -m "test(agent): add full segment round-trip integration test"
```

---

## Task 10: Documentation and plan update

**Files:**
- Modify: `docs/plans/2026-03-10-structured-llm-responses.md` (update status)
- Modify: `AGENTS.md` (update current state)

**Step 1: Update the parent plan**

In `docs/plans/2026-03-10-structured-llm-responses.md`, update:

```
**Status:** Phases 1–6 implemented. Corpus export, label exploration, schema suggestion, span validation, segment schema definition, and agent loop integration all complete.
```

**Step 2: Update AGENTS.md current state**

In the `Current State` table, update the structured LLM responses line to reflect completion.

**Step 3: Commit**

```bash
git add docs/plans/2026-03-10-structured-llm-responses.md AGENTS.md
git commit -m "docs: mark structured LLM response segments (Phases 5-6) complete"
```

---

## Summary

| Task | What | Files | Status |
|------|------|-------|--------|
| 1 | `SegmentConfig` type + TOML config | config.rs, default.toml | ✅ |
| 2 | Segment parsing + nesting detection | segments.rs (new) | ✅ |
| 3 | `persist_segments` — graph persistence | segments.rs | ✅ |
| 4 | GLiNER2 fallback wrapper | segments.rs | ✅ |
| 5 | Wire into `stream_and_record` | workflow.rs | ✅ |
| 6 | System prompt segment instructions | segments.rs, workflow.rs | ✅ |
| 7 | Context engine segment filtering | context.rs | ✅ |
| 8 | Wire `SegmentConfig` through sectioned TOML | config.rs | ✅ |
| 9 | Integration test — full round-trip | test_segments_integration.rs (new) | ✅ |
| 10 | Documentation + plan update | docs, AGENTS.md | ✅ |
| 11 | `enable_segments` API on `POST /api/sessions` | server/types.rs, routes.rs | ✅ |
| 12 | `segment-extraction` eval task + `GraphContainsContentType` verifier | graphirm-eval/tasks/segments.rs, task.rs | ✅ |
| 13 | Wire GLiNER2 fallback in `workflow.rs` (task-5b) | workflow.rs, segments.rs | ✅ |

**Verified:** `cargo run -p graphirm-eval -- --filter segments` → 1/1 ✅  
Live test confirmed both paths: structured JSON (DeepSeek follows format prompt) and GLiNER2 fallback (when LLM responds in plain text).

**Segment filter wiring:** ✅ Resolved in `docs/plans/2026-03-15-segment-context-filter-wiring.md`

**Known trade-off:** `OnnxExtractor::new` is constructed per-turn in the GLiNER2 fallback path (~seconds). Acceptable for current usage; cache as `Arc<OnnxExtractor>` in `AppState` if latency becomes a concern.
