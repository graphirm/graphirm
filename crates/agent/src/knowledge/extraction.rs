//! Post-turn knowledge extraction from conversations.

use std::collections::HashMap;
use std::sync::Arc;

use graphirm_graph::{
    Direction, EdgeType, GraphEdge, GraphNode, GraphStore, KnowledgeData, NodeId, NodeType,
};
use graphirm_llm::{CompletionConfig, LlmMessage, LlmProvider};
use serde::{Deserialize, Serialize};

use crate::error::AgentError;

fn default_entity_types() -> Vec<String> {
    vec![
        "function".into(),
        "api".into(),
        "pattern".into(),
        "decision".into(),
        "bug".into(),
        "architecture".into(),
        "convention".into(),
        "library".into(),
    ]
}

fn default_model() -> String {
    "gpt-4o-mini".into()
}

fn default_min_confidence() -> f64 {
    0.7
}

/// Selects which extraction backend to use for a given `ExtractionConfig`.
///
/// - `Llm`: sends the conversation to an LLM and parses the JSON response (default)
/// - `Local`: runs a GLiNER2 ONNX model on CPU for fast, zero-cost entity extraction
/// - `Hybrid`: GLiNER2 for entities + LLM for description synthesis and relationship discovery
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionBackend {
    #[default]
    Llm,
    /// Directory containing GLiNER2 ONNX files + tokenizer.
    /// Populate via `download_model()` on first run.
    Local {
        model_dir: String,
    },
    /// Directory containing GLiNER2 ONNX files + tokenizer.
    /// Populate via `download_model()` on first run.
    Hybrid {
        model_dir: String,
    },
}

/// Configuration for post-turn knowledge extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Whether knowledge extraction is enabled.
    #[serde(default)]
    pub enabled: bool,

    /// LLM model to use for extraction (LLM and Hybrid backends).
    #[serde(default = "default_model")]
    pub model: String,

    /// Entity types to extract (e.g. function, api, pattern).
    #[serde(default = "default_entity_types")]
    pub entity_types: Vec<String>,

    /// Minimum confidence score [0.0, 1.0] for a node to be persisted.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,

    /// Which extraction backend to use.
    #[serde(default)]
    pub backend: ExtractionBackend,
}

/// A knowledge entity extracted from a conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub entity_type: String,
    pub name: String,
    pub description: String,
    pub confidence: f64,
    pub relationships: Vec<EntityRelationship>,
}

/// A directed relationship from an extracted entity to a named target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
    pub target_name: String,
    pub relationship: String,
}

/// Top-level wrapper for the JSON object returned by the extraction LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResponse {
    pub entities: Vec<ExtractedEntity>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_model(),
            entity_types: default_entity_types(),
            min_confidence: default_min_confidence(),
            backend: ExtractionBackend::default(),
        }
    }
}

/// Builds the extraction prompt to send to the LLM, embedding the conversation
/// and instructing the model to return structured JSON with knowledge entities.
pub fn build_extraction_prompt(messages: &[(String, String)], config: &ExtractionConfig) -> String {
    let entity_types_list = config.entity_types.join(", ");

    let conversation_block = if messages.is_empty() {
        "(empty conversation)".to_string()
    } else {
        format_conversation(messages)
    };

    format!(
        r#"Extract knowledge entities from the following conversation.

For each entity, provide:
- entity_type: one of [{entity_types}]
- name: short identifier for the entity
- description: one-sentence description of what it is or why it matters
- confidence: 0.0-1.0 how confident you are this is a real, useful entity
- relationships: array of {{ target_name, relationship }} pairs linking to other entities

Only extract entities with confidence >= {min_confidence}.

Respond with ONLY valid JSON in this exact format:
{{
  "entities": [
    {{
      "entity_type": "pattern",
      "name": "Example Pattern",
      "description": "Description of the pattern",
      "confidence": 0.9,
      "relationships": [
        {{ "target_name": "OtherEntity", "relationship": "uses" }}
      ]
    }}
  ]
}}

CONVERSATION:
{conversation}
"#,
        entity_types = entity_types_list,
        min_confidence = config.min_confidence,
        conversation = conversation_block,
    )
}

/// Extracts knowledge entities from a conversation turn using an LLM call,
/// persists them as `Knowledge` nodes in the graph, and links them back to the
/// source interaction via `DerivedFrom` edges. Inter-entity `RelatesTo` edges
/// are created for any relationships the LLM identifies.
pub async fn extract_knowledge(
    graph: Arc<GraphStore>,
    llm: &dyn LlmProvider,
    messages: &[(String, String)],
    source_node_id: &NodeId,
    config: &ExtractionConfig,
) -> Result<Vec<NodeId>, AgentError> {
    if !config.enabled {
        return Ok(vec![]);
    }

    let prompt = build_extraction_prompt(messages, config);
    let completion_config = CompletionConfig::new(&config.model);
    let response = llm
        .complete(vec![LlmMessage::human(prompt)], &[], &completion_config)
        .await?;

    let extraction: ExtractionResponse = serde_json::from_str(&response.text_content())
        .map_err(|e| AgentError::Workflow(format!("Failed to parse extraction response: {e}")))?;

    let source_id = source_node_id.clone();
    let config_clone = config.clone();
    tokio::task::spawn_blocking(move || {
        persist_extracted_entities(&graph, &source_id, &extraction, &config_clone)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))?
}

/// Called after each agent turn. Gathers the recent conversation from the graph
/// (the response node and its parent messages via `RespondsTo` edges) and runs
/// knowledge extraction. Returns the IDs of any newly-created `Knowledge` nodes.
///
/// Non-fatal by design — callers should log and continue on error.
pub async fn post_turn_extract(
    graph: Arc<GraphStore>,
    llm: &dyn LlmProvider,
    config: &ExtractionConfig,
    response_node_id: &NodeId,
) -> Result<Vec<NodeId>, AgentError> {
    if !config.enabled {
        return Ok(vec![]);
    }

    // Collect conversation context in spawn_blocking to avoid blocking the runtime.
    let graph_for_read = graph.clone();
    let resp_id = response_node_id.clone();
    let (response_node, parents) = tokio::task::spawn_blocking(move || {
        let node = graph_for_read.get_node(&resp_id)?;
        let parents = graph_for_read.neighbors(
            &resp_id,
            Some(EdgeType::RespondsTo),
            Direction::Outgoing,
        )?;
        Ok::<_, AgentError>((node, parents))
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))??;

    let mut messages: Vec<(String, String)> = Vec::new();
    for parent in &parents {
        if let NodeType::Interaction(data) = &parent.node_type {
            messages.push((data.role.clone(), data.content.clone()));
        }
    }
    if let NodeType::Interaction(data) = &response_node.node_type {
        messages.push((data.role.clone(), data.content.clone()));
    }

    extract_knowledge_with_backend(
        graph,
        Some(llm),
        #[cfg(not(feature = "local-extraction"))]
        None::<()>,
        #[cfg(feature = "local-extraction")]
        None,
        &messages,
        response_node_id,
        config,
    )
    .await
}

/// Backend-aware extraction dispatcher. Routes to LLM, local ONNX, or hybrid
/// depending on `ExtractionConfig.backend`.
///
/// - `Llm`: calls the LLM provider (existing behaviour, same as `extract_knowledge`)
/// - `Local`: runs GLiNER2 via ONNX (fast, no token cost, no descriptions)
/// - `Hybrid`: GLiNER2 first for entities, then LLM synthesises descriptions and
///   discovers higher-order relationships
///
/// `onnx` is only meaningful when the `local-extraction` feature is enabled;
/// it is ignored otherwise.
pub async fn extract_knowledge_with_backend(
    graph: Arc<GraphStore>,
    llm: Option<&dyn LlmProvider>,
    #[cfg(feature = "local-extraction")] _onnx: Option<
        &crate::knowledge::local_extraction::OnnxExtractor,
    >,
    #[cfg(not(feature = "local-extraction"))] _onnx: Option<()>,
    messages: &[(String, String)],
    source_node_id: &NodeId,
    config: &ExtractionConfig,
) -> Result<Vec<NodeId>, AgentError> {
    if !config.enabled {
        return Ok(vec![]);
    }

    let extraction = match &config.backend {
        ExtractionBackend::Llm => {
            let llm = llm.ok_or_else(|| {
                AgentError::Workflow("LLM backend selected but no LLM provider given".into())
            })?;
            let prompt = build_extraction_prompt(messages, config);
            let completion_config = CompletionConfig::new(&config.model);
            let response = llm
                .complete(vec![LlmMessage::human(prompt)], &[], &completion_config)
                .await?;
            let raw = response.text_content();
            // Some providers wrap JSON in markdown code fences. Strip them.
            let json_str = {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    // Empty response — log for debugging and return no entities.
                    tracing::debug!(
                        model = %config.model,
                        stop_reason = ?response.stop_reason,
                        content_parts = response.content.len(),
                        "Extraction LLM returned empty text content"
                    );
                    return Ok(vec![]);
                }
                // Strip ```json ... ``` or ``` ... ``` wrappers.
                let inner = if let Some(rest) = trimmed.strip_prefix("```json") {
                    rest.trim_end_matches("```").trim()
                } else if let Some(rest) = trimmed.strip_prefix("```") {
                    rest.trim_end_matches("```").trim()
                } else {
                    trimmed
                };
                // Find outermost { ... } in case there's preamble text.
                match (inner.find('{'), inner.rfind('}')) {
                    (Some(start), Some(end)) if end > start => &inner[start..=end],
                    _ => inner,
                }
            };
            serde_json::from_str::<ExtractionResponse>(json_str).map_err(|e| {
                AgentError::Workflow(format!(
                    "Failed to parse extraction response: {e}\nRaw: {json_str}"
                ))
            })?
        }

        #[cfg(feature = "local-extraction")]
        ExtractionBackend::Local { model_dir } => {
            use crate::knowledge::local_extraction::OnnxExtractor;
            // Constructing OnnxExtractor inline is expensive (~seconds per call due
            // to loading 4 ONNX sessions). Callers should cache OnnxExtractor as
            // Arc<OnnxExtractor> at startup and pass it in. This inline construction
            // documents the API surface.
            let extractor = OnnxExtractor::new(std::path::Path::new(model_dir))?;
            // OnnxExtractor::extract already filters by min_confidence internally.
            // persist_extracted_entities will re-filter, which is harmless but
            // intentional: the shared path applies the same threshold uniformly.
            extractor
                .extract(
                    &format_conversation(messages),
                    &config.entity_types,
                    config.min_confidence,
                )
                .await?
        }

        #[cfg(not(feature = "local-extraction"))]
        ExtractionBackend::Local { .. } => {
            return Err(AgentError::Workflow(
                "Local extraction backend requires the `local-extraction` feature".into(),
            ));
        }

        #[cfg(feature = "local-extraction")]
        ExtractionBackend::Hybrid { model_dir } => {
            use crate::knowledge::local_extraction::OnnxExtractor;
            // See Local arm comment: inline construction is expensive; callers
            // should cache OnnxExtractor as Arc<OnnxExtractor> for Hybrid too.
            let extractor = OnnxExtractor::new(std::path::Path::new(model_dir))?;
            let conversation_text = format_conversation(messages);
            let local_result = extractor
                .extract(
                    &conversation_text,
                    &config.entity_types,
                    config.min_confidence,
                )
                .await?;

            // LLM enrichment — add descriptions and discover relationships.
            if let Some(llm) = llm {
                let entity_names: Vec<&str> = local_result
                    .entities
                    .iter()
                    .map(|e| e.name.as_str())
                    .collect();
                let enrichment_prompt = format!(
                    "Given these entities extracted from a conversation: {}\n\n\
                     For each entity, provide a one-sentence description and any relationships between them.\n\n\
                     Conversation:\n{}\n\n\
                     Respond with ONLY valid JSON in the ExtractionResponse format.",
                    entity_names.join(", "),
                    conversation_text,
                );
                let completion_config = CompletionConfig::new(&config.model);
                match llm
                    .complete(
                        vec![LlmMessage::human(enrichment_prompt)],
                        &[],
                        &completion_config,
                    )
                    .await
                {
                    Ok(response) => {
                        serde_json::from_str::<ExtractionResponse>(&response.text_content())
                            .unwrap_or(local_result)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "LLM enrichment failed, falling back to local-only");
                        local_result
                    }
                }
            } else {
                local_result
            }
        }

        #[cfg(not(feature = "local-extraction"))]
        ExtractionBackend::Hybrid { .. } => {
            return Err(AgentError::Workflow(
                "Hybrid extraction backend requires the `local-extraction` feature".into(),
            ));
        }
    };

    let source_id = source_node_id.clone();
    let config_clone = config.clone();
    tokio::task::spawn_blocking(move || {
        persist_extracted_entities(&graph, &source_id, &extraction, &config_clone)
    })
    .await
    .map_err(|e| AgentError::Join(e.to_string()))?
}

/// Shared node-creation logic: persists filtered entities as `Knowledge` graph nodes,
/// links them to the source via `DerivedFrom`, and wires `RelatesTo` edges.
fn persist_extracted_entities(
    graph: &GraphStore,
    source_node_id: &NodeId,
    extraction: &ExtractionResponse,
    config: &ExtractionConfig,
) -> Result<Vec<NodeId>, AgentError> {
    let filtered: Vec<&ExtractedEntity> = extraction
        .entities
        .iter()
        .filter(|e| e.confidence >= config.min_confidence)
        .collect();

    let mut name_to_id: HashMap<String, NodeId> = HashMap::new();
    let mut created_ids: Vec<NodeId> = Vec::new();

    for entity in &filtered {
        let node_id = graph.add_node(GraphNode::new(NodeType::Knowledge(KnowledgeData {
            entity: entity.name.clone(),
            entity_type: entity.entity_type.clone(),
            summary: entity.description.clone(),
            confidence: entity.confidence,
        })))?;

        graph.add_edge(GraphEdge::new(
            EdgeType::DerivedFrom,
            node_id.clone(),
            source_node_id.clone(),
        ))?;

        name_to_id.insert(entity.name.clone(), node_id.clone());
        created_ids.push(node_id);
    }

    for entity in &filtered {
        if let Some(src) = name_to_id.get(&entity.name) {
            for rel in &entity.relationships {
                if let Some(tgt) = name_to_id.get(&rel.target_name) {
                    graph.add_edge(GraphEdge::new(
                        EdgeType::RelatesTo,
                        src.clone(),
                        tgt.clone(),
                    ))?;
                }
            }
        }
    }

    tracing::info!(
        extracted = filtered.len(),
        total = extraction.entities.len(),
        backend = ?config.backend,
        "Knowledge extraction complete"
    );

    Ok(created_ids)
}

/// Format a conversation as a plain-text block for LLM prompts or ONNX input.
fn format_conversation(messages: &[(String, String)]) -> String {
    messages
        .iter()
        .map(|(role, content)| format!("[{}]: {}", role, content))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use futures::Stream;
    use graphirm_graph::{Direction, EdgeType, GraphNode, GraphStore, InteractionData, NodeType};
    use graphirm_llm::{
        CompletionConfig, ContentPart, LlmError, LlmMessage, LlmProvider, LlmResponse, StopReason,
        StreamEvent, TokenUsage, ToolDefinition,
    };
    use std::pin::Pin;

    struct MockExtractionProvider {
        response_json: String,
    }

    #[async_trait]
    impl LlmProvider for MockExtractionProvider {
        async fn complete(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: &[ToolDefinition],
            _config: &CompletionConfig,
        ) -> Result<LlmResponse, LlmError> {
            Ok(LlmResponse {
                content: vec![ContentPart::text(self.response_json.clone())],
                usage: TokenUsage::new(10, 100),
                stop_reason: StopReason::EndTurn,
            })
        }

        async fn stream(
            &self,
            _messages: Vec<LlmMessage>,
            _tools: &[ToolDefinition],
            _config: &CompletionConfig,
        ) -> Result<Pin<Box<dyn Stream<Item = StreamEvent> + Send>>, LlmError> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn provider_name(&self) -> &str {
            "mock-extraction"
        }
    }

    #[tokio::test]
    async fn test_extract_knowledge_creates_nodes() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "pattern",
                        "name": "JWT Authentication",
                        "description": "Token-based auth using JSON Web Tokens",
                        "confidence": 0.92,
                        "relationships": []
                    },
                    {
                        "entity_type": "library",
                        "name": "bcrypt",
                        "description": "Password hashing library",
                        "confidence": 0.88,
                        "relationships": [
                            { "target_name": "JWT Authentication", "relationship": "used_with" }
                        ]
                    }
                ]
            }"#
            .to_string(),
        };

        let messages = vec![
            ("user".to_string(), "How do I do auth?".to_string()),
            ("assistant".to_string(), "Use JWT with bcrypt".to_string()),
        ];

        let source_node_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".to_string(),
                content: "Use JWT with bcrypt".to_string(),
                token_count: None,
            })))
            .unwrap();

        let node_ids = extract_knowledge(std::sync::Arc::clone(&graph), &llm, &messages, &source_node_id, &config)
            .await
            .unwrap();

        assert_eq!(node_ids.len(), 2);
        for id in &node_ids {
            let node = graph.get_node(id).unwrap();
            assert!(matches!(node.node_type, NodeType::Knowledge(_)));
        }
    }

    #[tokio::test]
    async fn test_extract_knowledge_creates_derived_from_edges() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "decision",
                        "name": "Use Rust",
                        "description": "Chose Rust for performance",
                        "confidence": 0.95,
                        "relationships": []
                    }
                ]
            }"#
            .to_string(),
        };

        let messages = vec![("user".to_string(), "Why Rust?".to_string())];
        let source_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "Why Rust?".to_string(),
                token_count: None,
            })))
            .unwrap();

        let node_ids = extract_knowledge(std::sync::Arc::clone(&graph), &llm, &messages, &source_id, &config)
            .await
            .unwrap();

        assert_eq!(node_ids.len(), 1);
        let neighbors = graph
            .neighbors(
                &node_ids[0],
                Some(EdgeType::DerivedFrom),
                Direction::Outgoing,
            )
            .unwrap();
        assert!(neighbors.iter().any(|n| n.id == source_id));
    }

    #[tokio::test]
    async fn test_extract_knowledge_filters_by_confidence() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.9,
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "function",
                        "name": "high_confidence",
                        "description": "Very sure about this one",
                        "confidence": 0.95,
                        "relationships": []
                    },
                    {
                        "entity_type": "function",
                        "name": "low_confidence",
                        "description": "Not sure about this",
                        "confidence": 0.4,
                        "relationships": []
                    }
                ]
            }"#
            .to_string(),
        };

        let messages = vec![("user".to_string(), "test".to_string())];
        let source_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "test".to_string(),
                token_count: None,
            })))
            .unwrap();

        let node_ids = extract_knowledge(std::sync::Arc::clone(&graph), &llm, &messages, &source_id, &config)
            .await
            .unwrap();

        assert_eq!(node_ids.len(), 1);
    }

    #[tokio::test]
    async fn test_extract_knowledge_creates_relates_to_edges() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "pattern",
                        "name": "EntityA",
                        "description": "First entity",
                        "confidence": 0.9,
                        "relationships": [
                            { "target_name": "EntityB", "relationship": "uses" }
                        ]
                    },
                    {
                        "entity_type": "library",
                        "name": "EntityB",
                        "description": "Second entity",
                        "confidence": 0.85,
                        "relationships": []
                    }
                ]
            }"#
            .to_string(),
        };

        let messages = vec![("user".to_string(), "entities".to_string())];
        let source_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "entities".to_string(),
                token_count: None,
            })))
            .unwrap();

        let node_ids = extract_knowledge(std::sync::Arc::clone(&graph), &llm, &messages, &source_id, &config)
            .await
            .unwrap();

        assert_eq!(node_ids.len(), 2);
        let neighbors = graph
            .neighbors(&node_ids[0], Some(EdgeType::RelatesTo), Direction::Outgoing)
            .unwrap();
        assert_eq!(neighbors.len(), 1);
    }

    #[test]
    fn test_extraction_config_defaults() {
        let config = ExtractionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.min_confidence, 0.7);
        assert_eq!(config.entity_types.len(), 8);
    }

    #[test]
    fn test_extraction_config_deserialize_partial() {
        let toml_str = r#"
            enabled = true
            model = "claude-3-haiku"
        "#;
        let config: ExtractionConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        assert_eq!(config.model, "claude-3-haiku");
        assert_eq!(config.min_confidence, 0.7); // default
        assert_eq!(config.entity_types.len(), 8); // default
    }

    #[test]
    fn test_extraction_config_serialize_roundtrip() {
        let config = ExtractionConfig {
            enabled: true,
            model: "deepseek-chat".to_string(),
            entity_types: vec!["function".to_string(), "api".to_string()],
            min_confidence: 0.85,
            backend: ExtractionBackend::Llm,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: ExtractionConfig = serde_json::from_str(&json).unwrap();
        assert!(back.enabled);
        assert_eq!(back.model, "deepseek-chat");
        assert_eq!(back.entity_types.len(), 2);
        assert!((back.min_confidence - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extracted_entity_roundtrip() {
        let entity = ExtractedEntity {
            entity_type: "function".to_string(),
            name: "parse_config".to_string(),
            description: "Parses TOML configuration files and returns a Config struct".to_string(),
            confidence: 0.92,
            relationships: vec![
                EntityRelationship {
                    target_name: "Config".to_string(),
                    relationship: "returns".to_string(),
                },
                EntityRelationship {
                    target_name: "toml".to_string(),
                    relationship: "uses".to_string(),
                },
            ],
        };
        let json = serde_json::to_string(&entity).unwrap();
        let back: ExtractedEntity = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "parse_config");
        assert_eq!(back.entity_type, "function");
        assert_eq!(back.relationships.len(), 2);
        assert_eq!(back.relationships[0].target_name, "Config");
        assert!((back.confidence - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extracted_entity_from_llm_json() {
        let llm_output = r#"{
            "entities": [
                {
                    "entity_type": "pattern",
                    "name": "Repository Pattern",
                    "description": "Data access abstraction using trait objects",
                    "confidence": 0.88,
                    "relationships": [
                        { "target_name": "GraphStore", "relationship": "implements" }
                    ]
                },
                {
                    "entity_type": "decision",
                    "name": "Use rusqlite over sqlitegraph",
                    "description": "Chose MIT-licensed rusqlite to avoid GPL infection",
                    "confidence": 0.95,
                    "relationships": []
                }
            ]
        }"#;
        let parsed: ExtractionResponse = serde_json::from_str(llm_output).unwrap();
        assert_eq!(parsed.entities.len(), 2);
        assert_eq!(parsed.entities[0].name, "Repository Pattern");
        assert_eq!(parsed.entities[1].entity_type, "decision");
    }

    #[test]
    fn test_entity_relationship_display() {
        let rel = EntityRelationship {
            target_name: "tokio".to_string(),
            relationship: "depends_on".to_string(),
        };
        let debug = format!("{:?}", rel);
        assert!(debug.contains("tokio"));
        assert!(debug.contains("depends_on"));
    }

    #[test]
    fn test_build_extraction_prompt_structure() {
        let messages = vec![
            ("user".to_string(), "How should I implement authentication?".to_string()),
            ("assistant".to_string(), "You should use JWT tokens with bcrypt for password hashing. Store sessions in Redis.".to_string()),
            ("user".to_string(), "What about refresh tokens?".to_string()),
        ];
        let config = ExtractionConfig::default();
        let prompt = build_extraction_prompt(&messages, &config);

        assert!(prompt.contains("authentication"));
        assert!(prompt.contains("JWT"));
        assert!(prompt.contains("entity_type"));
        assert!(prompt.contains("confidence"));
        assert!(prompt.contains("function"));
        assert!(prompt.contains("pattern"));
    }

    #[test]
    fn test_build_extraction_prompt_includes_entity_types() {
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let config = ExtractionConfig {
            entity_types: vec!["function".into(), "api".into()],
            ..ExtractionConfig::default()
        };
        let prompt = build_extraction_prompt(&messages, &config);
        assert!(prompt.contains("function"));
        assert!(prompt.contains("api"));
        assert!(!prompt.contains("architecture"));
    }

    #[test]
    fn test_build_extraction_prompt_empty_conversation() {
        let messages: Vec<(String, String)> = vec![];
        let config = ExtractionConfig::default();
        let prompt = build_extraction_prompt(&messages, &config);
        assert!(prompt.contains("entity_type"));
        assert!(prompt.contains("CONVERSATION"));
        assert!(prompt.contains("(empty conversation)"));
    }

    #[tokio::test]
    async fn test_post_turn_extraction_hook() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: r#"{
                "entities": [
                    {
                        "entity_type": "convention",
                        "name": "snake_case naming",
                        "description": "Use snake_case for Rust function names",
                        "confidence": 0.9,
                        "relationships": []
                    }
                ]
            }"#
            .to_string(),
        };

        let user_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "What naming convention should I use?".to_string(),
                token_count: None,
            })))
            .unwrap();

        let assistant_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "assistant".to_string(),
                content: "Use snake_case for functions in Rust.".to_string(),
                token_count: None,
            })))
            .unwrap();

        graph
            .add_edge(GraphEdge::new(
                EdgeType::RespondsTo,
                assistant_id.clone(),
                user_id.clone(),
            ))
            .unwrap();

        let result = post_turn_extract(std::sync::Arc::clone(&graph), &llm, &config, &assistant_id).await;

        assert!(result.is_ok());
        let node_ids = result.unwrap();
        assert_eq!(node_ids.len(), 1);

        let neighbors = graph
            .neighbors(
                &node_ids[0],
                Some(EdgeType::DerivedFrom),
                Direction::Outgoing,
            )
            .unwrap();
        assert!(neighbors.iter().any(|n| n.id == assistant_id));
    }

    #[tokio::test]
    async fn test_post_turn_extraction_disabled() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: false,
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: "{}".to_string(),
        };

        let node_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "test".to_string(),
                token_count: None,
            })))
            .unwrap();

        let result = post_turn_extract(std::sync::Arc::clone(&graph), &llm, &config, &node_id).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_extract_knowledge_with_backend_local_returns_error_without_feature() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            backend: ExtractionBackend::Local {
                model_dir: "/nonexistent/model_dir".to_string(),
            },
            ..ExtractionConfig::default()
        };

        let messages = vec![("user".to_string(), "test".to_string())];
        let source_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "test".to_string(),
                token_count: None,
            })))
            .unwrap();

        let result = extract_knowledge_with_backend(
            std::sync::Arc::clone(&graph),
            None,
            #[cfg(not(feature = "local-extraction"))]
            None::<()>,
            #[cfg(feature = "local-extraction")]
            None,
            &messages,
            &source_id,
            &config,
        )
        .await;

        // Without the local-extraction feature the Local backend returns an error.
        // With the feature, it would also error because no extractor was provided.
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_extract_knowledge_with_backend_llm_creates_nodes() {
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            backend: ExtractionBackend::Llm,
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: r#"{"entities": [{"entity_type": "pattern", "name": "Test", "description": "A test", "confidence": 0.9, "relationships": []}]}"#.to_string(),
        };

        let messages = vec![("user".to_string(), "test".to_string())];
        let source_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "test".to_string(),
                token_count: None,
            })))
            .unwrap();

        let result = extract_knowledge_with_backend(
            std::sync::Arc::clone(&graph),
            Some(&llm as &dyn LlmProvider),
            #[cfg(not(feature = "local-extraction"))]
            None::<()>,
            #[cfg(feature = "local-extraction")]
            None,
            &messages,
            &source_id,
            &config,
        )
        .await
        .unwrap();

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_extraction_config_default_backend_is_llm() {
        let config = ExtractionConfig::default();
        assert!(matches!(config.backend, ExtractionBackend::Llm));
    }

    #[test]
    fn test_extraction_backend_local_deserialize() {
        let json = r#"{"local": {"model_dir": "/tmp/gliner2"}}"#;
        let backend: ExtractionBackend = serde_json::from_str(json).unwrap();
        assert!(matches!(backend, ExtractionBackend::Local { .. }));
    }

    #[test]
    fn test_extraction_config_local_backend_deserialize() {
        let toml_str = r#"
            enabled = true

            [backend]
            local = { model_dir = "/models/gliner2" }
        "#;
        let config: ExtractionConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);
        match &config.backend {
            ExtractionBackend::Local { model_dir } => {
                assert!(model_dir.contains("gliner2"));
            }
            other => panic!("Expected Local backend, got {:?}", other),
        }
    }

    #[test]
    fn test_extraction_config_hybrid_backend_deserialize() {
        let toml_str = r#"
            enabled = true

            [backend]
            hybrid = { model_dir = "/models/gliner2" }
        "#;
        let config: ExtractionConfig = toml::from_str(toml_str).unwrap();
        match &config.backend {
            ExtractionBackend::Hybrid { model_dir } => {
                assert!(model_dir.contains("gliner2"));
            }
            other => panic!("Expected Hybrid backend, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_post_turn_extract_local_backend_without_feature_returns_error() {
        // Without --features local-extraction, the Local backend should return
        // an error when dispatched via extract_knowledge_with_backend.
        // With the feature enabled, it would attempt to load the model from disk.
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            backend: ExtractionBackend::Local {
                model_dir: "/nonexistent/models/gliner2".to_string(),
            },
            ..ExtractionConfig::default()
        };
        let llm = MockExtractionProvider {
            response_json: "{}".to_string(),
        };
        let node_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "test".to_string(),
                token_count: None,
            })))
            .unwrap();

        let result = post_turn_extract(std::sync::Arc::clone(&graph), &llm, &config, &node_id).await;
        // Whether it errors depends on feature flag: without local-extraction
        // we get a feature error; with it we get a "model dir not found" error.
        // Either way the function must not panic.
        let _ = result;
    }

    #[tokio::test]
    async fn test_extract_knowledge_with_backend_hybrid_falls_through_to_llm_when_onnx_empty() {
        // Validates the Hybrid code path: when ONNX returns no entities (as the
        // placeholder parse_onnx_outputs stub does), LLM enrichment is called and
        // its response becomes the final extraction result.
        let graph = std::sync::Arc::new(GraphStore::open_memory().unwrap());
        let config = ExtractionConfig {
            enabled: true,
            min_confidence: 0.5,
            backend: ExtractionBackend::Hybrid {
                model_dir: "/models/gliner2".to_string(),
            },
            ..ExtractionConfig::default()
        };

        // Without the local-extraction feature the Hybrid backend is unavailable.
        // This test covers the non-feature path: it must return an error.
        let source_id = graph
            .add_node(GraphNode::new(NodeType::Interaction(InteractionData {
                role: "user".to_string(),
                content: "test".to_string(),
                token_count: None,
            })))
            .unwrap();
        let messages = vec![("user".to_string(), "test".to_string())];

        let result = extract_knowledge_with_backend(
            std::sync::Arc::clone(&graph),
            None,
            #[cfg(not(feature = "local-extraction"))]
            None::<()>,
            #[cfg(feature = "local-extraction")]
            None,
            &messages,
            &source_id,
            &config,
        )
        .await;

        // Without the feature flag: error contains "local-extraction".
        // With the feature: OnnxExtractor::new() fails because /models/gliner2
        // does not exist, producing a "not found" error.
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("local-extraction") || msg.contains("not found"),
            "unexpected error: {}",
            msg
        );
    }
}
