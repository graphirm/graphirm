use crate::error::AgentError;
use graphirm_graph::{EdgeType, GraphEdge, GraphNode, NodeId, NodeType};

/// Represents a single turn (user query + assistant response) in a parsed transcript.
#[derive(Debug, Clone)]
pub struct ParsedTurn {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
}

/// Represents a complete parsed transcript from a Cursor agent session.
#[derive(Debug, Clone)]
pub struct ParsedTranscript {
    pub source_file: String,
    pub turns: Vec<ParsedTurn>,
}

/// Parse a Cursor agent transcript file.
///
/// Parser rules:
/// - `user:` line → start user turn; collect lines between `<user_query>` and `</user_query>`
/// - `A:` line → start assistant segment; multiple `A:` blocks can appear
/// - `[Thinking] ...` → collect into thinking field; stop on next marker
/// - `[Tool call] Name` + indented lines → discard
/// - `[Tool result] Name` lines → discard
/// - Last non-tool `A:` block before next `user:` becomes assistant content
pub fn parse_transcript(source_file: &str, text: &str) -> ParsedTranscript {
    let mut turns: Vec<ParsedTurn> = Vec::new();
    let mut current_turn: Option<ParsedTurn> = None;
    let mut current_thinking: Option<String> = None;
    let mut in_user_query = false;
    let mut in_tool_call = false;
    let mut in_tool_result = false;
    let mut in_thinking = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Check for user marker
        if trimmed.starts_with("user:") {
            // Save previous turn if exists
            if let Some(mut turn) = current_turn.take() {
                // Normalise: trim trailing blank lines, keep one trailing newline
                let trimmed_content = turn.content.trim_end().to_string();
                turn.content = if trimmed_content.is_empty() {
                    trimmed_content
                } else {
                    format!("{trimmed_content}\n")
                };
                turns.push(turn);
            }
            current_turn = Some(ParsedTurn {
                role: "user".to_string(),
                content: String::new(),
                thinking: None,
            });
            in_user_query = true;
            in_tool_call = false;
            in_tool_result = false;
            in_thinking = false;
            continue;
        }

        // Check for assistant marker
        if trimmed.starts_with("A:") {
            // Save previous turn if exists
            if let Some(mut turn) = current_turn.take() {
                let trimmed_content = turn.content.trim_end().to_string();
                turn.content = if trimmed_content.is_empty() {
                    trimmed_content
                } else {
                    format!("{trimmed_content}\n")
                };
                turns.push(turn);
            }
            current_turn = Some(ParsedTurn {
                role: "assistant".to_string(),
                content: String::new(),
                thinking: None,
            });
            in_user_query = false;
            in_tool_call = false;
            in_tool_result = false;
            in_thinking = false;
            continue;
        }

        // Check for tool call marker
        if trimmed.starts_with("[Tool call]") {
            in_tool_call = true;
            in_tool_result = false;
            in_thinking = false;
            continue;
        }

        // Check for tool result marker
        if trimmed.starts_with("[Tool result]") {
            in_tool_result = true;
            in_tool_call = false;
            in_thinking = false;
            continue;
        }

        // Check for thinking marker
        if trimmed.starts_with("[Thinking]") {
            current_thinking = Some(String::new());
            in_thinking = true;
            in_user_query = false;
            in_tool_call = false;
            in_tool_result = false;
            continue;
        }

        // Collect content based on context
        if let Some(turn) = &mut current_turn {
            // Check for closing tags first (before content collection)
            if in_user_query && trimmed == "</user_query>" {
                in_user_query = false;
                // Reset tool flags since we're leaving user query mode
                in_tool_call = false;
                in_tool_result = false;
                continue;
            }

            if in_thinking && trimmed == "[/Thinking]" {
                // Finalize thinking content
                if let Some(t) = current_thinking.take() {
                    turn.thinking = Some(t);
                }
                in_thinking = false;
                in_user_query = false;
                in_tool_call = false;
                in_tool_result = false;
                continue;
            }

            if in_user_query {
                // Skip the <user_query> tag line itself
                if trimmed == "<user_query>" {
                    continue;
                }
                // Inside <user_query> block - collect all content
                turn.content.push_str(line);
                turn.content.push('\n');
                continue;
            }

            // Check if we're at the end of a tool block (non-indented line after tool content)
            if in_tool_call || in_tool_result {
                // If this line is not indented, we're out of the tool block
                if !line.starts_with(' ') && !line.starts_with('\t') {
                    in_tool_call = false;
                    in_tool_result = false;
                    // Don't continue - process this line as assistant content
                } else {
                    // Still in tool block, discard
                    continue;
                }
            }

            // Collect thinking content
            if in_thinking {
                if let Some(thinking) = &mut current_thinking {
                    // Skip the [Thinking] tag line itself
                    if trimmed == "[Thinking]" {
                        continue;
                    }
                    thinking.push_str(line);
                    thinking.push('\n');
                }
                continue;
            }

            // For user turns, only collect content inside <user_query> tags
            if turn.role == "user" {
                continue;
            }

            // Default: collect as assistant content
            turn.content.push_str(line);
            turn.content.push('\n');
        }
    }

    // Save final turn if exists
    if let Some(mut turn) = current_turn.take() {
        let trimmed_content = turn.content.trim_end().to_string();
        turn.content = if trimmed_content.is_empty() {
            trimmed_content
        } else {
            format!("{trimmed_content}\n")
        };
        turns.push(turn);
    }

    ParsedTranscript {
        source_file: source_file.to_string(),
        turns,
    }
}

/// Result of importing a transcript into the graph.
#[derive(Debug, Clone)]
pub struct ImportResult {
    pub agent_id: NodeId,
    pub turns_written: usize,
    pub skipped: bool,
}

/// Find an existing agent node for this source file.
fn find_imported_agent(
    store: &graphirm_graph::GraphStore,
    source_file: &str,
) -> Result<Option<NodeId>, AgentError> {
    let filter = serde_json::json!({"source_file": source_file});
    let agents = store.list_nodes_by_type("agent", None, Some(&filter), 1)?;
    Ok(agents.into_iter().next().map(|a| a.id))
}

/// Write a parsed transcript to the graph store.
///
/// This function is idempotent - if the same source file has already been imported,
/// it returns `ImportResult { skipped: true }` without creating new nodes.
pub fn write_transcript(
    store: &graphirm_graph::GraphStore,
    transcript: &ParsedTranscript,
) -> Result<ImportResult, AgentError> {
    // Check if already imported (idempotency)
    if let Some(agent_id) = find_imported_agent(store, &transcript.source_file)? {
        return Ok(ImportResult {
            agent_id,
            turns_written: 0,
            skipped: true,
        });
    }

    // Create agent node
    let mut agent_node = GraphNode::new(NodeType::Agent(graphirm_graph::AgentData {
        name: "cursor-import".to_string(),
        model: "unknown".to_string(),
        system_prompt: None,
        status: "imported".to_string(),
    }));
    agent_node.metadata["source_file"] = serde_json::json!(transcript.source_file);
    agent_node.metadata["imported_at"] = serde_json::json!(chrono::Utc::now().to_rfc3339());

    let agent_id = store.add_node(agent_node)?;

    // Track interaction IDs for edge creation
    let mut interaction_ids: Vec<NodeId> = Vec::new();

    // Create interaction nodes for each turn
    for turn in &transcript.turns {
        let mut interaction_node =
            GraphNode::new(NodeType::Interaction(graphirm_graph::InteractionData {
                role: turn.role.clone(),
                content: turn.content.clone(),
                token_count: None,
            }));
        interaction_node.metadata["session_id"] = serde_json::json!(agent_id.0.clone());

        let interaction_id = store.add_node(interaction_node)?;
        interaction_ids.push(interaction_id);
    }

    // Create Produces edge from agent to first interaction
    if !interaction_ids.is_empty() {
        let produces_edge = GraphEdge::new(
            EdgeType::Produces,
            agent_id.clone(),
            interaction_ids[0].clone(),
        );
        store.add_edge(produces_edge)?;
    }

    // Create RespondsTo edges between interactions
    for i in 1..interaction_ids.len() {
        let responds_to_edge = GraphEdge::new(
            EdgeType::RespondsTo,
            interaction_ids[i].clone(),
            interaction_ids[i - 1].clone(),
        );
        store.add_edge(responds_to_edge)?;
    }

    Ok(ImportResult {
        agent_id,
        turns_written: interaction_ids.len(),
        skipped: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_user_assistant_pair() {
        let text = r#"user:
<user_query>
Hello, how are you?
</user_query>

A:
I'm doing well, thank you for asking!
"#;

        let result = parse_transcript("test.txt", text);

        assert_eq!(result.source_file, "test.txt");
        assert_eq!(result.turns.len(), 2);

        assert_eq!(result.turns[0].role, "user");
        assert!(result.turns[0].content.contains("Hello, how are you?"));
        assert!(result.turns[0].thinking.is_none());

        assert_eq!(result.turns[1].role, "assistant");
        assert!(result.turns[1].content.contains("I'm doing well"));
        assert!(result.turns[1].thinking.is_none());
    }

    #[test]
    fn strips_tool_call_blocks() {
        let text = r#"user:
<user_query>
What files are in the directory?
</user_query>

A:
[Tool call] bash
  command: ls -la
  description: List files in current directory

[Tool result] bash
  stdout: src/ Cargo.toml
  stderr:

I can see the files now.
"#;

        let result = parse_transcript("test.txt", text);

        assert_eq!(result.turns.len(), 2);
        assert!(result.turns[1].content.contains("I can see the files now"));
        assert!(!result.turns[1].content.contains("[Tool call]"));
        assert!(!result.turns[1].content.contains("[Tool result]"));
    }

    #[test]
    fn captures_thinking_in_metadata() {
        let text = r#"user:
<user_query>
How do I fix this bug?
</user_query>

A:
[Thinking]
The user is asking about a bug fix. I should ask for more details about the error
message and what they've already tried. This will help me provide a more targeted
solution.
[/Thinking]

I'd be happy to help fix your bug! Can you tell me what error message you're seeing?
"#;

        let result = parse_transcript("test.txt", text);

        assert_eq!(result.turns.len(), 2);
        assert!(result.turns[1].thinking.is_some());
        let thinking = result.turns[1].thinking.as_ref().unwrap();
        assert!(thinking.contains("The user is asking about a bug fix"));
        assert!(thinking.contains("I should ask for more details"));
    }

    #[test]
    fn handles_empty_transcript() {
        let text = "";

        let result = parse_transcript("empty.txt", text);

        assert_eq!(result.source_file, "empty.txt");
        assert!(result.turns.is_empty());
    }

    #[test]
    fn multiple_pairs() {
        let text = r#"user:
<user_query>
Hello
</user_query>

A:
Hi there!

user:
<user_query>
How are you?
</user_query>

A:
I'm doing well, thanks!
"#;

        let result = parse_transcript("test.txt", text);

        assert_eq!(result.turns.len(), 4);
        assert_eq!(result.turns[0].role, "user");
        assert_eq!(result.turns[0].content, "Hello\n");
        assert_eq!(result.turns[1].role, "assistant");
        assert_eq!(result.turns[1].content, "Hi there!\n");
        assert_eq!(result.turns[2].role, "user");
        assert_eq!(result.turns[2].content, "How are you?\n");
        assert_eq!(result.turns[3].role, "assistant");
        assert_eq!(result.turns[3].content, "I'm doing well, thanks!\n");
    }

    #[test]
    fn write_transcript_creates_nodes() {
        use graphirm_graph::GraphStore;

        let store = GraphStore::open_memory().unwrap();
        let transcript = ParsedTranscript {
            source_file: "test-turns.txt".to_string(),
            turns: vec![
                ParsedTurn {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                    thinking: None,
                },
                ParsedTurn {
                    role: "assistant".to_string(),
                    content: "Hi".to_string(),
                    thinking: None,
                },
            ],
        };

        let result = write_transcript(&store, &transcript).unwrap();

        assert_eq!(result.turns_written, 2);
        assert!(!result.skipped);
        assert_eq!(result.agent_id.0.len(), 36); // UUID length
    }

    #[test]
    fn write_transcript_is_idempotent() {
        use graphirm_graph::GraphStore;

        let store = GraphStore::open_memory().unwrap();
        let transcript = ParsedTranscript {
            source_file: "idempotent-test.txt".to_string(),
            turns: vec![ParsedTurn {
                role: "user".to_string(),
                content: "Hello".to_string(),
                thinking: None,
            }],
        };

        // First write
        let result1 = write_transcript(&store, &transcript).unwrap();
        assert!(!result1.skipped);
        assert_eq!(result1.turns_written, 1);

        // Second write should be skipped
        let result2 = write_transcript(&store, &transcript).unwrap();
        assert!(result2.skipped);
        assert_eq!(result2.turns_written, 0);
    }

    #[test]
    fn write_transcript_sets_session_id_on_interactions() {
        use graphirm_graph::GraphStore;

        let store = GraphStore::open_memory().unwrap();
        let transcript = ParsedTranscript {
            source_file: "session-test.txt".to_string(),
            turns: vec![
                ParsedTurn {
                    role: "user".to_string(),
                    content: "First".to_string(),
                    thinking: None,
                },
                ParsedTurn {
                    role: "assistant".to_string(),
                    content: "Second".to_string(),
                    thinking: None,
                },
            ],
        };

        let result = write_transcript(&store, &transcript).unwrap();

        // Get the agent node to check session_id
        let agent = store.get_node(&result.agent_id).unwrap();
        assert_eq!(agent.metadata["source_file"], "session-test.txt");

        // Get interactions and verify session_id
        let interactions = store
            .list_nodes_by_type("interaction", None, None, 10)
            .unwrap();
        assert_eq!(interactions.len(), 2);

        for interaction in &interactions {
            assert_eq!(
                interaction.metadata["session_id"],
                serde_json::json!(result.agent_id.0)
            );
        }
    }
}
