# Agent Trace Ingestion (Cursor Import) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Parse Cursor agent transcript files (`.txt`, one per conversation) and import them into the Graphirm graph as `Interaction` + `Agent` nodes, making past Cursor sessions queryable via `session_trace` and `graph_query semantic`.

**Architecture:** A pure-Rust parser in `crates/agent/src/import/cursor.rs` reads the plain-text Cursor format line-by-line using a state machine, producing a `ParsedTranscript`. A `write_transcript` function maps turns to `GraphNode`s and writes them to `GraphStore`. A new `graphirm import-cursor <path>` CLI command drives it. Idempotent — re-importing the same file is a no-op.

**Tech Stack:** Rust std only (no new deps). `graphirm_graph::{GraphStore, GraphNode, GraphEdge, NodeType, EdgeType}`. `graphirm_agent::import::cursor`.

**Key decisions:**
- Parser lives in `graphirm-agent` (not `graphirm-graph`) — keeps parsing logic away from the storage layer, mirrors existing pattern where agent crate owns session lifecycle
- Tool call/result blocks are stripped from content — raw tool JSON adds noise to knowledge extraction; the user/assistant text is what matters
- Thinking blocks are preserved as `metadata["thinking"]` on the assistant `Interaction` node — useful for understanding reasoning without polluting the main content
- Idempotency via `metadata["source_file"]` on the synthetic `Agent` node — checked with a direct SQLite query before writing anything

---

## Transcript Format Reference

Cursor transcript files (at `~/.cursor/projects/<project>/agent-transcripts/<uuid>.txt`) use this plain-text format:

```
user:
<user_query>
...user message content...
</user_query>

A:
[Thinking] ...thinking text (optional, may span lines until next marker)...
[Tool call] ToolName
  key: value

[Tool result] ToolName
...result content...

A:
...final assistant response text...

user:
<user_query>
...next user message...
</user_query>
```

**Parser rules:**
- `user:\n<user_query>` → start of user turn; ends at `</user_query>`
- `A:\n` → start of assistant segment; multiple `A:` blocks can appear before the next `user:` block
- `[Thinking] ` → thinking text; continues until `[Tool call]`, `[Tool result]`, `A:`, or `user:` marker
- `[Tool call] Name` + indented lines → tool call block, **discarded**
- `[Tool result] Name` → tool result block, **discarded**
- Last `A:` block before the next `user:` is the assistant's final response (what gets stored)
- Consecutive `A:` blocks that are all tool-call/result pairs are collapsed; only the final non-tool `A:` content becomes the assistant `Interaction`

**Parsed output per conversation pair:**
- `ParsedTurn { role: "user", content: String, thinking: None }`
- `ParsedTurn { role: "assistant", content: String, thinking: Option<String> }`

---

## Data Model

```
Agent node (synthetic, one per transcript)
  metadata: { "source_file": "<uuid>.txt", "model": "cursor", "imported_at": "<rfc3339>" }
  └─ Produces → Interaction(user turn 1)
  └─ Produces → Interaction(assistant turn 1)
       └─ RespondsTo → Interaction(user turn 1)
  └─ Produces → Interaction(user turn 2)
  └─ Produces → Interaction(assistant turn 2)
       └─ RespondsTo → Interaction(user turn 2)
  ...
```

`session_id` set to the synthetic `Agent` node's `NodeId` in every `Interaction` node's metadata — makes `session_trace replay <id>` work without any changes to that tool.

---

## Success Criteria

- `graphirm import-cursor <path/to/file.txt>` prints `Imported N turns from file.txt` or `Already imported, skipping`
- `graphirm import-cursor <path/to/dir/>` imports all `.txt` files in the directory
- `graphirm graph query --keyword "SWE-bench"` finds nodes from a Cursor session that discussed SWE-bench
- Re-running the same import is a no-op (idempotent)
- `cargo test -p graphirm-agent` passes

---

## Task 1: Parser module

**Files:**
- Create: `crates/agent/src/import/mod.rs`
- Create: `crates/agent/src/import/cursor.rs`
- Modify: `crates/agent/src/lib.rs` (add `pub mod import;`)

**Step 1: Create the module files**

`crates/agent/src/import/mod.rs`:
```rust
pub mod cursor;
```

`crates/agent/src/lib.rs` — add after the existing `pub mod` declarations:
```rust
pub mod import;
```

**Step 2: Write failing tests in `crates/agent/src/import/cursor.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_user_assistant_pair() {
        let text = "user:\n<user_query>\nhello world\n</user_query>\n\nA:\nhi there\n\nuser:\n";
        let transcript = parse_transcript("test.txt", text);
        assert_eq!(transcript.turns.len(), 2);
        assert_eq!(transcript.turns[0].role, "user");
        assert_eq!(transcript.turns[0].content.trim(), "hello world");
        assert_eq!(transcript.turns[1].role, "assistant");
        assert_eq!(transcript.turns[1].content.trim(), "hi there");
    }

    #[test]
    fn strips_tool_call_blocks() {
        let text = "user:\n<user_query>\nrun ls\n</user_query>\n\nA:\n[Tool call] Shell\n  command: ls\n\n[Tool result] Shell\nfoo.rs\n\nA:\nDone, found foo.rs\n\nuser:\n";
        let transcript = parse_transcript("test.txt", text);
        assert_eq!(transcript.turns.len(), 2);
        let assistant = &transcript.turns[1];
        assert!(assistant.content.contains("Done, found foo.rs"));
        assert!(!assistant.content.contains("[Tool call]"));
        assert!(!assistant.content.contains("[Tool result]"));
    }

    #[test]
    fn captures_thinking_in_metadata() {
        let text = "user:\n<user_query>\nwhat is rust?\n</user_query>\n\nA:\n[Thinking] The user wants to know about Rust.\nRust is a systems language.\n\nuser:\n";
        let transcript = parse_transcript("test.txt", text);
        assert_eq!(transcript.turns.len(), 2);
        let assistant = &transcript.turns[1];
        assert!(assistant.thinking.as_deref().unwrap_or("").contains("The user wants to know about Rust"));
        assert!(!assistant.content.contains("[Thinking]"));
    }

    #[test]
    fn handles_empty_transcript() {
        let transcript = parse_transcript("empty.txt", "");
        assert!(transcript.turns.is_empty());
    }

    #[test]
    fn multiple_pairs() {
        let text = "user:\n<user_query>\nfirst\n</user_query>\n\nA:\nfirst reply\n\nuser:\n<user_query>\nsecond\n</user_query>\n\nA:\nsecond reply\n\n";
        let transcript = parse_transcript("multi.txt", text);
        assert_eq!(transcript.turns.len(), 4);
        assert_eq!(transcript.turns[2].role, "user");
        assert_eq!(transcript.turns[3].role, "assistant");
    }
}
```

Run: `cargo test -p graphirm-agent import::cursor` — expected: FAIL (module doesn't exist yet)

**Step 3: Implement `parse_transcript`**

```rust
pub struct ParsedTurn {
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
}

pub struct ParsedTranscript {
    pub source_file: String,
    pub turns: Vec<ParsedTurn>,
}

pub fn parse_transcript(source_file: &str, text: &str) -> ParsedTranscript {
    let mut turns = Vec::new();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        if line == "user:" {
            // consume <user_query>...</user_query>
            let mut content = String::new();
            let mut in_tag = false;
            for l in lines.by_ref() {
                if l.trim_start().starts_with("<user_query>") {
                    in_tag = true;
                    // content after <user_query> tag on same line (usually empty)
                    let after = l.trim_start().trim_start_matches("<user_query>");
                    if !after.is_empty() { content.push_str(after); content.push('\n'); }
                } else if l.contains("</user_query>") {
                    break;
                } else if in_tag {
                    content.push_str(l);
                    content.push('\n');
                }
            }
            turns.push(ParsedTurn {
                role: "user".to_string(),
                content: content.trim().to_string(),
                thinking: None,
            });
        } else if line == "A:" {
            // collect all A: blocks until next user: or EOF, keep last non-tool content
            let mut final_content = String::new();
            let mut thinking = String::new();
            let mut in_tool = false;
            let mut current_block = String::new();

            'outer: loop {
                for l in lines.by_ref() {
                    if l == "user:" {
                        // push user: back — we can't un-consume, so handle via flag
                        // Instead: treat user: as end-of-assistant signal
                        // Save current_block as final content if non-tool
                        if !in_tool && !current_block.trim().is_empty() {
                            final_content = current_block.trim().to_string();
                        }
                        // Re-push user turn — parse it in the outer while loop next iteration
                        // Since we can't un-consume, we handle by pushing a synthetic user marker
                        turns.push(ParsedTurn {
                            role: "assistant".to_string(),
                            content: final_content.clone(),
                            thinking: if thinking.trim().is_empty() { None } else { Some(thinking.trim().to_string()) },
                        });
                        // Now manually process the user: block
                        let mut ucontent = String::new();
                        let mut in_tag = false;
                        for ul in lines.by_ref() {
                            if ul.trim_start().starts_with("<user_query>") {
                                in_tag = true;
                                let after = ul.trim_start().trim_start_matches("<user_query>");
                                if !after.is_empty() { ucontent.push_str(after); ucontent.push('\n'); }
                            } else if ul.contains("</user_query>") {
                                break;
                            } else if in_tag {
                                ucontent.push_str(ul);
                                ucontent.push('\n');
                            }
                        }
                        turns.push(ParsedTurn {
                            role: "user".to_string(),
                            content: ucontent.trim().to_string(),
                            thinking: None,
                        });
                        break 'outer;
                    } else if l == "A:" {
                        // new A: block — save current if non-tool
                        if !in_tool && !current_block.trim().is_empty() {
                            final_content = current_block.trim().to_string();
                        }
                        current_block = String::new();
                        in_tool = false;
                    } else if l.starts_with("[Thinking]") {
                        let t = l.trim_start_matches("[Thinking]").trim();
                        thinking.push_str(t);
                        thinking.push('\n');
                        in_tool = false;
                    } else if l.starts_with("[Tool call]") || l.starts_with("[Tool result]") {
                        if !in_tool && !current_block.trim().is_empty() {
                            final_content = current_block.trim().to_string();
                        }
                        current_block = String::new();
                        in_tool = true;
                    } else if in_tool && l.starts_with("  ") {
                        // indented tool params — discard
                    } else {
                        if in_tool && !l.trim().is_empty() && !l.starts_with("  ") {
                            // non-indented line after tool block — likely result content, discard
                        } else if !in_tool {
                            current_block.push_str(l);
                            current_block.push('\n');
                        }
                    }
                }
                // EOF
                if !in_tool && !current_block.trim().is_empty() {
                    final_content = current_block.trim().to_string();
                }
                if !final_content.is_empty() || !thinking.is_empty() {
                    turns.push(ParsedTurn {
                        role: "assistant".to_string(),
                        content: final_content,
                        thinking: if thinking.trim().is_empty() { None } else { Some(thinking.trim().to_string()) },
                    });
                }
                break 'outer;
            }
        }
    }

    ParsedTranscript {
        source_file: source_file.to_string(),
        turns,
    }
}
```

> Note: The parsing logic above handles the core cases. The agent implementing this should read actual transcript samples from `~/.cursor/projects/home-krs-graphirm-repo/agent-transcripts/` to validate edge cases before finalising. Keep it simple — if a line doesn't match a known marker it's treated as content or discarded.

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent import::cursor -- --nocapture`
Expected: all 5 tests pass

**Step 5: Commit**

```bash
git add crates/agent/src/import/ crates/agent/src/lib.rs
git commit -m "feat(agent): cursor transcript parser — parse_transcript with state machine"
```

---

## Task 2: Graph write function

**Files:**
- Modify: `crates/agent/src/import/cursor.rs` (add `write_transcript`, `ImportResult`, `check_already_imported`)

**Step 1: Write failing tests**

Add to the `tests` module in `cursor.rs`:

```rust
    #[test]
    fn write_transcript_creates_nodes() {
        let store = graphirm_graph::GraphStore::open_memory().unwrap();
        let text = "user:\n<user_query>\nhello\n</user_query>\n\nA:\nworld\n\n";
        let transcript = parse_transcript("abc123.txt", text);
        let result = write_transcript(&store, &transcript).unwrap();
        assert_eq!(result.turns_written, 2);
        assert!(!result.skipped);
    }

    #[test]
    fn write_transcript_is_idempotent() {
        let store = graphirm_graph::GraphStore::open_memory().unwrap();
        let text = "user:\n<user_query>\nhello\n</user_query>\n\nA:\nworld\n\n";
        let transcript = parse_transcript("abc123.txt", text);
        let r1 = write_transcript(&store, &transcript).unwrap();
        let r2 = write_transcript(&store, &transcript).unwrap();
        assert!(!r1.skipped);
        assert!(r2.skipped);
        assert_eq!(r2.turns_written, 0);
    }

    #[test]
    fn write_transcript_sets_session_id_on_interactions() {
        let store = graphirm_graph::GraphStore::open_memory().unwrap();
        let text = "user:\n<user_query>\nfoo\n</user_query>\n\nA:\nbar\n\n";
        let transcript = parse_transcript("sess.txt", text);
        let result = write_transcript(&store, &transcript).unwrap();
        // session_id on interactions should match the agent node id
        let interactions = store.list_nodes_by_type("interaction", Some(&result.agent_id.0), None, 100).unwrap();
        assert_eq!(interactions.len(), 2);
    }
```

Run: `cargo test -p graphirm-agent import::cursor` — expected: FAIL (function not defined)

**Step 2: Implement `write_transcript`**

```rust
use graphirm_graph::{
    AgentData, EdgeType, GraphEdge, GraphNode, GraphStore, NodeType,
    nodes::{InteractionData},
};
use crate::error::AgentError;

pub struct ImportResult {
    pub agent_id: graphirm_graph::nodes::NodeId,
    pub turns_written: usize,
    pub skipped: bool,
}

pub fn write_transcript(
    store: &GraphStore,
    transcript: &ParsedTranscript,
) -> Result<ImportResult, AgentError> {
    // Idempotency check
    if let Some(existing_id) = find_imported_agent(store, &transcript.source_file)? {
        return Ok(ImportResult {
            agent_id: existing_id,
            turns_written: 0,
            skipped: true,
        });
    }

    // Create synthetic Agent node
    let agent_node = {
        let mut n = GraphNode::new(NodeType::Agent(AgentData {
            name: format!("cursor/{}", transcript.source_file),
            model: "cursor".to_string(),
            system_prompt: None,
            status: "completed".to_string(),
        }));
        n.metadata["source_file"] = serde_json::json!(transcript.source_file);
        n.metadata["imported_at"] = serde_json::json!(chrono::Utc::now().to_rfc3339());
        n
    };
    let agent_id = store.add_node(agent_node).map_err(AgentError::Graph)?;
    let session_id = agent_id.0.clone();

    let mut prev_user_id: Option<graphirm_graph::nodes::NodeId> = None;
    let mut turns_written = 0;

    for turn in &transcript.turns {
        let mut interaction = GraphNode::new(NodeType::Interaction(InteractionData {
            role: turn.role.clone(),
            content: turn.content.clone(),
            token_count: None,
        }));
        interaction.metadata["session_id"] = serde_json::json!(session_id);
        if let Some(ref thinking) = turn.thinking {
            interaction.metadata["thinking"] = serde_json::json!(thinking);
        }

        let interaction_id = store.add_node(interaction).map_err(AgentError::Graph)?;

        // Agent Produces each interaction
        store.add_edge(GraphEdge::new(
            EdgeType::Produces,
            agent_id.clone(),
            interaction_id.clone(),
        )).map_err(AgentError::Graph)?;

        // assistant RespondsTo previous user turn
        if turn.role == "assistant" {
            if let Some(ref uid) = prev_user_id {
                store.add_edge(GraphEdge::new(
                    EdgeType::RespondsTo,
                    interaction_id.clone(),
                    uid.clone(),
                )).map_err(AgentError::Graph)?;
            }
        } else if turn.role == "user" {
            prev_user_id = Some(interaction_id.clone());
        }

        turns_written += 1;
    }

    tracing::info!(
        source = %transcript.source_file,
        turns = turns_written,
        "imported cursor transcript"
    );

    Ok(ImportResult { agent_id, turns_written, skipped: false })
}

fn find_imported_agent(
    store: &GraphStore,
    source_file: &str,
) -> Result<Option<graphirm_graph::nodes::NodeId>, AgentError> {
    // Use list_nodes_by_type with metadata filter to find existing import
    let filter = serde_json::json!({ "source_file": source_file });
    // Note: list_nodes_by_type metadata_filter checks top-level metadata keys
    // source_file is stored in metadata, so this works
    let agents = store
        .list_nodes_by_type("agent", None, Some(&filter), 1)
        .map_err(AgentError::Graph)?;
    Ok(agents.into_iter().next().map(|n| n.id))
}
```

**Step 3: Add `graphirm_graph` to test imports at the top of the test module**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use graphirm_graph::GraphStore;
    // ... existing tests ...
}
```

**Step 4: Run tests**

Run: `cargo test -p graphirm-agent import::cursor -- --nocapture`
Expected: all 8 tests pass

**Step 5: Commit**

```bash
git add crates/agent/src/import/cursor.rs
git commit -m "feat(agent): write_transcript — idempotent graph write for cursor imports"
```

---

## Task 3: CLI command + integration test + docs

**Files:**
- Modify: `src/main.rs` (add `ImportCursor` variant + handler, lines ~25-67 for enum, ~200+ for dispatch)
- Create: `crates/agent/src/import/cursor.rs` integration test fixture (inline in tests)
- Modify: `docs/backlog.md`
- Modify: `AGENTS.md`

**Step 1: Add CLI variant**

In `src/main.rs`, find the `Commands` enum (around line 26) and add after `ExportCorpus`:

```rust
    /// Import Cursor agent transcript file(s) into the graph.
    ///
    /// Accepts a single .txt transcript file or a directory of .txt files.
    /// Idempotent — re-importing the same file is a no-op.
    ImportCursor {
        /// Path to a .txt transcript file or a directory containing .txt files
        path: PathBuf,
        /// Print what would be imported without writing to the graph
        #[arg(long)]
        dry_run: bool,
    },
```

**Step 2: Add handler dispatch**

Find the `match cli.command` block in `main.rs` (search for `Commands::ExportCorpus`) and add:

```rust
        Commands::ImportCursor { path, dry_run } => {
            let store = graphirm_graph::GraphStore::open(&db_path)
                .map_err(|e| GraphirmError::Other(e.to_string()))?;

            let files: Vec<PathBuf> = if path.is_dir() {
                std::fs::read_dir(&path)
                    .map_err(|e| GraphirmError::Other(e.to_string()))?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().map_or(false, |ext| ext == "txt"))
                    .collect()
            } else {
                vec![path]
            };

            if files.is_empty() {
                println!("No .txt files found.");
                return Ok(());
            }

            for file in &files {
                let text = std::fs::read_to_string(file)
                    .map_err(|e| GraphirmError::Other(format!("{}: {e}", file.display())))?;
                let source_name = file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");
                let transcript = graphirm_agent::import::cursor::parse_transcript(source_name, &text);

                if dry_run {
                    println!(
                        "[dry-run] {} — {} turns",
                        source_name,
                        transcript.turns.len()
                    );
                    continue;
                }

                let result = graphirm_agent::import::cursor::write_transcript(&store, &transcript)
                    .map_err(|e| GraphirmError::Other(e.to_string()))?;

                if result.skipped {
                    println!("Already imported, skipping: {source_name}");
                } else {
                    println!(
                        "Imported {} turns from {source_name}",
                        result.turns_written
                    );
                }
            }
            Ok(())
        }
```

**Step 3: Smoke test**

```bash
# dry run on this session's own transcripts
./target/debug/graphirm import-cursor \
  ~/.cursor/projects/home-krs-graphirm-repo/agent-transcripts/ \
  --dry-run
```
Expected: prints `[dry-run] <uuid>.txt — N turns` for each file, no DB writes.

```bash
# real import of one file
./target/debug/graphirm import-cursor \
  ~/.cursor/projects/home-krs-graphirm-repo/agent-transcripts/<any-uuid>.txt
```
Expected: `Imported N turns from <uuid>.txt`

```bash
# re-import same file — idempotent
./target/debug/graphirm import-cursor \
  ~/.cursor/projects/home-krs-graphirm-repo/agent-transcripts/<same-uuid>.txt
```
Expected: `Already imported, skipping: <uuid>.txt`

**Step 4: Run full test suite**

```bash
cargo test -p graphirm-agent
cargo test -p graphirm-graph
cargo clippy -p graphirm-agent -- -D warnings
cargo fmt --check
```
All expected to pass.

**Step 5: Update docs**

- `docs/backlog.md` — mark "Agent Trace ingestion" as ✅ Done with summary
- `AGENTS.md` — add Phase 30 row + detail block

**Step 6: Commit**

```bash
git add src/main.rs docs/backlog.md AGENTS.md
git commit -m "feat: import-cursor CLI command — ingest Cursor transcripts into graph"
git push origin main
```

---

## Risk Areas

- **Format variation**: Cursor may update its transcript format. The parser is deliberately lenient — unknown lines are treated as content or discarded. If a transcript produces 0 turns, it's a silent no-op (not an error). Add a `--verbose` flag in a future iteration to debug.
- **`list_nodes_by_type` metadata filter**: The existing `metadata_filter` in `list_nodes_by_type` does a JSON value comparison, not a `json_extract` SQL call. It fetches all agent nodes then filters in Rust. With 130+ sessions this is fast enough; at 10k sessions it would need a dedicated SQL query. Acceptable for now.
- **`AgentError::Graph` mapping**: Verify `AgentError::Graph` accepts `graphirm_graph::GraphError` directly via `#[from]` — check `crates/agent/src/error.rs` before writing the `map_err` calls.
