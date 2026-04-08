# Chat Pane Structured Message Rendering — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop rendering raw JSON in the chat pane by (A) cleaning the assistant message content server-side so it always contains readable markdown, and (B) splitting assistant messages into per-type cards on the client when segment Content nodes exist.

**Architecture:** Two-layer fix. **Server** (Rust): when `text_content()` returns a JSON segment envelope (`{"segments": [...]}`) the stored `InteractionData.content` is the concatenated text of all segments, not the raw JSON wrapper. **Client** (React): `ChatPane` detects segment children on the assistant interaction (via `Contains` edges in the graph), renders each segment as a separate styled card with a type badge and appropriate formatting (markdown for text, syntax-highlighted `<pre>` for code, muted for reasoning).

**Tech Stack:** Rust (`crates/agent/src/workflow.rs`, `crates/llm/src/provider.rs`), TypeScript/React (`web-app/src/components/ChatPane.tsx`, `web-app/src/hooks/useSession.ts`, `web-app/src/api/client.ts`), CSS modules.

**Key decisions:**

- **Server strips JSON envelope at persistence time** — not at read time — so the Interaction node always has human-readable `content` regardless of how it's queried (API, TUI, export). This is the minimal-surprise approach.
- **Client segment cards are opt-in per message** — only when the Interaction node has `metadata.segmented === true` do we fetch and render segment children; plain messages stay as-is (no extra API call).
- **No new dependencies** — use existing `marked` + `hljs` (`MarkdownBody`) for text/code; pure CSS for card chrome.

---

## Current data flow

```text
LLM response (structured_output: true)
  → text_content() joins ContentPart::Text parts → raw JSON envelope string
  → InteractionData { content: raw_json, ... }
  → Interaction node in graph (content = raw JSON)
  → GET /api/sessions/:id/messages → { role: "assistant", content: raw_json }
  → ChatPane → MarkdownBody → renders raw JSON as-is 😱

Separately (non-fatal, async):
  parse_structured_segments(raw_text) → Vec<Segment>
  → persist_segments → Content nodes linked via Contains edges
  → Interaction node metadata["segmented"] = true
```

## What changes

1. **Server:** After `parse_structured_segments` succeeds, **update** the Interaction node's `content` from raw JSON to the **concatenated segment text** (newline-joined). This means `text_content()` still stores the raw string initially, but the post-segmentation path patches it.

2. **Client:** When `message.segmented === true`, fetch segment children (Content nodes linked via `Contains` from the Interaction) and render each as a typed card instead of one `MarkdownBody`.

---

## Success criteria

- [x] Assistant messages in the chat pane never show raw JSON — always readable markdown/text.
- [x] When segments are available, each segment renders as a visually distinct card with a type badge.
- [x] Code segments use syntax highlighting; reasoning segments are muted/collapsible.
- [x] Non-segmented messages (most messages) render exactly as before — no regression.
- [x] SSE streaming still works (streaming shows incremental text; cards appear on `message_end` when segments are persisted).
- [x] Existing tests pass; new tests cover the content-patching logic.

---

## Risks

- **Race condition:** Segment persistence is async and happens after the Interaction node is already saved. The content patch must happen in the same `spawn_blocking` scope as `persist_segments`.
- **Backward compat:** Existing Interaction nodes in old DBs have raw JSON content. A client-side fallback (`tryParseAndFlatten`) handles them gracefully.
- **Streaming:** During SSE streaming, the content is raw text deltas. The card split only applies after `message_end` when `getMessages` re-fetches. This is fine — streaming already shows a single `MarkdownBody`.

---

## Dependency order

```text
Task 1 (server: patch content after segmentation)
  → Task 2 (client: segment types + API)
  → Task 3 (client: ChatPane card rendering)
  → Task 4 (CSS styling for segment cards)
  → Task 5 (client fallback for legacy JSON content)
  → Task 6 (docs)
```

Tasks 2–4 can be developed together; Task 5 is a safety net.

---

### Task 1: Server — patch Interaction content after segment persistence

**Files:**

- Modify: `crates/agent/src/workflow.rs` — after `persist_segments` succeeds, update the Interaction node's `content` field

**Behavior:**

After `persist_segments` returns `Ok(node_ids)`, concatenate all segment contents (newline-joined) and patch the Interaction node:

```rust
// After persist_segments succeeds (around line 755–785 in workflow.rs):
let clean_text = segments.iter().map(|s| s.content.as_str()).collect::<Vec<_>>().join("\n\n");
if clean_text != raw_text {
    let graph_for_patch = session.graph.clone();
    let patch_id = node_id.clone();
    let _ = tokio::task::spawn_blocking(move || {
        if let Ok(mut node) = graph_for_patch.get_node(&patch_id) {
            if let NodeType::Interaction(ref mut data) = node.node_type {
                data.content = clean_text;
            }
            let _ = graph_for_patch.update_node(&patch_id, node);
        }
    }).await;
}
```

Non-fatal: if the patch fails, the message still renders (just with raw JSON, same as today).

**Step 1:** Implement the patch.

**Step 2:** `cargo test -p graphirm-agent`

**Step 3:** Commit: `fix(agent): strip JSON segment envelope from assistant message content`

---

### Task 2: Client — expose segment children in Message type

**Files:**

- Modify: `web-app/src/types/graph.ts` — add `segmented?: boolean` and `segments?: SegmentPart[]` to `Message`
- Modify: `web-app/src/api/client.ts` — `getMessages` maps `metadata.segmented` and fetches segment Content children

**Types:**

```typescript
export interface SegmentPart {
  type: string;       // 'code' | 'reasoning' | 'observation' | 'plan' | 'answer' | string
  content: string;
  language?: string;   // for code segments
}

export interface Message {
  id: string;
  role: NodeRole;
  content: string;
  created_at: string;
  segmented?: boolean;
  segments?: SegmentPart[];
}
```

**`getMessages` change:** After mapping Interaction nodes to `Message[]`, for each message where `metadata.segmented === true`:

- Query the graph subgraph (`GET /api/graph/:session_id`) — already fetched — or add a lightweight endpoint. Simplest: use edges from the already-loaded graph data in `useSession` to find `Contains` children of that Interaction node where `node_type.type === 'Content'`.

Actually, the graph data is already in the client (`graphData` in `useSession` / `useGraphData`). So the segment lookup can happen **entirely client-side** by walking `Contains` edges from the Interaction id to Content nodes. No new API call needed.

**Step 1:** Add types.

**Step 2:** `cd web-app && npx tsc --noEmit`

**Step 3:** Commit: `feat(web-app): SegmentPart type and segmented flag on Message`

---

### Task 3: Client — ChatPane renders segment cards

**Files:**

- Modify: `web-app/src/components/ChatPane.tsx`
- Create: `web-app/src/components/SegmentCard.tsx` (small component)
- Modify: `web-app/src/hooks/useSession.ts` — enrich messages with segments from graph data

**Behavior in `useSession`:** After `setMessages(msgs)`, for each message with `segmented === true`, look up Content children via `Contains` edges in `graphData`. Map to `SegmentPart[]` and set `msg.segments`.

**`ChatPane` rendering change:**

```tsx
{msg.role === 'user' ? (
  <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
) : msg.segments && msg.segments.length > 0 ? (
  <div className={styles.segmentStack}>
    {msg.segments.map((seg, i) => (
      <SegmentCard key={i} segment={seg} />
    ))}
  </div>
) : (
  <MarkdownBody content={msg.content} maxHeight={250} />
)}
```

**`SegmentCard` component:**

```tsx
function SegmentCard({ segment }: { segment: SegmentPart }) {
  return (
    <div className={`${styles.segmentCard} ${styles[`seg_${segment.type}`] ?? ''}`}>
      <span className={styles.segBadge}>{segment.type}</span>
      {segment.type === 'code' ? (
        <pre><code dangerouslySetInnerHTML={{
          __html: hljs.highlightAuto(segment.content).value
        }} /></pre>
      ) : (
        <MarkdownBody content={segment.content} maxHeight={200} />
      )}
    </div>
  );
}
```

**Step 1:** Implement `SegmentCard`.

**Step 2:** Wire into `ChatPane`.

**Step 3:** `cd web-app && npm run build`

**Step 4:** Commit: `feat(web-app): render assistant segments as typed cards in ChatPane`

---

### Task 4: CSS styling for segment cards

**Files:**

- Modify: `web-app/src/styles/chat.module.css`

**Styles:**

```css
.segmentStack {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.segmentCard {
  padding: 6px 8px;
  border-radius: 4px;
  border-left: 2px solid var(--border);
  font-size: 12px;
  line-height: 1.5;
}

.segBadge {
  display: inline-block;
  font-size: 9px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 1px 5px;
  border-radius: 3px;
  margin-bottom: 4px;
  background: var(--surface-2);
  color: var(--fg-muted);
}

.seg_code {
  background: var(--surface-0);
  border-left-color: var(--node-content);
  font-family: var(--font-mono);
}

.seg_reasoning {
  background: color-mix(in srgb, var(--fg-muted) 5%, var(--surface-0));
  border-left-color: var(--fg-muted);
  opacity: 0.8;
}

.seg_answer {
  border-left-color: var(--accent);
}

.seg_plan {
  border-left-color: var(--node-knowledge);
}

.seg_observation {
  border-left-color: var(--info);
}
```

**Step 1:** Add styles.

**Step 2:** `cd web-app && npm run build`

**Step 3:** Commit: `style(web-app): segment card styling for ChatPane`

---

### Task 5: Client fallback for legacy JSON content

**Files:**

- Modify: `web-app/src/components/ChatPane.tsx` (or a small utility)

**Behavior:** For old messages where `segmented` is falsy but `content` starts with `{` or `[`, attempt to parse and extract readable text:

```typescript
function cleanLegacyContent(content: string): string {
  const t = content.trim();
  if (!t.startsWith('{') && !t.startsWith('[')) return content;
  try {
    const parsed = JSON.parse(t);
    if (parsed.segments && Array.isArray(parsed.segments)) {
      return parsed.segments
        .map((s: { content?: string }) => s.content ?? '')
        .join('\n\n');
    }
    if (Array.isArray(parsed)) {
      return parsed
        .map((s: { content?: string }) => s.content ?? JSON.stringify(s))
        .join('\n\n');
    }
  } catch { /* not JSON */ }
  return content;
}
```

Apply before `MarkdownBody`:

```tsx
<MarkdownBody content={cleanLegacyContent(msg.content)} maxHeight={250} />
```

**Step 1:** Add utility and wire in.

**Step 2:** `cd web-app && npm run build`

**Step 3:** Commit: `fix(web-app): graceful fallback for legacy JSON-wrapped assistant messages`

---

### Task 6: Documentation

**Files:**

- Modify: `docs/backlog.md` — mark relevant items done / narrowed
- Modify: `AGENTS.md` — brief note under web-app or relevant phase

**Step 1:** Commit: `docs: chat pane structured rendering`

---

## Verification (before claiming done)

```bash
cargo fmt
cargo clippy --workspace -- -D warnings
cargo test -p graphirm-agent
cargo test --workspace
cd web-app && npm run build
```

---

## Execution handoff

**Plan saved to:** `docs/plans/2026-04-09-chat-pane-structured-rendering.md`

**Execution options:**

1. **Subagent-driven (this session)** — one task at a time with reviews.
2. **Separate session** — use **executing-plans** in a **git worktree**.

**Which approach?**
