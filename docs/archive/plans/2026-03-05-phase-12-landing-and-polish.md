# Phase 12: Landing Page + Product Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the graphirm.ai static landing page and three product-quality improvements (session restore, DAG timeline, Agent Trace export) that make the product credible to first-time visitors.

**Architecture:** Phase 12 has two parallel tracks. Track A is a static website deployed to Cloudflare Pages — pure HTML/CSS/JS, no backend, lives in `graphirm-site/` at the repo root. Track B is engineering work inside the existing Rust codebase and VS Code extension — session restoration in `crates/server/`, DAG timeline in `graphirm-vscode/media/`, and an `agent-trace` CLI subcommand in `src/main.rs`.

**Tech Stack:** HTML/CSS/JS (landing page), Cloudflare Pages (hosting), Rust/axum (session restore), TypeScript/d3 (DAG layout), JSON (Agent Trace export)

---

## Track A: graphirm.ai Static Landing Page

### Task 1: Scaffold the site directory

**Files:**
- Create: `graphirm-site/index.html`
- Create: `graphirm-site/styles.css`
- Create: `graphirm-site/_headers` (Cloudflare security headers)

**Step 1: Create the directory structure**

```bash
mkdir -p graphirm-site
```

**Step 2: Create `graphirm-site/_headers`**

```
/*
  X-Frame-Options: DENY
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), microphone=(), geolocation=()
```

**Step 3: Create `graphirm-site/styles.css`** — dark, technical, minimal

```css
:root {
  --bg: #0d0d0d;
  --surface: #141414;
  --border: #222;
  --text: #e8e8e8;
  --muted: #888;
  --accent: #7c6af7;
  --green: #4ade80;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-sans);
  font-size: 16px;
  line-height: 1.6;
}

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

code, pre {
  font-family: var(--font-mono);
  font-size: 0.875em;
}

/* Layout */
.container { max-width: 860px; margin: 0 auto; padding: 0 24px; }

/* Nav */
nav {
  border-bottom: 1px solid var(--border);
  padding: 16px 0;
}
nav .container {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.logo { font-family: var(--font-mono); font-weight: 700; font-size: 1.1rem; color: var(--text); }
.logo span { color: var(--accent); }
nav .links { display: flex; gap: 24px; }
nav .links a { color: var(--muted); font-size: 0.9rem; }
nav .links a:hover { color: var(--text); text-decoration: none; }

/* Hero */
.hero { padding: 96px 0 72px; }
.hero h1 {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.02em;
  margin-bottom: 20px;
}
.hero h1 em { font-style: normal; color: var(--accent); }
.hero p {
  font-size: 1.15rem;
  color: var(--muted);
  max-width: 600px;
  margin-bottom: 36px;
}
.hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  border-radius: 6px;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  border: none;
  transition: opacity 0.15s;
}
.btn:hover { opacity: 0.85; text-decoration: none; }
.btn-primary { background: var(--accent); color: #fff; }
.btn-secondary {
  background: transparent;
  color: var(--text);
  border: 1px solid var(--border);
}

/* Terminal demo */
.demo {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin: 48px 0;
  overflow-x: auto;
}
.demo-bar {
  display: flex;
  gap: 6px;
  margin-bottom: 16px;
}
.demo-bar span {
  width: 12px; height: 12px; border-radius: 50%;
}
.demo-bar .r { background: #ff5f56; }
.demo-bar .y { background: #ffbd2e; }
.demo-bar .g { background: #27c93f; }
.demo pre { color: #ccc; line-height: 1.7; }
.demo .prompt { color: var(--accent); }
.demo .comment { color: #555; }
.demo .output { color: var(--green); }

/* Features */
.features { padding: 72px 0; border-top: 1px solid var(--border); }
.features h2 { font-size: 1.5rem; margin-bottom: 48px; }
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 32px;
}
.feature-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 24px;
}
.feature-card .icon { font-size: 1.5rem; margin-bottom: 12px; }
.feature-card h3 { font-size: 1rem; margin-bottom: 8px; }
.feature-card p { color: var(--muted); font-size: 0.9rem; }

/* Architecture */
.architecture { padding: 72px 0; border-top: 1px solid var(--border); }
.architecture h2 { font-size: 1.5rem; margin-bottom: 24px; }
.arch-diagram {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 24px;
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: #ccc;
  line-height: 1.8;
  white-space: pre;
  overflow-x: auto;
}

/* Install */
.install { padding: 72px 0; border-top: 1px solid var(--border); }
.install h2 { font-size: 1.5rem; margin-bottom: 32px; }
.install-steps { display: flex; flex-direction: column; gap: 20px; }
.install-step {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}
.step-num {
  width: 28px; height: 28px;
  background: var(--accent);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.8rem; font-weight: 700;
  flex-shrink: 0;
}
.step-body h4 { margin-bottom: 6px; }
.step-body code {
  display: block;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 14px;
  color: var(--green);
  margin-top: 8px;
}

/* Footer */
footer {
  border-top: 1px solid var(--border);
  padding: 32px 0;
  color: var(--muted);
  font-size: 0.85rem;
}
footer .container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}
footer a { color: var(--muted); }
footer a:hover { color: var(--text); }
```

**Step 4: Create `graphirm-site/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Graphirm — Graph-native coding agent</title>
  <meta name="description" content="A coding agent where the graph is the product. Every decision, tool call, and file edit is a node you can navigate, replay, and steer." />
  <meta property="og:title" content="Graphirm" />
  <meta property="og:description" content="Graph-native coding agent. Every interaction is a node." />
  <meta property="og:image" content="https://graphirm.ai/og.png" />
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⬡</text></svg>" />
  <link rel="stylesheet" href="styles.css" />
</head>
<body>

<nav>
  <div class="container">
    <span class="logo">graph<span>irm</span></span>
    <div class="links">
      <a href="https://github.com/graphirm/graphirm">GitHub</a>
      <a href="https://github.com/graphirm/graphirm/releases">Releases</a>
      <a href="https://github.com/graphirm/graphirm/blob/main/docs/plans/00-execution-strategy.md">Roadmap</a>
    </div>
  </div>
</nav>

<main>
  <section class="hero">
    <div class="container">
      <h1>A coding agent where<br/><em>the graph is the product</em></h1>
      <p>Every decision, tool call, and file edit is a node you can navigate, replay, and steer. Not a chat log — a knowledge graph.</p>
      <div class="hero-actions">
        <a class="btn btn-primary" href="https://github.com/graphirm/graphirm">View on GitHub</a>
        <a class="btn btn-secondary" href="#install">Install VS Code Extension ↓</a>
      </div>

      <div class="demo">
        <div class="demo-bar">
          <span class="r"></span><span class="y"></span><span class="g"></span>
        </div>
        <pre><span class="comment"># Start the Graphirm server</span>
<span class="prompt">$</span> graphirm serve
<span class="output">INFO graphirm_server: Starting on 127.0.0.1:3000</span>

<span class="comment"># Every message, tool call, and file edit → graph node</span>
<span class="prompt">$</span> curl -X POST localhost:3000/api/sessions \
    -d '{"name":"fix-auth-bug"}'
<span class="output">{"id":"3fa85f64","name":"fix-auth-bug","status":"idle"}</span>

<span class="comment"># Navigate sessions in VS Code — chat + live graph side by side</span></pre>
      </div>
    </div>
  </section>

  <section class="features">
    <div class="container">
      <h2>Why graph-native?</h2>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="icon">⬡</div>
          <h3>Cross-session memory</h3>
          <p>Knowledge nodes persist across sessions. PageRank surfaces what matters — not what was said last.</p>
        </div>
        <div class="feature-card">
          <div class="icon">↗</div>
          <h3>Context by relevance</h3>
          <p>Graph traversal + edge weights build the context window. Relevant history from 10 sessions ago beats verbatim recency.</p>
        </div>
        <div class="feature-card">
          <div class="icon">⎇</div>
          <h3>Multi-agent coordination</h3>
          <p>Subagents write nodes into a shared graph. Any agent can traverse them. No message-passing, no shared state bugs.</p>
        </div>
        <div class="feature-card">
          <div class="icon">◉</div>
          <h3>Navigate your reasoning</h3>
          <p>The VS Code extension shows chat and graph side-by-side. Click any node to see what triggered it and what it produced.</p>
        </div>
        <div class="feature-card">
          <div class="icon">△</div>
          <h3>Task DAG</h3>
          <p>Tasks form a directed acyclic graph with <code>depends_on</code> edges — visible, trackable, resumable.</p>
        </div>
        <div class="feature-card">
          <div class="icon">▣</div>
          <h3>Single Rust binary</h3>
          <p>No Docker, no runtime dependencies, no Python. One static binary + SQLite. Ships anywhere.</p>
        </div>
      </div>
    </div>
  </section>

  <section class="architecture">
    <div class="container">
      <h2>Architecture</h2>
      <div class="arch-diagram">graphirm/
├── crates/
│   ├── graph/    ← GraphStore (rusqlite + petgraph) — every node + edge
│   ├── llm/      ← 17+ providers via rig-core (Anthropic, OpenAI, DeepSeek…)
│   ├── tools/    ← bash, read, write, edit, grep, find, ls
│   ├── agent/    ← async loop, context engine, multi-agent, knowledge extraction
│   ├── server/   ← axum REST + SSE API
│   └── tui/      ← ratatui terminal UI
└── graphirm-vscode/   ← VS Code extension (chat + d3 graph explorer)</div>
    </div>
  </section>

  <section class="install" id="install">
    <div class="container">
      <h2>Get started</h2>
      <div class="install-steps">
        <div class="install-step">
          <div class="step-num">1</div>
          <div class="step-body">
            <h4>Build from source (Rust required)</h4>
            <code>git clone https://github.com/graphirm/graphirm && cd graphirm && cargo build --release</code>
          </div>
        </div>
        <div class="install-step">
          <div class="step-num">2</div>
          <div class="step-body">
            <h4>Set your LLM key and start the server</h4>
            <code>export ANTHROPIC_API_KEY=sk-... && ./target/release/graphirm serve</code>
          </div>
        </div>
        <div class="install-step">
          <div class="step-num">3</div>
          <div class="step-body">
            <h4>Install the VS Code extension</h4>
            <p>Download the latest <code>.vsix</code> from <a href="https://github.com/graphirm/graphirm/releases">Releases</a>, then in VS Code: <strong>Extensions → ⋯ → Install from VSIX</strong></p>
          </div>
        </div>
        <div class="install-step">
          <div class="step-num">4</div>
          <div class="step-body">
            <h4>Open the panel</h4>
            <code>Ctrl+Shift+P → Graphirm: Open</code>
          </div>
        </div>
      </div>
    </div>
  </section>
</main>

<footer>
  <div class="container">
    <span>Graphirm — MIT License</span>
    <div style="display:flex;gap:20px">
      <a href="https://github.com/graphirm/graphirm">GitHub</a>
      <a href="https://github.com/graphirm/graphirm/blob/main/README.md">Docs</a>
    </div>
  </div>
</footer>

</body>
</html>
```

**Step 5: Commit**

```bash
git add graphirm-site/
git commit -m "feat: add graphirm.ai static landing page"
```

---

### Task 2: GitHub Pages deploy + custom domain

**Files:**
- Create: `graphirm-site/CNAME` (already done — contains `graphirm.ai`)
- No CI workflow needed — GitHub Pages deploys automatically from a branch/folder

**Step 1: Enable GitHub Pages in repo settings**

1. Go to `https://github.com/graphirm/graphirm/settings/pages`
2. Source: **Deploy from a branch**
3. Branch: `main`, Folder: `/graphirm-site`
4. Click Save — GitHub will build and publish within ~60 seconds

**Step 2: Point Namecheap DNS at GitHub Pages**

In Namecheap → Domain List → `graphirm.ai` → Manage → Advanced DNS:

Add these four `A` records (GitHub Pages IPs):

| Type | Host | Value |
|------|------|-------|
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |

Add one `CNAME` record:

| Type | Host | Value |
|------|------|-------|
| CNAME | www | graphirm.github.io |

DNS propagates within 5–30 minutes. GitHub auto-provisions HTTPS once the domain resolves.

**Step 3: Verify**

```bash
# After DNS propagates:
curl -I https://graphirm.ai
# Expected: HTTP/2 200
```

**Step 4: Commit**

```bash
git add graphirm-site/
git commit -m "feat: graphirm.ai static landing page (GitHub Pages)"
git push origin main
```

---

## Track B: Product Polish

### Task 3: Session restoration on server restart

**Files:**
- Modify: `crates/server/src/routes.rs` — restore sessions in `start_server`
- Modify: `crates/server/src/state.rs` — add `restore_sessions` helper

**Background:** Currently `AppState.sessions` is an empty `HashMap` on every boot. Conversation content is safe in SQLite, but the API returns an empty session list after restart. Fix: query the graph for all `Agent` nodes on startup and reconstruct `SessionHandle` entries.

**Step 1: Add `restore_sessions` to `state.rs`**

```rust
/// Rebuild the session registry from persisted Agent nodes in the graph.
/// Called once at server startup. Restored sessions are non-live (no
/// cancellation token, no join handle) — they support GET but not new prompts
/// until a full Session is re-attached.
pub async fn restore_sessions(
    graph: &GraphStore,
    sessions: &tokio::sync::RwLock<HashMap<SessionId, SessionHandle>>,
) {
    use graphirm_graph::nodes::NodeType;

    let agent_nodes = match graph.all_nodes_of_type("Agent") {
        Ok(nodes) => nodes,
        Err(e) => {
            tracing::warn!("Session restore failed — could not query graph: {e}");
            return;
        }
    };

    let mut map = sessions.write().await;
    let mut count = 0usize;

    for node in agent_nodes {
        let NodeType::Agent(ref data) = node.node_type else { continue };
        let id = SessionId(node.id.to_string());
        if map.contains_key(&id) { continue; } // already live

        let name = node
            .metadata
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("restored")
            .to_string();
        let status = data.status;
        let created_at = node.created_at;

        map.insert(id, SessionHandle {
            session: Arc::new(Session::restore(graph.clone(), node.id.clone(), data.clone())),
            name,
            signal: CancellationToken::new(),
            join_handle: None,
            status,
            created_at,
        });
        count += 1;
    }

    if count > 0 {
        tracing::info!("Restored {count} sessions from graph");
    }
}
```

**Step 2: Add `Session::restore` constructor in `crates/agent/src/session.rs`**

```rust
/// Reconstruct a Session from a persisted Agent node.
/// The session is read-only until a new agent loop is attached.
pub fn restore(graph: Arc<GraphStore>, agent_id: NodeId, data: AgentData) -> Self {
    Self {
        id: agent_id,
        graph,
        agent_config: AgentConfig {
            name: data.name,
            model: data.model,
            ..AgentConfig::default()
        },
    }
}
```

**Step 3: Call `restore_sessions` in `start_server`**

In `crates/server/src/routes.rs`, after building `AppState`:

```rust
// Restore persisted sessions from graph before accepting requests
restore_sessions(&state.graph, &state.sessions).await;
```

**Step 4: Add `all_nodes_of_type` to `GraphStore`**

In `crates/graph/src/store.rs`:

```rust
pub fn all_nodes_of_type(&self, type_name: &str) -> Result<Vec<GraphNode>, GraphError> {
    let conn = self.pool.get()?;
    let mut stmt = conn.prepare(
        "SELECT id, node_type, created_at, updated_at, metadata
         FROM nodes WHERE json_extract(node_type, '$.type') = ?1"
    )?;
    let rows = stmt.query_map([type_name], |row| {
        // ... deserialise row to GraphNode (same pattern as existing query methods)
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(GraphError::Sqlite)
}
```

**Step 5: Test manually**

```bash
# Start server, create a session, kill server, restart, verify session appears
cargo run -- serve &
curl -s -X POST localhost:3000/api/sessions -d '{"name":"persist-test"}' | jq .id
pkill -f "graphirm serve"
cargo run -- serve &
sleep 3
curl -s localhost:3000/api/sessions | jq '.[].name'
# Expected: "persist-test" appears
```

**Step 6: Commit**

```bash
git add crates/
git commit -m "feat: restore sessions from graph on server restart"
```

---

### Task 4: DAG timeline layout in graph explorer

**Files:**
- Modify: `graphirm-vscode/media/graph.js` — add timeline layout option
- Modify: `graphirm-vscode/media/index.html` — add layout toggle button
- Modify: `graphirm-vscode/media/styles.css` — style the toggle

**Background:** The current d3-force layout is a blob — all nodes repel each other into a random arrangement. A timeline layout assigns x-position from `created_at` (oldest left, newest right) and y-position from node type (`Agent` top, `Interaction` middle, `Content`/`Task` bottom). Edge types are colour-coded.

**Step 1: Add layout toggle to `index.html`**

In the graph pane toolbar, after `#reset-zoom-btn`:

```html
<button id="layout-toggle-btn" title="Toggle layout">Timeline</button>
```

**Step 2: Add timeline layout logic to `graph.js`**

```javascript
let _layoutMode = 'force'; // 'force' | 'timeline'

// Y-position by node type
const TYPE_Y = {
  Agent: 80,
  Task: 160,
  Interaction: 260,
  Content: 360,
  Knowledge: 440,
};

function timelinePositions(nodes) {
  if (!nodes.length) return;
  const times = nodes.map(n => new Date(n.created_at).getTime());
  const tMin = Math.min(...times);
  const tMax = Math.max(...times);
  const tRange = tMax - tMin || 1;
  const padding = 60;
  nodes.forEach(n => {
    const t = new Date(n.created_at).getTime();
    n.fx = padding + ((t - tMin) / tRange) * (_width - padding * 2);
    n.fy = TYPE_Y[n.node_type?.type] ?? 260;
  });
}

function releasePositions(nodes) {
  nodes.forEach(n => { n.fx = null; n.fy = null; });
}

// Edge colours by type
const EDGE_COLORS = {
  RespondsTo: '#ffffff44',
  Reads: '#3b82f688',
  Modifies: '#f9731688',
  Produces: '#4ade8088',
  DependsOn: '#a855f788',
  SpawnedBy: '#ec489988',
};
```

In `renderGraph`, colour edges by type:

```javascript
const link = g.append('g').selectAll('line')
  .data(edges)
  .join('line')
  .attr('stroke', d => EDGE_COLORS[d.edge_type] ?? '#ffffff22')
  .attr('stroke-width', 1.5);
```

Add toggle handler in `initGraph`:

```javascript
document.getElementById('layout-toggle-btn').addEventListener('click', () => {
  _layoutMode = _layoutMode === 'force' ? 'timeline' : 'force';
  document.getElementById('layout-toggle-btn').textContent =
    _layoutMode === 'force' ? 'Timeline' : 'Force';
  if (_currentData) renderGraph(_currentData);
});
```

In `renderGraph`, branch on layout mode before starting simulation:

```javascript
if (_layoutMode === 'timeline') {
  timelinePositions(nodes);
  _simulation.alpha(0).stop();
} else {
  releasePositions(nodes);
  _simulation.alpha(0.3).restart();
}
```

**Step 3: Store current graph data for re-render on toggle**

```javascript
let _currentData = null;

export function handleGraphMessage(msg) {
  if (msg.type === 'graph' && msg.data) {
    _currentData = msg.data;
    renderGraph(msg.data);
  }
  // ...
}
```

**Step 4: Build extension**

```bash
cd graphirm-vscode && npm run build
```

**Step 5: Commit**

```bash
git add graphirm-vscode/
git commit -m "feat: add DAG timeline layout toggle to graph explorer"
```

---

### Task 5: Agent Trace export CLI subcommand

**Files:**
- Modify: `src/main.rs` — add `export` subcommand
- Create: `crates/graph/src/export.rs` — Agent Trace serialiser

**Background:** Agent Trace is an open spec (CC BY 4.0) for linking code changes to AI conversations. Graphirm's graph is a strict superset; this is a JSON serialiser over existing data.

**Step 1: Define Agent Trace types in `crates/graph/src/export.rs`**

```rust
use serde::Serialize;
use crate::nodes::{GraphNode, NodeType};
use crate::store::GraphStore;
use crate::error::GraphError;

#[derive(Serialize)]
pub struct AgentTraceRecord {
    pub version: &'static str,
    pub session_id: String,
    pub turns: Vec<TraceTurn>,
}

#[derive(Serialize)]
pub struct TraceTurn {
    pub id: String,
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<TraceToolCall>,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct TraceToolCall {
    pub id: String,
    pub name: String,
    pub result: String,
}

pub fn export_session(graph: &GraphStore, session_id: &str) -> Result<AgentTraceRecord, GraphError> {
    let nodes = graph.get_session_thread(session_id)?;
    let mut turns = Vec::new();

    for node in &nodes {
        let NodeType::Interaction(ref data) = node.node_type else { continue };
        if data.role == "tool" { continue; } // merged into parent turn below

        let tool_calls = graph
            .get_tool_results_for(&node.id)?
            .into_iter()
            .filter_map(|n| {
                if let NodeType::Interaction(ref d) = n.node_type {
                    Some(TraceToolCall {
                        id: n.id.to_string(),
                        name: n.metadata.get("tool_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        result: d.content.clone(),
                    })
                } else { None }
            })
            .collect();

        turns.push(TraceTurn {
            id: node.id.to_string(),
            role: data.role.clone(),
            content: data.content.clone(),
            tool_calls,
            created_at: node.created_at.to_rfc3339(),
        });
    }

    Ok(AgentTraceRecord { version: "0.1", session_id: session_id.to_string(), turns })
}
```

**Step 2: Add `export` subcommand to `src/main.rs`**

```rust
/// Export a session to a standard interchange format.
#[derive(clap::Args, Debug)]
struct ExportArgs {
    /// Session ID to export.
    session_id: String,
    /// Output format.
    #[arg(long, default_value = "agent-trace")]
    format: String,
    /// Output file (default: stdout).
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,
}
```

In the match arm:

```rust
Commands::Export(args) => {
    let graph = open_graph(&config)?;
    let record = graphirm_graph::export::export_session(&graph, &args.session_id)?;
    let json = serde_json::to_string_pretty(&record)?;
    match args.output {
        Some(path) => std::fs::write(&path, json)?,
        None => println!("{json}"),
    }
}
```

**Step 3: Test**

```bash
# Get a session ID
curl -s localhost:3000/api/sessions | jq '.[0].id'
# Export it
cargo run -- export <session-id> | jq '.turns | length'
```

**Step 4: Commit**

```bash
git add src/main.rs crates/graph/src/export.rs crates/graph/src/lib.rs
git commit -m "feat: add 'graphirm export --format agent-trace' CLI subcommand"
```

---

## Progress Tracking

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Static landing page HTML/CSS | ✅ |
| Task 2 | GitHub Pages deploy + custom domain | ✅ |
| Task 3 | Session restoration on restart | ✅ |
| Task 4 | DAG timeline layout toggle | ✅ |
| Task 5 | Agent Trace export CLI | ⬜ |
