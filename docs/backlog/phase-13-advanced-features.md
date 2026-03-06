# Phase 13: Advanced Features — Backlog

**Status:** Planning document for post-v3.0 development

**Vision:** Extend Graphirm from single-session tool to enterprise-grade knowledge platform with cross-session intelligence, extensibility, and production scalability.

---

## Overview

Phase 13 contains 7 independent features that can be tackled in parallel or sequenced based on priority. Each has been sized and assessed for dependencies.

---

## 1. Cross-Session Knowledge Extraction

**Complexity:** Medium | **Time estimate:** 2-3 weeks | **Dependencies:** Graph store (Phase 1)

### Problem
Currently, knowledge is trapped within individual sessions. When a developer starts a new session, Graphirm cannot leverage learnings from similar problems solved in past sessions.

### Solution
Automatic entity recognition and linking system that:
- Extracts **concepts, patterns, code snippets, errors, solutions** from any session
- Creates **Knowledge nodes** with embeddings (HNSW vector search)
- Cross-references similar Knowledge nodes across sessions
- Surfaces relevant past solutions when new problems appear
- Builds a searchable **knowledge graph** that grows with usage

### Value
- Dramatically reduced context switching (agent can reference past work)
- Developer tribal knowledge becomes system memory
- 10-100x faster problem-solving on repeated issue types
- Emergent expertise: system learns what works in your codebase

### Technical approach
- Entity extraction: LLM-driven or rule-based pattern recognition
- Vector embeddings: Use existing LLM to embed extracted entities
- Storage: New Knowledge nodes in graph store with HNSW indexing
- Query: Semantic search on Knowledge nodes during context building
- Deduplication: Similar Knowledge nodes merged by cosine similarity

### Key decisions needed
- When to extract? (after each session, batch job, or on-demand)
- Extract from what? (conversations, tool outputs, test results)
- How aggressive with deduplication? (false negatives vs. storage)
- Privacy: Store embeddings or just hashes?

---

## 2. Custom Tool Plugins

**Complexity:** Medium | **Time estimate:** 2-3 weeks | **Dependencies:** Tool system (Phase 3), TUI/Server (Phases 7-8)

### Problem
Graphirm ships with built-in tools (bash, read, write, grep) but cannot run user-defined tools like:
- Python REPL for data analysis
- SQL queries against databases
- Docker commands for containerized workflows
- Custom scripts (deploy.sh, test.sh, etc.)
- API calls to external services

### Solution
Plugin system for user-defined tools:
- **Plugin interface:** Simple trait-based system (`Tool` trait in graphirm-tools)
- **Registration:** CLI flag or config file to load plugins at startup
- **Execution:** Same security sandbox as built-in tools (token limits, timeouts)
- **Caching:** Tool result caching to avoid redundant expensive operations
- **Versioning:** Multiple versions of same tool runnable in same session

### Value
- Users can extend Graphirm without recompiling Rust
- Enables domain-specific workflows (ML teams, DevOps, data teams, etc.)
- Reduces need for "agent prompts" to work around missing tools
- Opens ecosystem for tool marketplace

### Technical approach
- Define `Tool` trait with execute(), name(), description() methods
- Load plugins from `~/.graphirm/plugins/` directory (*.so or WASM)
- OR: Support script-based tools (shell, Python) with subprocess spawning
- Tool registry updated dynamically at startup
- Permissions model: whitelist allowed tools by session/user

### Key decisions needed
- Plugin format: Native rust (.so) vs. WASM vs. scripts?
- Sandboxing: How strict? (separate process, containers, limits)
- Error handling: What if plugin crashes? Isolation?
- Permission model: Trust all plugins or per-session allowlist?

---

## 3. Agent Trace Ingestion

**Complexity:** Medium | **Time estimate:** 1-2 weeks | **Dependencies:** Graph store (Phase 1), Agent Trace export (Phase 12)

### Problem
Graphirm exports sessions as Agent Trace, but cannot import them back or import traces from other agents (Claude Projects, Anthropic Workbench, other frameworks).

### Solution
Bidirectional Agent Trace support:
- **Import:** Read Agent Trace JSON and reconstruct session in Graphirm graph
- **Normalize:** Map other agents' tool calls to Graphirm's tool model
- **Merge:** Combine multiple traces into composite sessions
- **Compare:** Analyze how different agents solved same problem
- **Replay:** Step through imported session in TUI/extension

### Value
- Consolidate work from multiple AI tools into single Graphirm graph
- Compare approaches across agents
- Migrate sessions between Graphirm instances
- Contribute shared sessions to community
- Integration with other agents' outputs

### Technical approach
- Define `import_agent_trace(json_path) -> Result<Session>`
- Validate trace structure against Agent Trace spec
- Transform turns → Interaction nodes
- Map tool calls → tool execution metadata
- Preserve timestamps, nesting, IDs
- Handle edge cases (malformed traces, unknown tool types)

### Key decisions needed
- Preserve original tool names or normalize to Graphirm tools?
- How to handle UUID collisions on import?
- Should imported sessions be read-only or editable?
- Merge strategy for duplicate sessions?

---

## 4. Web UI Enhancements

**Complexity:** Medium-High | **Time estimate:** 3-4 weeks | **Dependencies:** HTTP server (Phase 8), existing web UI (Phase 11)

### Problem
Current web UI (VS Code extension) is basic: chat panel + static graph view. Enterprise users need:
- **Graph filtering:** Hide/show node types, edge types, time ranges
- **Search:** Full-text search over message content and metadata
- **Timeline scrubbing:** Jump to any point in session, replay from there
- **Annotation:** Add tags, notes, highlights to turns or nodes
- **Export variants:** HTML report, PDF, Markdown summary
- **Collaboration:** Multi-user viewing (read-only or edit)
- **Performance:** Handle 10k+ node graphs smoothly

### Solution
Phased web UI improvements:
1. **Graph controls:** Filter pane (show/hide node types, time range slider)
2. **Search:** Elasticsearch or SQLite FTS integration, search UI
3. **Timeline scrubber:** Interactive timeline at bottom, click to jump
4. **Annotations:** Right-click → Add tag/note, persist to Knowledge nodes
5. **Export:** Generate HTML/PDF/Markdown exports
6. **Collaboration:** Session sharing link (read-only), collaborative cursors (optional)

### Value
- Enterprise readiness: multi-user, search, reporting
- Knowledge capture: annotations create institutional memory
- Analytics: Export enables dashboards and analysis
- Usability: Graph filtering reduces cognitive overload on large sessions

### Technical approach
- **Graph filtering:** Client-side + graph query parameters
- **Search:** Add FTS index to graph store, query endpoint
- **Timeline:** SVG timeline component with event markers
- **Annotations:** New Annotation node type, UI modal
- **Export:** Template-based HTML + print-to-PDF
- **Collaboration:** Session ACL, WebSocket for collaborative cursors

### Key decisions needed
- Search backend: FTS vs. Elasticsearch vs. Meilisearch?
- Timeline granularity: Per-turn or per-minute?
- Annotation storage: Separate table or Annotation nodes?
- Real-time collaboration or just sharing links?

---

## 5. Performance Optimization

**Complexity:** Medium-High | **Time estimate:** 2-3 weeks | **Dependencies:** Graph store (Phase 1), context engine (Phase 6)

### Problem
Graphirm's context engine (Phase 6) traverses the entire graph for every request. On large graphs (10k+ nodes), this becomes slow:
- Session list: 500ms → should be 50ms
- Message fetch: 1s → should be 100ms
- Graph query: 2-3s → should be 200ms

### Solution
Multi-layer performance improvements:
- **Query caching:** Cache frequent queries (session list, node by ID) with TTL
- **Graph indexing:** B-tree indices on session_id, node_type, created_at
- **Lazy loading:** Return summary at first, details on demand
- **Pagination:** Limit results to 100 items by default, allow offset/limit
- **Connection pooling:** Use r2d2 pool more aggressively
- **Async I/O:** Make context engine traversals async
- **Batch operations:** Combine multiple queries when possible

### Value
- **User experience:** Fast UI response, no loading spinners
- **Scale:** Support sessions with 100k+ nodes
- **Cost:** Reduce memory/CPU usage, fewer server resources needed
- **Reliability:** Better handling of concurrent users

### Technical approach
- **Caching:** Redis or in-memory cache with TTL
- **Indexing:** Add SQLite indices, profile slow queries
- **Pagination:** Offset/limit parameters on all list endpoints
- **Lazy loading:** Separate "list" and "detail" endpoints
- **Async:** Convert context engine to async streams
- **Profiling:** Use flamegraph, criterion benchmarks

### Key decisions needed
- Cache backend: Redis vs. in-memory vs. SQLite query cache?
- Cache TTL strategy: Per-query or global?
- Pagination defaults: 50 vs. 100 vs. 500?
- Async: Full async/await or thread pool?

---

## 6. API Versioning & Stability

**Complexity:** Low-Medium | **Time estimate:** 1-2 weeks | **Dependencies:** HTTP server (Phase 8)

### Problem
Current REST API (Phase 8) is v0, unstable, breaking changes expected. Users need stability guarantees.

### Solution
Formal API versioning and contract:
- **Versioning:** `/api/v1/`, `/api/v2/`, etc. with sunset schedule
- **Stability:** Semantic versioning, backwards compatibility within major version
- **Deprecation:** 6-month notice for breaking changes
- **Documentation:** OpenAPI/Swagger spec for each version
- **SDKs:** Official SDK for Python, JavaScript, Go
- **Rate limiting:** Per-user rate limits, quota management

### Value
- **Trust:** Production users can depend on API stability
- **Ecosystem:** Third-party tools can build on top
- **Migration path:** Clear upgrade path for users
- **Discoverability:** OpenAPI/Swagger enables auto-generated clients

### Technical approach
- Create `/api/v1/` routes alongside current endpoints
- Extract request/response types into versioned structs
- Add ApiVersion header/parameter handling
- Generate OpenAPI spec from route definitions
- Create SDKs using code generation from OpenAPI

### Key decisions needed
- V1 scope: Just stabilize Phase 8, or include new features?
- SDK priority: Python first, or all three in parallel?
- Rate limiting: Token bucket, sliding window, or quota per plan?

---

## 7. Multi-User Support

**Complexity:** High | **Time estimate:** 3-4 weeks | **Dependencies:** HTTP server (Phase 8), API versioning (Feature 6 above)

### Problem
Currently, Graphirm has no user concept—all sessions are stored in one shared database. For teams:
- Can't share sessions between users
- No audit trail of who changed what
- No permission model (read, write, delete)
- No user accounts or authentication

### Solution
Multi-user architecture:
- **Users:** User accounts, identity via OAuth2 (GitHub, Google) or API keys
- **Sessions:** Sessions owned by user, shareable with other users
- **Permissions:** Per-session: owner, collaborator, viewer
- **Audit trail:** Track who created/modified each session
- **Quotas:** Rate limits, storage limits per user/plan
- **Admin panel:** Manage users, quotas, billing (if SaaS)

### Value
- **Teams:** Collaborate on sessions, share findings
- **Enterprise:** Audit, compliance, multi-tenancy
- **SaaS:** Foundation for paid tiers, feature gating
- **Trust:** Clear ownership and permission model

### Technical approach
- Add `User` table with OAuth integration
- Add `SessionPermission` table (user, session, role)
- Add `AuditLog` table tracking mutations
- Add auth middleware to all endpoints
- Add quota checking in request handlers
- Optional: Stripe integration for billing

### Key decisions needed
- Authentication: OAuth2, SAML, both, or just API keys?
- Sharing model: Direct user invite or public links?
- Billing: Freemium model, seat-based, usage-based?
- Multi-tenancy: Shared database or separate DBs per tenant?

---

## Feature Dependencies & Sequencing

```
Legend: → (depends on)

Core Infrastructure:
  Phase 1 (Graph Store) → Phase 3 (Tools) → Phase 4 (Agent Loop)

Phase 13 Features:
  1. Knowledge Extraction: Phase 6 (Context) + embeddings
  2. Tool Plugins: Phase 3 (Tool System)
  3. Trace Ingestion: Phase 1 + Phase 12 (Export)
  4. Web UI: Phase 8 (HTTP Server) + Phase 11 (Web UI)
  5. Performance: Phase 1 (Graph Store)
  6. API Versioning: Phase 8 (HTTP Server) ← Phase 5 (Multi-Agent)
  7. Multi-User: Feature 6 (API Versioning)

Suggested order:
  Priority 1 (Early wins):
    - Performance Optimization (improves everything)
    - Tool Plugins (extends use cases)
    
  Priority 2 (Scalability):
    - Knowledge Extraction (cross-session learning)
    - Trace Ingestion (ecosystem integration)
    
  Priority 3 (Enterprise):
    - API Versioning (stability)
    - Web UI Enhancements (usability)
    - Multi-User Support (teams)
```

---

## Estimation Summary

| Feature | Complexity | Est. Time | Dependencies | Priority |
|---------|------------|-----------|--------------|----------|
| Knowledge Extraction | Medium | 2-3w | Phase 6 | High |
| Tool Plugins | Medium | 2-3w | Phase 3 | High |
| Trace Ingestion | Medium | 1-2w | Phase 1, 12 | Medium |
| Web UI Enhancements | Med-High | 3-4w | Phase 8, 11 | Medium |
| Performance | Med-High | 2-3w | Phase 1 | High |
| API Versioning | Low-Med | 1-2w | Phase 8 | Low* |
| Multi-User | High | 3-4w | Feature 6 | Low* |

*Low priority for solo dev, high for teams/SaaS

---

## Success Criteria (Phase 13 Complete)

- ✅ At least 3 features shipped (recommended: Knowledge, Plugins, Performance)
- ✅ All features have unit tests (>80% coverage)
- ✅ Performance tests show 3-5x improvement on large graphs
- ✅ New features documented in README and API docs
- ✅ v3.1 released with features

---

## Next Action

Choose 2-3 features from this backlog and create detailed implementation plans using the `writing-plans` skill. Start with Priority 1 (Performance Optimization + Tool Plugins) for maximum impact.

---

**Backlog created:** March 6, 2026
**Graphirm version:** 3.0.0
**Status:** Ready for Phase 13 planning
