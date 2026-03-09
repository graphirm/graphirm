# Graphirm Execution Strategy

> **For Claude:** This document describes the overall execution strategy. Each phase has its own implementation plan in this directory. Use superpowers:subagent-driven-development to execute each phase plan.

**Goal:** Build a graph-native coding agent in Rust — from single-agent MVP through full platform with web visualization.

**Architecture:** Cargo workspace with 6 crates (graph, llm, tools, agent, tui, server). Graph-native data model where every interaction is a node. Hand-rolled async agent loop for MVP, adk-graph migration post-MVP.

**Tech Stack:** Rust, rusqlite + petgraph (graph), rig-core (LLM), tokio (async), ratatui (TUI), axum (server), thiserror + clap + tracing + serde

---

## Phase Plans

| Phase | Plan File | Tasks | Status | Dependencies |
|-------|-----------|-------|--------|-------------|
| Phase 0: Scaffold | `2026-03-03-phase-0-scaffold.md` | 11 | **done** | none |
| Phase 1: Graph Store | `2026-03-03-phase-1-graph-store.md` | 14 | **done** | Phase 0 |
| Phase 2: LLM Provider | `2026-03-03-phase-2-llm-provider.md` | 11 | **done** | Phase 1 |
| Phase 3: Tool System | `2026-03-03-phase-3-tool-system.md` | 13 | **done** | Phase 1 |
| Phase 4: Agent Loop | `2026-03-03-phase-4-agent-loop.md` | 11 | **done** | Phase 2 + 3 |
| Phase 5: Multi-Agent | `2026-03-03-phase-5-multi-agent.md` | 11 | **done** | Phase 4 + 6 |
| Phase 6: Context Engine | `2026-03-03-phase-6-context-engine.md` | 12 | **done** | Phase 1 + 4 |
| Phase 7: TUI | `2026-03-03-phase-7-tui.md` | 10 | **done** | Phase 4 |
| Phase 8: HTTP Server | `2026-03-03-phase-8-http-server.md` | 14 | **done** | Phase 5 + 7 |
| Phase 9: Knowledge Layer | `2026-03-03-phase-9-knowledge-layer.md` | 16 | **in progress** | Phase 8 |
| Phase 10: Usage Discovery | `2026-03-05-phase-10-usage-discovery.md` | 13 | **done** | Phase 8 |
| Phase 11: VS Code Extension | `2026-03-05-phase-11-vscode-extension.md` | 11 | **done** | Phase 8 + 10 |
| Phase 12: Landing + Polish | `2026-03-05-phase-12-landing-and-polish.md` | 5 | **done** | Phase 11 |

## Release Milestones

**MVP (Phases 0-4 + 7):** Single-agent coding assistant with graph-tracked interactions and TUI. 70 tasks. ✅ complete.

**v1.0 (+ Phases 5-6):** Multi-agent with graph-based context engine. +23 tasks. ✅ complete.

**v2.0 (+ Phases 8-10):** Full platform with HTTP server, knowledge layer, and usage discovery. +43 tasks. ⏳ in progress (Phase 9 knowledge layer: GLiNER2 ONNX inference complete, memory wiring pending).

**v2.1 (+ Phase 11):** VS Code/Cursor extension — two-pane chat + d3-force graph, SSE streaming, session management. +11 tasks. ✅ complete.

**v3.0 (+ Phase 12):** graphirm.ai landing page + product polish — static site, session restore, DAG timeline layout, Agent Trace export. +5 tasks. ✅ complete.

**Total: 154 tasks across 13 phases.**

## Session State Fixes (2026-03-09)

Two bugs found while running programmatic multi-turn session tests:

| Fix | Crate | What |
|-----|-------|------|
| Wrong system prompt in `AgentConfig::default()` | `graphirm-agent` | Bare `"You are a helpful coding assistant."` caused DeepSeek to wrap answers in `bash echo` calls, keeping sessions stuck in `Running` permanently. Fixed by moving the full Graphirm system prompt (with explicit tool-use guidance) into `AgentConfig::default()`. |
| Premature knowledge extraction hang | `config/default.toml` | `[knowledge] enabled = true` triggered a post-turn LLM call using `ExtractionConfig` default model `"gpt-4o-mini"` (wrong for DeepSeek), which hung indefinitely. Disabled until Phase 9 wiring is complete. |

Verified fix: 15-turn session, all turns `completed`, 31 nodes / 59 edges, ~70s total.

## Post-Dogfooding Fixes (2026-03-05)

Significant bugs discovered and fixed during real-LLM dogfooding session on a Hetzner spoke (DeepSeek `deepseek-chat`):

| Fix | Crate | What |
|-----|-------|------|
| Context builder interleaving | `graphirm-agent` | `Content`/`Knowledge` nodes were sorted chronologically between tool-call and tool-result messages, causing 400 "insufficient tool messages" errors from all OpenAI-compatible providers. Fixed by placing context documents before the conversation thread. |
| Silent agent loop failures | `graphirm-server` | `SessionStatus::Failed` was set without logging the error. Added `tracing::error!` before status update. |
| Silent LLM call failures | `graphirm-agent` | Per-turn LLM errors propagated via `?` without any log entry. Added per-turn error logging. |
| Model name not passed | `graphirm` (binary) | `GRAPHIRM_MODEL` env var was parsed but model name was discarded; `AgentConfig` used default Claude model. Fixed to pass extracted model name. |

## Critical Path

```
P0 (scaffold) → P1 (graph store) → P2 + P3 (parallel) → P4 (agent loop) → P7 (TUI) = MVP ✅
                                                          ↓
                                                   P6 (context) → P5 (multi-agent) = v1.0 ✅
                                                                        ↓
                                   P7 + P5 → P8 (server) → P9 (knowledge) + P10 (discovery) = v2.0 ✅
                                                                                 ↓
                                                                           P11 (VS Code ext) = v2.1 ✅
```

## Execution Model

### Skills Chain Per Phase

```
using-git-worktrees → subagent-driven-development → finishing-a-development-branch
```

### Agent Roles

| Role | Type | What It Does |
|------|------|-------------|
| **Controller** | Parent agent | Reads plan, extracts tasks, dispatches subagents, answers questions, integrates results |
| **Implementer** | generalPurpose subagent | Implements one task following TDD, self-reviews, commits |
| **Spec Reviewer** | generalPurpose subagent (readonly) | Verifies implementation matches plan spec |
| **Code Quality Reviewer** | generalPurpose subagent (readonly) | Reviews Rust idioms, error handling, thread safety |

### Phase Grouping

**MVP:**
- **Phase 0:** Direct execution (too small for subagents)
- **Phase 1:** Subagent-driven development with TDD — foundation, gets most scrutiny
- **Phase 2 + 3:** Dispatched as parallel agents (independent after Phase 1)
- **Phase 4:** Subagent-driven development — depends on P2+P3 merge
- **Phase 7:** Subagent-driven development — depends on P4

**v1.0:**
- **Phase 6:** Subagent-driven — upgrades MVP context builder
- **Phase 5:** Subagent-driven — depends on P4+P6

**v2.0:**
- **Phase 8:** Subagent-driven — HTTP server wrapping agent + graph
- **Phase 9 + 10:** Parallel agents — knowledge layer and web UI are independent

### Branching Strategy

Each phase gets its own branch off main:
- `phase-0/scaffold`
- `phase-1/graph-store`
- `phase-2/llm-provider`
- `phase-3/tool-system`
- `phase-4/agent-loop`
- `phase-5/multi-agent`
- `phase-6/context-engine`
- `phase-7/tui`
- `phase-8/http-server`
- `phase-9/knowledge-layer`
- `phase-10/usage-discovery`
- `phase-11-vscode`

Merge to main after each phase passes review.
