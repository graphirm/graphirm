# Graphirm Documentation

Welcome to the Graphirm project documentation. This folder contains planning documents, feature documentation, architecture notes, and completion logs.

## Structure

### 📋 **Core Documents**

- **`backlog.md`** — Active backlog with 2 deferred items for Phase 12+
- **`completion-log.md`** — Complete list of shipped features with implementation details
- **`usage-discovery-findings.md`** — Findings from real-world dogfooding sessions

### 🎯 **Planning & Execution**

**`plans/`** folder contains:
- **`00-execution-strategy.md`** — Master plan for all 13 phases (154 total tasks)
- **`2026-03-06-dag-timeline-layout-completion.md`** — Latest active feature (DAG timeline layout)
- **`2026-03-06-phase-12-task-5-agent-trace-export.md`** — Agent Trace export task plan

**Completed phase plans** have been archived to `archive/plans/` for reference.

### 🎁 **Features**

**`features/`** folder documents shipped capabilities:
- **`session-restoration.md`** — Sessions survive server restarts with full history
- **`phase-13-planning.md`** — Early planning notes for Phase 13+

### 📦 **Advanced Topics**

**`backlog/`** folder:
- **`phase-13-advanced-features.md`** — 7 major feature categories for Phase 13+
  - Cross-session knowledge extraction
  - Custom tool plugins
  - Agent Trace ingestion
  - DevOps automation suite
  - Graph-native debugging
  - Team collaboration layer
  - Web UI with real-time collaboration

### 📚 **Archive**

**`archive/plans/`** — Completed phase plans (Phases 0-12) kept for reference:
- 13 phase implementation plans (Phases 0-12)
- 2 superseded DAG timeline variants (design & implementation drafts)

---

## Quick Navigation

### I want to...

**Understand the project** → Start with `plans/00-execution-strategy.md`

**See what's shipped** → Read `completion-log.md`

**Check what's next** → See `backlog.md` (2 Phase 12 items) or `backlog/phase-13-advanced-features.md` (7 Phase 13+ items)

**Review a phase** → Check `plans/` for current/active work, or `archive/plans/` for completed phases

**Learn about a feature** → See `features/` folder

**Understand real-world issues** → Read `usage-discovery-findings.md`

---

## Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| execution-strategy.md | Active (154/154 tasks complete) | 2026-03-06 |
| backlog.md | Active (2 items, Phases 12+) | 2026-03-06 |
| completion-log.md | Active | 2026-03-06 |
| phase-13-advanced-features.md | Planning | 2026-03-05 |
| session-restoration.md | Shipped ✅ | 2026-03-06 |
| dag-timeline-layout-completion.md | Shipped ✅ | 2026-03-06 |
| agent-trace-export task | Shipped ✅ | 2026-03-06 |

---

## Current Release Status

- **v3.0 (Main)** — Full platform shipped with graph-native agent, VS Code extension, landing page, session restoration, DAG timeline layout, and Agent Trace export
- **v3.1 (Next)** — Phase 13 advanced features (parallel work on multiple feature tracks)

---

## Notes

- Plans are MIT licensed
- All completed work is documented in `completion-log.md`
- Archived phase plans are kept for reference and historical context
- This docs folder is not version-controlled (see `.gitignore`) but snapshots are committed when major work completes
