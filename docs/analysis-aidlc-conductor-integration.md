# Analysis: AI-DLC & Conductor — Should We Integrate?

**Date:** 2026-03-20
**Repos analysed:**
- [awslabs/aidlc-workflows](https://github.com/awslabs/aidlc-workflows) (799 stars) — MIT-0
- [gemini-cli-extensions/conductor](https://github.com/gemini-cli-extensions/conductor) (3 249 stars) — Apache 2.0

**Our stack:** Cursor + superpowers skills (`~/.cursor/skills/`), Cursor rules (`.cursor/rules/*.mdc`), `AGENTS.md` hierarchy, `docs/plans/` for implementation plans. Also use **OpenCode** on other projects (reads `AGENTS.md`, supports skills in `~/.agents/skills/*/SKILL.md` and `.agents/skills/*/SKILL.md`). Cross-project: same skill directory shared across all repos.

**Cross-platform constraint:** Any reusable skills/rules must work across **both Cursor and OpenCode**. The portable intersection is:
- **Skills:** `~/.agents/skills/*/SKILL.md` — discovered by both OpenCode (natively) and Cursor (via `~/.cursor/skills/` or Claude-compatible paths)
- **Project context:** `AGENTS.md` — read by both Cursor and OpenCode; project-level source of truth
- **Project instructions:** `opencode.json` `instructions` field can glob `.cursor/rules/*.md`, so Cursor rules can be shared
- **Cursor-only:** `.cursor/rules/*.mdc` frontmatter format (OpenCode doesn't load these directly, but can reference them via `opencode.json`)

---

## 1. What Each Project Actually Does

### AI-DLC (AI-Driven Development Life Cycle)

A **methodology encoded as agent rules** — markdown files that steer any coding agent through a three-phase software lifecycle:

| Phase | Purpose | Always/Conditional |
|-------|---------|-------------------|
| **Inception** | What to build and why | Workspace Detection (always), Reverse Engineering (brownfield), Requirements (always, adaptive depth), User Stories (conditional), Workflow Planning (always), Application Design (conditional), Units Generation (conditional) |
| **Construction** | How to build it | Per-unit loop: Functional Design → NFR → Infrastructure → Code Generation (always) → Build & Test (always) |
| **Operations** | Deploy & run (placeholder) | Not yet implemented |

**Key mechanisms:**
- **Adaptive depth** — simple requests get minimal treatment, complex ones get comprehensive analysis
- **Brownfield/greenfield detection** — scans workspace for existing code before deciding what to do
- **Mandatory audit trail** — every user input logged with ISO timestamps in `aidlc-docs/audit.md`
- **Extension system** — opt-in rules (security, compliance) loaded lazily to save context
- **Overconfidence prevention** — explicit philosophy of "ask more questions, make fewer assumptions"
- **Content validation** — Mermaid diagram syntax checking, ASCII art standards
- **Human-in-the-loop gates** — explicit approval required between every phase
- **Plan-level checkbox tracking** — two-tier checkbox system (task + stage)

**Form factor:** Pure markdown rules. No code, no runtime, no dependencies. Copy files into `.cursor/rules/`, `CLAUDE.md`, `.clinerules/`, etc. Agent reads them and follows the protocol.

### Conductor (Gemini CLI Extension)

A **project management framework** implemented as a Gemini CLI extension. Structures development into context → spec → plan → implement cycles:

| Command | Purpose |
|---------|---------|
| `/conductor:setup` | One-time project scaffolding — product definition, tech stack, guidelines, workflow, code style guides, skill catalog |
| `/conductor:newTrack` | Create a feature/bug "track" — interactive spec → plan generation with user approval |
| `/conductor:implement` | Execute plan tasks sequentially — TDD workflow, phase checkpoints, git notes, coverage gates |
| `/conductor:status` | Progress dashboard — parsed from `tracks.md` + individual `plan.md` files |
| `/conductor:revert` | Git-aware undo — resolves logical units (track/phase/task) to commit SHAs |
| `/conductor:review` | Code review — diff analysis, style compliance, test execution, suggested fixes |

**Key mechanisms:**
- **Track = unit of work** — each track gets `spec.md`, `plan.md`, `metadata.json` in `conductor/tracks/<id>/`
- **TDD workflow** — write failing test → implement → refactor → verify coverage → commit → git note
- **Phase completion verification** — automated tests + manual verification plan presented to user
- **Git notes for audit** — task summaries attached to commits via `git notes`
- **Post-completion doc sync** — updates `product.md`, `tech-stack.md` when a track finishes
- **Skill catalog** — detection signals match project files to downloadable skills (Firebase, GCP, etc.)
- **Policies** — TOML rules restricting file operations in plan mode

**Form factor:** Gemini CLI extension (TOML command files + markdown templates). Requires Gemini CLI. Uses `ask_user` tool for structured interaction.

---

## 2. What We Actually Have (Corrected Picture)

After reading the project in depth, the setup is materially more sophisticated than a surface scan suggests. There are **two layers**, and an important **native advantage** in Graphirm's own architecture:

### Layer 1: Cursor rules (`.cursor/rules/`)
- `000-skills-first.mdc` — mandatory skills-first gate before any action
- `001-router.mdc` — context-aware rule loading
- `002-meta-generator.mdc` — self-improving rule system: detects patterns, proposes new rules, scores effectiveness. Already an "extension system."
- `003-code-quality.mdc` — code standards
- `100-rust-standards.mdc` — Rust-specific
- `102-project-management.mdc` — design-first task workflow
- `103-response-quality.mdc` — communication standards
- `tasklist.mdc` — 295-line design-first task file system with full template, git integration, progress tracking
- `graphirm-project.mdc`, `graphirm-context-always.mdc` — project-specific always-loaded context
- `106-git.mdc`, `107-documentation.mdc`, `103-testing.mdc`

### Layer 2: Superpowers skills (`~/.cursor/skills/`)
14 skills covering the full lifecycle: brainstorming → writing-plans → using-git-worktrees → subagent-driven-development → executing-plans → verification-before-completion → finishing-a-development-branch → requesting-code-review → receiving-code-review → test-driven-development → systematic-debugging → dispatching-parallel-agents → writing-skills + using-superpowers.

Plus Cursor-specific meta-skills: `create-rule`, `create-skill`, `create-subagent`, `migrate-to-skills`.

### Execution pipeline (`00-execution-strategy.md`)
A documented multi-agent architecture: **Controller** reads plan → dispatches **Implementer subagent** per task (TDD, self-review, commit) → dispatches **Spec Reviewer** subagent → dispatches **Code Quality Reviewer** subagent → two-stage review loops until both approve → `finishing-a-development-branch`. This is more rigorous than Conductor's implement loop.

### Important distinction: Graphirm is the product, not the tool
Graphirm is what we're *building*. We build it using Cursor + Claude + superpowers skills. Our development decisions, plan approvals, brainstorming outcomes — they happen in Cursor chat sessions, not in Graphirm sessions. Cursor transcripts are ephemeral `.jsonl` files, unstructured and not queryable. None of our development workflow is stored in the Graphirm graph.

This means gaps like "structured audit trail" and "decision history" are **real gaps in our development workflow** — not things we can wave away by pointing at Graphirm's own graph data model.

**Future possibility:** If we eventually dogfood Graphirm to build Graphirm, then yes — the graph *would* be the audit trail, Task nodes *would* be track management, etc. But that's a product maturity milestone, not a current workflow solution.

### Where the actual gaps are

| Concept | AI-DLC | Conductor | Our Stack | Gap? |
|---------|--------|-----------|-----------|------|
| Requirements / brainstorming | Inception phase | Setup + newTrack spec | `brainstorming` skill | Minor |
| Implementation planning | Workflow Planning | `plan.md` generation | `writing-plans` skill | Minor |
| Plan execution | Code Generation + checkbox | `/conductor:implement` | `subagent-driven-development` | **None — ours is more rigorous** |
| Per-task two-stage review | Not explicit | Not explicit | Spec + Code Quality reviewer subagents | **None — ours has this, theirs don't** |
| TDD | Construction phase | Full TDD workflow | `test-driven-development` skill | None |
| Debugging | Not explicit | Not explicit | `systematic-debugging` skill | None |
| Git workflow | Not explicit | Git-aware revert | `using-git-worktrees` + `finishing-a-development-branch` | Partial (revert by logical unit is missing) |
| Verification gate | Approval at each phase | Phase completion verification | `verification-before-completion` | Minor |
| Project context | Workspace detection, RE artifacts | `product.md`, `tech-stack.md` | `AGENTS.md` hierarchy | Minor |
| **Adaptive depth** | Explicit 3-level system | Implicit | **Not formalised** | **Real gap** |
| **Overconfidence prevention** | Explicit guide + red flags | Implicit | **Not formalised** | **Real gap** |
| Audit trail | `audit.md` with timestamps | Git notes | **Graph IS the audit** — native to Graphirm | Gap only in surfacing it |
| Extension system | Opt-in rule extensions | Skill catalog | `002-meta-generator.mdc` (auto-generates rules) | Minor |
| Task registry / progress dashboard | `aidlc-state.md` | `tracks.md` | `docs/plans/00-execution-strategy.md` + plan checkboxes | Minor |
| **Internal workflow fragmentation** | N/A | N/A | `tasklist.mdc` vs `writing-plans` skill — two competing approaches | **Real gap (internal)** |
| **Structured audit / decision log** | `audit.md` with timestamps | Git notes + plan updates | **Nothing** — Cursor chat transcripts are ephemeral | **Real gap** |
| Post-completion doc sync | N/A | Updates `product.md` / `tech-stack.md` | Manual `AGENTS.md` updates | Real but manageable |
| Git-aware revert by logical unit | N/A | `/conductor:revert` | Not present | Nice to have |

---

## 3. What's Actually Novel and Valuable (Revised)

### Real gaps — worth addressing:

**1. Adaptive depth formalisation (AI-DLC)**
The single most applicable idea. Our skills don't scale their output to problem complexity. AI-DLC explicitly defines factors: request clarity, complexity, scope, risk level, available context, user preferences. The principle "create exactly the detail needed — no more, no less" should be embedded in `brainstorming` and `writing-plans`. A one-line config change and a multi-crate architecture shift currently get treated identically.

**2. Overconfidence prevention (AI-DLC)**
AI-DLC found through production use that agents skip questions and make assumptions. Their explicit guide has a specific anti-pattern catalogue and red flags ("stages completing without asking any questions on complex projects", "proceeding with vague responses", "making assumptions instead of asking"). Our `brainstorming` skill says "ask clarifying questions" but lacks this systematic stance. Worth adding to `brainstorming` directly.

**3. Internal workflow fragmentation (our own problem)**
We have two competing task management approaches: `tasklist.mdc` / `102-project-management.mdc` (design-first, physical task files) versus `writing-plans` / `executing-plans` / `subagent-driven-development` skills (TDD-oriented, plans in `docs/plans/`). They have different philosophies, different file formats, and are invoked inconsistently. This isn't from AI-DLC or Conductor — it's a gap we need to resolve independently. Any integration should pick one and consolidate.

**4. Post-completion doc sync (Conductor)**
`AGENTS.md` is updated manually after each phase and frequently drifts. Conductor's pattern of proposing diffs to project context documents after completing a track is the right model. For us this means: after `finishing-a-development-branch`, automatically propose what needs updating in `AGENTS.md` based on what changed. This is small but high-value.

**5. Structured audit / decision log (AI-DLC)**
Our development decisions happen in ephemeral Cursor chat sessions. When we brainstorm, approve a plan, or choose approach A over B — there's no structured record of *why*. Plan checkboxes record *what* was done, not *what was considered and rejected*. AI-DLC's audit approach (ISO-timestamped log of user inputs, AI responses, and decision context per stage) fills a real gap. This doesn't need to be as heavyweight as AI-DLC's mandatory logging — a lightweight decision log appended during `brainstorming` and `finishing-a-development-branch` would capture the high-value moments without overhead.

### Worth considering but lower priority:

**6. Git-aware revert by logical unit (Conductor)**
Nice to have. Conductor's revert resolves a "track" to all its commits, handles rewritten history, presents an execution plan. Ours is `git revert <sha>`. Not urgent but elegant.

**7. Brownfield/greenfield detection (AI-DLC)**
Our `brainstorming` skill says "check files, docs, recent commits" informally. AI-DLC's systematic check (dependency manifests, source dirs, git state) is slightly more rigorous. Low priority since we know our own projects.

### Not worth adopting:

1. **AI-DLC's heavyweight inception phases** — user stories, application design, units generation. Enterprise greenfield complexity; we're extending a known Rust codebase.
2. **AI-DLC's content validation** — Mermaid syntax checking, ASCII art standards. Niche.
3. **Conductor's Gemini-specific infrastructure** — `ask_user` tool, plan mode policies, `/skills reload`. Implementation is tightly coupled to Gemini CLI.
4. **Conductor's skill catalog** — Firebase/GCP detection signals. Wrong domain.
5. **AI-DLC's question-file format** — `[Answer]:` in markdown files. Workaround for tools without interactive prompts; Cursor handles this natively.
6. **Track-based file management from Conductor** — `conductor/tracks/<id>/spec.md + plan.md + metadata.json`. We already have `docs/plans/` + `00-execution-strategy.md`. The *concept* is sound but the implementation is already covered.
7. **Structured audit.md** — the graph IS the audit trail. Duplicating it in markdown would be regression.

### Cross-platform skill placement strategy

Both Cursor and OpenCode support skills in `SKILL.md` format with YAML frontmatter. The discovery paths:

| Location | Cursor | OpenCode |
|----------|--------|----------|
| `~/.cursor/skills/*/SKILL.md` | Native | Not discovered |
| `~/.agents/skills/*/SKILL.md` | Not discovered by default | Native |
| `~/.claude/skills/*/SKILL.md` | Not discovered by default | Native (Claude compat) |
| `.agents/skills/*/SKILL.md` (project) | Not discovered by default | Native |
| `~/.config/opencode/skills/*/SKILL.md` | Not discovered | Native |

Current superpowers skills live at `~/.cursor/skills/superpowers/skills/`. For OpenCode portability, options:
1. **Symlink:** `ln -s ~/.cursor/skills ~/.agents/skills` — simplest, one source of truth
2. **Dual placement:** Copy or script-sync between directories
3. **Move to `~/.agents/skills/`** and configure Cursor to read from there

OpenCode's name validation is stricter: lowercase alphanumeric with single hyphens only (`using-superpowers` is valid, but the parent directory `superpowers/skills/` nesting may not match OpenCode's flat `skills/<name>/SKILL.md` layout).

### Future: Graphirm as its own development tool
Once Graphirm is mature enough to dogfood (use Graphirm to develop Graphirm), several of these concepts could be handled natively by the product:

| Concept | External approach | Future Graphirm-native approach |
|---------|-----------------|------------------------|
| Track management | `conductor/tracks/*.md` files | `Task` nodes in the graph, linked via `DependsOn` edges |
| Audit trail | `audit.md` with timestamps | Every interaction already stored as a node with timestamp |
| Project context | `product.md`, `tech-stack.md` | Knowledge nodes surfaced by `repo_briefing` |
| Progress tracking | Checkbox files | Task node status (`pending` → `in_progress` → `done`) |
| Decision history | Audit entries | `session_trace` tool querying Interaction chains |

This is Graphirm's long-term product opportunity: both AI-DLC and Conductor manage project context in markdown files that drift. A graph-native approach could manage it in a persistent, queryable, cross-session graph. **But this is a product roadmap item, not a current workflow solution.** For now, we build with Cursor skills and markdown plans like everyone else.

---

## 4. Recommendation (Revised)

**Three things to actually do, in priority order:**

### Priority 1 — Fix the internal inconsistency (no external source needed)
Decide between `tasklist.mdc` + `102-project-management.mdc` (design-first task files) and `writing-plans` + `executing-plans` + `subagent-driven-development` (TDD-oriented plan skills). They're parallel systems. Either consolidate into one, or define clear triggers in `001-router.mdc` for which to use when. This is more valuable than anything from AI-DLC or Conductor.

### Priority 2 — Add adaptive depth + overconfidence prevention (from AI-DLC)
Both are additions to existing skills, not new infrastructure:
- **`brainstorming` skill**: Add explicit depth assessment (5 factors), add overconfidence anti-pattern catalogue, add structured brownfield/greenfield detection check
- **`writing-plans` skill**: Add depth-scaling instruction — minimal for simple changes, comprehensive for complex ones

These are reusable across any project since they live in `~/.cursor/skills/`.

### Priority 3 — Add post-completion doc sync (from Conductor)
Add to `finishing-a-development-branch` skill: after all tasks complete and tests pass, explicitly prompt to review `AGENTS.md` (or the project's equivalent context file) and propose specific diffs for what changed. Reusable across projects — every project has some equivalent of `AGENTS.md`.

### Priority 4 — Add lightweight decision log (from AI-DLC)
Add to `brainstorming` and `finishing-a-development-branch` skills: append key decisions (what was considered, what was chosen, why) to a `docs/decisions.md` or per-plan decision section. Not mandatory per-interaction logging like AI-DLC — just the high-value decision points. Reusable across projects.

### Lower priority / optional:
- Git-aware revert skill (new skill, portable)
- Graphirm product features: Task nodes in UI, `session_trace` tool (already in backlog) — these are product work, not workflow improvements

**Reusability and portability:**
- Skills in priorities 2-4 should live in a location both Cursor and OpenCode discover: **`~/.agents/skills/*/SKILL.md`** is the portable path (OpenCode reads it natively; Cursor can be configured to read it)
- Alternatively, keep skills in `~/.cursor/skills/` (current location) and symlink or copy to `~/.agents/skills/` for OpenCode
- Skill names must be **lowercase-alphanumeric-with-hyphens** to satisfy OpenCode's stricter validation (e.g. `adaptive-depth`, not `adaptiveDepth`)
- Priority 1 (workflow fragmentation) involves `.cursor/rules/*.mdc` which is Cursor-only — for OpenCode projects, the equivalent guidance goes in `AGENTS.md` or a referenced file via `opencode.json` `instructions`
- `AGENTS.md` is the universal project context file — works in Cursor, OpenCode, Claude Code, and AI-DLC
- Nothing should be graphirm-specific except Graphirm product features

---

## 5. What Not To Do

- Don't import the AI-DLC rule files directly — they're 500+ lines of mandatory enterprise protocol
- Don't build a Gemini CLI extension — we're Cursor + Claude, skills are our extension mechanism
- Don't add `aidlc-docs/` or `conductor/` directory structures — `docs/plans/` and `AGENTS.md` are correct
- Don't add AI-DLC's mandatory per-interaction `audit.md` logging — too heavyweight; a lightweight decision log at brainstorming/completion gates is sufficient
- Don't add `product.md`, `tech-stack.md` etc. — `AGENTS.md` hierarchy already serves this purpose
- Don't implement NFR/infrastructure design stages — enterprise complexity we don't need
- Don't create a third task management approach — resolve the existing fragmentation first
- Don't mistake sophistication for quality — our two-stage review pipeline (spec + quality) is already more rigorous than either AI-DLC or Conductor's equivalent
- Don't confuse the product with the tool — Graphirm is what we're building, not what we're building with; our development workflow gaps are real and can't be solved by pointing at features in the unfinished product

---

## 6. Files Consulted

**AI-DLC:**
- `aidlc-rules/aws-aidlc-rules/core-workflow.md` (540 lines — the entire workflow)
- `aidlc-rules/aws-aidlc-rule-details/common/process-overview.md` (with Mermaid diagram)
- `aidlc-rules/aws-aidlc-rule-details/common/depth-levels.md` (adaptive depth philosophy)
- `aidlc-rules/aws-aidlc-rule-details/common/overconfidence-prevention.md`
- `aidlc-rules/aws-aidlc-rule-details/inception/workspace-detection.md`

**Conductor:**
- `commands/conductor/setup.toml` (584 lines — the full setup protocol)
- `commands/conductor/newTrack.toml` (spec + plan generation)
- `commands/conductor/implement.toml` (TDD execution loop)
- `commands/conductor/revert.toml` (git-aware revert)
- `commands/conductor/review.toml` (code review protocol)
- `commands/conductor/status.toml` (progress dashboard)
- `policies/conductor.toml` (plan mode access control)
- `templates/workflow.md` (TDD workflow template)
- `skills/catalog.md` (skill detection signals)
- `GEMINI.md` (file resolution protocol)
- `gemini-extension.json` (extension metadata)

**Our stack:**
- `AGENTS.md` (project context)
- `docs/backlog.md` (current state)
- `.cursor/rules/*.mdc` (workspace rules)
- `~/.cursor/skills/superpowers/skills/` (14 skills)
