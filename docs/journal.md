# Graphirm Decision Journal

Why things are the way they are. `docs/completion-log.md` records **what** shipped;
this file records **why** — decisions, rejected alternatives, and reversals.
Newest entry first. One entry per decision, not per session.

Add an entry whenever a choice is made that a future reader could not reconstruct from
the code or the plan alone: architecture, dependency, convention, measured trade-off,
something tried and abandoned. Skip routine work.

Entry template:

```markdown
## YYYY-MM-DD — <decision in one line>

**Context:** what forced the choice.
**Decision:** what was chosen.
**Alternatives:** what was considered and why it lost.
**Consequences:** what this commits us to, what to watch.
**Refs:** plan / commit / eval / transcript.
```

---

## 2026-09-28 — Governance docs must be updated in the same commit as the work

**Context:** Audit found the scaffolding (root + 7 scoped `AGENTS.md`, 3 Cursor rules, hooks,
1,500-line backlog, completion log) was mature but drifting: `backlog.md` header said
"Phases 0–35" while `AGENTS.md` said 0–55; `docs/README.md` was stuck at Phase 12;
`CHANGELOG.md` frozen since 2026-03-08; ~60% of `backlog.md` was completed items despite the
file's own convention to move them out. Root cause: no rule, hook, or skill obliged anyone to
update these files. The only doc-update obligation (`000-skills-first.mdc` → update
`00-execution-strategy.md`) pointed at a document abandoned in March. No decision journal existed.

**Decision:** Extend `.cursor/rules/000-skills-first.mdc` with a "Governance docs" block —
plan link on start, ✅ + completion-log entry on ship, journal entry on non-obvious decisions,
backlog entry for discovered-but-unstarted work — all in the same commit as the work. Drop the
`00-execution-strategy.md` line. Create this file. Add a pointer block to `AGENTS.md` because
`.cursor/` is gitignored and only `AGENTS.md` reaches the spoke and other clones.

**Alternatives:** (a) A separate `200-governance.mdc` rule — rejected; the existing rule already
owns "commit and progress rules", and a fourth always-apply rule adds context cost for no
isolation benefit. (b) A pre-commit hook that fails when `crates/` changes without a
`docs/` change — deferred; too blunt for refactors and hotfixes, revisit if drift recurs.
(c) ADR directory (`docs/decisions/NNNN-*.md`) instead of a single journal — rejected for now;
one file is easier to grep and append to at the current decision rate.

**Consequences:** Every agent session that ships anything now touches at least two docs. The
known drift (backlog header, `docs/README.md`, `CHANGELOG.md`, stale `src/main.rs` pointers in
`AGENTS.md`) is **not** fixed by this entry — it is separate cleanup work.

**Refs:** `.cursor/rules/000-skills-first.mdc`, `AGENTS.md` → Governance docs, commit `85ffc1a`
(same day: phase table moved out of `AGENTS.md` into `completion-log.md`).
