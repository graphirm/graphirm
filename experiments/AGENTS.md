# Experiments

One-off probes, demos, and scratch tools **outside** the main workspace crates. They are optional;
the root `graphirm` binary does **not** depend on them.

## Conventions

- **Each subdirectory** that ships a runnable (Rust crate, script, notebook) should have either:
  - a short **`AGENTS.md`** in that subdirectory (purpose, prerequisites, run command, how to read output), or
  - a clearly marked subsection **here** if the folder is tiny and not worth a second file.
- Prefer documenting **env vars** (e.g. `GLINER2_MODEL_DIR`) and **feature flags** next to the run instructions.
- When an experiment graduates into the product, fold behavior into the relevant crate and **`docs/plans/`**, then remove or archive the experiment.

## Contents

| Path | What |
|------|------|
| `gliner2-segment-probe/` | CLI: GLiNER2 segment spans on a file/stdin (`AGENTS.md` inside). |

## How to test

There is no unified `cargo test` for this tree. Run commands listed in each subfolder’s `AGENTS.md` or `README.md`.
