//! Delegation to the external `pi` coding agent (`@earendil-works/pi-coding-agent`).
//!
//! Pi is run as a subprocess with `--mode json`, which prints one JSON event per
//! line on stdout. This module owns the pieces needed to drive such a run:
//!
//! - [`events`] — pure line parser turning each JSON line into a [`PiEvent`].
//!   No process or graph dependencies; reused unchanged for `--mode rpc`.
//!
//! - [`process`] — spawns `pi --mode json` as a subprocess, drains its stdout
//!   into a stream of [`PiEvent`]s, and kills the process group on cancel or
//!   timeout ([`spawn_pi`], [`probe_version`], [`build_argv`]).
//!
//! - [`graph`] — writes one delegation to the graph in the same shape as an
//!   in-process subagent: Task, Pi Agent, `role:"tool"` / `role:"assistant"`
//!   Interaction nodes, and the final Task result ([`PiRun`]).
//!
//! The `delegate_pi` tool itself lands in a later task of
//! `docs/plans/2026-09-28-pi-delegate-executor.md`.

pub mod events;
pub mod graph;
pub mod process;

pub use events::{MAX_ERROR_CHARS, PiEvent, flatten_content, flatten_result, parse_line};
pub use graph::{PI_EXECUTOR, PI_TASK_TITLE, PiRun, PiRunFinish, PiToolCall};
pub use process::{
    MAX_INLINE_TASK_BYTES, MAX_LINE_BYTES, PiProcessError, PiRunHandle, PiRunOutcome, PiSpawnSpec,
    STDERR_TAIL_BYTES, build_argv, expand_binary, probe_version, spawn_pi,
};
