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
//! Graph recording and the `delegate_pi` tool land in later tasks of
//! `docs/plans/2026-09-28-pi-delegate-executor.md`.

pub mod events;
pub mod process;

pub use events::{MAX_ERROR_CHARS, PiEvent, flatten_content, flatten_result, parse_line};
pub use process::{
    MAX_INLINE_TASK_BYTES, MAX_LINE_BYTES, PiProcessError, PiRunHandle, PiRunOutcome, PiSpawnSpec,
    STDERR_TAIL_BYTES, build_argv, expand_binary, probe_version, spawn_pi,
};
