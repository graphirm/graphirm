//! Delegation to the external `pi` coding agent (`@earendil-works/pi-coding-agent`).
//!
//! Pi is run as a subprocess with `--mode json`, which prints one JSON event per
//! line on stdout. This module owns the pieces needed to drive such a run:
//!
//! - [`events`] — pure line parser turning each JSON line into a [`PiEvent`].
//!   No process or graph dependencies; reused unchanged for `--mode rpc`.
//!
//! Process spawning and graph recording land in later tasks of
//! `docs/plans/2026-09-28-pi-delegate-executor.md`.

pub mod events;

pub use events::{PiEvent, flatten_content, flatten_result, parse_line};
