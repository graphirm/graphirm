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
//! - [`tool`] — the `delegate_pi` [`Tool`](graphirm_tools::Tool) that ties the
//!   three together ([`PiDelegateTool`]), its registration
//!   ([`register_pi_delegate`]) and the system-prompt notice
//!   ([`apply_pi_delegate_system_notice`]).

pub mod events;
pub mod graph;
pub mod label;
pub mod pieces;
pub mod process;
pub mod tool;

pub use events::{MAX_ERROR_CHARS, PiEvent, flatten_content, flatten_result, parse_line};
pub use graph::{FAILURE_ABANDONED, PI_EXECUTOR, PI_TASK_TITLE, PiRun, PiRunFinish, PiToolCall};
pub use label::{LabeledReply, final_reply_texts, load_final_replies, run_label_session};
pub use process::{
    MAX_INLINE_TASK_BYTES, MAX_LINE_BYTES, PiProcessError, PiRunHandle, PiRunOutcome, PiSpawnSpec,
    STDERR_TAIL_BYTES, build_argv, expand_binary, probe_version, spawn_pi,
};
pub use tool::{
    PI_DELEGATE_TOOL_NAME, PiDelegateTool, apply_pi_delegate_system_notice, register_pi_delegate,
};
