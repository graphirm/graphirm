//! Pi subprocess wrapper: spawn `pi --mode json`, stream-parse its JSONL
//! stdout into [`PiEvent`]s, and tear the process (group) down on cancel or
//! timeout.
//!
//! Process model (design D1):
//!
//! ```text
//! <binary> --mode json -p --no-session <--approve|--no-approve>
//!          --provider <p> --model <m> <extra_args...> -- <task>
//! cwd = spec.cwd; stdin = null; stdout/stderr piped separately;
//! env inherited + PI_SKIP_VERSION_CHECK=1 − GRAPHIRM_API_KEY;
//! kill_on_drop; process_group(0) on unix
//! ```
//!
//! Pi reads its provider key from its own environment / auth store; this
//! module never reads, sets, or logs it. The env changes are exactly two:
//! `PI_SKIP_VERSION_CHECK=1` is added, and `GRAPHIRM_API_KEY` is removed —
//! graphirm's own server key is not something Pi or its `bash` children need.
//!
//! The task is passed inline unless it is longer than
//! [`MAX_INLINE_TASK_BYTES`] **or starts with `@`** (Pi treats a leading `@`
//! as a file reference and fails with `File not found`). In either case it is
//! written to a private (`0600`) temp file and passed as
//! `-- @<path> "<instruction>"`; Pi concatenates the file text and the message.
//!
//! A run is finished only when Pi's stdout closes *and* the process has
//! exited. `agent_end` is not a terminator: Pi auto-retries transient provider
//! errors (`AgentEnd { will_retry: true }` is followed by another
//! `agent_start`), so the reader drains to EOF unconditionally.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use graphirm_tools::process::{kill_group_and_reap, kill_process_group};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdout, Command};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::events::{PiEvent, parse_line};
use crate::config::PiConfig;

/// Longest stdout line we are willing to buffer. Longer lines are dropped
/// (counted in [`PiRunOutcome::oversized_lines`]) and reading resumes at the
/// next newline, so one pathological tool result cannot exhaust memory.
pub const MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

/// How much of stderr is kept, from the end.
pub const STDERR_TAIL_BYTES: usize = 4 * 1024;

/// Tasks longer than this are handed to Pi as `@<tempfile>` instead of argv.
pub const MAX_INLINE_TASK_BYTES: usize = 64 * 1024;

/// Message that follows the `@<file>` argument when the task goes through a
/// file, so the model receives an instruction rather than a bare file block.
pub const TASK_FILE_INSTRUCTION: &str = "Carry out the task described in the attached file.";

/// After Pi exits, how long the pipes may stay open (held by a straggling tool
/// child) before the group is killed and the run reported with
/// [`PiRunOutcome::pipes_lingered`].
pub const POST_EXIT_GRACE: Duration = Duration::from_secs(3);

/// Floor of the post-exit grace. When Pi exits right at the deadline the
/// remaining budget is ~0, which would kill the pipes before the final events
/// (already written by the exited process) were drained. The run may thus
/// overshoot `timeout` by at most this much.
pub const MIN_POST_EXIT_GRACE: Duration = Duration::from_millis(250);

/// Bound of the event channel handed to the consumer.
const EVENT_CHANNEL_CAP: usize = 256;

/// Wall-clock cap for `pi --version`.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Everything needed to start one Pi run. Borrows; [`spawn_pi`] clones what
/// the background driver needs.
pub struct PiSpawnSpec<'a> {
    /// `[agent.pi]` settings (binary, provider, model, trust, extra args).
    pub config: &'a PiConfig,
    /// Working directory for Pi (the session workspace). Must exist.
    pub cwd: &'a Path,
    /// Task text handed to Pi as the final positional argument.
    pub task: &'a str,
    /// Hard wall-clock cap; the process group is killed when exceeded.
    pub timeout: Duration,
}

/// Failures of the process layer. Provider/model errors reported by Pi itself
/// arrive as events, not here.
#[derive(Debug, thiserror::Error)]
pub enum PiProcessError {
    /// The binary could not be located (absolute path missing or not on PATH).
    #[error("pi not found at '{0}'")]
    NotFound(String),
    /// Any other failure to start the process (permissions, temp file, …).
    #[error("failed to spawn pi: {0}")]
    Spawn(String),
    /// The run exceeded its deadline and was killed.
    #[error("pi timed out after {0:?}")]
    Timeout(Duration),
    /// The cancellation token fired and the run was killed.
    #[error("cancelled")]
    Cancelled,
}

/// How a completed run ended. Only produced when Pi exited on its own.
#[derive(Debug, Clone)]
pub struct PiRunOutcome {
    /// Process exit code; `None` when Pi died from a signal.
    pub exit_code: Option<i32>,
    /// Last [`STDERR_TAIL_BYTES`] of stderr (lossy UTF-8).
    pub stderr_tail: String,
    /// Non-blank stdout lines that were not valid JSON.
    pub malformed_lines: u32,
    /// Stdout lines longer than [`MAX_LINE_BYTES`] that were skipped.
    pub oversized_lines: u32,
    /// Wall-clock time from spawn to exit.
    pub duration: Duration,
    /// Pi exited but something it left behind kept stdout/stderr open past
    /// [`POST_EXIT_GRACE`]; the process group was killed. Output captured up
    /// to that point is still reported.
    pub pipes_lingered: bool,
}

/// A running Pi process: a stream of parsed events plus the completion handle.
///
/// Dropping the handle aborts the driver task; the group-kill guard armed at
/// spawn then SIGKILLs Pi's whole process group (and `kill_on_drop` the direct
/// child). Consumers that want the outcome must await [`PiRunHandle::wait`]
/// (or `&mut handle.done`) before dropping.
pub struct PiRunHandle {
    /// Parsed events in stdout order. `PiEvent::Ignored` lines are not sent.
    /// Closes when Pi's stdout closes or the run is killed.
    pub events: mpsc::Receiver<PiEvent>,
    /// Resolves when Pi exits, times out, or is cancelled.
    pub done: JoinHandle<Result<PiRunOutcome, PiProcessError>>,
    pid: Option<u32>,
}

impl PiRunHandle {
    /// Pi's pid (also its process-group id), captured at spawn.
    pub fn pid(&self) -> Option<u32> {
        self.pid
    }

    /// Stop consuming events and await the run's outcome.
    ///
    /// Closes `events` first: the reader switches to discard mode, so a
    /// consumer that stopped reading cannot stall Pi on a full pipe (which
    /// would otherwise surface as a bogus `Timeout`). Events already buffered
    /// remain readable via `events.recv()` afterwards. A panicked driver task
    /// is reported as [`PiProcessError::Spawn`] rather than propagated.
    pub async fn wait(&mut self) -> Result<PiRunOutcome, PiProcessError> {
        self.events.close();
        match (&mut self.done).await {
            Ok(res) => res,
            Err(join_err) => Err(PiProcessError::Spawn(format!(
                "pi driver task failed: {join_err}"
            ))),
        }
    }
}

impl Drop for PiRunHandle {
    fn drop(&mut self) {
        self.done.abort();
    }
}

/// Exact argv (excluding argv\[0\]) per design D1. `task_args` is the
/// positional tail after `--`: the task itself, or `@<file>` + instruction.
pub fn build_argv(config: &PiConfig, task_args: &[&str]) -> Vec<String> {
    let mut argv: Vec<String> = ["--mode", "json", "-p", "--no-session"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    argv.push(
        if config.trust_project {
            "--approve"
        } else {
            "--no-approve"
        }
        .to_string(),
    );
    argv.push("--provider".to_string());
    argv.push(config.provider.clone());
    argv.push("--model".to_string());
    argv.push(config.model.clone());
    argv.extend(config.extra_args.iter().cloned());
    argv.push("--".to_string());
    argv.extend(task_args.iter().map(|s| s.to_string()));
    argv
}

/// Expand a leading `~` / `~/` to `$HOME`. Anything else (bare names resolved
/// via PATH, absolute paths, `~user/…`) is returned unchanged. When `HOME` is
/// unset the input is returned as-is.
pub fn expand_binary(binary: &str) -> PathBuf {
    let rest = if binary == "~" {
        Some("")
    } else {
        binary.strip_prefix("~/")
    };
    match (rest, std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => {
            let home = PathBuf::from(home);
            if rest.is_empty() {
                home
            } else {
                home.join(rest)
            }
        }
        _ => PathBuf::from(binary),
    }
}

/// Whether the task must travel through a file: too long for argv, or it
/// starts with `@`, which Pi would otherwise parse as a file path.
pub fn task_needs_file(task: &str) -> bool {
    task.len() > MAX_INLINE_TASK_BYTES || task.starts_with('@')
}

/// Base command shared by [`probe_version`] and [`spawn_pi`]: binary, stdio,
/// env policy, `kill_on_drop`.
fn pi_command(binary: &str) -> Command {
    let mut cmd = Command::new(expand_binary(binary));
    cmd
        // Pi runs in its own process group (spawn_pi); an inherited tty would
        // SIGTTIN it on read. Null stdin gives a clean EOF.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("PI_SKIP_VERSION_CHECK", "1")
        // graphirm's own API key is for graphirm's HTTP server; Pi and the
        // shells it spawns have no business seeing it.
        .env_remove("GRAPHIRM_API_KEY")
        // Safety net: if the owning future is dropped without reaching the
        // kill path, tokio SIGKILLs the direct child.
        .kill_on_drop(true);
    cmd
}

/// Run `<binary> --version` under a 5 s cap and return the trimmed first line
/// of stdout.
pub async fn probe_version(config: &PiConfig) -> Result<String, PiProcessError> {
    let mut cmd = pi_command(&config.binary);
    cmd.arg("--version");
    let child = cmd
        .spawn()
        .map_err(|e| map_spawn_error(e, &config.binary))?;
    // On timeout the future (and the `Child`) is dropped → kill_on_drop.
    let output = tokio::time::timeout(PROBE_TIMEOUT, child.wait_with_output())
        .await
        .map_err(|_| PiProcessError::Timeout(PROBE_TIMEOUT))?
        .map_err(|e| PiProcessError::Spawn(format!("pi --version: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PiProcessError::Spawn(format!(
            "pi --version exited with {}: {}",
            output.status,
            stderr.trim()
        )));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().next().unwrap_or("").trim().to_string())
}

/// Spawn Pi. Spawn-time failures ([`PiProcessError::NotFound`] /
/// [`PiProcessError::Spawn`], including a missing `cwd`) are returned
/// immediately; everything after that — exit, timeout, cancellation —
/// resolves through the returned handle.
///
/// Async only because the workspace check and an oversized / `@`-prefixed
/// task's temp file go through `tokio::fs` before the process starts.
pub async fn spawn_pi(
    spec: PiSpawnSpec<'_>,
    cancel: CancellationToken,
) -> Result<PiRunHandle, PiProcessError> {
    let binary = spec.config.binary.clone();
    match tokio::fs::metadata(spec.cwd).await {
        Ok(meta) if meta.is_dir() => {}
        _ => {
            return Err(PiProcessError::Spawn(format!(
                "workspace directory {} does not exist",
                spec.cwd.display()
            )));
        }
    }
    let temp_task = TempTask::for_task(spec.task).await?;
    let file_arg;
    let task_args: Vec<&str> = match &temp_task {
        Some(t) => {
            file_arg = format!("@{}", t.path.display());
            vec![file_arg.as_str(), TASK_FILE_INSTRUCTION]
        }
        None => vec![spec.task],
    };
    let argv = build_argv(spec.config, &task_args);

    let mut cmd = pi_command(&binary);
    cmd.args(&argv).current_dir(spec.cwd);
    // Own process group so the kill path takes down Pi *and* its tool children.
    #[cfg(unix)]
    cmd.process_group(0);

    let started = Instant::now();
    let mut child = cmd.spawn().map_err(|e| map_spawn_error(e, &binary))?;
    // Capture now: `child.id()` is `None` once the direct child is reaped, and
    // a grandchild outliving Pi is exactly when the group kill matters.
    let pgid = child.id();
    // Armed here, not in the driver: a handle dropped before the driver task
    // is ever polled must still take the whole group down (the guard is
    // dropped with the un-polled future).
    let group_guard = GroupKillGuard { pgid };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    tracing::debug!(
        pid = ?pgid,
        cwd = %spec.cwd.display(),
        provider = %spec.config.provider,
        model = %spec.config.model,
        via_temp_file = temp_task.is_some(),
        "spawned pi"
    );

    let (events_tx, events_rx) = mpsc::channel(EVENT_CHANNEL_CAP);
    let run = SpawnedRun {
        child,
        pgid,
        group_guard,
        stdout,
        stderr,
        started,
        timeout: spec.timeout,
        cancel,
    };
    let done = tokio::spawn(async move {
        let result = drive(run, events_tx).await;
        if let Some(t) = temp_task {
            t.cleanup().await;
        }
        result
    });
    Ok(PiRunHandle {
        events: events_rx,
        done,
        pid: pgid,
    })
}

fn map_spawn_error(e: std::io::Error, binary: &str) -> PiProcessError {
    if e.kind() == std::io::ErrorKind::NotFound {
        PiProcessError::NotFound(binary.to_string())
    } else {
        PiProcessError::Spawn(format!("{binary}: {e}"))
    }
}

/// Why the exit-wait `select!` returned.
enum Step {
    Exited(std::io::Result<std::process::ExitStatus>),
    Timeout,
    Cancelled,
}

/// Why the post-exit pipe-drain `select!` returned.
enum Drained {
    Closed,
    GraceExpired,
    Cancelled,
}

/// A freshly spawned Pi process plus everything the driver needs to see it out.
struct SpawnedRun {
    child: Child,
    pgid: Option<u32>,
    group_guard: GroupKillGuard,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
    started: Instant,
    timeout: Duration,
    cancel: CancellationToken,
}

/// Own the run until exit / timeout / cancel. Readers are spawned as tasks and
/// aborted on the way out if still running.
async fn drive(
    run: SpawnedRun,
    events_tx: mpsc::Sender<PiEvent>,
) -> Result<PiRunOutcome, PiProcessError> {
    let SpawnedRun {
        mut child,
        pgid,
        mut group_guard,
        stdout,
        stderr,
        started,
        timeout,
        cancel,
    } = run;
    let child = &mut child;
    let capture = Arc::new(Capture::new());
    let mut stdout_task = AbortOnDrop(tokio::spawn(drain_stdout(
        stdout,
        events_tx,
        Arc::clone(&capture),
    )));
    let mut stderr_task = AbortOnDrop(tokio::spawn(drain_stderr(stderr, Arc::clone(&capture))));

    // Phase 1: wait for the process to exit.
    let step = tokio::select! {
        status = child.wait() => Step::Exited(status),
        _ = tokio::time::sleep(timeout) => Step::Timeout,
        _ = cancel.cancelled() => Step::Cancelled,
    };
    // Whatever happens next, the guard's job is done: either the direct child
    // has exited (its pid, hence the pgid, may be recycled once the group is
    // empty — never signal it blindly), or we kill the group explicitly.
    group_guard.disarm();
    let status = match step {
        Step::Exited(status) => status,
        Step::Timeout => {
            tracing::warn!(?timeout, "pi timed out; killing process group");
            kill_group_and_reap(child, pgid).await;
            return Err(PiProcessError::Timeout(timeout));
        }
        Step::Cancelled => {
            tracing::info!("pi run cancelled; killing process group");
            kill_group_and_reap(child, pgid).await;
            return Err(PiProcessError::Cancelled);
        }
    };
    let exit_code = match status {
        Ok(s) => s.code(),
        Err(e) => {
            tracing::warn!(error = %e, "waiting on pi failed");
            None
        }
    };

    // Phase 2: the pipes close when every holder exits. A tool child Pi left
    // behind can keep them open; give it a short grace (bounded by what is
    // left of the deadline, floored at MIN_POST_EXIT_GRACE so an exit at the
    // deadline still drains the last events), then kill the group through the
    // captured pgid and report what was captured so far.
    let grace = timeout
        .saturating_sub(started.elapsed())
        .min(POST_EXIT_GRACE)
        .max(MIN_POST_EXIT_GRACE);
    let drained = {
        let readers = async {
            let _ = (&mut stdout_task.0).await;
            let _ = (&mut stderr_task.0).await;
        };
        tokio::pin!(readers);
        tokio::select! {
            _ = &mut readers => Drained::Closed,
            _ = tokio::time::sleep(grace) => Drained::GraceExpired,
            _ = cancel.cancelled() => Drained::Cancelled,
        }
    };
    let pipes_lingered = match drained {
        Drained::Closed => false,
        Drained::GraceExpired => {
            tracing::warn!(
                ?exit_code,
                grace_ms = grace.as_millis() as u64,
                "pi exited but its pipes stayed open; killing stragglers"
            );
            kill_group_and_reap(child, pgid).await;
            stdout_task.0.abort();
            stderr_task.0.abort();
            true
        }
        Drained::Cancelled => {
            kill_group_and_reap(child, pgid).await;
            return Err(PiProcessError::Cancelled);
        }
    };

    let duration = started.elapsed();
    let malformed_lines = capture.malformed.load(Ordering::Relaxed);
    let oversized_lines = capture.oversized.load(Ordering::Relaxed);
    tracing::debug!(
        ?exit_code,
        malformed_lines,
        oversized_lines,
        pipes_lingered,
        duration_ms = duration.as_millis() as u64,
        "pi exited"
    );
    Ok(PiRunOutcome {
        exit_code,
        stderr_tail: capture.stderr_tail(),
        malformed_lines,
        oversized_lines,
        duration,
        pipes_lingered,
    })
}

/// SIGKILL Pi's process group if dropped while armed (the driver future was
/// dropped — or never polled — before Pi exited). `kill(2)` is a single
/// non-blocking syscall, so this is safe to do in `Drop` on the runtime.
struct GroupKillGuard {
    pgid: Option<u32>,
}

impl GroupKillGuard {
    fn disarm(&mut self) {
        self.pgid = None;
    }
}

impl Drop for GroupKillGuard {
    fn drop(&mut self) {
        if let Some(pgid) = self.pgid {
            match kill_process_group(pgid) {
                Ok(()) => tracing::info!(pgid, "pi driver dropped; killed process group"),
                Err(e) => tracing::debug!(pgid, error = %e, "group kill on drop"),
            }
        }
    }
}

/// Abort a spawned task when dropped (e.g. when `drive` returns early).
struct AbortOnDrop<T>(JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Where a Pi `--mode json` run is copied, if recording is on.
///
/// `GRAPHIRM_PI_RUNS_DIR=off` or an empty value disables recording. A path
/// overrides the default. With no variable, production runs record under
/// `$HOME/.graphirm/pi-runs`. Tests do not record unless the variable is set,
/// so the suite does not write into the home directory.
pub(super) fn resolve_pi_runs_dir(
    env_value: Option<&str>,
    home: Option<&Path>,
    record_by_default: bool,
) -> Option<PathBuf> {
    match env_value {
        Some("") | Some("off") => None,
        Some(path) => Some(PathBuf::from(path)),
        None if record_by_default => home.map(|dir| dir.join(".graphirm/pi-runs")),
        None => None,
    }
}

fn open_pi_run_record() -> Option<std::fs::File> {
    let dir = resolve_pi_runs_dir(
        std::env::var("GRAPHIRM_PI_RUNS_DIR").ok().as_deref(),
        std::env::var_os("HOME").as_deref().map(Path::new),
        !cfg!(test),
    )?;
    if let Err(error) = std::fs::create_dir_all(&dir) {
        tracing::warn!(dir = %dir.display(), %error, "pi run record directory was not created");
        return None;
    }
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("{millis}-{}.jsonl", std::process::id()));
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
    {
        Ok(file) => {
            tracing::info!(path = %path.display(), "recording pi run");
            Some(file)
        }
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "pi run record was not opened");
            None
        }
    }
}

/// Counters and stderr tail shared between the reader tasks and the driver,
/// so partial results survive a reader that has to be aborted (straggler
/// holding the pipe after Pi exited).
#[derive(Default)]
struct Capture {
    malformed: AtomicU32,
    oversized: AtomicU32,
    stderr_tail: std::sync::Mutex<Vec<u8>>,
    /// Raw stdout of this run, when recording is enabled. `None` after a write
    /// failure so a full disk cannot stall Pi.
    record: std::sync::Mutex<Option<std::fs::File>>,
}

impl Capture {
    fn new() -> Self {
        Self {
            record: std::sync::Mutex::new(open_pi_run_record()),
            ..Self::default()
        }
    }

    fn record_stdout_line(&self, line: &[u8]) {
        let mut slot = self
            .record
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(file) = slot.as_mut() else {
            return;
        };
        if std::io::Write::write_all(file, line).is_err() {
            tracing::warn!("pi run record write failed; recording stopped");
            *slot = None;
        }
    }

    fn stderr_tail(&self) -> String {
        let tail = self
            .stderr_tail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        String::from_utf8_lossy(&tail).into_owned()
    }

    fn push_stderr(&self, chunk: &[u8]) {
        let mut tail = self
            .stderr_tail
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        tail.extend_from_slice(chunk);
        if tail.len() > STDERR_TAIL_BYTES {
            let excess = tail.len() - STDERR_TAIL_BYTES;
            tail.drain(..excess);
        }
    }
}

/// Result of one capped line read.
enum LineRead {
    /// `buf` holds a complete line (trailing newline included when present).
    Line,
    /// The line exceeded the cap; it was skipped up to and including its newline.
    Oversized,
    /// No more data.
    Eof,
}

/// Read one line into `buf` without ever buffering more than `cap` bytes.
///
/// Unlike `read_until`, a line longer than `cap` is discarded rather than
/// grown: the remainder up to the next `\n` is consumed and `Oversized`
/// returned. A final line without a trailing newline is still a `Line`.
async fn read_line_capped<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    cap: usize,
) -> std::io::Result<LineRead> {
    buf.clear();
    let mut oversized = false;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(if oversized {
                LineRead::Oversized
            } else if buf.is_empty() {
                LineRead::Eof
            } else {
                LineRead::Line
            });
        }
        let newline = available.iter().position(|b| *b == b'\n');
        let take = newline.map_or(available.len(), |i| i + 1);
        if !oversized {
            if buf.len() + take > cap {
                oversized = true;
                buf.clear();
            } else {
                buf.extend_from_slice(&available[..take]);
            }
        }
        reader.consume(take);
        if newline.is_some() {
            return Ok(if oversized {
                LineRead::Oversized
            } else {
                LineRead::Line
            });
        }
    }
}

/// Drain Pi's stdout to EOF, parsing each line and forwarding events.
///
/// If the consumer drops or closes the receiver we keep draining (discarding
/// events) so Pi never blocks on a full pipe.
async fn drain_stdout(
    stdout: Option<ChildStdout>,
    events_tx: mpsc::Sender<PiEvent>,
    capture: Arc<Capture>,
) {
    let Some(stdout) = stdout else {
        tracing::warn!("pi stdout was not captured");
        return;
    };
    let mut reader = BufReader::with_capacity(64 * 1024, stdout);
    let mut buf = Vec::with_capacity(8 * 1024);
    let mut receiver_open = true;
    loop {
        match read_line_capped(&mut reader, &mut buf, MAX_LINE_BYTES).await {
            Ok(LineRead::Eof) => break,
            Ok(LineRead::Oversized) => {
                capture.oversized.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(cap = MAX_LINE_BYTES, "pi stdout line exceeded cap; skipped");
            }
            Ok(LineRead::Line) => {
                capture.record_stdout_line(&buf);
                let line = String::from_utf8_lossy(&buf);
                match parse_line(&line) {
                    Ok(PiEvent::Ignored) => {}
                    Ok(event) => {
                        if receiver_open && events_tx.send(event).await.is_err() {
                            receiver_open = false;
                            tracing::debug!("pi event receiver closed; draining stdout");
                        }
                    }
                    Err(e) => {
                        capture.malformed.fetch_add(1, Ordering::Relaxed);
                        let preview: String = line.trim_end().chars().take(200).collect();
                        tracing::debug!(error = %e, line = %preview, "malformed pi stdout line");
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "reading pi stdout failed");
                break;
            }
        }
    }
}

/// Drain stderr to EOF into the shared tail (last [`STDERR_TAIL_BYTES`]).
async fn drain_stderr(stderr: Option<ChildStderr>, capture: Arc<Capture>) {
    let Some(mut stderr) = stderr else {
        return;
    };
    let mut chunk = [0u8; 4096];
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => capture.push_stderr(&chunk[..n]),
            Err(e) => {
                tracing::warn!(error = %e, "reading pi stderr failed");
                break;
            }
        }
    }
}

/// A task written to a private temp file for `@<path>` hand-off; removed when
/// the run ends (explicitly via [`TempTask::cleanup`], or in `Drop` as a
/// fallback).
struct TempTask {
    path: PathBuf,
    removed: bool,
}

impl TempTask {
    /// `Ok(None)` when the task can go inline (see [`task_needs_file`]).
    async fn for_task(task: &str) -> Result<Option<Self>, PiProcessError> {
        if !task_needs_file(task) {
            return Ok(None);
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path =
            std::env::temp_dir().join(format!("graphirm-pi-{}-{nanos}.md", std::process::id()));
        let mut options = tokio::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(&path).await.map_err(|e| {
            PiProcessError::Spawn(format!("creating task file {}: {e}", path.display()))
        })?;
        let guard = Self {
            path,
            removed: false,
        };
        file.write_all(task.as_bytes()).await.map_err(|e| {
            PiProcessError::Spawn(format!("writing task file {}: {e}", guard.path.display()))
        })?;
        file.flush().await.map_err(|e| {
            PiProcessError::Spawn(format!("flushing task file {}: {e}", guard.path.display()))
        })?;
        Ok(Some(guard))
    }

    async fn cleanup(mut self) {
        if let Err(e) = tokio::fs::remove_file(&self.path).await
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(path = %self.path.display(), error = %e, "failed to remove pi task file");
        }
        self.removed = true;
    }
}

impl Drop for TempTask {
    fn drop(&mut self) {
        if !self.removed {
            // Fallback only (e.g. spawn failed after the write); the happy
            // path removes asynchronously in `cleanup`.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn pi_runs_dir_follows_the_env_override() {
        let home = Path::new("/home/user");
        assert_eq!(
            resolve_pi_runs_dir(None, Some(home), true),
            Some(home.join(".graphirm/pi-runs"))
        );
        assert_eq!(resolve_pi_runs_dir(None, Some(home), false), None);
        assert_eq!(resolve_pi_runs_dir(Some("off"), Some(home), true), None);
        assert_eq!(resolve_pi_runs_dir(Some(""), Some(home), true), None);
        assert_eq!(
            resolve_pi_runs_dir(Some("/tmp/pi-runs"), Some(home), false),
            Some(PathBuf::from("/tmp/pi-runs"))
        );
    }

    #[test]
    fn record_stdout_line_appends_the_raw_line() {
        let dir = std::env::temp_dir().join(format!("graphirm-pi-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("run.jsonl");
        let file = std::fs::File::create(&path).unwrap();
        let capture = Capture {
            record: std::sync::Mutex::new(Some(file)),
            ..Capture::default()
        };
        capture.record_stdout_line(b"{\"type\":\"session\"}\n");
        let got = std::fs::read_to_string(&path).unwrap();
        assert_eq!(got, "{\"type\":\"session\"}\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    const FAKE_PI: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pi/fake_pi.sh");

    /// Config pointing at the fake script. Knobs travel as `--fake-knob K=V`
    /// pairs in `extra_args` (see the script header); a 1 ms per-line delay
    /// keeps the replay fast while still exercising streaming.
    fn fake_pi_config() -> PiConfig {
        with_knobs(&[])
    }

    fn with_knobs(knobs: &[(&str, &str)]) -> PiConfig {
        let mut extra_args = vec!["--fake-knob".to_string(), "FAKE_PI_DELAY_MS=1".to_string()];
        for (k, v) in knobs {
            extra_args.push("--fake-knob".to_string());
            extra_args.push(format!("{k}={v}"));
        }
        PiConfig {
            enabled: true,
            binary: FAKE_PI.to_string(),
            provider: "openrouter".to_string(),
            model: "test/model".to_string(),
            timeout_seconds: 60,
            trust_project: false,
            extra_args,
            max_result_chars: 4000,
        }
    }

    async fn run_to_end(
        config: &PiConfig,
        cwd: &Path,
        task: &str,
        timeout: Duration,
        cancel: CancellationToken,
    ) -> (Vec<PiEvent>, Result<PiRunOutcome, PiProcessError>) {
        let spec = PiSpawnSpec {
            config,
            cwd,
            task,
            timeout,
        };
        let mut handle = spawn_pi(spec, cancel).await.expect("spawn");
        let mut events = Vec::new();
        while let Some(ev) = handle.events.recv().await {
            events.push(ev);
        }
        let outcome = handle.wait().await;
        (events, outcome)
    }

    async fn wait_for_file(path: &Path) {
        for _ in 0..100 {
            if path.exists() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("{} never appeared", path.display());
    }

    async fn read_pid(path: &Path) -> u32 {
        wait_for_file(path).await;
        std::fs::read_to_string(path)
            .expect("pidfile")
            .trim()
            .parse()
            .expect("pid")
    }

    fn argv_lines(path: &Path) -> Vec<String> {
        std::fs::read_to_string(path)
            .expect("argv file")
            .lines()
            .map(str::to_string)
            .collect()
    }

    // ---- /proc-based liveness helpers (Linux only) ----

    #[cfg(target_os = "linux")]
    fn proc_state(pid: u32) -> Option<char> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        // "pid (comm) S ..." — comm may contain spaces/parens, so split after the last ')'.
        let rest = stat.rsplit_once(')')?.1;
        rest.trim_start().chars().next()
    }

    /// True when the pid is running (not gone, not a zombie).
    #[cfg(target_os = "linux")]
    fn is_running(pid: u32) -> bool {
        !matches!(proc_state(pid), None | Some('Z'))
    }

    /// Pids of live (non-zombie) processes whose process group is `pgid`.
    #[cfg(target_os = "linux")]
    fn group_members(pgid: u32) -> Vec<u32> {
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return Vec::new();
        };
        entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().to_str()?.parse::<u32>().ok())
            .filter(|pid| {
                let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
                    return false;
                };
                let Some((_, rest)) = stat.rsplit_once(')') else {
                    return false;
                };
                // rest = " S ppid pgrp session ..." → fields[0]=state, [2]=pgrp
                let fields: Vec<&str> = rest.split_whitespace().collect();
                fields.first() != Some(&"Z") && fields.get(2) == Some(&pgid.to_string().as_str())
            })
            .collect()
    }

    /// Poll up to 5 s for the pid to be gone (or a zombie).
    #[cfg(target_os = "linux")]
    async fn assert_dead_within_5s(pid: u32, what: &str) {
        for _ in 0..50 {
            if !is_running(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("{what} {pid} still running after 5 s");
    }

    /// Poll up to 5 s for the process group to have no live members.
    #[cfg(target_os = "linux")]
    async fn assert_group_empty_within_5s(pgid: u32) {
        let mut members = Vec::new();
        for _ in 0..50 {
            members = group_members(pgid);
            if members.is_empty() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("process group {pgid} still has members after 5 s: {members:?}");
    }

    #[test]
    fn build_argv_matches_design_d1() {
        let mut cfg = PiConfig {
            enabled: true,
            binary: "pi".into(),
            provider: "openrouter".into(),
            model: "deepseek/deepseek-v4-flash".into(),
            timeout_seconds: 900,
            trust_project: false,
            extra_args: vec!["--no-extensions".into(), "--thinking".into(), "off".into()],
            max_result_chars: 4000,
        };
        assert_eq!(
            build_argv(&cfg, &["do the thing"]),
            vec![
                "--mode",
                "json",
                "-p",
                "--no-session",
                "--no-approve",
                "--provider",
                "openrouter",
                "--model",
                "deepseek/deepseek-v4-flash",
                "--no-extensions",
                "--thinking",
                "off",
                "--",
                "do the thing",
            ]
        );
        cfg.trust_project = true;
        cfg.extra_args.clear();
        assert_eq!(
            build_argv(&cfg, &["@/tmp/task.md", TASK_FILE_INSTRUCTION]),
            vec![
                "--mode",
                "json",
                "-p",
                "--no-session",
                "--approve",
                "--provider",
                "openrouter",
                "--model",
                "deepseek/deepseek-v4-flash",
                "--",
                "@/tmp/task.md",
                TASK_FILE_INSTRUCTION,
            ]
        );
    }

    #[test]
    fn task_needs_file_rules() {
        assert!(!task_needs_file("hello"));
        assert!(!task_needs_file(&"a".repeat(MAX_INLINE_TASK_BYTES)));
        assert!(task_needs_file(&"a".repeat(MAX_INLINE_TASK_BYTES + 1)));
        assert!(task_needs_file("@notes.md please"));
        assert!(!task_needs_file("see @notes.md"));
    }

    #[test]
    fn expand_binary_expands_tilde() {
        let home = std::env::var("HOME").expect("HOME set in tests");
        assert_eq!(expand_binary("~/x/pi"), PathBuf::from(&home).join("x/pi"),);
        assert_eq!(expand_binary("~"), PathBuf::from(&home));
        assert_eq!(expand_binary("pi"), PathBuf::from("pi"));
        assert_eq!(expand_binary("/abs/pi"), PathBuf::from("/abs/pi"));
        // `~user` forms are not expanded.
        assert_eq!(expand_binary("~bob/pi"), PathBuf::from("~bob/pi"));
    }

    #[test]
    fn pi_command_env_policy() {
        let cmd = pi_command("pi");
        let envs: Vec<(String, Option<String>)> = cmd
            .as_std()
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        assert!(
            envs.contains(&("PI_SKIP_VERSION_CHECK".to_string(), Some("1".to_string()))),
            "{envs:?}"
        );
        // `None` is std's marker for `env_remove`.
        assert!(
            envs.contains(&("GRAPHIRM_API_KEY".to_string(), None)),
            "{envs:?}"
        );
        // Only those two entries: everything else is inherited untouched.
        assert_eq!(envs.len(), 2, "{envs:?}");
    }

    #[tokio::test]
    async fn probe_version_reports_fake_version() {
        let v = probe_version(&fake_pi_config()).await.expect("version");
        assert_eq!(v, "0.85.1-fake");
    }

    #[tokio::test]
    async fn probe_version_missing_binary_is_not_found() {
        let mut cfg = fake_pi_config();
        cfg.binary = "/nonexistent/pi".into();
        match probe_version(&cfg).await {
            Err(PiProcessError::NotFound(b)) => assert_eq!(b, "/nonexistent/pi"),
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn temp_task_file_is_private_and_removed() {
        let t = TempTask::for_task("@x")
            .await
            .expect("ok")
            .expect("needs file");
        let path = t.path.clone();
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "@x");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "mode {mode:o}");
        }
        t.cleanup().await;
        assert!(!path.exists());

        // Inline tasks never touch the filesystem.
        assert!(TempTask::for_task("plain").await.expect("ok").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn runs_fixture_to_completion_and_yields_events() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let cfg = fake_pi_config();
        let (events, outcome) = run_to_end(
            &cfg,
            dir.path(),
            "make hello.txt",
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        let outcome = outcome.expect("ok outcome");
        assert_eq!(outcome.exit_code, Some(0));
        assert_eq!(outcome.malformed_lines, 0);
        assert_eq!(outcome.oversized_lines, 0);
        assert!(!outcome.pipes_lingered);
        assert!(outcome.duration > Duration::ZERO);

        assert!(
            events.contains(&PiEvent::AgentEnd { will_retry: false }),
            "agent_end seen"
        );
        let starts = events
            .iter()
            .filter(|e| matches!(e, PiEvent::ToolStart { .. }))
            .count();
        let ends = events
            .iter()
            .filter(|e| matches!(e, PiEvent::ToolEnd { .. }))
            .count();
        assert!(starts >= 4, "tool starts: {starts}");
        assert!(ends >= 4, "tool ends: {ends}");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PiEvent::AssistantMessage { .. })),
            "assistant message seen"
        );
        assert!(
            !events.iter().any(|e| matches!(e, PiEvent::Ignored)),
            "Ignored is never forwarded"
        );
        assert!(matches!(events.first(), Some(PiEvent::Session { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn argv_passed_to_binary_matches_build_argv() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let argv_file = dir.path().join("argv.txt");
        let cfg = with_knobs(&[("FAKE_PI_ARGV", &argv_file.display().to_string())]);
        let task = "write a haiku about pipes";
        let (_, outcome) = run_to_end(
            &cfg,
            dir.path(),
            task,
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(outcome.expect("ok").exit_code, Some(0));
        assert_eq!(argv_lines(&argv_file), build_argv(&cfg, &[task]));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn child_env_has_skip_version_check_and_no_graphirm_key() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let env_file = dir.path().join("env.txt");
        let cfg = with_knobs(&[("FAKE_PI_ENV_DUMP", &env_file.display().to_string())]);
        let (_, outcome) = run_to_end(
            &cfg,
            dir.path(),
            "t",
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(outcome.expect("ok").exit_code, Some(0));
        // `export -p` format: `declare -x KEY="value"`.
        let env = std::fs::read_to_string(&env_file).expect("env dump");
        assert!(
            env.contains("PI_SKIP_VERSION_CHECK=\"1\""),
            "PI_SKIP_VERSION_CHECK missing:\n{env}"
        );
        // Whether or not the test process has GRAPHIRM_API_KEY set, the child
        // must not see it (env_remove is asserted structurally in
        // `pi_command_env_policy`).
        assert!(
            !env.contains("GRAPHIRM_API_KEY"),
            "GRAPHIRM_API_KEY leaked into pi's env:\n{env}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nonzero_exit_is_reported() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let cfg = with_knobs(&[("FAKE_PI_EXIT", "3")]);
        let (_, outcome) = run_to_end(
            &cfg,
            dir.path(),
            "t",
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(outcome.expect("ok").exit_code, Some(3));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn garbage_lines_are_counted_not_fatal() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let cfg = with_knobs(&[("FAKE_PI_GARBAGE", "1")]);
        let (events, outcome) = run_to_end(
            &cfg,
            dir.path(),
            "t",
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        let outcome = outcome.expect("ok");
        assert_eq!(outcome.exit_code, Some(0));
        assert!(outcome.malformed_lines > 0, "{outcome:?}");
        assert!(events.contains(&PiEvent::AgentEnd { will_retry: false }));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PiEvent::AssistantMessage { .. }))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn oversized_line_is_counted_and_skipped() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let fixture = dir.path().join("big.jsonl");
        let mut big = String::with_capacity(5 * 1024 * 1024 + 64);
        big.push_str(r#"{"type":"error","message":""#);
        big.extend(std::iter::repeat_n('x', 5 * 1024 * 1024));
        big.push_str("\"}\n");
        big.push_str("{\"type\":\"agent_end\",\"willRetry\":false}\n");
        std::fs::write(&fixture, big).expect("write fixture");

        let cfg = with_knobs(&[
            ("FAKE_PI_FIXTURE", &fixture.display().to_string()),
            ("FAKE_PI_DELAY_MS", "0"),
        ]);
        let (events, outcome) = run_to_end(
            &cfg,
            dir.path(),
            "t",
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await;
        let outcome = outcome.expect("ok");
        assert_eq!(outcome.oversized_lines, 1, "{outcome:?}");
        assert_eq!(outcome.malformed_lines, 0);
        assert_eq!(events, vec![PiEvent::AgentEnd { will_retry: false }]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_without_draining_events_completes_quickly() {
        // More events than the channel holds (but small enough to fit the
        // pipe buffer, so the fake exits normally): without `events.close()`
        // in `wait()` the stdout reader would sit on a full channel `send`
        // through phase 2, the run would end only after POST_EXIT_GRACE with
        // `pipes_lingered = true`, and the tail events would be lost. With
        // `close()` the reader drains stdout and the run ends promptly.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let fixture = dir.path().join("many.jsonl");
        let mut body = String::new();
        for _ in 0..(EVENT_CHANNEL_CAP * 4) {
            body.push_str("{\"type\":\"agent_start\"}\n");
        }
        body.push_str("{\"type\":\"agent_end\",\"willRetry\":false}\n");
        std::fs::write(&fixture, body).expect("write fixture");
        let cfg = with_knobs(&[
            ("FAKE_PI_FIXTURE", &fixture.display().to_string()),
            ("FAKE_PI_DELAY_MS", "0"),
        ]);
        let spec = PiSpawnSpec {
            config: &cfg,
            cwd: dir.path(),
            task: "t",
            timeout: Duration::from_secs(60),
        };
        let started = Instant::now();
        let mut handle = spawn_pi(spec, CancellationToken::new())
            .await
            .expect("spawn");
        let outcome = handle.wait().await.expect("ok");
        assert_eq!(outcome.exit_code, Some(0));
        assert!(!outcome.pipes_lingered);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "wait() stalled for {:?}",
            started.elapsed()
        );
        // The channel is closed, not poisoned: draining what (if anything) was
        // buffered before `close()` terminates with `None`.
        let mut leftover = 0;
        while handle.events.recv().await.is_some() {
            leftover += 1;
        }
        assert!(leftover <= EVENT_CHANNEL_CAP, "{leftover}");
    }

    #[tokio::test]
    async fn read_line_capped_handles_edge_cases() {
        let data = b"short\n".to_vec();
        let mut long = vec![b'a'; 20];
        long.push(b'\n');
        let mut input = data.clone();
        input.extend_from_slice(&long);
        input.extend_from_slice(b"tail-no-newline");
        let mut reader = BufReader::with_capacity(4, std::io::Cursor::new(input));
        let mut buf = Vec::new();

        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 10).await.unwrap(),
            LineRead::Line
        ));
        assert_eq!(buf, b"short\n");
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 10).await.unwrap(),
            LineRead::Oversized
        ));
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 100).await.unwrap(),
            LineRead::Line
        ));
        assert_eq!(buf, b"tail-no-newline");
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 100).await.unwrap(),
            LineRead::Eof
        ));

        // Oversized final line without newline is still counted as oversized.
        let mut reader = BufReader::with_capacity(4, std::io::Cursor::new(vec![b'z'; 30]));
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 10).await.unwrap(),
            LineRead::Oversized
        ));
        assert!(matches!(
            read_line_capped(&mut reader, &mut buf, 10).await.unwrap(),
            LineRead::Eof
        ));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_kills_child_within_5s() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let pidfile = dir.path().join("pid");
        let cfg = with_knobs(&[
            ("FAKE_PI_HANG_AT", "3"),
            ("FAKE_PI_PIDFILE", &pidfile.display().to_string()),
            ("FAKE_PI_CHILD", "1"),
        ]);
        let cancel = CancellationToken::new();
        let spec = PiSpawnSpec {
            config: &cfg,
            cwd: dir.path(),
            task: "t",
            timeout: Duration::from_secs(60),
        };
        let mut handle = spawn_pi(spec, cancel.clone()).await.expect("spawn");
        let script_pid = read_pid(&pidfile).await;
        let child_pid = read_pid(&dir.path().join("pid.child")).await;
        assert_eq!(handle.pid(), Some(script_pid));
        assert!(is_running(script_pid));
        assert!(is_running(child_pid));

        // Let it reach the hang so the cancel hits a blocked process.
        tokio::time::sleep(Duration::from_millis(300)).await;
        cancel.cancel();
        match handle.wait().await {
            Err(PiProcessError::Cancelled) => {}
            other => panic!("expected Cancelled, got {other:?}"),
        }
        assert_dead_within_5s(script_pid, "fake pi").await;
        assert_dead_within_5s(child_pid, "grandchild sleep").await;
        // Events channel closes too.
        while handle.events.recv().await.is_some() {}
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timeout_kills_child() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let pidfile = dir.path().join("pid");
        let cfg = with_knobs(&[
            ("FAKE_PI_HANG_AT", "3"),
            ("FAKE_PI_PIDFILE", &pidfile.display().to_string()),
        ]);
        let timeout = Duration::from_secs(1);
        let started = Instant::now();
        let (events, outcome) =
            run_to_end(&cfg, dir.path(), "t", timeout, CancellationToken::new()).await;
        match outcome {
            Err(PiProcessError::Timeout(t)) => assert_eq!(t, timeout),
            other => panic!("expected Timeout, got {other:?}"),
        }
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "kill path took {:?}",
            started.elapsed()
        );
        // The three lines before the hang were still delivered.
        assert!(events.len() >= 2, "{events:?}");
        let script_pid = read_pid(&pidfile).await;
        assert_dead_within_5s(script_pid, "fake pi").await;
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn post_exit_straggler_sets_pipes_lingered() {
        // The fake exits normally (code 2) but forks `sleep 3600` that keeps
        // stdout/stderr open. Phase 2 must give up after POST_EXIT_GRACE,
        // kill the group, and still report exit code and captured output.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let pidfile = dir.path().join("pid");
        let cfg = with_knobs(&[
            ("FAKE_PI_PIDFILE", &pidfile.display().to_string()),
            ("FAKE_PI_CHILD", "1"),
            ("FAKE_PI_CHILD_HOLDS_STDOUT", "1"),
            ("FAKE_PI_STDERR", "1"),
            ("FAKE_PI_EXIT", "2"),
        ]);
        let started = Instant::now();
        let (events, outcome) = run_to_end(
            &cfg,
            dir.path(),
            "t",
            Duration::from_secs(60),
            CancellationToken::new(),
        )
        .await;
        let outcome = outcome.expect("ok despite straggler");
        assert!(outcome.pipes_lingered, "{outcome:?}");
        assert_eq!(outcome.exit_code, Some(2));
        assert!(outcome.stderr_tail.contains("warn"), "{outcome:?}");
        assert!(events.contains(&PiEvent::AgentEnd { will_retry: false }));
        let elapsed = started.elapsed();
        assert!(
            elapsed >= POST_EXIT_GRACE && elapsed < POST_EXIT_GRACE + Duration::from_secs(6),
            "took {elapsed:?}"
        );
        let child_pid = read_pid(&dir.path().join("pid.child")).await;
        assert_dead_within_5s(child_pid, "straggler sleep").await;
    }

    #[tokio::test]
    async fn missing_binary_is_not_found() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let mut cfg = fake_pi_config();
        cfg.binary = "/nonexistent/pi".into();
        let spec = PiSpawnSpec {
            config: &cfg,
            cwd: dir.path(),
            task: "t",
            timeout: Duration::from_secs(5),
        };
        match spawn_pi(spec, CancellationToken::new()).await {
            Err(PiProcessError::NotFound(b)) => assert_eq!(b, "/nonexistent/pi"),
            Ok(_) => panic!("spawn unexpectedly succeeded"),
            Err(other) => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_cwd_is_spawn_error() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let missing = dir.path().join("nope");
        let cfg = fake_pi_config();
        let spec = PiSpawnSpec {
            config: &cfg,
            cwd: &missing,
            task: "t",
            timeout: Duration::from_secs(5),
        };
        match spawn_pi(spec, CancellationToken::new()).await {
            Err(PiProcessError::Spawn(msg)) => {
                assert!(msg.contains("workspace directory"), "{msg}");
                assert!(msg.contains("nope"), "{msg}");
            }
            Ok(_) => panic!("spawn unexpectedly succeeded"),
            Err(other) => panic!("expected Spawn, got {other:?}"),
        }
        // A file is not a directory either.
        let file = dir.path().join("file");
        std::fs::write(&file, "x").unwrap();
        let spec = PiSpawnSpec {
            config: &cfg,
            cwd: &file,
            task: "t",
            timeout: Duration::from_secs(5),
        };
        assert!(matches!(
            spawn_pi(spec, CancellationToken::new()).await,
            Err(PiProcessError::Spawn(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stderr_tail_is_captured() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let cfg = with_knobs(&[("FAKE_PI_STDERR", "1")]);
        let (_, outcome) = run_to_end(
            &cfg,
            dir.path(),
            "t",
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        let outcome = outcome.expect("ok");
        assert!(outcome.stderr_tail.contains("warn"), "{outcome:?}");
    }

    /// Runs a task that must go through the file route and checks argv shape,
    /// the file's presence during the run, and its removal afterwards.
    async fn assert_file_route(task: &str) {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let argv_file = dir.path().join("argv.txt");
        let cfg = with_knobs(&[("FAKE_PI_ARGV", &argv_file.display().to_string())]);
        let (_, outcome) = run_to_end(
            &cfg,
            dir.path(),
            task,
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        let outcome = outcome.expect("ok");
        assert_eq!(outcome.exit_code, Some(0));
        // The fake reports the referenced file's size while it exists.
        assert!(
            outcome
                .stderr_tail
                .contains(&format!("task-file-bytes: {}", task.len())),
            "{}",
            outcome.stderr_tail
        );
        let argv = argv_lines(&argv_file);
        let n = argv.len();
        assert_eq!(argv[n - 1], TASK_FILE_INSTRUCTION);
        let path = argv[n - 2].strip_prefix('@').expect("task passed as @file");
        assert!(path.contains("graphirm-pi-"), "{path}");
        assert_eq!(argv[n - 3], "--");
        assert!(
            !Path::new(path).exists(),
            "temp task file {path} should be removed after the run"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn large_task_goes_through_temp_file_and_is_removed() {
        assert_file_route(&"y".repeat(70 * 1024)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn at_prefixed_task_goes_through_temp_file() {
        // Pi would parse a bare leading `@` as a file path → "File not found".
        assert_file_route("@todo.md is not a file, just the first word").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn small_task_is_passed_inline() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let argv_file = dir.path().join("argv.txt");
        let cfg = with_knobs(&[("FAKE_PI_ARGV", &argv_file.display().to_string())]);
        let task = "z".repeat(MAX_INLINE_TASK_BYTES);
        let (_, outcome) = run_to_end(
            &cfg,
            dir.path(),
            &task,
            Duration::from_secs(30),
            CancellationToken::new(),
        )
        .await;
        assert_eq!(outcome.expect("ok").exit_code, Some(0));
        assert_eq!(argv_lines(&argv_file).last(), Some(&task));
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_handle_kills_child() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let pidfile = dir.path().join("pid");
        let cfg = with_knobs(&[
            ("FAKE_PI_HANG_AT", "3"),
            ("FAKE_PI_PIDFILE", &pidfile.display().to_string()),
            ("FAKE_PI_CHILD", "1"),
        ]);
        let spec = PiSpawnSpec {
            config: &cfg,
            cwd: dir.path(),
            task: "t",
            timeout: Duration::from_secs(60),
        };
        let handle = spawn_pi(spec, CancellationToken::new())
            .await
            .expect("spawn");
        let script_pid = read_pid(&pidfile).await;
        let child_pid = read_pid(&dir.path().join("pid.child")).await;
        assert!(is_running(script_pid));
        assert!(is_running(child_pid));
        drop(handle);
        assert_dead_within_5s(script_pid, "fake pi").await;
        // Not just the direct child: the group guard takes the grandchild too.
        assert_dead_within_5s(child_pid, "grandchild sleep").await;
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_handle_immediately_after_spawn_kills_group() {
        // No await between spawn and drop: the driver task may never have
        // been polled, so the guard must already be armed at spawn time.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let cfg = with_knobs(&[("FAKE_PI_HANG_AT", "3"), ("FAKE_PI_CHILD", "1")]);
        let spec = PiSpawnSpec {
            config: &cfg,
            cwd: dir.path(),
            task: "t",
            timeout: Duration::from_secs(60),
        };
        let handle = spawn_pi(spec, CancellationToken::new())
            .await
            .expect("spawn");
        let pgid = handle.pid().expect("pid captured at spawn");
        drop(handle);
        assert_dead_within_5s(pgid, "fake pi").await;
        assert_group_empty_within_5s(pgid).await;
    }
}
