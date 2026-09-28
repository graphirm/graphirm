//! Child-process teardown helpers shared by tools that spawn subprocesses
//! (`bash` today, `delegate_pi` next).
//!
//! The pattern: spawn the child as leader of its own process group
//! (`Command::process_group(0)` on unix), capture `child.id()` immediately after
//! `spawn()`, and on cancel/timeout call [`kill_group_and_reap`] with that pgid.
//! Killing the *group* rather than the direct child is what stops
//! `bash -c 'sleep 30'` from orphaning `sleep`, or `bash -c 'npm run dev &'`
//! from leaving a grandchild holding the stdout pipe after the shell exited.

use std::time::Duration;

use tokio::process::Child;

/// How long to wait for the killed child to be reaped before giving up.
const REAP_TIMEOUT: Duration = Duration::from_secs(5);

/// SIGKILL an entire process group.
///
/// `pgid` must be greater than 1: `kill(0, …)` would signal *our own* process
/// group and `kill(-1, …)` every process we are allowed to signal. Returns the OS
/// error when the signal could not be delivered; `ESRCH` (no such group) is a
/// normal outcome when the group already exited and callers should log it at
/// debug level.
#[cfg(unix)]
pub fn kill_process_group(pgid: u32) -> std::io::Result<()> {
    debug_assert!(pgid > 1, "refusing to signal pgid {pgid}");
    if pgid <= 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing to signal process group {pgid}"),
        ));
    }
    let pgid = i32::try_from(pgid).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("pgid {pgid} does not fit in pid_t"),
        )
    })?;
    // SAFETY: `kill(2)` takes plain integers and has no memory-safety
    // preconditions; a negative pid targets the process group `-pid`.
    let rc = unsafe { libc::kill(-pgid, libc::SIGKILL) };
    if rc == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// No-op on platforms without process groups; callers fall back to
/// `Child::start_kill`.
#[cfg(not(unix))]
pub fn kill_process_group(_pgid: u32) -> std::io::Result<()> {
    Ok(())
}

/// Tear down a child spawned with `process_group(0)`.
///
/// 1. SIGKILL the whole process group (when `pgid` is `Some`), so descendants
///    the child forked die with it.
/// 2. `start_kill()` the direct child as a portable fallback (harmless if the
///    group kill already landed, or if the child was already reaped).
/// 3. Reap with `child.wait()` under [`REAP_TIMEOUT`] so no zombie lingers.
///
/// `pgid` should be captured via `child.id()` right after `spawn()`: once the
/// direct child has been reaped `child.id()` is `None`, and that is exactly the
/// case (shell exited, grandchild still running) where the group kill matters.
pub async fn kill_group_and_reap(child: &mut Child, pgid: Option<u32>) {
    if let Some(pgid) = pgid {
        match kill_process_group(pgid) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(no_such_process()) => {
                tracing::debug!(pgid, "process group already gone");
            }
            Err(e) => {
                tracing::warn!(pgid, error = %e, "failed to kill process group");
            }
        }
    }
    if let Err(e) = child.start_kill() {
        // `InvalidInput` is tokio's "already exited / already reaped" — routine.
        if e.kind() != std::io::ErrorKind::InvalidInput {
            tracing::warn!(error = %e, "failed to kill child process");
        }
    }
    match tokio::time::timeout(REAP_TIMEOUT, child.wait()).await {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => tracing::warn!(error = %e, "failed to reap killed child"),
        Err(_) => tracing::warn!(
            timeout_secs = REAP_TIMEOUT.as_secs(),
            "killed child was not reaped in time"
        ),
    }
}

#[cfg(unix)]
fn no_such_process() -> i32 {
    libc::ESRCH
}

#[cfg(not(unix))]
fn no_such_process() -> i32 {
    // No errno to compare against; pick a value that never matches.
    i32::MIN
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use tokio::process::Command;

    fn proc_state(pid: u32) -> Option<char> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        // "pid (comm) S ..." — comm may contain spaces/parens, so split after the last ')'.
        let rest = stat.rsplit_once(')')?.1;
        rest.trim_start().chars().next()
    }

    /// True when the pid is running (not gone, not a zombie).
    fn is_running(pid: u32) -> bool {
        !matches!(proc_state(pid), None | Some('Z'))
    }

    #[tokio::test]
    async fn kill_group_takes_down_descendants_and_reaps() {
        let dir = tempfile::TempDir::new().unwrap();
        let pidfile = dir.path().join("sleep_pid");
        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(format!("sleep 30 & echo $! > {}; wait", pidfile.display()))
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .process_group(0);
        let mut child = cmd.spawn().unwrap();
        let pgid = child.id();
        assert!(pgid.is_some());

        for _ in 0..50 {
            if pidfile.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let sleep_pid: u32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!(is_running(sleep_pid));

        kill_group_and_reap(&mut child, pgid).await;

        // Shell reaped by `wait()`; `sleep` reparented to init, may take a beat to vanish.
        assert!(child.try_wait().unwrap().is_some());
        let mut alive = true;
        for _ in 0..20 {
            alive = is_running(sleep_pid);
            if !alive {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(!alive, "sleep {sleep_pid} survived the group kill");
    }

    #[tokio::test]
    async fn kill_missing_group_is_esrch() {
        // Spawn and fully reap a short-lived group leader so its pgid is free.
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg("exit 0").process_group(0);
        let mut child = cmd.spawn().unwrap();
        let pgid = child.id().unwrap();
        child.wait().await.unwrap();
        let err = kill_process_group(pgid).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::ESRCH));
    }

    #[tokio::test]
    async fn kill_reaped_child_with_no_pgid_is_a_noop() {
        let mut child = Command::new("bash")
            .arg("-c")
            .arg("exit 0")
            .spawn()
            .unwrap();
        child.wait().await.unwrap();
        // Must not panic or hang.
        kill_group_and_reap(&mut child, None).await;
    }

    #[test]
    fn refuses_dangerous_pgids() {
        // Would be kill(0)/kill(-1): must be rejected without signalling anything.
        // Under debug the `debug_assert!` panics first (also acceptable), so guard
        // with catch_unwind; in release the runtime check must return `Err`.
        if let Ok(r) = std::panic::catch_unwind(|| kill_process_group(1)) {
            assert!(r.is_err());
        }
    }
}
