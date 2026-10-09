//! Per-task workspaces for the eval harness.
//!
//! Each task gets a fresh directory under the server's `workspaces_root`.
//! The harness copies the repo slice those tasks read (`crates/` and `src/`)
//! into that directory, and deletes the shared `/tmp` files a task may write.

use std::path::Path;

/// Top-level trees copied into a session workspace.
pub const REPO_SLICE: &[&str] = &["crates", "src"];

/// Absolute paths tasks write. Cleared before every task so a leftover cannot score.
pub const SHARED_EVAL_FILES: &[&str] = &[
    "/tmp/eval_fib.rs",
    "/tmp/eval_functions.py",
    "/tmp/eval_secret.txt",
    "/tmp/eval_reader.sh",
    "/tmp/eval_broken.py",
    "/tmp/eval_marker.txt",
];

pub fn eval_workspace_name(task_id: &str, nonce: u64) -> String {
    let safe: String = task_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect();
    format!("eval-{safe}-{nonce}")
}

pub fn clear_shared_eval_files() {
    for path in SHARED_EVAL_FILES {
        let _ = std::fs::remove_file(path);
    }
}

/// Copy `crates/` and `src/` from `repo` into `dest`, skipping any `target` directory.
pub fn copy_repo_slice(repo: &Path, dest: &Path) -> std::io::Result<()> {
    for name in REPO_SLICE {
        let from = repo.join(name);
        if from.is_dir() {
            copy_dir_skipping_targets(&from, &dest.join(name))?;
        }
    }
    Ok(())
}

/// Token that appears only in `selection/early.txt`. Prompts must not include it.
pub const SELECTION_TOKEN: &str = "TOKEN_EARLY_184729";

/// Words of filler in each selection file. At the word estimator this is about
/// 2,900 tokens, so one file is larger than the tail share of a 4,000 budget.
const SELECTION_WORDS: usize = 2_200;

/// Write four large files under `dest/selection/`. The early file holds the token.
/// Later files are filler, so a tight budget has to drop the early read.
pub fn write_selection_fixtures(dest: &Path) -> std::io::Result<()> {
    let dir = dest.join("selection");
    std::fs::create_dir_all(&dir)?;
    for (name, seed, head) in [
        ("early.txt", 1u32, SELECTION_TOKEN),
        ("mid_a.txt", 2, "FILLER_A"),
        ("mid_b.txt", 3, "FILLER_B"),
        ("late.txt", 4, "LABEL=unset"),
    ] {
        let mut body = String::new();
        body.push_str(head);
        body.push('\n');
        for i in 0..SELECTION_WORDS {
            if i > 0 {
                body.push(' ');
            }
            body.push_str(&format!("w{seed}_{i}"));
        }
        body.push('\n');
        std::fs::write(dir.join(name), body)?;
    }
    Ok(())
}

/// Make `dest` a git repo with one commit of the copied slice, so `git diff`
/// shows only what the agent changes.
pub fn init_workspace_repo(dest: &Path) -> std::io::Result<()> {
    fn git(dest: &Path, args: &[&str]) -> std::io::Result<()> {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(dest)
            .args(args)
            .env("GIT_AUTHOR_NAME", "graphirm-eval")
            .env("GIT_AUTHOR_EMAIL", "eval@graphirm.local")
            .env("GIT_COMMITTER_NAME", "graphirm-eval")
            .env("GIT_COMMITTER_EMAIL", "eval@graphirm.local")
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!(
                "git {} exited {}",
                args.join(" "),
                status
            )))
        }
    }
    git(dest, &["init", "-q"])?;
    git(dest, &["add", "-A"])?;
    git(
        dest,
        &["commit", "-m", "eval workspace slice", "--allow-empty"],
    )?;
    Ok(())
}

fn copy_dir_skipping_targets(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        if entry.file_name() == "target" {
            continue;
        }
        let dest = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_skipping_targets(&entry.path(), &dest)?;
        } else {
            std::fs::copy(entry.path(), &dest)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_name_is_unique_per_nonce() {
        assert_eq!(
            eval_workspace_name("read-line-count", 7),
            "eval-read-line-count-7"
        );
        assert_ne!(
            eval_workspace_name("read-line-count", 7),
            eval_workspace_name("read-line-count", 8)
        );
    }

    #[test]
    fn copy_repo_slice_keeps_sources_and_skips_target() {
        let root = std::env::temp_dir().join(format!("graphirm-eval-slice-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("crates/agent/target")).unwrap();
        std::fs::write(root.join("crates/agent/lib.rs"), "fn line() {}\n").unwrap();
        std::fs::write(root.join("crates/agent/target/skip.rs"), "nope").unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();

        let dest = root.join("workspace");
        copy_repo_slice(&root, &dest).unwrap();

        assert_eq!(
            std::fs::read_to_string(dest.join("crates/agent/lib.rs")).unwrap(),
            "fn line() {}\n"
        );
        assert!(dest.join("src/main.rs").is_file());
        assert!(!dest.join("crates/agent/target/skip.rs").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn init_workspace_repo_commits_the_slice() {
        let root = std::env::temp_dir().join(format!("graphirm-eval-git-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        let dest = root.join("workspace");
        copy_repo_slice(&root, &dest).unwrap();
        init_workspace_repo(&dest).unwrap();
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(&dest)
            .args(["status", "--porcelain"])
            .output()
            .unwrap();
        assert!(status.status.success());
        assert!(
            status.stdout.is_empty(),
            "{}",
            String::from_utf8_lossy(&status.stdout)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn selection_fixtures_hide_the_token_in_the_first_file() {
        let root =
            std::env::temp_dir().join(format!("graphirm-eval-selection-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        write_selection_fixtures(&root).unwrap();
        let early = std::fs::read_to_string(root.join("selection/early.txt")).unwrap();
        let late = std::fs::read_to_string(root.join("selection/late.txt")).unwrap();
        assert!(early.starts_with(SELECTION_TOKEN));
        assert!(!late.contains(SELECTION_TOKEN));
        assert!(late.starts_with("LABEL=unset"));
        assert!(early.split_whitespace().count() > SELECTION_WORDS);
        let _ = std::fs::remove_dir_all(&root);
    }
}
