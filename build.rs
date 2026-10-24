use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn main() {
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = match git(&["status", "--porcelain"]) {
        None => "unknown",
        Some(text) if text.is_empty() => "false",
        Some(_) => "true",
    };
    println!("cargo:rustc-env=GRAPHIRM_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=GRAPHIRM_GIT_DIRTY={dirty}");
    println!("cargo:rerun-if-changed=.git/HEAD");
}
