//! Workspace name sanitization for session and subagent directories.

/// Sanitise a user-provided name into a safe directory component.
/// Trim, lowercase, replace non-`[a-z0-9_-]` with `-`, collapse consecutive
/// dashes, strip leading/trailing dashes. Returns `None` if the result is empty.
pub fn sanitize_workspace_name(name: &str) -> Option<String> {
    let lowered = name.trim().to_lowercase();
    let replaced: String = lowered
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let mut result = String::new();
    let mut last_dash = false;
    for c in replaced.chars() {
        if c == '-' {
            if !last_dash {
                result.push(c);
            }
            last_dash = true;
        } else {
            result.push(c);
            last_dash = false;
        }
    }
    let result = result.trim_matches('-').to_string();
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_basic() {
        assert_eq!(
            sanitize_workspace_name("Hello World"),
            Some("hello-world".to_string())
        );
    }

    #[test]
    fn sanitize_empty_returns_none() {
        assert_eq!(sanitize_workspace_name(""), None);
        assert_eq!(sanitize_workspace_name("   "), None);
    }

    #[test]
    fn sanitize_special_chars() {
        assert_eq!(
            sanitize_workspace_name("my/project@v2"),
            Some("my-project-v2".to_string())
        );
    }

    #[test]
    fn sanitize_preserves_alphanumeric_and_dash_underscore() {
        assert_eq!(
            sanitize_workspace_name("auth_service-v2"),
            Some("auth_service-v2".to_string())
        );
    }

    #[test]
    fn sanitize_collapses_consecutive_dashes() {
        assert_eq!(
            sanitize_workspace_name("foo--bar"),
            Some("foo-bar".to_string())
        );
        assert_eq!(sanitize_workspace_name("--foo--"), Some("foo".to_string()));
    }

    #[test]
    fn sanitize_only_dashes_returns_none() {
        assert_eq!(sanitize_workspace_name("---"), None);
    }
}
