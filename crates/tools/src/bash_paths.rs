use std::path::PathBuf;

/// Extract literal file paths from a bash command string using tree-sitter.
///
/// Walks the AST looking for `word` or `string` nodes that look like file paths.
/// Skips nodes inside command substitutions (`$(...)`) and variable expansions
/// (`${...}`) since those are dynamic and can't be statically resolved.
///
/// Returns an empty Vec (not an error) when:
/// - The command can't be parsed
/// - No path-like tokens are found
/// - The command only uses dynamic paths
pub fn extract_paths(command: &str) -> Vec<PathBuf> {
    let mut parser = tree_sitter::Parser::new();
    let language = tree_sitter_bash::LANGUAGE;
    parser
        .set_language(&language.into())
        .expect("tree-sitter-bash grammar");

    let Some(tree) = parser.parse(command, None) else {
        return Vec::new();
    };

    let mut paths = Vec::new();
    let source = command.as_bytes();
    collect_paths(tree.root_node(), source, &mut paths, 0);

    paths.sort();
    paths.dedup();
    paths
}

/// Known file extensions for code and config files.
const PATH_EXTENSIONS: &[&str] = &[
    "rs", "toml", "lock", "md", "txt", "json", "yaml", "yml", "ts", "tsx",
    "js", "jsx", "py", "sh", "bash", "sql", "html", "css", "xml", "csv",
    "env", "cfg", "ini", "conf", "log", "gitignore", "dockerfile",
];

fn is_path_like(text: &str) -> bool {
    if text.is_empty() || text.starts_with('-') {
        return false;
    }

    // Contains a path separator → likely a path
    if text.contains('/') && !text.starts_with("http://") && !text.starts_with("https://") {
        return true;
    }

    // Has a known file extension
    if let Some((_base, ext)) = text.rsplit_once('.') {
        if PATH_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
            return true;
        }
    }

    false
}

/// Returns true if this node is inside a command substitution or variable expansion,
/// making its value dynamic and unresolvable at parse time.
fn is_inside_expansion(node: tree_sitter::Node) -> bool {
    let mut cursor = node;
    while let Some(parent) = cursor.parent() {
        let kind = parent.kind();
        if kind == "command_substitution" || kind == "expansion" {
            return true;
        }
        cursor = parent;
    }
    false
}

fn collect_paths(node: tree_sitter::Node, source: &[u8], paths: &mut Vec<PathBuf>, depth: usize) {
    // Guard against pathologically deep trees
    if depth > 100 {
        return;
    }

    let kind = node.kind();

    // Skip entire subtrees that represent dynamic values
    if kind == "command_substitution" || kind == "expansion" {
        return;
    }

    // Check leaf-like nodes for path patterns
    if (kind == "word" || kind == "raw_string" || kind == "string_content")
        && !is_inside_expansion(node)
    {
        if let Ok(text) = node.utf8_text(source) {
            let cleaned = text.trim_matches(|c| c == '"' || c == '\'');
            if is_path_like(cleaned) {
                paths.push(PathBuf::from(cleaned));
            }
        }
    }

    // Also check redirect targets: `> file.txt`, `>> log.txt`
    if kind == "file_redirect" || kind == "heredoc_redirect" {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                if child.kind() == "word" || child.kind() == "string_content" {
                    if let Ok(text) = child.utf8_text(source) {
                        let cleaned = text.trim_matches(|c| c == '"' || c == '\'');
                        if is_path_like(cleaned) {
                            paths.push(PathBuf::from(cleaned));
                        }
                    }
                }
            }
        }
    }

    // Recurse into children
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            collect_paths(child, source, paths, depth + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_path_from_sed_command() {
        let paths = extract_paths("sed -i 's/foo/bar/' src/lib.rs");
        assert!(paths.contains(&PathBuf::from("src/lib.rs")));
    }

    #[test]
    fn extracts_path_from_redirect() {
        let paths = extract_paths("echo hello > output.txt");
        assert!(paths.contains(&PathBuf::from("output.txt")));
    }

    #[test]
    fn extracts_multiple_paths() {
        let paths = extract_paths("cp src/main.rs backup/main.rs.bak");
        assert!(paths.len() >= 2);
        assert!(paths.contains(&PathBuf::from("src/main.rs")));
    }

    #[test]
    fn skips_flags_and_options() {
        let paths = extract_paths("ls -la --color=auto");
        assert!(paths.is_empty());
    }

    #[test]
    fn skips_urls() {
        let paths = extract_paths("curl https://example.com/api/data");
        // Should not treat the URL as a file path
        for p in &paths {
            assert!(
                !p.to_str().unwrap().starts_with("https://"),
                "URL should not be extracted as path"
            );
        }
    }

    #[test]
    fn returns_empty_for_simple_echo() {
        let paths = extract_paths("echo hello world");
        assert!(paths.is_empty());
    }

    #[test]
    fn returns_empty_for_unparseable() {
        // Deeply malformed — tree-sitter may still parse it but won't find paths
        let paths = extract_paths("((((");
        // No assertion on count — just shouldn't panic
        let _ = paths;
    }

    #[test]
    fn handles_path_with_known_extension() {
        let paths = extract_paths("cat Cargo.toml");
        assert!(paths.contains(&PathBuf::from("Cargo.toml")));
    }

    #[test]
    fn deduplicates_paths() {
        let paths = extract_paths("cat src/lib.rs && grep foo src/lib.rs");
        let count = paths.iter().filter(|p| *p == &PathBuf::from("src/lib.rs")).count();
        assert_eq!(count, 1, "duplicate paths should be deduplicated");
    }
}
