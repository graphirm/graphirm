//! Repo briefing — compact summary injected at session start.

use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

// ── Language breakdown ────────────────────────────────────────────────────────

/// Walk `root` (async, up to `max_files` entries) and return a map of
/// file-extension → file-count. Hidden directories (starting with `.`) and
/// `target/` are skipped.
pub async fn count_files_by_extension(root: &Path, max_files: usize) -> HashMap<String, usize> {
    let mut map: HashMap<String, usize> = HashMap::new();
    let mut count = 0usize;
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let mut entries = match fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            if count >= max_files {
                return map;
            }
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            // Skip hidden dirs and target/
            if name_str.starts_with('.') || name_str == "target" || name_str == "node_modules" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    *map.entry(ext.to_string()).or_insert(0) += 1;
                }
                count += 1;
            }
        }
    }
    map
}

/// Format the top extensions as a compact string, e.g. `"rs: 312, ts: 89, toml: 24"`.
/// Keeps at most `top_n` entries, sorted by count descending.
pub fn format_language_breakdown(map: &HashMap<String, usize>, top_n: usize) -> String {
    let mut pairs: Vec<(&String, &usize)> = map.iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    pairs.truncate(top_n);
    pairs
        .iter()
        .map(|(ext, count)| format!("{}: {}", ext, count))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn format_empty_map() {
        let map = HashMap::new();
        assert_eq!(format_language_breakdown(&map, 5), "");
    }

    #[test]
    fn format_sorts_by_count_descending() {
        let mut map = HashMap::new();
        map.insert("toml".to_string(), 10);
        map.insert("rs".to_string(), 50);
        map.insert("md".to_string(), 5);
        let result = format_language_breakdown(&map, 5);
        // rs (50) should come first
        assert!(result.starts_with("rs: 50"));
    }

    #[test]
    fn format_truncates_to_top_n() {
        let mut map = HashMap::new();
        for i in 0..10u32 {
            map.insert(format!("ext{}", i), i as usize + 1);
        }
        let result = format_language_breakdown(&map, 3);
        // Only 3 entries — 2 commas
        assert_eq!(result.matches(", ").count(), 2);
    }
}
