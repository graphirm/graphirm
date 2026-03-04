// Multi-agent: coordinator pattern, subagent spawning, result aggregation

use std::collections::HashMap;
use std::path::Path;

use crate::config::{AgentConfig, AgentMode};
use crate::error::AgentError;

/// Registry of agent configurations loaded from TOML files.
pub struct AgentRegistry {
    configs: HashMap<String, AgentConfig>,
}

impl AgentRegistry {
    /// Load all `*.toml` agent config files from a directory.
    pub fn load_from_dir(path: &Path) -> Result<Self, AgentError> {
        let mut configs = HashMap::new();

        if !path.exists() {
            return Ok(Self { configs });
        }

        let entries = std::fs::read_dir(path).map_err(|e| {
            AgentError::Workflow(format!("Failed to read dir {}: {}", path.display(), e))
        })?;

        for entry in entries {
            let entry = entry
                .map_err(|e| AgentError::Workflow(format!("Failed to read entry: {}", e)))?;
            let file_path = entry.path();

            if file_path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }

            let config = AgentConfig::from_file(&file_path)?;
            configs.insert(config.name.clone(), config);
        }

        Ok(Self { configs })
    }

    /// Create a registry from a pre-built map (useful for testing).
    pub fn from_configs(configs: HashMap<String, AgentConfig>) -> Self {
        Self { configs }
    }

    /// Get an agent config by name.
    pub fn get(&self, name: &str) -> Option<&AgentConfig> {
        self.configs.get(name)
    }

    /// List all registered agent names.
    pub fn list(&self) -> Vec<&str> {
        self.configs.keys().map(|s| s.as_str()).collect()
    }

    /// Find the primary agent config (mode = "primary").
    pub fn primary(&self) -> Option<&AgentConfig> {
        self.configs.values().find(|c| c.mode == AgentMode::Primary)
    }

    /// Find all subagent configs.
    pub fn subagents(&self) -> Vec<&AgentConfig> {
        self.configs
            .values()
            .filter(|c| c.mode == AgentMode::Subagent)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_toml(dir: &TempDir, name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.path().join(format!("{}.toml", name));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn build_toml() -> &'static str {
        r#"
[agent]
name = "build"
mode = "primary"
model = "anthropic/claude-sonnet-4"
description = "Default agent with full tool access"
system_prompt = "You are a coding assistant."
max_turns = 50
tools = ["bash", "read", "write", "edit"]

[permissions]
bash = "allow"
write = "allow"
edit = "allow"
"#
    }

    fn explore_toml() -> &'static str {
        r#"
[agent]
name = "explore"
mode = "subagent"
model = "anthropic/claude-haiku-4"
description = "Fast, read-only codebase exploration"
system_prompt = "You explore code. Read files and report findings."
max_turns = 10
tools = ["read", "grep", "find", "ls"]

[permissions]
bash = "deny"
write = "deny"
edit = "deny"
"#
    }

    #[test]
    fn test_registry_load_from_dir() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert_eq!(registry.list().len(), 2);
        assert!(registry.get("build").is_some());
        assert!(registry.get("explore").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_primary() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let primary = registry.primary().unwrap();
        assert_eq!(primary.name, "build");
        assert_eq!(primary.mode, AgentMode::Primary);
    }

    #[test]
    fn test_registry_list_names() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        write_toml(&dir, "explore", explore_toml());

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["build", "explore"]);
    }

    #[test]
    fn test_registry_empty_dir() {
        let dir = TempDir::new().unwrap();
        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert!(registry.list().is_empty());
        assert!(registry.primary().is_none());
    }

    #[test]
    fn test_registry_skips_non_toml_files() {
        let dir = TempDir::new().unwrap();
        write_toml(&dir, "build", build_toml());
        std::fs::write(dir.path().join("README.md"), "# Agents").unwrap();

        let registry = AgentRegistry::load_from_dir(dir.path()).unwrap();
        assert_eq!(registry.list().len(), 1);
    }
}
