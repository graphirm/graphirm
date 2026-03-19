use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::process::Command;

use crate::{Tool, ToolContext, ToolError, ToolOutput};

/// A tool plugin defined by a TOML manifest and executed as a shell command.
///
/// Manifest format (`~/.graphirm/plugins/<name>/plugin.toml`):
/// ```toml
/// name = "deploy"
/// description = "Deploy the current project to staging"
/// destructive = true
/// command = "bash ${plugin_dir}/deploy.sh"
///
/// [parameters]
/// type = "object"
///
/// [parameters.properties.target]
/// type = "string"
/// description = "Deployment target (staging, production)"
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub destructive: bool,
    pub command: String,
    #[serde(default = "default_parameters")]
    pub parameters: serde_json::Value,
}

fn default_parameters() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

pub struct ScriptTool {
    manifest: PluginManifest,
    plugin_dir: PathBuf,
}

impl ScriptTool {
    pub fn new(manifest: PluginManifest, plugin_dir: PathBuf) -> Self {
        Self {
            manifest,
            plugin_dir,
        }
    }

    pub fn is_destructive(&self) -> bool {
        self.manifest.destructive
    }

    /// Load a plugin from a directory containing `plugin.toml`.
    pub fn from_dir(dir: &std::path::Path) -> Result<Self, ToolError> {
        let manifest_path = dir.join("plugin.toml");
        let content = std::fs::read_to_string(&manifest_path).map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to read plugin manifest at {}: {e}",
                manifest_path.display()
            ))
        })?;
        let manifest: PluginManifest = toml::from_str(&content).map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Invalid plugin manifest at {}: {e}",
                manifest_path.display()
            ))
        })?;
        Ok(Self::new(manifest, dir.to_path_buf()))
    }
}

#[async_trait]
impl Tool for ScriptTool {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn description(&self) -> &str {
        &self.manifest.description
    }

    fn parameters(&self) -> serde_json::Value {
        self.manifest.parameters.clone()
    }

    fn is_destructive(&self) -> bool {
        self.manifest.destructive
    }

    async fn execute(
        &self,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let command = self
            .manifest
            .command
            .replace("${plugin_dir}", &self.plugin_dir.to_string_lossy());

        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(&command);
        cmd.current_dir(&ctx.working_dir);

        // Pass all tool arguments as GRAPHIRM_ARGS env var (JSON string)
        cmd.env(
            "GRAPHIRM_ARGS",
            serde_json::to_string(&args).unwrap_or_default(),
        );

        // Pass individual top-level string args as GRAPHIRM_ARG_<KEY> env vars
        if let Some(obj) = args.as_object() {
            for (key, value) in obj {
                let env_key = format!("GRAPHIRM_ARG_{}", key.to_uppercase());
                match value {
                    serde_json::Value::String(s) => {
                        cmd.env(&env_key, s);
                    }
                    other => {
                        cmd.env(&env_key, other.to_string());
                    }
                }
            }
        }

        let output = cmd.output().await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Plugin '{}' failed to start: {e}",
                self.manifest.name
            ))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            let content = if stderr.is_empty() {
                stdout.to_string()
            } else {
                format!("{stdout}\n\n[stderr]\n{stderr}")
            };
            Ok(ToolOutput::success(content))
        } else {
            let content = format!(
                "Plugin '{}' exited with code {}\n\n[stdout]\n{}\n\n[stderr]\n{}",
                self.manifest.name,
                output.status.code().unwrap_or(-1),
                stdout,
                stderr,
            );
            Ok(ToolOutput::error(content))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    use graphirm_graph::GraphStore;
    use graphirm_graph::nodes::NodeId;
    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;

    fn test_context(working_dir: &std::path::Path) -> ToolContext {
        ToolContext {
            graph: Arc::new(GraphStore::open_memory().unwrap()),
            agent_id: NodeId::from("test-agent"),
            interaction_id: NodeId::from("test-interaction"),
            working_dir: working_dir.to_path_buf(),
            signal: CancellationToken::new(),
            turn: 0,
            turn_pos_counter: Arc::new(AtomicU32::new(0)),
            knowledge_retriever: None,
        }
    }

    #[test]
    fn test_parse_manifest() {
        let toml_str = r#"
            name = "hello"
            description = "Says hello"
            command = "echo hello"
            destructive = false

            [parameters]
            type = "object"

            [parameters.properties.name]
            type = "string"
            description = "Name to greet"
        "#;
        let manifest: PluginManifest = toml::from_str(toml_str).unwrap();
        assert_eq!(manifest.name, "hello");
        assert!(!manifest.destructive);
    }

    #[test]
    fn test_from_dir() {
        let dir = tempdir().unwrap();
        let manifest = r#"
            name = "test_plugin"
            description = "A test plugin"
            command = "echo test"
        "#;
        std::fs::write(dir.path().join("plugin.toml"), manifest).unwrap();
        let tool = ScriptTool::from_dir(dir.path()).unwrap();
        assert_eq!(tool.name(), "test_plugin");
        assert!(!tool.is_destructive());
    }

    #[test]
    fn test_from_dir_missing_manifest() {
        let dir = tempdir().unwrap();
        let result = ScriptTool::from_dir(dir.path());
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_success() {
        let dir = tempdir().unwrap();
        let manifest = PluginManifest {
            name: "echo_test".to_string(),
            description: "Echoes args".to_string(),
            destructive: false,
            command: "echo $GRAPHIRM_ARG_MESSAGE".to_string(),
            parameters: default_parameters(),
        };
        let tool = ScriptTool::new(manifest, dir.path().to_path_buf());
        let ctx = test_context(dir.path());
        let result = tool
            .execute(serde_json::json!({"message": "hello world"}), &ctx)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("hello world"));
    }

    #[tokio::test]
    async fn test_execute_failure() {
        let dir = tempdir().unwrap();
        let manifest = PluginManifest {
            name: "fail_test".to_string(),
            description: "Always fails".to_string(),
            destructive: false,
            command: "exit 1".to_string(),
            parameters: default_parameters(),
        };
        let tool = ScriptTool::new(manifest, dir.path().to_path_buf());
        let ctx = test_context(dir.path());
        let result = tool
            .execute(serde_json::json!({}), &ctx)
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[test]
    fn test_destructive_flag() {
        let manifest = PluginManifest {
            name: "dangerous".to_string(),
            description: "Does dangerous things".to_string(),
            destructive: true,
            command: "rm -rf /tmp/test".to_string(),
            parameters: default_parameters(),
        };
        let tool = ScriptTool::new(manifest, PathBuf::from("/tmp"));
        assert!(tool.is_destructive());
    }

    #[tokio::test]
    async fn test_plugin_dir_substitution() {
        let dir = tempdir().unwrap();
        let manifest = PluginManifest {
            name: "dir_test".to_string(),
            description: "Tests plugin_dir substitution".to_string(),
            destructive: false,
            command: "echo ${plugin_dir}".to_string(),
            parameters: default_parameters(),
        };
        let plugin_dir = dir.path().to_path_buf();
        let tool = ScriptTool::new(manifest, plugin_dir.clone());
        let ctx = test_context(dir.path());
        let result = tool
            .execute(serde_json::json!({}), &ctx)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains(plugin_dir.to_string_lossy().as_ref()));
    }
}
