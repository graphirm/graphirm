use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    pub fn with_properties(
        name: impl Into<String>,
        description: impl Into<String>,
        properties: Vec<(&str, &str, &str)>,
        required: Vec<&str>,
    ) -> Self {
        let mut props = serde_json::Map::new();
        for (prop_name, prop_type, prop_desc) in properties {
            props.insert(
                prop_name.to_string(),
                serde_json::json!({
                    "type": prop_type,
                    "description": prop_desc,
                }),
            );
        }
        Self {
            name: name.into(),
            description: description.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": props,
                "required": required,
            }),
        }
    }
}

impl From<ToolDefinition> for rig::completion::ToolDefinition {
    fn from(td: ToolDefinition) -> Self {
        rig::completion::ToolDefinition {
            name: td.name,
            description: td.description,
            parameters: td.parameters,
        }
    }
}

impl From<rig::completion::ToolDefinition> for ToolDefinition {
    fn from(td: rig::completion::ToolDefinition) -> Self {
        Self {
            name: td.name,
            description: td.description,
            parameters: td.parameters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_definition_new() {
        let params = serde_json::json!({"type": "object"});
        let td = ToolDefinition::new("my_tool", "Does something", params.clone());
        assert_eq!(td.name, "my_tool");
        assert_eq!(td.description, "Does something");
        assert_eq!(td.parameters, params);
    }

    #[test]
    fn test_tool_definition_with_properties() {
        let td = ToolDefinition::with_properties(
            "read_file",
            "Reads a file",
            vec![("path", "string", "The file path")],
            vec!["path"],
        );
        assert_eq!(td.name, "read_file");
        assert!(td.parameters["properties"]["path"]["type"] == "string");
        assert!(
            td.parameters["required"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("path"))
        );
    }

    #[test]
    fn test_tool_definition_serde_roundtrip() {
        let td = ToolDefinition::new("tool", "description", serde_json::json!({}));
        let json = serde_json::to_string(&td).unwrap();
        let restored: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(td, restored);
    }

    #[test]
    fn test_tool_definition_into_rig() {
        let td = ToolDefinition::new("bash", "Run bash", serde_json::json!({"type": "object"}));
        let rig_td: rig::completion::ToolDefinition = td.clone().into();
        assert_eq!(rig_td.name, td.name);
        assert_eq!(rig_td.description, td.description);
        assert_eq!(rig_td.parameters, td.parameters);
    }

    #[test]
    fn test_tool_definition_from_rig() {
        let rig_td = rig::completion::ToolDefinition {
            name: "bash".to_string(),
            description: "Run bash".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let td: ToolDefinition = rig_td.into();
        assert_eq!(td.name, "bash");
    }

    #[test]
    fn test_tool_definition_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ToolDefinition>();
    }

    #[test]
    fn test_tool_definition_clone() {
        let td = ToolDefinition::new("tool", "desc", serde_json::json!({}));
        let cloned = td.clone();
        assert_eq!(td, cloned);
    }
}
