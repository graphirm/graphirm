use std::collections::HashMap;
use std::sync::Arc;

use crate::{Tool, ToolCall, ToolContext, ToolDefinition, ToolError, ToolOutput};

pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Result<Arc<dyn Tool>, ToolError> {
        self.tools
            .get(name)
            .cloned()
            .ok_or_else(|| ToolError::NotFound(name.to_string()))
    }

    pub fn list(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs: Vec<ToolDefinition> = self.tools.values().map(|t| t.definition()).collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub async fn execute(
        &self,
        call: &ToolCall,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let tool = self.get(&call.name)?;
        tool.execute(call.arguments.clone(), ctx).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use async_trait::async_trait;
    use serde_json::json;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn execute(
            &self,
            args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success(args.to_string()))
        }
    }

    struct PingTool;

    #[async_trait]
    impl Tool for PingTool {
        fn name(&self) -> &str {
            "ping"
        }
        fn description(&self) -> &str {
            "Pings"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("pong"))
        }
    }

    struct AlphaTool;

    #[async_trait]
    impl Tool for AlphaTool {
        fn name(&self) -> &str {
            "alpha"
        }
        fn description(&self) -> &str {
            "Alpha tool"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::success("alpha"))
        }
    }

    #[test]
    fn new_registry_is_empty() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn register_and_get() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        assert_eq!(reg.len(), 1);
        let tool = reg.get("echo").unwrap();
        assert_eq!(tool.name(), "echo");
    }

    #[test]
    fn get_not_found() {
        let reg = ToolRegistry::new();
        let result = reg.get("nonexistent");
        assert!(matches!(result, Err(ToolError::NotFound(_))));
    }

    #[test]
    fn list_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(PingTool));
        reg.register(Arc::new(EchoTool));
        reg.register(Arc::new(AlphaTool));
        let names = reg.list();
        assert_eq!(names, vec!["alpha", "echo", "ping"]);
    }

    #[test]
    fn definitions_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(PingTool));
        reg.register(Arc::new(EchoTool));
        let defs = reg.definitions();
        assert_eq!(defs[0].name, "echo");
        assert_eq!(defs[1].name, "ping");
    }

    #[test]
    fn register_overwrites() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        reg.register(Arc::new(EchoTool));
        assert_eq!(reg.len(), 1);
    }

    #[tokio::test]
    async fn execute_by_name() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(PingTool));
        let ctx = make_test_context();
        let call = ToolCall {
            id: "c1".into(),
            name: "ping".into(),
            arguments: json!({}),
        };
        let out = reg.execute(&call, &ctx).await.unwrap();
        assert_eq!(out.content, "pong");
    }

    #[tokio::test]
    async fn execute_not_found() {
        let reg = ToolRegistry::new();
        let ctx = make_test_context();
        let call = ToolCall {
            id: "c2".into(),
            name: "missing".into(),
            arguments: json!({}),
        };
        let err = reg.execute(&call, &ctx).await.unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }
}
