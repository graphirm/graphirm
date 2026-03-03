use std::sync::Arc;

use tokio::task::JoinSet;
use tracing::{debug, warn};

use crate::{Tool, ToolCall, ToolContext, ToolError, ToolOutput};
use crate::registry::ToolRegistry;

#[derive(Debug)]
pub struct ToolCallResult {
    pub call_id: String,
    pub result: Result<ToolOutput, ToolError>,
}

pub async fn execute_parallel(
    registry: &ToolRegistry,
    calls: Vec<ToolCall>,
    ctx: &ToolContext,
) -> Vec<ToolCallResult> {
    let mut set: JoinSet<(String, Result<ToolOutput, ToolError>)> = JoinSet::new();

    for call in calls {
        let tool: Arc<dyn Tool> = match registry.get(&call.name) {
            Ok(t) => t,
            Err(e) => {
                let call_id = call.id.clone();
                let err_msg = e.to_string();
                set.spawn(async move {
                    (call_id, Err::<ToolOutput, ToolError>(ToolError::NotFound(err_msg)))
                });
                continue;
            }
        };

        let args = call.arguments.clone();
        let call_id = call.id.clone();
        let ctx = ctx.clone();

        set.spawn(async move {
            debug!(tool = %call_id, "executing tool");
            let result = tool.execute(args, &ctx).await;
            (call_id, result)
        });
    }

    let mut results = Vec::new();
    while let Some(join_result) = set.join_next().await {
        match join_result {
            Ok((call_id, result)) => results.push(ToolCallResult { call_id, result }),
            Err(join_error) => {
                warn!(error = %join_error, "tool task panicked");
                results.push(ToolCallResult {
                    call_id: "unknown".into(),
                    result: Err(ToolError::ExecutionFailed(format!(
                        "task panicked: {}",
                        join_error
                    ))),
                });
            }
        }
    }
    results
}

pub async fn execute_single(
    registry: &ToolRegistry,
    call: &ToolCall,
    ctx: &ToolContext,
) -> Result<ToolOutput, ToolError> {
    let tool = registry.get(&call.name)?;
    tool.execute(call.arguments.clone(), ctx).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_test_context;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    struct CounterTool {
        counter: Arc<AtomicUsize>,
    }

    impl CounterTool {
        fn new(counter: Arc<AtomicUsize>) -> Self {
            Self { counter }
        }
    }

    #[async_trait]
    impl Tool for CounterTool {
        fn name(&self) -> &str {
            "counter"
        }
        fn description(&self) -> &str {
            "Increments a counter"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(ToolOutput::success(count.to_string()))
        }
    }

    struct SlowTool;

    #[async_trait]
    impl Tool for SlowTool {
        fn name(&self) -> &str {
            "slow"
        }
        fn description(&self) -> &str {
            "Sleeps for 50ms"
        }
        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }
        async fn execute(
            &self,
            _args: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolOutput, ToolError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(ToolOutput::success("done"))
        }
    }

    fn make_registry_with_counter(counter: Arc<AtomicUsize>) -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(CounterTool::new(counter)));
        reg
    }

    #[tokio::test]
    async fn parallel_three_tools() {
        let counter = Arc::new(AtomicUsize::new(0));
        let reg = make_registry_with_counter(counter.clone());
        let ctx = make_test_context();

        let calls = vec![
            ToolCall { id: "1".into(), name: "counter".into(), arguments: json!({}) },
            ToolCall { id: "2".into(), name: "counter".into(), arguments: json!({}) },
            ToolCall { id: "3".into(), name: "counter".into(), arguments: json!({}) },
        ];

        let results = execute_parallel(&reg, calls, &ctx).await;
        assert_eq!(results.len(), 3);
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        for r in &results {
            assert!(r.result.is_ok());
        }
    }

    #[tokio::test]
    async fn parallel_preserves_call_ids() {
        let counter = Arc::new(AtomicUsize::new(0));
        let reg = make_registry_with_counter(counter);
        let ctx = make_test_context();

        let calls = vec![
            ToolCall { id: "abc".into(), name: "counter".into(), arguments: json!({}) },
            ToolCall { id: "def".into(), name: "counter".into(), arguments: json!({}) },
        ];

        let results = execute_parallel(&reg, calls, &ctx).await;
        let ids: Vec<&str> = results.iter().map(|r| r.call_id.as_str()).collect();
        assert!(ids.contains(&"abc"));
        assert!(ids.contains(&"def"));
    }

    #[tokio::test]
    async fn parallel_with_not_found() {
        let counter = Arc::new(AtomicUsize::new(0));
        let reg = make_registry_with_counter(counter);
        let ctx = make_test_context();

        let calls = vec![
            ToolCall { id: "ok".into(), name: "counter".into(), arguments: json!({}) },
            ToolCall { id: "bad".into(), name: "missing_tool".into(), arguments: json!({}) },
        ];

        let results = execute_parallel(&reg, calls, &ctx).await;
        assert_eq!(results.len(), 2);

        let bad = results.iter().find(|r| r.call_id == "bad").unwrap();
        assert!(matches!(bad.result, Err(ToolError::NotFound(_))));

        let ok = results.iter().find(|r| r.call_id == "ok").unwrap();
        assert!(ok.result.is_ok());
    }

    #[tokio::test]
    async fn parallel_actually_concurrent() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(SlowTool));
        let ctx = make_test_context();

        let calls: Vec<ToolCall> = (0..5)
            .map(|i| ToolCall {
                id: i.to_string(),
                name: "slow".into(),
                arguments: json!({}),
            })
            .collect();

        let start = std::time::Instant::now();
        let results = execute_parallel(&reg, calls, &ctx).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 5);
        // 5 serial 50ms calls would be 250ms; concurrent should finish < 200ms
        assert!(
            elapsed < Duration::from_millis(200),
            "expected concurrent execution, elapsed: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn execute_single_success() {
        let counter = Arc::new(AtomicUsize::new(0));
        let reg = make_registry_with_counter(counter.clone());
        let ctx = make_test_context();

        let call = ToolCall { id: "x".into(), name: "counter".into(), arguments: json!({}) };
        let out = execute_single(&reg, &call, &ctx).await.unwrap();
        assert_eq!(out.content, "1");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
