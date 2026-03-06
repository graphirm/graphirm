use std::collections::HashMap;
use graphirm_llm::ContentPart;

#[derive(Debug, Clone)]
pub struct ToolCallKey {
    pub tool_name: String,
    pub file_path: Option<String>, // For deduplication key
}

impl ToolCallKey {
    pub fn from_content_part(part: &ContentPart) -> Option<Self> {
        match part {
            ContentPart::ToolCall { name, arguments, .. } => {
                let file_path = arguments
                    .get("path")
                    .or_else(|| arguments.get("file_path"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                
                Some(ToolCallKey {
                    tool_name: name.clone(),
                    file_path,
                })
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct EscalationDetector {
    /// Track tool calls in a sliding window of recent turns
    /// Key: (turn_number, tool_key) -> count
    recent_calls: HashMap<(usize, String), usize>,
    /// Configuration thresholds
    pub soft_escalation_turn: usize,
    pub soft_escalation_threshold: usize,
}

impl EscalationDetector {
    pub fn new(soft_turn: usize, threshold: usize) -> Self {
        Self {
            recent_calls: HashMap::new(),
            soft_escalation_turn: soft_turn,
            soft_escalation_threshold: threshold,
        }
    }

    /// Record a tool call in the current turn
    pub fn record_tool_call(&mut self, turn: usize, key: ToolCallKey) {
        let key_str = format!("{}:{:?}", key.tool_name, key.file_path);
        let entry_key = (turn, key_str);
        *self.recent_calls.entry(entry_key).or_insert(0) += 1;
    }

    /// Check if soft escalation should trigger
    /// Looks at recent tool call history and detects repeated patterns
    pub fn should_escalate(&self, turn: usize, _tool_calls: &[&ContentPart]) -> (bool, usize) {
        if turn <= self.soft_escalation_turn {
            return (false, 0);
        }

        // Count all tool call occurrences in the sliding window
        let mut tool_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        for check_turn in (turn.saturating_sub(3))..=turn {
            for ((t, key_str), count) in self.recent_calls.iter() {
                if *t == check_turn {
                    *tool_counts.entry(key_str.clone()).or_insert(0) += count;
                }
            }
        }

        // Find the max count for any tool
        let max_count = tool_counts.values().max().copied().unwrap_or(0);

        (
            max_count >= self.soft_escalation_threshold,
            max_count,
        )
    }

    /// Clean up old entries to prevent memory growth
    pub fn cleanup_old_turns(&mut self, current_turn: usize, window_size: usize) {
        self.recent_calls.retain(|(turn, _), _| *turn + window_size >= current_turn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escalation_detector_creation() {
        let detector = EscalationDetector::new(7, 2);
        assert_eq!(detector.soft_escalation_turn, 7);
        assert_eq!(detector.soft_escalation_threshold, 2);
    }

    #[test]
    fn test_escalation_threshold_triggers() {
        let mut detector = EscalationDetector::new(3, 2);
        
        let key1 = ToolCallKey {
            tool_name: "read".to_string(),
            file_path: Some("file.rs".to_string()),
        };
        
        detector.record_tool_call(1, key1.clone());
        detector.record_tool_call(2, key1.clone());
        detector.record_tool_call(3, key1.clone());
        
        // Should not escalate yet (turn < soft_escalation_turn)
        assert!(!detector.should_escalate(3, &[]).0);
        
        // At turn 4, we should detect the pattern
        detector.record_tool_call(4, key1);
        let (should_escalate, count) = detector.should_escalate(4, &[]);
        assert!(should_escalate);
        assert!(count >= 2);
    }
}
