use std::collections::HashMap;
use graphirm_llm::ContentPart;

/// A key representing a unique tool call for deduplication and pattern tracking.
///
/// Combines the tool name and optional file path to identify repeated tool invocations.
#[derive(Debug, Clone)]
pub struct ToolCallKey {
    /// The name of the tool being called (e.g., "read", "write").
    pub tool_name: String,
    /// Optional file path for tools that operate on files, used for deduplication.
    pub file_path: Option<String>,
}

impl ToolCallKey {
    /// Extracts a `ToolCallKey` from a `ContentPart`, if it represents a tool call.
    ///
    /// Returns `Some` if the part is a tool call, extracting the tool name and
    /// optional file path from the arguments ("path" or "file_path" keys).
    /// Returns `None` if the part is not a tool call.
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
    /// Soft escalation triggers after this turn number if threshold is exceeded
    pub soft_escalation_turn: usize,
    /// Minimum number of repeated tool calls to trigger escalation
    pub soft_escalation_threshold: usize,
}

impl EscalationDetector {
    /// Size of the sliding window for tracking tool calls.
    /// Window includes current turn + 3 previous turns = 4 turns total.
    const WINDOW_SIZE: usize = 4;

    /// Creates a new `EscalationDetector` with specified escalation parameters.
    ///
    /// # Arguments
    ///
    /// * `soft_turn` - The turn number after which escalation detection begins
    /// * `threshold` - The maximum number of repeated tool calls before escalation triggers
    pub fn new(soft_turn: usize, threshold: usize) -> Self {
        Self {
            recent_calls: HashMap::new(),
            soft_escalation_turn: soft_turn,
            soft_escalation_threshold: threshold,
        }
    }

    /// Records a tool call in the current turn for pattern tracking.
    ///
    /// # Arguments
    ///
    /// * `turn` - The turn number in which the tool was called
    /// * `key` - The tool call key containing tool name and optional file path
    pub fn record_tool_call(&mut self, turn: usize, key: ToolCallKey) {
        let key_str = format!("{}:{:?}", key.tool_name, key.file_path);
        let entry_key = (turn, key_str);
        *self.recent_calls.entry(entry_key).or_insert(0) += 1;
    }

    /// Checks if soft escalation should be triggered based on recent tool call patterns.
    ///
    /// Analyzes tool calls in the sliding window (current turn and 3 previous turns)
    /// to detect if any tool exceeds the configured threshold, indicating repetitive
    /// behavior that may warrant human escalation.
    ///
    /// # Arguments
    ///
    /// * `turn` - The current turn number
    /// * `_tool_calls` - Reserved for future use (forward compatibility)
    ///
    /// # Returns
    ///
    /// A tuple of `(should_escalate, max_count)` where:
    /// - `should_escalate` is true if any tool's count meets or exceeds the threshold
    /// - `max_count` is the highest repetition count observed in the window
    pub fn should_escalate(&self, turn: usize, _tool_calls: &[&ContentPart]) -> (bool, usize) {
        if turn < self.soft_escalation_turn {
            return (false, 0);
        }

        // Count all tool call occurrences in the sliding window
        let mut tool_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        
        for check_turn in (turn.saturating_sub(Self::WINDOW_SIZE - 1))..=turn {
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

    /// Cleans up old entries to prevent unbounded memory growth.
    ///
    /// Removes entries from turns older than the retention window to maintain
    /// constant memory usage as the detector processes many turns.
    ///
    /// # Arguments
    ///
    /// * `current_turn` - The current turn number
    /// * `window_size` - Number of turns to retain
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
        let mut detector = EscalationDetector::new(4, 2);
        
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
