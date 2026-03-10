//! Shared TUI state and widget types. Extracted to break app <-> events <-> ui dependency cycles.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use graphirm_graph::nodes::NodeId;
use graphirm_llm::Role;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppState {
    Idle,
    WaitingForAgent,
    Streaming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPanel {
    Chat,
    Graph,
    Input,
}

pub struct ChatView {
    pub messages: Vec<ChatMessage>,
    pub scroll_offset: u16,
    /// When true the view auto-scrolls to the bottom on new content.
    /// Set to false when the user manually scrolls up.
    pub pinned_to_bottom: bool,
    /// The scroll offset that was actually rendered last frame.
    /// Used to initialize scroll_offset correctly when unpinning from bottom.
    pub last_computed_scroll: u16,
}

impl ChatView {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            scroll_offset: 0,
            pinned_to_bottom: true,
            last_computed_scroll: 0,
        }
    }
}

impl Default for ChatView {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub node_id: Option<NodeId>,
    pub is_tool_call: bool,
    pub tool_name: Option<String>,
}

pub struct GraphExplorer {
    pub nodes: Vec<GraphNodeEntry>,
    pub selected: usize,
    pub expanded: HashSet<String>,
}

impl GraphExplorer {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            selected: 0,
            expanded: HashSet::new(),
        }
    }
}

impl Default for GraphExplorer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GraphNodeEntry {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub depth: u16,
    pub has_children: bool,
}

pub struct InputField {
    pub content: String,
    pub cursor: usize,
}

impl InputField {
    pub fn new() -> Self {
        Self {
            content: String::new(),
            cursor: 0,
        }
    }

    pub fn insert(&mut self, c: char) {
        self.content.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.content[..self.cursor]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor -= prev;
            self.content.remove(self.cursor);
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.content.len() {
            self.content.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            let prev = self.content[..self.cursor]
                .chars()
                .last()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor -= prev;
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor < self.content.len() {
            let next = self.content[self.cursor..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(0);
            self.cursor += next;
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.content.len();
    }

    /// Drain the input content, returning it and resetting the field.
    pub fn take(&mut self) -> String {
        let content = std::mem::take(&mut self.content);
        self.cursor = 0;
        content
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

impl Default for InputField {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StatusBar {
    pub tokens_in: u32,
    pub tokens_out: u32,
    pub cost: f64,
    pub model: String,
    pub agent_state: String,
}

impl StatusBar {
    pub fn new(model: String) -> Self {
        Self {
            tokens_in: 0,
            tokens_out: 0,
            cost: 0.0,
            model,
            agent_state: "Idle".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_field_insert_char() {
        let mut input = InputField::new();
        input.insert('h');
        input.insert('i');
        assert_eq!(input.content, "hi");
        assert_eq!(input.cursor, 2);
    }

    #[test]
    fn test_input_field_insert_at_cursor() {
        let mut input = InputField::new();
        input.insert('h');
        input.insert('o');
        input.move_left();
        input.insert('e');
        input.insert('l');
        input.insert('l');
        assert_eq!(input.content, "hello");
        assert_eq!(input.cursor, 4);
    }

    #[test]
    fn test_input_field_delete_backspace() {
        let mut input = InputField::new();
        input.insert('a');
        input.insert('b');
        input.insert('c');
        input.backspace();
        assert_eq!(input.content, "ab");
        assert_eq!(input.cursor, 2);
    }

    #[test]
    fn test_input_field_backspace_at_start() {
        let mut input = InputField::new();
        input.backspace();
        assert_eq!(input.content, "");
        assert_eq!(input.cursor, 0);
    }

    #[test]
    fn test_input_field_move_left_right() {
        let mut input = InputField::new();
        input.insert('a');
        input.insert('b');
        input.insert('c');
        assert_eq!(input.cursor, 3);

        input.move_left();
        assert_eq!(input.cursor, 2);

        input.move_left();
        assert_eq!(input.cursor, 1);

        input.move_right();
        assert_eq!(input.cursor, 2);

        input.move_right();
        input.move_right();
        assert_eq!(input.cursor, 3);

        input.move_left();
        input.move_left();
        input.move_left();
        input.move_left();
        assert_eq!(input.cursor, 0);
    }

    #[test]
    fn test_input_field_clear() {
        let mut input = InputField::new();
        input.insert('x');
        input.insert('y');
        let content = input.take();
        assert_eq!(content, "xy");
        assert_eq!(input.content, "");
        assert_eq!(input.cursor, 0);
    }

    #[test]
    fn test_input_field_home_end() {
        let mut input = InputField::new();
        input.insert('a');
        input.insert('b');
        input.insert('c');

        input.home();
        assert_eq!(input.cursor, 0);

        input.end();
        assert_eq!(input.cursor, 3);
    }
}
