use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use graphirm_llm::Role;

use crate::app::{ChatMessage, ChatView};

impl ChatView {
    pub fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
    }

    /// Append a text delta to the last message (for streaming).
    pub fn append_delta(&mut self, delta: &str) {
        if let Some(last) = self.messages.last_mut() {
            last.content.push_str(delta);
        }
    }

    /// Scroll toward older messages. Unpins from bottom.
    /// When transitioning out of pinned mode, starts from the last rendered
    /// bottom position so the view scrolls smoothly rather than jumping to top.
    pub fn scroll_up(&mut self) {
        if self.pinned_to_bottom {
            self.scroll_offset = self.last_computed_scroll.saturating_sub(3);
            self.pinned_to_bottom = false;
        } else {
            self.scroll_offset = self.scroll_offset.saturating_sub(3);
        }
    }

    /// Scroll toward newer messages. Re-pins when reaching the bottom.
    pub fn scroll_down(&mut self) {
        self.scroll_offset = self.scroll_offset.saturating_add(3);
    }

    /// Pin to the bottom so new messages are always visible.
    /// The render function computes the correct offset from content + area height.
    pub fn scroll_to_bottom(&mut self) {
        self.pinned_to_bottom = true;
    }

    pub fn render_widget(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::default().borders(Borders::ALL).title(" Chat ");

        let inner = block.inner(area);
        block.render(area, buf);

        if self.messages.is_empty() {
            let placeholder = Paragraph::new("No messages yet. Type below to start.")
                .style(Style::default().fg(Color::DarkGray));
            placeholder.render(inner, buf);
            return;
        }

        let mut lines: Vec<Line> = Vec::new();

        for msg in &self.messages {
            let (role_label, role_style) = match msg.role {
                Role::Human => (
                    "You",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Role::Assistant => (
                    "Assistant",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Role::ToolResult => (
                    msg.tool_name.as_deref().unwrap_or("Tool"),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Role::System => (
                    "System",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ),
            };

            lines.push(Line::from(vec![Span::styled(
                role_label.to_string(),
                role_style,
            )]));

            for line in msg.content.lines() {
                lines.push(Line::from(line.to_string()));
            }

            lines.push(Line::from(""));
        }

        let text = Text::from(lines);

        // When pinned, scroll to show the bottom of the content.
        // Ratatui does not clamp automatically, so we must compute the correct
        // offset ourselves: total content lines minus the visible area height.
        let content_lines = text.lines.len() as u16;
        let scroll_offset = if self.pinned_to_bottom {
            content_lines.saturating_sub(inner.height)
        } else {
            self.scroll_offset.min(content_lines.saturating_sub(inner.height))
        };

        // Store the computed offset so scroll_up() can start from here
        // when transitioning out of pinned mode.
        self.last_computed_scroll = scroll_offset;

        let paragraph = Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .scroll((scroll_offset, 0));

        paragraph.render(inner, buf);
    }
}

#[cfg(test)]
mod tests {
    use crate::app::{ChatMessage, ChatView};
    use chrono::Utc;
    use graphirm_llm::Role;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    fn buffer_to_lines(buf: &Buffer) -> Vec<String> {
        let area = buf.area;
        (area.y..area.y + area.height)
            .map(|y| {
                (area.x..area.x + area.width)
                    .map(|x| {
                        buf.cell((x, y))
                            .map(|c| c.symbol().to_string())
                            .unwrap_or_else(|| " ".to_string())
                    })
                    .collect::<Vec<_>>()
                    .join("")
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    fn make_message(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.to_string(),
            timestamp: Utc::now(),
            node_id: None,
            is_tool_call: false,
            tool_name: None,
        }
    }

    #[test]
    fn test_chat_view_renders_messages() {
        let mut chat = ChatView::new();
        chat.add_message(make_message(Role::Human, "Hello!"));
        chat.add_message(make_message(Role::Assistant, "Hi there!"));

        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        chat.render_widget(area, &mut buf); // &mut self

        let lines = buffer_to_lines(&buf);
        let text = lines.join("\n");

        assert!(text.contains("You"), "Should show 'You' role label");
        assert!(text.contains("Hello!"), "Should show user message");
        assert!(text.contains("Assistant"), "Should show 'Assistant' label");
        assert!(text.contains("Hi there!"), "Should show assistant message");
    }

    #[test]
    fn test_chat_view_scroll() {
        let mut chat = ChatView::new();
        // Start unpinned to test raw offset arithmetic (otherwise scroll_up
        // would initialize from last_computed_scroll which is 0 in unit tests
        // where no render has occurred yet).
        chat.pinned_to_bottom = false;

        // scroll_down increases offset (toward newer content)
        chat.scroll_down();
        assert_eq!(chat.scroll_offset, 3);
        chat.scroll_down();
        assert_eq!(chat.scroll_offset, 6);
        // scroll_up decreases offset (toward older content)
        chat.scroll_up();
        assert_eq!(chat.scroll_offset, 3);
        chat.scroll_up();
        chat.scroll_up(); // clamps to 0
        assert_eq!(chat.scroll_offset, 0);
    }

    #[test]
    fn test_scroll_up_from_pinned_uses_last_computed() {
        let mut chat = ChatView::new();
        // Simulate having rendered at offset 20 (e.g. many messages)
        chat.last_computed_scroll = 20;
        assert!(chat.pinned_to_bottom);

        // scroll_up from pinned should start from last_computed_scroll, not 0
        chat.scroll_up();
        assert!(!chat.pinned_to_bottom);
        assert_eq!(chat.scroll_offset, 17); // 20 - 3
    }

    #[test]
    fn test_chat_view_scroll_to_bottom() {
        let mut chat = ChatView::new();
        chat.add_message(make_message(Role::Human, "Hello"));
        chat.add_message(make_message(Role::Assistant, "Hi\nSecond line"));
        chat.scroll_to_bottom();
        // pinned_to_bottom flag should be set
        assert!(
            chat.pinned_to_bottom,
            "scroll_to_bottom should pin to bottom"
        );
    }

    #[test]
    fn test_scroll_up_unpins() {
        let mut chat = ChatView::new();
        assert!(chat.pinned_to_bottom);
        chat.scroll_up();
        assert!(!chat.pinned_to_bottom, "scroll_up should unpin from bottom");
    }

    #[test]
    fn test_chat_view_empty() {
        let mut chat = ChatView::new();
        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        chat.render_widget(area, &mut buf);

        let lines = buffer_to_lines(&buf);
        let text = lines.join("\n");
        assert!(
            text.contains("No messages"),
            "Empty chat should show placeholder"
        );
    }

    #[test]
    fn test_chat_view_append_delta() {
        let mut chat = ChatView::new();
        chat.add_message(make_message(Role::Assistant, "Hello"));
        chat.append_delta(", world!");

        assert_eq!(chat.messages[0].content, "Hello, world!");
    }
}
