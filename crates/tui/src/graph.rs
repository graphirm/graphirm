use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Widget};

use crate::app::GraphExplorer;

impl GraphExplorer {
    pub fn select_next(&mut self) {
        if !self.nodes.is_empty() && self.selected < self.nodes.len() - 1 {
            self.selected += 1;
        }
    }

    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn toggle_expand(&mut self) {
        if let Some(node) = self.nodes.get(self.selected) {
            let id = node.id.clone();
            if self.expanded.contains(&id) {
                self.expanded.remove(&id);
            } else {
                self.expanded.insert(id);
            }
        }
    }

    pub fn render_widget(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default().borders(Borders::ALL).title(" Graph ");

        let inner = block.inner(area);
        block.render(area, buf);

        if self.nodes.is_empty() {
            let empty = ratatui::widgets::Paragraph::new("No graph data")
                .style(Style::default().fg(Color::DarkGray));
            empty.render(inner, buf);
            return;
        }

        let items: Vec<ListItem> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let indent = "  ".repeat(node.depth as usize);
                let expand_marker = if node.has_children {
                    if self.expanded.contains(&node.id) {
                        "▼ "
                    } else {
                        "▶ "
                    }
                } else {
                    "  "
                };

                let type_color = match node.node_type.as_str() {
                    "Agent" => Color::Cyan,
                    "Interaction" => Color::Green,
                    "Content" => Color::Yellow,
                    "Task" => Color::Magenta,
                    "Knowledge" => Color::Blue,
                    _ => Color::White,
                };

                let mut style = Style::default().fg(type_color);
                if i == self.selected {
                    style = style.bg(Color::DarkGray).add_modifier(Modifier::BOLD);
                }

                let line = Line::from(vec![
                    Span::raw(indent),
                    Span::raw(expand_marker.to_string()),
                    Span::styled(
                        format!("[{}] ", node.node_type),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::styled(node.label.chars().take(30).collect::<String>(), style),
                ]);

                ListItem::new(line)
            })
            .collect();

        let list = List::new(items);
        Widget::render(list, inner, buf);
    }
}

#[cfg(test)]
mod tests {
    use crate::app::{GraphExplorer, GraphNodeEntry};
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

    #[test]
    fn test_graph_explorer_renders_nodes() {
        let mut explorer = GraphExplorer::new();
        explorer.nodes.push(GraphNodeEntry {
            id: "agent-1".to_string(),
            label: "graphirm".to_string(),
            node_type: "Agent".to_string(),
            depth: 0,
            has_children: true,
        });
        explorer.nodes.push(GraphNodeEntry {
            id: "msg-1".to_string(),
            label: "Hello!".to_string(),
            node_type: "Interaction".to_string(),
            depth: 1,
            has_children: false,
        });

        let area = Rect::new(0, 0, 40, 10);
        let mut buf = Buffer::empty(area);
        explorer.render_widget(area, &mut buf);

        let lines = buffer_to_lines(&buf);
        let text = lines.join("\n");
        assert!(text.contains("graphirm"), "Should show agent node label");
    }

    #[test]
    fn test_graph_explorer_select_next_prev() {
        let mut explorer = GraphExplorer::new();
        for i in 0..5 {
            explorer.nodes.push(GraphNodeEntry {
                id: format!("n{}", i),
                label: format!("Node {}", i),
                node_type: "Interaction".to_string(),
                depth: 0,
                has_children: false,
            });
        }

        assert_eq!(explorer.selected, 0);
        explorer.select_next();
        assert_eq!(explorer.selected, 1);
        explorer.select_next();
        explorer.select_next();
        explorer.select_next();
        assert_eq!(explorer.selected, 4);
        explorer.select_next();
        assert_eq!(explorer.selected, 4);

        explorer.select_prev();
        assert_eq!(explorer.selected, 3);
    }

    #[test]
    fn test_graph_explorer_toggle_expand() {
        let mut explorer = GraphExplorer::new();
        explorer.nodes.push(GraphNodeEntry {
            id: "n1".to_string(),
            label: "Agent".to_string(),
            node_type: "Agent".to_string(),
            depth: 0,
            has_children: true,
        });

        assert!(!explorer.expanded.contains("n1"));
        explorer.toggle_expand();
        assert!(explorer.expanded.contains("n1"));
        explorer.toggle_expand();
        assert!(!explorer.expanded.contains("n1"));
    }
}
