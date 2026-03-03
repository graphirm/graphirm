use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::app::StatusBar;

impl StatusBar {
    pub fn update_tokens(&mut self, input: u32, output: u32) {
        self.tokens_in += input;
        self.tokens_out += output;
        self.recalculate_cost();
    }

    fn recalculate_cost(&mut self) {
        // Rough estimate: $3/MTok input, $15/MTok output (Claude pricing)
        self.cost = (self.tokens_in as f64 * 3.0 / 1_000_000.0)
            + (self.tokens_out as f64 * 15.0 / 1_000_000.0);
    }

    pub fn render_widget(&self, area: Rect, buf: &mut Buffer) {
        let tokens_text = format!(
            "Tokens: {} in / {} out",
            format_number(self.tokens_in),
            format_number(self.tokens_out),
        );
        let cost_text = format!("Cost: ${:.4}", self.cost);
        let state_text = &self.agent_state;
        let model_text = &self.model;

        let line = Line::from(vec![
            Span::styled(
                format!(" {} ", model_text),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(" | "),
            Span::styled(tokens_text, Style::default().fg(Color::White)),
            Span::raw(" | "),
            Span::styled(cost_text, Style::default().fg(Color::Yellow)),
            Span::raw(" | "),
            Span::styled(
                format!("{} ", state_text),
                Style::default().fg(match state_text.as_str() {
                    "Working" | "Thinking..." => Color::Yellow,
                    "Tool Error" => Color::Red,
                    "Idle" => Color::DarkGray,
                    // "Running <tool>..." prefix
                    s if s.starts_with("Running") => Color::Cyan,
                    _ => Color::DarkGray,
                }),
            ),
        ]);

        let paragraph =
            Paragraph::new(line).style(Style::default().bg(Color::DarkGray).fg(Color::White));
        paragraph.render(area, buf);
    }
}

fn format_number(n: u32) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::StatusBar;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    fn buffer_text(buf: &Buffer) -> String {
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
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_status_bar_renders_model() {
        let status = StatusBar {
            tokens_in: 12_450,
            tokens_out: 3_200,
            cost: 0.04,
            model: "claude-4".to_string(),
            agent_state: "Idle".to_string(),
        };

        let area = Rect::new(0, 0, 80, 1);
        let mut buf = Buffer::empty(area);
        status.render_widget(area, &mut buf);

        let text = buffer_text(&buf);
        assert!(text.contains("claude-4"), "Should show model name");
        assert!(text.contains("Idle"), "Should show agent state");
    }

    #[test]
    fn test_status_bar_renders_token_counts() {
        let status = StatusBar {
            tokens_in: 500,
            tokens_out: 100,
            cost: 0.01,
            model: "gpt-4o".to_string(),
            agent_state: "Streaming".to_string(),
        };

        let area = Rect::new(0, 0, 80, 1);
        let mut buf = Buffer::empty(area);
        status.render_widget(area, &mut buf);

        let text = buffer_text(&buf);
        assert!(text.contains("500"), "Should show input token count");
        assert!(text.contains("100"), "Should show output token count");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1_500), "1.5K");
        assert_eq!(format_number(12_450), "12.4K");
        assert_eq!(format_number(1_500_000), "1.5M");
    }

    #[test]
    fn test_status_bar_update_tokens() {
        let mut status = StatusBar::new("test".to_string());
        status.update_tokens(1000, 200);
        assert_eq!(status.tokens_in, 1000);
        assert_eq!(status.tokens_out, 200);
        assert!(status.cost > 0.0);

        status.update_tokens(500, 100);
        assert_eq!(status.tokens_in, 1500);
        assert_eq!(status.tokens_out, 300);
    }
}
