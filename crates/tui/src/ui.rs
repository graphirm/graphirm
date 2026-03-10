use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::App;
use crate::types::{AppState, FocusPanel};

pub fn render_ui(frame: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(5),    // main content (chat + graph)
            Constraint::Length(3), // input
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

    render_header(frame, chunks[0], app);
    render_main_content(frame, chunks[1], app);
    render_input(frame, chunks[2], app);
    app.status_bar.render_widget(chunks[3], frame.buffer_mut());
}

fn render_header(frame: &mut Frame, area: Rect, app: &mut App) {
    let header = Line::from(vec![
        Span::styled(" graphirm ", Style::default().fg(Color::Cyan)),
        Span::raw("| "),
        Span::styled(
            format!("Model: {} ", app.status_bar.model),
            Style::default().fg(Color::White),
        ),
        Span::raw("| "),
        Span::styled(
            match app.state {
                AppState::Idle => "Ready",
                AppState::WaitingForAgent => "Thinking...",
                AppState::Streaming => "Streaming...",
            },
            Style::default().fg(match app.state {
                AppState::Idle => Color::Green,
                AppState::WaitingForAgent => Color::Yellow,
                AppState::Streaming => Color::Cyan,
            }),
        ),
    ]);

    let header_widget = Paragraph::new(header).style(Style::default().bg(Color::DarkGray));
    frame.render_widget(header_widget, area);
}

fn render_main_content(frame: &mut Frame, area: Rect, app: &mut App) {
    if app.show_graph {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);

        app.chat.render_widget(chunks[0], frame.buffer_mut());
        app.graph_explorer
            .render_widget(chunks[1], frame.buffer_mut());
    } else {
        app.chat.render_widget(area, frame.buffer_mut());
    }
}

fn render_input(frame: &mut Frame, area: Rect, app: &mut App) {
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(" Input ")
        .border_style(
            Style::default().fg(if matches!(app.focus, FocusPanel::Input) {
                Color::Cyan
            } else {
                Color::DarkGray
            }),
        );

    let inner = input_block.inner(area);
    frame.render_widget(input_block, area);

    let display_text = if app.input.is_empty() {
        Span::styled(
            "Type a message... (Ctrl+C to quit, F2 for graph, PgUp/PgDn to scroll)",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::raw(app.input.content.as_str())
    };

    let input_paragraph = Paragraph::new(Line::from(display_text));
    frame.render_widget(input_paragraph, inner);

    // Set cursor position when input panel is focused.
    // Use char count (not byte index) for the display column so multi-byte
    // characters (e.g. `é`, `→`) don't offset the visible caret.
    if matches!(app.focus, FocusPanel::Input) {
        let display_col = app.input.content[..app.input.cursor].chars().count() as u16;
        frame.set_cursor_position((inner.x + display_col, inner.y));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use tokio::sync::mpsc;

    #[test]
    fn test_layout_renders_without_panic() {
        let (_tx, rx) = mpsc::channel(16);
        let mut app = App::new(rx, "claude-4".to_string());

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render_ui(frame, &mut app);
            })
            .unwrap();
    }

    #[test]
    fn test_layout_with_graph_panel() {
        let (_tx, rx) = mpsc::channel(16);
        let mut app = App::new(rx, "claude-4".to_string());
        app.show_graph = true;

        let backend = TestBackend::new(120, 30);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render_ui(frame, &mut app);
            })
            .unwrap();
    }

    #[test]
    fn test_layout_shows_input_field() {
        let (_tx, rx) = mpsc::channel(16);
        let mut app = App::new(rx, "test-model".to_string());
        app.input.insert('H');
        app.input.insert('i');

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| {
                render_ui(frame, &mut app);
            })
            .unwrap();

        let buf = terminal.backend().buffer().clone();
        let text: String = (0..24)
            .flat_map(|y| {
                let buf = buf.clone();
                (0..80).map(move |x| {
                    buf.cell((x, y))
                        .map(|c| c.symbol().to_string())
                        .unwrap_or_default()
                })
            })
            .collect();

        assert!(text.contains("Hi"), "Input field content should be visible");
    }
}
