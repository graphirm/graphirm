use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::app::{App, FocusPanel};

#[derive(Debug)]
pub enum KeyAction {
    None,
    SubmitMessage(String),
    Quit,
}

pub fn handle_key_event(app: &mut App, key: KeyEvent) -> KeyAction {
    // Global shortcuts (work regardless of focus)
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') => {
                app.should_quit = true;
                return KeyAction::Quit;
            }
            _ => {}
        }
    }

    // F2 toggles the graph panel (Ctrl+G is captured by Cursor/code-server)
    if key.code == KeyCode::F(2) {
        app.show_graph = !app.show_graph;
        if !app.show_graph && matches!(app.focus, FocusPanel::Graph) {
            app.focus = FocusPanel::Input;
        }
        return KeyAction::None;
    }

    // PgUp / PgDn scroll the chat from any panel — no Tab required.
    // When pinned to bottom, start from the last rendered offset so the view
    // scrolls up smoothly rather than jumping to the top of all messages.
    match key.code {
        KeyCode::PageUp => {
            let start = if app.chat.pinned_to_bottom {
                app.chat.last_computed_scroll
            } else {
                app.chat.scroll_offset
            };
            app.chat.pinned_to_bottom = false;
            app.chat.scroll_offset = start.saturating_sub(10);
            return KeyAction::None;
        }
        KeyCode::PageDown => {
            app.chat.scroll_offset = app.chat.scroll_offset.saturating_add(10);
            return KeyAction::None;
        }
        _ => {}
    }

    // Tab to cycle focus
    if key.code == KeyCode::Tab {
        app.focus = next_focus(app.focus, app.show_graph);
        return KeyAction::None;
    }

    // Panel-specific handling
    match app.focus {
        FocusPanel::Input => handle_input_key(app, key),
        FocusPanel::Chat => handle_chat_key(app, key),
        FocusPanel::Graph => handle_graph_key(app, key),
    }
}

fn handle_input_key(app: &mut App, key: KeyEvent) -> KeyAction {
    match key.code {
        KeyCode::Enter => {
            if app.input.is_empty() {
                return KeyAction::None;
            }
            let content = app.input.take();
            KeyAction::SubmitMessage(content)
        }
        KeyCode::Char(c) => {
            app.input.insert(c);
            KeyAction::None
        }
        KeyCode::Backspace => {
            app.input.backspace();
            KeyAction::None
        }
        KeyCode::Delete => {
            app.input.delete();
            KeyAction::None
        }
        KeyCode::Left => {
            app.input.move_left();
            KeyAction::None
        }
        KeyCode::Right => {
            app.input.move_right();
            KeyAction::None
        }
        KeyCode::Home => {
            app.input.home();
            KeyAction::None
        }
        KeyCode::End => {
            app.input.end();
            KeyAction::None
        }
        _ => KeyAction::None,
    }
}

fn handle_chat_key(app: &mut App, key: KeyEvent) -> KeyAction {
    match key.code {
        KeyCode::Up => {
            app.chat.scroll_up();
            KeyAction::None
        }
        KeyCode::Down => {
            app.chat.scroll_down();
            KeyAction::None
        }
        KeyCode::End => {
            app.chat.scroll_to_bottom();
            KeyAction::None
        }
        KeyCode::Char(c) => {
            // Typing in chat panel switches focus back to input automatically
            app.focus = FocusPanel::Input;
            app.input.insert(c);
            KeyAction::None
        }
        _ => KeyAction::None,
    }
}

fn handle_graph_key(app: &mut App, key: KeyEvent) -> KeyAction {
    match key.code {
        KeyCode::Up => {
            app.graph_explorer.select_prev();
            KeyAction::None
        }
        KeyCode::Down => {
            app.graph_explorer.select_next();
            KeyAction::None
        }
        KeyCode::Enter | KeyCode::Char(' ') => {
            app.graph_explorer.toggle_expand();
            KeyAction::None
        }
        _ => KeyAction::None,
    }
}

fn next_focus(current: FocusPanel, show_graph: bool) -> FocusPanel {
    match current {
        FocusPanel::Input => FocusPanel::Chat,
        FocusPanel::Chat => {
            if show_graph {
                FocusPanel::Graph
            } else {
                FocusPanel::Input
            }
        }
        FocusPanel::Graph => FocusPanel::Input,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use tokio::sync::mpsc;

    fn make_app() -> App {
        let (_tx, rx) = mpsc::channel(16);
        App::new(rx, "test".to_string())
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl_key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::CONTROL)
    }

    #[test]
    fn test_ctrl_c_quits() {
        let mut app = make_app();
        assert!(!app.should_quit);

        handle_key_event(&mut app, ctrl_key(KeyCode::Char('c')));
        assert!(app.should_quit);
    }

    #[test]
    fn test_typing_inserts_chars() {
        let mut app = make_app();
        handle_key_event(&mut app, key(KeyCode::Char('h')));
        handle_key_event(&mut app, key(KeyCode::Char('i')));

        assert_eq!(app.input.content, "hi");
    }

    #[test]
    fn test_backspace_deletes() {
        let mut app = make_app();
        handle_key_event(&mut app, key(KeyCode::Char('a')));
        handle_key_event(&mut app, key(KeyCode::Char('b')));
        handle_key_event(&mut app, key(KeyCode::Backspace));

        assert_eq!(app.input.content, "a");
    }

    #[test]
    fn test_enter_submits_message() {
        let mut app = make_app();
        handle_key_event(&mut app, key(KeyCode::Char('H')));
        handle_key_event(&mut app, key(KeyCode::Char('i')));

        let action = handle_key_event(&mut app, key(KeyCode::Enter));

        assert!(
            matches!(action, KeyAction::SubmitMessage(ref msg) if msg == "Hi"),
            "Enter should submit the input content"
        );
        assert!(app.input.is_empty(), "Input should be cleared after submit");
    }

    #[test]
    fn test_enter_on_empty_is_noop() {
        let mut app = make_app();
        let action = handle_key_event(&mut app, key(KeyCode::Enter));
        assert!(matches!(action, KeyAction::None));
    }

    #[test]
    fn test_f2_toggles_graph() {
        let mut app = make_app();
        assert!(!app.show_graph);

        handle_key_event(&mut app, key(KeyCode::F(2)));
        assert!(app.show_graph);

        handle_key_event(&mut app, key(KeyCode::F(2)));
        assert!(!app.show_graph);
    }

    #[test]
    fn test_arrow_keys_in_chat() {
        let mut app = make_app();
        app.focus = crate::app::FocusPanel::Chat;

        // Down scrolls toward newer content (higher offset, step 3)
        handle_key_event(&mut app, key(KeyCode::Down));
        assert_eq!(app.chat.scroll_offset, 3);

        // Up scrolls toward older content (lower offset, step 3)
        handle_key_event(&mut app, key(KeyCode::Up));
        assert_eq!(app.chat.scroll_offset, 0);
    }

    #[test]
    fn test_tab_cycles_focus() {
        let mut app = make_app();
        app.show_graph = true;
        assert!(matches!(app.focus, crate::app::FocusPanel::Input));

        handle_key_event(&mut app, key(KeyCode::Tab));
        assert!(matches!(app.focus, crate::app::FocusPanel::Chat));

        handle_key_event(&mut app, key(KeyCode::Tab));
        assert!(matches!(app.focus, crate::app::FocusPanel::Graph));

        handle_key_event(&mut app, key(KeyCode::Tab));
        assert!(matches!(app.focus, crate::app::FocusPanel::Input));
    }

    #[test]
    fn test_tab_skips_graph_when_hidden() {
        let mut app = make_app();
        app.show_graph = false;
        assert!(matches!(app.focus, crate::app::FocusPanel::Input));

        handle_key_event(&mut app, key(KeyCode::Tab));
        assert!(matches!(app.focus, crate::app::FocusPanel::Chat));

        handle_key_event(&mut app, key(KeyCode::Tab));
        assert!(matches!(app.focus, crate::app::FocusPanel::Input));
    }

    #[test]
    fn test_arrow_keys_move_cursor_in_input() {
        let mut app = make_app();
        handle_key_event(&mut app, key(KeyCode::Char('a')));
        handle_key_event(&mut app, key(KeyCode::Char('b')));
        handle_key_event(&mut app, key(KeyCode::Left));
        assert_eq!(app.input.cursor, 1);
        handle_key_event(&mut app, key(KeyCode::Right));
        assert_eq!(app.input.cursor, 2);
    }
}
