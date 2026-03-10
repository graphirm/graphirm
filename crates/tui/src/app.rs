use std::io;
use std::time::Duration;

use chrono::Utc;
use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use graphirm_agent::AgentEvent;
use graphirm_llm::Role;

use crate::types::{AppState, ChatMessage, FocusPanel, GraphExplorer, InputField, StatusBar};

pub struct App {
    pub state: AppState,
    pub chat: crate::types::ChatView,
    pub graph_explorer: GraphExplorer,
    pub input: InputField,
    pub status_bar: StatusBar,
    pub event_rx: mpsc::Receiver<AgentEvent>,
    pub should_quit: bool,
    pub show_graph: bool,
    pub focus: FocusPanel,
}

impl App {
    pub fn new(event_rx: mpsc::Receiver<AgentEvent>, model: String) -> Self {
        Self {
            state: AppState::Idle,
            chat: crate::types::ChatView::new(),
            graph_explorer: GraphExplorer::new(),
            input: InputField::new(),
            status_bar: StatusBar::new(model),
            event_rx,
            should_quit: false,
            show_graph: false,
            focus: FocusPanel::Input,
        }
    }

    /// Run the TUI event loop. Blocks until the user quits.
    ///
    /// `on_submit` is called when the user presses Enter with non-empty input.
    /// It receives the message text and should send it to the agent.
    ///
    /// Requires `Send + 'static` so the caller can hand this off to
    /// `tokio::task::spawn_blocking` without blocking the async executor.
    pub fn run<F>(mut self, mut on_submit: F) -> io::Result<()>
    where
        F: FnMut(String) + Send + 'static,
    {
        use crate::events::handle_agent_event;
        use crate::input::{KeyAction, handle_key_event};
        use crate::ui::render_ui;

        // Install a panic hook that restores the terminal before printing the
        // panic message. Without this, any panic leaves the shell in raw mode
        // with the alternate screen active — the user loses their terminal.
        let original_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let _ = execute!(io::stdout(), LeaveAlternateScreen);
            original_hook(info);
        }));

        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        terminal.clear()?;

        loop {
            terminal.draw(|frame| {
                render_ui(frame, &mut self);
            })?;

            // Poll for keyboard events with 50ms timeout
            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match handle_key_event(&mut self, key) {
                        KeyAction::SubmitMessage(msg) => {
                            self.chat.add_message(ChatMessage {
                                role: Role::Human,
                                content: msg.clone(),
                                timestamp: Utc::now(),
                                node_id: None,
                                is_tool_call: false,
                                tool_name: None,
                            });
                            self.chat.scroll_to_bottom();
                            self.state = AppState::WaitingForAgent;
                            on_submit(msg);
                        }
                        KeyAction::Quit | KeyAction::None => {}
                    }
                }
            }

            // Drain agent events (non-blocking)
            while let Ok(agent_event) = self.event_rx.try_recv() {
                handle_agent_event(&mut self, agent_event);
            }

            if self.should_quit {
                break;
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        Ok(())
    }
}
