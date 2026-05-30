use eframe::egui;

pub struct BrowserPanel;

impl BrowserPanel {
    pub fn new() -> Self {
        Self
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.label("Browser panel (TODO)");
    }
}
