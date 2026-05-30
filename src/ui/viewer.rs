use eframe::egui;

pub struct ViewerPanel;

impl ViewerPanel {
    pub fn new() -> Self {
        Self
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        ui.label("Viewer panel (TODO)");
    }
}
