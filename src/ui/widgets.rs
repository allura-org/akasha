use eframe::egui;

pub fn placeholder(ui: &mut egui::Ui, text: &str) {
    ui.label(format!("[Widget: {}]", text));
}
