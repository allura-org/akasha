use eframe::egui;

use crate::config::Config;

pub enum SettingsAction {
    ThumbnailSizeChanged(u32),
    ThemeChanged(bool),
}

pub fn show(ctx: &egui::Context, open: &mut bool, config: &mut Config) -> Vec<SettingsAction> {
    let mut actions = Vec::new();

    egui::Window::new("Settings")
        .open(open)
        .resizable(false)
        .collapsible(false)
        .default_width(400.0)
        .show(ctx, |ui| {
            ui.heading("Appearance");
            ui.separator();

            let mut dark = config.ui.theme == "dark";
            if ui.checkbox(&mut dark, "Dark theme").changed() {
                config.ui.theme = if dark { "dark".to_string() } else { "light".to_string() };
                actions.push(SettingsAction::ThemeChanged(dark));
            }

            ui.add_space(16.0);
            ui.heading("Thumbnails");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Size:");
                let mut size = config.ui.thumbnail_size as f32;
                if ui.add(egui::Slider::new(&mut size, 64.0..=512.0).step_by(16.0)).changed() {
                    config.ui.thumbnail_size = size as u32;
                    actions.push(SettingsAction::ThumbnailSizeChanged(size as u32));
                }
            });

            ui.add_space(16.0);
            ui.heading("Folders");
            ui.separator();
            ui.label("Edit ~/.config/akasha/config.toml to add or remove folders.");
            ui.label("Changes require a restart to take full effect.");

            ui.add_space(8.0);
            for folder in &config.folders {
                ui.label(format!("• {}", folder.path));
            }
        });

    actions
}
