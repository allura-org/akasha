use eframe::egui;

use crate::config::{Config, ViewerScaleMode};

pub enum SettingsAction {
    ThumbnailSizeChanged(u32),
    ThemeChanged(bool),
    DoubleClickDebounceChanged,
    ScrollSpeedChanged(f32),
    ViewerDefaultScaleModeChanged,
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
            ui.heading("Interaction");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Double-click debounce:");
                let mut ms = config.ui.double_click_debounce_ms as f32;
                if ui.add(egui::Slider::new(&mut ms, 100.0..=1000.0).step_by(50.0).suffix(" ms")).changed() {
                    config.ui.double_click_debounce_ms = ms as u64;
                    actions.push(SettingsAction::DoubleClickDebounceChanged);
                }
            });

            ui.horizontal(|ui| {
                ui.label("Scroll speed:");
                let mut speed = config.ui.scroll_speed;
                if ui.add(egui::Slider::new(&mut speed, 0.5..=3.0).step_by(0.1).suffix("×")).changed() {
                    config.ui.scroll_speed = speed;
                    actions.push(SettingsAction::ScrollSpeedChanged(speed));
                }
            });

            ui.add_space(16.0);
            ui.heading("Viewer");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Default scale mode:");
                egui::ComboBox::from_id_salt("viewer_default_scale_mode")
                    .selected_text(config.ui.viewer_default_scale_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [ViewerScaleMode::Fit, ViewerScaleMode::OneToOne, ViewerScaleMode::Smallest] {
                            if ui.selectable_label(config.ui.viewer_default_scale_mode == mode, mode.label()).clicked() {
                                config.ui.viewer_default_scale_mode = mode;
                                actions.push(SettingsAction::ViewerDefaultScaleModeChanged);
                            }
                        }
                    });
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
