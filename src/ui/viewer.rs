use eframe::egui;
use crate::db::media::MediaFile;

pub struct ViewerResponse {
    pub close: bool,
    pub prev: bool,
    pub next: bool,
    pub toggle_zoom: bool,
}

pub fn show(
    ctx: &egui::Context,
    media: &MediaFile,
    texture: &Option<egui::TextureHandle>,
    zoom_to_fit: bool,
) -> ViewerResponse {
    let mut resp = ViewerResponse {
        close: false,
        prev: false,
        next: false,
        toggle_zoom: false,
    };

    // Keyboard shortcuts
    ctx.input(|i| {
        if i.key_pressed(egui::Key::Escape) {
            resp.close = true;
        }
        if i.key_pressed(egui::Key::ArrowLeft) {
            resp.prev = true;
        }
        if i.key_pressed(egui::Key::ArrowRight) {
            resp.next = true;
        }
    });

    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(egui::Color32::from_black_alpha(245)))
        .show(ctx, |ui| {
            let full_rect = ui.max_rect();

            // Top bar
            let top_height = 36.0;
            let top_rect = egui::Rect::from_min_size(
                full_rect.min,
                egui::vec2(full_rect.width(), top_height),
            );
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(top_rect), |ui| {
                ui.horizontal(|ui| {
                    ui.visuals_mut().button_frame = false;
                    ui.visuals_mut().override_text_color = Some(egui::Color32::WHITE);
                    if ui.button("✕  Close").clicked() {
                        resp.close = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!(
                            "{}  •  {}x{}  •  {}",
                            media.format.as_deref().unwrap_or("unknown"),
                            media.width.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                            media.height.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                            media.absolute_path,
                        ));
                    });
                });
            });

            // Bottom bar
            let bottom_height = 44.0;
            let bottom_rect = egui::Rect::from_min_size(
                egui::pos2(full_rect.min.x, full_rect.max.y - bottom_height),
                egui::vec2(full_rect.width(), bottom_height),
            );
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(bottom_rect), |ui| {
                ui.horizontal_centered(|ui| {
                    ui.visuals_mut().button_frame = true;
                    if ui.button("← Previous").clicked() {
                        resp.prev = true;
                    }
                    ui.add_space(16.0);
                    if ui.button(if zoom_to_fit { "🔍 1:1" } else { "🔍 Fit" }).clicked() {
                        resp.toggle_zoom = true;
                    }
                    ui.add_space(16.0);
                    if ui.button("Next →").clicked() {
                        resp.next = true;
                    }
                });
            });

            // Image area (between top and bottom bars)
            let image_rect = egui::Rect::from_min_max(
                egui::pos2(full_rect.min.x, full_rect.min.y + top_height),
                egui::pos2(full_rect.max.x, full_rect.max.y - bottom_height),
            );
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(image_rect), |ui| {
                ui.centered_and_justified(|ui| {
                    if let Some(texture) = texture {
                        let tex_w = texture.size()[0] as f32;
                        let tex_h = texture.size()[1] as f32;
                        if tex_w > 0.0 && tex_h > 0.0 {
                            let avail = ui.available_size();
                            let display_size = if zoom_to_fit {
                                let scale = (avail.x / tex_w).min(avail.y / tex_h);
                                egui::vec2(tex_w * scale, tex_h * scale)
                            } else {
                                egui::vec2(tex_w, tex_h)
                            };
                            let img_resp = ui.add(
                                egui::Image::new((texture.id(), display_size))
                                    .fit_to_exact_size(display_size),
                            );
                            if img_resp.double_clicked() {
                                resp.toggle_zoom = true;
                            }
                            if img_resp.clicked() && ui.ctx().input(|i| i.modifiers.shift) {
                                resp.toggle_zoom = true;
                            }
                        }
                    } else {
                        ui.spinner();
                        ui.label("Loading image...");
                    }
                });
            });
        });

    resp
}
