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

    let screen = ctx.screen_rect();
    let top_height = 40.0;
    let bottom_height = 48.0;

    let mut consumed_click = false;

    egui::Area::new(egui::Id::new("viewer_overlay"))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            // Top bar
            let top_rect = egui::Rect::from_min_size(
                screen.min,
                egui::vec2(screen.width(), top_height),
            );
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(top_rect), |ui| {
                ui.horizontal(|ui| {
                    ui.visuals_mut().override_text_color = Some(egui::Color32::WHITE);
                    let close = ui.button("✕  Close");
                    if close.clicked() {
                        resp.close = true;
                    }
                    consumed_click |= close.clicked();

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
            let bottom_rect = egui::Rect::from_min_size(
                egui::pos2(screen.min.x, screen.max.y - bottom_height),
                egui::vec2(screen.width(), bottom_height),
            );
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(bottom_rect), |ui| {
                ui.horizontal_centered(|ui| {
                    let prev = ui.button("← Previous");
                    if prev.clicked() {
                        resp.prev = true;
                    }
                    consumed_click |= prev.clicked();

                    let zoom = ui.button(if zoom_to_fit { "🔍 1:1" } else { "🔍 Fit" });
                    if zoom.clicked() {
                        resp.toggle_zoom = true;
                    }
                    consumed_click |= zoom.clicked();

                    let next = ui.button("Next →");
                    if next.clicked() {
                        resp.next = true;
                    }
                    consumed_click |= next.clicked();
                });
            });

            // Image area
            let image_rect = egui::Rect::from_min_max(
                egui::pos2(screen.min.x, screen.min.y + top_height),
                egui::pos2(screen.max.x, screen.max.y - bottom_height),
            );

            if let Some(texture) = texture {
                let tex_w = texture.size()[0] as f32;
                let tex_h = texture.size()[1] as f32;
                if tex_w > 0.0 && tex_h > 0.0 {
                    let avail = image_rect.size();
                    let display_size = if zoom_to_fit {
                        let scale = (avail.x / tex_w).min(avail.y / tex_h);
                        egui::vec2(tex_w * scale, tex_h * scale)
                    } else {
                        egui::vec2(tex_w, tex_h)
                    };

                    let img_rect = egui::Rect::from_center_size(image_rect.center(), display_size);
                    let img_resp = ui.put(
                        img_rect,
                        egui::Image::new((texture.id(), display_size))
                            .sense(egui::Sense::click()),
                    );
                    if img_resp.double_clicked() {
                        resp.toggle_zoom = true;
                    }
                    consumed_click |= img_resp.clicked() || img_resp.double_clicked();
                }
            } else {
                let spinner_rect = egui::Rect::from_center_size(
                    image_rect.center(),
                    egui::vec2(100.0, 100.0),
                );
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(spinner_rect), |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.spinner();
                        ui.label("Loading...");
                    });
                });
            }
        });

    // Close viewer when clicking on empty space (not on any widget)
    if !consumed_click && ctx.input(|i| i.pointer.primary_clicked()) {
        resp.close = true;
    }

    resp
}
