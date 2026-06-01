use eframe::egui;
use crate::db::media::MediaSummary;

pub struct ViewerResponse {
    pub close: bool,
    pub prev: bool,
    pub next: bool,
    pub toggle_zoom: bool,
}

pub fn show(
    ctx: &egui::Context,
    media: &MediaSummary,
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
    let bottom_height = 80.0;

    let bottom_rect = egui::Rect::from_min_size(
        egui::pos2(screen.min.x, screen.max.y - bottom_height),
        egui::vec2(screen.width(), bottom_height),
    );
    let content_rect = egui::Rect::from_min_max(
        screen.min,
        egui::pos2(screen.max.x, screen.max.y - bottom_height),
    );
    let mut img_rect: Option<egui::Rect> = None;

    egui::Area::new(egui::Id::new("viewer_overlay"))
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.scope(|ui| {
                // Scale everything up ~50%
                let style = ui.style_mut();
                style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size *= 1.5;
                style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size *= 1.5;
                style.text_styles.get_mut(&egui::TextStyle::Heading).unwrap().size *= 1.5;

                // Translucent backdrop
                ui.painter().rect_filled(screen, 0.0, egui::Color32::from_black_alpha(180));

                // Image
                if let Some(texture) = texture {
                    let tex_w = texture.size()[0] as f32;
                    let tex_h = texture.size()[1] as f32;
                    if tex_w > 0.0 && tex_h > 0.0 {
                        let avail = content_rect.size();
                        let display_size = if zoom_to_fit {
                            let scale = (avail.x / tex_w).min(avail.y / tex_h);
                            egui::vec2(tex_w * scale, tex_h * scale)
                        } else {
                            egui::vec2(tex_w, tex_h)
                        };

                        let img_r = egui::Rect::from_center_size(content_rect.center(), display_size);
                        img_rect = Some(img_r);
                        let img_resp = ui.put(
                            img_r,
                            egui::Image::new((texture.id(), display_size))
                                .sense(egui::Sense::click()),
                        );
                        if img_resp.double_clicked() {
                            resp.toggle_zoom = true;
                        }
                    }
                } else {
                    let spinner_rect = egui::Rect::from_center_size(
                        content_rect.center(),
                        egui::vec2(100.0, 100.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(spinner_rect), |ui| {
                        ui.centered_and_justified(|ui| {
                            ui.spinner();
                            ui.label("Loading...");
                        });
                    });
                }

                // Bottom bar layout — exact positioning
                let pad = 12.0;
                let button_h = 48.0;
                let y = bottom_rect.max.y - button_h - pad;

                // Close — snapped to bottom-left
                let close_rect = egui::Rect::from_min_size(
                    egui::pos2(bottom_rect.min.x + pad, y),
                    egui::vec2(100.0, button_h),
                );
                if ui.put(close_rect, egui::Button::new("Close")).clicked() {
                    resp.close = true;
                }

                // Nav cluster — exactly centered on screen
                let nav_w = 320.0;
                let nav_rect = egui::Rect::from_min_size(
                    egui::pos2(bottom_rect.center().x - nav_w / 2.0, y),
                    egui::vec2(nav_w, button_h),
                );
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(nav_rect), |ui| {
                    ui.horizontal_centered(|ui| {
                        if ui.button("< Previous").clicked() {
                            resp.prev = true;
                        }
                        if ui.button(if zoom_to_fit { "1:1" } else { "Fit" }).clicked() {
                            resp.toggle_zoom = true;
                        }
                        if ui.button("Next >").clicked() {
                            resp.next = true;
                        }
                    });
                });

                // Info — snapped to bottom-right
                let info_left = bottom_rect.center().x + nav_w / 2.0 + 20.0;
                let info_rect = egui::Rect::from_min_size(
                    egui::pos2(info_left, y),
                    egui::vec2(bottom_rect.max.x - info_left - pad, button_h),
                );
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(info_rect), |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::BOTTOM), |ui| {
                        ui.label(format!(
                            "{}  \u{2022}  {}x{}  \u{2022}  {}",
                            media.format.as_deref().unwrap_or("unknown"),
                            media.width.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                            media.height.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                            media.absolute_path,
                        ));
                    });
                });
            });
        });

    // Close viewer when clicking on empty space
    if let Some(pos) = ctx.input(|i| i.pointer.interact_pos()) {
        if ctx.input(|i| i.pointer.primary_clicked()) {
            let on_image = img_rect.map_or_else(
                || content_rect.contains(pos),
                |r| r.contains(pos),
            );
            let on_bottom = bottom_rect.contains(pos);
            if !on_image && !on_bottom {
                resp.close = true;
            }
        }
    }

    resp
}
