use eframe::egui;
use crate::config::ViewerScaleMode;
use crate::db::media::MediaSummary;

pub struct ViewerResponse {
    pub close: bool,
    pub prev: bool,
    pub next: bool,
    pub cycle_scale_mode: bool,
    pub show_in_file_manager: bool,
    pub copy_to_clipboard: bool,
    pub process_with_ai: bool,
    pub context_menu_used: bool,
}

pub fn show(
    ctx: &egui::Context,
    media: &MediaSummary,
    texture: &Option<egui::TextureHandle>,
    scale_mode: ViewerScaleMode,
    missing: bool,
) -> ViewerResponse {
    let mut resp = ViewerResponse {
        close: false,
        prev: false,
        next: false,
        cycle_scale_mode: false,
        show_in_file_manager: false,
        copy_to_clipboard: false,
        process_with_ai: false,
        context_menu_used: false,
    };

    // Keyboard shortcuts
    ctx.input(|i| {
        if i.key_pressed(egui::Key::Escape) || i.key_pressed(egui::Key::ArrowDown) {
            resp.close = true;
        }
        if i.key_pressed(egui::Key::ArrowLeft) {
            resp.prev = true;
        }
        if i.key_pressed(egui::Key::ArrowRight) {
            resp.next = true;
        }
        if i.key_pressed(egui::Key::ArrowUp) {
            resp.cycle_scale_mode = true;
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
    let mut img_resp: Option<egui::Response> = None;

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
                if missing {
                    let text_rect = egui::Rect::from_center_size(
                        content_rect.center(),
                        egui::vec2(200.0, 100.0),
                    );
                    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(text_rect), |ui| {
                        ui.centered_and_justified(|ui| {
                            ui.label(egui::RichText::new("File is missing").size(24.0).color(egui::Color32::LIGHT_GRAY));
                        });
                    });
                } else if let Some(texture) = texture {
                    let tex_w = texture.size()[0] as f32;
                    let tex_h = texture.size()[1] as f32;
                    if tex_w > 0.0 && tex_h > 0.0 {
                        let avail = content_rect.size();
                        let display_size = if scale_mode == ViewerScaleMode::Fit {
                            let scale = (avail.x / tex_w).min(avail.y / tex_h);
                            egui::vec2(tex_w * scale, tex_h * scale)
                        } else {
                            egui::vec2(tex_w, tex_h)
                        };

                        let img_r = egui::Rect::from_center_size(content_rect.center(), display_size);
                        img_rect = Some(img_r);
                        let image_response = ui.put(
                            img_r,
                            egui::Image::new((texture.id(), display_size))
                                .sense(egui::Sense::click()),
                        );
                        image_response.context_menu(|ui| {
                            if ui.button("Process with AI…").clicked() {
                                resp.process_with_ai = true;
                                resp.context_menu_used = true;
                                ui.close_menu();
                            }
                            if ui.button("Show in file manager").clicked() {
                                resp.show_in_file_manager = true;
                                resp.context_menu_used = true;
                                ui.close_menu();
                            }
                            if ui.button("Copy to clipboard").clicked() {
                                resp.copy_to_clipboard = true;
                                resp.context_menu_used = true;
                                ui.close_menu();
                            }
                        });
                        if image_response.middle_clicked() {
                            resp.cycle_scale_mode = true;
                        }
                        img_resp = Some(image_response);
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
                let close_resp = ui.put(close_rect, egui::Button::new("Close"));
                if close_resp.clicked() {
                    resp.close = true;
                }

                // Nav cluster — exactly centered on screen
                let nav_w = 320.0;
                let nav_rect = egui::Rect::from_min_size(
                    egui::pos2(bottom_rect.center().x - nav_w / 2.0, y),
                    egui::vec2(nav_w, button_h),
                );
                let mut prev_clicked = false;
                let mut scale_clicked = false;
                let mut next_clicked = false;
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(nav_rect), |ui| {
                    ui.horizontal_centered(|ui| {
                        prev_clicked = ui.button("< Previous").clicked();
                        if prev_clicked {
                            resp.prev = true;
                        }
                        scale_clicked = ui.button(scale_mode.label()).clicked();
                        if scale_clicked {
                            resp.cycle_scale_mode = true;
                        }
                        next_clicked = ui.button("Next >").clicked();
                        if next_clicked {
                            resp.next = true;
                        }
                    });
                });
                let button_clicked = close_resp.clicked()
                    || prev_clicked
                    || scale_clicked
                    || next_clicked;

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

                // Scroll wheel navigates images one per notch.
                // Use raw_scroll_delta so a single wheel notch produces one event,
                // not several frames of smoothed velocity.
                let scroll_delta = ui.input(|i| i.raw_scroll_delta.y);
                if scroll_delta < -1.0 {
                    resp.next = true;
                } else if scroll_delta > 1.0 {
                    resp.prev = true;
                }
                ui.input_mut(|i| i.raw_scroll_delta = egui::Vec2::ZERO);

                // Close viewer on left-click anywhere that isn't a button or context menu item.
                if !button_clicked && !resp.context_menu_used && ui.input(|i| i.pointer.primary_clicked()) {
                    resp.close = true;
                }
            });
        });

    resp
}
