use eframe::egui;

pub fn apply(ctx: &egui::Context, dark: bool) {
    let mut visuals = if dark { egui::Visuals::dark() } else { egui::Visuals::light() };

    // Content-first dark palette
    if dark {
        visuals.panel_fill = egui::Color32::from_rgb(24, 24, 26);
        visuals.extreme_bg_color = egui::Color32::from_rgb(16, 16, 18);
        visuals.faint_bg_color = egui::Color32::from_rgb(32, 32, 36);
        visuals.window_fill = egui::Color32::from_rgb(28, 28, 32);
        visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(48, 48, 52));
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(32, 32, 36);
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(42, 42, 48);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(52, 52, 60);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(62, 62, 72);
        visuals.selection.bg_fill = egui::Color32::from_rgb(70, 100, 140);
        visuals.hyperlink_color = egui::Color32::from_rgb(100, 160, 220);
    }

    // Flat, sharp corners — no rounding
    let zero = egui::CornerRadius::ZERO;
    visuals.window_corner_radius = zero;
    visuals.menu_corner_radius = zero;
    visuals.widgets.noninteractive.corner_radius = zero;
    visuals.widgets.inactive.corner_radius = zero;
    visuals.widgets.hovered.corner_radius = zero;
    visuals.widgets.active.corner_radius = zero;
    visuals.widgets.open.corner_radius = zero;

    ctx.set_visuals(visuals);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(10.0, 10.0);
    style.spacing.window_margin = egui::Margin::same(16);
    ctx.set_style(style);
}
