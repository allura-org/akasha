use eframe::egui;

use crate::db::media::PropertiesData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertiesTab {
    General,
    Tags,
    Descriptions,
    Classifications,
    Embeddings,
}

impl PropertiesTab {
    fn label(&self) -> &'static str {
        match self {
            PropertiesTab::General => "General",
            PropertiesTab::Tags => "Tags",
            PropertiesTab::Descriptions => "Descriptions",
            PropertiesTab::Classifications => "Classifications",
            PropertiesTab::Embeddings => "Embeddings",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PropertiesState {
    pub open: bool,
    pub media_id: Option<i64>,
}

impl Default for PropertiesState {
    fn default() -> Self {
        Self { open: false, media_id: None }
    }
}

pub enum PropertiesAction {
    Open(i64),
}

pub fn show(
    ctx: &egui::Context,
    open: &mut bool,
    media_id: Option<i64>,
    data: Option<&PropertiesData>,
    advanced: bool,
) -> Vec<PropertiesAction> {
    let mut actions = Vec::new();

    egui::Window::new("Properties")
        .open(open)
        .resizable(true)
        .collapsible(false)
        .default_width(500.0)
        .default_height(600.0)
        .show(ctx, |ui| {
            let Some(media_id) = media_id else {
                ui.label("No media selected.");
                return;
            };

            let Some(data) = data else {
                ui.label("Loading...");
                return;
            };

            // Tabs
            let mut tab = ctx.memory_mut(|mem| {
                mem.data
                    .get_persisted(egui::Id::new("properties_tab"))
                    .unwrap_or(PropertiesTab::General)
            });

            ui.horizontal(|ui| {
                for t in [
                    PropertiesTab::General,
                    PropertiesTab::Tags,
                    PropertiesTab::Descriptions,
                    PropertiesTab::Classifications,
                    PropertiesTab::Embeddings,
                ] {
                    if ui.selectable_label(tab == t, t.label()).clicked() {
                        tab = t;
                    }
                }
            });
            ui.separator();

            ctx.memory_mut(|mem| {
                mem.data.insert_persisted(egui::Id::new("properties_tab"), tab);
            });

            egui::ScrollArea::vertical().show(ui, |ui| {
                match tab {
                    PropertiesTab::General => show_general(ui, data, advanced),
                    PropertiesTab::Tags => show_tags(ui, data),
                    PropertiesTab::Descriptions => show_descriptions(ui, data),
                    PropertiesTab::Classifications => show_classifications(ui, data),
                    PropertiesTab::Embeddings => show_embeddings(ui, data),
                }
            });
        });

    actions
}

fn format_bytes(bytes: i64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn show_general(ui: &mut egui::Ui, data: &PropertiesData, advanced: bool) {
    let m = &data.media;
    ui.label(format!("Filename: {}", std::path::Path::new(&m.relative_path).file_name().map(|s| s.to_string_lossy()).unwrap_or_default()));
    ui.horizontal(|ui| {
        ui.label("Absolute path:");
        ui.add(
            egui::TextEdit::singleline(&mut m.absolute_path.clone())
                .desired_width(f32::INFINITY),
        );
    });
    ui.label(format!("Folder ID: {}", m.folder_id));
    ui.label(format!("Dimensions: {}x{}", m.width.unwrap_or(0), m.height.unwrap_or(0)));
    ui.label(format!("Format: {}", m.format.as_deref().unwrap_or("unknown")));
    ui.label(format!("Size: {}", format_bytes(m.file_size.unwrap_or(0))));
    ui.label(format!("Created: {}", m.created_at));
    if let Some(modified) = m.modified_at {
        ui.label(format!("Modified: {}", modified));
    }
    ui.label(format!("Present: {}", if m.is_present { "yes" } else { "no" }));
    ui.label(format!("Hash: {}", m.blake3_hash));

    if advanced {
        ui.separator();
        ui.heading("Advanced");
        ui.label(format!("ID: {}", m.id));
        ui.label(format!("Folder ID: {}", m.folder_id));
        if let Some(missing) = m.missing_since {
            ui.label(format!("Missing since: {}", missing));
        }
    }
}

fn source_switcher(
    ui: &mut egui::Ui,
    sources: &[String],
    selected: &mut usize,
    id_salt: &str,
) {
    ui.horizontal(|ui| {
        let available_width = ui.available_width();
        let dropdown_width = 200.0f32.min(available_width - 80.0).max(100.0);

        let prev_enabled = !sources.is_empty() && *selected > 0;
        if ui.add_enabled(prev_enabled, egui::Button::new("<-")).clicked() {
            *selected = selected.saturating_sub(1);
        }

        egui::ComboBox::from_id_salt(id_salt)
            .width(dropdown_width)
            .selected_text(sources.get(*selected).map(|s| s.as_str()).unwrap_or("(none)"))
            .show_ui(ui, |ui| {
                for (i, source) in sources.iter().enumerate() {
                    if ui.selectable_label(*selected == i, source).clicked() {
                        *selected = i;
                    }
                }
            });

        let next_enabled = !sources.is_empty() && *selected + 1 < sources.len();
        if ui.add_enabled(next_enabled, egui::Button::new("->")).clicked() {
            *selected += 1;
        }
    });
}

fn show_tags(ui: &mut egui::Ui, data: &PropertiesData) {
    let sources: Vec<String> = data.tags.keys().cloned().collect();
    let mut selected = ui.memory_mut(|mem| {
        mem.data
            .get_persisted(egui::Id::new("properties_tags_source"))
            .unwrap_or(0usize)
    });
    selected = selected.min(sources.len().saturating_sub(1));

    source_switcher(ui, &sources, &mut selected, "properties_tags_source");

    ui.memory_mut(|mem| {
        mem.data.insert_persisted(egui::Id::new("properties_tags_source"), selected);
    });

    if let Some(source) = sources.get(selected) {
        if let Some(tags) = data.tags.get(source) {
            let mut sorted: Vec<_> = tags.iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (tag, score) in sorted {
                ui.horizontal(|ui| {
                    ui.label(tag);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("{:.3}", score));
                    });
                });
            }
        }
    }
}

fn show_descriptions(ui: &mut egui::Ui, data: &PropertiesData) {
    let sources: Vec<String> = data.descriptions.keys().cloned().collect();
    let mut selected = ui.memory_mut(|mem| {
        mem.data
            .get_persisted(egui::Id::new("properties_desc_source"))
            .unwrap_or(0usize)
    });
    selected = selected.min(sources.len().saturating_sub(1));

    source_switcher(ui, &sources, &mut selected, "properties_desc_source");

    ui.memory_mut(|mem| {
        mem.data.insert_persisted(egui::Id::new("properties_desc_source"), selected);
    });

    if let Some(source) = sources.get(selected) {
        if let Some(text) = data.descriptions.get(source) {
            let mut text = text.clone();
            ui.add(
                egui::TextEdit::multiline(&mut text)
                    .desired_width(f32::INFINITY)
                    .interactive(false),
            );
        }
    }
}

fn show_classifications(ui: &mut egui::Ui, data: &PropertiesData) {
    let sources: Vec<String> = data.classifications.keys().cloned().collect();
    let mut selected = ui.memory_mut(|mem| {
        mem.data
            .get_persisted(egui::Id::new("properties_cls_source"))
            .unwrap_or(0usize)
    });
    selected = selected.min(sources.len().saturating_sub(1));

    source_switcher(ui, &sources, &mut selected, "properties_cls_source");

    ui.memory_mut(|mem| {
        mem.data.insert_persisted(egui::Id::new("properties_cls_source"), selected);
    });

    if let Some(source) = sources.get(selected) {
        if let Some(classes) = data.classifications.get(source) {
            for class in classes {
                ui.label(class);
            }
        }
    }
}

fn show_embeddings(ui: &mut egui::Ui, _data: &PropertiesData) {
    ui.label("Embeddings viewer not yet implemented.");
}
