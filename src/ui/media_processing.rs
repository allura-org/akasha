use eframe::egui;

use crate::config::{Config, ModelConfig};

#[derive(Debug, Clone)]
pub enum MediaProcessingTarget {
    Single(i64),
    Folder(i64, bool),
}

impl MediaProcessingTarget {
    pub fn label(&self) -> String {
        match self {
            MediaProcessingTarget::Single(_) => "1 file".to_string(),
            MediaProcessingTarget::Folder(_, recursive) => {
                if *recursive {
                    "folder (recursive)".to_string()
                } else {
                    "folder".to_string()
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MediaProcessingAction {
    pub target: MediaProcessingTarget,
    pub source_name: String,
    pub output_kind: String,
    pub model_name: String,
    pub overwrite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AiSubTab {
    VisionLanguage,
    Tagger,
    Classifier,
}

impl AiSubTab {
    fn label(&self) -> &'static str {
        match self {
            AiSubTab::VisionLanguage => "Vision-Language",
            AiSubTab::Tagger => "Tagger",
            AiSubTab::Classifier => "Classifier",
        }
    }

    fn output_kind(&self) -> &'static str {
        match self {
            AiSubTab::VisionLanguage => "description",
            AiSubTab::Tagger => "tags",
            AiSubTab::Classifier => "classification",
        }
    }

    fn models<'a>(&self, config: &'a Config) -> Vec<&'a ModelConfig> {
        config
            .models
            .models
            .iter()
            .filter(|m| match self {
                AiSubTab::VisionLanguage => m.description.is_some(),
                AiSubTab::Tagger => m.tags.is_some(),
                AiSubTab::Classifier => m.classification.is_some(),
            })
            .collect()
    }
}

pub fn show(
    ctx: &egui::Context,
    open: &mut bool,
    config: &Config,
    target: Option<&MediaProcessingTarget>,
    pending_jobs: usize,
    running: bool,
    toggle: &mut bool,
) -> Option<MediaProcessingAction> {
    let mut action = None;
    let mut close = false;

    egui::Window::new("Media Processing")
        .open(open)
        .resizable(false)
        .collapsible(false)
        .default_width(450.0)
        .show(ctx, |ui| {
            // Top-level tabs. Only AI exists for now.
            ui.horizontal(|ui| {
                let _ = ui.selectable_label(true, "AI");
            });
            ui.separator();

            // AI subtabs.
            let mut sub_tab = ctx.memory_mut(|mem| {
                mem.data
                    .get_persisted(egui::Id::new("media_processing_sub_tab"))
                    .unwrap_or(AiSubTab::Tagger)
            });

            ui.horizontal(|ui| {
                for tab in [AiSubTab::VisionLanguage, AiSubTab::Tagger, AiSubTab::Classifier] {
                    if ui
                        .selectable_label(sub_tab == tab, tab.label())
                        .clicked()
                    {
                        sub_tab = tab;
                    }
                }
            });
            ui.separator();

            ctx.memory_mut(|mem| {
                mem.data
                    .insert_persisted(egui::Id::new("media_processing_sub_tab"), sub_tab);
            });

            let models = sub_tab.models(config);

            if let Some(target) = target {
                ui.label(format!("Target: {}", target.label()));
            } else {
                ui.label("Target: nothing selected");
            }
            ui.horizontal(|ui| {
                let status = if running { "running" } else { "paused" };
                ui.label(format!("Queue: {} ({} pending/running)", status, pending_jobs));
                if ui.button(if running { "Pause" } else { "Resume" }).clicked() {
                    *toggle = true;
                }
            });
            ui.add_space(8.0);

            if models.is_empty() {
                ui.label(format!(
                    "No {} models configured. Add them to config.toml under [[models]].",
                    sub_tab.label()
                ));
            } else {
                ui.label("Model:");
                let selected_key = format!("media_processing_model_{}", sub_tab.output_kind());
                let mut selected: usize = ctx.memory_mut(|mem| {
                    mem.data
                        .get_persisted(egui::Id::new(&selected_key))
                        .unwrap_or(0)
                });
                selected = selected.min(models.len().saturating_sub(1));

                egui::ComboBox::from_id_salt(&selected_key)
                    .selected_text(&models[selected].name)
                    .show_ui(ui, |ui| {
                        for (i, model) in models.iter().enumerate() {
                            ui.horizontal(|ui| {
                                if ui.selectable_label(selected == i, &model.name).clicked() {
                                    selected = i;
                                }
                                ui.label(model.kind_label());
                            });
                        }
                    });

                ctx.memory_mut(|mem| {
                    mem.data.insert_persisted(egui::Id::new(selected_key), selected);
                });

                ui.add_space(8.0);
                if let Some(path) = &models[selected].path {
                    ui.horizontal(|ui| {
                        ui.label("Path:");
                        ui.label(path);
                    });
                }
                if let Some(base_url) = &models[selected].base_url {
                    ui.horizontal(|ui| {
                        ui.label("Base URL:");
                        ui.label(base_url);
                    });
                }
                if let Some(model_id) = &models[selected].model_id {
                    ui.horizontal(|ui| {
                        ui.label("Model ID:");
                        ui.label(model_id);
                    });
                }

                let any_local_configured = config
                    .models
                    .models
                    .iter()
                    .any(|m| m.kind == crate::config::ModelKind::Local);
                if any_local_configured && !cfg!(feature = "candle") {
                    ui.add_space(8.0);
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "Local models are configured, but candle support was not compiled in. \
                         Enable the 'candle' feature and rebuild to run them.",
                    );
                }

                if models[selected].kind == crate::config::ModelKind::Local {
                    ui.add_space(8.0);
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "Local CPU inference can take days on very large collections and only runs while Akasha is open.",
                    );
                }

                ui.add_space(8.0);
                let overwrite_key = format!("media_processing_overwrite_{}", sub_tab.output_kind());
                let mut overwrite: bool = ctx.memory_mut(|mem| {
                    mem.data
                        .get_persisted(egui::Id::new(&overwrite_key))
                        .unwrap_or(false)
                });
                ui.checkbox(&mut overwrite, "Overwrite existing predictions");
                ctx.memory_mut(|mem| {
                    mem.data.insert_persisted(egui::Id::new(overwrite_key), overwrite);
                });

                ui.add_space(16.0);
                let can_go = target.is_some();
                if ui.add_enabled(can_go, egui::Button::new("Go")).clicked() {
                    if let Some(target) = target {
                        action = Some(MediaProcessingAction {
                            target: target.clone(),
                            source_name: models[selected].name.clone(),
                            output_kind: sub_tab.output_kind().to_string(),
                            model_name: models[selected].name.clone(),
                            overwrite,
                        });
                        close = true;
                    }
                }
            }
        });

    if close {
        *open = false;
    }

    action
}

trait ModelKindLabel {
    fn kind_label(&self) -> &'static str;
}

impl ModelKindLabel for ModelConfig {
    fn kind_label(&self) -> &'static str {
        match self.kind {
            crate::config::ModelKind::Local => "local",
            crate::config::ModelKind::Remote => "remote",
        }
    }
}
