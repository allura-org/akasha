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
    pub job_kind: String,
    pub model_name: String,
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

    fn job_kind(&self) -> &'static str {
        match self {
            AiSubTab::VisionLanguage => "visionlanguage",
            AiSubTab::Tagger => "tagger",
            AiSubTab::Classifier => "classifier",
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
            ui.label(format!("Pending/running jobs: {}", pending_jobs));
            ui.add_space(8.0);

            if models.is_empty() {
                ui.label(format!(
                    "No {} models configured. Add them to config.toml under [[models]].",
                    sub_tab.label()
                ));
            } else {
                ui.label("Model:");
                let selected_key = format!("media_processing_model_{}", sub_tab.job_kind());
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
                            if ui.selectable_label(selected == i, &model.name).clicked() {
                                selected = i;
                            }
                        }
                    });

                ctx.memory_mut(|mem| {
                    mem.data.insert_persisted(egui::Id::new(selected_key), selected);
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Kind:");
                    ui.label(models[selected].kind_label());
                });
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

                ui.add_space(16.0);
                let can_go = target.is_some();
                if ui.add_enabled(can_go, egui::Button::new("Go")).clicked() {
                    if let Some(target) = target {
                        action = Some(MediaProcessingAction {
                            target: target.clone(),
                            job_kind: sub_tab.job_kind().to_string(),
                            model_name: models[selected].name.clone(),
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
