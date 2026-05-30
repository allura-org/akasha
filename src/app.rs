use eframe::egui;
use sqlx::SqlitePool;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

use crate::config::Config;

pub struct AkashaApp {
    pub config: Config,
    pub pool: Arc<Mutex<SqlitePool>>,
    pub rt: Arc<Runtime>,
}

impl AkashaApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config: Config,
        pool: SqlitePool,
        rt: Runtime,
    ) -> Self {
        if config.ui.theme == "dark" {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
        } else {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
        }

        Self {
            config,
            pool: Arc::new(Mutex::new(pool)),
            rt: Arc::new(rt),
        }
    }
}

impl eframe::App for AkashaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Akasha");
            ui.label("Scaffolding in progress...");
        });
    }
}
