mod app;
mod config;
mod db;
mod scanner;
mod theme;
mod thumbnailer;
mod ui;

use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "heif")]
    libheif_rs::integration::image::register_all_decoding_hooks();

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("Akasha starting up...");

    let config = config::Config::load()?;
    info!("Loaded config: {:?}", config);

    let rt = tokio::runtime::Runtime::new()?;
    let db_path = config::Config::data_dir()?.join("akasha.db");
    let pool = rt.block_on(db::init_pool(db_path))?;
    info!("Database initialized");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Akasha"),
        ..Default::default()
    };

    eframe::run_native(
        "Akasha",
        options,
        Box::new(|cc| Ok(Box::new(app::AkashaApp::new(cc, config, pool, rt)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}
