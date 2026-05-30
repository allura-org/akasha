use eframe::egui;
use sqlx::SqlitePool;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

use crate::config::{Config, FolderConfig};
use crate::db;
use crate::thumbnailer::{CacheMode, Thumbnailer};

#[derive(Debug, Clone)]
pub enum ScanEvent {
    Started(String),
    Progress(String, usize),
    Complete(String, usize),
    Error(String, String),
}

pub struct Toast {
    pub message: String,
    pub level: ToastLevel,
    pub created_at: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastLevel {
    Info,
    Warning,
    Error,
}

pub struct AkashaApp {
    pub config: Config,
    pub pool: Arc<Mutex<SqlitePool>>,
    pub rt: Arc<Runtime>,
    pub thumbnailer: Thumbnailer,

    // UI state
    pub folders: Vec<db::folder::Folder>,
    pub selected_folder: Option<usize>,
    pub media_items: Vec<db::media::MediaFile>,
    pub textures: HashMap<String, egui::TextureHandle>,
    pub pending_thumbnails: HashSet<String>,
    pub scan_status: String,
    pub is_scanning: bool,
    pub scan_rx: std::sync::mpsc::Receiver<ScanEvent>,
    pub thumbnail_tx: std::sync::mpsc::Sender<(String, Result<Vec<u8>, String>)>,
    pub thumbnail_rx: std::sync::mpsc::Receiver<(String, Result<Vec<u8>, String>)>,

    // Viewer state
    pub viewer_open: bool,
    pub viewer_index: Option<usize>,
    pub viewer_texture: Option<egui::TextureHandle>,
    pub viewer_zoom_to_fit: bool,
    pub viewer_image_tx: std::sync::mpsc::Sender<(String, Result<egui::ColorImage, String>)>,
    pub viewer_image_rx: std::sync::mpsc::Receiver<(String, Result<egui::ColorImage, String>)>,

    // Polish
    pub toasts: Vec<Toast>,
    pub settings_open: bool,
}

impl AkashaApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config: Config,
        pool: SqlitePool,
        rt: Runtime,
    ) -> Self {
        crate::theme::apply(&cc.egui_ctx, config.ui.theme == "dark");

        let cache_mode = CacheMode::from_config(
            &config.thumbnails.cache_mode,
            &config.thumbnails.custom_path,
        );
        let thumbnailer = Thumbnailer::new(config.ui.thumbnail_size, cache_mode);

        let pool_arc = Arc::new(Mutex::new(pool));
        let rt_arc = Arc::new(rt);
        let (scan_tx, scan_rx) = std::sync::mpsc::channel();
        let (thumb_tx, thumbnail_rx) = std::sync::mpsc::channel::<(String, Result<Vec<u8>, String>)>();
        let (viewer_img_tx, viewer_img_rx) = std::sync::mpsc::channel::<(String, Result<egui::ColorImage, String>)>();

        // Clone for background scan task
        let pool_clone = Arc::clone(&pool_arc);
        let rt_clone = Arc::clone(&rt_arc);
        let folders_config: Vec<FolderConfig> = config.folders.clone();

        rt_clone.spawn(async move {
            let pool = pool_clone.lock().await;
            for folder_cfg in &folders_config {
                let _ = scan_tx.send(ScanEvent::Started(folder_cfg.path.clone()));

                let cache_mode = folder_cfg.thumbnail_cache_mode.as_deref();
                match db::folder::insert_or_get(
                    &*pool,
                    &folder_cfg.path,
                    folder_cfg.recursive,
                    &folder_cfg.blacklist,
                    cache_mode,
                )
                .await
                {
                    Ok(folder_id) => {
                        let path = std::path::Path::new(&folder_cfg.path);
                        match crate::scanner::scan_folder(
                            &*pool,
                            folder_id,
                            path,
                            folder_cfg.recursive,
                            &folder_cfg.blacklist,
                        )
                        .await
                        {
                            Ok(count) => {
                                let _ = scan_tx.send(ScanEvent::Complete(
                                    folder_cfg.path.clone(),
                                    count,
                                ));
                            }
                            Err(e) => {
                                let _ = scan_tx.send(ScanEvent::Error(
                                    folder_cfg.path.clone(),
                                    e.to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        let _ = scan_tx.send(ScanEvent::Error(
                            folder_cfg.path.clone(),
                            e.to_string(),
                        ));
                    }
                }
            }
        });

        let mut app = Self {
            config,
            pool: pool_arc,
            rt: rt_arc,
            thumbnailer,
            folders: Vec::new(),
            selected_folder: None,
            media_items: Vec::new(),
            textures: HashMap::new(),
            pending_thumbnails: HashSet::new(),
            scan_status: "Initializing...".to_string(),
            is_scanning: true,
            scan_rx,
            thumbnail_tx: thumb_tx,
            thumbnail_rx,
            viewer_open: false,
            viewer_index: None,
            viewer_texture: None,
            viewer_zoom_to_fit: true,
            viewer_image_tx: viewer_img_tx,
            viewer_image_rx: viewer_img_rx,
            toasts: Vec::new(),
            settings_open: false,
        };

        app.refresh_folders_blocking();
        app
    }

    fn refresh_folders_blocking(&mut self) {
        let pool = Arc::clone(&self.pool);
        match self.rt.block_on(async move {
            let p = pool.lock().await;
            db::folder::list_all(&*p).await
        }) {
            Ok(folders) => self.folders = folders,
            Err(e) => self.scan_status = format!("Failed to load folders: {e}"),
        }
    }

    fn refresh_media_blocking(&mut self) {
        if let Some(idx) = self.selected_folder {
            if let Some(folder) = self.folders.get(idx) {
                let folder_id = folder.id;
                let pool = Arc::clone(&self.pool);
                match self.rt.block_on(async move {
                    let p = pool.lock().await;
                    db::media::list_by_folder(&*p, folder_id).await
                }) {
                    Ok(items) => {
                        self.media_items = items;
                        // Clear pending for items no longer visible
                        self.pending_thumbnails.clear();
                    }
                    Err(e) => self.scan_status = format!("Failed to load media: {e}"),
                }
            }
        }
    }

    fn poll_scan_events(&mut self) {
        while let Ok(event) = self.scan_rx.try_recv() {
            match event {
                ScanEvent::Started(path) => {
                    self.scan_status = format!("Scanning: {path}...");
                    self.is_scanning = true;
                }
                ScanEvent::Progress(path, count) => {
                    self.scan_status = format!("Scanning: {path} ({count} files)...");
                }
                ScanEvent::Complete(path, count) => {
                    self.scan_status = format!("Done scanning {path}: {count} files");
                    self.is_scanning = false;
                    self.refresh_folders_blocking();
                    self.refresh_media_blocking();
                }
                ScanEvent::Error(path, err) => {
                    self.scan_status = format!("Error scanning {path}: {err}");
                    self.is_scanning = false;
                    self.push_toast(format!("Scan error in {path}: {err}"), ToastLevel::Error);
                }
            }
        }
    }

    fn poll_thumbnail_events(&mut self, ctx: &egui::Context) {
        while let Ok((hash, result)) = self.thumbnail_rx.try_recv() {
            self.pending_thumbnails.remove(&hash);
            match result {
                Ok(bytes) => {
                    if let Ok(image) = image::load_from_memory(&bytes) {
                        let size = [image.width() as usize, image.height() as usize];
                        let rgba = image.to_rgba8();
                        let color_image =
                            egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                        let texture = ctx.load_texture(
                            &hash,
                            color_image,
                            egui::TextureOptions::default(),
                        );
                        self.textures.insert(hash, texture);
                    }
                }
                Err(e) => {
                    tracing::warn!("Thumbnail failed for {}: {}", hash, e);
                    self.push_toast(format!("Thumbnail failed: {}", e), ToastLevel::Warning);
                }
            }
        }
    }

    fn load_missing_thumbnails(&mut self) {
        for media in &self.media_items {
            if self.textures.contains_key(&media.blake3_hash)
                || self.pending_thumbnails.contains(&media.blake3_hash)
            {
                continue;
            }

            self.pending_thumbnails.insert(media.blake3_hash.clone());

            let source = std::path::PathBuf::from(&media.absolute_path);
            let hash = media.blake3_hash.clone();
            let size = self.thumbnailer.size;
            let cache_mode = self.thumbnailer.cache_mode.clone();
            let tx = self.thumbnail_tx.clone();

            self.rt.spawn_blocking(move || {
                let thumbnailer = Thumbnailer::new(size, cache_mode);
                match thumbnailer.load_thumbnail_bytes(&source, &hash, None) {
                    Ok(bytes) => {
                        let _ = tx.send((hash, Ok(bytes)));
                    }
                    Err(e) => {
                        let _ = tx.send((hash, Err(e.to_string())));
                    }
                }
            });
        }
    }

    fn open_viewer(&mut self, index: usize) {
        self.viewer_open = true;
        self.viewer_index = Some(index);
        self.viewer_zoom_to_fit = true;
        self.viewer_texture = None;
        self.load_viewer_image();
    }

    fn close_viewer(&mut self) {
        self.viewer_open = false;
        self.viewer_index = None;
        self.viewer_texture = None;
    }

    fn navigate_viewer(&mut self, delta: isize) {
        if let Some(idx) = self.viewer_index {
            let len = self.media_items.len();
            if len == 0 {
                self.close_viewer();
                return;
            }
            let new_idx = if delta > 0 {
                (idx + delta as usize) % len
            } else {
                let d = (-delta) as usize;
                (idx + len - (d % len)) % len
            };
            self.viewer_index = Some(new_idx);
            self.viewer_texture = None;
            self.load_viewer_image();
        }
    }

    fn load_viewer_image(&mut self) {
        if let Some(idx) = self.viewer_index {
            if let Some(media) = self.media_items.get(idx) {
                let source = std::path::PathBuf::from(&media.absolute_path);
                let hash = media.blake3_hash.clone();
                let tx = self.viewer_image_tx.clone();

                self.rt.spawn_blocking(move || {
                    match image::open(&source) {
                        Ok(img) => {
                            let rgba = img.to_rgba8();
                            let size = [rgba.width() as usize, rgba.height() as usize];
                            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                                size,
                                rgba.as_raw(),
                            );
                            let _ = tx.send((hash, Ok(color_image)));
                        }
                        Err(e) => {
                            let _ = tx.send((hash, Err(e.to_string())));
                        }
                    }
                });
            }
        }
    }

    fn poll_viewer_images(&mut self, ctx: &egui::Context) {
        while let Ok((hash, result)) = self.viewer_image_rx.try_recv() {
            match result {
                Ok(color_image) => {
                    let texture = ctx.load_texture(
                        &format!("viewer-{}", hash),
                        color_image,
                        egui::TextureOptions::default(),
                    );
                    self.viewer_texture = Some(texture);
                }
                Err(e) => {
                    tracing::warn!("Viewer image failed for {}: {}", hash, e);
                    self.push_toast(format!("Failed to load image: {}", e), ToastLevel::Error);
                }
            }
        }
    }

    fn push_toast(&mut self, message: String, level: ToastLevel) {
        self.toasts.push(Toast {
            message,
            level,
            created_at: std::time::Instant::now(),
        });
    }

    fn prune_toasts(&mut self) {
        let now = std::time::Instant::now();
        self.toasts.retain(|t| now.duration_since(t.created_at).as_secs() < 5);
    }

    fn show_toasts(&mut self, ctx: &egui::Context) {
        self.prune_toasts();
        if self.toasts.is_empty() {
            return;
        }

        let screen = ctx.screen_rect();
        let margin = 16.0;
        let toast_width = 320.0;
        let toast_height = 48.0;
        let spacing = 8.0;

        let mut y = screen.max.y - margin;

        for toast in self.toasts.iter().rev() {
            y -= toast_height;
            let rect = egui::Rect::from_min_size(
                egui::pos2(screen.max.x - margin - toast_width, y),
                egui::vec2(toast_width, toast_height - spacing),
            );

            let (bg, fg) = match toast.level {
                ToastLevel::Info => (egui::Color32::from_rgb(40, 80, 120), egui::Color32::WHITE),
                ToastLevel::Warning => (egui::Color32::from_rgb(140, 110, 40), egui::Color32::WHITE),
                ToastLevel::Error => (egui::Color32::from_rgb(140, 50, 50), egui::Color32::WHITE),
            };

            egui::Area::new(egui::Id::new(("toast", toast.created_at)))
                .order(egui::Order::Foreground)
                .fixed_pos(rect.min)
                .show(ctx, |ui| {
                    let frame = egui::Frame::none()
                        .fill(bg)
                        .rounding(8.0)
                        .inner_margin(12.0);
                    frame.show(ui, |ui| {
                        ui.set_min_size(rect.size());
                        ui.colored_label(fg, &toast.message);
                    });
                });

            y -= spacing;
        }
    }

    fn show_settings(&mut self, ctx: &egui::Context) {
        let mut open = self.settings_open;
        egui::Window::new("Settings")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Appearance");
                ui.separator();

                let mut dark = self.config.ui.theme == "dark";
                if ui.checkbox(&mut dark, "Dark theme").changed() {
                    self.config.ui.theme = if dark { "dark".to_string() } else { "light".to_string() };
                    crate::theme::apply(ctx, dark);
                    if let Err(e) = self.config.save() {
                        self.push_toast(format!("Failed to save config: {}", e), ToastLevel::Error);
                    }
                }

                ui.add_space(16.0);
                ui.heading("Thumbnails");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Size:");
                    let mut size = self.config.ui.thumbnail_size as f32;
                    if ui.add(egui::Slider::new(&mut size, 64.0..=512.0).step_by(16.0)).changed() {
                        self.config.ui.thumbnail_size = size as u32;
                        self.thumbnailer.size = size as u32;
                        self.textures.clear();
                        self.pending_thumbnails.clear();
                        if let Err(e) = self.config.save() {
                            self.push_toast(format!("Failed to save config: {}", e), ToastLevel::Error);
                        }
                    }
                });

                ui.add_space(16.0);
                ui.heading("Folders");
                ui.separator();
                ui.label("Edit ~/.config/akasha/config.toml to add or remove folders.");
                ui.label("Changes require a restart to take full effect.");

                ui.add_space(8.0);
                for folder in &self.config.folders {
                    ui.label(format!("• {}", folder.path));
                }
            });
        self.settings_open = open;
    }

    fn trigger_rescan(&mut self) {
        if self.is_scanning {
            return;
        }
        self.is_scanning = true;
        self.scan_status = "Rescanning...".to_string();

        let pool_clone = Arc::clone(&self.pool);
        let folders_config: Vec<FolderConfig> = self.config.folders.clone();
        let (scan_tx, scan_rx) = std::sync::mpsc::channel();
        self.scan_rx = scan_rx;

        let rt = Arc::clone(&self.rt);
        rt.spawn(async move {
            let pool = pool_clone.lock().await;
            for folder_cfg in &folders_config {
                let _ = scan_tx.send(ScanEvent::Started(folder_cfg.path.clone()));
                let path = std::path::Path::new(&folder_cfg.path);
                let cache_mode = folder_cfg.thumbnail_cache_mode.as_deref();

                match db::folder::insert_or_get(
                    &*pool,
                    &folder_cfg.path,
                    folder_cfg.recursive,
                    &folder_cfg.blacklist,
                    cache_mode,
                )
                .await
                {
                    Ok(folder_id) => {
                        match crate::scanner::scan_folder(
                            &*pool,
                            folder_id,
                            path,
                            folder_cfg.recursive,
                            &folder_cfg.blacklist,
                        )
                        .await
                        {
                            Ok(count) => {
                                let _ = scan_tx.send(ScanEvent::Complete(
                                    folder_cfg.path.clone(),
                                    count,
                                ));
                            }
                            Err(e) => {
                                let _ = scan_tx.send(ScanEvent::Error(
                                    folder_cfg.path.clone(),
                                    e.to_string(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        let _ = scan_tx.send(ScanEvent::Error(
                            folder_cfg.path.clone(),
                            e.to_string(),
                        ));
                    }
                }
            }
        });
    }
}

impl eframe::App for AkashaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_scan_events();
        self.poll_thumbnail_events(ctx);
        self.poll_viewer_images(ctx);
        self.load_missing_thumbnails();

        if self.viewer_open {
            if let Some(idx) = self.viewer_index {
                if let Some(media) = self.media_items.get(idx).cloned() {
                    let resp = crate::ui::viewer::show(
                        ctx,
                        &media,
                        &self.viewer_texture,
                        self.viewer_zoom_to_fit,
                    );
                    if resp.close {
                        self.close_viewer();
                    }
                    if resp.prev {
                        self.navigate_viewer(-1);
                    }
                    if resp.next {
                        self.navigate_viewer(1);
                    }
                    if resp.toggle_zoom {
                        self.viewer_zoom_to_fit = !self.viewer_zoom_to_fit;
                    }
                } else {
                    self.close_viewer();
                }
            } else {
                self.close_viewer();
            }

            ctx.request_repaint_after(std::time::Duration::from_millis(16));
            return;
        }

        // Top bar
        egui::TopBottomPanel::top("top_bar")
            .frame(egui::Frame::new()
                .fill(ctx.style().visuals.panel_fill)
                .inner_margin(egui::Margin::same(12)))
            .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Akasha");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⚙ Settings").clicked() {
                        self.settings_open = !self.settings_open;
                    }
                    if ui.button("⟳ Rescan").clicked() && !self.is_scanning {
                        self.trigger_rescan();
                    }
                    ui.label(&self.scan_status);
                });
            });
        });

        // Left sidebar: folders
        egui::SidePanel::left("folders")
            .resizable(true)
            .default_width(250.0)
            .frame(egui::Frame::new()
                .fill(ctx.style().visuals.panel_fill)
                .inner_margin(egui::Margin::same(12)))
            .show(ctx, |ui| {
                ui.heading("Folders");
                ui.separator();

                if self.folders.is_empty() {
                    ui.label("No folders configured.");
                    ui.label("Add folders in config.toml");
                } else {
                    let mut clicked_idx = None;
                    for (idx, folder) in self.folders.iter().enumerate() {
                        let selected = self.selected_folder == Some(idx);
                        let response = ui.selectable_label(
                            selected,
                            format!("{}", std::path::Path::new(&folder.path).file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or(&folder.path)),
                        );
                        if response.clicked() && !selected {
                            clicked_idx = Some(idx);
                        }
                        response.on_hover_text(&folder.path);
                    }
                    if let Some(idx) = clicked_idx {
                        self.selected_folder = Some(idx);
                        self.refresh_media_blocking();
                    }
                }
            });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.is_scanning && self.media_items.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.heading("Scanning...");
                    ui.label(&self.scan_status);
                });
            } else if self.media_items.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.heading("No images found");
                    ui.label("Add a folder in config.toml and restart, or click Rescan.");
                });
            } else {
                ui.heading(format!("{} images", self.media_items.len()));
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let cols = (ui.available_width() / 220.0).max(1.0) as usize;
                    egui::Grid::new("media_grid")
                        .num_columns(cols)
                        .spacing([16.0, 16.0])
                        .show(ui, |ui| {
                            let mut clicked_index = None;
                            for (i, media) in self.media_items.iter().enumerate() {
                                if i > 0 && i % cols == 0 {
                                    ui.end_row();
                                }
                                let clicked = ui.vertical(|ui| {
                                    let thumb_size = egui::vec2(200.0, 200.0);
                                    let response = if let Some(texture) = self.textures.get(&media.blake3_hash) {
                                        let mut size = thumb_size;
                                        let tex_w = texture.size()[0] as f32;
                                        let tex_h = texture.size()[1] as f32;
                                        if tex_w > 0.0 && tex_h > 0.0 {
                                            let aspect = tex_w / tex_h;
                                            if aspect > 1.0 {
                                                size.y = size.x / aspect;
                                            } else {
                                                size.x = size.y * aspect;
                                            }
                                        }
                                        ui.add(
                                            egui::Image::new((texture.id(), size))
                                                .fit_to_exact_size(size)
                                                .sense(egui::Sense::click()),
                                        )
                                    } else {
                                        ui.add_sized(thumb_size, egui::Spinner::new())
                                    };
                                    ui.label(
                                        std::path::Path::new(&media.relative_path)
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or(&media.relative_path),
                                    );
                                    response.clicked()
                                }).inner;
                                if clicked {
                                    clicked_index = Some(i);
                                }
                            }
                            if let Some(i) = clicked_index {
                                self.open_viewer(i);
                            }
                        });
                });
            }
        });

        if self.settings_open {
            self.show_settings(ctx);
        }
        self.show_toasts(ctx);

        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}
