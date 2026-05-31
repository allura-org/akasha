use eframe::egui;
use sqlx::SqlitePool;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use tokio::runtime::Runtime;

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
    pub pool: Arc<SqlitePool>,
    pub rt: Arc<Runtime>,
    pub thumbnailer: Thumbnailer,

    // UI state
    pub folders: Vec<db::folder::Folder>,
    pub selected_folder_id: Option<i64>,
    pub media_items: Vec<db::media::MediaFile>,
    pub textures: HashMap<String, egui::TextureHandle>,
    pub pending_thumbnails: HashSet<String>,
    pub thumbnail_queue: VecDeque<(usize, String)>,
    pub queued_indices: HashSet<usize>,
    pub scroll_offset: f32,
    pub thumbnail_epoch: u64,
    pub scan_status: String,
    pub is_scanning: bool,
    pub scan_rx: std::sync::mpsc::Receiver<ScanEvent>,
    pub thumbnail_tx: std::sync::mpsc::Sender<(String, u64, Result<egui::ColorImage, String>)>,
    pub thumbnail_rx: std::sync::mpsc::Receiver<(String, u64, Result<egui::ColorImage, String>)>,
    pub media_tx: std::sync::mpsc::Sender<Result<Vec<db::media::MediaFile>, String>>,
    pub media_rx: std::sync::mpsc::Receiver<Result<Vec<db::media::MediaFile>, String>>,
    pub folders_tx: std::sync::mpsc::Sender<Result<Vec<db::folder::Folder>, String>>,
    pub folders_rx: std::sync::mpsc::Receiver<Result<Vec<db::folder::Folder>, String>>,
    pub last_refresh: std::time::Instant,
    pub expanded_folders: HashSet<i64>,

    // Viewer state
    pub viewer_open: bool,
    pub viewer_index: Option<usize>,
    pub viewer_texture: Option<egui::TextureHandle>,
    pub viewer_zoom_to_fit: bool,
    pub viewer_image_tx: std::sync::mpsc::Sender<(String, Result<egui::ColorImage, String>)>,
    pub viewer_image_rx: std::sync::mpsc::Receiver<(String, Result<egui::ColorImage, String>)>,
    pub viewer_just_opened: bool,

    // Polish
    pub toasts: Vec<Toast>,
    pub settings_open: bool,
}

const MAX_CONCURRENT_THUMBNAILS: usize = 8;
const THUMB_CELL_HEIGHT: f32 = 230.0;

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

        let pool_arc = Arc::new(pool);
        let rt_arc = Arc::new(rt);
        let (scan_tx, scan_rx) = std::sync::mpsc::channel();
        let (thumb_tx, thumbnail_rx) = std::sync::mpsc::channel::<(String, u64, Result<egui::ColorImage, String>)>();
        let (viewer_img_tx, viewer_img_rx) = std::sync::mpsc::channel::<(String, Result<egui::ColorImage, String>)>();
        let (media_tx, media_rx) = std::sync::mpsc::channel::<Result<Vec<db::media::MediaFile>, String>>();
        let (folders_tx, folders_rx) = std::sync::mpsc::channel::<Result<Vec<db::folder::Folder>, String>>();

        // On startup: scan any configured folders that are new or incomplete
        let pool_clone = Arc::clone(&pool_arc);
        let folders_config: Vec<FolderConfig> = config.folders.clone();

        let incomplete_folders: Vec<FolderConfig> = rt_arc.block_on(async {
            let mut incomplete = Vec::new();
            for folder_cfg in &folders_config {
                match db::folder::get_by_path(&pool_clone, &folder_cfg.path).await {
                    Ok(Some(folder)) if folder.scan_complete => {
                        // Already fully scanned, skip
                    }
                    _ => {
                        // New or incomplete — needs scanning
                        incomplete.push(folder_cfg.clone());
                    }
                }
            }
            incomplete
        });

        if incomplete_folders.is_empty() {
            let _ = scan_tx.send(ScanEvent::Complete("Existing data loaded".to_string(), 0));
        } else {
            let rt_clone = Arc::clone(&rt_arc);
            let pool_clone = Arc::clone(&pool_arc);
            rt_clone.spawn(async move {
                for folder_cfg in &incomplete_folders {
                    let _ = scan_tx.send(ScanEvent::Started(folder_cfg.path.clone()));

                    let cache_mode = folder_cfg.thumbnail_cache_mode.as_deref();
                    match db::folder::get_or_create(
                        &pool_clone,
                        None,
                        &folder_cfg.path,
                        folder_cfg.recursive,
                        folder_cfg.show_recursive,
                        false,
                        &folder_cfg.blacklist,
                        cache_mode,
                    )
                    .await
                    {
                        Ok(folder_id) => {
                            let path = std::path::Path::new(&folder_cfg.path);
                            match crate::scanner::scan_folder(
                                &pool_clone,
                                folder_id,
                                path,
                                folder_cfg.recursive,
                                folder_cfg.show_recursive,
                                &folder_cfg.blacklist,
                                Some(&scan_tx),
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

        let mut app = Self {
            config,
            pool: pool_arc,
            rt: rt_arc,
            thumbnailer,
            folders: Vec::new(),
            selected_folder_id: None,
            media_items: Vec::new(),
            textures: HashMap::new(),
            pending_thumbnails: HashSet::new(),
            thumbnail_queue: VecDeque::new(),
            queued_indices: HashSet::new(),
            scroll_offset: 0.0,
            thumbnail_epoch: 0,
            scan_status: "Initializing...".to_string(),
            is_scanning: true,
            scan_rx,
            thumbnail_tx: thumb_tx,
            thumbnail_rx,
            media_tx,
            media_rx,
            folders_tx,
            folders_rx,
            last_refresh: std::time::Instant::now(),
            expanded_folders: HashSet::new(),
            viewer_open: false,
            viewer_index: None,
            viewer_texture: None,
            viewer_zoom_to_fit: true,
            viewer_image_tx: viewer_img_tx,
            viewer_image_rx: viewer_img_rx,
            viewer_just_opened: false,
            toasts: Vec::new(),
            settings_open: false,
        };

        app.refresh_folders_async();
        app
    }

    fn folder_depth(&self, folder: &db::folder::Folder) -> usize {
        let mut depth = 0;
        let mut current = folder.parent_id;
        while let Some(pid) = current {
            depth += 1;
            current = self.folders.iter().find(|f| f.id == pid).and_then(|f| f.parent_id);
        }
        depth
    }

    fn refresh_folders_async(&mut self) {
        let pool = Arc::clone(&self.pool);
        let tx = self.folders_tx.clone();
        self.rt.spawn(async move {
            let result = db::folder::list_all(&pool).await;
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
    }

    fn poll_folders_events(&mut self) {
        if let Ok(result) = self.folders_rx.try_recv() {
            match result {
                Ok(folders) => {
                    // Auto-expand root folders
                    for folder in &folders {
                        if folder.parent_id.is_none() {
                            self.expanded_folders.insert(folder.id);
                        }
                    }
                    self.folders = folders;
                }
                Err(e) => self.scan_status = format!("Failed to load folders: {e}"),
            }
        }
    }

    fn render_folder_tree(&mut self, ui: &mut egui::Ui, folder_id: i64, depth: usize, clicked_id: &mut Option<i64>) {
        let Some(folder) = self.folders.iter().find(|f| f.id == folder_id) else { return; };
        let selected = self.selected_folder_id == Some(folder.id);
        let has_children = self.folders.iter().any(|f| f.parent_id == Some(folder.id));
        let is_root = folder.parent_id.is_none();

        ui.horizontal(|ui| {
            ui.add_space(depth as f32 * 16.0);

            if has_children {
                let arrow = if self.expanded_folders.contains(&folder.id) { "▼" } else { "▶" };
                if ui.small_button(arrow).clicked() {
                    if self.expanded_folders.contains(&folder.id) {
                        self.expanded_folders.remove(&folder.id);
                    } else {
                        self.expanded_folders.insert(folder.id);
                    }
                }
            } else {
                ui.add_space(24.0);
            }

            let name = std::path::Path::new(&folder.path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&folder.path);

            let label = if is_root {
                egui::RichText::new(name).strong()
            } else {
                egui::RichText::new(name)
            };

            if ui.selectable_label(selected, label).clicked() && !selected {
                *clicked_id = Some(folder.id);
            }
        });

        if self.expanded_folders.contains(&folder.id) {
            let children: Vec<i64> = self.folders
                .iter()
                .filter(|f| f.parent_id == Some(folder.id))
                .map(|f| f.id)
                .collect();
            for child_id in children {
                self.render_folder_tree(ui, child_id, depth + 1, clicked_id);
            }
        }
    }

    fn refresh_media_async(&mut self, hard_reset: bool) {
        let Some(folder_id) = self.selected_folder_id else { return; };
        let Some(folder) = self.folders.iter().find(|f| f.id == folder_id) else { return; };

        let pool = Arc::clone(&self.pool);
        let tx = self.media_tx.clone();
        let show_recursive = folder.show_recursive;

        if hard_reset {
            self.scan_status = "Loading images...".to_string();
            self.media_items.clear();
            self.textures.clear();
            self.pending_thumbnails.clear();
            self.thumbnail_queue.clear();
            self.queued_indices.clear();
            self.scroll_offset = 0.0;
        }

        self.rt.spawn(async move {
            let result = if show_recursive {
                db::media::list_by_folder_recursive(&pool, folder_id).await
            } else {
                db::media::list_by_folder(&pool, folder_id).await
            };
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
    }

    fn poll_media_events(&mut self) {
        if let Ok(result) = self.media_rx.try_recv() {
            match result {
                Ok(items) => {
                    // Prune textures for hashes no longer in the current folder
                    let needed: HashSet<String> = items.iter().map(|m| m.blake3_hash.clone()).collect();
                    self.textures.retain(|hash, _| needed.contains(hash));
                    self.pending_thumbnails.retain(|hash| needed.contains(hash));
                    self.thumbnail_queue.retain(|(idx, _)| items.get(*idx).is_some());
                    self.queued_indices.retain(|idx| items.get(*idx).is_some());
                    self.media_items = items;
                    self.scan_status = format!("{} images", self.media_items.len());
                }
                Err(e) => {
                    self.scan_status = format!("Failed to load media: {e}");
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
                    self.refresh_folders_async();
                    if self.selected_folder_id.is_none() {
                        // Select first root folder if nothing selected
                        if let Some(root) = self.folders.iter().find(|f| f.parent_id.is_none()) {
                            self.selected_folder_id = Some(root.id);
                            self.refresh_media_async(true);
                        }
                    } else {
                        self.refresh_media_async(false);
                    }
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
        while let Ok((hash, epoch, result)) = self.thumbnail_rx.try_recv() {
            self.pending_thumbnails.remove(&hash);
            if epoch != self.thumbnail_epoch {
                continue; // stale result from before a size change
            }
            match result {
                Ok(color_image) => {
                    let texture = ctx.load_texture(
                        &hash,
                        color_image,
                        egui::TextureOptions::default(),
                    );
                    self.textures.insert(hash, texture);
                }
                Err(e) => {
                    tracing::warn!("Thumbnail failed for {}: {}", hash, e);
                    self.push_toast(format!("Thumbnail failed: {}", e), ToastLevel::Warning);
                }
            }
        }
    }

    fn queue_visible_thumbnails(&mut self, viewport_height: f32, cols: usize) {
        if cols == 0 || self.media_items.is_empty() {
            return;
        }
        let first_visible_row = (self.scroll_offset / THUMB_CELL_HEIGHT).floor() as usize;
        let last_visible_row = ((self.scroll_offset + viewport_height) / THUMB_CELL_HEIGHT).ceil() as usize;
        // Prefetch 5 rows above and below the viewport (small fixed window)
        let prefetch_rows = 5;
        let start_idx = first_visible_row.saturating_sub(prefetch_rows) * cols;
        let end_idx = ((last_visible_row + prefetch_rows) * cols).min(self.media_items.len());

        let mut to_queue = Vec::new();
        for i in start_idx..end_idx {
            let hash = &self.media_items[i].blake3_hash;
            if !self.textures.contains_key(hash)
                && !self.pending_thumbnails.contains(hash)
                && !self.queued_indices.contains(&i)
            {
                let row = i / cols;
                let dist = if row < first_visible_row {
                    first_visible_row - row
                } else if row > last_visible_row {
                    row - last_visible_row
                } else {
                    0
                };
                to_queue.push((i, hash.clone(), dist));
            }
        }

        to_queue.sort_by_key(|(_, _, dist)| *dist);
        for (idx, hash, _) in to_queue {
            self.queued_indices.insert(idx);
            self.thumbnail_queue.push_back((idx, hash));
        }
    }

    fn process_thumbnail_queue(&mut self) {
        let can_spawn = MAX_CONCURRENT_THUMBNAILS.saturating_sub(self.pending_thumbnails.len());
        if can_spawn == 0 {
            return;
        }

        for _ in 0..can_spawn {
            let Some((idx, hash)) = self.thumbnail_queue.pop_front() else {
                break;
            };
            self.queued_indices.remove(&idx);
            // Double-check it's still needed
            if self.textures.contains_key(&hash) || self.pending_thumbnails.contains(&hash) {
                continue;
            }

            let Some(media) = self.media_items.get(idx) else {
                continue;
            };

            self.pending_thumbnails.insert(hash.clone());
            let source = std::path::PathBuf::from(&media.absolute_path);
            let size = self.thumbnailer.size;
            let cache_mode = self.thumbnailer.cache_mode.clone();
            let tx = self.thumbnail_tx.clone();
            let epoch = self.thumbnail_epoch;

            self.rt.spawn_blocking(move || {
                let thumbnailer = Thumbnailer::new(size, cache_mode);
                let result = thumbnailer.load_thumbnail_bytes(&source, &hash, None)
                    .and_then(|bytes| {
                        let img = image::load_from_memory(&bytes)
                            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
                        let rgba = img.to_rgba8();
                        let sz = [rgba.width() as usize, rgba.height() as usize];
                        Ok(egui::ColorImage::from_rgba_unmultiplied(sz, rgba.as_raw()))
                    });
                let _ = tx.send((hash, epoch, result.map_err(|e| e.to_string())));
            });
        }
    }

    fn open_viewer(&mut self, index: usize) {
        self.viewer_open = true;
        self.viewer_just_opened = true;
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
                    match crate::image_loader::open_image(&source) {
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
                        self.thumbnail_queue.clear();
                        self.queued_indices.clear();
                        self.thumbnail_epoch += 1;
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
            for folder_cfg in &folders_config {
                let _ = scan_tx.send(ScanEvent::Started(folder_cfg.path.clone()));
                let path = std::path::Path::new(&folder_cfg.path);
                let cache_mode = folder_cfg.thumbnail_cache_mode.as_deref();

                match db::folder::get_or_create(
                    &pool_clone,
                    None,
                    &folder_cfg.path,
                    folder_cfg.recursive,
                    folder_cfg.show_recursive,
                    false,
                    &folder_cfg.blacklist,
                    cache_mode,
                )
                .await
                {
                    Ok(folder_id) => {
                        // Mark entire tree as incomplete before scanning so interruption is recoverable
                        let _ = db::folder::update_scan_complete_recursive(&pool_clone, folder_id, false).await;
                        match crate::scanner::scan_folder(
                            &pool_clone,
                            folder_id,
                            path,
                            folder_cfg.recursive,
                            folder_cfg.show_recursive,
                            &folder_cfg.blacklist,
                            Some(&scan_tx),
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
        self.poll_folders_events();
        self.poll_media_events();
        self.poll_thumbnail_events(ctx);
        self.poll_viewer_images(ctx);
        self.process_thumbnail_queue();

        // Periodic refresh while scanning so the tree populates and counts update live
        if self.is_scanning && self.last_refresh.elapsed() > std::time::Duration::from_secs(2) {
            self.last_refresh = std::time::Instant::now();
            self.refresh_folders_async();
            if self.selected_folder_id.is_some() {
                self.refresh_media_async(false);
            }
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

        // Left sidebar: folder tree
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
                    let mut clicked_id = None;
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let roots: Vec<i64> = self.folders
                            .iter()
                            .filter(|f| f.parent_id.is_none())
                            .map(|f| f.id)
                            .collect();
                        for root_id in roots {
                            self.render_folder_tree(ui, root_id, 0, &mut clicked_id);
                        }
                    });
                    if let Some(id) = clicked_id {
                        self.selected_folder_id = Some(id);
                        self.refresh_media_async(true);
                    }
                }
            });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.is_scanning && self.media_items.is_empty() && self.selected_folder_id.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.heading("Scanning...");
                    ui.label(&self.scan_status);
                });
            } else if self.media_items.is_empty() {
                ui.centered_and_justified(|ui| {
                    if self.selected_folder_id.is_some() {
                        ui.heading("No images in this folder");
                    } else {
                        ui.heading("Select a folder");
                    }
                });
            } else {
                ui.heading(format!("{} images", self.media_items.len()));
                ui.separator();

                let cols = (ui.available_width() / 220.0).max(1.0) as usize;
                let rows = (self.media_items.len() + cols - 1) / cols;
                let item_size = egui::vec2(200.0, 200.0);
                let label_h = 30.0;
                let row_height = item_size.y + label_h;

                let scroll = egui::ScrollArea::vertical()
                    .show_rows(ui, row_height, rows, |ui, row_range| {
                        let mut clicked_index = None;
                        for row in row_range {
                            ui.horizontal(|ui| {
                                for col in 0..cols {
                                    let idx = row * cols + col;
                                    if idx >= self.media_items.len() {
                                        break;
                                    }
                                    let media = &self.media_items[idx];

                                    let clicked = ui.allocate_ui_with_layout(
                                        egui::vec2(item_size.x, row_height),
                                        egui::Layout::top_down(egui::Align::Center),
                                        |ui| {
                                            let response = if let Some(texture) = self.textures.get(&media.blake3_hash) {
                                                let mut size = item_size;
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
                                                ui.add_sized(item_size, egui::Spinner::new())
                                            };
                                            ui.label(
                                                std::path::Path::new(&media.relative_path)
                                                    .file_name()
                                                    .and_then(|n| n.to_str())
                                                    .unwrap_or(&media.relative_path),
                                            );
                                            response.clicked()
                                        },
                                    ).inner;
                                    if clicked {
                                        clicked_index = Some(idx);
                                    }
                                }
                            });
                        }
                        if let Some(i) = clicked_index {
                            self.open_viewer(i);
                        }
                    });
                self.scroll_offset = scroll.state.offset.y;
                let viewport_h = scroll.inner_rect.height();
                self.queue_visible_thumbnails(viewport_h, cols);
            }
        });

        // Viewer overlay (drawn on top of browser)
        if self.viewer_open {
            if let Some(idx) = self.viewer_index {
                if let Some(media) = self.media_items.get(idx).cloned() {
                    let resp = crate::ui::viewer::show(
                        ctx,
                        &media,
                        &self.viewer_texture,
                        self.viewer_zoom_to_fit,
                    );
                    if resp.close && !self.viewer_just_opened {
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
            self.viewer_just_opened = false;
        }

        if self.settings_open {
            self.show_settings(ctx);
        }
        self.show_toasts(ctx);

        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}
