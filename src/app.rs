use eframe::egui;
use sqlx::SqlitePool;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::config::{Config, FolderConfig};
use crate::db;
use crate::searchables::{SearchEngine, SearchQuery};
use crate::thumbnailer::{CacheMode, Thumbnailer};
use crate::ui::browser::BrowserPanel;

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
    pub browser: BrowserPanel,
    pub search_engine: SearchEngine,

    pub scan_rx: std::sync::mpsc::Receiver<ScanEvent>,
    pub thumbnail_rx: std::sync::mpsc::Receiver<(String, u64, Result<egui::ColorImage, String>)>,
    pub media_tx: std::sync::mpsc::Sender<(u64, bool, Result<Vec<db::media::MediaSummary>, String>)>,
    pub media_rx: std::sync::mpsc::Receiver<(u64, bool, Result<Vec<db::media::MediaSummary>, String>)>,
    pub folders_tx: std::sync::mpsc::Sender<Result<Vec<db::folder::Folder>, String>>,
    pub folders_rx: std::sync::mpsc::Receiver<Result<Vec<db::folder::Folder>, String>>,
    pub search_names_tx: std::sync::mpsc::Sender<Result<Vec<String>, String>>,
    pub search_names_rx: std::sync::mpsc::Receiver<Result<Vec<String>, String>>,
    pub viewer_image_tx: std::sync::mpsc::Sender<(String, Result<egui::ColorImage, String>)>,
    pub viewer_image_rx: std::sync::mpsc::Receiver<(String, Result<egui::ColorImage, String>)>,

    pub media_refresh_in_flight: bool,
    pub last_refresh: std::time::Instant,

    pub viewer_open: bool,
    pub viewer_index: Option<usize>,
    pub viewer_texture: Option<egui::TextureHandle>,
    pub viewer_scale_mode: crate::config::ViewerScaleMode,
    pub viewer_just_opened: bool,
    pub pending_viewer_images: HashSet<String>,

    pub toasts: Vec<Toast>,
}

impl AkashaApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        config: Config,
        pool: SqlitePool,
        rt: Runtime,
    ) -> Self {
        crate::theme::apply(&cc.egui_ctx, config.ui.theme == "dark");
        Self::apply_input_options(&cc.egui_ctx, &config);

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
        let (media_tx, media_rx) = std::sync::mpsc::channel::<(u64, bool, Result<Vec<db::media::MediaSummary>, String>)>();
        let (folders_tx, folders_rx) = std::sync::mpsc::channel::<Result<Vec<db::folder::Folder>, String>>();
        let (search_names_tx, search_names_rx) = std::sync::mpsc::channel::<Result<Vec<String>, String>>();

        // On startup: scan any configured folders that are new or incomplete
        let pool_clone = Arc::clone(&pool_arc);
        let folders_config: Vec<FolderConfig> = config.folders.clone();

        let incomplete_folders: Vec<FolderConfig> = rt_arc.block_on(async {
            let mut incomplete = Vec::new();
            for folder_cfg in &folders_config {
                match db::folder::get_by_path(&pool_clone, &folder_cfg.path).await {
                    Ok(Some(folder)) if folder.scan_complete => {}
                    _ => {
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

        let scroll_speed = config.ui.scroll_speed;
        let sort_key = config.ui.sort_key;
        let sort_order = config.ui.sort_order;
        let mut app = Self {
            config,
            pool: pool_arc.clone(),
            rt: rt_arc.clone(),
            thumbnailer,
            browser: BrowserPanel::new(rt_arc, thumb_tx, scroll_speed, sort_key, sort_order),
            search_engine: SearchEngine::with_defaults(),
            scan_rx,
            thumbnail_rx,
            media_tx,
            media_rx,
            folders_tx,
            folders_rx,
            search_names_tx,
            search_names_rx,
            viewer_image_tx: viewer_img_tx,
            viewer_image_rx: viewer_img_rx,
            media_refresh_in_flight: false,
            last_refresh: std::time::Instant::now(),
            viewer_open: false,
            viewer_index: None,
            viewer_texture: None,
            viewer_scale_mode: crate::config::ViewerScaleMode::Fit,
            viewer_just_opened: false,
            pending_viewer_images: HashSet::new(),
            toasts: Vec::new(),
        };

        app.refresh_folders_async();
        app.refresh_searchable_names_async();

        // Start the background Searchables worker stub.
        let worker_pool = Arc::clone(&app.pool);
        app.rt.spawn(async move {
            crate::searchables::SearchWorker::new(worker_pool).run().await;
        });

        app
    }

    fn apply_input_options(ctx: &egui::Context, config: &Config) {
        let delay_secs = config.ui.double_click_debounce_ms as f64 / 1000.0;
        ctx.memory_mut(|mem| {
            mem.options.input_options.max_double_click_delay = delay_secs;
        });
    }

    fn refresh_folders_async(&mut self) {
        let pool = Arc::clone(&self.pool);
        let tx = self.folders_tx.clone();
        self.rt.spawn(async move {
            let result = db::folder::list_all(&pool).await;
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
    }

    fn refresh_searchable_names_async(&mut self) {
        let pool = Arc::clone(&self.pool);
        let tx = self.search_names_tx.clone();
        self.rt.spawn(async move {
            let result = db::searchable::list_enabled_configs(&pool).await;
            let _ = tx.send(result.map_err(|e| e.to_string()).map(|configs| {
                configs.into_iter().map(|c| c.name).collect()
            }));
        });
    }

    fn poll_folders_events(&mut self) {
        if let Ok(result) = self.folders_rx.try_recv() {
            match result {
                Ok(folders) => {
                    for folder in &folders {
                        if folder.parent_id.is_none() {
                            self.browser.expanded_folders.insert(folder.id);
                        }
                    }
                    self.browser.folders = folders;
                }
                Err(e) => self.browser.scan_status = format!("Failed to load folders: {e}"),
            }
        }
    }

    fn poll_search_names_events(&mut self) {
        if let Ok(result) = self.search_names_rx.try_recv() {
            match result {
                Ok(names) => {
                    self.browser.search_available_names = names.clone();
                    self.browser.search_enabled_names = names.into_iter().collect();
                }
                Err(e) => tracing::warn!("Failed to load Searchable names: {e}"),
            }
        }
    }

    fn refresh_search_async(&mut self, query: SearchQuery) {
        let Some(folder_id) = self.browser.selected_folder_id else { return };
        let Some(folder) = self.browser.folders.iter().find(|f| f.id == folder_id) else { return };

        if self.media_refresh_in_flight {
            return;
        }

        let pool = Arc::clone(&self.pool);
        let tx = self.media_tx.clone();
        let show_recursive = folder.show_recursive;
        let engine = self.search_engine.clone();

        self.browser.clear_for_refresh(true);
        self.media_refresh_in_flight = true;
        let epoch = self.browser.media_epoch;

        self.rt.spawn(async move {
            let result = engine
                .execute(&pool, folder_id, show_recursive, &query)
                .await;
            let summaries: Result<Vec<db::media::MediaSummary>, String> = result.map(|hits| {
                hits.into_iter()
                    .map(|hit| {
                        let mut summary = hit.media_summary;
                        summary.search_score = Some(hit.score);
                        summary
                    })
                    .collect()
            }).map_err(|e| e.to_string());
            let _ = tx.send((epoch, true, summaries));
        });
    }

    fn refresh_media_async(&mut self, hard_reset: bool) {
        let Some(folder_id) = self.browser.selected_folder_id else { return };
        let Some(folder) = self.browser.folders.iter().find(|f| f.id == folder_id) else { return };

        if self.media_refresh_in_flight && !hard_reset {
            return;
        }

        let pool = Arc::clone(&self.pool);
        let tx = self.media_tx.clone();
        let show_recursive = folder.show_recursive;

        if hard_reset {
            self.browser.clear_for_refresh(true);
        }

        self.media_refresh_in_flight = true;
        let epoch = self.browser.media_epoch;
        self.rt.spawn(async move {
            let result = if show_recursive {
                db::media::list_summaries_by_folder_recursive(&pool, folder_id).await
            } else {
                db::media::list_summaries_by_folder(&pool, folder_id).await
            };
            let _ = tx.send((epoch, false, result.map_err(|e| e.to_string())));
        });
    }

    fn poll_media_events(&mut self) {
        let mut latest: Option<(bool, Result<Vec<db::media::MediaSummary>, String>)> = None;
        while let Ok((epoch, is_search, result)) = self.media_rx.try_recv() {
            if epoch == self.browser.media_epoch {
                latest = Some((is_search, result));
            }
        }
        if let Some((is_search, result)) = latest {
            self.media_refresh_in_flight = false;
            match result {
                Ok(items) => {
                    let needed: HashSet<String> = items.iter().map(|m| m.blake3_hash.clone()).collect();
                    self.browser.textures.retain(|hash, _| needed.contains(hash));
                    self.browser.pending_thumbnails.retain(|hash| needed.contains(hash));
                    self.browser.thumbnail_queue.clear();
                    self.browser.queued_indices.clear();
                    self.browser.media_summaries = items;
                    if is_search {
                        self.browser.search_active = true;
                        self.browser.sort_key = crate::config::SortKey::Score;
                        self.browser.sort_order = crate::config::SortOrder::Descending;
                        self.browser.scan_status = format!("{} results", self.browser.media_summaries.len());
                    } else {
                        self.browser.search_active = false;
                        self.browser.scan_status = format!("{} images", self.browser.media_summaries.len());
                    }
                }
                Err(e) => {
                    self.browser.scan_status = format!("Failed to load media: {e}");
                }
            }
        }
    }

    fn poll_scan_events(&mut self) {
        while let Ok(event) = self.scan_rx.try_recv() {
            match event {
                ScanEvent::Started(path) => {
                    self.browser.scan_status = format!("Scanning: {path}...");
                    self.browser.is_scanning = true;
                }
                ScanEvent::Progress(path, count) => {
                    self.browser.scan_status = format!("Scanning: {path} ({count} files)...");
                }
                ScanEvent::Complete(path, count) => {
                    self.browser.scan_status = format!("Done scanning {path}: {count} files");
                    self.browser.is_scanning = false;
                    self.refresh_folders_async();
                    if self.browser.selected_folder_id.is_none() {
                        if let Some(root) = self.browser.folders.iter().find(|f| f.parent_id.is_none()) {
                            self.browser.selected_folder_id = Some(root.id);
                            self.refresh_media_async(true);
                        }
                    } else {
                        self.refresh_media_async(false);
                    }
                }
                ScanEvent::Error(path, err) => {
                    self.browser.scan_status = format!("Error scanning {path}: {err}");
                    self.browser.is_scanning = false;
                    self.push_toast(format!("Scan error in {path}: {err}"), ToastLevel::Error);
                }
            }
        }
    }

    fn poll_thumbnail_events(&mut self, ctx: &egui::Context) {
        while let Ok((hash, epoch, result)) = self.thumbnail_rx.try_recv() {
            self.browser.pending_thumbnails.remove(&hash);
            if epoch != self.browser.thumbnail_epoch {
                continue;
            }
            match result {
                Ok(color_image) => {
                    let texture = ctx.load_texture(
                        &hash,
                        color_image,
                        egui::TextureOptions::default(),
                    );
                    self.browser.textures.insert(hash, texture);
                }
                Err(e) => {
                    tracing::warn!("Thumbnail failed for {}: {}", hash, e);
                    self.push_toast(format!("Thumbnail failed: {}", e), ToastLevel::Warning);
                }
            }
        }
    }

    fn open_viewer(&mut self, ctx: &egui::Context, index: usize) {
        self.viewer_open = true;
        self.viewer_just_opened = true;
        self.viewer_index = Some(index);
        self.viewer_scale_mode = self.resolve_initial_scale_mode(ctx, index);
        self.viewer_texture = None;
        self.load_viewer_image();
    }

    fn resolve_initial_scale_mode(
        &self,
        ctx: &egui::Context,
        index: usize,
    ) -> crate::config::ViewerScaleMode {
        match self.config.ui.viewer_default_scale_mode {
            crate::config::ViewerScaleMode::Fit => crate::config::ViewerScaleMode::Fit,
            crate::config::ViewerScaleMode::OneToOne => crate::config::ViewerScaleMode::OneToOne,
            crate::config::ViewerScaleMode::Smallest => {
                let bottom_height = 80.0;
                let screen = ctx.screen_rect();
                let avail = egui::vec2(screen.width(), screen.height() - bottom_height);
                if let Some(media) = self.browser.media_summaries.get(index) {
                    let fits = match (media.width, media.height) {
                        (Some(w), Some(h)) => {
                            let w = w as f32;
                            let h = h as f32;
                            w <= avail.x && h <= avail.y
                        }
                        _ => false,
                    };
                    if fits {
                        crate::config::ViewerScaleMode::OneToOne
                    } else {
                        crate::config::ViewerScaleMode::Fit
                    }
                } else {
                    crate::config::ViewerScaleMode::Fit
                }
            }
        }
    }

    fn close_viewer(&mut self) {
        self.viewer_open = false;
        self.viewer_index = None;
        self.viewer_texture = None;
        self.pending_viewer_images.clear();
    }

    fn navigate_viewer(&mut self, delta: isize) {
        if let Some(idx) = self.viewer_index {
            let len = self.browser.media_summaries.len();
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
            if let Some(media) = self.browser.media_summaries.get(idx) {
                let hash = media.blake3_hash.clone();
                if self.pending_viewer_images.contains(&hash) {
                    return;
                }
                self.pending_viewer_images.insert(hash.clone());
                let source = std::path::PathBuf::from(&media.absolute_path);
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
            self.pending_viewer_images.remove(&hash);
            let current_hash = self.viewer_index
                .and_then(|idx| self.browser.media_summaries.get(idx))
                .map(|m| m.blake3_hash.as_str());
            if current_hash != Some(hash.as_str()) {
                continue;
            }
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
                    let frame = egui::Frame::new()
                        .fill(bg)
                        .corner_radius(8.0)
                        .inner_margin(12.0);
                    frame.show(ui, |ui| {
                        ui.set_min_size(rect.size());
                        ui.colored_label(fg, &toast.message);
                    });
                });

            y -= spacing;
        }
    }

    fn trigger_rescan(&mut self) {
        if self.browser.is_scanning {
            return;
        }
        self.browser.is_scanning = true;
        self.browser.scan_status = "Rescanning...".to_string();

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
        self.poll_search_names_events();
        self.poll_media_events();
        self.poll_thumbnail_events(ctx);
        self.poll_viewer_images(ctx);
        self.browser.process_thumbnail_queue(&self.thumbnailer);

        if self.browser.is_scanning && self.last_refresh.elapsed() > std::time::Duration::from_secs(2) {
            self.last_refresh = std::time::Instant::now();
            self.refresh_folders_async();
            if self.browser.selected_folder_id.is_some() {
                self.refresh_media_async(false);
            }
        }

        let actions = self.browser.show(ctx);

        if let Some(id) = actions.selected_folder {
            self.browser.selected_folder_id = Some(id);
            self.browser.search_query.clear();
            self.browser.search_active = false;
            self.refresh_media_async(true);
        }
        if let Some(query) = actions.search_changed {
            if query.is_empty() {
                self.refresh_media_async(true);
            } else {
                self.refresh_search_async(query);
            }
        }
        if let Some(idx) = actions.opened_thumbnail {
            self.open_viewer(ctx, idx);
        }
        if actions.rescan_requested {
            self.trigger_rescan();
        }
        if actions.settings_toggled {
            self.browser.settings_open = !self.browser.settings_open;
        }
        if let Some(path) = actions.show_in_file_manager {
            if let Err(e) = crate::ui::context_menu::open_containing_folder(&path) {
                self.push_toast(format!("Failed to open folder: {e}"), ToastLevel::Error);
            }
        }
        if let Some(path) = actions.copy_to_clipboard {
            if let Err(e) = crate::ui::context_menu::copy_image_to_clipboard(&path) {
                self.push_toast(format!("Failed to copy to clipboard: {e}"), ToastLevel::Error);
            } else {
                self.push_toast("Image copied to clipboard".to_string(), ToastLevel::Info);
            }
        }
        if let Some(key) = actions.sort_key_changed {
            self.config.ui.sort_key = key;
            if let Err(e) = self.config.save() {
                self.push_toast(format!("Failed to save config: {e}"), ToastLevel::Error);
            }
        }
        if let Some(order) = actions.sort_order_changed {
            self.config.ui.sort_order = order;
            if let Err(e) = self.config.save() {
                self.push_toast(format!("Failed to save config: {e}"), ToastLevel::Error);
            }
        }

        if self.viewer_open {
            if let Some(idx) = self.viewer_index {
                let media = self.browser.media_summaries.get(idx).cloned();
                if let Some(media) = media {
                    let resp = crate::ui::viewer::show(
                        ctx,
                        &media,
                        &self.viewer_texture,
                        self.viewer_scale_mode,
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
                    if resp.cycle_scale_mode {
                        self.viewer_scale_mode = match self.viewer_scale_mode {
                            crate::config::ViewerScaleMode::Fit => crate::config::ViewerScaleMode::OneToOne,
                            _ => crate::config::ViewerScaleMode::Fit,
                        };
                    }
                    if resp.show_in_file_manager {
                        if let Err(e) = crate::ui::context_menu::open_containing_folder(&media.absolute_path) {
                            self.push_toast(format!("Failed to open folder: {e}"), ToastLevel::Error);
                        }
                    }
                    if resp.copy_to_clipboard {
                        if let Err(e) = crate::ui::context_menu::copy_image_to_clipboard(&media.absolute_path) {
                            self.push_toast(format!("Failed to copy to clipboard: {e}"), ToastLevel::Error);
                        } else {
                            self.push_toast("Image copied to clipboard".to_string(), ToastLevel::Info);
                        }
                    }
                } else {
                    self.close_viewer();
                }
            } else {
                self.close_viewer();
            }
            self.viewer_just_opened = false;
        }

        if self.browser.settings_open {
            let settings_actions = crate::ui::settings::show(
                ctx,
                &mut self.browser.settings_open,
                &mut self.config,
            );
            let mut settings_changed = false;
            for action in settings_actions {
                match action {
                    crate::ui::settings::SettingsAction::ThumbnailSizeChanged(size) => {
                        self.thumbnailer.size = size;
                        self.browser.textures.clear();
                        self.browser.pending_thumbnails.clear();
                        self.browser.thumbnail_queue.clear();
                        self.browser.queued_indices.clear();
                        self.browser.thumbnail_epoch += 1;
                        settings_changed = true;
                    }
                    crate::ui::settings::SettingsAction::ThemeChanged(dark) => {
                        crate::theme::apply(ctx, dark);
                        settings_changed = true;
                    }
                    crate::ui::settings::SettingsAction::DoubleClickDebounceChanged => {
                        Self::apply_input_options(ctx, &self.config);
                        settings_changed = true;
                    }
                    crate::ui::settings::SettingsAction::ScrollSpeedChanged(speed) => {
                        self.browser.scroll_speed = speed;
                        settings_changed = true;
                    }
                    crate::ui::settings::SettingsAction::ViewerDefaultScaleModeChanged => {
                        settings_changed = true;
                    }
                }
            }
            if settings_changed {
                if let Err(e) = self.config.save() {
                    self.push_toast(format!("Failed to save config: {}", e), ToastLevel::Error);
                }
            }
        }

        self.show_toasts(ctx);
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}
