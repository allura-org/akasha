use eframe::egui;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::db;
use crate::thumbnailer::Thumbnailer;

#[derive(Debug, Clone)]
pub struct ThumbnailJob {
    pub idx: usize,
    pub hash: String,
    pub priority: i64,
}

pub struct BrowserActions {
    pub selected_folder: Option<i64>,
    pub opened_thumbnail: Option<usize>,
    pub rescan_requested: bool,
    pub settings_toggled: bool,
    pub show_in_file_manager: Option<String>,
    pub copy_to_clipboard: Option<String>,
}

const MAX_CONCURRENT_THUMBNAILS: usize = 8;
const THUMB_CELL_HEIGHT: f32 = 230.0;

pub struct BrowserPanel {
    pub folders: Vec<db::folder::Folder>,
    pub selected_folder_id: Option<i64>,
    pub expanded_folders: HashSet<i64>,
    pub media_summaries: Vec<db::media::MediaSummary>,
    pub media_items: Vec<db::media::MediaFile>,
    pub textures: HashMap<String, egui::TextureHandle>,
    pub pending_thumbnails: HashSet<String>,
    pub thumbnail_queue: Vec<ThumbnailJob>,
    pub queued_indices: HashSet<usize>,
    pub scroll_offset: f32,
    pub scroll_velocity: f32,
    pub last_scroll_offset: f32,
    pub last_scroll_time: std::time::Instant,
    pub thumbnail_epoch: u64,
    pub media_epoch: u64,
    pub scan_status: String,
    pub is_scanning: bool,
    pub settings_open: bool,
    pub scroll_speed: f32,
    pub folder_filter: String,

    rt: Arc<Runtime>,
    thumbnail_tx: std::sync::mpsc::Sender<(String, u64, Result<egui::ColorImage, String>)>,
}

impl BrowserPanel {
    pub fn new(
        rt: Arc<Runtime>,
        thumbnail_tx: std::sync::mpsc::Sender<(String, u64, Result<egui::ColorImage, String>)>,
        scroll_speed: f32,
    ) -> Self {
        Self {
            folders: Vec::new(),
            selected_folder_id: None,
            expanded_folders: HashSet::new(),
            media_summaries: Vec::new(),
            media_items: Vec::new(),
            textures: HashMap::new(),
            pending_thumbnails: HashSet::new(),
            thumbnail_queue: Vec::new(),
            queued_indices: HashSet::new(),
            scroll_offset: 0.0,
            scroll_velocity: 0.0,
            last_scroll_offset: 0.0,
            last_scroll_time: std::time::Instant::now(),
            thumbnail_epoch: 0,
            media_epoch: 0,
            scan_status: "Initializing...".to_string(),
            is_scanning: true,
            settings_open: false,
            scroll_speed,
            folder_filter: String::new(),
            rt,
            thumbnail_tx,
        }
    }

    pub fn clear_for_refresh(&mut self, hard_reset: bool) {
        if hard_reset {
            self.media_epoch += 1;
            self.scan_status = "Loading images...".to_string();
            self.media_summaries.clear();
            self.media_items.clear();
            self.textures.clear();
            self.pending_thumbnails.clear();
            self.thumbnail_queue.clear();
            self.queued_indices.clear();
            self.scroll_offset = 0.0;
        }
    }

    pub fn show(&mut self, ctx: &egui::Context) -> BrowserActions {
        let mut actions = BrowserActions {
            selected_folder: None,
            opened_thumbnail: None,
            rescan_requested: false,
            settings_toggled: false,
            show_in_file_manager: None,
            copy_to_clipboard: None,
        };

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
                            actions.settings_toggled = true;
                        }
                        if ui.button("⟳ Rescan").clicked() && !self.is_scanning {
                            actions.rescan_requested = true;
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
                    ui.text_edit_singleline(&mut self.folder_filter);
                    ui.separator();

                    let visible = self.visible_folder_ids();
                    let mut clicked_id = None;
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let mut roots: Vec<i64> = self.folders
                            .iter()
                            .filter(|f| f.parent_id.is_none() && visible.contains(&f.id))
                            .map(|f| f.id)
                            .collect();
                        roots.sort_by_key(|id| {
                            self.folders
                                .iter()
                                .find(|f| f.id == *id)
                                .map(|f| self.folder_name(f).to_lowercase())
                        });
                        for root_id in roots {
                            self.render_folder_tree(ui, root_id, 0, &mut clicked_id, &visible);
                        }
                    });
                    if let Some(id) = clicked_id {
                        actions.selected_folder = Some(id);
                    }
                }
            });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.is_scanning && self.media_summaries.is_empty() && self.selected_folder_id.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.heading("Scanning...");
                    ui.label(&self.scan_status);
                });
            } else if self.media_summaries.is_empty() {
                ui.centered_and_justified(|ui| {
                    if self.selected_folder_id.is_some() {
                        ui.heading("No images in this folder");
                    } else {
                        ui.heading("Select a folder");
                    }
                });
            } else {
                ui.heading(format!("{} images", self.media_summaries.len()));
                ui.separator();

                let cols = (ui.available_width() / 220.0).max(1.0) as usize;
                let rows = (self.media_summaries.len() + cols - 1) / cols;
                let item_size = egui::vec2(200.0, 200.0);
                let label_h = 30.0;
                let row_height = item_size.y + label_h;

                // Apply configured scroll speed multiplier to this ScrollArea only.
                // We scale the input delta, let the ScrollArea consume it, then restore
                // the original so other UI (e.g. the viewer) sees unscaled wheel events.
                let scroll_speed = self.scroll_speed.max(0.1);
                let original_delta = ui.input(|i| i.smooth_scroll_delta);
                ui.input_mut(|i| i.smooth_scroll_delta *= scroll_speed);

                let mut visible_rows: Option<(usize, usize)> = None;
                let scroll = egui::ScrollArea::vertical()
                    .show_rows(ui, row_height, rows, |ui, row_range| {
                        visible_rows = Some((row_range.start, row_range.end));
                        let mut clicked_index = None;
                        for row in row_range {
                            ui.horizontal(|ui| {
                                for col in 0..cols {
                                    let idx = row * cols + col;
                                    if idx >= self.media_summaries.len() {
                                        break;
                                    }
                                    let media = &self.media_summaries[idx];

                                    let clicked = ui.allocate_ui_with_layout(
                                        egui::vec2(item_size.x, row_height),
                                        egui::Layout::top_down(egui::Align::Center),
                                        |ui| {
                                            ui.set_min_height(row_height);
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
                                                ui.add_sized(item_size,
                                                    egui::Image::new((texture.id(), size))
                                                        .fit_to_exact_size(size)
                                                        .sense(egui::Sense::click()),
                                                )
                                            } else {
                                                ui.add_sized(item_size, egui::Spinner::new())
                                            };
                                            response.context_menu(|ui| {
                                                if ui.button("Show in file manager").clicked() {
                                                    actions.show_in_file_manager = Some(media.absolute_path.clone());
                                                    ui.close_menu();
                                                }
                                                if ui.button("Copy to clipboard").clicked() {
                                                    actions.copy_to_clipboard = Some(media.absolute_path.clone());
                                                    ui.close_menu();
                                                }
                                            });
                                            let filename = std::path::Path::new(&media.relative_path)
                                                .file_name()
                                                .and_then(|n| n.to_str())
                                                .unwrap_or(&media.relative_path);
                                            ui.add(egui::Label::new(filename).truncate());
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
                            actions.opened_thumbnail = Some(i);
                        }
                    });
                ui.input_mut(|i| i.smooth_scroll_delta = original_delta);
                self.scroll_offset = scroll.state.offset.y;
                let viewport_h = scroll.inner_rect.height();
                if let Some((first, last)) = visible_rows {
                    self.queue_visible_thumbnails(viewport_h, cols, first, last);
                }
            }
        });

        actions
    }

    fn folder_name<'a>(&self, folder: &'a db::folder::Folder) -> &'a str {
        std::path::Path::new(&folder.path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&folder.path)
    }

    fn visible_folder_ids(&self) -> HashSet<i64> {
        if self.folder_filter.is_empty() {
            return self.folders.iter().map(|f| f.id).collect();
        }

        let filter = self.folder_filter.to_lowercase();
        let mut visible: HashSet<i64> = self
            .folders
            .iter()
            .filter(|f| {
                let name = self.folder_name(f).to_lowercase();
                name.contains(&filter)
            })
            .map(|f| f.id)
            .collect();

        // Propagate visibility up to ancestors.
        let mut changed = true;
        while changed {
            changed = false;
            for folder in &self.folders {
                if visible.contains(&folder.id) {
                    if let Some(parent_id) = folder.parent_id {
                        if visible.insert(parent_id) {
                            changed = true;
                        }
                    }
                }
            }
        }

        visible
    }

    fn render_folder_tree(
        &mut self,
        ui: &mut egui::Ui,
        folder_id: i64,
        depth: usize,
        clicked_id: &mut Option<i64>,
        visible: &HashSet<i64>,
    ) {
        let Some(folder) = self.folders.iter().find(|f| f.id == folder_id) else { return };
        let selected = self.selected_folder_id == Some(folder.id);
        let is_root = folder.parent_id.is_none();
        let name = self.folder_name(folder).to_string();
        let filtering = !self.folder_filter.is_empty();

        // Pre-compute visible children so the mutable closure below doesn't borrow `folder`.
        let mut children: Vec<i64> = self
            .folders
            .iter()
            .filter(|f| f.parent_id == Some(folder.id) && visible.contains(&f.id))
            .map(|f| f.id)
            .collect();
        children.sort_by_key(|id| {
            self.folders
                .iter()
                .find(|f| f.id == *id)
                .map(|f| self.folder_name(f).to_lowercase())
        });
        let has_visible_children = !children.is_empty();

        ui.horizontal(|ui| {
            ui.add_space(depth as f32 * 16.0);

            if has_visible_children {
                let expanded = filtering || self.expanded_folders.contains(&folder_id);
                let arrow = if expanded { "▼" } else { "▶" };
                if ui.small_button(arrow).clicked() && !filtering {
                    if self.expanded_folders.contains(&folder_id) {
                        self.expanded_folders.remove(&folder_id);
                    } else {
                        self.expanded_folders.insert(folder_id);
                    }
                }
            } else {
                ui.add_space(24.0);
            }

            let label = if is_root {
                egui::RichText::new(&name).strong()
            } else {
                egui::RichText::new(&name)
            };

            if ui.selectable_label(selected, label).clicked() && !selected {
                *clicked_id = Some(folder_id);
            }
        });

        let expand = filtering || self.expanded_folders.contains(&folder_id);
        if expand {
            for child_id in children {
                self.render_folder_tree(ui, child_id, depth + 1, clicked_id, visible);
            }
        }
    }

    fn queue_visible_thumbnails(&mut self, viewport_height: f32, cols: usize, first_visible_row: usize, last_visible_row: usize) {
        if cols == 0 || self.media_summaries.is_empty() {
            return;
        }
        let center_row = (first_visible_row + last_visible_row) / 2;

        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_scroll_time).as_secs_f32().max(0.001);
        let rows_delta = (self.scroll_offset - self.last_scroll_offset).abs() / THUMB_CELL_HEIGHT;
        let instant_velocity = rows_delta / dt;
        self.scroll_velocity = self.scroll_velocity * 0.8 + instant_velocity * 0.2;
        self.last_scroll_offset = self.scroll_offset;
        self.last_scroll_time = now;

        let (prefetch_rows, max_dist, fast_scroll) = if self.scroll_velocity > 240.0 {
            (1, 2, true)
        } else if self.scroll_velocity > 60.0 {
            (2, usize::MAX, false)
        } else {
            (5, usize::MAX, false)
        };

        let start_idx = first_visible_row.saturating_sub(prefetch_rows) * cols;
        let end_idx = ((last_visible_row + prefetch_rows) * cols).min(self.media_summaries.len());

        let evict_margin = prefetch_rows + 10;
        let keep_start = first_visible_row.saturating_sub(evict_margin) * cols;
        let keep_end = ((last_visible_row + evict_margin) * cols).min(self.media_summaries.len());
        let keep_hashes: HashSet<&str> = self.media_summaries[keep_start..keep_end]
            .iter()
            .map(|m| m.blake3_hash.as_str())
            .collect();
        self.textures.retain(|hash, _| keep_hashes.contains(hash.as_str()));

        let mut to_queue = Vec::new();
        for i in start_idx..end_idx {
            let hash = &self.media_summaries[i].blake3_hash;
            if !self.textures.contains_key(hash)
                && !self.pending_thumbnails.contains(hash)
            {
                let row = i / cols;
                let dist = if row > center_row {
                    row - center_row
                } else {
                    center_row - row
                };
                if !fast_scroll || dist <= max_dist {
                    to_queue.push(ThumbnailJob {
                        idx: i,
                        hash: hash.clone(),
                        priority: dist as i64,
                    });
                }
            }
        }

        to_queue.sort_by_key(|job| job.priority);
        to_queue.reverse();

        self.thumbnail_queue.clear();
        self.queued_indices.clear();
        for job in to_queue {
            self.queued_indices.insert(job.idx);
            self.thumbnail_queue.push(job);
        }
    }

    pub fn process_thumbnail_queue(&mut self, thumbnailer: &Thumbnailer) {
        let can_spawn = MAX_CONCURRENT_THUMBNAILS.saturating_sub(self.pending_thumbnails.len());
        if can_spawn == 0 {
            return;
        }

        for _ in 0..can_spawn {
            let Some(job) = self.thumbnail_queue.pop() else {
                break;
            };
            self.queued_indices.remove(&job.idx);

            if self.textures.contains_key(&job.hash) || self.pending_thumbnails.contains(&job.hash) {
                continue;
            }

            let Some(media) = self.media_summaries.get(job.idx) else {
                continue;
            };

            let hash = media.blake3_hash.clone();
            self.pending_thumbnails.insert(hash.clone());
            let source = std::path::PathBuf::from(&media.absolute_path);
            let size = thumbnailer.size;
            let cache_mode = thumbnailer.cache_mode.clone();
            let tx = self.thumbnail_tx.clone();
            let epoch = self.thumbnail_epoch;
            let rt = Arc::clone(&self.rt);

            rt.spawn_blocking(move || {
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
}
