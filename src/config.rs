use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub ui: UiConfig,
    pub thumbnails: ThumbnailConfig,
    pub folders: Vec<FolderConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewerScaleMode {
    Fit,
    OneToOne,
    Smallest,
}

impl ViewerScaleMode {
    pub fn label(&self) -> &'static str {
        match self {
            ViewerScaleMode::Fit => "Fit",
            ViewerScaleMode::OneToOne => "1:1",
            ViewerScaleMode::Smallest => "Smallest",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortKey {
    Filename,
    Size,
    DateCreated,
    DateModified,
    Score,
}

impl SortKey {
    pub fn label(&self) -> &'static str {
        match self {
            SortKey::Filename => "Filename",
            SortKey::Size => "Size",
            SortKey::DateCreated => "Date created",
            SortKey::DateModified => "Date modified",
            SortKey::Score => "Score",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    Ascending,
    Descending,
}

impl SortOrder {
    pub fn label(&self) -> &'static str {
        match self {
            SortOrder::Ascending => "Ascending",
            SortOrder::Descending => "Descending",
        }
    }

    pub fn toggle(&self) -> Self {
        match self {
            SortOrder::Ascending => SortOrder::Descending,
            SortOrder::Descending => SortOrder::Ascending,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub theme: String,
    pub thumbnail_size: u32,
    pub double_click_debounce_ms: u64,
    pub scroll_speed: f32,
    pub viewer_default_scale_mode: ViewerScaleMode,
    pub sort_key: SortKey,
    pub sort_order: SortOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ThumbnailConfig {
    pub cache_mode: String, // "disabled" | "global" | "per_folder" | "custom"
    pub custom_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderConfig {
    pub path: String,
    #[serde(default = "default_true")]
    pub recursive: bool,
    #[serde(default)]
    pub show_recursive: bool,
    #[serde(default)]
    pub blacklist: Vec<String>,
    pub thumbnail_cache_mode: Option<String>,
}

fn default_true() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ui: UiConfig::default(),
            thumbnails: ThumbnailConfig::default(),
            folders: Vec::new(),
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            thumbnail_size: 256,
            double_click_debounce_ms: 300,
            scroll_speed: 1.0,
            viewer_default_scale_mode: ViewerScaleMode::Smallest,
            sort_key: SortKey::Filename,
            sort_order: SortOrder::Ascending,
        }
    }
}

impl Default for ThumbnailConfig {
    fn default() -> Self {
        Self {
            cache_mode: "global".to_string(),
            custom_path: String::new(),
        }
    }
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let path = Self::config_path()?;
        if path.exists() {
            let text = std::fs::read_to_string(&path)?;
            let config: Config = toml::from_str(&text)?;
            Ok(config)
        } else {
            let config = Config::default();
            config.save()?;
            Ok(config)
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(&path, text)?;
        Ok(())
    }

    fn config_path() -> anyhow::Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("", "", "akasha")
            .ok_or_else(|| anyhow::anyhow!("Could not determine project directories"))?;
        Ok(dirs.config_dir().join("config.toml"))
    }

    pub fn data_dir() -> anyhow::Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("", "", "akasha")
            .ok_or_else(|| anyhow::anyhow!("Could not determine project directories"))?;
        Ok(dirs.data_dir().to_path_buf())
    }
}
