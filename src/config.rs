use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub ui: UiConfig,
    pub thumbnails: ThumbnailsConfig,
    pub debug: DebugConfig,
    pub models: ModelsConfig,
    #[serde(alias = "folders")]
    pub imports: Vec<ImportConfig>,
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
    pub double_click_debounce_ms: u64,
    pub scroll_speed: f32,
    pub viewer_default_scale_mode: ViewerScaleMode,
    pub sort_key: SortKey,
    pub sort_order: SortOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ThumbnailsConfig {
    /// The size of thumbnails, by longest side.
    pub thumbnail_size: u32,
    /// The location of the global cache. Empty means `$HOME/.cache/akasha`.
    pub cache_folder: String,
    /// Disables writing to *any* thumbnail cache. Caches are still read.
    pub disable_cache: bool,
    /// Writes all new thumbnails to `/tmp/.akasha_thumbnails`. Deleted on exit.
    pub temporary_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugConfig {
    /// Disable all thumbnail cache reads, forcing regeneration.
    pub no_cache_read: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub tagger: Vec<ModelConfig>,
    pub classifier: Vec<ModelConfig>,
    pub visionlanguage: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: ModelKind,
    /// Local model: HuggingFace slug or on-disk directory.
    pub source: Option<String>,
    /// Remote OpenAI-compatible base URL.
    pub base_url: Option<String>,
    /// Remote model identifier.
    pub model_id: Option<String>,
    /// API key for remote models. Prefer env vars / keyring in production.
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Local,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImportConfig {
    pub path: String,
    #[serde(default = "default_true")]
    pub recursive: bool,
    #[serde(default)]
    pub flatten: bool,
    #[serde(default, alias = "blacklist")]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub include: Vec<String>,
    pub thumbnails: ImportThumbnailsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ImportThumbnailsConfig {
    pub cache_mode: String,
    pub cache_folder: String,
    pub cache_fallback: String,
}

fn default_true() -> bool {
    true
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ui: UiConfig::default(),
            thumbnails: ThumbnailsConfig::default(),
            debug: DebugConfig::default(),
            models: ModelsConfig::default(),
            imports: Vec::new(),
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "dark".to_string(),
            double_click_debounce_ms: 300,
            scroll_speed: 1.0,
            viewer_default_scale_mode: ViewerScaleMode::Smallest,
            sort_key: SortKey::Filename,
            sort_order: SortOrder::Ascending,
        }
    }
}

impl Default for ThumbnailsConfig {
    fn default() -> Self {
        Self {
            thumbnail_size: 512,
            cache_folder: String::new(),
            disable_cache: false,
            temporary_cache: false,
        }
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            no_cache_read: false,
        }
    }
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            recursive: true,
            flatten: false,
            exclude: Vec::new(),
            include: Vec::new(),
            thumbnails: ImportThumbnailsConfig::default(),
        }
    }
}

impl Default for ImportThumbnailsConfig {
    fn default() -> Self {
        Self {
            cache_mode: "global".to_string(),
            cache_folder: String::new(),
            cache_fallback: "disable".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_models_config() {
        let text = r#"
[ui]
theme = "dark"

[[models.tagger]]
name = "wd14"
type = "local"
source = "SmilingWolf/wd-v1-4-convnext-tagger-v2"

[[models.classifier]]
name = "nsfw"
type = "remote"
base_url = "http://localhost:8000/v1"
model_id = "my-classifier"
"#;

        let config: Config = toml::from_str(text).unwrap();
        assert_eq!(config.models.tagger.len(), 1);
        assert_eq!(config.models.tagger[0].name, "wd14");
        assert_eq!(config.models.tagger[0].kind, ModelKind::Local);
        assert_eq!(config.models.tagger[0].source.as_deref(), Some("SmilingWolf/wd-v1-4-convnext-tagger-v2"));
        assert_eq!(config.models.classifier.len(), 1);
        assert_eq!(config.models.classifier[0].kind, ModelKind::Remote);
        assert_eq!(config.models.classifier[0].base_url.as_deref(), Some("http://localhost:8000/v1"));
    }
}
