use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub ui: UiConfig,
    pub thumbnails: ThumbnailsConfig,
    pub debug: DebugConfig,
    pub models: ModelsConfig,
    pub remote: RemoteConfig,
    #[serde(alias = "folders")]
    pub imports: Vec<ImportConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RemoteConfig {
    pub chat_endpoint: String,
    pub tag_endpoint: String,
    pub classify_endpoint: String,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            chat_endpoint: "/chat/completions".into(),
            tag_endpoint: "/tags".into(),
            classify_endpoint: "/classify".into(),
        }
    }
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
    #[serde(default)]
    pub show_advanced_media_properties: bool,
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
#[serde(default, transparent)]
pub struct ModelsConfig {
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: ModelKind,
    pub backend: Option<String>,
    pub path: Option<String>,
    pub base_url: Option<String>,
    pub model_id: Option<String>,
    pub api_key: Option<String>,
    pub tags: Option<ModelTagsOptions>,
    pub description: Option<ModelDescriptionOptions>,
    pub classification: Option<ModelClassificationOptions>,
    pub remote: Option<ModelRemoteOptions>,
    pub onnx: Option<ModelOnnxOptions>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            kind: ModelKind::Local,
            backend: None,
            path: None,
            base_url: None,
            model_id: None,
            api_key: None,
            tags: None,
            description: None,
            classification: None,
            remote: None,
            onnx: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRemoteOptions {
    #[serde(default = "default_chat_endpoint")]
    pub chat_endpoint: String,
    #[serde(default = "default_tag_endpoint")]
    pub tag_endpoint: String,
    #[serde(default = "default_classify_endpoint")]
    pub classify_endpoint: String,
}

impl Default for ModelRemoteOptions {
    fn default() -> Self {
        Self {
            chat_endpoint: default_chat_endpoint(),
            tag_endpoint: default_tag_endpoint(),
            classify_endpoint: default_classify_endpoint(),
        }
    }
}

fn default_chat_endpoint() -> String { "/chat/completions".into() }
fn default_tag_endpoint() -> String { "/tags".into() }
fn default_classify_endpoint() -> String { "/classify".into() }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOnnxOptions {
    /// Explicit ONNX model file name. If omitted, OrtBackend searches the model folder.
    pub model_file: Option<String>,
    /// Explicit tags/labels file. If omitted, OrtBackend searches the model folder.
    pub tags_file: Option<String>,
    /// Explicit preprocessing config file. If omitted, OrtBackend searches the model folder.
    pub config_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTagsOptions {
    #[serde(default = "default_threshold")]
    pub threshold: f32,
    #[serde(default = "default_top_k")]
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelDescriptionOptions {
    pub prompt: Option<String>,
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// Optional per-channel mean for image normalization. If both `image_mean`
    /// and `image_std` are provided, preprocessing applies `(pixel/255 - mean) / std`
    /// per channel, overriding the model-specific defaults. If omitted, the model's
    /// default normalization is used (for Gemma 4 this is a no-op: mean `[0,0,0]`,
    /// std `[1,1,1]`).
    pub image_mean: Option<Vec<f32>>,
    /// Optional per-channel standard deviation for image normalization. See `image_mean`.
    pub image_std: Option<Vec<f32>>,
}

impl Default for ModelDescriptionOptions {
    fn default() -> Self {
        Self {
            prompt: None,
            max_tokens: 128,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            image_mean: None,
            image_std: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelClassificationOptions {}

fn default_threshold() -> f32 {
    0.35
}

fn default_top_k() -> Option<usize> {
    Some(100)
}

impl Default for ModelTagsOptions {
    fn default() -> Self {
        Self {
            threshold: default_threshold(),
            top_k: default_top_k(),
        }
    }
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
            remote: RemoteConfig::default(),
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
            show_advanced_media_properties: false,
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
    fn parse_unified_models_config() {
        let text = r#"
[ui]
theme = "dark"

[[models]]
name = "wd-vit-tagger-v3"
type = "local"
path = "SmilingWolf/wd-vit-tagger-v3"

[models.tags]
threshold = 0.35
"#;

        let config: Config = toml::from_str(text).unwrap();
        assert_eq!(config.models.models.len(), 1);
        assert_eq!(config.models.models[0].name, "wd-vit-tagger-v3");
        assert_eq!(config.models.models[0].kind, ModelKind::Local);
        assert_eq!(config.models.models[0].path.as_deref(), Some("SmilingWolf/wd-vit-tagger-v3"));
        assert_eq!(config.models.models[0].tags.as_ref().unwrap().threshold, 0.35);
        assert_eq!(config.models.models[0].tags.as_ref().unwrap().top_k, Some(100));
    }

    #[test]
    fn parse_model_with_backend_and_remote_options() {
        let text = r#"
[remote]
chat_endpoint = "/v1/chat"
tag_endpoint = "/v1/tag"

[[models]]
name = "remote-model"
type = "remote"
backend = "remote"
base_url = "https://example.com"
model_id = "m1"
api_key = "secret"

[models.remote]
classify_endpoint = "/v1/classify"
"#;

        let config: Config = toml::from_str(text).unwrap();
        let model = &config.models.models[0];
        assert_eq!(model.backend.as_deref(), Some("remote"));
        assert_eq!(model.remote.as_ref().unwrap().classify_endpoint.as_str(), "/v1/classify");
        assert_eq!(config.remote.chat_endpoint.as_str(), "/v1/chat");
    }

    #[test]
    fn parse_model_description_options() {
        let text = r#"
[[models]]
name = "gemma-4-E2B-it"
type = "local"
path = "google/gemma-4-E2B-it"

[models.description]
prompt = "Describe this image in one sentence."
max_tokens = 64
temperature = 0.5
top_p = 0.9
top_k = 20
repeat_penalty = 1.1
repeat_last_n = 32
"#;
        let config: Config = toml::from_str(text).unwrap();
        let desc = config.models.models[0].description.as_ref().unwrap();
        assert_eq!(desc.prompt.as_deref(), Some("Describe this image in one sentence."));
        assert_eq!(desc.max_tokens, 64);
        assert_eq!(desc.temperature, Some(0.5));
        assert_eq!(desc.top_p, Some(0.9));
        assert_eq!(desc.top_k, Some(20));
        assert!((desc.repeat_penalty - 1.1).abs() < f32::EPSILON);
        assert_eq!(desc.repeat_last_n, 32);
    }
}
