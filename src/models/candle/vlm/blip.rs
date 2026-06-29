use anyhow::Result;
use candle_core::Device;

use crate::config::ModelConfig;
use crate::models::loader::ModelFiles;

use super::{VlmArchitecture, VlmModel};

pub struct BlipArchitecture;

impl VlmArchitecture for BlipArchitecture {
    fn name(&self) -> &'static str {
        "blip"
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        if config.backend.as_deref() == Some("candle-blip") {
            return true;
        }
        let haystack = format!(
            "{} {}",
            config.name,
            config.path.as_deref().unwrap_or("")
        )
        .to_lowercase();
        haystack.contains("blip")
    }

    fn load(
        &self,
        _config: &ModelConfig,
        _files: &ModelFiles,
        _device: &Device,
    ) -> Result<Box<dyn VlmModel>> {
        anyhow::bail!("BLIP VLM is not yet implemented; use Gemma4 for descriptions")
    }
}
