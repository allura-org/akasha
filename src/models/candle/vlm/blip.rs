use std::path::Path;

use anyhow::Result;
use candle_core::Device;

use crate::config::ModelConfig;
use crate::models::loader::ModelFiles;
use super::{VlmArchitecture, VlmModel};

pub struct BlipArchitecture;

struct BlipModel;

impl VlmModel for BlipModel {
    fn generate(&mut self, _image_path: &Path, _prompt: Option<&str>) -> Result<String> {
        anyhow::bail!("BLIP VLM not yet implemented")
    }
}

impl VlmArchitecture for BlipArchitecture {
    fn name(&self) -> &'static str {
        "blip"
    }

    fn supports(&self, _config: &ModelConfig) -> bool {
        false
    }

    fn load(
        &self,
        _config: &ModelConfig,
        _files: &ModelFiles,
        _device: &Device,
    ) -> Result<Box<dyn VlmModel>> {
        Ok(Box::new(BlipModel))
    }
}
