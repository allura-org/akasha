use std::path::Path;

use anyhow::Result;
use candle_core::Device;

use crate::config::ModelConfig;
use crate::models::loader::ModelFiles;
use super::{VlmArchitecture, VlmModel};

pub struct Gemma4Architecture;

struct Gemma4Model;

impl VlmModel for Gemma4Model {
    fn generate(&mut self, _image_path: &Path, _prompt: Option<&str>) -> Result<String> {
        anyhow::bail!("Gemma 4 VLM not yet implemented")
    }
}

impl VlmArchitecture for Gemma4Architecture {
    fn name(&self) -> &'static str {
        "gemma4"
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
        Ok(Box::new(Gemma4Model))
    }
}
