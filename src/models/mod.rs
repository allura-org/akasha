use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;

pub mod loader;
#[cfg(feature = "candle")]
pub mod preprocess;
#[cfg(feature = "candle")]
pub mod tagger;
#[cfg(feature = "candle")]
pub mod worker;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelOutputKind {
    Tags,
    Description,
    Classification,
    Vector,
}

#[derive(Debug, Clone)]
pub enum ModelOutput {
    Tags(HashMap<String, f32>),
    Description(String),
    Classification { label: String, score: f32 },
    Vector(Vec<f32>),
}

#[async_trait::async_trait]
pub trait CandleModel: Send + Sync {
    fn name(&self) -> &str;
    fn kind(&self) -> ModelOutputKind;
    fn infer(&self, image_path: &Path) -> Result<ModelOutput>;
}
