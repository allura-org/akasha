//! Burn backend for Akasha.
//!
//! This is a generic backend that runs models implemented directly in Burn.
//! The first supported model is Hydra-3.5 (RedRocket/Hydra), loaded from its
//! original `.safetensors` checkpoint.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use burn::backend::{ndarray::NdArrayDevice, NdArray};

use crate::config::{ModelConfig, ModelKind};

use super::{Backend, Model};

pub mod hydra;

/// Burn backend selected at compile time.
type BurnBackendType = NdArray;

pub struct BurnBackend;

impl Backend for BurnBackend {
    fn id(&self) -> &'static str {
        "burn"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        if config.kind != ModelKind::Local {
            return false;
        }

        if config.backend.as_deref() == Some("burn") {
            return true;
        }

        if let Some(path) = &config.path {
            if is_hydra_checkpoint(path) {
                return true;
            }
        }

        false
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let device = NdArrayDevice::default();
        let model = hydra::HydraModel::<BurnBackendType>::load(config, device)
            .context("failed to load Hydra-3.5 Burn model")?;
        Ok(Arc::new(model))
    }
}

fn is_hydra_checkpoint(path: &str) -> bool {
    let p = Path::new(path);
    let file = if p.is_file() {
        p.to_path_buf()
    } else {
        p.join("hydra-3.5.safetensors")
    };

    if !file.is_file() {
        return false;
    }

    match read_safetensors_metadata(&file) {
        Ok(meta) => meta
            .get("modelspec.architecture")
            .map(|s| s.as_str())
            == Some("naflexvit_so400m_patch16_siglip+rr_hydra2"),
        Err(_) => false,
    }
}

pub(crate) fn read_safetensors_metadata(path: &Path) -> Result<HashMap<String, String>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read safetensors file: {}", path.display()))?;

    // The safetensors header is a JSON object prefixed by its length as a
    // little-endian u64. We avoid relying on version-specific crate APIs by
    // parsing the header directly.
    let header_len = u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .context("safetensors file is too small for header length")?,
    ) as usize;
    anyhow::ensure!(
        bytes.len() >= 8 + header_len,
        "safetensors header length exceeds file size"
    );

    #[derive(Debug, serde::Deserialize)]
    struct Header {
        #[serde(rename = "__metadata__", default)]
        metadata: Option<HashMap<String, String>>,
    }

    let header: Header = serde_json::from_slice(&bytes[8..8 + header_len])
        .with_context(|| format!("failed to parse safetensors header JSON from {}", path.display()))?;

    Ok(header.metadata.unwrap_or_default())
}
