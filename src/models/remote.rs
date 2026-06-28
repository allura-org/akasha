//! Remote HTTP backend for OpenAI-compatible and custom inference endpoints.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};
use base64::Engine;

use crate::config::{ModelConfig, ModelKind, RemoteConfig};
use super::{Backend, Model, ModelOutput};

pub struct RemoteBackend {
    global: RemoteConfig,
}

impl RemoteBackend {
    pub fn new(global: RemoteConfig) -> Self {
        Self { global }
    }
}

impl Backend for RemoteBackend {
    fn id(&self) -> &'static str { "remote" }
    fn is_available(&self) -> bool { true }

    fn supports(&self, config: &ModelConfig) -> bool {
        config.kind == ModelKind::Remote || config.base_url.is_some()
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        if config.tags.is_none() {
            anyhow::bail!("remote backend only supports tags output kind right now");
        }

        let base_url = config.base_url.as_deref().context("remote model missing base_url")?;
        let model_id = config.model_id.as_deref().context("remote model missing model_id")?;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()
            .context("failed to build remote HTTP client")?;

        let endpoints = config.remote.clone().unwrap_or_else(|| crate::config::ModelRemoteOptions {
            chat_endpoint: self.global.chat_endpoint.clone(),
            tag_endpoint: self.global.tag_endpoint.clone(),
            classify_endpoint: self.global.classify_endpoint.clone(),
        });

        Ok(Arc::new(RemoteModel {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model_id: model_id.to_string(),
            api_key: config.api_key.clone(),
            endpoints,
        }))
    }
}

struct RemoteModel {
    client: reqwest::blocking::Client,
    base_url: String,
    model_id: String,
    api_key: Option<String>,
    endpoints: crate::config::ModelRemoteOptions,
}

impl Model for RemoteModel {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        // Default to tags for this milestone.
        self.tag_image(image_path)
    }
}

impl RemoteModel {
    fn tag_image(&self, image_path: &Path) -> Result<ModelOutput> {
        let image_bytes = std::fs::read(image_path)
            .with_context(|| format!("failed to read image: {}", image_path.display()))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

        let url = format!("{}{}", self.base_url, self.endpoints.tag_endpoint);
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "model": self.model_id,
            "image": b64,
        }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response: serde_json::Value = req.send()
            .context("remote tag request failed")?
            .error_for_status()
            .context("remote tag request returned error status")?
            .json()
            .context("failed to parse remote tag response")?;

        let mut tags = HashMap::new();
        if let Some(obj) = response.as_object() {
            for (k, v) in obj {
                if let Some(score) = v.as_f64() {
                    tags.insert(k.clone(), score as f32);
                }
            }
        }
        Ok(ModelOutput::Tags(tags))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind, ModelRemoteOptions, ModelTagsOptions};
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn remote_backend_supports_base_url() {
        let backend = RemoteBackend::new(RemoteConfig::default());
        let cfg = ModelConfig {
            name: "remote".into(),
            kind: ModelKind::Remote,
            backend: None,
            path: None,
            base_url: Some("https://example.com".into()),
            model_id: Some("m1".into()),
            api_key: None,
            tags: Some(ModelTagsOptions { threshold: 0.35 }),
            description: None,
            classification: None,
            remote: Some(ModelRemoteOptions::default()),
        };
        assert!(backend.supports(&cfg));
    }

    #[test]
    fn remote_backend_http_stub_returns_tags() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();

        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let body = r#"{"tag_a": 0.9, "tag_b": 0.5}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).expect("write");
        });

        let temp_path = std::env::temp_dir().join("akasha_remote_stub_test.bin");
        std::fs::write(&temp_path, b"fake-image-bytes").expect("write temp file");

        let config = ModelConfig {
            name: "remote-stub".into(),
            kind: ModelKind::Remote,
            backend: None,
            path: None,
            base_url: Some(format!("http://127.0.0.1:{}", port)),
            model_id: Some("stub-model".into()),
            api_key: None,
            tags: Some(ModelTagsOptions { threshold: 0.35 }),
            description: None,
            classification: None,
            remote: Some(ModelRemoteOptions::default()),
        };

        let output = RemoteBackend::new(RemoteConfig::default())
            .load(&config)
            .expect("load")
            .infer(&temp_path)
            .expect("infer");
        std::fs::remove_file(&temp_path).ok();

        match output {
            ModelOutput::Tags(tags) => {
                assert!((tags.get("tag_a").copied().unwrap_or(0.0) - 0.9).abs() < 1e-6);
                assert!((tags.get("tag_b").copied().unwrap_or(0.0) - 0.5).abs() < 1e-6);
            }
            other => panic!("expected ModelOutput::Tags, got {:?}", other),
        }
    }
}
