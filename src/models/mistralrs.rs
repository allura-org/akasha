//! mistral.rs local inference backend for vision-language description jobs.

use std::path::Path;
use std::sync::Arc;
use anyhow::{Context, Result};
use mistralrs::{
    IsqType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};
use tokio::runtime::Handle;

use crate::config::{ModelConfig, ModelDescriptionOptions};
use crate::models::{Backend, Model, ModelOutput};

pub struct MistralRsBackend;

impl Backend for MistralRsBackend {
    fn id(&self) -> &'static str {
        "mistralrs"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports(&self, config: &ModelConfig) -> bool {
        config.kind == crate::config::ModelKind::Local
            && config.description.is_some()
            && config.backend.as_deref() == Some("mistralrs")
    }

    fn load(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        let model_id = config
            .model_id
            .clone()
            .or_else(|| config.path.clone())
            .context("mistralrs backend requires a model_id or path")?;
        let opts = config.description.clone().unwrap_or_default();
        // `Backend::load` is called from `tokio::task::spawn_blocking`, so a
        // runtime handle is available. We keep only the handle (not a full
        // `Runtime`) so dropping the model on an async worker thread does not
        // try to tear down a nested runtime.
        let handle = Handle::current();
        let isq = opts.isq.as_deref().unwrap_or("Q8_0");
        let mut builder = MultimodalModelBuilder::new(&model_id).with_logging();
        if !isq.eq_ignore_ascii_case("none") {
            let isq_type = parse_isq(isq)
                .with_context(|| format!("invalid mistralrs isq value: {isq}"))?;
            builder = builder.with_isq(isq_type);
        }
        let model = match handle.block_on(async {
            builder
                .build()
                .await
                .context("failed to build mistralrs multimodal model")
        }) {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(error = ?e, "mistralrs model build failed");
                for (i, cause) in e.chain().enumerate() {
                    tracing::error!(cause_index = i, "{}", cause);
                }
                return Err(e);
            }
        };
        Ok(Arc::new(MistralRsModel {
            model,
            opts,
            handle,
        }))
    }
}

struct MistralRsModel {
    model: mistralrs::Model,
    opts: ModelDescriptionOptions,
    handle: Handle,
}

impl Model for MistralRsModel {
    fn infer(&self, image_path: &Path) -> Result<ModelOutput> {
        let image = image::open(image_path)
            .with_context(|| format!("failed to open image {image_path:?}"))?;
        let prompt = self
            .opts
            .prompt
            .clone()
            .unwrap_or_else(|| "Describe this image in detail.".into());

        self.handle.block_on(async {
            let messages = MultimodalMessages::new()
                .add_image_message(TextMessageRole::User, &prompt, vec![image]);
            let mut request =
                RequestBuilder::from(messages).set_sampler_max_len(self.opts.max_tokens);
            if let Some(temperature) = self.opts.temperature {
                request = request.set_sampler_temperature(temperature);
            }
            if let Some(top_p) = self.opts.top_p {
                request = request.set_sampler_topp(top_p);
            }
            if let Some(top_k) = self.opts.top_k {
                request = request.set_sampler_topk(top_k);
            }
            let response = self
                .model
                .send_chat_request(request)
                .await
                .context("mistralrs inference failed")?;
            let text = response
                .choices
                .first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();
            Ok(ModelOutput::Description(text))
        })
    }
}

fn parse_isq(s: &str) -> Result<IsqType> {
    match s.to_ascii_uppercase().as_str() {
        "Q4_0" => Ok(IsqType::Q4_0),
        "Q4_1" => Ok(IsqType::Q4_1),
        "Q5_0" => Ok(IsqType::Q5_0),
        "Q5_1" => Ok(IsqType::Q5_1),
        "Q8_0" => Ok(IsqType::Q8_0),
        "Q8_1" => Ok(IsqType::Q8_1),
        "Q2K" => Ok(IsqType::Q2K),
        "Q3K" => Ok(IsqType::Q3K),
        "Q4K" => Ok(IsqType::Q4K),
        "Q5K" => Ok(IsqType::Q5K),
        "Q6K" => Ok(IsqType::Q6K),
        "Q8K" => Ok(IsqType::Q8K),
        "HQQ8" => Ok(IsqType::HQQ8),
        "HQQ4" => Ok(IsqType::HQQ4),
        "F8E4M3" => Ok(IsqType::F8E4M3),
        "AFQ8" => Ok(IsqType::AFQ8),
        "AFQ6" => Ok(IsqType::AFQ6),
        "AFQ4" => Ok(IsqType::AFQ4),
        "AFQ3" => Ok(IsqType::AFQ3),
        "AFQ2" => Ok(IsqType::AFQ2),
        "F8Q8" => Ok(IsqType::F8Q8),
        "MXFP4" => Ok(IsqType::MXFP4),
        _ => anyhow::bail!("unknown ISQ type: {s}"),
    }
}
