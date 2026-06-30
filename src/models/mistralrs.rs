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
        config.description.is_some()
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
        let model = handle.block_on(async {
            MultimodalModelBuilder::new(&model_id)
                .with_isq(IsqType::Q4K)
                .build()
                .await
                .context("failed to build mistralrs multimodal model")
        })?;
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
