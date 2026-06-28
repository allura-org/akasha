//! Background worker for candle inference jobs.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::Device;
use sqlx::SqlitePool;

use super::{loader, CandleModel, ModelOutput};

pub struct CandleWorker {
    pool: Arc<SqlitePool>,
    device: Device,
    resident: Option<Arc<dyn CandleModel>>,
    resident_config_id: Option<i64>,
}

impl CandleWorker {
    pub fn new(pool: Arc<SqlitePool>) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let device = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("CUDA device unavailable, falling back to CPU: {e}");
                Device::Cpu
            }
        };

        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;

        Ok(Self {
            pool,
            device,
            resident: None,
            resident_config_id: None,
        })
    }

    #[cfg(test)]
    pub fn set_resident(&mut self, model: Arc<dyn CandleModel>, config_id: i64) {
        self.resident = Some(model);
        self.resident_config_id = Some(config_id);
    }

    pub fn resident_config_id(&self) -> Option<i64> {
        self.resident_config_id
    }

    pub async fn process_jobs(&mut self, jobs: &[crate::db::searchable::JobRow]) -> Result<()> {
        for (i, job) in jobs.iter().enumerate() {
            if i > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
            tracing::info!(
                job_id = job.id,
                media_file_id = job.media_file_id,
                searchable_config_id = job.searchable_config_id,
                "CandleWorker: starting job"
            );
            match self.process_one(job).await {
                Ok(()) => tracing::info!(job_id = job.id, "CandleWorker: job completed"),
                Err(e) => {
                    tracing::warn!(job_id = job.id, error = %e, "CandleWorker: job failed");
                    let _ = crate::db::searchable::fail_job(&self.pool, job.id, &e.to_string()).await;
                }
            }
        }
        Ok(())
    }

    async fn process_one(&mut self, job: &crate::db::searchable::JobRow) -> Result<()> {
        let cfg = crate::db::searchable::get_config_by_id(
            &self.pool,
            job.searchable_config_id.unwrap_or(0),
        )
        .await?
        .context("missing searchable_config for job")?;

        // Load/replace model if needed. All model construction (file I/O,
        // memory-mapping, and ViT build) runs on the blocking thread pool so it
        // does not stall the async runtime.
        if self.resident_config_id != Some(cfg.id) {
            tracing::info!(model = cfg.name, "CandleWorker: loading model");
            let device = self.device.clone();
            let cfg_for_load = cfg.clone();
            self.resident = Some(
                tokio::task::spawn_blocking(move || load_model_for_config(&cfg_for_load, &device))
                    .await
                    .map_err(|e| anyhow::anyhow!("model loading task panicked: {e}"))?
                    .with_context(|| format!("failed to load model for {}", cfg.name))?,
            );
            self.resident_config_id = Some(cfg.id);
            tracing::info!(model = cfg.name, "CandleWorker: model loaded");
        }

        let model = self.resident.clone().unwrap();
        let media = crate::db::media::get_by_id(&self.pool, job.media_file_id)
            .await?
            .context("missing media file")?;
        let image_path = media.absolute_path.clone();

        let output = tokio::task::spawn_blocking(move || model.infer(Path::new(&image_path)))
            .await
            .map_err(|e| anyhow::anyhow!("inference task panicked: {e}"))??;

        match output {
            ModelOutput::Tags(tags) => {
                let count = tags.len();
                crate::db::searchable::update_tags_json(
                    &self.pool,
                    job.media_file_id,
                    &cfg.name,
                    tags,
                )
                .await?;
                tracing::info!(
                    job_id = job.id,
                    source = cfg.name,
                    tag_count = count,
                    "CandleWorker: wrote tags"
                );
            }
            ModelOutput::Description(text) => {
                crate::db::searchable::update_description_json(
                    &self.pool,
                    job.media_file_id,
                    &cfg.name,
                    &text,
                )
                .await?;
            }
            _ => {}
        }

        crate::db::searchable::complete_job(&self.pool, job.id).await?;
        Ok(())
    }
}

fn load_model_for_config(
    cfg: &crate::db::searchable::SearchableConfig,
    device: &Device,
) -> Result<Arc<dyn CandleModel>> {
    let options_value = cfg.options.clone();

    if options_value
        .get("base_url")
        .and_then(|v| v.as_str())
        .is_some()
    {
        anyhow::bail!("remote inference is not implemented in the candle worker");
    }

    let path = options_value
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or(&cfg.name);
    let source = loader::resolve_source(path)?;
    let files = loader::load_model_files(&source)
        .with_context(|| format!("failed to load model files for {}", cfg.name))?;

    match cfg.kind.as_str() {
        "tags" => {
            let options: crate::config::ModelTagsOptions =
                serde_json::from_value(options_value).unwrap_or_default();
            let tagger =
                super::tagger::ViTTagger::load(&cfg.name, &files, device.clone(), options.threshold)?;
            Ok(Arc::new(tagger))
        }
        other => anyhow::bail!("unsupported model kind: {other}"),
    }
}

#[cfg(all(test, feature = "candle"))]
mod tests {
    use std::sync::Arc;
    use sqlx::SqlitePool;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    #[tokio::test]
    async fn candle_worker_writes_tags() {
        let pool = setup_pool().await;
        let fid = crate::db::folder::insert(&pool, None, "/tmp", true, false, &[], &[], None, None, "disable").await.unwrap();
        let mid = crate::db::media::upsert(&pool, fid, "a.jpg", "/tmp/a.jpg", "hash", None, None, None, None, None).await.unwrap();

        let cfg_id = crate::db::searchable::upsert_config(&pool, "stub", "tags", true, serde_json::json!({"threshold":0.0})).await.unwrap();
        crate::db::searchable::enqueue_job(&pool, mid, "inference", "{}", Some(cfg_id)).await.unwrap();

        let jobs = crate::db::searchable::claim_pending_jobs(&pool, 10).await.unwrap();
        let mut worker = crate::models::worker::CandleWorker::new(Arc::new(pool.clone())).unwrap();
        // Override resident with stub for the test.
        worker.set_resident(Arc::new(crate::models::stub::StubTagger::new("stub")), cfg_id);
        worker.process_jobs(&jobs).await.unwrap();

        let row: (String,) = sqlx::query_as("SELECT tags_json FROM media_files WHERE id = ?1")
            .bind(mid).fetch_one(&pool).await.unwrap();
        assert!(row.0.contains("stub_tag"));

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM searchable_tags WHERE media_file_id = ?1")
                .bind(mid)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 1);
    }
}
