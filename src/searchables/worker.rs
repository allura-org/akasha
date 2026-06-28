use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use sqlx::SqlitePool;
use tokio::time::interval;

use crate::models::{BackendRegistry, Model};

/// Background worker that polls the `job_queue` table and runs AI inference
/// through the backend-agnostic `BackendRegistry`.
pub struct SearchWorker {
    pool: Arc<SqlitePool>,
    batch_size: i64,
    registry: BackendRegistry,
    resident: Option<ResidentModel>,
}

struct ResidentModel {
    config_id: i64,
    backend_id: String,
    model: Arc<dyn Model>,
}

impl SearchWorker {
    pub fn new(pool: Arc<SqlitePool>) -> Self {
        Self {
            pool,
            batch_size: 4,
            registry: BackendRegistry::default(),
            resident: None,
        }
    }

    #[cfg(test)]
    pub fn with_registry(pool: Arc<SqlitePool>, registry: BackendRegistry) -> Self {
        Self {
            pool,
            batch_size: 4,
            registry,
            resident: None,
        }
    }

    pub async fn run(mut self) {
        let mut ticker = interval(Duration::from_secs(5));
        loop {
            ticker.tick().await;
            match self.tick().await {
                Ok(0) => {}
                Ok(n) => tracing::info!("SearchWorker processed {} jobs", n),
                Err(e) => tracing::warn!("SearchWorker error: {e}"),
            }
        }
    }

    async fn tick(&mut self) -> anyhow::Result<usize> {
        let jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
        if jobs.is_empty() {
            return Ok(0);
        }

        let ai_kinds = ["tagger", "classifier", "visionlanguage"];
        let (mut to_process, ignored): (Vec<_>, Vec<_>) = jobs
            .into_iter()
            .partition(|j| ai_kinds.contains(&j.job_kind.as_str()));

        for job in ignored {
            let _ = crate::db::searchable::fail_job(
                &self.pool,
                job.id,
                &format!("unknown job kind: {}", job.job_kind),
            )
            .await;
        }

        let count = to_process.len();
        if count == 0 {
            return Ok(0);
        }

        cluster_jobs(&mut to_process, self.resident.as_ref().map(|r| r.config_id));

        for job in &to_process {
            if let Err(e) = self.process_one(job).await {
                tracing::warn!(job_id = job.id, error = %e, "SearchWorker: job failed");
                let _ = crate::db::searchable::fail_job(&self.pool, job.id, &e.to_string()).await;
            }
        }

        Ok(count)
    }

    async fn process_one(&mut self, job: &crate::db::searchable::JobRow) -> anyhow::Result<()> {
        use std::path::Path;

        let cfg = crate::db::searchable::get_config_by_id(
            &self.pool,
            job.searchable_config_id.unwrap_or(0),
        )
        .await?
        .context("missing searchable_config for job")?;

        let model_config = crate::db::searchable::model_config_from_searchable_config(&cfg)?;
        let backend = self
            .registry
            .select(&model_config)
            .with_context(|| format!("no backend available for model {}", model_config.name))?;

        let backend_id = backend.id().to_string();
        let needs_load = self
            .resident
            .as_ref()
            .map(|r| r.config_id != cfg.id || r.backend_id != backend_id)
            .unwrap_or(true);

        if needs_load {
            tracing::info!(
                model = model_config.name,
                backend = backend_id,
                "SearchWorker: loading model"
            );
            let model = tokio::task::spawn_blocking({
                let backend = backend.clone();
                let model_config = model_config.clone();
                move || backend.load(&model_config)
            })
            .await
            .map_err(|e| anyhow::anyhow!("model loading task panicked: {e}"))??;
            self.resident = Some(ResidentModel {
                config_id: cfg.id,
                backend_id,
                model,
            });
            tracing::info!(model = model_config.name, "SearchWorker: model loaded");
        }

        let model = self.resident.as_ref().unwrap().model.clone();
        let media = crate::db::media::get_by_id(&self.pool, job.media_file_id)
            .await?
            .context("missing media file")?;
        let image_path = media.absolute_path.clone();

        let output = tokio::task::spawn_blocking(move || model.infer(Path::new(&image_path)))
            .await
            .map_err(|e| anyhow::anyhow!("inference task panicked: {e}"))??;

        match output {
            crate::models::ModelOutput::Tags(tags) => {
                crate::db::searchable::update_tags_json(
                    &self.pool,
                    job.media_file_id,
                    &cfg.name,
                    tags,
                )
                .await?;
            }
            crate::models::ModelOutput::Description(text) => {
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

/// Reorder claimed jobs so jobs sharing the same `searchable_config_id` are
/// grouped together. If a model is currently resident, its jobs are placed
/// first to minimize expensive reloads.
fn cluster_jobs(jobs: &mut [crate::db::searchable::JobRow], resident_config_id: Option<i64>) {
    jobs.sort_by_key(|j| {
        let config_id = j.searchable_config_id.unwrap_or(0);
        let resident_first = resident_config_id
            .map(|rid| if config_id == rid { 0 } else { 1 })
            .unwrap_or(1);
        (resident_first, config_id)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelKind, ModelTagsOptions};
    use crate::models::{Backend, Model, ModelOutput};
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;

    struct MockModel;
    impl Model for MockModel {
        fn infer(&self, _path: &Path) -> anyhow::Result<ModelOutput> {
            let mut tags = HashMap::new();
            tags.insert("mock_tag".to_string(), 0.99);
            Ok(ModelOutput::Tags(tags))
        }
    }

    struct MockBackend;
    impl Backend for MockBackend {
        fn id(&self) -> &'static str {
            "mock"
        }
        fn is_available(&self) -> bool {
            true
        }
        fn supports(&self, config: &ModelConfig) -> bool {
            config.backend.as_deref() == Some("mock")
        }
        fn load(&self, _config: &ModelConfig) -> anyhow::Result<Arc<dyn Model>> {
            Ok(Arc::new(MockModel))
        }
    }

    #[tokio::test]
    async fn search_worker_runs_mock_backend_job() {
        use crate::db;
        let pool = sqlx::SqlitePool::connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();

        let fid = db::folder::insert(&pool, None, "/tmp", true, false, &[], &[], None, None, "disable")
            .await
            .unwrap();
        let mid = db::media::upsert(
            &pool, fid, "a.jpg", "/tmp/a.jpg", "hash", None, None, None, None, None,
        )
        .await
        .unwrap();

        let cfg_id = db::searchable::upsert_config(
            &pool,
            "mock",
            "tags",
            true,
            serde_json::json!({"backend": "mock", "kind": "local", "threshold": 0.0}),
        )
        .await
        .unwrap();
        db::searchable::enqueue_job(&pool, mid, "tagger", "{}", Some(cfg_id))
            .await
            .unwrap();

        let mut reg = BackendRegistry::empty();
        reg.register(MockBackend);
        let mut worker = SearchWorker::with_registry(Arc::new(pool.clone()), reg);
        worker.tick().await.unwrap();

        let row: (String,) = sqlx::query_as("SELECT tags_json FROM media_files WHERE id = ?1")
            .bind(mid)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(row.0.contains("mock_tag"));
    }
}
