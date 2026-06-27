use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;
use tokio::time::interval;

/// Background worker that polls the `job_queue` table.
///
/// For now this only dispatches AI inference jobs as dummy work (log + sleep).
/// Real ONNX/remote inference will be plugged into `process_ai_job` later.
pub struct SearchWorker {
    pool: Arc<SqlitePool>,
    batch_size: i64,
}

impl SearchWorker {
    pub fn new(pool: Arc<SqlitePool>) -> Self {
        Self {
            pool,
            batch_size: 4,
        }
    }

    pub async fn run(self) {
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

    #[cfg(feature = "candle")]
    async fn tick(&self) -> anyhow::Result<usize> {
        let jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
        let count = jobs.len();
        if count == 0 {
            return Ok(0);
        }

        let mut candle = crate::models::worker::CandleWorker::new(Arc::clone(&self.pool))?;
        candle.process_jobs(&jobs).await?;
        Ok(count)
    }

    #[cfg(not(feature = "candle"))]
    async fn tick(&self) -> anyhow::Result<usize> {
        let jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
        let count = jobs.len();
        for job in jobs {
            match job.job_kind.as_str() {
                "tagger" | "classifier" | "visionlanguage" => {
                    if let Err(e) = self.process_ai_job(&job).await {
                        let _ = crate::db::searchable::fail_job(&self.pool, job.id, &e.to_string()).await;
                    }
                }
                other => {
                    tracing::warn!("SearchWorker: unknown job kind '{}' for job {}", other, job.id);
                    let _ = crate::db::searchable::fail_job(
                        &self.pool,
                        job.id,
                        &format!("unknown job kind: {}", other),
                    ).await;
                }
            }
        }
        Ok(count)
    }

    #[cfg(not(feature = "candle"))]
    async fn process_ai_job(&self, job: &crate::db::searchable::JobRow) -> anyhow::Result<()> {
        let model_name: String = job
            .params_json
            .as_deref()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .and_then(|v| v.get("model_name").and_then(|m| m.as_str()).map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        tracing::info!(
            "SearchWorker: running {} job {} for media {} (model: {})",
            job.job_kind,
            job.id,
            job.media_file_id,
            model_name
        );

        // Dummy work: pretend inference takes a moment.
        tokio::time::sleep(Duration::from_millis(50)).await;

        tracing::info!(
            "SearchWorker: completed {} job {} for media {} (model: {})",
            job.job_kind,
            job.id,
            job.media_file_id,
            model_name
        );

        crate::db::searchable::complete_job(&self.pool, job.id).await?;
        Ok(())
    }
}
