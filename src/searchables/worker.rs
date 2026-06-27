use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;
use tokio::time::interval;

/// Background worker that polls the `job_queue` table.
///
/// When the `candle` feature is enabled, the worker owns a resident
/// `CandleWorker` and reuses it across ticks. Jobs are clustered by
/// `searchable_config_id` so the same model stays loaded for consecutive
/// jobs. Without `candle`, AI jobs are processed as no-ops (log + mark done).
pub struct SearchWorker {
    pool: Arc<SqlitePool>,
    batch_size: i64,
    #[cfg(feature = "candle")]
    candle: Option<crate::models::worker::CandleWorker>,
}

impl SearchWorker {
    pub fn new(pool: Arc<SqlitePool>) -> Self {
        Self {
            pool,
            batch_size: 4,
            #[cfg(feature = "candle")]
            candle: None,
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

    #[cfg(feature = "candle")]
    async fn tick(&mut self) -> anyhow::Result<usize> {
        let mut jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
        let count = jobs.len();
        if count == 0 {
            return Ok(0);
        }

        let resident_id = self.candle.as_ref().and_then(|c| c.resident_config_id());
        cluster_jobs(&mut jobs, resident_id);

        if self.candle.is_none() {
            match crate::models::worker::CandleWorker::new(Arc::clone(&self.pool)) {
                Ok(c) => self.candle = Some(c),
                Err(e) => {
                    for job in &jobs {
                        let _ = crate::db::searchable::fail_job(&self.pool, job.id, &e.to_string()).await;
                    }
                    return Ok(count);
                }
            }
        }

        let candle = self.candle.as_mut().unwrap();
        candle.process_jobs(&jobs).await?;
        Ok(count)
    }

    #[cfg(not(feature = "candle"))]
    async fn tick(&mut self) -> anyhow::Result<usize> {
        let jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
        let count = jobs.len();
        for (i, job) in jobs.iter().enumerate() {
            if i > 0 {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            match job.job_kind.as_str() {
                "tagger" | "classifier" | "visionlanguage" => {
                    if let Err(e) = self.process_ai_job(job).await {
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

#[cfg(feature = "candle")]
fn cluster_jobs(jobs: &mut [crate::db::searchable::JobRow], resident_id: Option<i64>) {
    jobs.sort_by_key(|j| {
        let is_resident = resident_id == j.searchable_config_id;
        (!is_resident, j.searchable_config_id, j.created_at)
    });
}
