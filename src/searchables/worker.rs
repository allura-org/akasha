use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;
use tokio::time::interval;

/// Background worker that polls the `job_queue` table.
///
/// AI jobs are currently processed as no-ops (log + mark done) while the
/// backend-agnostic inference path is being refactored.
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

    async fn process_ai_job(&self, job: &crate::db::searchable::JobRow) -> anyhow::Result<()> {
        let model_name: String = job
            .params_json
            .as_deref()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .and_then(|v| v.get("model_name").and_then(|m| m.as_str()).map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        tracing::info!(
            "SearchWorker: failing {} job {} for media {} (model: {}) — backend-registry integration pending (Task 5)",
            job.job_kind,
            job.id,
            job.media_file_id,
            model_name
        );

        anyhow::bail!(
            "AI inference is temporarily disabled; backend-registry integration is pending (Task 5)."
        )
    }
}
