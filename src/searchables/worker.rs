use std::sync::Arc;
use std::time::Duration;

use sqlx::SqlitePool;
use tokio::time::interval;

/// Background worker stub that polls the `job_queue` table.
///
/// For the first Searchables milestone this simply claims pending jobs and
/// marks them `done` immediately. Future milestones will replace this with
/// real ONNX inference for tags, embeddings, and classifications.
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
                Ok(n) => tracing::info!("SearchWorker processed {} stub jobs", n),
                Err(e) => tracing::warn!("SearchWorker error: {e}"),
            }
        }
    }

    async fn tick(&self) -> anyhow::Result<usize> {
        let jobs = crate::db::searchable::claim_pending_jobs(&self.pool, self.batch_size).await?;
        let count = jobs.len();
        for job in jobs {
            tracing::info!(
                "SearchWorker: would process job {} for media {} / config {}",
                job.id,
                job.media_file_id,
                job.searchable_config_id
            );
            crate::db::searchable::complete_job(&self.pool, job.id).await?;
        }
        Ok(count)
    }
}
