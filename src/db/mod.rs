use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::PathBuf;

pub mod folder;
pub mod media;
pub mod searchable;

pub async fn init_pool(db_path: PathBuf) -> anyhow::Result<SqlitePool> {
    if let Some(parent) = db_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    tracing::info!("Connecting to database at: {}", db_path.display());

    let options = SqliteConnectOptions::new()
        .filename(&db_path)
        .create_if_missing(true)
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        // WAL + NORMAL only fsyncs at checkpoints: commits stay durable
        // across process crashes, just not OS/power loss. This is the
        // standard WAL trade-off and keeps bulk commits cheap.
        .synchronous(sqlx::sqlite::SqliteSynchronous::Normal)
        .busy_timeout(std::time::Duration::from_secs(5));

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await?;

    sqlx::migrate!()
        .run(&pool)
        .await?;

    Ok(pool)
}

/// A `BEGIN IMMEDIATE` transaction on a pooled SQLite connection.
///
/// sqlx's `begin()` issues a deferred `BEGIN`: the write lock is only taken
/// at the first write statement, so a transaction that reads before it writes
/// can fail with `SQLITE_BUSY_SNAPSHOT` ("database is locked") when another
/// writer commits in between — an error `busy_timeout` does not cover.
/// `BEGIN IMMEDIATE` acquires the write lock up front, where `busy_timeout`
/// does apply, so concurrent writers queue instead of erroring.
///
/// If dropped without `commit()`, the transaction is rolled back before the
/// connection returns to the pool.
pub struct ImmediateTx {
    conn: Option<sqlx::pool::PoolConnection<sqlx::Sqlite>>,
}

impl ImmediateTx {
    pub async fn begin(pool: &SqlitePool) -> anyhow::Result<Self> {
        let mut conn = pool.acquire().await?;
        sqlx::query("BEGIN IMMEDIATE").execute(&mut *conn).await?;
        Ok(Self { conn: Some(conn) })
    }

    pub async fn commit(mut self) -> anyhow::Result<()> {
        let mut conn = self.conn.take().expect("transaction already finished");
        sqlx::query("COMMIT").execute(&mut *conn).await?;
        Ok(())
    }
}

impl std::ops::Deref for ImmediateTx {
    type Target = sqlx::SqliteConnection;
    fn deref(&self) -> &Self::Target {
        self.conn.as_deref().expect("transaction already finished")
    }
}

impl std::ops::DerefMut for ImmediateTx {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.conn.as_deref_mut().expect("transaction already finished")
    }
}

impl Drop for ImmediateTx {
    fn drop(&mut self) {
        if let Some(mut conn) = self.conn.take() {
            // Only reachable from async contexts (the pool is a tokio pool),
            // so a runtime handle is always available here.
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                handle.spawn(async move {
                    let _ = sqlx::query("ROLLBACK").execute(&mut *conn).await;
                });
            }
        }
    }
}
