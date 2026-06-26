use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use notify_debouncer_full::{
    new_debouncer, notify::RecursiveMode, DebouncedEvent, Debouncer as FullDebouncer,
};
use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::config::ImportConfig;
use crate::scanner::is_supported;

/// A single filesystem change the watcher wants the app to apply.
#[derive(Debug, Clone)]
pub struct WatcherChange {
    pub absolute_path: PathBuf,
    pub kind: WatcherChangeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatcherChangeKind {
    /// File was created or modified; upsert it into the DB.
    Upsert,
    /// File was removed; delete it from the DB.
    Remove,
}

/// Events emitted by the watcher task to `AkashaApp`.
#[derive(Debug, Clone)]
pub enum WatcherEvent {
    Changed(Vec<WatcherChange>),
    Error(String),
}

/// Opaque handle that keeps the debouncer alive. Dropping it stops the watcher.
pub struct WatcherHandle {
    _debouncer: Box<dyn std::any::Any + Send>,
}

/// Spawn a debounced filesystem watcher for every configured folder.
///
/// Returns a `WatcherHandle` and the receiver for `WatcherEvent`s. The watcher
/// stops when the handle is dropped.
pub fn spawn(
    _pool: Arc<SqlitePool>,
    imports: Vec<ImportConfig>,
) -> anyhow::Result<(WatcherHandle, Receiver<WatcherEvent>)> {
    let (app_tx, app_rx) = channel::<WatcherEvent>();
    let (internal_tx, internal_rx) = channel::<WatcherEvent>();

    let mut debouncer: FullDebouncer<
        notify_debouncer_full::notify::RecommendedWatcher,
        notify_debouncer_full::NoCache,
    > = new_debouncer(
        Duration::from_millis(500),
        None,
        make_handler(internal_tx.clone()),
    )?;

    for import in imports {
        let path = PathBuf::from(&import.path);
        let mode = if import.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        match debouncer.watch(&path, mode) {
            Ok(_) => info!("Watching folder: {}", path.display()),
            Err(e) => {
                warn!("Failed to watch {}: {}", path.display(), e);
                let _ = internal_tx.send(WatcherEvent::Error(format!(
                    "Failed to watch {}: {}",
                    path.display(),
                    e
                )));
            }
        }
    }

    // Forward internal events to the app channel. The thread exits when the
    // debouncer (and therefore internal_tx) is dropped.
    std::thread::spawn(move || {
        while let Ok(event) = internal_rx.recv() {
            if app_tx.send(event).is_err() {
                break;
            }
        }
    });

    let handle = WatcherHandle {
        _debouncer: Box::new(debouncer),
    };

    Ok((handle, app_rx))
}

fn make_handler(
    tx: Sender<WatcherEvent>,
) -> impl Fn(notify_debouncer_full::DebounceEventResult) + Send + 'static {
    move |result: notify_debouncer_full::DebounceEventResult| {
        match result {
            Ok(events) => {
                let changes = classify_events(events);
                if !changes.is_empty() {
                    let _ = tx.send(WatcherEvent::Changed(changes));
                }
            }
            Err(errors) => {
                for err in errors {
                    let _ = tx.send(WatcherEvent::Error(format!("Watcher error: {err}")));
                }
            }
        }
    }
}

/// Collapse a batch of debounced notify events into a set of per-path changes.
fn classify_events(events: Vec<DebouncedEvent>) -> Vec<WatcherChange> {
    // Map absolute path -> kind. Remove wins over Upsert.
    let mut by_path: HashMap<PathBuf, WatcherChangeKind> = HashMap::new();

    for event in events {
        for path in &event.paths {
            // Skip directories and unsupported files early.
            if path.is_dir() {
                continue;
            }
            if path.is_file() && !is_supported(path) {
                continue;
            }

            use notify_debouncer_full::notify::EventKind::*;
            let kind = match &event.event.kind {
                Create(_) if !path.exists() => WatcherChangeKind::Remove,
                Create(_) => WatcherChangeKind::Upsert,
                Modify(_) if !path.exists() => WatcherChangeKind::Remove,
                Modify(_) => WatcherChangeKind::Upsert,
                Remove(_) => WatcherChangeKind::Remove,
                // Access events are ignored — they fire when a file is merely read
                // (e.g. thumbnail generation) and must not trigger re-upserts.
                Access(_) => continue,
                Any | Other => {
                    // For ambiguous events, check whether the file still exists on disk.
                    if path.exists() {
                        WatcherChangeKind::Upsert
                    } else {
                        WatcherChangeKind::Remove
                    }
                }
            };

            match by_path.get(path) {
                // Once a path is marked Remove, it stays Remove for this batch.
                Some(WatcherChangeKind::Remove) => {}
                _ => {
                    by_path.insert(path.clone(), kind);
                }
            }
        }
    }

    by_path
        .into_iter()
        .map(|(absolute_path, kind)| WatcherChange { absolute_path, kind })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "akasha_watcher_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn detects_file_creation() {
        let dir = temp_dir();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let cfg = ImportConfig {
            path: dir.to_string_lossy().to_string(),
            recursive: true,
            flatten: true,
            exclude: Vec::new(),
            include: Vec::new(),
            thumbnails: crate::config::ImportThumbnailsConfig::default(),
        };

        let pool = Arc::new(tokio::runtime::Runtime::new().unwrap().block_on(async {
            let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
            sqlx::migrate!("./migrations").run(&pool).await.unwrap();
            pool
        }));

        let (_handle, rx) = spawn(pool, vec![cfg]).unwrap();

        // Create a small PNG file.
        let test_file = dir.join("test.png");
        {
            let mut f = std::fs::File::create(&test_file).unwrap();
            // Minimal 1x1 PNG.
            let png: &[u8] = &[
                0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
                0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
                0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8, 0x0f, 0x00, 0x00,
                0x01, 0x01, 0x00, 0x05, 0x18, 0xd8, 0x4e, 0x00, 0x00, 0x00, 0x00, 0x49,
                0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
            ];
            f.write_all(png).unwrap();
        }

        // Wait for debounce + notification.
        std::thread::sleep(Duration::from_millis(1200));

        let mut found = false;
        while let Ok(event) = rx.try_recv() {
            if let WatcherEvent::Changed(changes) = event {
                if changes.iter().any(|c| c.absolute_path == test_file) {
                    found = true;
                }
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
        assert!(found, "expected watcher to detect created file");
    }

    #[test]
    fn detects_file_deletion() {
        let dir = temp_dir();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let cfg = ImportConfig {
            path: dir.to_string_lossy().to_string(),
            recursive: true,
            flatten: true,
            exclude: Vec::new(),
            include: Vec::new(),
            thumbnails: crate::config::ImportThumbnailsConfig::default(),
        };

        let pool = Arc::new(tokio::runtime::Runtime::new().unwrap().block_on(async {
            let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
            sqlx::migrate!("./migrations").run(&pool).await.unwrap();
            pool
        }));

        let (_handle, rx) = spawn(pool, vec![cfg]).unwrap();

        let test_file = dir.join("test.png");
        {
            let mut f = std::fs::File::create(&test_file).unwrap();
            let png: &[u8] = &[
                0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
                0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
                0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8, 0x0f, 0x00, 0x00,
                0x01, 0x01, 0x00, 0x05, 0x18, 0xd8, 0x4e, 0x00, 0x00, 0x00, 0x00, 0x49,
                0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
            ];
            f.write_all(png).unwrap();
        }

        // Wait for the creation event to be emitted.
        std::thread::sleep(Duration::from_millis(1200));
        while rx.try_recv().is_ok() {}

        // Delete the file and wait for the watcher to report it.
        std::fs::remove_file(&test_file).unwrap();
        std::thread::sleep(Duration::from_millis(1200));

        let mut found_remove = false;
        while let Ok(event) = rx.try_recv() {
            if let WatcherEvent::Changed(changes) = event {
                if changes.iter().any(|c| {
                    c.absolute_path == test_file && c.kind == WatcherChangeKind::Remove
                }) {
                    found_remove = true;
                }
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
        assert!(found_remove, "expected watcher to detect deleted file as Remove");
    }
}
