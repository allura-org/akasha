use std::io::Write;
use std::process::{Command, Stdio};

/// Open the directory containing `path` in the user's default file manager.
pub fn open_containing_folder(path: &str) -> anyhow::Result<()> {
    let p = std::path::Path::new(path);
    let dir = p
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("/"));

    Command::new("xdg-open").arg(dir).spawn()?;
    Ok(())
}

/// Copy the image file at `path` to the system clipboard as a file reference.
/// Uses `text/uri-list` so the receiving application reads the original file
/// from disk, preserving its name, extension, and encoding. Uses `wl-copy` on
/// Wayland and `xclip` on X11.
pub fn copy_image_to_clipboard(path: &str) -> anyhow::Result<()> {
    let uri = file_uri(path)?;
    // text/uri-list entries are CRLF-terminated.
    let uri_list = format!("{uri}\r\n");
    let bytes = uri_list.into_bytes();

    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        let mut child = Command::new("wl-copy")
            .args(["--type", "text/uri-list"])
            .stdin(Stdio::piped())
            .spawn()?;
        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("failed to open wl-copy stdin"))?;
            stdin.write_all(&bytes)?;
        }
    } else {
        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard", "-t", "text/uri-list"])
            .stdin(Stdio::piped())
            .spawn()?;
        {
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| anyhow::anyhow!("failed to open xclip stdin"))?;
            stdin.write_all(&bytes)?;
        }
    }

    Ok(())
}

fn file_uri(path: &str) -> anyhow::Result<String> {
    url::Url::from_file_path(path)
        .map(|u| u.to_string())
        .map_err(|_| anyhow::anyhow!("invalid file path: {path}"))
}
