## Summary

Add an optional `mistralrs` backend for local vision-language model (VLM) description jobs. This lets Akasha run multimodal captioning models (e.g., `google/gemma-4-E2B-it`, `Qwen/Qwen3-VL-2B-Instruct`) through `mistral.rs` instead of Candle, because Candle does not yet support the Gemma 4 architecture.

## What changed

- Added optional `mistralrs` Cargo feature and `mistralrs = "0.8.1"` dependency.
- Implemented `MistralRsBackend` / `MistralRsModel` in `src/models/mistralrs.rs`:
  - Bridges `mistral.rs`'s async `MultimodalModelBuilder` to the synchronous `Model::infer` trait by keeping a `tokio::runtime::Handle` (not a full `Runtime`), avoiding the nested-runtime drop panic seen when switching models.
  - Supports configurable in-situ quantization via the new `isq` field in `ModelDescriptionOptions` (default `"Q8_0"`; `"none"` disables).
  - Requires an explicit `backend = "mistralrs"` config value so it does not accidentally claim Candle or Remote description models.
- Registered `MistralRsBackend` in `BackendRegistry`.
- Improved `SearchWorker` robustness:
  - Deletes existing predictions only after successful inference, preventing overwrite-enabled jobs from wiping data on failure.
  - Fails the job if tags or description outputs are empty/whitespace.
  - Resets any jobs left in `running` status to `failed` on startup (crash cleanup, no auto-resume).
- Made the Properties window refresh every 2 seconds while open so inference results appear without closing/reopening.
- Unified all HTTP/TLS usage on `native-tls` (`hf-hub`, `reqwest`, `ort`) to avoid `rustls` TLS `close_notify` and truncated-response failures when downloading large model files from Hugging Face.
- Added `scripts/regenerate_imports.py` to rebuild `[[imports]]` config blocks from the database after the config key fix.
- Added `run-gpu.sh` as a local test helper for the 32-bit `/usr/lib/libcuda.so` linker workaround (gitignored, not shipped).
- Updated `AGENTS.md` and `config.example.toml` to document the new backend and `isq` option.

## Testing

- `cargo check --all-features` passes.
- `cargo test --features "hevc simd-thumbnails candle remote onnx mistralrs"` passes: 67 passed, 0 failed, 4 ignored.
- Verified end-to-end on GPU: `google/gemma-4-E2B-it` and `Qwen/Qwen3-VL-2B-Instruct` load via `mistralrs` and produce descriptions.

## Known limitations

- `mistral.rs` handles its own HuggingFace downloads; on some networks large model files may fail with truncated-body errors and need a retry/re-queue.
- `mistralrs` models must specify `backend = "mistralrs"` explicitly.
- Models still need to fit in VRAM; quantization via `isq` helps but very large VLMs will need a bigger card or CPU fallback.
