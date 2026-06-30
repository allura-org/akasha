## Summary

Adds end-to-end support for vision-language model (VLM) description jobs inside the Candle backend, so visionlanguage / description AI jobs can run against local multimodal checkpoints. The first supported architecture is google/gemma-4-E2B-it.

## What changed

- Upgraded candle-core, candle-nn, and candle-transformers to 0.11.0; added optional tokenizers = "0.22" under the candle feature.
- Extended ModelFiles / model loader to discover tokenizer.json and sharded model.safetensors.index.json files.
- Added generation controls to ModelDescriptionOptions (max_tokens, temperature, top_p, top_k, repeat_penalty, repeat_last_n, plus optional image normalization overrides).
- Introduced VlmModel / VlmArchitecture traits and VlmArchitectureRegistry, mirroring the existing backend registry pattern.
- Implemented Gemma4Vlm with image preprocessing aligned to the official processor_config.json (rescale 1/255, Bicubic resize, 280 image tokens) and a manual prompt tokenization fallback.
- Added a BLIP stub architecture for future expansion.
- Wired the VLM registry into CandleBackend::load via VlmModelWrapper; description jobs now produce ModelOutput::Description through the existing SearchWorker pipeline.
- Added integration and unit tests, including mock VLM description job coverage.

## Testing

- cargo test passes: 66 passed, 0 failed, 4 ignored.

## Known limitations

- The tokenizer fallback is hand-rolled because tokenizers 0.22 does not expose apply_chat_template.
- Architecture selection currently relies on model name/path heuristics; explicit candle-* backend hints are not fully wired.
- Real inference against the full Gemma 4 checkpoint has not been smoke-tested yet.
