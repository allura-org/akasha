# akasha - a Linux-native image gallery for data hoarders

<img width="1692" height="1356" alt="image" src="https://github.com/user-attachments/assets/487cffaf-4d05-42c4-8dc2-2833763971bb" />

akasha (ah-kuh-sha) is a modern, performance-forward, database-backed image gallery written in Rust with (currently) egui-based GUI. Built to handle hundreds of thousands of images in a single import.

⚠️  akasha is currently in ALPHA. Most of its features are present, but it is still in heavy flux and the biggest lifts are yet to come!

## features

- native GUI with Rust+egui (for now...) no webjank.
- opens in one click. configured in TOML. optimized for desktops, not homelabs.
- simplistic, comfortable, content-over-chrome design.
- fast as hell, hundreds of images index per second on a LUKS2-encrypted SATA SSD.
- media stays where it is, no duplication. akasha gets you to the content you want, it doesn't demand control over it.
- updates to existing folders are processed automatically
- backwards-compatible schema means no database rebuilds. migration scripts provided for breaking changes.

## planned

- killer feature: modular AI classification via a "Searchables" schema abstraction that combines classifiers, taggers, embedding models, VLMs, under one roof
- ~~gallery-dl integration~~ being moved to another project :3
- videos
- remuxing, transcoding
- interaction API
- more skinnable UI

## generative AI disclosure

akasha was coded primarily by an LLM called Kimi (2.6, 2.7). through its creation, it was actively directed and observed, as an exercise in learning about databases and Rust. it's primarily made for the use of its author and shared in hopes it'll be useful. akasha's design principles mean it shouldn't be of any harm to your media (it currently doesn't even have any means of manipulation that Could be harmful), but it's regardless a good idea to **keep backups.**

## rationale

my collection stretches to nearly half a million images, accrued over nearly a decade. browsing it via file manager is a non-starter. i Need semantic search.

every image gallery sucks. half of them are giant docker containers meant to reimplement the entirety of google photos. hydrus network is a respectable project but it's an absolute behemoth that demands i put too much trust in it, and i see no reason for it to be as complex as it is for what it does.

and even among the solutions that Let you use an AI model for content search, they use piddly models from the stone ages that are dogwater at recognizing the things I'm actually interested in.
so there is an absolute dearth of appropriate solutions for creatures like me.

thankfully we live in the era of vibe-coding. so i slopped harder than ive ever slopped before :3

## install

clone it and `cargo build` homeboy. (or grab the latest nightly over [hyea](https://github.com/allura-org/akasha/releases/tag/nightly))

### extra features

when building from source, akasha is pure-rust by default for ease of setup, but some features require non-rust libraries.
if u want an optional feature, use `--features <feature>` once for each.

#### HEVC support

`hevc`

HEIC images, H.265 in MP4 (eventually), etc. HEVC is patent-encumbered, so instead of shipping a decoder ourselves we rely on system libraries.

**dependencies:**

- `libheif` >= 1.17.0
- `libde265` (for HEVC decoding)

```bash
# Debian / Ubuntu
sudo apt install libheif-dev libde265-dev

# Fedora
sudo dnf install libheif-devel libde265-devel
```

#### SIMD thumbnail generation

`simd-thumbnails`

accelerates thumbnail generation.

**dependencies:**

- `libwebp`

```bash
# Debian / Ubuntu
sudo apt install libwebp-dev

# Fedora
sudo dnf install libwebp-devel
```

#### Local AI inference (candle)

`candle`

enables local inference using Hugging Face `candle`. Models are downloaded on first use via `hf-hub` and cached in `$HF_HOME` (default `~/.cache/huggingface`). Add a `[[models]]` entry to `config.toml` with `type = "local"` and the relevant output subtable (`[models.tags]`, `[models.description]`, etc.).

```bash
cargo build --features candle
# or with CUDA support:
cargo build --features candle,cuda
```

**notes:**

- Local CPU inference can be very slow on large collections and only runs while Akasha is open.
- The popular `SmilingWolf/wd-vit-tagger-v3` checkpoint uses a `timm`-style model config and is not directly compatible with `candle_transformers::models::vit`; use standard Hugging Face ViT checkpoints (e.g. `google/vit-base-patch16-224`) for the current scaffold.

## setup

`~.config/akasha/config.toml` should be generated on first launch.
modify the `[[import]]` table array (including the `[import.thumbnails]` tables) to your preferences.
you can copy these table arrays multiple times for multiple imports.

refer to `config.example.toml` in this repo for info on all the available options.
