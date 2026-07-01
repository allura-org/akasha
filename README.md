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
(note that ONNX is currently not included in nightly builds because of a packaging headache)

### features
when building from source, akasha's default features require no extra dependencies besides a working C toolchain:

```bash
# Debian / Ubuntu
sudo apt install build-essential

# Fedora
sudo dnf install gcc make
```

for `cargo build --all-features`, you'll need `perl` (for OpenSSL), `libheif` and `libde265` (for HEVC), and CUDA Toolkit (for use ur imagination).

```bash
# Debian / Ubuntu
sudo apt install libheif-dev libde265-dev perl

# Fedora
sudo dnf install libheif-devel libde265-devel perl
```
(if i could give u a one-liner for CUDA on both families i'd have 2 more hours on my lifespan r/n)

#### remote AI inference

`remote`

remote (OpenAI-compatible) inference for classifiers (WIP) and descriptions, with `reqwest`.

#### SIMD thumbnail generation (non-rust)

`simd-thumbnails`

accelerates thumbnail generation with `libwebp`. compiled at build time (no external dependencies.)
requires a working C toolchain.

#### inference backends: `onnx`

`onnx`

enables local inference with `ort` for ONNX-format taggers. when `hf-hub` is enabled, parses .json files from the repo to configure the preprocessor, which is a Very Big Win.

#### inference backends: `candle`

`candle`

enables local inference using Hugging Face `candle`. currently this isn't well-supported for the implemented inference modes (taggers, VLMs where it straight-up isn't finished), but it's listed for the sake of completeness. support for customized models like JTP-3 is planned, and these will be implemented through `candle` primitives.

### optional features

these features require non-rust components. if u want optional features, use `--features "feature1 feature2..."`

#### huggingface model downloads

`hf-hub`

enables putting an HF slug into config.toml to automatically download models.
this uses statically-vendored `openssl` via `native-tls`, due to bugs with `rustls-tls`.
in addition to a working C toolchain you also need `perl`.

**dependencies:**
- `base-devel`/`gcc`+`make`
- `perl`

#### inference backends: mistralrs

`mistralrs`

enables VLMs to generate descriptions using mistral.rs.
mistral.rs itself *requires* `hf-hub`, and thus external dependencies, which is why it's not a default feature.

**dependencies**
- see `hf-hub`

#### CUDA acceleration

`cuda`

lets you generate tags/descriptions at speeds that *may* process your library *before* you shuffle off your mortal coil. (currently ONNX CUDA isn't wired up.)
requires CUDA toolkit installed. get that wherever proprietary compute libraries are sold.

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

## setup

`~/.config/akasha/config.toml` should be generated on first launch.

### import folders
modify the `[[imports]]` table array (including the `[imports.thumbnails]` tables) to your preferences. the defaults should be fine for most people.
you can copy these table arrays multiple times for multiple imports.

refer to `config.example.toml` in this repo for info on all the available options.
