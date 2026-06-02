# akasha - a Linux-native image gallery for data hoarders

<img width="1692" height="1356" alt="image" src="https://github.com/user-attachments/assets/487cffaf-4d05-42c4-8dc2-2833763971bb" />

akasha (ah-kuh-sha) is a modern, performance-forward, database-backed image gallery written in Rust with (currently) egui-based GUI. Built to handle hundreds of thousands of images in a single import.

⚠️  akasha is in pre-alpha and is not suitable to daily-drive yet.

## features

- native GUI with Rust+egui (for now...) no webjank.
- opens in one click. configured in TOML. optimized for desktops, not homelabs.
- simplistic, comfortable, content-over-chrome design.
- fast as hell, hundreds of images index per second on a LUKS2-encrypted SATA SSD.
- media stays where it is, no duplication. akasha gets you to the content you want, it doesn't demand control over it.
- backwards-compatible schema means no database rebuilds. migration scripts provided for breaking changes.

## planned

- killer feature: modular AI classification via a "Searchables" schema abstraction that combines classifiers, taggers, embedding models, VLMs, under one roof
- file watcher
- gallery-dl integration
- videos
- remuxing, transcoding
- interaction API
- more skinnable UI

## generative AI disclosure

akasha was coded primarily by an LLM called Kimi 2.6. through its creation, it was actively directed and observed, as an exercise in learning about databases and Rust. it's primarily made for the use of its author and shared in hopes it'll be useful. akasha's design principles mean it shouldn't be of any harm to your media (it currently doesn't even have any means of manipulation that Could be harmful), but it's regardless a good idea to **keep backups.**

## rationale

my collection stretches to nearly half a million images, accrued over nearly a decade. browsing it via file manager is a non-starter. i Need semantic search.

every image gallery sucks. half of them are giant docker containers meant to reimplement the entirety of google photos. hydrus network is a respectable project but it's an absolute behemoth that demands i put too much trust in it, and i see no reason for it to be as complex as it is for what it does.

there is an absolute dearth of appropriate solutions for creatures like me.

thankfully we live in the era of vibe-coding. so i slopped harder than ive ever slopped before :3

## install

clone it and `cargo build` homeboy.

### extra features

akasha is pure-rust by default for ease of setup, but some features require non-rust libraries.
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

## setup

`~.config/akasha/config.toml` should be generated on first launch. modify the `[[folders]]` section, and copy it for every directory you want akasha to scan.

each `[[folders]]` section is an import. when `recursive = true`, its children will be scanned, except for any folders with names in the blacklist string array. when `show_recursive = true`, the import will show images found in the root folder *and* all children recursively.

when `thumbnails.cache_mode` is "global", thumbnails are cached to `~/.cache/akasha`
when `thumbnails.cache_mode` is `custom`, you change where thumbnails are cached to.
when `thumbnails.cache_mode` is `per_folder`, thumbnails are cached alongside media. (WIP)
when `thumbnails.cache_mode` is `disabled`, no thumbnails are cached.

thats all the setup that exists rn. thumbnail sizes and theme can be changed in the app
