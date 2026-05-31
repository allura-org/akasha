I have a brain-dump of this concept:

# Akasha a.k.a. Yet Another Image Gallery

A simple database-backed image gallery with AI classification and search, focused on getting data hoarders to the things they want to look at.

- Fast, Linux-native; optimized for desktops, not homelabs
- Open in one click, human-readable configuration format
- Simplistic, comfortable design language, put user content front and center
- Add folders to import, make them recursive or not, blacklist directories
- "Watch" folders for new additions
- Browse library by folder tree with each imported folder as a separate root
- Keep media where it is, use hashes to avoid duplicates within the app
- Modular classification with a "Searchables" abstraction
    - Use whatever model you want with ONNX; typical classifiers/embedding models/VLMs/whatever.
    - If it takes image/video in and spits out some way to search for that image/video, it ought to work
    - Classifications, vectors, descriptions, tags, etc. are stored as Searchables, and all can be searched simultaneously or turned on/off independently (may be limited by memory impact of retrieval? no reason text-based Searchables can't be pooled tho, I think?)
    - maybe a "meta-Searchable" that concatenates and vectorizes text-based Searchables? that way you can have one Searchable for all your text-based forms? might be too complex
- Extensible, modular design with backwards-compatible schema (in future i want to add things like a gallery-dl downloader, remuxer/transcoder, interaction API, but all that is distant atm)
