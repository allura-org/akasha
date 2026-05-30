# Akasha a.k.a. Yet Another Image Gallery

A simple database-backed image gallery with AI classification and search, focused on getting data hoarders to the things they want to look at.

- Fast, Linux-native; optimized for desktops, not homelabs
- Open in one click, human-readable configuration format

- Add folders to watch, make them recursive or not
- Browse library by folder tree
- Keep media where it is, use hashes to avoid duplicates within the app
- Modular classification with a "Searchables" abstraction
	- Use whatever model you want with ONNX; typical classifiers/embedding models/VLMs/whatever.
	- If it takes image/video in and spits out some way to search for that image/video, it ought to work
	- Classifications, vectors, descriptions, tags, etc. are stored as Searchables, and all can be searched simultaneously or turned on/off independently (may be limited by memory impact of retrieval? no reason text-based Searchables can't be pooled tho IME)
	- Text-based Searchables can be concatenated and turned into vectors
- Extensible, modular, backwards-compatible schema
