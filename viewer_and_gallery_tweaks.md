# Viewer & Gallery Tweaks — Backlog

## Viewer

### Buttons
- [ ] The buttons used to have icons that have since disappeared.
- [ ] The Fit/1:1 button currently changes size when you toggle it, pushing against the Next button, which is a bit of a UI crime. Should be a determinate size with the text centered within it.

### Info ticker
- [ ] Currently is nearly the same color as the background when in light mode.
- [ ] Currently overlaps the navbar when the window is narrow. There's a few ways to approach this:
    - Give it a background and stick it in another corner, so it doesn't clash with the gallery elements below (also solves the above)
    - Move the navbar and center the ticker
    - Place the ticker between the navbar and the image
    - Make the ticker's width reactive to the window so it retreats when the navbar moves over it (IMO probably too much faff)

### Close button
- [ ] Should be the same height as the navbar, keeping the two horizontally aligned.

## Gallery

### Thumbnail slider
- [ ] The thumbnail slider doesn't affect the size of the image previews, only the resolution of the thumbnails.
    - Maybe preview size should be a different slider? Or maybe it should replace it. With 3 images, this is fine, but with many, this could become a problem.
