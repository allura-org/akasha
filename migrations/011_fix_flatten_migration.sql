-- Migration 010 flipped show_recursive into flatten, which was incorrect:
-- old show_recursive=false (tree view, direct files) became flatten=true (flat folder).
-- Reset flatten to match the original show_recursive value.
UPDATE folders SET flatten = show_recursive;
