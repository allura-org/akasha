-- Fix imports that were incorrectly flattened by migration 010.
-- Old `show_recursive = false` should map to `flatten = false` (show tree),
-- and old `show_recursive = true` should map to `flatten = true` (show as one folder).
UPDATE folders SET flatten = show_recursive;
