#!/usr/bin/env python3
"""Regenerate [[imports]] TOML blocks from Akasha's folder table.

Run with no args to print to stdout, or pass an output file:

    python3 scripts/regenerate_imports.py ~/.config/akasha/regenerated_imports.toml

Then copy the printed/created blocks into ~/.config/akasha/config.toml.
"""
import json
import os
import sqlite3
import sys

DB_PATH = os.path.expanduser("~/.local/share/akasha/akasha.db")


def toml_str_array(items: list[str]) -> str:
    """Format a list of strings as a TOML array."""
    return json.dumps(items)


def main() -> int:
    if not os.path.exists(DB_PATH):
        print(f"Database not found: {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Root folders are the configured imports.
    rows = cur.execute(
        """
        SELECT path, recursive, exclude, include,
               thumbnail_cache_mode, thumbnail_cache_folder, thumbnail_cache_fallback
        FROM folders
        WHERE parent_id IS NULL
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        print("No root imports found in the database.", file=sys.stderr)
        return 0

    lines: list[str] = []
    for row in rows:
        exclude = json.loads(row["exclude"] or "[]")
        include = json.loads(row["include"] or "[]")
        recursive = bool(row["recursive"])

        lines.append("[[imports]]")
        lines.append(f'path = {json.dumps(row["path"])}')
        lines.append(f"recursive = {str(recursive).lower()}")
        # `flatten` is not currently persisted in the DB; default to false.
        lines.append("flatten = false")
        lines.append(f"exclude = {toml_str_array(exclude)}")
        lines.append(f"include = {toml_str_array(include)}")

        cache_mode = row["thumbnail_cache_mode"]
        cache_folder = row["thumbnail_cache_folder"]
        cache_fallback = row["thumbnail_cache_fallback"]

        # Only emit the thumbnails subtable if any non-default value is set.
        if cache_mode or cache_folder or cache_fallback != "disable":
            lines.append("")
            lines.append("[imports.thumbnails]")
            if cache_mode:
                lines.append(f'cache_mode = {json.dumps(cache_mode)}')
            if cache_folder:
                lines.append(f'cache_folder = {json.dumps(cache_folder)}')
            if cache_fallback != "disable":
                lines.append(f'cache_fallback = {json.dumps(cache_fallback)}')

        lines.append("")

    output = "\n".join(lines)

    if len(sys.argv) > 1:
        out_path = sys.argv[1]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Wrote {len(rows)} import block(s) to {out_path}")
    else:
        print(output, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
