#!/usr/bin/env python
"""Rename exported frame files into the numeric format preferred by official SAM3."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Rename frames like "frame_00000.jpg" to "00000.jpg" for SAM3.'
    )
    parser.add_argument("--frame-dir", required=True, help="Directory containing image frames.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without modifying files.",
    )
    return parser.parse_args()


def extract_numeric_stem(path: Path) -> str:
    stem = path.stem
    if stem.isdigit():
        return stem
    if stem.startswith("frame_"):
        suffix = stem.split("_", 1)[1]
        if suffix.isdigit():
            return suffix
    raise ValueError(f"Unsupported frame name format: {path.name}")


def main() -> int:
    args = parse_args()
    frame_dir = Path(args.frame_dir).expanduser().resolve()
    if not frame_dir.is_dir():
        print(f"Error: not a directory: {frame_dir}")
        return 1

    frame_paths = sorted(
        [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not frame_paths:
        print(f"Error: no frames found in {frame_dir}")
        return 1

    rename_pairs: list[tuple[Path, Path]] = []
    for path in frame_paths:
        numeric_stem = extract_numeric_stem(path)
        target = path.with_name(f"{numeric_stem}{path.suffix.lower()}")
        rename_pairs.append((path, target))

    collisions = [target for src, target in rename_pairs if src != target and target.exists()]
    if collisions:
        print("Error: target filenames already exist:")
        for target in collisions[:10]:
            print(f"  - {target}")
        return 1

    changed = 0
    for src, target in rename_pairs:
        if src == target:
            continue
        changed += 1
        print(f"{src.name} -> {target.name}")
        if not args.dry_run:
            src.rename(target)

    print(f"[ok] processed {len(frame_paths)} frames, renamed {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
