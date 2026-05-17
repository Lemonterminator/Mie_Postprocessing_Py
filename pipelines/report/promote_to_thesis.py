"""Promote selected generated artifacts into Thesis/images canonical filenames."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import THESIS_GENERATED, resolve_latest

DEFAULT_MAP = Path(__file__).resolve().parent / "canonical_figure_map.yaml"
THESIS_IMAGES = REPO_ROOT / "Thesis" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-run-dir", type=Path, default=None)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--thesis-image-dir", type=Path, default=THESIS_IMAGES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_simple_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        mapping[key.strip()] = value.strip().strip("'\"")
    return mapping


def run(args: argparse.Namespace) -> None:
    generated = args.generated_run_dir or resolve_latest(THESIS_GENERATED)
    mapping = read_simple_map(args.map)
    args.thesis_image_dir.mkdir(parents=True, exist_ok=True)
    promoted = []
    for role, target_name in mapping.items():
        matches = sorted(generated.glob(f"{role}.*"))
        if not matches:
            print(f"[promote] missing role {role} in {generated}")
            continue
        src = matches[0]
        dest = args.thesis_image_dir / target_name
        print(f"[promote] {src.name} -> {dest}")
        promoted.append((src, dest))
        if not args.dry_run:
            shutil.copy2(src, dest)
    if not args.dry_run:
        (args.thesis_image_dir / "_promoted_from.txt").write_text(
            f"{generated}\n" + "\n".join(f"{src.name} -> {dest.name}" for src, dest in promoted) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
