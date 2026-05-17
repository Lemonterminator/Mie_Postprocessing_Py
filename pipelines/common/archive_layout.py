"""Path conventions and latest-run resolution for all pipeline phases."""
from __future__ import annotations

import csv
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Archive roots for each phase
SYNTHETIC_DATA_RUNS = REPO_ROOT / "MLP" / "synthetic_data_runs"
AUDIT_RUNS = REPO_ROOT / "MLP" / "audit_runs"
THESIS_GENERATED = REPO_ROOT / "Thesis" / "generated"

# Canonical path consumed by legacy scripts (must remain intact while --mirror-canonical is on)
SYNTHETIC_DATA_CANONICAL = REPO_ROOT / "MLP" / "synthetic_data"

LATEST_FILE = "latest.txt"


def resolve_latest(archive_root: Path) -> Path:
    """Return the path of the most-recent run directory.

    Priority: (1) latest.txt contents, (2) mtime of _metadata.json glob.
    Raises FileNotFoundError if no runs exist.
    """
    latest_txt = archive_root / LATEST_FILE
    if latest_txt.exists():
        run_id = latest_txt.read_text().strip()
        candidate = archive_root / run_id
        if candidate.is_dir():
            return candidate

    # Fallback: glob by _metadata.json mtime (matches existing baseline script convention)
    candidates = sorted(
        archive_root.glob("*/_metadata.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under {archive_root}")
    return candidates[-1].parent


def update_latest(archive_root: Path, run_dir: Path) -> None:
    """Write latest.txt pointing at run_dir.name."""
    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / LATEST_FILE).write_text(run_dir.name)


def make_run_dir(archive_root: Path, run_id: str) -> Path:
    """Create and return a fresh run directory. Fails if it already exists."""
    run_dir = archive_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# ---------- Manifest helpers ----------

MANIFEST_FIELDS = ("role", "filename", "description")


def append_manifest(run_dir: Path, role: str, filename: str, description: str = "") -> None:
    """Append one row to _manifest.csv in run_dir."""
    manifest_path = run_dir / "_manifest.csv"
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow({"role": role, "filename": filename, "description": description})


def read_manifest(run_dir: Path) -> list[dict[str, str]]:
    """Return all manifest rows from run_dir/_manifest.csv."""
    manifest_path = run_dir / "_manifest.csv"
    if not manifest_path.exists():
        return []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
