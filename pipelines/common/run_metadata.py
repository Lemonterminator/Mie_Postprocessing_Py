"""Capture run-level provenance metadata (git hash, config snapshot, timing)."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else False
    except Exception:
        return False


def make_run_id(prefix: str) -> str:
    """Return a timestamped run identifier, e.g. 'fit_20260517_143022'."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def write_metadata(
    run_dir: Path,
    *,
    phase: str,
    config: dict[str, Any],
    parent_run_ids: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Write _metadata.json into run_dir. Returns the metadata dict."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    started_at = datetime.now(tz=timezone.utc).isoformat()
    metadata: dict[str, Any] = {
        "phase": phase,
        "run_id": run_dir.name,
        "started_at": started_at,
        "finished_at": None,
        "wall_seconds": None,
        "git_hash": _git_hash(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "config": config,
        "parent_run_ids": parent_run_ids or {},
    }
    (run_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def finalize_metadata(run_dir: Path, started_wall: float) -> None:
    """Update _metadata.json with finish time and wall-clock seconds."""
    meta_path = run_dir / "_metadata.json"
    if not meta_path.exists():
        return
    metadata = json.loads(meta_path.read_text())
    metadata["finished_at"] = datetime.now(tz=timezone.utc).isoformat()
    metadata["wall_seconds"] = round(time.monotonic() - started_wall, 2)
    meta_path.write_text(json.dumps(metadata, indent=2))
