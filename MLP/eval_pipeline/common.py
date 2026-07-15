"""Shared plumbing for the layered evaluation pipeline.

Path resolution, junction guard, timestamped run directories and JSON I/O.
Every module in ``MLP.eval_pipeline`` imports from here instead of
re-implementing the per-script helpers that used to be copy-pasted across
``MLP/eval/*.py``.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYNTHETIC_JUNCTION = PROJECT_ROOT / "MLP" / "synthetic_data"


def resolve_path(path: Path | str) -> Path:
    """Resolve a possibly repo-relative path to an absolute one.

    A leading ``/`` with no drive letter (``p.root`` set, ``p.drive`` empty)
    is neither absolute on Windows nor safely joinable: ``PROJECT_ROOT / p``
    silently keeps only PROJECT_ROOT's drive and drops its directory
    components, redirecting to the drive root instead of the repo. Reject it
    explicitly rather than resolving to a plausible-looking wrong path.
    """
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return p
    if p.root and not p.drive:
        raise ValueError(
            f"{path!r} starts with a path separator but has no drive letter; "
            "joining it to PROJECT_ROOT would silently discard PROJECT_ROOT. "
            "Use a repo-relative path (no leading slash) or a full absolute path."
        )
    return PROJECT_ROOT / p


def guard_dataset_root(root: Path) -> Path:
    """Refuse the ``MLP/synthetic_data`` junction; demand an explicit root.

    The junction silently retargets between lv2/lv3 datasets, which is exactly
    the ambiguity this pipeline exists to remove.
    """
    resolved = resolve_path(root)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
    try:
        is_junction = resolved.resolve() != resolved and resolved.samefile(SYNTHETIC_JUNCTION)
    except (OSError, ValueError):
        is_junction = False
    if is_junction or resolved == SYNTHETIC_JUNCTION:
        raise ValueError(
            "Refusing to evaluate through the MLP/synthetic_data junction. "
            "Pass an explicit dataset root (e.g. MLP/synthetic_data_clean_lv3_qc_gated)."
        )
    return resolved


_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def timestamped_dir(root: Path, prefix: str, tag: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = _UNSAFE_NAME_CHARS.sub("_", tag).strip(" .") if tag else None
    name = f"{prefix}_{stamp}" + (f"_{safe_tag}" if safe_tag else "")
    out = resolve_path(root) / name
    out.mkdir(parents=True, exist_ok=False)
    return out


def jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas/Path values into JSON-safe types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_points(df: pd.DataFrame, path_stem: Path) -> Path:
    """Write a points table as parquet when pyarrow is available, else CSV.

    Falls back to CSV on ANY parquet failure (missing pyarrow, or a
    serialization error from e.g. an inconsistently-typed metadata column
    assembled across per-condition groups) rather than losing the already
    computed metrics for the whole eval set to an uncaught exception.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401

        out = path_stem.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    except ImportError:
        pass
    except Exception as exc:
        print(f"[warn] parquet write failed ({exc!r}); falling back to CSV for {path_stem}")
    out = path_stem.with_suffix(".csv")
    df.to_csv(out, index=False)
    return out


def read_points(dir_path: Path, stem: str = "points") -> pd.DataFrame:
    """Read a points table written by :func:`write_points` (parquet or CSV)."""
    parquet = Path(dir_path) / f"{stem}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    csv = Path(dir_path) / f"{stem}.csv"
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError(f"No {stem}.parquet/{stem}.csv under {dir_path}")
