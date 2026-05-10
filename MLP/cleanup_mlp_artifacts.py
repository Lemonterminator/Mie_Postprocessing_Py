from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "MLP" / "runs_mlp"
EVAL_ROOT = PROJECT_ROOT / "MLP" / "eval"


KEEP_RUNS = {
    "stage3_ablation_suites",
}

KEEP_DATE_TOKENS = {
    "20260508",  # five-seed stability test pipeline
    "20260509",  # latest selected Stage-3 ablation suite and winner eval
}


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "K", "M", "G", "T"]:
        if value < 1024.0 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


def collect_top_level_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def is_kept_artifact_name(name: str) -> bool:
    return name in KEEP_RUNS or any(token in name for token in KEEP_DATE_TOKENS)


def classify_runs() -> list[dict[str, Any]]:
    rows = []
    for path in collect_top_level_dirs(RUNS_ROOT):
        keep = is_kept_artifact_name(path.name)
        rows.append(
            {
                "root": "runs_mlp",
                "name": path.name,
                "path": str(path),
                "action": "keep" if keep else "delete",
                "reason": "20260508 stability run or latest 20260509 suite" if keep else "older training artifact",
                "size_bytes": path_size_bytes(path),
            }
        )
    return rows


def classify_evals() -> list[dict[str, Any]]:
    rows = []
    for path in collect_top_level_dirs(EVAL_ROOT):
        if not path.name.startswith("rmse_eval_") and path.name != "__pycache__":
            keep = True
            reason = "non-evaluation support directory"
        elif is_kept_artifact_name(path.name):
            keep = True
            reason = "20260508 stability eval or latest 20260509 eval"
        else:
            keep = False
            reason = "older evaluation artifact"
        rows.append(
            {
                "root": "eval",
                "name": path.name,
                "path": str(path),
                "action": "keep" if keep else "delete",
                "reason": reason,
                "size_bytes": path_size_bytes(path),
            }
        )
    return rows


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["root", "name", "path", "action", "reason", "size_bytes", "size_human"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["size_human"] = human_size(int(out["size_bytes"]))
            writer.writerow(out)


def delete_paths(rows: list[dict[str, Any]]) -> list[str]:
    deleted = []
    for row in rows:
        if row["action"] != "delete":
            continue
        path = Path(row["path"])
        if path.exists():
            shutil.rmtree(path)
            deleted.append(str(path))
    return deleted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean old MLP run/eval artifacts while keeping the latest Stage-3 suite.")
    p.add_argument("--apply", action="store_true", help="Actually delete old artifacts. Default is dry-run.")
    p.add_argument("--manifest", type=Path, default=None, help="CSV manifest path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = classify_runs() + classify_evals()
    delete_rows = [r for r in rows if r["action"] == "delete"]
    keep_rows = [r for r in rows if r["action"] == "keep"]
    reclaim_bytes = sum(int(r["size_bytes"]) for r in delete_rows)
    keep_bytes = sum(int(r["size_bytes"]) for r in keep_rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = args.manifest or (PROJECT_ROOT / "MLP" / f"cleanup_manifest_{stamp}.csv")
    write_manifest(rows, manifest)

    payload = {
        "mode": "apply" if args.apply else "dry_run",
        "manifest": str(manifest),
        "n_keep": len(keep_rows),
        "n_delete": len(delete_rows),
        "keep_size": human_size(keep_bytes),
        "reclaim_size": human_size(reclaim_bytes),
        "keep_runs": sorted(KEEP_RUNS),
        "keep_date_tokens": sorted(KEEP_DATE_TOKENS),
    }
    if args.apply:
        payload["deleted_paths"] = delete_paths(delete_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
