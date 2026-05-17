"""Phase 4: collect role-tagged artifacts into Thesis/generated/{run_id}."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import AUDIT_RUNS, SYNTHETIC_DATA_RUNS, THESIS_GENERATED, read_manifest, resolve_latest
from pipelines.common.run_metadata import write_metadata, finalize_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run-dir", type=Path, default=None)
    parser.add_argument("--audit-run-dir", type=Path, default=None)
    parser.add_argument("--train-root", type=Path, default=REPO_ROOT / "MLP" / "runs_mlp")
    parser.add_argument("--output-root", type=Path, default=THESIS_GENERATED)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def _copy_manifest_files(phase: str, run_dir: Path, generated_dir: Path, rows_out: list[dict[str, str]]) -> None:
    used: dict[str, int] = {}
    for row in read_manifest(run_dir):
        role = row.get("role", "").strip()
        rel = row.get("filename", "").strip()
        if not role or not rel:
            continue
        src = run_dir / rel
        if not src.exists() or not src.is_file():
            continue
        suffix = src.suffix
        stem = role
        used[stem] = used.get(stem, 0) + 1
        dest_name = f"{stem}{suffix}" if used[stem] == 1 else f"{stem}_{used[stem]}{suffix}"
        dest = generated_dir / dest_name
        shutil.copy2(src, dest)
        rows_out.append(
            {
                "phase": phase,
                "source_run_id": run_dir.name,
                "role": role,
                "source_file": str(src),
                "generated_file": dest.name,
                "description": row.get("description", ""),
            }
        )


def _latest_metric_files(train_root: Path) -> list[Path]:
    files = sorted(train_root.rglob("_thesis_metrics.json"), key=lambda p: p.stat().st_mtime)
    by_stage: dict[str, Path] = {}
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stage = str(payload.get("stage", path.parent.name))
        except Exception:
            stage = path.parent.name
        by_stage[stage] = path
    return [by_stage[key] for key in sorted(by_stage)]


def _write_metrics_table(metric_files: list[Path], generated_dir: Path, rows_out: list[dict[str, str]]) -> None:
    rows = []
    for path in metric_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stage = str(payload.get("stage", path.parent.name))
        for metric, item in (payload.get("metrics") or {}).items():
            value = item.get("value") if isinstance(item, dict) else item
            target = item.get("target", payload.get("metric_target", "")) if isinstance(item, dict) else payload.get("metric_target", "")
            if isinstance(value, (int, float)):
                value_text = f"{value:.4g}"
            else:
                value_text = str(value)
            rows.append({"stage": stage, "metric": metric, "value": value_text, "target": str(target)})
    if not rows:
        return
    csv_path = generated_dir / "thesis_metrics_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "metric", "value", "target"])
        writer.writeheader()
        writer.writerows(rows)
    tex_lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Stage & Metric & Value & Target population \\",
        r"\midrule",
    ]
    for row in rows:
        metric = row["metric"].replace("_", r"\_")
        target = row["target"].replace("_", r"\_")
        tex_lines.append(
            f"{row['stage']} & {metric} & {row['value']} & {target} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path = generated_dir / "thesis_metrics_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    for role, path in (("thesis_metrics_table_csv", csv_path), ("thesis_metrics_table_tex", tex_path)):
        rows_out.append(
            {
                "phase": "train",
                "source_run_id": "latest_by_stage",
                "role": role,
                "source_file": ";".join(str(p) for p in metric_files),
                "generated_file": path.name,
                "description": "Phase 3 metrics with explicit target populations",
            }
        )


def _write_manifest(generated_dir: Path, rows: list[dict[str, str]]) -> None:
    fields = ["phase", "source_run_id", "role", "source_file", "generated_file", "description"]
    for name in ("_provenance.csv", "_manifest.csv"):
        with (generated_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    run_id = args.run_id or f"thesis_{datetime.now(tz=timezone.utc):%Y%m%d_%H%M%S}"
    generated_dir = args.output_root / run_id
    generated_dir.mkdir(parents=True, exist_ok=False)
    write_metadata(
        generated_dir,
        phase="report_assembly",
        config={
            "fit_run_dir": None if args.fit_run_dir is None else str(args.fit_run_dir),
            "audit_run_dir": None if args.audit_run_dir is None else str(args.audit_run_dir),
            "train_root": str(args.train_root),
        },
    )
    rows: list[dict[str, str]] = []
    fit_run = args.fit_run_dir or resolve_latest(SYNTHETIC_DATA_RUNS)
    audit_run = args.audit_run_dir or resolve_latest(AUDIT_RUNS)
    _copy_manifest_files("fit", fit_run, generated_dir, rows)
    _copy_manifest_files("audit", audit_run, generated_dir, rows)

    metric_files = _latest_metric_files(args.train_root)
    for path in metric_files:
        dest = generated_dir / f"{path.parent.name}_thesis_metrics.json"
        shutil.copy2(path, dest)
        rows.append(
            {
                "phase": "train",
                "source_run_id": path.parent.name,
                "role": "thesis_metrics_json",
                "source_file": str(path),
                "generated_file": dest.name,
                "description": "Flat thesis metrics export",
            }
        )
    _write_metrics_table(metric_files, generated_dir, rows)
    _write_manifest(generated_dir, rows)
    current_dir = args.output_root / "current"
    if current_dir.exists():
        shutil.rmtree(current_dir)
    current_dir.mkdir(parents=True, exist_ok=True)
    for path in generated_dir.iterdir():
        if path.is_file() and path.name != "_metadata.json":
            shutil.copy2(path, current_dir / path.name)
    finalize_metadata(generated_dir, started_wall=started)
    print(f"[report] Assembled thesis artifacts: {generated_dir}")
    return generated_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
