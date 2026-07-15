"""Layer 2 CLI: render the full figure suite from a Layer-1 run directory.

Reads only artifacts produced by ``run_eval.py`` (no model loading, no
inference), so figures can be regenerated instantly and repeatedly while
polishing styles::

    .venv/Scripts/python.exe MLP/eval_pipeline/make_figures.py \
        --run MLP/eval_pipeline/runs/evalrun_<ts>_<tag> [--formats png,pdf]

Outputs::

    <run>/figures/models/<label>/<dataset>/<eval_set>/*.png   per-model suite
    <run>/figures/comparison/<dataset>/<eval_set>/*.png       cross-model suite
    <run>/figure_manifest.json                                catalog of all files
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.eval_pipeline.common import read_json, resolve_path, write_json  # noqa: E402
from MLP.eval_pipeline.figures.style import apply_style, family_color_map  # noqa: E402
from MLP.eval_pipeline.figures.per_model import render_eval_set_figures  # noqa: E402
from MLP.eval_pipeline.figures.comparison import render_comparison_figures  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True,
                        help="Layer-1 run directory (evalrun_*)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only these model labels (default: all found)")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--eval-sets", nargs="*", default=None)
    parser.add_argument("--formats", default="png",
                        help="Comma-separated: png[,pdf,svg]")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=80_000,
                        help="Subsample cap for scatter/hexbin figures")
    parser.add_argument("--top-n-conditions", type=int, default=30)
    parser.add_argument("--skip-per-model", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_style()
    run_dir = resolve_path(args.run)
    models_root = run_dir / "models"
    if not models_root.is_dir():
        raise SystemExit(f"No models/ directory under {run_dir} — is this a Layer-1 run?")
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())

    manifest = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    family_order = [m.get("family") or m.get("label") for m in manifest.get("models", [])]
    colors = family_color_map(family_order)

    written: list[str] = []
    failures: list[dict[str, str]] = []

    # ---- per-model suites ----------------------------------------------------
    metric_files = sorted(models_root.glob("*/*/*/metrics.json"))
    for metrics_path in metric_files:
        meta = read_json(metrics_path)
        if args.models and meta.get("model_label") not in args.models:
            continue
        if args.datasets and meta.get("dataset") not in args.datasets:
            continue
        if args.eval_sets and meta.get("eval_set") not in args.eval_sets:
            continue
        eval_dir = metrics_path.parent
        if not args.skip_per_model:
            fig_dir = (run_dir / "figures" / "models" / meta["model_label"]
                       / meta["dataset"] / meta["eval_set"])
            try:
                paths = render_eval_set_figures(
                    eval_dir, fig_dir, meta,
                    max_points=args.max_points,
                    top_n_conditions=args.top_n_conditions,
                    dpi=args.dpi, formats=formats,
                )
                written.extend(str(p) for p in paths)
                print(f"[fig] {meta['model_label']}/{meta['dataset']}/{meta['eval_set']}: "
                      f"{len(paths)} figures")
            except Exception:
                failures.append({"where": str(eval_dir), "error": traceback.format_exc(limit=4)})
                print(f"[FAIL] per-model figures for {eval_dir}")
                traceback.print_exc()

    # ---- comparison suites ---------------------------------------------------
    wide_path = run_dir / "metrics_wide.csv"
    if not args.skip_comparison and wide_path.exists():
        wide = pd.read_csv(wide_path)
        overall = wide[(wide.get("slice") == "overall")]
        if "error" in overall.columns:
            overall = overall[overall["error"].isna()]
        if args.models:
            overall = overall[overall["model_label"].isin(args.models)]
        # Preserve manifest model order so hues stay stable across figures.
        label_order = [m["label"] for m in manifest.get("models", [])]
        if label_order:
            overall = overall.copy()
            overall["_order"] = overall["model_label"].map(
                {label: i for i, label in enumerate(label_order)}).fillna(len(label_order))
            overall = overall.sort_values(["_order", "model_label"])
        for (dataset, eval_set), sub in overall.groupby(["dataset", "eval_set"], sort=True):
            if args.datasets and dataset not in args.datasets:
                continue
            if args.eval_sets and eval_set not in args.eval_sets:
                continue
            if len(sub) < 2:
                continue
            model_dirs = {row["model_label"]: models_root / row["model_label"] / dataset / eval_set
                          for _, row in sub.iterrows()}
            fig_dir = run_dir / "figures" / "comparison" / dataset / eval_set
            title = f"{dataset} | {eval_set}"
            try:
                paths = render_comparison_figures(sub, model_dirs, fig_dir, colors,
                                                  title=title, dpi=args.dpi, formats=formats)
                written.extend(str(p) for p in paths)
                print(f"[fig] comparison {dataset}/{eval_set}: {len(paths)} figures")
            except Exception:
                failures.append({"where": f"comparison/{dataset}/{eval_set}",
                                 "error": traceback.format_exc(limit=4)})
                print(f"[FAIL] comparison figures for {dataset}/{eval_set}")
                traceback.print_exc()

    write_json(run_dir / "figure_manifest.json", {
        "run_dir": str(run_dir),
        "n_figures": len(written),
        "formats": list(formats),
        "figures": written,
        "failures": failures,
    })
    print(f"[done] {len(written)} figure files -> {run_dir / 'figures'}"
          + (f" ({len(failures)} failures)" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
