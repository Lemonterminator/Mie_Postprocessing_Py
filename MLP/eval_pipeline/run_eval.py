"""Layer 1 CLI: full evaluation from (dataset roots, model checkpoints).

Examples
--------
Full thesis roster (13 checkpoints + HA/NS baselines) on both dataset roots::

    .venv/Scripts/python.exe MLP/eval_pipeline/run_eval.py \
        --manifest MLP/eval_pipeline/manifests/thesis_production.json

Ad-hoc single model on one dataset::

    .venv/Scripts/python.exe MLP/eval_pipeline/run_eval.py \
        --model "seed42=mlp:MLP/best_models/thesis_baselines/production_mlp_modeA_5seed/seed_42" \
        --dataset "lv3_qc_gated=MLP/synthetic_data_clean_lv3_qc_gated" \
        --eval-set cdf_uncensored

Smoke test (first 10 conditions only)::

    ... run_eval.py --manifest ... --limit-conditions 10 --tag smoke

Outputs one run directory (default under ``MLP/eval_pipeline/runs``)::

    evalrun_<timestamp>_<tag>/
      manifest.json          resolved models/datasets/options
      models/<label>/<dataset>/<eval_set>/{points.parquet, metrics.json, *.csv}
      metrics_wide.csv       one row per model x dataset x eval_set x slice
      known_number_checks.json  regression guard against published thesis numbers

Layer 2 (``make_figures.py``) consumes the run directory; this script never
plots anything.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.eval_pipeline.common import (  # noqa: E402
    read_json,
    resolve_path,
    timestamped_dir,
    write_json,
)
from MLP.eval_pipeline.datasets import DatasetSpec, make_dataset_spec  # noqa: E402
from MLP.eval_pipeline.predictors import MODEL_KINDS, ModelSpec  # noqa: E402
from MLP.eval_pipeline.runner import evaluate_model, flatten_payloads  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "eval_pipeline" / "runs"
DEFAULT_MANIFEST = PROJECT_ROOT / "MLP" / "eval_pipeline" / "manifests" / "thesis_production.json"

#: Published thesis numbers used as regression guards (mm, cdf_uncensored).
KNOWN_NUMBER_CHECKS = (
    {"model_label": "residual_film", "dataset": "lv3_qc_gated",
     "eval_set": "cdf_uncensored", "metric": "rmse_mm", "expected": 4.401, "tol": 0.08},
    {"model_label": "qc_retrained_residual_svgp", "dataset": "lv3_qc_gated",
     "eval_set": "cdf_uncensored", "metric": "rmse_mm", "expected": 3.848, "tol": 0.08},
)


def parse_model_arg(text: str) -> ModelSpec:
    """Parse ``label=kind:path[:seed]`` into a ModelSpec."""
    label, _, rest = text.partition("=")
    kind, _, path_part = rest.partition(":")
    if not label or kind not in MODEL_KINDS or not path_part:
        raise argparse.ArgumentTypeError(
            f"Expected label=kind:path with kind in {MODEL_KINDS}, got {text!r}")
    seed: int | None = None
    if ":" in path_part:
        maybe_path, _, maybe_seed = path_part.rpartition(":")
        if maybe_seed.isdigit():
            path_part, seed = maybe_path, int(maybe_seed)
    return ModelSpec(label=label, kind=kind, path=resolve_path(path_part), seed=seed)


def parse_dataset_arg(text: str) -> tuple[str, str]:
    name, _, path = text.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected name=path, got {text!r}")
    return name, path


def models_from_manifest(payload: dict[str, Any]) -> list[ModelSpec]:
    models = []
    for entry in payload.get("models", []):
        models.append(ModelSpec(
            label=str(entry["label"]),
            kind=str(entry["kind"]),
            path=resolve_path(entry["path"]),
            family=str(entry.get("family", "")),
            seed=(int(entry["seed"]) if entry.get("seed") is not None else None),
            meta=dict(entry.get("meta", {})),
        ))
    if not models:
        raise ValueError("Manifest contains no models.")
    return models


def datasets_from_manifest(payload: dict[str, Any],
                           eval_sets: list[str] | None) -> list[DatasetSpec]:
    datasets = []
    for entry in payload.get("datasets", []):
        datasets.append(make_dataset_spec(str(entry["name"]), entry["root"], eval_sets))
    if not datasets:
        raise ValueError("Manifest contains no datasets.")
    return datasets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"JSON manifest of models+datasets (default: {DEFAULT_MANIFEST} "
                             "when no --model/--dataset given)")
    parser.add_argument("--model", action="append", default=[],
                        help="Ad-hoc model as label=kind:path[:seed]; repeatable. "
                             f"Kinds: {', '.join(MODEL_KINDS)}")
    parser.add_argument("--dataset", action="append", default=[],
                        help="Ad-hoc dataset as name=path; repeatable.")
    parser.add_argument("--model-label", action="append", default=[],
                        help="Only evaluate manifest models with these labels; repeatable.")
    parser.add_argument("--eval-set", action="append", default=[],
                        choices=["cdf_uncensored", "full_clean", "p50_observed", "q1_grid_all"],
                        help="Restrict eval sets (default: all canonical sets).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-points-mlp", type=int, default=262_144)
    parser.add_argument("--batch-points-gp", type=int, default=65_536)
    parser.add_argument("--t-min-ms", type=float, default=0.0)
    parser.add_argument("--t-max-ms", type=float, default=5.0)
    parser.add_argument("--rel-err-floor-mm", type=float, default=5.0)
    parser.add_argument("--filter-experiment", default=None)
    parser.add_argument("--limit-conditions", type=int, default=None,
                        help="Smoke-test hook: keep only the first N conditions per table.")
    parser.add_argument("--no-save-points", action="store_true",
                        help="Skip writing per-point tables (metrics only).")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Add seeded bootstrap 95%% CIs for RMSE/MAE/bias.")
    parser.add_argument("--bootstrap-n", type=int, default=2000,
                        help="Number of bootstrap resamples when --bootstrap is enabled.")
    parser.add_argument("--bootstrap-workers", type=int, default=1,
                        help="Worker threads for vectorized bootstrap chunks.")
    parser.add_argument("--strict-known-checks", action="store_true",
                        help="Exit non-zero when a known-number regression check fails.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    eval_sets = args.eval_set or None
    if args.model or args.dataset:
        if not (args.model and args.dataset):
            raise SystemExit("Ad-hoc mode needs at least one --model AND one --dataset.")
        models = [parse_model_arg(t) for t in args.model]
        datasets = [make_dataset_spec(name, path, eval_sets)
                    for name, path in (parse_dataset_arg(t) for t in args.dataset)]
        manifest_path = None
    else:
        manifest_path = resolve_path(args.manifest or DEFAULT_MANIFEST)
        payload = read_json(manifest_path)
        models = models_from_manifest(payload)
        datasets = datasets_from_manifest(payload, eval_sets)

    if args.model_label:
        wanted = set(args.model_label)
        unknown = wanted - {m.label for m in models}
        if unknown:
            raise SystemExit(f"--model-label not in roster: {sorted(unknown)}")
        models = [m for m in models if m.label in wanted]

    # Duplicate labels/names would silently overwrite each other's on-disk
    # artifacts (same out_dir) while metrics_wide.csv kept both rows —
    # reject up front instead of producing artifacts disconnected from the
    # aggregate table.
    dupe_labels = [label for label, count in Counter(m.label for m in models).items() if count > 1]
    if dupe_labels:
        raise SystemExit(f"Duplicate model labels (would overwrite each other's artifacts): {sorted(dupe_labels)}")
    dupe_names = [name for name, count in Counter(d.name for d in datasets).items() if count > 1]
    if dupe_names:
        raise SystemExit(f"Duplicate dataset names: {sorted(dupe_names)}")

    run_dir = timestamped_dir(args.output_root, "evalrun", args.tag)
    models_root = run_dir / "models"
    print(f"[run] {run_dir}")
    print(f"[run] {len(models)} model(s) x {len(datasets)} dataset(s)")

    write_json(run_dir / "manifest.json", {
        "manifest_path": str(manifest_path) if manifest_path else None,
        "models": [{"label": m.label, "kind": m.kind, "path": str(m.path),
                    "family": m.family, "seed": m.seed, "meta": dict(m.meta)} for m in models],
        "datasets": [{"name": d.name, "root": str(d.root),
                      "eval_sets": [s.name for s in d.eval_sets]} for d in datasets],
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    })

    payloads: list[dict[str, Any]] = []
    for model in models:
        for dataset in datasets:
            payloads.extend(evaluate_model(
                model, dataset, models_root,
                device=args.device,
                filter_experiment=args.filter_experiment,
                t_min_ms=args.t_min_ms,
                t_max_ms=args.t_max_ms,
                rel_err_floor_mm=args.rel_err_floor_mm,
                batch_points_mlp=args.batch_points_mlp,
                batch_points_gp=args.batch_points_gp,
                save_points=not args.no_save_points,
                limit_conditions=args.limit_conditions,
                bootstrap=args.bootstrap,
                bootstrap_n=args.bootstrap_n,
                bootstrap_workers=args.bootstrap_workers,
            ))

    wide = flatten_payloads(payloads)
    wide.to_csv(run_dir / "metrics_wide.csv", index=False)

    checks = run_known_number_checks(wide) if args.limit_conditions is None else []
    if checks:
        write_json(run_dir / "known_number_checks.json", checks)
        failed = [c for c in checks if c["status"] == "FAIL"]
        for check in checks:
            print(f"[check] {check['status']}: {check['model_label']}/{check['dataset']}"
                  f"/{check['eval_set']} {check['metric']}={check.get('actual')}"
                  f" (expected {check['expected']}+/-{check['tol']})")
        if failed and args.strict_known_checks:
            return 1

    errors = wide["error"].notna().sum() if "error" in wide.columns else 0
    print(f"[done] {len(wide)} metric rows -> {run_dir / 'metrics_wide.csv'}"
          + (f" ({errors} FAILED combos)" if errors else ""))
    return 2 if errors else 0


def run_known_number_checks(wide) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for check in KNOWN_NUMBER_CHECKS:
        mask = (
            (wide.get("model_label") == check["model_label"])
            & (wide.get("dataset") == check["dataset"])
            & (wide.get("eval_set") == check["eval_set"])
            & (wide.get("slice") == "overall")
        )
        row = wide.loc[mask]
        result = dict(check)
        if row.empty or check["metric"] not in row.columns:
            result["status"] = "SKIP"
        else:
            actual = float(row.iloc[0][check["metric"]])
            result["actual"] = actual
            result["status"] = "PASS" if abs(actual - check["expected"]) <= check["tol"] else "FAIL"
        checks.append(result)
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
