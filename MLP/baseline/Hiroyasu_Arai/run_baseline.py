from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_io import prepare_wide_table, wide_to_points
    from evaluate import evaluate_split, split_counts, write_json
    from fit import (
        bootstrap_params,
        calibrate_residual_uncertainty,
        fit_model,
        uncertainty_to_frame,
    )
    from models import ALL_VARIANTS, params_to_dict
    from plots import save_standard_plots
else:
    from .data_io import prepare_wide_table, wide_to_points
    from .evaluate import evaluate_split, split_counts, write_json
    from .fit import (
        bootstrap_params,
        calibrate_residual_uncertainty,
        fit_model,
        uncertainty_to_frame,
    )
    from .models import ALL_VARIANTS, params_to_dict
    from .plots import save_standard_plots


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
DEFAULT_CONFIG = THIS_DIR / "config_default.json"


def load_config(path: Path | None) -> dict[str, Any]:
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            override = json.load(f)
        config = _deep_update(config, override)
    return config


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def apply_variant_preset(config: dict[str, Any], variant: str | None) -> dict[str, Any]:
    if variant is None:
        variant = str(config.get("variant", "ha_hiroyasu"))
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choices: {ALL_VARIANTS}")
    config = dict(config)
    config["variant"] = variant
    return config


def resolve_output_root(config: dict[str, Any], override: Path | None) -> Path:
    root = override or Path(config.get("outputs", {}).get("root", "MLP/baseline/Hiroyasu_Arai/outputs"))
    root = Path(root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def maybe_subsample_wide(wide: pd.DataFrame, max_trajectories: int | None, seed: int) -> pd.DataFrame:
    if max_trajectories is None or len(wide) <= int(max_trajectories):
        return wide
    parts = []
    per_split = max(int(max_trajectories) // max(wide["sample_split"].nunique(), 1), 1)
    for _split, group in wide.groupby("sample_split", dropna=False):
        take = min(len(group), per_split)
        parts.append(group.sample(n=take, random_state=int(seed)))
    out = pd.concat(parts, sort=False)
    if len(out) < int(max_trajectories):
        remaining = wide.drop(index=out.index, errors="ignore")
        if not remaining.empty:
            extra = remaining.sample(
                n=min(int(max_trajectories) - len(out), len(remaining)),
                random_state=int(seed) + 1,
            )
            out = pd.concat([out, extra], sort=False)
    return out.sample(frac=1.0, random_state=int(seed) + 2).reset_index(drop=True)


def run_once(
    config: dict[str, Any],
    *,
    output_root: Path,
    tag: str | None = None,
    max_trajectories: int | None = None,
    save_plots: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    variant = str(config.get("variant", "ha_hiroyasu"))

    wide = prepare_wide_table(config)
    wide = maybe_subsample_wide(wide, max_trajectories=max_trajectories, seed=int(config.get("seed", 42)))
    points = wide_to_points(
        wide,
        time_min_ms=float(config.get("time_min_ms", 0.0)),
        time_max_ms=float(config.get("time_max_ms", 5.0)),
    )

    train_points = points.loc[points["sample_split"] == "train"].reset_index(drop=True)
    params, fit_info = fit_model(train_points, variant, config)

    unc_cfg = dict(config.get("uncertainty", {}))
    cal_split = str(unc_cfg.get("calibration_split", "val"))
    cal_points = points.loc[points["sample_split"] == cal_split].reset_index(drop=True)
    if cal_points.empty:
        cal_split = "train"
        cal_points = train_points
    residual_unc = calibrate_residual_uncertainty(cal_points, params, config)
    boot_params_list, boot_df = bootstrap_params(train_points, variant, config)

    primary_split = str(config.get("primary_eval_split", "test"))
    pred_points, per_traj, per_folder, per_nozzle, eval_payload = evaluate_split(
        points,
        split=primary_split,
        params=params,
        residual_unc=residual_unc,
        boot_params=boot_params_list,
        config=config,
    )
    time_bins = eval_payload["time_bins"]
    overall = eval_payload["overall"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_parts = [stamp, variant, str(config.get("split_mode", "split")), primary_split]
    if config.get("holdout_nozzle"):
        name_parts.append(f"holdout_{config['holdout_nozzle']}")
    if tag:
        name_parts.append(tag)
    run_dir = output_root / "_".join(name_parts)
    run_dir.mkdir(parents=True, exist_ok=False)

    write_json(run_dir / "config_effective.json", dict(config))
    write_json(run_dir / "model_params.json", params_to_dict(params))
    write_json(run_dir / "fit_info.json", fit_info)
    wide.to_csv(run_dir / "split_manifest.csv", index=False)
    uncertainty_to_frame(residual_unc).to_csv(run_dir / "residual_uncertainty_by_time.csv", index=False)
    if not boot_df.empty:
        boot_df.to_csv(run_dir / "bootstrap_params.csv", index=False)
    if bool(config.get("outputs", {}).get("save_points", True)):
        pred_points.to_csv(run_dir / "points.csv", index=False)
    per_traj.to_csv(run_dir / "per_trajectory.csv", index=False)
    per_folder.to_csv(run_dir / "per_folder.csv", index=False)
    per_nozzle.to_csv(run_dir / "per_nozzle.csv", index=False)
    time_bins.to_csv(run_dir / "time_bins.csv", index=False)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "model": "Hiroyasu-Arai",
        "variant": variant,
        "split_mode": str(config.get("split_mode")),
        "holdout_nozzle": config.get("holdout_nozzle"),
        "source": str(config.get("source", "cdf")),
        "series_split": str(config.get("series_split", "clean")),
        "primary_eval_split": primary_split,
        "split_counts_trajectories": split_counts(wide),
        "params": params_to_dict(params),
        "fit_info": fit_info,
        "uncertainty": {
            "residual_mode": residual_unc.mode,
            "global_sigma_mm": float(residual_unc.global_sigma_mm),
            "time_bin_ms": float(residual_unc.time_bin_ms),
            "sigma_floor_mm": float(residual_unc.sigma_floor_mm),
            "bootstrap_n_requested": int(unc_cfg.get("bootstrap_n", 0)),
            "bootstrap_n_success": int(len(boot_params_list)),
            "calibration_split_used": cal_split,
        },
        "overall": overall,
    }
    write_json(run_dir / "metrics_summary.json", summary)

    should_plot = bool(config.get("outputs", {}).get("save_plots", True)) if save_plots is None else bool(save_plots)
    if should_plot:
        save_standard_plots(
            points_df=pred_points,
            per_traj=per_traj,
            per_folder=per_folder,
            per_nozzle=per_nozzle,
            time_bins=time_bins,
            out_dir=run_dir,
            title_prefix=f"H-A {variant} ({primary_split})",
            max_trajectory_plots=int(config.get("outputs", {}).get("max_trajectory_plots", 24)),
        )

    return run_dir, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit and evaluate a Hiroyasu-Arai / Zhou penetration baseline.")
    p.add_argument("--config", type=Path, default=None, help="Optional JSON config override.")
    p.add_argument("--variant", choices=ALL_VARIANTS, default=None)
    p.add_argument("--split-mode", choices=("row_random_stage3", "grouped_condition", "leave_one_nozzle"), default=None)
    p.add_argument("--holdout-nozzle", default=None, help="Required for leave_one_nozzle, e.g. nozzle3.")
    p.add_argument("--bootstrap-n", type=int, default=None,
                   help="Override bootstrap count (only used for calibrated variants).")
    p.add_argument("--primary-split", choices=("train", "val", "test", "all"), default=None)
    p.add_argument("--max-trajectories", type=int, default=None, help="Smoke-test subset size.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_variant_preset(config, args.variant)
    if args.split_mode:
        config["split_mode"] = args.split_mode
    if args.holdout_nozzle:
        config["holdout_nozzle"] = args.holdout_nozzle
    if args.bootstrap_n is not None:
        config.setdefault("uncertainty", {})["bootstrap_n"] = int(args.bootstrap_n)
    if args.primary_split is not None:
        config["primary_eval_split"] = args.primary_split

    output_root = resolve_output_root(config, args.output_root)
    run_dir, summary = run_once(
        config,
        output_root=output_root,
        tag=args.tag,
        max_trajectories=args.max_trajectories,
        save_plots=not args.no_plots,
    )
    print(f"Wrote H-A baseline to {run_dir}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
