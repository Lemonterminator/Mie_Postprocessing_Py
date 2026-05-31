"""Post-training evaluation hooks.

Default Stage-3 evaluation now runs the canonical point-table suite from
``MLP/eval/``:

* ``cdf_points_uncensored.csv`` (primary metric; right-censored tail removed)
* ``p50_q1_observed_fit_points.csv``
* ``p50_q1_predictions.csv``

The legacy clean-series evaluator remains available via ``eval_kind="series"``.
Full probabilistic diagnostics are written by the point-table evaluator for
``cdf_uncensored`` and ``p50_observed``. The lighter sigma-bin coverage audit is
still run on the primary point-table subdirectory when points are saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class PostEvalResult:
    eval_dir: Path
    rmse_summary: dict[str, Any]
    calibration_summary: dict[str, Any] | None
    probabilistic_summaries: dict[str, Any]
    figure_summary: dict[str, Any] | None


def _collect_probabilistic_summaries(rmse_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, payload in (rmse_summary.get("eval_sets") or {}).items():
        if not isinstance(payload, dict):
            continue
        probabilistic = payload.get("probabilistic")
        if probabilistic:
            out[str(name)] = probabilistic
    return out


def run_default_post_eval(
    *,
    run_dir: Path,
    device: torch.device | str | None = None,
    eval_kind: str = "point_tables",
    split: str = "clean",
    filter_experiment: str | None = None,
    synthetic_root: Path | None = None,
    t_min_ms: float = 0.0,
    t_max_ms: float = 5.0,
    rel_err_floor_mm: float = 5.0,
    output_root: Path | None = None,
    tag: str | None = None,
    batch_points: int = 65536,
    fast: bool = False,
    save_points: bool = True,
    save_plots: bool = True,
    max_traj_plots: int | None = None,
    run_calibration: bool = True,
    calibration_n_bins: int = 10,
) -> PostEvalResult:
    """Run default post-training eval then optional calibration/coverage audit.

    Calibration is automatically skipped (with a printed notice) when
    ``save_points=False`` or ``run_calibration=False``.
    """
    run_dir = Path(run_dir)
    eval_tag = tag or run_dir.name
    print()
    eval_kind_norm = str(eval_kind).strip().lower()
    if eval_kind_norm in {"point", "points", "point_table", "point_tables"}:
        from MLP.eval.inference_rmse_on_point_tables import run_point_table_evaluation

        print("Running automatic post-training point-table evaluation...")
        eval_dir, rmse_summary = run_point_table_evaluation(
            refinement_run=run_dir,
            filter_experiment=filter_experiment,
            device=device,
            output_root=output_root,
            tag=eval_tag,
            t_min_ms=float(t_min_ms),
            t_max_ms=float(t_max_ms),
            rel_err_floor_mm=float(rel_err_floor_mm),
            batch_points=int(batch_points),
            save_points=bool(save_points),
            save_plots=bool(save_plots),
            max_traj_plots=max_traj_plots,
        )
    elif eval_kind_norm == "series":
        from MLP.eval.inference_rmse_on_series import run_rmse_evaluation

        print("Running automatic post-training clean-series RMSE evaluation...")
        eval_dir, rmse_summary = run_rmse_evaluation(
            refinement_run=run_dir,
            split=split,
            filter_experiment=filter_experiment,
            device=device,
            synthetic_root=synthetic_root,
            t_min_ms=float(t_min_ms),
            t_max_ms=float(t_max_ms),
            rel_err_floor_mm=float(rel_err_floor_mm),
            output_root=output_root,
            tag=eval_tag,
            batch_points=int(batch_points),
            fast=bool(fast),
            save_points=bool(save_points),
            save_plots=bool(save_plots),
            max_traj_plots=max_traj_plots,
        )
    else:
        raise ValueError(f"Unsupported eval_kind={eval_kind!r}; use 'point_tables' or 'series'.")
    print("Post-training evaluation saved to:", eval_dir)
    probabilistic_summaries = _collect_probabilistic_summaries(dict(rmse_summary))
    if probabilistic_summaries:
        print(
            "Probabilistic diagnostics saved for:",
            ", ".join(sorted(probabilistic_summaries)),
        )
    figure_summary = rmse_summary.get("figures") if isinstance(rmse_summary.get("figures"), dict) else None
    if figure_summary:
        print("Point-eval figures saved to:", figure_summary.get("figures_dir", Path(eval_dir) / "figures"))

    calibration_summary: dict[str, Any] | None = None
    if not run_calibration:
        print("Calibration/coverage audit skipped (run_calibration=False).")
    elif not save_points:
        print("Calibration/coverage audit skipped: requires save_points=True.")
    else:
        from MLP.eval.calibration_coverage_audit import run_audit

        print("Running calibration/coverage audit...")
        calibration_eval_dir = Path(eval_dir)
        if str(rmse_summary.get("eval_kind")) == "point_tables":
            primary = str(rmse_summary.get("primary_eval_set", ""))
            if primary:
                calibration_eval_dir = Path(eval_dir) / primary
        calibration_summary = run_audit(
            eval_dir=calibration_eval_dir,
            n_bins=int(calibration_n_bins),
            thesis_image_dir=None,
        )
        print("Calibration/coverage audit saved to:", calibration_eval_dir / "calibration")

    return PostEvalResult(
        eval_dir=Path(eval_dir),
        rmse_summary=dict(rmse_summary),
        calibration_summary=calibration_summary,
        probabilistic_summaries=probabilistic_summaries,
        figure_summary=figure_summary,
    )
