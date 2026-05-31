"""External (n=795k clean diagnostic) eval for the 5 new ΔP^0.5 winners.

Loads the 5 winner_run_dirs from the latest mode-C bootstrap_summary.json,
calls run_rmse_evaluation on each, and writes an aggregated comparison CSV
(alongside HA / NS / old-MLP rows from the prior baseline CSV).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.eval.inference_rmse_on_series import run_rmse_evaluation  # noqa: E402


NEW_BOOTSTRAP_SUMMARY = (
    PROJECT_ROOT
    / "MLP"
    / "runs_mlp"
    / "full_pipeline_C_20260509_110100"
    / "bootstrap_summary.json"
)
BASELINE_CSV = (
    PROJECT_ROOT
    / "MLP"
    / "baseline"
    / "comparison_reports"
    / "stage3_vs_HA_NS_20260509"
    / "overall_metrics.csv"
)
OUT_DIR = (
    PROJECT_ROOT
    / "MLP"
    / "baseline"
    / "comparison_reports"
    / "stage3_dp_exp_0p5_vs_HA_NS_20260509"
)
EVAL_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "eval"

METRIC_KEYS = [
    "n_points",
    "n_trajectories",
    "rmse_mm",
    "mae_mm",
    "bias_mm",
    "median_abs_err_mm",
    "p90_abs_err_mm",
    "p95_abs_err_mm",
    "coverage_1sigma",
    "coverage_2sigma",
    "mean_pred_std_mm",
    "mean_rel_err",
    "median_rel_err",
    "nrmse_range",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    boot = json.loads(NEW_BOOTSTRAP_SUMMARY.read_text(encoding="utf-8"))
    seeds_records = [(rec["seed"], rec["winner_name"], Path(rec["winner_run_dir"]))
                     for rec in boot["per_seed"]]

    print(f"Found {len(seeds_records)} new winners. Evaluating each on n=795k clean ...")

    new_rows: list[dict] = []
    for seed, winner_name, run_dir in seeds_records:
        print(f"\n--- seed {seed} winner={winner_name} ---")
        print(f"    run_dir: {run_dir}")
        out_dir, summary = run_rmse_evaluation(
            refinement_run=run_dir,
            split="clean",
            device=None,  # auto
            t_min_ms=0.0,
            t_max_ms=5.0,
            rel_err_floor_mm=5.0,
            output_root=EVAL_OUTPUT_ROOT,
            tag=f"newdp_seed{seed}",
            batch_points=262144,
            fast=False,
            save_points=False,
            save_plots=False,
            max_traj_plots=0,
        )
        overall = summary["overall"]
        row = {
            "model": f"Stage-3 MLP {winner_name} (ΔP^0.5, seed {seed})",
            "kind": "stage3_mlp_dp05",
            "seed": seed,
            "winner_name": winner_name,
            "run_dir": str(run_dir),
            "eval_dir": str(out_dir),
        }
        for k in METRIC_KEYS:
            row[k] = overall.get(k)
        new_rows.append(row)

    # Aggregate the new MLP rows: mean / std / 95% bootstrap CI on key metrics.
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(OUT_DIR / "per_seed_new_mlp_eval.csv", index=False)

    rng = np.random.default_rng(20260509)
    agg_rows = []

    def bootstrap_ci(values: np.ndarray, n_resample: int = 5000, alpha: float = 0.05):
        if len(values) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        if len(values) == 1:
            v = float(values[0])
            return v, 0.0, v, v
        means = rng.choice(values, size=(n_resample, len(values)), replace=True).mean(axis=1)
        return (
            float(values.mean()),
            float(values.std(ddof=1)),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)),
        )

    summary_metrics = ["rmse_mm", "mae_mm", "bias_mm", "median_abs_err_mm",
                       "p90_abs_err_mm", "p95_abs_err_mm",
                       "coverage_1sigma", "coverage_2sigma", "mean_pred_std_mm",
                       "mean_rel_err", "median_rel_err", "nrmse_range"]
    agg_row = {"model": "Stage-3 MLP ΔP^0.5 (5-seed mean)", "kind": "stage3_mlp_dp05_5seed"}
    for k in summary_metrics:
        vals = np.asarray([r[k] for r in new_rows if r.get(k) is not None and np.isfinite(r[k])])
        m, s, lo, hi = bootstrap_ci(vals)
        agg_row[k] = m
        agg_row[f"{k}_std"] = s
        agg_row[f"{k}_ci_lo"] = lo
        agg_row[f"{k}_ci_hi"] = hi
    # n_points / n_trajectories: take from first row (should be identical across seeds)
    agg_row["n_points"] = new_rows[0]["n_points"]
    agg_row["n_trajectories"] = new_rows[0]["n_trajectories"]
    agg_rows.append(agg_row)

    pd.DataFrame(agg_rows).to_csv(OUT_DIR / "new_mlp_5seed_aggregate.csv", index=False)

    # Build the headline comparison table side by side with HA / NS / old-MLP.
    base_df = pd.read_csv(BASELINE_CSV)
    headline_cols = ["model", "n_points", "n_trajectories", "rmse_mm", "mae_mm",
                     "bias_mm", "median_abs_err_mm", "p90_abs_err_mm", "p95_abs_err_mm",
                     "coverage_1sigma", "coverage_2sigma", "mean_pred_std_mm",
                     "mean_rel_err", "median_rel_err", "nrmse_range"]
    base_headline = base_df[[c for c in headline_cols if c in base_df.columns]].copy()
    base_headline["seed"] = ""
    base_headline["std"] = ""

    new_headline_rows = []
    for r in new_rows:
        nrow = {c: r.get(c) for c in headline_cols if c not in {"model"}}
        nrow["model"] = r["model"]
        nrow["seed"] = r["seed"]
        nrow["std"] = ""
        new_headline_rows.append(nrow)
    agg_headline = {c: agg_row.get(c) for c in headline_cols if c != "model"}
    agg_headline["model"] = agg_row["model"]
    agg_headline["seed"] = "mean(5)"
    agg_headline["std"] = (
        f"rmse±{agg_row['rmse_mm_std']:.2f} | "
        f"mae±{agg_row['mae_mm_std']:.2f} | "
        f"cov1±{agg_row['coverage_1sigma_std']:.3f}"
    )
    new_headline_rows.append(agg_headline)

    headline = pd.concat(
        [base_headline, pd.DataFrame(new_headline_rows)],
        ignore_index=True,
        sort=False,
    )
    headline_path = OUT_DIR / "headline_comparison.csv"
    headline.to_csv(headline_path, index=False)

    # Also write a markdown for quick visual inspection.
    md_lines = ["# External eval (n=795k clean) — ΔP^0.5 vs baselines", ""]
    md_lines.append(
        "| model | rmse_mm | mae_mm | bias_mm | p95_abs_err_mm | cov_1σ | cov_2σ |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for _, r in headline.iterrows():
        md_lines.append(
            f"| {r['model']} | "
            f"{r['rmse_mm']:.3f} | {r['mae_mm']:.3f} | {r['bias_mm']:.3f} | "
            f"{r['p95_abs_err_mm']:.3f} | "
            f"{r['coverage_1sigma']:.3f} | {r['coverage_2sigma']:.3f} |"
        )
    (OUT_DIR / "headline_comparison.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    print(f"\n=== Wrote {headline_path} ===")
    print("\n=== Aggregate (n=5 seeds) ===")
    for k in ["rmse_mm", "mae_mm", "bias_mm", "p95_abs_err_mm",
              "coverage_1sigma", "coverage_2sigma"]:
        print(
            f"  {k:>22s}: {agg_row[k]:.4f} ± {agg_row[f'{k}_std']:.4f}  "
            f"[{agg_row[f'{k}_ci_lo']:.4f}, {agg_row[f'{k}_ci_hi']:.4f}]"
        )

    print("\n=== Per-seed new MLP rows ===")
    for r in new_rows:
        print(
            f"  seed {r['seed']:>4}  {r['winner_name']:>22s}  "
            f"rmse={r['rmse_mm']:.3f}  mae={r['mae_mm']:.3f}  "
            f"bias={r['bias_mm']:.3f}  cov1={r['coverage_1sigma']:.3f}  "
            f"cov2={r['coverage_2sigma']:.3f}"
        )


if __name__ == "__main__":
    main()
