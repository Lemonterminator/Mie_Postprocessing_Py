from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HA_OUTPUTS = PROJECT_ROOT / "MLP" / "baseline" / "Hiroyasu_Arai" / "outputs"
NS_OUTPUTS = PROJECT_ROOT / "MLP" / "baseline" / "Naber_Siebers" / "outputs"
MLP_EVAL_ROOT = PROJECT_ROOT / "MLP" / "eval"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: str
    run_dir: Path
    color: str


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def safe_name(text: str, max_len: int = 170) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return out[:max_len] or "condition"


def display_dataset_key(value: Any) -> str:
    text = str(value)
    return "nozzle0" if text.lower() == "ds300" else text


def latest_baseline_run(variant: str, outputs_root: Path = HA_OUTPUTS, primary_split: str = "all") -> Path:
    candidates: list[Path] = []
    for metrics_path in outputs_root.glob("*/metrics_summary.json"):
        try:
            payload = read_json(metrics_path)
        except Exception:
            continue
        if payload.get("variant") != variant:
            continue
        if payload.get("primary_eval_split") != primary_split:
            continue
        if not (metrics_path.parent / "points.csv").exists():
            continue
        candidates.append(metrics_path.parent)
    if not candidates:
        raise FileNotFoundError(f"No {variant!r} run with primary_eval_split={primary_split!r} under {outputs_root}")
    return max(candidates, key=lambda p: (p / "metrics_summary.json").stat().st_mtime)


def latest_ns_run(outputs_root: Path = NS_OUTPUTS, primary_split: str = "all") -> Path:
    return latest_baseline_run("ns_delay", outputs_root=outputs_root, primary_split=primary_split)


def latest_winner_eval(eval_root: Path = MLP_EVAL_ROOT) -> Path:
    candidates = [p.parent for p in eval_root.glob("rmse_eval_clean_*_winner_full/metrics_summary.json")]
    candidates = [p for p in candidates if (p / "points.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No winner_full eval with points.csv found under {eval_root}")
    return max(candidates, key=lambda p: (p / "metrics_summary.json").stat().st_mtime)


def load_points(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "points.csv"
    columns = [
        "dataset_key",
        "test_name",
        "injection_duration_us",
        "time_ms",
        "pen_true_mm",
        "pen_pred_mm",
        "pen_std_mm",
        "resid_mm",
        "traj_key",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False)
    missing = sorted(set(columns[:-1]) - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["dataset_key"] = df["dataset_key"].map(display_dataset_key)
    df["test_name"] = df["test_name"].astype(str)
    df["injection_duration_us"] = pd.to_numeric(df["injection_duration_us"], errors="coerce")
    for col in ["time_ms", "pen_true_mm", "pen_pred_mm", "pen_std_mm", "resid_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_ms", "pen_true_mm", "pen_pred_mm", "pen_std_mm", "resid_mm"])
    return df.reset_index(drop=True)


def add_condition_columns(df: pd.DataFrame, time_bin_ms: float) -> pd.DataFrame:
    out = df.copy()
    duration = out["injection_duration_us"].round(6).astype(str)
    out["condition_key"] = out["dataset_key"] + "|" + out["test_name"] + "|dur=" + duration
    out["time_bin"] = np.floor(out["time_ms"].to_numpy(dtype=float) / float(time_bin_ms)).astype(int)
    out["time_bin_center_ms"] = (out["time_bin"].to_numpy(dtype=float) + 0.5) * float(time_bin_ms)
    return out


def finite_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    truth = df["pen_true_mm"].to_numpy(dtype=float)
    pred = df["pen_pred_mm"].to_numpy(dtype=float)
    std = np.maximum(df["pen_std_mm"].to_numpy(dtype=float), 1e-12)
    resid = pred - truth
    abs_err = np.abs(resid)
    truth_range = float(np.max(truth) - np.min(truth)) if len(truth) else float("nan")
    return {
        "n_points": int(len(df)),
        "n_trajectories": int(df["traj_key"].nunique()) if "traj_key" in df.columns else 0,
        "rmse_mm": float(np.sqrt(np.mean(resid * resid))) if len(df) else float("nan"),
        "mae_mm": float(np.mean(abs_err)) if len(df) else float("nan"),
        "bias_mm": float(np.mean(resid)) if len(df) else float("nan"),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)) if len(df) else float("nan"),
        "coverage_1sigma": float(np.mean(abs_err <= std)) if len(df) else float("nan"),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std)) if len(df) else float("nan"),
        "mean_pred_std_mm": float(np.mean(std)) if len(df) else float("nan"),
        "nrmse_range": float(np.sqrt(np.mean(resid * resid)) / truth_range) if len(df) and truth_range > 0 else float("nan"),
    }


def binned_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    def rmse(x: pd.Series) -> float:
        arr = x.to_numpy(dtype=float)
        return float(np.sqrt(np.mean(arr * arr)))

    grouped = df.groupby(["condition_key", "time_bin", "time_bin_center_ms"], dropna=False)
    out = grouped.agg(
        n_points=("pen_true_mm", "size"),
        truth_mean_mm=("pen_true_mm", "mean"),
        truth_std_mm=("pen_true_mm", "std"),
        pred_mean_mm=("pen_pred_mm", "mean"),
        pred_std_mean_mm=("pen_std_mm", "mean"),
        resid_mean_mm=("resid_mm", "mean"),
        resid_rmse_mm=("resid_mm", rmse),
    ).reset_index()
    return out


def condition_metrics(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby("condition_key", dropna=False):
        first = group.iloc[0]
        row = {
            "model": model_label,
            "condition_key": key,
            "dataset_key": first["dataset_key"],
            "test_name": first["test_name"],
            "injection_duration_us": float(first["injection_duration_us"]),
        }
        row.update(finite_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def write_overall_tables(specs: list[ModelSpec], points_by_model: dict[str, pd.DataFrame], out_dir: Path) -> None:
    rows = []
    for spec in specs:
        summary_path = spec.run_dir / "metrics_summary.json"
        payload = read_json(summary_path) if summary_path.exists() else {}
        overall = payload.get("overall") or finite_metrics(points_by_model[spec.label])
        row = {"model": spec.label, "kind": spec.kind, "run_dir": str(spec.run_dir)}
        for key in [
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
        ]:
            if key in overall:
                row[key] = overall[key]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "overall_metrics.csv", index=False)


def plot_pred_vs_actual_grid(specs: list[ModelSpec], points_by_model: dict[str, pd.DataFrame], out_path: Path, max_points: int) -> None:
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.6), dpi=170, sharex=True, sharey=True)
    if len(specs) == 1:
        axes = [axes]
    low = min(float(points_by_model[s.label]["pen_true_mm"].min()) for s in specs)
    high = max(float(points_by_model[s.label]["pen_true_mm"].max()) for s in specs)
    pred_low = min(float(points_by_model[s.label]["pen_pred_mm"].min()) for s in specs)
    pred_high = max(float(points_by_model[s.label]["pen_pred_mm"].max()) for s in specs)
    low = min(low, pred_low)
    high = max(high, pred_high)
    for ax, spec in zip(axes, specs):
        df = points_by_model[spec.label]
        sample = df if len(df) <= max_points else df.sample(max_points, random_state=42)
        ax.scatter(sample["pen_true_mm"], sample["pen_pred_mm"], s=2, alpha=0.16, linewidths=0, color=spec.color)
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        m = finite_metrics(df)
        ax.set_title(f"{spec.label}\nRMSE={m['rmse_mm']:.2f} mm, P95={m['p95_abs_err_mm']:.2f} mm")
        ax.set_xlabel("Measured penetration [mm]")
        ax.grid(True, alpha=0.23)
    axes[0].set_ylabel("Predicted penetration [mm]")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_residual_hist_grid(specs: list[ModelSpec], points_by_model: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 4.1), dpi=170, sharex=True)
    if len(specs) == 1:
        axes = [axes]
    all_resid = np.concatenate([points_by_model[s.label]["resid_mm"].to_numpy(dtype=float) for s in specs])
    finite = all_resid[np.isfinite(all_resid)]
    lo, hi = np.quantile(finite, [0.005, 0.995])
    bins = np.linspace(float(lo), float(hi), 90)
    for ax, spec in zip(axes, specs):
        df = points_by_model[spec.label]
        ax.hist(df["resid_mm"], bins=bins, color=spec.color, alpha=0.86)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        m = finite_metrics(df)
        ax.set_title(f"{spec.label}\nbias={m['bias_mm']:.2f} mm")
        ax.set_xlabel("Residual [mm]")
        ax.grid(True, axis="y", alpha=0.20)
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_condition(
    *,
    condition_key: str,
    specs: list[ModelSpec],
    points_by_model: dict[str, pd.DataFrame],
    bins_by_model: dict[str, pd.DataFrame],
    out_path: Path,
    scatter_max_points: int,
) -> dict[str, Any]:
    base = points_by_model[specs[-1].label]
    truth = base.loc[base["condition_key"] == condition_key].copy()
    if truth.empty:
        truth = points_by_model[specs[0].label].loc[points_by_model[specs[0].label]["condition_key"] == condition_key].copy()
    if len(truth) > scatter_max_points:
        truth_scatter = truth.sample(scatter_max_points, random_state=7)
    else:
        truth_scatter = truth

    first = truth.iloc[0]
    title = (
        f"{first['dataset_key']} | {first['test_name']} | "
        f"inj={float(first['injection_duration_us']):.0f} us"
    )
    truth_mean = (
        truth.groupby("time_bin_center_ms", dropna=False)["pen_true_mm"]
        .mean()
        .reset_index(name="truth_mean_mm")
        .sort_values("time_bin_center_ms")
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=155)
    ax.scatter(
        truth_scatter["time_ms"],
        truth_scatter["pen_true_mm"],
        s=8,
        color="black",
        alpha=0.12,
        linewidths=0,
        label="Measured points",
    )
    ax.plot(
        truth_mean["time_bin_center_ms"],
        truth_mean["truth_mean_mm"],
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Measured mean",
    )

    manifest: dict[str, Any] = {
        "condition_key": condition_key,
        "dataset_key": first["dataset_key"],
        "test_name": first["test_name"],
        "injection_duration_us": float(first["injection_duration_us"]),
        "plot_path": str(out_path),
    }
    for spec in specs:
        pts = points_by_model[spec.label]
        condition_points = pts.loc[pts["condition_key"] == condition_key]
        if condition_points.empty:
            continue
        metrics = finite_metrics(condition_points)
        b = bins_by_model[spec.label]
        b = b.loc[b["condition_key"] == condition_key].sort_values("time_bin_center_ms")
        x = b["time_bin_center_ms"].to_numpy(dtype=float)
        mu = b["pred_mean_mm"].to_numpy(dtype=float)
        sigma = b["pred_std_mean_mm"].to_numpy(dtype=float)
        label = (
            f"{spec.label} "
            f"cov={100.0 * metrics['coverage_1sigma']:.0f}/{100.0 * metrics['coverage_2sigma']:.0f}%"
        )
        ax.plot(x, mu, color=spec.color, linewidth=1.55, label=label)
        ax.fill_between(x, mu - sigma, mu + sigma, color=spec.color, alpha=0.08, linewidth=0)
        manifest[f"{safe_name(spec.label)}_rmse_mm"] = metrics["rmse_mm"]
        manifest[f"{safe_name(spec.label)}_coverage_1sigma"] = metrics["coverage_1sigma"]
        manifest[f"{safe_name(spec.label)}_coverage_2sigma"] = metrics["coverage_2sigma"]

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Penetration [mm]")
    ax.set_title(title)
    ax.grid(True, alpha=0.24)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return manifest


def select_conditions(condition_table: pd.DataFrame, max_conditions: int | None, min_points: int) -> list[str]:
    table = condition_table.loc[condition_table["n_points"] >= int(min_points)].copy()
    table = table.sort_values(["dataset_key", "test_name", "injection_duration_us", "condition_key"])
    keys = table["condition_key"].astype(str).tolist()
    if max_conditions is not None and max_conditions >= 0:
        keys = keys[: int(max_conditions)]
    return keys


def write_html_index(manifest: pd.DataFrame, out_dir: Path, title: str) -> None:
    rows = []
    metric_cols = [c for c in manifest.columns if c.endswith("_rmse_mm")]
    metric_cols.sort()
    for _, row in manifest.iterrows():
        plot_path = Path(str(row["plot_path"]))
        rel = plot_path.relative_to(out_dir).as_posix() if plot_path.is_relative_to(out_dir) else plot_path.as_posix()
        metrics = " | ".join(
            f"{col.replace('_rmse_mm', '')}: {float(row[col]):.2f} mm"
            for col in metric_cols
            if pd.notna(row.get(col))
        )
        rows.append(
            "<tr>"
            f"<td><a href='{rel}'>{row['dataset_key']}</a></td>"
            f"<td>{row['test_name']}</td>"
            f"<td>{float(row['injection_duration_us']):.0f}</td>"
            f"<td>{metrics}</td>"
            f"<td><img src='{rel}' loading='lazy'></td>"
            "</tr>"
        )
    html = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            f"<title>{title}</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:20px;color:#222}",
            "table{border-collapse:collapse;width:100%}",
            "td,th{border-bottom:1px solid #ddd;padding:6px;vertical-align:top}",
            "th{position:sticky;top:0;background:#fff;text-align:left}",
            "img{width:420px;max-width:45vw}",
            "</style></head><body>",
            f"<h1>{title}</h1>",
            f"<p>Total condition plots: {len(manifest)}</p>",
            "<table>",
            "<thead><tr><th>Dataset</th><th>Test</th><th>Injection us</th><th>RMSE summary</th><th>Plot</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table></body></html>",
        ]
    )
    (out_dir / "condition_plot_index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Stage-3 MLP against physics baselines using condition-level plots.")
    p.add_argument(
        "--baseline-set",
        choices=("ha_zhou", "ha_ns"),
        default="ha_ns",
        help="ha_ns compares H-A, Naber-Siebers, and MLP; ha_zhou compares H-A, Zhou, and MLP.",
    )
    p.add_argument("--ha-run-dir", type=Path, default=None)
    p.add_argument("--zhou-run-dir", type=Path, default=None)
    p.add_argument("--ns-run-dir", type=Path, default=None)
    p.add_argument("--mlp-eval-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--time-bin-ms", type=float, default=0.05)
    p.add_argument("--max-condition-plots", type=int, default=None, help="Default: all conditions. Use a small value for smoke tests.")
    p.add_argument("--min-condition-points", type=int, default=8)
    p.add_argument("--scatter-max-points", type=int, default=6000)
    p.add_argument("--pred-actual-sample", type=int, default=90000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ha_dir = resolve_path(args.ha_run_dir) or latest_baseline_run("ha_calibrated")
    mlp_dir = resolve_path(args.mlp_eval_dir) or latest_winner_eval()
    if args.baseline_set == "ha_ns":
        ns_dir = resolve_path(args.ns_run_dir) or latest_ns_run()
        default_name = f"stage3_vs_HA_NS_condition_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        specs = [
            ModelSpec("Hiroyasu-Arai calibrated", "ha_calibrated", ha_dir, "#8f6bb3"),
            ModelSpec("Naber-Siebers delay", "ns_delay", ns_dir, "#c44e52"),
            ModelSpec("Stage-3 MLP anchor_off", "stage3_mlp", mlp_dir, "#4878d0"),
        ]
        html_title = "Stage-3 vs H-A vs Naber-Siebers condition plots"
    else:
        zhou_dir = resolve_path(args.zhou_run_dir) or latest_baseline_run("zhou_calibrated")
        default_name = f"stage3_ha_zhou_condition_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        specs = [
            ModelSpec("Hiroyasu-Arai calibrated", "ha_calibrated", ha_dir, "#8f6bb3"),
            ModelSpec("Zhou calibrated", "zhou_calibrated", zhou_dir, "#c44e52"),
            ModelSpec("Stage-3 MLP anchor_off", "stage3_mlp", mlp_dir, "#4878d0"),
        ]
        html_title = "Stage-3 vs H-A vs Zhou condition plots"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = resolve_path(args.output_dir) or (DEFAULT_OUTPUT_ROOT / default_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_dir = out_dir / "condition_plots"
    condition_dir.mkdir(exist_ok=True)

    points_by_model: dict[str, pd.DataFrame] = {}
    bins_by_model: dict[str, pd.DataFrame] = {}
    condition_metric_frames: list[pd.DataFrame] = []
    for spec in specs:
        df = add_condition_columns(load_points(spec.run_dir), time_bin_ms=float(args.time_bin_ms))
        points_by_model[spec.label] = df
        bins_by_model[spec.label] = binned_condition_summary(df)
        condition_metric_frames.append(condition_metrics(df, spec.label))

    write_overall_tables(specs, points_by_model, out_dir)
    pd.concat(condition_metric_frames, ignore_index=True).to_csv(out_dir / "per_condition_metrics.csv", index=False)
    plot_pred_vs_actual_grid(specs, points_by_model, out_dir / "pred_vs_actual_side_by_side.png", int(args.pred_actual_sample))
    plot_residual_hist_grid(specs, points_by_model, out_dir / "residual_histogram_side_by_side.png")

    condition_basis = condition_metrics(points_by_model[specs[-1].label], specs[-1].label)
    condition_keys = select_conditions(condition_basis, args.max_condition_plots, int(args.min_condition_points))
    manifest_rows = []
    for idx, key in enumerate(condition_keys, start=1):
        cond = condition_basis.loc[condition_basis["condition_key"] == key].iloc[0]
        name = safe_name(
            f"{idx:04d}_{cond['dataset_key']}_{cond['test_name']}_inj_{float(cond['injection_duration_us']):.0f}us"
        )
        out_path = condition_dir / f"{name}.png"
        manifest_rows.append(
            plot_condition(
                condition_key=key,
                specs=specs,
                points_by_model=points_by_model,
                bins_by_model=bins_by_model,
                out_path=out_path,
                scatter_max_points=int(args.scatter_max_points),
            )
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "condition_plot_manifest.csv", index=False)
    write_html_index(manifest, out_dir, title=html_title)

    run_manifest = {
        "baseline_set": str(args.baseline_set),
        "output_dir": str(out_dir),
        "time_bin_ms": float(args.time_bin_ms),
        "n_condition_plots": int(len(manifest_rows)),
        "models": [{"label": s.label, "kind": s.kind, "run_dir": str(s.run_dir)} for s in specs],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"Wrote comparison plots to {out_dir}")
    print(f"Condition plots: {len(manifest_rows)}")
    print(out_dir / "pred_vs_actual_side_by_side.png")
    print(out_dir / "residual_histogram_side_by_side.png")
    print(condition_dir)


if __name__ == "__main__":
    main()
