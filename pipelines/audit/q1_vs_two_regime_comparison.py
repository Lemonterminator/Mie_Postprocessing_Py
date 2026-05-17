"""B.3 audit: compare production q1 fits against legacy two-regime fits.

This script expects a dual-fit archive produced by
``pipelines/fit/run_fit_pipeline.py --ablation-dual-fit`` after the current
refactor.  In that archive the production q1 columns remain the canonical
``log_k_quarter/log_t0/log_s`` fields, while the legacy blend is stored in
``*_two_regime`` columns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.archive_layout import SYNTHETIC_DATA_RUNS, append_manifest, resolve_latest

REQUIRED_TWO_COLS = {
    "log_k_sqrt_two_regime",
    "log_k_quarter_two_regime",
    "log_t0_two_regime",
    "log_s_two_regime",
    "success_two_regime",
}
REQUIRED_Q1_COLS = {"log_k_quarter", "log_t0", "log_s", "success"}
KEY_COLS = ["experiment_name", "folder", "file_path", "file_name", "file_stem", "plume_idx"]
np = None
pd = None


def _data_deps():
    global np, pd
    if np is None or pd is None:
        import numpy as _np
        import pandas as _pd

        np = _np
        pd = _pd
    return np, pd


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dualfit-run-dir", type=Path, default=None)
    parser.add_argument("--fit-run-dir", type=Path, default=None, help="Alias for --dualfit-run-dir.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--horizon-ms", type=float, default=5.0)
    return parser.parse_args()


def find_dualfit_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    latest = resolve_latest(SYNTHETIC_DATA_RUNS)
    if latest.name.endswith("_dualfit"):
        return latest
    candidates = sorted(
        [p for p in SYNTHETIC_DATA_RUNS.glob("*_dualfit") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No *_dualfit archive found. Run Phase 1 with --ablation-dual-fit.")


def sigmoid(x: np.ndarray) -> np.ndarray:
    _data_deps()
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def q1_model(log_k_quarter: float, log_t0: float, log_s: float, t_s: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t_s, dtype=float), 1e-9, None)
    kq = np.exp(float(log_k_quarter))
    t0 = np.exp(float(log_t0))
    s = np.exp(float(log_s))
    w = sigmoid((t - t0) / max(s, 1e-12))
    return w * kq * np.power(t, 0.25)


def two_regime_model(row: pd.Series, t_s: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t_s, dtype=float), 1e-9, None)
    ks = np.exp(float(row["log_k_sqrt_two_regime"]))
    kq = np.exp(float(row["log_k_quarter_two_regime"]))
    t0 = np.exp(float(row["log_t0_two_regime"]))
    s = np.exp(float(row["log_s_two_regime"]))
    w = sigmoid((t - t0) / max(s, 1e-12))
    return (1.0 - w) * ks * np.sqrt(t) + w * kq * np.power(t, 0.25)


def _read_clean_tables(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*/cdf/clean/*.csv")):
        df = pd.read_csv(path)
        df.insert(0, "folder", path.stem)
        df.insert(0, "experiment_name", path.parents[2].name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No cdf/clean CSVs found under {run_dir}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    missing = sorted((REQUIRED_Q1_COLS | REQUIRED_TWO_COLS) - set(out.columns))
    if missing:
        raise KeyError(
            "Dual-fit columns are missing: "
            + ", ".join(missing)
            + ". Re-run Phase 1 with --ablation-dual-fit after the dual-fit export patch."
        )
    return out


def _read_wide_tables(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("*/cdf/series_wide_clean/*.csv")):
        df = pd.read_csv(path)
        df.insert(0, "folder", path.stem)
        df.insert(0, "experiment_name", path.parents[2].name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _time_pen_cols(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    time_cols = sorted([c for c in row.index if c.startswith("time_ms_")])
    pen_cols = sorted([c for c in row.index if c.startswith("penetration_mm_")])
    t = pd.to_numeric(row[time_cols], errors="coerce").to_numpy(dtype=float) if time_cols else np.array([])
    y = pd.to_numeric(row[pen_cols], errors="coerce").to_numpy(dtype=float) if pen_cols else np.array([])
    n = min(len(t), len(y))
    mask = np.isfinite(t[:n]) & np.isfinite(y[:n])
    return t[:n][mask], y[:n][mask]


def build_ranked_cases(clean_df: pd.DataFrame, wide_df: pd.DataFrame, horizon_ms: float) -> pd.DataFrame:
    work = clean_df.copy()
    ok = work["success"].astype(str).str.lower().isin(["true", "1", "yes"]) & work["success_two_regime"].astype(str).str.lower().isin(["true", "1", "yes"])
    work = work.loc[ok].copy()
    horizon_s = float(horizon_ms) * 1e-3
    t_grid = np.linspace(1e-6, horizon_s, 256)
    rows = []
    for idx, row in work.iterrows():
        q1_curve = q1_model(row["log_k_quarter"], row["log_t0"], row["log_s"], t_grid)
        two_curve = two_regime_model(row, t_grid)
        q1_horizon = float(q1_model(row["log_k_quarter"], row["log_t0"], row["log_s"], np.array([horizon_s]))[0])
        two_horizon = float(two_regime_model(row, np.array([horizon_s]))[0])
        rows.append(
            {
                "row_index": int(idx),
                "experiment_name": row.get("experiment_name", ""),
                "folder": row.get("folder", ""),
                "file_name": row.get("file_name", ""),
                "file_stem": row.get("file_stem", ""),
                "plume_idx": row.get("plume_idx", ""),
                "q1_at_horizon_mm": q1_horizon,
                "two_regime_at_horizon_mm": two_horizon,
                "abs_diff_horizon_mm": abs(two_horizon - q1_horizon),
                "max_abs_diff_0_horizon_mm": float(np.nanmax(np.abs(two_curve - q1_curve))),
                "rmse_q1_mm": row.get("rmse", np.nan),
                "rmse_two_regime_mm": row.get("rmse_two_regime", np.nan),
            }
        )
    ranked = pd.DataFrame(rows).sort_values("abs_diff_horizon_mm", ascending=False).reset_index(drop=True)
    if wide_df.empty or ranked.empty:
        return ranked
    merge_cols = [c for c in KEY_COLS if c in wide_df.columns and c in clean_df.columns]
    if not merge_cols:
        return ranked
    return ranked.merge(clean_df.reset_index().rename(columns={"index": "row_index"}), on=["row_index"], how="left").merge(
        wide_df,
        on=merge_cols,
        how="left",
        suffixes=("", "_wide"),
    )


def write_plot(cases: pd.DataFrame, out_path: Path, *, top_n: int, horizon_ms: float) -> None:
    plt = _pyplot()
    selected = cases.head(top_n).copy()
    if selected.empty:
        raise ValueError("No converged q1/two-regime comparison cases were found.")
    fig, axes = plt.subplots(len(selected), 1, figsize=(8.0, 3.0 * len(selected)), dpi=170, squeeze=False)
    t_grid_ms = np.linspace(0.0, horizon_ms, 300)
    t_grid_s = np.clip(t_grid_ms * 1e-3, 1e-9, None)
    for ax, (_, row) in zip(axes[:, 0], selected.iterrows()):
        raw_t, raw_y = _time_pen_cols(row)
        if len(raw_t):
            ax.scatter(raw_t, raw_y, s=10, alpha=0.55, label="raw cleaned CDF")
        q1 = q1_model(row["log_k_quarter"], row["log_t0"], row["log_s"], t_grid_s)
        two = two_regime_model(row, t_grid_s)
        ax.plot(t_grid_ms, q1, linewidth=1.8, label="q1 production")
        ax.plot(t_grid_ms, two, linewidth=1.5, linestyle="--", label="two-regime legacy")
        ax.axvline(horizon_ms, color="0.25", linewidth=0.8, linestyle=":")
        title = (
            f"{row.get('experiment_name', '')} {row.get('folder', '')} "
            f"{row.get('file_stem', '')}/plume {row.get('plume_idx', '')}: "
            f"|Δ{horizon_ms:g}ms|={row['abs_diff_horizon_mm']:.1f} mm"
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Penetration (mm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(dualfit_run_dir: Path, out_dir: Path, *, top_n: int, horizon_ms: float) -> None:
    _data_deps()
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_df = _read_clean_tables(dualfit_run_dir)
    wide_df = _read_wide_tables(dualfit_run_dir)
    ranked = build_ranked_cases(clean_df, wide_df, horizon_ms)
    ranked.head(max(top_n, 20)).to_csv(out_dir / "q1_vs_two_regime_hardest_cases.csv", index=False)
    write_plot(ranked, out_dir / "q1_vs_two_regime_hardest.png", top_n=top_n, horizon_ms=horizon_ms)
    append_manifest(out_dir.parent, "q1_two_regime_cases", f"{out_dir.name}/q1_vs_two_regime_hardest_cases.csv", "B.3 hardest q1 vs two-regime cases")
    append_manifest(out_dir.parent, "q1_two_regime_figure", f"{out_dir.name}/q1_vs_two_regime_hardest.png", "B.3 q1 vs two-regime overlay")


def main() -> None:
    args = parse_args()
    dualfit_dir = find_dualfit_dir(args.dualfit_run_dir or args.fit_run_dir)
    run(dualfit_dir, args.out_dir, top_n=args.top_n, horizon_ms=args.horizon_ms)


if __name__ == "__main__":
    main()
