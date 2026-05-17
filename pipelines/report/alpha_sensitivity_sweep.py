"""E.2 audit: sweep the Gaussian cone-width alpha used in impingement screening.

The sweep is a post-processing pass over an ``impingement_frames.npz`` cache.
It reuses the saved tip path, piston-top profile, bore radius, and penetration
uncertainty, then recomputes wall/piston screening probabilities for each alpha.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipelines.common.latex_helpers import write_newcommands
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
    parser.add_argument("--frames-npz", type=Path, default=REPO_ROOT / "outputs" / "impingement_gui" / "latest" / "impingement_frames.npz")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.0])
    return parser.parse_args()


def _axis_from_npz(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, float]:
    x_axis = data["x_axis"].astype(float)
    if len(x_axis) < 2:
        raise ValueError("x_axis must contain at least two points.")
    grid = float(np.nanmedian(np.diff(x_axis)))
    canvas_h = float(data["canvas_h_mm"])
    y_axis = np.arange(int(round(canvas_h / grid)), dtype=float) * grid
    return x_axis, y_axis, grid


def _piston_mask_from_top(top_profile: np.ndarray, valid: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    mask = np.zeros((len(y_axis), len(top_profile)), dtype=bool)
    for col, ok in enumerate(valid):
        if not ok or not np.isfinite(top_profile[col]):
            continue
        mask[:, col] = y_axis >= float(top_profile[col])
    return mask


def recompute_for_alpha(data: np.lib.npyio.NpzFile, alpha: float) -> dict[str, float]:
    _data_deps()
    from MLP.impingement_core import gaussian_2d_rotated

    x_axis, y_axis, grid = _axis_from_npz(data)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")
    bore = float(data["cylinder_radius"])
    wall_mask = X >= bore
    time_ms = data["toy_time_ms"].astype(float)
    tip_x = data["tip_x_arr"].astype(float)
    tip_y = data["tip_y_arr"].astype(float)
    sigma_axis = np.maximum(data["sigma_axis_arr"].astype(float), 1.5 * grid)
    mu_plot = data["mu_plot_arr"].astype(float)
    half_cone = float(data["half_cone_rad"])
    tilt = float(data["tilt"])
    piston_top = data["piston_top_y_frames"].astype(float)
    piston_valid = data["piston_top_valid"].astype(bool)

    wall_prob = np.zeros_like(time_ms, dtype=float)
    piston_prob = np.zeros_like(time_ms, dtype=float)
    for i in range(len(time_ms)):
        sigma_ortho = max(max(float(mu_plot[i]), float(sigma_axis[i])) * np.tan(half_cone) / float(alpha), 2.0 * grid)
        pdf = gaussian_2d_rotated(X, Y, float(tip_x[i]), float(tip_y[i]), float(sigma_axis[i]), sigma_ortho, tilt)
        pdf /= pdf.sum() * grid**2 + 1e-30
        wall_prob[i] = float((wall_mask * pdf).sum() * grid**2)
        piston_mask = _piston_mask_from_top(piston_top[i], piston_valid[i], y_axis)
        piston_prob[i] = float((piston_mask * pdf).sum() * grid**2)

    return {
        "alpha": float(alpha),
        "peak_wall_prob": float(np.nanmax(wall_prob)),
        "peak_piston_prob": float(np.nanmax(piston_prob)),
        "p95_wall_prob": float(np.nanpercentile(wall_prob, 95)),
        "p95_piston_prob": float(np.nanpercentile(piston_prob, 95)),
        "cumulative_wall_prob_ms": float(np.trapz(wall_prob, time_ms)),
        "cumulative_piston_prob_ms": float(np.trapz(piston_prob, time_ms)),
        "fraction_wall_gt_1pct": float(np.mean(wall_prob > 0.01)),
        "fraction_piston_gt_1pct": float(np.mean(piston_prob > 0.01)),
    }


def write_plot(df: pd.DataFrame, path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=170)
    ax.plot(df["alpha"], df["peak_wall_prob"], marker="o", label="peak wall probability")
    ax.plot(df["alpha"], df["peak_piston_prob"], marker="s", label="peak piston probability")
    ax.axvline(3.0, color="0.25", linestyle="--", linewidth=1.0, label="chosen alpha=3")
    ax.set_xlabel("Cone-width alpha")
    ax.set_ylabel("Peak probability")
    ax.set_title("Impingement screening sensitivity to Gaussian cone width")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(frames_npz: Path, out_dir: Path, alphas: list[float]) -> None:
    _data_deps()
    if not frames_npz.exists():
        raise FileNotFoundError(f"Missing impingement frame cache: {frames_npz}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(frames_npz) as data:
        rows = [recompute_for_alpha(data, alpha) for alpha in alphas]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "alpha_sensitivity_screening_rate.csv", index=False)
    write_plot(df, out_dir / "alpha_sensitivity_curves.png")
    chosen = df.iloc[(df["alpha"] - 3.0).abs().argsort()].iloc[0]
    write_newcommands(
        out_dir / "alpha_sensitivity_summary.tex",
        {
            "alphaSensitivityChosenAlpha": f"{float(chosen['alpha']):.1f}",
            "alphaSensitivityPeakWallPct": f"{100.0 * float(chosen['peak_wall_prob']):.2f}",
            "alphaSensitivityPeakPistonPct": f"{100.0 * float(chosen['peak_piston_prob']):.2f}",
        },
    )


def main() -> None:
    args = parse_args()
    run(args.frames_npz, args.out_dir, args.alphas)


if __name__ == "__main__":
    main()
