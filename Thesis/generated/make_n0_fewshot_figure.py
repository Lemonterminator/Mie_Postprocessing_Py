"""Generate the Nozzle0 few-shot adaptation figure.

Reads the NLL- and MSE-loss adaptation curves and renders RMSE-vs-k and
bias-vs-k panels with the in-family LONO band and the zero-shot baseline for
reference, into the slides figs/ folder.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLIDE_DIR = PROJECT_ROOT / "Thesis" / "slides" / "slides_residual_family_head_production"
DATA_DIR = SLIDE_DIR / "n0_fewshot_adaptation"
OUT_PNG = SLIDE_DIR / "figs" / "n0_fewshot_adaptation_curve.png"

# Reference bands from the LONO study (project_lono_residual_recipe / winner notes).
IN_FAMILY_RMSE = 5.16          # residual_fh 4-fold (excl N6) LONO mean
ZERO_SHOT_RMSE = 33.16         # N0 held-out zero-shot

CURVES = [
    ("fewshot_curve_mse.csv", "MSE adapt (mean direct)", "#2F6FA3"),
    ("fewshot_curve_nll.csv", "NLL adapt (frozen sigma)", "#C05050"),
]


def _xpos(curve: pd.DataFrame) -> tuple[list[int], list[str]]:
    ks = list(curve["k"].astype(int))
    labels = [("all" if k == ks[-1] and k > 100 else str(k)) for k in ks]
    return list(range(len(ks))), labels


def main() -> None:
    frames = {name: pd.read_csv(DATA_DIR / name) for name, _, _ in CURVES}
    ref = frames[CURVES[0][0]]
    x, labels = _xpos(ref)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=180)

    # ── left: RMSE vs k with error band ─────────────────────────────────────
    ax = axes[0]
    ax.axhline(ZERO_SHOT_RMSE, color="#888888", linestyle=":", linewidth=1.1,
               label=f"zero-shot ({ZERO_SHOT_RMSE:.0f} mm)")
    ax.axhline(IN_FAMILY_RMSE, color="#6BA56B", linestyle="--", linewidth=1.2,
               label=f"in-family LONO band (~{IN_FAMILY_RMSE:.1f} mm)")
    for name, lab, col in CURVES:
        df = frames[name]
        xv, _ = _xpos(df)
        mean = df["rmse_mean"].to_numpy()
        std = df["rmse_std"].to_numpy()
        ax.plot(xv, mean, "-o", color=col, label=lab, linewidth=1.8, markersize=5)
        ax.fill_between(xv, mean - std, mean + std, color=col, alpha=0.18, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("adaptation budget k (N0 conditions)")
    ax.set_ylabel("uncensored RMSE [mm]")
    ax.set_title("Nozzle0 few-shot: RMSE vs k")
    ax.set_ylim(0, 36)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── right: bias vs k ────────────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    for name, lab, col in CURVES:
        df = frames[name]
        xv, _ = _xpos(df)
        ax.plot(xv, df["bias_mean"].to_numpy(), "-o", color=col, label=lab,
                linewidth=1.8, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("adaptation budget k (N0 conditions)")
    ax.set_ylabel("bias [mm]")
    ax.set_title("Nozzle0 few-shot: bias vs k")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
