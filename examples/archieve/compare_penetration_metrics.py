"""Plot penetration_cdf, penetration_bw_x, penetration_bw_polar side-by-side
for every CSV under Mie_scattering_top_view_results/.

For each CSV (one cine recording), the three metrics are drawn in three
subplots; every plume is overlaid as one curve in each subplot. Output PNGs
mirror the source folder layout under images/penetration_compare/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "Mie_scattering_top_view_results"
OUTPUT_ROOT = REPO_ROOT / "images" / "penetration_compare"

METRICS = [
    ("penetration_cdf(mm)_plume_", "penetration_cdf [mm]"),
    ("penetration_bw_x(mm)_plume_", "penetration_bw_x [mm]"),
    ("penetration_bw_polar(mm)_plume_", "penetration_bw_polar [mm]"),
]


def load_fps(meta_path: Path) -> float | None:
    if not meta_path.exists():
        return None
    try:
        return float(json.loads(meta_path.read_text())["fps"])
    except Exception:
        return None


def plot_one(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if "frame_idx" not in df.columns:
        return

    fps = load_fps(csv_path.with_suffix(".meta.json"))
    if fps and fps > 0:
        t = df["frame_idx"].to_numpy() / fps * 1e3  # ms
        xlabel = "time [ms]"
    else:
        t = df["frame_idx"].to_numpy()
        xlabel = "frame_idx"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
    any_plotted = False
    for ax, (prefix, ylabel) in zip(axes, METRICS):
        cols = sorted(c for c in df.columns if c.startswith(prefix))
        for col in cols:
            plume_idx = col.rsplit("_", 1)[-1]
            ax.plot(t, df[col].to_numpy(), lw=1.0, label=f"plume {plume_idx}")
            any_plotted = True
        ax.set_title(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    if not any_plotted:
        plt.close(fig)
        return

    axes[0].set_ylabel("penetration [mm]")
    axes[-1].legend(fontsize=7, loc="lower right", ncol=2)
    rel = csv_path.relative_to(RESULTS_ROOT)
    fig.suptitle(str(rel), fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not RESULTS_ROOT.exists():
        raise SystemExit(f"Results folder not found: {RESULTS_ROOT}")

    csv_files = [
        p for p in RESULTS_ROOT.rglob("*.csv")
        if "boundary_points" not in p.parts and not p.name.endswith(".meta.json")
    ]
    print(f"Found {len(csv_files)} CSV files.")
    for i, csv_path in enumerate(sorted(csv_files), 1):
        rel = csv_path.relative_to(RESULTS_ROOT).with_suffix(".png")
        out_path = OUTPUT_ROOT / rel
        try:
            plot_one(csv_path, out_path)
        except Exception as exc:
            print(f"[{i}/{len(csv_files)}] FAILED {rel}: {exc}")
            continue
        if i % 25 == 0 or i == len(csv_files):
            print(f"[{i}/{len(csv_files)}] {rel}")
    print(f"Saved figures under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
