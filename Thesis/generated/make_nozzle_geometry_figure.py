"""Generate the Ch.03 nozzle-geometry overview figure.

Visualizes Table 3.1 (tab:nozzle-overview): hole diameter, hole count, and
umbrella angle for the nine nozzle families, plus schematics defining the
two geometric quantities. Writes Thesis/images/nozzle_geometry_overview.png.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PNG = PROJECT_ROOT / "Thesis" / "images" / "nozzle_geometry_overview.png"

# Data transcribed from Table 3.1 (tab:nozzle-overview) in Ch.03.
NOZZLES = ["Nozzle0", "Nozzle1", "Nozzle2", "Nozzle3", "Nozzle4",
           "Nozzle5", "Nozzle6", "Nozzle7", "Nozzle8"]
D_N = [0.384, 0.384, 0.375, 0.365, 0.355, 0.348, 0.333, 0.365, 0.365]
N_HOLES = [10, 10, 10, 10, 10, 11, 12, 12, 10]
THETA_UMB = [140, 140, 140, 140, 140, 140, 140, 164, 130]

REF_COLOR = "#9A5B38"      # Nozzle0 — 2022 reference injector
PROTO_COLOR = "#2F6FA3"    # Nozzle1-8 — 2024 prototypes
VARIANT_COLOR = "#6BA56B"  # geometry variants that differ from the 140°/10-hole core


def draw_side_view(ax: plt.Axes) -> None:
    """Side view: umbrella angle definition."""
    theta = 140.0
    half = math.radians(theta / 2.0)
    length = 1.0
    # injector body
    ax.add_patch(plt.Rectangle((-0.09, 0.0), 0.18, 0.42, color="#555555"))
    # spray axis (downwards)
    ax.plot([0, 0], [0.0, -1.05], ls=":", lw=1.0, color="#888888")
    # two opposing plumes at +-theta/2 from the downward axis
    for sign in (-1, 1):
        dx = sign * math.sin(half) * length
        dy = -math.cos(half) * length
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=PROTO_COLOR))
        # light spray cone fill around each plume
        spread = math.radians(7)
        for d_ang in np.linspace(-spread, spread, 7):
            a = sign * half + d_ang
            ax.plot([0, math.sin(a) * length], [0, -math.cos(a) * length],
                    lw=0.6, color=PROTO_COLOR, alpha=0.18)
    # angle arc between the two plumes (through the downward axis)
    ax.add_patch(Arc((0, 0), 0.9, 0.9, angle=0,
                     theta1=270 - theta / 2, theta2=270 + theta / 2,
                     color="#333333", lw=1.2))
    ax.text(0, -0.58, r"$\theta_{\mathrm{umb}}$", ha="center", va="top", fontsize=11)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Side view: umbrella angle", fontsize=9)


def draw_top_view(ax: plt.Axes) -> None:
    """Top view: N holes of diameter d_n equally spaced around the tip."""
    n = 10
    r_ring = 0.72
    ax.add_patch(Circle((0, 0), 0.18, color="#555555"))
    ax.add_patch(Circle((0, 0), r_ring, fill=False, ls=":", lw=0.9, color="#888888"))
    for k in range(n):
        a = 2 * math.pi * k / n
        x, y = r_ring * math.cos(a), r_ring * math.sin(a)
        ax.add_patch(Circle((x, y), 0.075, color=PROTO_COLOR))
    # annotate one hole with d_n
    a0 = 2 * math.pi * 1.5 / n
    x0, y0 = r_ring * math.cos(a0), r_ring * math.sin(a0)
    ax.annotate(r"$d_n$", xy=(x0, y0), xytext=(x0 + 0.42, y0 + 0.34),
                fontsize=11, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#333333"))
    ax.text(0, -1.06, rf"$N$ holes (here $N={n}$)", ha="center", fontsize=9)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.25, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Top view: hole pattern", fontsize=9)


def bar_panel(ax: plt.Axes, values, ylabel: str, ylim, fmt: str,
              variant_idx: set[int], rotate_labels: bool = False) -> None:
    colors = [REF_COLOR if i == 0 else (VARIANT_COLOR if i in variant_idx else PROTO_COLOR)
              for i in range(len(NOZZLES))]
    x = np.arange(len(NOZZLES))
    bars = ax.bar(x, values, color=colors, width=0.7)
    for rect, val in zip(bars, values):
        if rotate_labels:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    " " + fmt.format(val), ha="center", va="bottom", fontsize=6.5,
                    rotation=90)
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    fmt.format(val), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("Nozzle", "N") for n in NOZZLES], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def main() -> None:
    fig = plt.figure(figsize=(10.6, 5.6), dpi=200)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.28, wspace=0.3)

    ax_side = fig.add_subplot(gs[0, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    draw_side_view(ax_side)
    draw_top_view(ax_top)

    # legend panel
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REF_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=PROTO_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLOR),
    ]
    ax_leg.legend(handles,
                  ["Nozzle0 — 2022 reference",
                   "Nozzle1–8 — 2024 prototypes\n(140°, 10-hole core)",
                   "geometry variants\n(hole count / umbrella angle)"],
                  loc="center", frameon=False, fontsize=9)

    bar_panel(fig.add_subplot(gs[1, 0]), D_N, r"hole diameter $d_n$ (mm)",
              (0.30, 0.405), "{:.3f}", variant_idx=set(), rotate_labels=True)
    bar_panel(fig.add_subplot(gs[1, 1]), N_HOLES, r"hole count $N$",
              (0, 14), "{:d}", variant_idx={5, 6, 7})
    bar_panel(fig.add_subplot(gs[1, 2]), THETA_UMB, r"umbrella angle $\theta_{\mathrm{umb}}$ (°)",
              (120, 172), "{:d}", variant_idx={7, 8})

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
