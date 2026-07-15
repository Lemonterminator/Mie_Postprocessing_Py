"""Shared matplotlib style for the eval-pipeline figure suite.

Colors follow the validated categorical palette (CVD-checked, light surface):
hues are assigned to *model families* in fixed order — never cycled — and
seeds within a family share the family hue at varied alpha/linestyle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Validated categorical palette, fixed order (light surface).
CATEGORICAL = (
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SINGLE_HUE = CATEGORICAL[0]
ACCENT_HUE = CATEGORICAL[1]
WARN_HUE = CATEGORICAL[2]
SEQUENTIAL_CMAP = "Blues"

RC = {
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "lines.linewidth": 2.0,
}


def apply_style() -> None:
    plt.rcParams.update(RC)


def family_color_map(families: Iterable[str]) -> dict[str, str]:
    """Fixed hue per family, in first-appearance order; >8 families fold to gray."""
    mapping: dict[str, str] = {}
    for family in families:
        if family in mapping:
            continue
        idx = len(mapping)
        mapping[family] = CATEGORICAL[idx] if idx < len(CATEGORICAL) else INK_MUTED
    return mapping


def save_fig(fig, path: Path, *, dpi: int = 200, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
    plt.close(fig)
    return written
