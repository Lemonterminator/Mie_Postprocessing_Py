"""Visual check for GUI-calibrated centre alignment on a Cine frame.

Example:
    python examples/verify_gui_centre_alignment.py ^
        --cine "G:\\data\\test.cine" ^
        --config "G:\\data\\config.json" ^
        --frame 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OSCC_postprocessing.cine.cine_utils import CineReader


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_frame(frame_u16: np.ndarray) -> np.ndarray:
    """Robustly scale one frame to [0, 1] for plotting."""
    frame = frame_u16.astype(np.float32, copy=False)
    low = float(np.percentile(frame, 1.0))
    high = float(np.percentile(frame, 99.5))
    if high <= low:
        return np.zeros_like(frame, dtype=np.float32)
    scaled = (frame - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cine", required=True, type=Path, help="Path to the input .cine file.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the GUI-generated config.json.")
    parser.add_argument("--frame", type=int, default=0, help="0-based frame index to display.")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=80.0,
        help="Scatter marker size for the centre point.",
    )
    parser.add_argument(
        "--x-shift",
        type=float,
        default=0.0,
        help="Optional extra x-shift for quick visual debugging.",
    )
    parser.add_argument(
        "--y-shift",
        type=float,
        default=0.0,
        help="Optional extra y-shift for quick visual debugging.",
    )
    parser.add_argument(
        "--hide-radii",
        action="store_true",
        help="Do not draw inner/outer radius circles from config.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure instead of only showing it.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = _load_config(args.config)

    reader = CineReader()
    reader.load(str(args.cine))

    frame_idx = int(np.clip(args.frame, 0, max(0, reader.frame_count - 1)))
    frame = reader.read_frame(frame_idx)
    frame_img = _normalize_frame(frame)

    centre_x = float(config["centre_x"]) + float(args.x_shift)
    centre_y = float(config["centre_y"]) + float(args.y_shift)
    inner_radius = float(config.get("inner_radius", 0.0))
    outer_radius = float(config.get("outer_radius", 0.0))
    offset = float(config.get("offset", 0.0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame_img, cmap="gray", origin="upper")
    ax.scatter(
        [centre_x],
        [centre_y],
        s=float(args.marker_size),
        c="red",
        marker="x",
        linewidths=2.0,
        label="centre",
    )

    # Crosshair makes sub-pixel misalignment easier to see than a point alone.
    ax.axvline(centre_x, color="red", alpha=0.35, linewidth=1.0)
    ax.axhline(centre_y, color="red", alpha=0.35, linewidth=1.0)

    if not args.hide_radii:
        if outer_radius > 0:
            ax.add_patch(plt.Circle((centre_x, centre_y), outer_radius, color="cyan", fill=False, linewidth=1.0))
        if inner_radius > 0:
            ax.add_patch(plt.Circle((centre_x, centre_y), inner_radius, color="yellow", fill=False, linewidth=1.0))

    ax.set_title(
        f"{args.cine.name} | frame={frame_idx} | centre=({centre_x:.2f}, {centre_y:.2f}) | offset={offset:.2f}"
    )
    ax.legend(loc="upper right")
    ax.set_xlim(0, frame.shape[1] - 1)
    ax.set_ylim(frame.shape[0] - 1, 0)
    ax.set_aspect("equal")
    plt.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=160)
        print(f"Saved overlay figure to {args.save}")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
