"""Interactive helpers for the packaged Masters-thesis workflow."""

from __future__ import annotations

import json
import os
from importlib import resources
from pathlib import Path

import cv2
import numpy as np


def default_state_dir() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    state_dir = root / "oscc-postprocessing" / "masters-thesis"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def ensure_spray_origins_file(origins_path: str | Path | None = None) -> Path:
    target = Path(origins_path) if origins_path is not None else default_state_dir() / "spray_origins.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        template = resources.files("OSCC_postprocessing.masters_thesis.resources").joinpath(
            "spray_origins.json"
        )
        target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def draw_freehand_mask(video_strip: np.ndarray, *, mask_output_path: str | Path) -> np.ndarray:
    nframes = video_strip.shape[0]
    frame = video_strip[nframes // 2]
    mask = edit_mask_overlay(frame, np.zeros(frame.shape[:2], dtype=np.uint8), window_name="Draw Mask")
    cv2.destroyAllWindows()
    mask_path = Path(mask_output_path)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    return mask


def edit_mask_overlay(
    frame: np.ndarray,
    initial_mask: np.ndarray,
    window_name: str = "Edit Mask",
) -> np.ndarray:
    mask = initial_mask.copy().astype(np.uint8)
    drawing_mode: str | None = None
    points: list[tuple[int, int]] = []

    def ensure_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    def apply_contour(target_mask: np.ndarray, contour_points: list[tuple[int, int]], value: bool) -> None:
        if len(contour_points) < 3:
            return
        contour = np.array(contour_points, dtype=np.int32)
        fill_value = 255 if value else 0
        cv2.fillPoly(target_mask, [contour], fill_value)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        nonlocal drawing_mode, points, mask

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_mode = "add"
            points = [(x, y)]
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing_mode = "remove"
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing_mode is not None:
            if not points or points[-1] != (x, y):
                points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP and drawing_mode == "add":
            if not points or points[-1] != (x, y):
                points.append((x, y))
            apply_contour(mask, points, value=True)
            drawing_mode = None
            points = []
        elif event == cv2.EVENT_RBUTTONUP and drawing_mode == "remove":
            if not points or points[-1] != (x, y):
                points.append((x, y))
            apply_contour(mask, points, value=False)
            drawing_mode = None
            points = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        overlay = ensure_bgr(frame)
        mask_bool = mask > 0
        overlay[mask_bool] = (
            0.55 * overlay[mask_bool] + 0.45 * np.array([0, 0, 255], dtype=np.float32)
        ).astype(np.uint8)

        if len(points) >= 2 and drawing_mode is not None:
            contour = np.array(points, dtype=np.int32)
            line_color = (0, 0, 255) if drawing_mode == "add" else (255, 0, 0)
            cv2.polylines(overlay, [contour], isClosed=False, color=line_color, thickness=2)

        cv2.imshow(window_name, overlay)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(40) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            mask[:] = 0
            drawing_mode = None
            points = []

    cv2.destroyWindow(window_name)
    return mask
