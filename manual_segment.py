import csv
import json
import os
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from OSCC_postprocessing.utils import resolve_sam2_paths

from OSCC_postprocessing.cine.cine_utils import CineReader
from OSCC_postprocessing.analysis.cone_angle import angle_signal_density
from OSCC_postprocessing.rotation.rotate_crop import generate_CropRect
from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
    rotate_video_nozzle_at_0_half_numpy,
)
from OSCC_postprocessing.binary_ops.functions_bw import (
    keep_largest_component,
    keep_largest_component_cuda,
)
from OSCC_postprocessing.binary_ops.masking import generate_plume_mask
from OSCC_postprocessing.utils.zoom_utils import enlarge_image

try:
    import cupy as cp
    from OSCC_postprocessing.rotation.rotate_with_alignment import (
        rotate_video_nozzle_at_0_half_cupy,
    )

    _ = cp.cuda.runtime.getDeviceCount()
    ROTATE_BACKEND = "cupy"
except Exception:
    cp = None
    rotate_video_nozzle_at_0_half_cupy = None
    ROTATE_BACKEND = "numpy"

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
except Exception:
    torch = None
    build_sam2 = None
    SAM2ImagePredictor = None
    SAM2_AVAILABLE = False

class ManualSegmenter:
    """Interactive tool to create and export pixel-wise masks for rotated plume videos."""

    INDEX_FIELDS = [
        "sample_id",
        "video_id",
        "video_path",
        "frame_idx",
        "plume_idx",
        "mask_area_px",
        "mask_area_ratio",
        "qa_flags",
        "timestamp_saved",
        "image_raw",
        "image_disp",
        "mask_path",
        "meta_path",
    ]

    REVIEW_FIELDS = [
        "sample_id",
        "video_id",
        "frame_idx",
        "plume_idx",
        "qa_flags",
        "timestamp_saved",
    ]

    def __init__(self, master):
        self.master = master
        master.title("Manual Segmenter")

        self.reader = None
        self.video_path = None
        self.video = None  # Original video frames
        self.plume_videos = []  # Rotated plume-specific videos
        self.plume_masks = []  # Per-plume, per-frame masks
        self.current_img = None  # Current display-processed frame (uint8)
        self.current_raw = None  # Current raw rotated frame (float)
        self.current_frame = 0
        self.current_plume = 0
        self.zoom = 1
        self.tool = "grabcut"
        self.brush_size = tk.IntVar(value=5)
        self.start_pos = None
        self.last_pos = None
        self.live_rect_id = None

        self.gain = tk.DoubleVar(value=1.0)
        self.gamma = tk.DoubleVar(value=1.0)
        self.grabcut_iters = tk.IntVar(value=3)
        self.sam_predictor = None
        self.sam_model = None
        self.sam_model_key = None
        self.sam_device = None
        self.sam_points_pos = []
        self.sam_points_neg = []
        self.sam_box = None
        self.sam_dragging = False

        self.dataset_root = None
        self.dataset_dirs = {}
        self.overwrite_existing = tk.BooleanVar(value=False)
        self.include_empty_masks = tk.BooleanVar(value=False)
        self.tool_version = "manual_segmenter_v2"
        self.mask_area_warn_ratio = 5e-4
        self.saturation_warn_ratio = 0.25

        self.config_path = None
        self.config_values = {}
        self.n_plumes = None
        self.centre_x = None
        self.centre_y = None
        self.inner_radius = 0
        self.outer_radius = 0
        self.rotation_offset_deg = 0.0

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(top)
        row1.pack(side=tk.TOP, fill=tk.X)
        row2 = ttk.Frame(top)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        ttk.Button(row1, text="Load Video", command=self.load_video).pack(side=tk.LEFT)
        ttk.Button(row1, text="Load Config", command=self.load_config).pack(side=tk.LEFT)
        ttk.Button(row1, text="No Config", command=self.load_no_config).pack(side=tk.LEFT)
        ttk.Button(row1, text="Prev Frame", command=self.prev_frame).pack(side=tk.LEFT)
        ttk.Button(row1, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT)
        ttk.Button(row1, text="Prev Plume", command=self.prev_plume).pack(side=tk.LEFT)
        ttk.Button(row1, text="Next Plume", command=self.next_plume).pack(side=tk.LEFT)
        ttk.Button(row1, text="Brush Tool", command=lambda: self.set_tool("brush")).pack(
            side=tk.LEFT
        )
        ttk.Button(row1, text="GrabCut Tool", command=lambda: self.set_tool("grabcut")).pack(
            side=tk.LEFT
        )
        ttk.Button(row1, text="Load SAM", command=self.load_sam_model).pack(side=tk.LEFT)
        ttk.Button(row1, text="SAM Tool", command=lambda: self.set_tool("sam")).pack(side=tk.LEFT)
        ttk.Button(row1, text="Apply SAM", command=self.apply_sam_current).pack(side=tk.LEFT)
        ttk.Button(row1, text="Clear SAM", command=self.clear_sam_prompts).pack(side=tk.LEFT)
        ttk.Label(row1, text="Brush").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(
            row1,
            from_=1,
            to=200,
            increment=1,
            width=5,
            textvariable=self.brush_size,
        ).pack(side=tk.LEFT)

        ttk.Label(row1, text="Gain").pack(side=tk.LEFT, padx=(10, 0))
        gain_box = ttk.Spinbox(
            row1,
            from_=0.1,
            to=10,
            increment=0.1,
            width=6,
            textvariable=self.gain,
            command=self.update_image,
        )
        gain_box.pack(side=tk.LEFT)
        ttk.Label(row1, text="Gamma").pack(side=tk.LEFT)
        gamma_box = ttk.Spinbox(
            row1,
            from_=0.1,
            to=3,
            increment=0.05,
            width=6,
            textvariable=self.gamma,
            command=self.update_image,
        )
        gamma_box.pack(side=tk.LEFT)
        gain_box.bind("<Return>", lambda _e: self.update_image())
        gain_box.bind("<FocusOut>", lambda _e: self.update_image())
        gamma_box.bind("<Return>", lambda _e: self.update_image())
        gamma_box.bind("<FocusOut>", lambda _e: self.update_image())

        ttk.Button(row2, text="Set Dataset Dir", command=self.set_dataset_root).pack(side=tk.LEFT)
        ttk.Button(row2, text="Save Current", command=self.save_current_sample).pack(side=tk.LEFT)
        ttk.Button(row2, text="Save Plume", command=self.save_current_plume_labeled).pack(
            side=tk.LEFT
        )
        ttk.Button(row2, text="Save All Labeled", command=self.save_all_labeled).pack(
            side=tk.LEFT
        )
        ttk.Button(row2, text="Generate Splits", command=self.generate_splits).pack(side=tk.LEFT)
        ttk.Checkbutton(row2, text="Overwrite", variable=self.overwrite_existing).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Checkbutton(row2, text="Save Empty Masks", variable=self.include_empty_masks).pack(
            side=tk.LEFT
        )
        self.dataset_label = ttk.Label(row2, text="Dataset: (not set)")
        self.dataset_label.pack(side=tk.LEFT, padx=(10, 0))

        cf = ttk.Frame(self.master)
        cf.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cf, bg="black")
        hbar = ttk.Scrollbar(cf, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = ttk.Scrollbar(cf, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)
        self.canvas.bind("<Button-5>", self._on_zoom)

        self.master.bind("<Key-n>", lambda _e: self.next_frame())
        self.master.bind("<Key-p>", lambda _e: self.prev_frame())
        self.master.bind("<Key-]>", lambda _e: self.next_plume())
        self.master.bind("<Key-[>", lambda _e: self.prev_plume())
        self.master.bind("<Key-s>", lambda _e: self.save_current_sample())
        self.master.bind("<Key-a>", lambda _e: self.apply_sam_current())
        self.master.bind("<Key-c>", lambda _e: self.clear_sam_prompts())

    # ---------------------------------------------------------------
    #                       Loading utilities
    # ---------------------------------------------------------------
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Cine", "*.cine"), ("All", "*.*")])
        if not path:
            return
        try:
            reader = CineReader()
            reader.load(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open video:\n{e}")
            return
        frames = []
        for i in range(reader.frame_count):
            frames.append(reader.read_frame(i).astype(np.float32))
        self.reader = reader
        self.video_path = path
        self.video = np.stack(frames, axis=0)
        self.current_frame = 0
        self.current_plume = 0
        self.plume_videos = []
        self.plume_masks = []
        self.current_raw = None
        self.current_img = None
        self.update_image()

    def load_config(self):
        if self.video is None:
            messagebox.showinfo("Info", "Load a video first")
            return
        path = filedialog.askopenfilename(filetypes=[("Config", "*.json")])
        if not path:
            return

        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        n_plumes = int(cfg["plumes"])
        cx = float(cfg["centre_x"])
        cy = float(cfg["centre_y"])
        bins, sig, _ = angle_signal_density(self.video, cx, cy, N_bins=360)
        summed = sig.sum(axis=0)
        fft_vals = np.fft.rfft(summed)
        phase = np.angle(fft_vals[n_plumes]) if n_plumes < len(fft_vals) else 0.0
        offset = (-phase / n_plumes) * 180.0 / np.pi
        offset %= 360.0
        if "offset" in cfg:
            offset = float(cfg["offset"]) % 360.0

        inner = int(cfg.get("inner_radius", cfg.get("calib_radius", 0)))
        outer = int(cfg.get("outer_radius", max(380, inner * 2)))
        if outer <= inner:
            outer = inner + 1

        crop = generate_CropRect(inner, outer, n_plumes, cx, cy)
        # Keep crop height from geometric crop, but use nozzle-aligned strip width like pipeline usage.
        frame_h, frame_w = int(self.video.shape[1]), int(self.video.shape[2])
        crop_h = int(min(max(1, crop[3]), frame_h))
        crop_w = int(min(max(1, outer), frame_w))
        out_shape = (crop_h, crop_w)
        calibration_point = (0.0, float(cy))

        plume_angle = None
        if n_plumes > 1:
            plume_angle_raw = 360.0 / float(n_plumes)
            # Avoid tan(90 deg) overflow in generate_plume_mask for 180-deg sectors.
            if plume_angle_raw < 179.0:
                plume_angle = plume_angle_raw
        x0 = int(max(0, min(inner, crop_w - 1)))
        plume_mask = generate_plume_mask(crop_w, crop_h, angle=plume_angle, x0=x0)
        if plume_mask.shape != (crop_h, crop_w):
            raise ValueError(
                f"Generated plume mask shape {plume_mask.shape} does not match strip shape {(crop_h, crop_w)}"
            )
        angles = np.linspace(0, 360, n_plumes, endpoint=False) + offset
        self.plume_videos = []
        use_cupy_rotate = ROTATE_BACKEND == "cupy" and cp is not None
        video_input = cp.asarray(self.video) if use_cupy_rotate else self.video
        for ang in angles:
            rotate_fn = (
                rotate_video_nozzle_at_0_half_cupy
                if use_cupy_rotate
                else rotate_video_nozzle_at_0_half_numpy
            )
            seg, _, _ = rotate_fn(
                video_input,
                (cx, cy),
                float(ang),
                interpolation="nearest",
                border_mode="constant",
                out_shape=out_shape,
                calibration_point=calibration_point,
                cval=0.0,
                stack=True,
            )
            if use_cupy_rotate and hasattr(seg, "__cuda_array_interface__"):
                seg = cp.asnumpy(seg)
            seg = np.asarray(seg).astype(np.float32, copy=False)
            self.plume_videos.append(seg)
        self._set_plume_videos(self.plume_videos)

        self.config_path = path
        self.config_values = cfg
        self.n_plumes = n_plumes
        self.centre_x = cx
        self.centre_y = cy
        self.inner_radius = inner
        self.outer_radius = outer
        self.rotation_offset_deg = float(offset)

        self.update_image()

    def load_no_config(self):
        if self.video is None:
            messagebox.showinfo("Info", "Load a video first")
            return

        raw_video = np.asarray(self.video, dtype=np.float32)
        self._set_plume_videos([raw_video])

        self.config_path = None
        self.config_values = {}
        self.n_plumes = 1
        self.centre_x = None
        self.centre_y = None
        self.inner_radius = 0
        self.outer_radius = 0
        self.rotation_offset_deg = 0.0

        self.update_image()

    def _set_plume_videos(self, plume_videos):
        self.plume_videos = plume_videos
        self.plume_masks = [
            [np.zeros(seg[0].shape, dtype=np.uint8) for _ in range(seg.shape[0])]
            for seg in self.plume_videos
        ]
        self.current_frame = 0
        self.current_plume = 0
        self.clear_sam_prompts(update=False)

    # ---------------------------------------------------------------
    #                       Navigation
    # ---------------------------------------------------------------
    def prev_frame(self):
        if not self.plume_videos:
            return
        self.current_frame = max(0, self.current_frame - 1)
        self.update_image()

    def next_frame(self):
        if not self.plume_videos:
            return
        old_frame = self.current_frame
        new_frame = min(self.plume_videos[0].shape[0] - 1, self.current_frame + 1)
        if new_frame == old_frame:
            return

        cur_mask = self.plume_masks[self.current_plume][old_frame]
        next_mask = self.plume_masks[self.current_plume][new_frame]
        if np.any(cur_mask) and not np.any(next_mask):
            next_mask[:] = cur_mask

        self.current_frame = new_frame
        self.update_image()

    def prev_plume(self):
        if not self.plume_videos:
            return
        self.current_plume = (self.current_plume - 1) % len(self.plume_videos)
        self.update_image()

    def next_plume(self):
        if not self.plume_videos:
            return
        self.current_plume = (self.current_plume + 1) % len(self.plume_videos)
        self.update_image()

    # ---------------------------------------------------------------
    #                       Image processing
    # ---------------------------------------------------------------
    def apply_gain_gamma(self, frame):
        g = self.gain.get()
        gm = self.gamma.get()
        img = frame / 4096.0 * g
        img = np.clip(img, 0, 1)
        if gm != 1:
            img = img ** gm
        return (img * 255).astype(np.uint8)

    def update_image(self):
        self.canvas.delete("all")
        self.live_rect_id = None
        if not self.plume_videos:
            return
        frame = self.plume_videos[self.current_plume][self.current_frame]
        img8 = self.apply_gain_gamma(frame)
        self.current_raw = frame
        self.current_img = img8
        disp = enlarge_image(Image.fromarray(img8), int(self.zoom))
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.config(scrollregion=(0, 0, disp.width, disp.height))
        mask = self.plume_masks[self.current_plume][self.current_frame]
        if np.any(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                pts = cnt.reshape(-1, 2) * self.zoom
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    self.canvas.create_line(x1, y1, x2, y2, fill="yellow", dash=(4, 2))

        for px, py in self.sam_points_pos:
            sx, sy = px * self.zoom, py * self.zoom
            r = max(3, int(self.zoom))
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, outline="lime", width=2)
        for px, py in self.sam_points_neg:
            sx, sy = px * self.zoom, py * self.zoom
            r = max(4, int(self.zoom) + 1)
            self.canvas.create_line(sx - r, sy - r, sx + r, sy + r, fill="red", width=2)
            self.canvas.create_line(sx - r, sy + r, sx + r, sy - r, fill="red", width=2)
        if self.sam_box is not None:
            x, y, w, h = self.sam_box
            self.canvas.create_rectangle(
                x * self.zoom,
                y * self.zoom,
                (x + w) * self.zoom,
                (y + h) * self.zoom,
                outline="cyan",
                dash=(4, 2),
                width=2,
            )

    # ---------------------------------------------------------------
    #                       Mask editing
    # ---------------------------------------------------------------
    def on_left_press(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if self.tool == "sam":
            self.start_pos = (x, y)
            self.sam_dragging = False
            return
        if self.tool == "brush":
            self.last_pos = (x, y)
            mask = self.plume_masks[self.current_plume][self.current_frame]
            cv2.circle(mask, (x, y), int(self.brush_size.get()), 1, -1)
            self.update_image()
        else:
            self.start_pos = (x, y)

    def on_left_drag(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if self.tool == "brush":
            if self.last_pos is None:
                return
            mask = self.plume_masks[self.current_plume][self.current_frame]
            width = max(1, int(self.brush_size.get()) * 2)
            cv2.line(mask, self.last_pos, (x, y), 1, width)
            self.last_pos = (x, y)
            self.update_image()
        elif self.tool == "sam" and self.start_pos is not None:
            self.sam_dragging = True
            self._draw_live_rect(self.start_pos, (x, y), outline="cyan")
        elif self.tool == "grabcut" and self.start_pos is not None:
            self._draw_live_rect(self.start_pos, (x, y), outline="lime")

    def on_left_release(self, event):
        if not self.plume_videos:
            return
        if self.tool == "sam" and self.start_pos is not None:
            x0, y0 = self.start_pos
            x1 = int(self.canvas.canvasx(event.x) / self.zoom)
            y1 = int(self.canvas.canvasy(event.y) / self.zoom)
            if self.sam_dragging and abs(x1 - x0) >= 2 and abs(y1 - y0) >= 2:
                self.sam_box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            else:
                self.sam_points_pos.append((x1, y1))
            self.start_pos = None
            self.sam_dragging = False
            self._clear_live_rect()
            self.update_image()
            return
        if not self.plume_videos or self.tool != "grabcut" or self.start_pos is None:
            return
        x0, y0 = self.start_pos
        x1 = int(self.canvas.canvasx(event.x) / self.zoom)
        y1 = int(self.canvas.canvasy(event.y) / self.zoom)
        rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        self.start_pos = None
        self._clear_live_rect()
        self.run_grabcut(rect, add=True)

    def on_right_press(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if self.tool == "sam":
            self.start_pos = (x, y)
            self.sam_dragging = False
            return
        if self.tool == "brush":
            self.last_pos = (x, y)
            mask = self.plume_masks[self.current_plume][self.current_frame]
            cv2.circle(mask, (x, y), int(self.brush_size.get()), 0, -1)
            self.update_image()
        else:
            self.start_pos = (x, y)

    def on_right_drag(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if self.tool == "brush":
            if self.last_pos is None:
                return
            mask = self.plume_masks[self.current_plume][self.current_frame]
            width = max(1, int(self.brush_size.get()) * 2)
            cv2.line(mask, self.last_pos, (x, y), 0, width)
            self.last_pos = (x, y)
            self.update_image()
        elif self.tool == "sam" and self.start_pos is not None:
            self.sam_dragging = True
        elif self.tool == "grabcut" and self.start_pos is not None:
            self._draw_live_rect(self.start_pos, (x, y), outline="red")

    def on_right_release(self, event):
        if not self.plume_videos:
            return
        if self.tool == "sam" and self.start_pos is not None:
            x1 = int(self.canvas.canvasx(event.x) / self.zoom)
            y1 = int(self.canvas.canvasy(event.y) / self.zoom)
            if not self.sam_dragging:
                self.sam_points_neg.append((x1, y1))
            self.start_pos = None
            self.sam_dragging = False
            self._clear_live_rect()
            self.update_image()
            return
        if not self.plume_videos or self.tool != "grabcut" or self.start_pos is None:
            return
        x0, y0 = self.start_pos
        x1 = int(self.canvas.canvasx(event.x) / self.zoom)
        y1 = int(self.canvas.canvasy(event.y) / self.zoom)
        mask = self.plume_masks[self.current_plume][self.current_frame]
        cv2.rectangle(mask, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), 0, -1)
        self.start_pos = None
        self._clear_live_rect()
        self.update_image()

    def _on_zoom(self, event):
        direction = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4 else -1
        self.zoom = max(1, self.zoom + direction)
        self.update_image()

    def set_tool(self, tool):
        self.tool = tool
        self._clear_live_rect()

    def clear_sam_prompts(self, update=True):
        self.sam_points_pos = []
        self.sam_points_neg = []
        self.sam_box = None
        self.sam_dragging = False
        if update:
            self.update_image()

    def load_sam_model(self):
        if not SAM2_AVAILABLE:
            messagebox.showerror(
                "SAM2 Not Available",
                "Missing SAM2 runtime. Install in this env:\n"
                "pip install sam2 torch torchvision",
            )
            return False

        try:
            paths = resolve_sam2_paths()
        except Exception as e:
            messagebox.showerror("SAM2 Paths", str(e))
            return False

        model_key = str(paths.get("model_key", "unknown"))
        config_path = str(paths["config"])
        checkpoint_path = str(paths["checkpoint"])
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        # Reuse existing predictor when already loaded with same model+device.
        if (
            self.sam_predictor is not None
            and self.sam_model_key == model_key
            and self.sam_device == device
        ):
            return True

        try:
            model = build_sam2(config_path, checkpoint_path, device=device)
            predictor = SAM2ImagePredictor(model)
        except Exception as e:
            messagebox.showerror("SAM2 Load Error", f"Failed to load SAM2:\n{e}")
            return False

        self.sam_model = model
        self.sam_predictor = predictor
        self.sam_model_key = model_key
        self.sam_device = device
        messagebox.showinfo("SAM2", f"SAM2 loaded ({model_key}) on {device}.")
        return True

    def apply_sam_current(self):
        if not self.plume_videos or self.current_img is None:
            return
        if not self.sam_points_pos and not self.sam_points_neg and self.sam_box is None:
            messagebox.showinfo(
                "SAM2 Prompts",
                "Add prompt(s) first: left click=positive, right click=negative, left drag=box.",
            )
            return
        if not self.load_sam_model():
            return

        frame = self.current_img
        if frame.ndim == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        kwargs = {"multimask_output": True}
        pts = []
        labels = []
        for p in self.sam_points_pos:
            pts.append([float(p[0]), float(p[1])])
            labels.append(1)
        for p in self.sam_points_neg:
            pts.append([float(p[0]), float(p[1])])
            labels.append(0)
        if pts:
            kwargs["point_coords"] = np.asarray(pts, dtype=np.float32)
            kwargs["point_labels"] = np.asarray(labels, dtype=np.int32)
        if self.sam_box is not None:
            x, y, w, h = self.sam_box
            kwargs["box"] = np.asarray([x, y, x + w, y + h], dtype=np.float32)

        try:
            self.sam_predictor.set_image(frame_rgb)
            masks, scores, _logits = self.sam_predictor.predict(**kwargs)
        except Exception as e:
            messagebox.showerror("SAM2 Predict Error", f"SAM2 failed on current frame:\n{e}")
            return

        if masks is None:
            return
        masks = np.asarray(masks)
        if masks.ndim == 2:
            best_mask = masks
        elif masks.ndim == 3:
            if scores is not None and len(scores) == masks.shape[0]:
                idx = int(np.argmax(scores))
            else:
                idx = 0
            best_mask = masks[idx]
        else:
            messagebox.showerror("SAM2 Predict Error", f"Unexpected mask shape: {masks.shape}")
            return

        out = (best_mask > 0).astype(np.uint8)
        # Keep only the dominant spray blob to suppress isolated SAM false positives.
        try:
            out = keep_largest_component_cuda(out, connectivity=2).astype(np.uint8, copy=False)
        except Exception:
            out = keep_largest_component(out, connectivity=2).astype(np.uint8, copy=False)
        if out.shape != self.plume_masks[self.current_plume][self.current_frame].shape:
            messagebox.showerror(
                "SAM2 Predict Error",
                f"Mask shape mismatch: SAM={out.shape}, target={self.plume_masks[self.current_plume][self.current_frame].shape}",
            )
            return
        target_mask = self.plume_masks[self.current_plume][self.current_frame]
        if self.sam_box is not None:
            # Keep mask unchanged outside the user-selected SAM patch.
            x, y, w, h = self.sam_box
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(target_mask.shape[1], x0 + int(w))
            y1 = min(target_mask.shape[0], y0 + int(h))
            if x1 > x0 and y1 > y0:
                target_mask[y0:y1, x0:x1] = out[y0:y1, x0:x1]
        else:
            # Without a box prompt, preserve existing mask and only add SAM positives.
            target_mask[:] = np.maximum(target_mask, out)
        self.update_image()

    def _draw_live_rect(self, p0, p1, outline="lime"):
        x0, y0 = p0
        x1, y1 = p1
        sx0, sy0 = x0 * self.zoom, y0 * self.zoom
        sx1, sy1 = x1 * self.zoom, y1 * self.zoom
        if self.live_rect_id is None:
            self.live_rect_id = self.canvas.create_rectangle(
                sx0, sy0, sx1, sy1, outline=outline, dash=(4, 2), width=2
            )
        else:
            self.canvas.coords(self.live_rect_id, sx0, sy0, sx1, sy1)
            self.canvas.itemconfig(self.live_rect_id, outline=outline)

    def _clear_live_rect(self):
        if self.live_rect_id is not None:
            self.canvas.delete(self.live_rect_id)
            self.live_rect_id = None

    def run_grabcut(self, rect, add=True):
        if self.current_img is None:
            return
        if rect[2] < 2 or rect[3] < 2:
            return

        # Use the display image (gain/gamma-adjusted uint8) for GrabCut to match what user sees.
        frame = self.current_img
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        mask = self.plume_masks[self.current_plume][self.current_frame]

        x, y, w, h = rect
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(frame.shape[1], x0 + int(w))
        y1 = min(frame.shape[0], y0 + int(h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return

        pad = 16
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(frame.shape[1], x1 + pad)
        ry1 = min(frame.shape[0], y1 + pad)
        roi_frame = frame[ry0:ry1, rx0:rx1]
        roi_mask = mask[ry0:ry1, rx0:rx1]
        gc_mask = np.where(roi_mask, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
        roi_rect = (x0 - rx0, y0 - ry0, x1 - x0, y1 - y0)

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                roi_frame,
                gc_mask,
                roi_rect,
                bgd,
                fgd,
                int(self.grabcut_iters.get()),
                cv2.GC_INIT_WITH_RECT,
            )
        except Exception:
            return
        new_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        if add:
            roi_mask[:] = np.maximum(roi_mask, new_mask)
        else:
            roi_mask[:] = roi_mask & (~new_mask)
        self.update_image()

    # ---------------------------------------------------------------
    #                       Dataset export utilities
    # ---------------------------------------------------------------
    def set_dataset_root(self):
        path = filedialog.askdirectory(title="Select dataset root folder")
        if not path:
            return
        self.dataset_root = path
        self._ensure_dataset_dirs()
        self.dataset_label.config(text=f"Dataset: {self.dataset_root}")

    def _ensure_dataset_dirs(self):
        if not self.dataset_root:
            return
        self.dataset_dirs = {
            "images_raw": os.path.join(self.dataset_root, "images_raw"),
            "images_disp": os.path.join(self.dataset_root, "images_disp"),
            "masks": os.path.join(self.dataset_root, "masks"),
            "meta": os.path.join(self.dataset_root, "meta"),
            "splits": os.path.join(self.dataset_root, "splits"),
        }
        for p in self.dataset_dirs.values():
            os.makedirs(p, exist_ok=True)

    def _require_dataset_root(self):
        if self.dataset_root:
            self._ensure_dataset_dirs()
            return True
        self.set_dataset_root()
        return bool(self.dataset_root)

    @staticmethod
    def _timestamp_utc():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _read_csv_rows(path):
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _write_csv_rows(path, fieldnames, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _upsert_row(self, path, fieldnames, row, key_field):
        rows = self._read_csv_rows(path)
        row = {k: row.get(k, "") for k in fieldnames}
        key = row[key_field]
        replaced = False
        for i, old in enumerate(rows):
            if old.get(key_field) == key:
                rows[i] = row
                replaced = True
                break
        if not replaced:
            rows.append(row)
        rows.sort(key=lambda r: r.get(key_field, ""))
        self._write_csv_rows(path, fieldnames, rows)

    def _sample_id(self, plume_idx, frame_idx):
        video_id = "video"
        if self.video_path:
            video_id = os.path.splitext(os.path.basename(self.video_path))[0]
        return f"{video_id}_p{plume_idx:02d}_f{frame_idx:04d}"

    def _sample_paths(self, sample_id):
        return {
            "raw": os.path.join(self.dataset_dirs["images_raw"], f"{sample_id}.png"),
            "disp": os.path.join(self.dataset_dirs["images_disp"], f"{sample_id}.png"),
            "mask": os.path.join(self.dataset_dirs["masks"], f"{sample_id}.png"),
            "meta": os.path.join(self.dataset_dirs["meta"], f"{sample_id}.json"),
        }

    @staticmethod
    def _to_uint16(frame):
        arr = np.nan_to_num(frame, nan=0.0, posinf=65535.0, neginf=0.0)
        arr = np.clip(arr, 0, 65535)
        return arr.astype(np.uint16)

    def _qa_flags(self, disp_u8, mask_bool):
        flags = []
        mask_area_px = int(mask_bool.sum())
        mask_area_ratio = float(mask_area_px / mask_bool.size) if mask_bool.size else 0.0

        if mask_area_px == 0:
            flags.append("empty_mask")
        elif mask_area_ratio < self.mask_area_warn_ratio:
            flags.append("small_mask")

        saturation_ratio = float(np.mean((disp_u8 <= 0) | (disp_u8 >= 255)))
        if saturation_ratio > self.saturation_warn_ratio:
            flags.append("high_saturation")

        touches_border = bool(
            np.any(mask_bool[0, :])
            or np.any(mask_bool[-1, :])
            or np.any(mask_bool[:, 0])
            or np.any(mask_bool[:, -1])
        )
        if touches_border:
            flags.append("touches_border")

        return flags, mask_area_px, mask_area_ratio

    def _save_sample(self, plume_idx, frame_idx, quiet=False):
        if not self.plume_videos or self.video_path is None:
            return "failed", "No plume video/config loaded."
        if not self._require_dataset_root():
            return "failed", "Dataset folder not set."

        raw_frame = self.plume_videos[plume_idx][frame_idx]
        disp_u8 = self.apply_gain_gamma(raw_frame)
        mask = self.plume_masks[plume_idx][frame_idx]
        mask_bool = mask.astype(bool)

        if not self.include_empty_masks.get() and not np.any(mask_bool):
            return "skipped", "Mask is empty and 'Save Empty Masks' is disabled."

        sample_id = self._sample_id(plume_idx, frame_idx)
        paths = self._sample_paths(sample_id)

        if not self.overwrite_existing.get():
            if all(os.path.exists(p) for p in paths.values()):
                return "skipped", "Sample exists (overwrite disabled)."

        raw_u16 = self._to_uint16(raw_frame)
        mask_u8 = (mask_bool.astype(np.uint8) * 255).astype(np.uint8)

        ok_raw = cv2.imwrite(paths["raw"], raw_u16)
        ok_disp = cv2.imwrite(paths["disp"], disp_u8)
        ok_mask = cv2.imwrite(paths["mask"], mask_u8)
        if not (ok_raw and ok_disp and ok_mask):
            return "failed", "Failed to write one or more PNG files."

        flags, mask_area_px, mask_area_ratio = self._qa_flags(disp_u8, mask_bool)
        ts = self._timestamp_utc()
        video_id = os.path.splitext(os.path.basename(self.video_path))[0]
        qa_text = ";".join(flags)

        metadata = {
            "sample_id": sample_id,
            "tool_version": self.tool_version,
            "timestamp_saved": ts,
            "video_id": video_id,
            "video_path": os.path.abspath(self.video_path),
            "config_path": os.path.abspath(self.config_path) if self.config_path else "",
            "frame_idx": int(frame_idx),
            "plume_idx": int(plume_idx),
            "height": int(raw_u16.shape[0]),
            "width": int(raw_u16.shape[1]),
            "raw_dtype": str(raw_u16.dtype),
            "disp_dtype": str(disp_u8.dtype),
            "mask_dtype": str(mask_u8.dtype),
            "raw_min": int(raw_u16.min()),
            "raw_max": int(raw_u16.max()),
            "gain": float(self.gain.get()),
            "gamma": float(self.gamma.get()),
            "n_plumes": int(self.n_plumes) if self.n_plumes is not None else len(self.plume_videos),
            "centre_x": float(self.centre_x) if self.centre_x is not None else None,
            "centre_y": float(self.centre_y) if self.centre_y is not None else None,
            "inner_radius": int(self.inner_radius),
            "outer_radius": int(self.outer_radius),
            "offset_deg": float(self.rotation_offset_deg),
            "mask_area_px": mask_area_px,
            "mask_area_ratio": mask_area_ratio,
            "qa_flags": flags,
            "image_raw": os.path.relpath(paths["raw"], self.dataset_root),
            "image_disp": os.path.relpath(paths["disp"], self.dataset_root),
            "mask_path": os.path.relpath(paths["mask"], self.dataset_root),
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        index_row = {
            "sample_id": sample_id,
            "video_id": video_id,
            "video_path": os.path.abspath(self.video_path),
            "frame_idx": str(frame_idx),
            "plume_idx": str(plume_idx),
            "mask_area_px": str(mask_area_px),
            "mask_area_ratio": f"{mask_area_ratio:.8f}",
            "qa_flags": qa_text,
            "timestamp_saved": ts,
            "image_raw": os.path.relpath(paths["raw"], self.dataset_root),
            "image_disp": os.path.relpath(paths["disp"], self.dataset_root),
            "mask_path": os.path.relpath(paths["mask"], self.dataset_root),
            "meta_path": os.path.relpath(paths["meta"], self.dataset_root),
        }
        index_path = os.path.join(self.dataset_root, "meta", "index.csv")
        self._upsert_row(index_path, self.INDEX_FIELDS, index_row, key_field="sample_id")

        if flags:
            review_row = {
                "sample_id": sample_id,
                "video_id": video_id,
                "frame_idx": str(frame_idx),
                "plume_idx": str(plume_idx),
                "qa_flags": qa_text,
                "timestamp_saved": ts,
            }
            review_path = os.path.join(self.dataset_root, "meta", "review_candidates.csv")
            self._upsert_row(
                review_path, self.REVIEW_FIELDS, review_row, key_field="sample_id"
            )

        if not quiet:
            detail = "Saved sample."
            if flags:
                detail += f"\nQA flags: {qa_text}"
            messagebox.showinfo("Saved", detail)

        return "saved", qa_text

    # ---------------------------------------------------------------
    #                       Saving actions
    # ---------------------------------------------------------------
    def save_current_sample(self):
        if not self.plume_videos:
            return
        status, msg = self._save_sample(self.current_plume, self.current_frame, quiet=False)
        if status == "failed":
            messagebox.showerror("Save Error", msg)
        elif status == "skipped":
            messagebox.showinfo("Skipped", msg)

    # Backward-compatible method name.
    def save_current(self):
        self.save_current_sample()

    def save_current_plume_labeled(self):
        if not self.plume_videos:
            return
        targets = [(self.current_plume, f) for f in range(self.plume_videos[self.current_plume].shape[0])]
        self._save_many(targets, "Save Current Plume")

    def save_all_labeled(self):
        if not self.plume_videos:
            return
        targets = []
        for p, seg in enumerate(self.plume_videos):
            for f in range(seg.shape[0]):
                targets.append((p, f))
        self._save_many(targets, "Save All Labeled")

    def _save_many(self, targets, title):
        if not self._require_dataset_root():
            return

        saved = 0
        skipped = 0
        failed = 0
        failed_messages = []

        for plume_idx, frame_idx in targets:
            status, msg = self._save_sample(plume_idx, frame_idx, quiet=True)
            if status == "saved":
                saved += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failed_messages.append(f"p{plume_idx:02d} f{frame_idx:04d}: {msg}")

        text = f"Saved: {saved}\nSkipped: {skipped}\nFailed: {failed}"
        if failed_messages:
            text += "\n\nFirst failures:\n" + "\n".join(failed_messages[:10])
        messagebox.showinfo(title, text)

    # ---------------------------------------------------------------
    #                       Split generation
    # ---------------------------------------------------------------
    def generate_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        if not self._require_dataset_root():
            return
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            messagebox.showerror("Split Error", "Split ratios must sum to 1.0.")
            return

        index_path = os.path.join(self.dataset_root, "meta", "index.csv")
        rows = self._read_csv_rows(index_path)
        if not rows:
            messagebox.showinfo("No Data", "No samples found in meta/index.csv.")
            return

        video_ids = sorted({r["video_id"] for r in rows if r.get("video_id")})
        if not video_ids:
            messagebox.showerror("Split Error", "index.csv has no valid video_id entries.")
            return

        rng = np.random.default_rng(seed)
        shuffled = list(video_ids)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))

        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                if n_train >= n_val:
                    n_train -= 1
                else:
                    n_val -= 1
        n_test = n - n_train - n_val

        train_videos = set(shuffled[:n_train])
        val_videos = set(shuffled[n_train : n_train + n_val])
        test_videos = set(shuffled[n_train + n_val :])

        train_ids = []
        val_ids = []
        test_ids = []
        for r in rows:
            vid = r.get("video_id", "")
            sid = r.get("sample_id", "")
            if not sid or not vid:
                continue
            if vid in train_videos:
                train_ids.append(sid)
            elif vid in val_videos:
                val_ids.append(sid)
            else:
                test_ids.append(sid)

        os.makedirs(self.dataset_dirs["splits"], exist_ok=True)
        train_path = os.path.join(self.dataset_dirs["splits"], "train.txt")
        val_path = os.path.join(self.dataset_dirs["splits"], "val.txt")
        test_path = os.path.join(self.dataset_dirs["splits"], "test.txt")
        with open(train_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(train_ids)))
        with open(val_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(val_ids)))
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(test_ids)))

        split_cfg = {
            "strategy": "by_video_id",
            "seed": int(seed),
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "num_videos": {
                "train": len(train_videos),
                "val": len(val_videos),
                "test": len(test_videos),
            },
            "num_samples": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "timestamp_generated": self._timestamp_utc(),
        }
        cfg_path = os.path.join(self.dataset_dirs["splits"], "split_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(split_cfg, f, indent=2)

        messagebox.showinfo(
            "Splits Generated",
            (
                "Generated splits by source video.\n"
                f"Train/Val/Test samples: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}"
            ),
        )

if __name__ == '__main__':
    root = tk.Tk()
    app = ManualSegmenter(root)
    root.mainloop()
