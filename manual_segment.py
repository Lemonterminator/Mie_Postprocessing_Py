"""Manual segmentation UI for plume masks.

Local changes:
- keep the original brush workflow for local touch-up
- add contour-fill and contour-erase tools for fast large-area editing with lower UI cost
- use SAM3 text/visual prompts for assisted foreground segmentation
- preserve the rest of the export/review pipeline so auto masks can be refined manually
"""

import csv
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from OSCC_postprocessing.cine.cine_utils import CineReader
from OSCC_postprocessing.analysis.cone_angle import angle_signal_density
from OSCC_postprocessing.rotation.segment_ops import generate_CropRect
from OSCC_postprocessing.rotation.rotate_with_alignment_cpu import (
    rotate_video_nozzle_at_0_half_numpy,
)
from OSCC_postprocessing.binary_ops.functions_bw import (
    keep_largest_component,
    keep_largest_component_cuda,
)
from OSCC_postprocessing.binary_ops.masking import generate_plume_mask
from OSCC_postprocessing.playback.video_playback import play_video_with_boundaries_cv2
from OSCC_postprocessing.utils.scaling import robust_scale
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
    from transformers import Sam3Model, Sam3Processor

    SAM3_AVAILABLE = True
except Exception:
    torch = None
    Sam3Model = None
    Sam3Processor = None
    SAM3_AVAILABLE = False

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
        self.current_raw = None  # Current rotated/cropped frame before display gain/gamma.
        self.current_frame = 0
        self.current_plume = 0
        self.zoom = 1
        self.tool = "sam"
        self.brush_size = tk.IntVar(value=5)
        self.start_pos = None
        self.last_pos = None
        self.live_rect_id = None
        self.contour_points = []

        self.gain = tk.DoubleVar(value=1.0)
        self.gamma = tk.DoubleVar(value=1.0)
        self.robust_qmin = tk.DoubleVar(value=5.0)
        self.robust_qmax = tk.DoubleVar(value=99.0)
        self.video_normalization = "raw"
        self.video_normalization_params = {}
        self.sam_model = None
        self.sam_processor = None
        self.sam_model_key = None
        self.sam_device = None
        self.sam_text_prompt = tk.StringVar(value="")
        self.sam_score_threshold = tk.DoubleVar(value=0.5)
        self.sam_mask_threshold = tk.DoubleVar(value=0.5)
        self.sam_point_box_radius = tk.IntVar(value=8)
        self.sam_merge_detections = tk.BooleanVar(value=False)
        self.sam_video_thread = None
        self.last_sam3_video_result = None
        self.sam_points_pos = []
        self.sam_points_neg = []
        self.sam_box = None
        self.sam_dragging = False
        self.static_block_masks = []  # Per-plume static 2D block masks (1=blocked)
        self.show_static_block = tk.BooleanVar(value=True)
        self.static_block_alpha = tk.IntVar(value=110)
        self.static_block_color = (255, 0, 0)

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
        row3 = ttk.Frame(top)
        row3.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        row4 = ttk.Frame(top)
        row4.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        row5 = ttk.Frame(top)
        row5.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        row1_left = ttk.Frame(row1)
        row1_left.pack(side=tk.LEFT)
        row1_right = ttk.Frame(row1)
        row1_right.pack(side=tk.RIGHT)

        row2_left = ttk.Frame(row2)
        row2_left.pack(side=tk.LEFT)
        row2_right = ttk.Frame(row2)
        row2_right.pack(side=tk.RIGHT)

        row3_left = ttk.Frame(row3)
        row3_left.pack(side=tk.LEFT)
        row3_right = ttk.Frame(row3)
        row3_right.pack(side=tk.RIGHT)

        row4_left = ttk.Frame(row4)
        row4_left.pack(side=tk.LEFT)
        row4_right = ttk.Frame(row4)
        row4_right.pack(side=tk.RIGHT)
        row5_left = ttk.Frame(row5)
        row5_left.pack(side=tk.LEFT)
        row5_right = ttk.Frame(row5)
        row5_right.pack(side=tk.RIGHT)

        ttk.Button(row1_left, text="Load Video", command=self.load_video).pack(side=tk.LEFT)
        ttk.Button(row1_left, text="Load Config", command=self.load_config).pack(side=tk.LEFT)
        ttk.Button(row1_left, text="No Config", command=self.load_no_config).pack(side=tk.LEFT)

        ttk.Label(row1_right, text="Brush").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(
            row1_right,
            from_=1,
            to=200,
            increment=1,
            width=5,
            textvariable=self.brush_size,
        ).pack(side=tk.LEFT)
        ttk.Label(row1_right, text="Gain").pack(side=tk.LEFT, padx=(10, 0))
        gain_box = ttk.Spinbox(
            row1_right,
            from_=0.1,
            to=10,
            increment=0.1,
            width=6,
            textvariable=self.gain,
            command=self.update_image,
        )
        gain_box.pack(side=tk.LEFT)
        ttk.Label(row1_right, text="Gamma").pack(side=tk.LEFT)
        gamma_box = ttk.Spinbox(
            row1_right,
            from_=0.1,
            to=3,
            increment=0.05,
            width=6,
            textvariable=self.gamma,
            command=self.update_image,
        )
        gamma_box.pack(side=tk.LEFT)

        ttk.Button(row2_left, text="Prev Frame", command=self.prev_frame).pack(side=tk.LEFT)
        ttk.Button(row2_left, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT)
        ttk.Button(row2_left, text="Prev Plume", command=self.prev_plume).pack(side=tk.LEFT)
        ttk.Button(row2_left, text="Next Plume", command=self.next_plume).pack(side=tk.LEFT)

        ttk.Button(row2_right, text="Brush Tool", command=lambda: self.set_tool("brush")).pack(
            side=tk.LEFT
        )
        ttk.Button(
            row2_right, text="Contour Fill", command=lambda: self.set_tool("contour_fill")
        ).pack(side=tk.LEFT)
        ttk.Button(
            row2_right, text="Contour Erase", command=lambda: self.set_tool("contour_erase")
        ).pack(side=tk.LEFT)
        ttk.Button(
            row2_right, text="Static Block Tool", command=lambda: self.set_tool("static_block")
        ).pack(side=tk.LEFT)
        gain_box.bind("<Return>", lambda _e: self.update_image())
        gain_box.bind("<FocusOut>", lambda _e: self.update_image())
        gamma_box.bind("<Return>", lambda _e: self.update_image())
        gamma_box.bind("<FocusOut>", lambda _e: self.update_image())

        ttk.Button(row3_left, text="Save Dataset Dir", command=self.set_dataset_root).pack(side=tk.LEFT)
        ttk.Button(row3_left, text="Save Current", command=self.save_current_sample).pack(side=tk.LEFT)
        ttk.Button(row3_left, text="Save Plume", command=self.save_current_plume_labeled).pack(
            side=tk.LEFT
        )
        ttk.Button(row3_left, text="Save All Labeled", command=self.save_all_labeled).pack(
            side=tk.LEFT
        )
        ttk.Button(row3_left, text="Generate Splits", command=self.generate_splits).pack(side=tk.LEFT)
        ttk.Button(
            row3_left, text="Apply Static Block", command=self.apply_static_block_current_plume
        ).pack(side=tk.LEFT)
        ttk.Button(row3_left, text="Clear Static Block", command=self.clear_static_block_current_plume).pack(
            side=tk.LEFT
        )

        ttk.Label(row3_right, text="SAM3 Text").pack(side=tk.LEFT, padx=(8, 0))
        self.sam_text_entry = ttk.Entry(row3_right, width=28, textvariable=self.sam_text_prompt)
        self.sam_text_entry.pack(side=tk.LEFT)
        self.sam_text_entry.bind("<Return>", lambda _e: self.apply_sam_current())
        ttk.Button(row3_right, text="Load SAM3", command=self.load_sam_model).pack(side=tk.LEFT)
        ttk.Button(row3_right, text="SAM3 Tool", command=lambda: self.set_tool("sam")).pack(side=tk.LEFT)
        ttk.Button(row3_right, text="Apply SAM3", command=self.apply_sam_current).pack(side=tk.LEFT)
        ttk.Button(row3_right, text="Apply SAM3 Video", command=self.apply_sam_video_current_plume).pack(
            side=tk.LEFT
        )
        ttk.Button(row3_right, text="Clear SAM3", command=self.clear_sam_prompts).pack(side=tk.LEFT)

        ttk.Checkbutton(row4_left, text="Overwrite", variable=self.overwrite_existing).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Checkbutton(row4_left, text="Save Empty Masks", variable=self.include_empty_masks).pack(
            side=tk.LEFT
        )
        ttk.Checkbutton(
            row4_left, text="Show Static Block", variable=self.show_static_block, command=self.update_image
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(row4_left, text="Block Alpha").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(
            row4_left,
            from_=0,
            to=255,
            increment=5,
            width=5,
            textvariable=self.static_block_alpha,
            command=self.update_image,
        ).pack(side=tk.LEFT)
        ttk.Button(row4_left, text="Block Color", command=self.choose_static_block_color).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        self.dataset_label = ttk.Label(row4_left, text="Dataset: (not set)")
        self.dataset_label.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(row4_right, text="SAM3 Score").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(
            row4_right,
            from_=0.0,
            to=1.0,
            increment=0.05,
            width=5,
            textvariable=self.sam_score_threshold,
        ).pack(side=tk.LEFT)
        ttk.Label(row4_right, text="Mask").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Spinbox(
            row4_right,
            from_=0.0,
            to=1.0,
            increment=0.05,
            width=5,
            textvariable=self.sam_mask_threshold,
        ).pack(side=tk.LEFT)
        ttk.Label(row4_right, text="Point Box").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Spinbox(
            row4_right,
            from_=1,
            to=80,
            increment=1,
            width=5,
            textvariable=self.sam_point_box_radius,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(row4_right, text="Merge Detections", variable=self.sam_merge_detections).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        ttk.Label(row5_left, text="Robust qmin").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(
            row5_left,
            from_=0.0,
            to=100.0,
            increment=0.5,
            width=6,
            textvariable=self.robust_qmin,
        ).pack(side=tk.LEFT)
        ttk.Label(row5_left, text="qmax").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Spinbox(
            row5_left,
            from_=0.0,
            to=100.0,
            increment=0.5,
            width=6,
            textvariable=self.robust_qmax,
        ).pack(side=tk.LEFT)
        ttk.Button(row5_left, text="Apply Robust Scale", command=self.apply_robust_scale_video).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(row5_right, text="Play SAM3 Probability", command=self.play_last_sam3_video_probability).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(row5_right, text="Play SAM3 Boundary", command=self.play_last_sam3_video_boundary).pack(
            side=tk.LEFT, padx=(6, 0)
        )

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
        self.master.bind("<Left>", lambda _e: self.prev_frame())
        self.master.bind("<Right>", lambda _e: self.next_frame())
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
        self.video_normalization = "raw"
        self.video_normalization_params = {}
        self.current_frame = 0
        self.current_plume = 0
        self.plume_videos = []
        self.plume_masks = []
        self.current_raw = None
        self.current_img = None
        self.last_sam3_video_result = None
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
        crop_h_raw = int(min(max(1, crop[3]), frame_h))
        crop_w_raw = int(min(max(1, outer), frame_w))
        crop_h, crop_w = self._even_spatial_shape(crop_h_raw, crop_w_raw)
        if (crop_h, crop_w) != (crop_h_raw, crop_w_raw):
            print(
                f"[manual-segment] adjusted rotate/crop shape from {(crop_h_raw, crop_w_raw)} "
                f"to even {(crop_h, crop_w)}",
                flush=True,
            )
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
            print(ang)
            rotate_fn = (
                rotate_video_nozzle_at_0_half_cupy
                if use_cupy_rotate
                else rotate_video_nozzle_at_0_half_numpy
            )
            seg, _, _ = rotate_fn(
                video_input,
                (cx, cy),
                float(ang)%360.0,
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

    @staticmethod
    def _even_spatial_shape(height, width):
        height = int(height)
        width = int(width)
        if height > 1 and height % 2:
            height -= 1
        if width > 1 and width % 2:
            width -= 1
        return max(1, height), max(1, width)

    @classmethod
    def _crop_video_even_spatial(cls, video):
        arr = np.asarray(video)
        if arr.ndim < 3:
            return arr
        height, width = int(arr.shape[-2]), int(arr.shape[-1])
        even_h, even_w = cls._even_spatial_shape(height, width)
        if (even_h, even_w) == (height, width):
            return arr
        slices = [slice(None)] * arr.ndim
        slices[-2] = slice(0, even_h)
        slices[-1] = slice(0, even_w)
        print(
            f"[manual-segment] cropped plume video shape from {(height, width)} to even {(even_h, even_w)}",
            flush=True,
        )
        return arr[tuple(slices)]

    def _set_plume_videos(self, plume_videos):
        self.plume_videos = [self._crop_video_even_spatial(seg) for seg in plume_videos]
        self.video_normalization = "raw"
        self.video_normalization_params = {}
        self.plume_masks = [
            [np.zeros(seg[0].shape, dtype=np.uint8) for _ in range(seg.shape[0])]
            for seg in self.plume_videos
        ]
        self.static_block_masks = [np.zeros(seg[0].shape, dtype=np.uint8) for seg in self.plume_videos]
        self.current_frame = 0
        self.current_plume = 0
        self.last_sam3_video_result = None
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
        if np.issubdtype(frame.dtype, np.integer) and frame.dtype.itemsize == 1:
            img = frame.astype(np.float32, copy=False) / 255.0
        else:
            frame_f = np.asarray(frame, dtype=np.float32)
            finite = frame_f[np.isfinite(frame_f)]
            if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.5:
                img = frame_f
            else:
                img = frame_f / 4096.0
        img = img * g
        img = np.clip(img, 0, 1)
        if gm != 1:
            img = img ** gm
        return (img * 255).astype(np.uint8)

    def apply_robust_scale_video(self):
        if not self.plume_videos:
            return
        try:
            qmin = float(self.robust_qmin.get())
            qmax = float(self.robust_qmax.get())
        except Exception:
            messagebox.showerror("Robust Scale", "qmin and qmax must be numeric.")
            return
        if not (0.0 <= qmin < qmax <= 100.0):
            messagebox.showerror("Robust Scale", "Expected 0 <= qmin < qmax <= 100.")
            return

        scaled_videos = []
        stats = []
        try:
            for idx, seg in enumerate(self.plume_videos):
                seg_f = np.asarray(seg, dtype=np.float32)
                low, high = np.percentile(seg_f, [qmin, qmax])
                scaled = robust_scale(seg_f, q_min=qmin, q_max=qmax)
                seg_u8 = np.clip(np.rint(np.asarray(scaled) * 255.0), 0, 255).astype(np.uint8)
                scaled_videos.append(seg_u8)
                stats.append(
                    {
                        "plume_idx": int(idx),
                        "qmin_value": float(low),
                        "qmax_value": float(high),
                    }
                )
        except Exception as e:
            messagebox.showerror("Robust Scale", f"Failed to robust-scale video:\n{e}")
            return

        self.plume_videos = scaled_videos
        self.video_normalization = "robust_u8"
        self.video_normalization_params = {
            "qmin": qmin,
            "qmax": qmax,
            "scope": "per_plume_full_video",
            "stats": stats,
        }
        self.gain.set(1.0)
        self.gamma.set(1.0)
        self.current_raw = None
        self.current_img = None
        self.last_sam3_video_result = None
        self.update_image()
        self.master.title(f"Manual Segmenter - robust scaled to uint8 q={qmin:g}-{qmax:g}")

    def update_image(self):
        self.canvas.delete("all")
        self.live_rect_id = None
        if not self.plume_videos:
            return
        frame = self.plume_videos[self.current_plume][self.current_frame]
        img8 = self.apply_gain_gamma(frame)
        self.current_raw = frame
        self.current_img = img8

        disp_rgb = np.stack([img8, img8, img8], axis=-1)
        if self.show_static_block.get() and self.static_block_masks:
            block = self.static_block_masks[self.current_plume].astype(bool)
            if np.any(block):
                a = float(self.static_block_alpha.get()) / 255.0
                color = np.array(self.static_block_color, dtype=np.float32)
                disp_rgb_f = disp_rgb.astype(np.float32)
                disp_rgb_f[block] = (1.0 - a) * disp_rgb_f[block] + a * color
                disp_rgb = np.clip(disp_rgb_f, 0, 255).astype(np.uint8)

        disp = enlarge_image(Image.fromarray(disp_rgb), int(self.zoom))
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
        if len(self.contour_points) >= 2 and self.tool in {"contour_fill", "contour_erase"}:
            pts = np.asarray(self.contour_points, dtype=np.float32) * float(self.zoom)
            line_color = "lime" if self.tool == "contour_fill" else "red"
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                self.canvas.create_line(x1, y1, x2, y2, fill=line_color, width=2)

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
        if self.tool == "static_block":
            self.last_pos = (x, y)
            static_mask = self.static_block_masks[self.current_plume]
            cv2.circle(static_mask, (x, y), int(self.brush_size.get()), 1, -1)
            self.update_image()
            return
        if self.tool == "brush":
            self.last_pos = (x, y)
            mask = self.plume_masks[self.current_plume][self.current_frame]
            cv2.circle(mask, (x, y), int(self.brush_size.get()), 1, -1)
            self.update_image()
        elif self.tool in {"contour_fill", "contour_erase"}:
            self.contour_points = [(x, y)]
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
        elif self.tool == "static_block":
            if self.last_pos is None:
                return
            static_mask = self.static_block_masks[self.current_plume]
            width = max(1, int(self.brush_size.get()) * 2)
            cv2.line(static_mask, self.last_pos, (x, y), 1, width)
            self.last_pos = (x, y)
            self.update_image()
        elif self.tool in {"contour_fill", "contour_erase"}:
            if not self.contour_points or self.contour_points[-1] != (x, y):
                self.contour_points.append((x, y))
                self.update_image()
        elif self.tool == "sam" and self.start_pos is not None:
            self.sam_dragging = True
            self._draw_live_rect(self.start_pos, (x, y), outline="cyan")

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
        if self.tool in {"contour_fill", "contour_erase"}:
            self._finish_contour(event)
            return
        self.start_pos = None
        self._clear_live_rect()

    def on_right_press(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if self.tool == "sam":
            self.start_pos = (x, y)
            self.sam_dragging = False
            return
        if self.tool == "static_block":
            self.last_pos = (x, y)
            static_mask = self.static_block_masks[self.current_plume]
            cv2.circle(static_mask, (x, y), int(self.brush_size.get()), 0, -1)
            self.update_image()
            return
        if self.tool == "brush":
            self.last_pos = (x, y)
            mask = self.plume_masks[self.current_plume][self.current_frame]
            cv2.circle(mask, (x, y), int(self.brush_size.get()), 0, -1)
            self.update_image()
        elif self.tool in {"contour_fill", "contour_erase"}:
            self.contour_points = [(x, y)]
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
        elif self.tool == "static_block":
            if self.last_pos is None:
                return
            static_mask = self.static_block_masks[self.current_plume]
            width = max(1, int(self.brush_size.get()) * 2)
            cv2.line(static_mask, self.last_pos, (x, y), 0, width)
            self.last_pos = (x, y)
            self.update_image()
        elif self.tool in {"contour_fill", "contour_erase"}:
            if not self.contour_points or self.contour_points[-1] != (x, y):
                self.contour_points.append((x, y))
                self.update_image()
        elif self.tool == "sam" and self.start_pos is not None:
            self.sam_dragging = True
            self._draw_live_rect(self.start_pos, (x, y), outline="red")

    def on_right_release(self, event):
        if not self.plume_videos:
            return
        if self.tool == "sam" and self.start_pos is not None:
            x0, y0 = self.start_pos
            x1 = int(self.canvas.canvasx(event.x) / self.zoom)
            y1 = int(self.canvas.canvasy(event.y) / self.zoom)
            if not self.sam_dragging:
                self.sam_points_neg.append((x1, y1))
            else:
                mask = self.plume_masks[self.current_plume][self.current_frame]
                rx0 = min(x0, x1)
                ry0 = min(y0, y1)
                rx1 = max(x0, x1)
                ry1 = max(y0, y1)
                cv2.rectangle(mask, (rx0, ry0), (rx1, ry1), 0, -1)
            self.start_pos = None
            self.sam_dragging = False
            self._clear_live_rect()
            self.update_image()
            return
        if self.tool in {"contour_fill", "contour_erase"}:
            self._finish_contour(event)
            return
        self.start_pos = None
        self._clear_live_rect()

    def _on_zoom(self, event):
        direction = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4 else -1
        self.zoom = max(1, self.zoom + direction)
        self.update_image()

    def set_tool(self, tool):
        self.tool = tool
        self._clear_live_rect()
        self.contour_points = []
        self.last_pos = None
        self.start_pos = None
        self.update_image()

    def _finish_contour(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if not self.contour_points:
            return
        if self.contour_points[-1] != (x, y):
            self.contour_points.append((x, y))
        if len(self.contour_points) >= 3:
            contour = np.asarray(self.contour_points, dtype=np.int32)
            mask = self.plume_masks[self.current_plume][self.current_frame]
            fill_value = 1 if self.tool == "contour_fill" else 0
            cv2.fillPoly(mask, [contour], fill_value)
        self.contour_points = []
        self.update_image()

    def clear_sam_prompts(self, update=True):
        self.sam_points_pos = []
        self.sam_points_neg = []
        self.sam_box = None
        self.sam_dragging = False
        if update:
            self.update_image()

    def choose_static_block_color(self):
        rgb, _hex = colorchooser.askcolor(color=self.static_block_color, title="Select static block color")
        if rgb is None:
            return
        self.static_block_color = tuple(int(v) for v in rgb)
        self.update_image()

    def clear_static_block_current_plume(self):
        if not self.plume_videos:
            return
        self.static_block_masks[self.current_plume][:] = 0
        self.update_image()

    def apply_static_block_current_plume(self):
        if not self.plume_videos:
            return
        block = self.static_block_masks[self.current_plume].astype(bool)
        if not np.any(block):
            messagebox.showinfo("Static Block", "Static block mask is empty for current plume.")
            return
        keep = (~block).astype(np.uint8)
        for f in range(self.plume_videos[self.current_plume].shape[0]):
            self.plume_masks[self.current_plume][f] &= keep
        self.update_image()

    @staticmethod
    def _default_sam3_model_dir():
        return Path(
            os.environ.get(
                "SAM3_MODEL_DIR",
                Path(__file__).resolve().parent / ".models" / "sam3_hf",
            )
        ).expanduser()

    def load_sam_model(self):
        if not SAM3_AVAILABLE:
            messagebox.showerror(
                "SAM3 Not Available",
                "Missing SAM3 runtime. Use the project venv or install:\n"
                "pip install torch torchvision transformers pillow",
            )
            return False

        model_dir = self._default_sam3_model_dir()
        config_path = model_dir / "config.json"
        has_weights = any(
            candidate.exists()
            for candidate in (
                model_dir / "model.safetensors",
                model_dir / "pytorch_model.bin",
                model_dir / "model.safetensors.index.json",
                model_dir / "pytorch_model.bin.index.json",
            )
        )
        if not model_dir.exists() or not config_path.exists() or not has_weights:
            messagebox.showerror(
                "SAM3 Model",
                f"Local SAM3 model is incomplete:\n{model_dir}\n\n"
                "Run scripts/download_sam3.py first, or set SAM3_MODEL_DIR.",
            )
            return False

        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        model_key = f"{model_dir.resolve()}|{device}"

        if (
            self.sam_model is not None
            and self.sam_processor is not None
            and self.sam_model_key == model_key
            and self.sam_device == device
        ):
            return True

        try:
            model = Sam3Model.from_pretrained(str(model_dir)).to(device)
            model.eval()
            processor = Sam3Processor.from_pretrained(str(model_dir))
        except Exception as e:
            messagebox.showerror("SAM3 Load Error", f"Failed to load SAM3:\n{e}")
            return False

        self.sam_model = model
        self.sam_processor = processor
        self.sam_model_key = model_key
        self.sam_device = device
        messagebox.showinfo("SAM3", f"SAM3 loaded on {device}:\n{model_dir}")
        return True

    def _sam3_boxes_and_labels(self, image_shape):
        h, w = int(image_shape[0]), int(image_shape[1])
        boxes = []
        labels = []

        def add_box(x0, y0, x1, y1, label):
            x0 = max(0, min(w - 1, int(round(x0))))
            y0 = max(0, min(h - 1, int(round(y0))))
            x1 = max(0, min(w, int(round(x1))))
            y1 = max(0, min(h, int(round(y1))))
            if x1 - x0 >= 1 and y1 - y0 >= 1:
                boxes.append([float(x0), float(y0), float(x1), float(y1)])
                labels.append(int(label))

        if self.sam_box is not None:
            x, y, bw, bh = self.sam_box
            add_box(x, y, x + bw, y + bh, 1)

        radius = max(1, int(self.sam_point_box_radius.get()))
        for px, py in self.sam_points_pos:
            add_box(px - radius, py - radius, px + radius + 1, py + radius + 1, 1)
        for px, py in self.sam_points_neg:
            add_box(px - radius, py - radius, px + radius + 1, py + radius + 1, 0)

        return boxes, labels

    def apply_sam_current(self):
        if not self.plume_videos or self.current_img is None:
            return
        prompt_text = self.sam_text_prompt.get().strip()
        boxes, box_labels = self._sam3_boxes_and_labels(self.current_img.shape)
        if not prompt_text and not boxes:
            messagebox.showinfo(
                "SAM3 Prompts",
                "Add a text prompt or visual prompt first:\n"
                "left click=positive point, right click=negative point, left drag=positive box, right drag=erase mask.",
            )
            return
        if not self.load_sam_model():
            return

        frame = self.current_img
        if frame.ndim == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            image = Image.fromarray(frame_rgb.astype(np.uint8, copy=False))
            processor_kwargs = {
                "images": image,
                "return_tensors": "pt",
            }
            if prompt_text:
                processor_kwargs["text"] = prompt_text
            if boxes:
                processor_kwargs["input_boxes"] = [boxes]
                processor_kwargs["input_boxes_labels"] = [box_labels]

            inputs = self.sam_processor(**processor_kwargs).to(self.sam_device)
            with torch.inference_mode():
                outputs = self.sam_model(**inputs)

            target_sizes = inputs["original_sizes"].detach().cpu().tolist()
            result = self.sam_processor.post_process_instance_segmentation(
                outputs,
                threshold=float(self.sam_score_threshold.get()),
                mask_threshold=float(self.sam_mask_threshold.get()),
                target_sizes=target_sizes,
            )[0]
        except Exception as e:
            messagebox.showerror("SAM3 Predict Error", f"SAM3 failed on current frame:\n{e}")
            return

        masks = result.get("masks")
        scores = result.get("scores")
        if masks is None:
            return
        if len(masks) == 0:
            messagebox.showinfo("SAM3", "No detections for the current prompt/threshold.")
            return

        if bool(self.sam_merge_detections.get()):
            best_score = float(scores.max().detach().cpu().item()) if scores is not None and len(scores) else None
            best_mask = torch.any(masks > 0, dim=0)
        else:
            if scores is not None and len(scores) == len(masks):
                idx = int(torch.argmax(scores).detach().cpu().item())
                best_score = float(scores[idx].detach().cpu().item())
            else:
                idx = 0
                best_score = None
            best_mask = masks[idx]

        out = best_mask.detach().cpu().numpy().astype(np.uint8)
        # Keep only the dominant spray blob to suppress isolated SAM false positives.
        if not bool(self.sam_merge_detections.get()):
            try:
                out = keep_largest_component_cuda(out, connectivity=2).astype(np.uint8, copy=False)
            except Exception:
                out = keep_largest_component(out, connectivity=2).astype(np.uint8, copy=False)
        if out.shape != self.plume_masks[self.current_plume][self.current_frame].shape:
            messagebox.showerror(
                "SAM3 Predict Error",
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
        score_text = "none" if best_score is None else f"{best_score:.3f}"
        self.master.title(f"Manual Segmenter - SAM3 detections={len(masks)} best_score={score_text}")
        self.update_image()

    def apply_sam_video_current_plume(self):
        if not self.plume_videos:
            return
        if self.sam_video_thread is not None and self.sam_video_thread.is_alive():
            messagebox.showinfo("SAM3 Video", "SAM3 video segmentation is already running.")
            return

        prompt_text = self.sam_text_prompt.get().strip()
        if not prompt_text:
            messagebox.showinfo(
                "SAM3 Video",
                "Enter a SAM3 Text prompt first. The full-video path currently uses the official text-prompt video predictor.",
            )
            return

        checkpoint_path = self._default_sam3_model_dir() / "sam3.pt"
        if not checkpoint_path.exists():
            messagebox.showerror("SAM3 Video", f"Missing official SAM3 checkpoint:\n{checkpoint_path}")
            return

        plume_idx = int(self.current_plume)
        frame_idx = int(self.current_frame)
        plume_video = np.asarray(self.plume_videos[plume_idx])
        export_root = self._sam3_video_export_root(plume_idx, frame_idx)
        input_avi = export_root / "input.avi"
        output_root = export_root / "sam3_video_export"

        try:
            self._write_sam3_input_video(plume_video, input_avi, fps=30)
        except Exception as e:
            messagebox.showerror("SAM3 Video", f"Failed to write temporary AVI:\n{e}")
            return

        cmd = self._build_sam3_video_command(
            video_path=input_avi,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            prompt=prompt_text,
            frame_index=frame_idx,
            fps=30,
        )
        self.master.title(f"Manual Segmenter - SAM3 video running plume={plume_idx} frame={frame_idx}")
        print(
            f"[sam3-video] start plume={plume_idx} prompt_frame={frame_idx} frames={plume_video.shape[0]} "
            f"input={input_avi}",
            flush=True,
        )
        print(f"[sam3-video] command: {subprocess.list2cmdline([str(part) for part in cmd])}", flush=True)

        def worker():
            try:
                env = os.environ.copy()
                env.setdefault("USE_PERFLIB", "0")
                process = subprocess.Popen(
                    cmd,
                    cwd=str(Path(__file__).resolve().parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    bufsize=1,
                )
                output_tail = []
                if process.stdout is not None:
                    for line in process.stdout:
                        line = line.rstrip()
                        if line:
                            print(f"[sam3-video] {line}", flush=True)
                            output_tail.append(line)
                            output_tail = output_tail[-60:]
                return_code = process.wait()
                if return_code != 0:
                    detail = "\n".join(output_tail)
                    raise RuntimeError(detail or f"command returned {return_code}")
                masks_loaded = self._load_sam3_video_masks(output_root, plume_video.shape[0], plume_idx)
            except Exception as exc:
                self.master.after(0, lambda exc=exc: self._finish_sam3_video_error(exc, export_root))
                return
            self.master.after(
                0,
                lambda masks_loaded=masks_loaded: self._finish_sam3_video_success(
                    plume_idx, masks_loaded, output_root, input_avi
                ),
            )

        self.sam_video_thread = threading.Thread(target=worker, daemon=True)
        self.sam_video_thread.start()

    def _sam3_video_export_root(self, plume_idx, frame_idx):
        repo_root = Path(__file__).resolve().parent
        video_stem = Path(self.video_path).stem if self.video_path else "video"
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in video_stem)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return repo_root / "Results" / "manual_segment_sam3_video" / (
            f"{safe_stem}_plume{plume_idx:02d}_frame{frame_idx:05d}_{stamp}"
        )

    def _write_sam3_input_video(self, plume_video, output_path, fps=30):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if plume_video.ndim != 3:
            raise ValueError(f"Expected grayscale plume video with shape (frames, h, w), got {plume_video.shape}")

        height, width = plume_video.shape[1:]
        encoded_height = int(height + (height % 2))
        encoded_width = int(width + (width % 2))
        pad_bottom = encoded_height - int(height)
        pad_right = encoded_width - int(width)
        if pad_bottom or pad_right:
            print(
                f"[sam3-video] padding input AVI from {(int(height), int(width))} "
                f"to {(encoded_height, encoded_width)} for codec compatibility",
                flush=True,
            )
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (encoded_width, encoded_height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {output_path}")
        try:
            for frame in plume_video:
                img8 = self.apply_gain_gamma(frame)
                bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
                if pad_bottom or pad_right:
                    bgr = cv2.copyMakeBorder(
                        bgr,
                        0,
                        pad_bottom,
                        0,
                        pad_right,
                        borderType=cv2.BORDER_REPLICATE,
                    )
                writer.write(bgr)
        finally:
            writer.release()

    @staticmethod
    def _to_wsl_path(path):
        path_str = str(Path(path).resolve())
        if len(path_str) >= 2 and path_str[1] == ":":
            drive = path_str[0].lower()
            rest = path_str[2:].replace("\\", "/").lstrip("/")
            return f"/mnt/{drive}/{rest}"
        return path_str.replace("\\", "/")

    def _build_sam3_video_command(self, video_path, checkpoint_path, output_root, prompt, frame_index, fps=30):
        repo_root = Path(__file__).resolve().parent
        script_path = repo_root / "scripts" / "export_sam3_video_results.py"
        prompt_args = self._sam3_video_prompt_args()
        args = [
            "--video-path",
            str(video_path),
            "--checkpoint",
            str(checkpoint_path),
            "--version",
            "sam3",
            "--prompt",
            str(prompt),
            "--frame-index",
            str(int(frame_index)),
            "--output-root",
            str(output_root),
            "--save-format",
            "npz",
            "--fps",
            str(int(fps)),
            "--mask-threshold",
            str(float(self.sam_mask_threshold.get())),
            "--score-threshold",
            str(float(self.sam_score_threshold.get())),
            "--progress-every",
            "1",
        ]
        args.extend(prompt_args)

        if os.name == "nt" and shutil.which("wsl.exe"):
            wsl_args = args.copy()
            for flag in ("--video-path", "--checkpoint", "--output-root"):
                value_index = wsl_args.index(flag) + 1
                wsl_args[value_index] = self._to_wsl_path(wsl_args[value_index])
            inner = [
                "./scripts/run_in_sam3_wsl_env.sh",
                "python",
                "-u",
                "scripts/export_sam3_video_results.py",
                *wsl_args,
            ]
            command = "cd " + shlex.quote(self._to_wsl_path(repo_root)) + " && export USE_PERFLIB=0 && " + " ".join(
                shlex.quote(part) for part in inner
            )
            return ["wsl.exe", "bash", "-lc", command]

        return [sys.executable, "-u", str(script_path), *args]

    def _sam3_video_prompt_args(self):
        if not self.plume_videos:
            return []

        frame_shape = self.plume_videos[self.current_plume][0].shape
        height, width = int(frame_shape[-2]), int(frame_shape[-1])
        args = []
        boxes = []

        if self.sam_box is not None:
            x, y, bw, bh = self.sam_box
            x0 = max(0.0, min(float(width), float(x)))
            y0 = max(0.0, min(float(height), float(y)))
            x1 = max(0.0, min(float(width), float(x) + float(bw)))
            y1 = max(0.0, min(float(height), float(y) + float(bh)))
            if x1 > x0 and y1 > y0:
                boxes = [[x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]]
                print(
                    f"[sam3-video] using drawn box prompt xywh=({int(x0)}, {int(y0)}, {int(x1 - x0)}, {int(y1 - y0)})",
                    flush=True,
                )
        elif self.plume_masks:
            mask = np.asarray(self.plume_masks[self.current_plume][self.current_frame]) > 0
            if mask.shape == (height, width) and np.any(mask):
                ys, xs = np.where(mask)
                pad = max(2, int(round(0.02 * max(width, height))))
                x0 = max(0, int(xs.min()) - pad)
                y0 = max(0, int(ys.min()) - pad)
                x1 = min(width, int(xs.max()) + 1 + pad)
                y1 = min(height, int(ys.max()) + 1 + pad)
                if x1 > x0 and y1 > y0:
                    boxes = [[x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]]
                    print(
                        f"[sam3-video] using current mask bbox prompt xywh=({x0}, {y0}, {x1 - x0}, {y1 - y0})",
                        flush=True,
                    )

        if boxes:
            args.extend(
                [
                    "--boxes-json",
                    json.dumps(boxes, separators=(",", ":")),
                    "--box-labels-json",
                    "[1]",
                ]
            )

        points = []
        labels = []
        for px, py in self.sam_points_pos:
            points.append([float(px) / width, float(py) / height])
            labels.append(1)
        for px, py in self.sam_points_neg:
            points.append([float(px) / width, float(py) / height])
            labels.append(0)

        if not points and self.plume_masks:
            mask = np.asarray(self.plume_masks[self.current_plume][self.current_frame]) > 0
            if mask.shape == (height, width) and np.any(mask):
                auto_points, auto_labels = self._sam3_video_points_from_mask(mask)
                for px, py in auto_points:
                    points.append([float(px) / width, float(py) / height])
                labels.extend(auto_labels)
                pos_count = int(sum(1 for label in auto_labels if label == 1))
                neg_count = int(sum(1 for label in auto_labels if label == 0))
                print(
                    f"[sam3-video] using current mask point prompts pos={pos_count} neg={neg_count}",
                    flush=True,
                )

        if points:
            clipped_points = [
                [max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))]
                for x, y in points
            ]
            args.extend(
                [
                    "--points-json",
                    json.dumps(clipped_points, separators=(",", ":")),
                    "--point-labels-json",
                    json.dumps(labels, separators=(",", ":")),
                ]
            )

        return args

    @staticmethod
    def _sam3_video_points_from_mask(mask):
        mask = np.asarray(mask) > 0
        height, width = mask.shape
        ys, xs = np.where(mask)
        if xs.size == 0:
            return [], []

        points = []
        labels = []

        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        points.append((float(max_loc[0]), float(max_loc[1])))
        labels.append(1)

        x0, x1 = int(xs.min()), int(xs.max())
        mask_width = max(1, x1 - x0 + 1)
        sample_xs = [x0 + 0.25 * mask_width, x0 + 0.50 * mask_width, x0 + 0.75 * mask_width]
        for sample_x in sample_xs:
            band = np.abs(xs.astype(np.float32) - float(sample_x)) <= max(3.0, 0.03 * mask_width)
            if not np.any(band):
                continue
            band_ys = ys[band]
            band_xs = xs[band]
            median_idx = int(np.argsort(band_ys)[len(band_ys) // 2])
            candidate = (float(band_xs[median_idx]), float(band_ys[median_idx]))
            if all((candidate[0] - px) ** 2 + (candidate[1] - py) ** 2 > 100 for px, py in points):
                points.append(candidate)
                labels.append(1)

        y0, y1 = int(ys.min()), int(ys.max())
        pad = max(8, int(round(0.05 * max(height, width))))
        neg_candidates = [
            (x0 - pad, y0 - pad),
            (x0 - pad, y1 + pad),
            (x1 + pad, y0 - pad),
            (x1 + pad, y1 + pad),
            ((x0 + x1) / 2.0, y0 - pad),
            ((x0 + x1) / 2.0, y1 + pad),
        ]
        for px, py in neg_candidates:
            px = float(np.clip(px, 0, width - 1))
            py = float(np.clip(py, 0, height - 1))
            if not mask[int(round(py)), int(round(px))]:
                points.append((px, py))
                labels.append(0)
            if sum(1 for label in labels if label == 0) >= 4:
                break

        return points[:8], labels[:8]

    @staticmethod
    def _sam3_video_mask_stats(mask_frames):
        areas = [int(np.count_nonzero(mask)) for mask in mask_frames]
        total_area = int(sum(areas))
        frames_with_mask = int(sum(area > 0 for area in areas))
        return total_area, frames_with_mask

    @staticmethod
    def _windows_path_from_wsl(path):
        path_str = str(path).replace("\\", "/")
        if os.name == "nt" and path_str.startswith("/mnt/"):
            parts = path_str.split("/")
            if len(parts) > 3 and parts[1] == "mnt":
                return Path(parts[2].upper() + ":\\" + "\\".join(parts[3:]))
        return Path(path)

    @staticmethod
    def _fit_sam3_mask_array_to_shape(mask_array, target_shape):
        arr = np.asarray(mask_array)
        target_h, target_w = int(target_shape[0]), int(target_shape[1])

        if arr.ndim == 2:
            out = np.zeros((target_h, target_w), dtype=arr.dtype)
            copy_h = min(target_h, arr.shape[0])
            copy_w = min(target_w, arr.shape[1])
            if copy_h > 0 and copy_w > 0:
                out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
            return out

        if arr.ndim == 3:
            out = np.zeros((arr.shape[0], target_h, target_w), dtype=arr.dtype)
            copy_h = min(target_h, arr.shape[1])
            copy_w = min(target_w, arr.shape[2])
            if copy_h > 0 and copy_w > 0:
                out[:, :copy_h, :copy_w] = arr[:, :copy_h, :copy_w]
            return out

        return arr

    def _sam3_video_summary_and_npz_dir(self, output_root):
        summary_path = Path(output_root) / "summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"Missing SAM3 video summary: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        npz_dir = self._windows_path_from_wsl(summary["artifacts"]["mask_npz_dir"])
        return summary, npz_dir

    def _load_sam3_video_masks(self, output_root, expected_frames, plume_idx, score_threshold=None):
        summary, npz_dir = self._sam3_video_summary_and_npz_dir(output_root)
        if score_threshold is None:
            score_threshold = float(self.sam_score_threshold.get())
        else:
            score_threshold = float(score_threshold)
        new_masks = [
            np.zeros_like(self.plume_masks[plume_idx][0], dtype=np.uint8)
            for _ in range(int(expected_frames))
        ]
        shape_warning_done = False
        for record in summary.get("records", []):
            frame_index = int(record["frame_index"])
            if frame_index < 0 or frame_index >= len(new_masks):
                continue
            target_shape = new_masks[frame_index].shape
            npz_path = npz_dir / f"{record['frame_name']}.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                if "raw_masks" in data.files:
                    masks = np.asarray(data["raw_masks"], dtype=np.uint8)
                    probs = np.asarray(
                        data["raw_probabilities"] if "raw_probabilities" in data.files else [],
                        dtype=np.float32,
                    )
                    if probs.ndim >= 1 and masks.ndim == 3 and masks.shape[0] == probs.shape[0]:
                        masks = masks[probs >= score_threshold]
                else:
                    masks = np.asarray(data["masks"], dtype=np.uint8)
            if masks.ndim in (2, 3) and masks.shape[-2:] != target_shape:
                if not shape_warning_done:
                    print(
                        f"[sam3-video] fitting SAM3 mask shape {masks.shape[-2:]} "
                        f"to GUI target shape {target_shape}; this is usually AVI codec padding/cropping.",
                        flush=True,
                    )
                    shape_warning_done = True
                masks = self._fit_sam3_mask_array_to_shape(masks, target_shape)
            if masks.ndim == 3 and masks.shape[0] > 0:
                mask = np.any(masks > 0, axis=0).astype(np.uint8)
            elif masks.ndim == 2:
                mask = (masks > 0).astype(np.uint8)
            else:
                mask = np.zeros_like(new_masks[frame_index], dtype=np.uint8)

            block = self.static_block_masks[plume_idx].astype(bool) if self.static_block_masks else None
            if block is not None and np.any(block):
                mask = mask.copy()
                mask[block] = 0
            new_masks[frame_index] = mask

        return new_masks

    def _load_sam3_probability_frames(self, output_root, expected_frames, plume_idx):
        summary, npz_dir = self._sam3_video_summary_and_npz_dir(output_root)
        target_shape = self.plume_masks[plume_idx][0].shape
        prob_frames = [np.zeros(target_shape, dtype=np.float32) for _ in range(int(expected_frames))]
        max_probs = [None for _ in range(int(expected_frames))]
        raw_counts = [0 for _ in range(int(expected_frames))]
        shape_warning_done = False

        for record in summary.get("records", []):
            frame_index = int(record["frame_index"])
            if frame_index < 0 or frame_index >= len(prob_frames):
                continue
            npz_path = npz_dir / f"{record['frame_name']}.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                if "raw_masks" in data.files:
                    masks = np.asarray(data["raw_masks"], dtype=np.uint8)
                    probs = np.asarray(
                        data["raw_probabilities"] if "raw_probabilities" in data.files else [],
                        dtype=np.float32,
                    )
                else:
                    masks = np.asarray(data["masks"], dtype=np.uint8)
                    probs = np.asarray(
                        data["probabilities"] if "probabilities" in data.files else [],
                        dtype=np.float32,
                    )

            if masks.ndim == 2:
                masks = masks[None, ...]
            if masks.ndim != 3:
                continue
            if masks.shape[1:] != target_shape:
                if not shape_warning_done:
                    print(
                        f"[sam3-video] fitting SAM3 probability mask shape {masks.shape[1:]} "
                        f"to GUI target shape {target_shape}; this is usually AVI codec padding/cropping.",
                        flush=True,
                    )
                    shape_warning_done = True
                masks = self._fit_sam3_mask_array_to_shape(masks, target_shape)
            if probs.ndim < 1 or probs.shape[0] != masks.shape[0]:
                probs = np.ones((masks.shape[0],), dtype=np.float32)

            raw_counts[frame_index] = int(masks.shape[0])
            if probs.size:
                max_probs[frame_index] = float(np.max(probs))
            if masks.shape[0] > 0:
                weighted = (masks > 0).astype(np.float32) * probs[:, None, None]
                prob_frames[frame_index] = np.max(weighted, axis=0).astype(np.float32, copy=False)

        return prob_frames, max_probs, raw_counts

    def _refresh_last_sam3_video_masks_from_score(self, warn_if_empty=False):
        if self.last_sam3_video_result is None:
            raise RuntimeError("No SAM3 video result to refresh.")

        plume_idx = int(self.last_sam3_video_result["plume_idx"])
        if plume_idx < 0 or plume_idx >= len(self.plume_videos):
            raise RuntimeError(f"SAM3 video plume index is no longer valid: {plume_idx}")

        output_root = Path(self.last_sam3_video_result["output_root"])
        expected_frames = len(self.plume_videos[plume_idx])
        score_threshold = float(self.sam_score_threshold.get())
        masks = self._load_sam3_video_masks(
            output_root,
            expected_frames,
            plume_idx,
            score_threshold=score_threshold,
        )
        total_area, frames_with_mask = self._sam3_video_mask_stats(masks)
        self.last_sam3_video_result["masks"] = masks
        self.last_sam3_video_result["score_threshold"] = score_threshold
        self.last_sam3_video_result["total_area"] = int(total_area)
        self.last_sam3_video_result["frames_with_mask"] = int(frames_with_mask)

        if total_area == 0:
            print(
                f"[sam3-video] score={score_threshold:.3f} generated empty masks; existing GUI masks kept.",
                flush=True,
            )
            if warn_if_empty:
                messagebox.showwarning(
                    "SAM3 Score",
                    "The current SAM3 Score produces no foreground from the last video result.\n"
                    "Lower SAM3 Score and try Play SAM3 Boundary again.",
                )
            return masks, total_area, frames_with_mask

        self.plume_masks[plume_idx] = masks
        print(
            f"[sam3-video] regenerated GUI masks from raw video result score={score_threshold:.3f} "
            f"frames_with_mask={frames_with_mask} total_area={total_area}",
            flush=True,
        )
        self.update_image()
        return masks, total_area, frames_with_mask

    def _finish_sam3_video_success(self, plume_idx, masks_loaded, output_root, input_avi):
        total_area, frames_with_mask = self._sam3_video_mask_stats(masks_loaded)
        self.last_sam3_video_result = {
            "plume_idx": int(plume_idx),
            "input_avi": str(input_avi),
            "output_root": str(output_root),
            "masks": masks_loaded,
            "score_threshold": float(self.sam_score_threshold.get()),
            "total_area": int(total_area),
            "frames_with_mask": int(frames_with_mask),
        }
        if total_area == 0:
            print(
                f"[sam3-video] result is empty for all {len(masks_loaded)} frames; keeping existing GUI masks. output={output_root}",
                flush=True,
            )
            self.master.title("Manual Segmenter - SAM3 video returned empty; existing masks kept")
            messagebox.showwarning(
                "SAM3 Video",
                "SAM3 video returned no foreground, so the existing GUI masks were kept.\n\n"
                f"Artifacts:\n{output_root}",
            )
            self.update_image()
            return

        self.plume_masks[plume_idx] = masks_loaded
        print(
            f"[sam3-video] loaded masks plume={plume_idx} frames={len(masks_loaded)} "
            f"frames_with_mask={frames_with_mask} total_area={total_area} output={output_root}",
            flush=True,
        )
        print(
            "[sam3-video] use 'Play SAM3 Probability' to inspect raw probabilities, "
            "then adjust SAM3 Score and click 'Play SAM3 Boundary'.",
            flush=True,
        )
        self.master.title(f"Manual Segmenter - SAM3 video masks loaded. Tune Score or preview boundary")
        self.update_image()

    def _finish_sam3_video_error(self, exc, export_root):
        self.master.title("Manual Segmenter - SAM3 video failed")
        messagebox.showerror(
            "SAM3 Video Error",
            f"{exc}\n\nArtifacts, if any:\n{export_root}",
        )

    @staticmethod
    def _read_grayscale_video_float(video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open SAM3 input video: {video_path}")
        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                frames.append(gray.astype(np.float32) / 255.0)
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from SAM3 input video: {video_path}")
        return np.stack(frames, axis=0)

    @staticmethod
    def _masks_to_boundary_points(mask_frames):
        boundaries = []
        empty = np.empty((0, 2), dtype=np.int32)
        kernel = np.ones((3, 3), dtype=np.uint8)
        for mask in mask_frames:
            mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
            if np.any(mask_u8):
                eroded = cv2.erode(mask_u8, kernel, iterations=1)
                edge = (mask_u8 > 0) & (eroded == 0)
                coords_yx = np.argwhere(edge > 0).astype(np.int32)
            else:
                coords_yx = empty
            boundaries.append((coords_yx, empty))
        return boundaries

    @classmethod
    def _fit_video_frames_to_shape(cls, video, target_shape):
        return np.stack(
            [cls._fit_sam3_mask_array_to_shape(frame, target_shape) for frame in np.asarray(video)],
            axis=0,
        )

    @staticmethod
    def _play_probability_overlay(video, prob_frames, max_probs, raw_counts, score_threshold, intv=17):
        video = np.asarray(video)
        total = min(len(video), len(prob_frames))
        for idx in range(total):
            frame = np.asarray(video[idx])
            prob = np.clip(np.asarray(prob_frames[idx], dtype=np.float32), 0.0, 1.0)
            frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            if frame_u8.shape != prob.shape:
                frame_u8 = ManualSegmenter._fit_sam3_mask_array_to_shape(frame_u8, prob.shape)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)

            prob_u8 = np.round(prob * 255.0).astype(np.uint8)
            color_map = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
            color = cv2.applyColorMap(prob_u8, color_map)
            mask_any = prob > 0
            if np.any(mask_any):
                blended = cv2.addWeighted(color, 0.55, frame_bgr, 0.45, 0.0)
                frame_bgr[mask_any] = blended[mask_any]

            threshold_mask = prob >= float(score_threshold)
            if np.any(threshold_mask):
                edge = threshold_mask.astype(np.uint8)
                edge = edge - cv2.erode(edge, np.ones((3, 3), dtype=np.uint8), iterations=1)
                frame_bgr[edge > 0] = (0, 255, 255)

            max_prob = max_probs[idx]
            max_text = "none" if max_prob is None else f"{max_prob:.3f}"
            text = f"frame {idx} raw={raw_counts[idx]} max_p={max_text} score={score_threshold:.3f}"
            cv2.putText(
                frame_bgr,
                text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("SAM3 Probability", frame_bgr)
            if cv2.waitKey(intv) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    def play_last_sam3_video_probability(self):
        if self.last_sam3_video_result is None:
            messagebox.showinfo("SAM3 Probability", "No SAM3 video result to preview yet.")
            return

        input_avi = Path(self.last_sam3_video_result["input_avi"])
        output_root = Path(self.last_sam3_video_result["output_root"])
        plume_idx = int(self.last_sam3_video_result["plume_idx"])
        try:
            video = self._read_grayscale_video_float(input_avi)
            prob_frames, max_probs, raw_counts = self._load_sam3_probability_frames(
                output_root,
                len(video),
                plume_idx,
            )
            total_raw = int(sum(raw_counts))
            print(
                f"[sam3-video] playing probability preview frames={len(video)} raw_objects_total={total_raw} "
                f"score={float(self.sam_score_threshold.get()):.3f} input={input_avi}",
                flush=True,
            )
            self._play_probability_overlay(
                video,
                prob_frames,
                max_probs,
                raw_counts,
                score_threshold=float(self.sam_score_threshold.get()),
                intv=17,
            )
        except Exception as e:
            messagebox.showerror("SAM3 Probability", f"Failed to play SAM3 probability preview:\n{e}")

    def play_last_sam3_video_boundary(self):
        if self.last_sam3_video_result is None:
            messagebox.showinfo("SAM3 Boundary", "No SAM3 video result to preview yet.")
            return

        input_avi = Path(self.last_sam3_video_result["input_avi"])
        try:
            masks, total_area, frames_with_mask = self._refresh_last_sam3_video_masks_from_score(
                warn_if_empty=True
            )
            if total_area == 0:
                return
            video = self._read_grayscale_video_float(input_avi)
            total = min(len(video), len(masks))
            if total <= 0:
                raise RuntimeError("SAM3 video preview is empty.")
            video = self._fit_video_frames_to_shape(video[:total], masks[0].shape)
            boundaries = self._masks_to_boundary_points(masks[:total])
            print(
                f"[sam3-video] playing boundary preview frames={total} frames_with_mask={frames_with_mask} "
                f"score={float(self.sam_score_threshold.get()):.3f} input={input_avi}",
                flush=True,
            )
            play_video_with_boundaries_cv2(
                video,
                boundaries,
                gain=1.0,
                intv=17,
                color_top=(0, 0, 255),
                color_bottom=(0, 0, 255),
                thickness=1,
                alpha=1.0,
            )
        except Exception as e:
            messagebox.showerror("SAM3 Boundary", f"Failed to play SAM3 boundary preview:\n{e}")

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
        if np.issubdtype(frame.dtype, np.integer) and frame.dtype.itemsize == 1:
            return frame.astype(np.uint16) * np.uint16(257)
        arr = np.nan_to_num(frame, nan=0.0, posinf=65535.0, neginf=0.0)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.5:
            arr = arr * 65535.0
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
            "video_normalization": self.video_normalization,
            "video_normalization_params": self.video_normalization_params,
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
