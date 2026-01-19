import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import json
import os
import cv2

from OSCC_postprocessing.cine.cine_utils import CineReader
from OSCC_postprocessing.analysis.cone_angle import angle_signal_density
from OSCC_postprocessing.rotation.rotate_crop import rotate_and_crop, generate_CropRect, generate_plume_mask
from OSCC_postprocessing.utils.zoom_utils import enlarge_image

class ManualSegmenter:
    """Interactive tool to create pixel-wise masks for rotated plume videos."""

    def __init__(self, master):
        self.master = master
        master.title("Manual Segmenter")

        self.reader = None
        self.video_path = None
        self.video = None  # Original video frames
        self.plume_videos = []  # Rotated plume-specific videos
        self.plume_masks = []  # Per-plume, per-frame masks
        self.current_frame = 0
        self.current_plume = 0
        self.zoom = 1
        self.tool = 'grabcut'
        self.brush_size = 5

        self.gain = tk.DoubleVar(value=1.0)
        self.gamma = tk.DoubleVar(value=1.0)

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.master)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Load Video", command=self.load_video).pack(side=tk.LEFT)
        ttk.Button(top, text="Load Config", command=self.load_config).pack(side=tk.LEFT)
        ttk.Button(top, text="Prev Frame", command=self.prev_frame).pack(side=tk.LEFT)
        ttk.Button(top, text="Next Frame", command=self.next_frame).pack(side=tk.LEFT)
        ttk.Button(top, text="Prev Plume", command=self.prev_plume).pack(side=tk.LEFT)
        ttk.Button(top, text="Next Plume", command=self.next_plume).pack(side=tk.LEFT)
        ttk.Button(top, text="Brush Tool", command=lambda: self.set_tool('brush')).pack(side=tk.LEFT)
        ttk.Button(top, text="GrabCut Tool", command=lambda: self.set_tool('grabcut')).pack(side=tk.LEFT)
        ttk.Button(top, text="Save", command=self.save_current).pack(side=tk.LEFT)

        ttk.Label(top, text="Gain").pack(side=tk.LEFT, padx=(10,0))
        ttk.Scale(top, from_=0.1, to=10, variable=self.gain, orient=tk.HORIZONTAL,
                  command=lambda e: self.update_image()).pack(side=tk.LEFT)
        ttk.Label(top, text="Gamma").pack(side=tk.LEFT)
        ttk.Scale(top, from_=0.1, to=3, variable=self.gamma, orient=tk.HORIZONTAL,
                  command=lambda e: self.update_image()).pack(side=tk.LEFT)

        cf = ttk.Frame(self.master)
        cf.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cf, bg='black')
        hbar = ttk.Scrollbar(cf, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = ttk.Scrollbar(cf, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        self.canvas.bind('<ButtonPress-1>', self.on_left_press)
        self.canvas.bind('<B1-Motion>', self.on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_left_release)
        self.canvas.bind('<ButtonPress-3>', self.on_right_press)
        self.canvas.bind('<B3-Motion>', self.on_right_drag)
        self.canvas.bind('<ButtonRelease-3>', self.on_right_release)
        self.canvas.bind('<MouseWheel>', self._on_zoom)
        self.canvas.bind('<Button-4>', self._on_zoom)
        self.canvas.bind('<Button-5>', self._on_zoom)

    # ---------------------------------------------------------------
    #                       Loading utilities
    # ---------------------------------------------------------------
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[('Cine', '*.cine'), ('All', '*.*')])
        if not path:
            return
        try:
            reader = CineReader()
            reader.load(path)
        except Exception as e:
            messagebox.showerror('Error', f'Could not open video:\n{e}')
            return
        frames = []
        for i in range(reader.frame_count):
            frames.append(reader.read_frame(i).astype(np.float32))
        self.reader = reader
        self.video_path = path
        self.video = np.stack(frames, axis=0)
        self.current_frame = 0
        self.plume_videos = []
        self.plume_masks = []
        self.update_image()

    def load_config(self):
        if self.video is None:
            messagebox.showinfo('Info', 'Load a video first')
            return
        path = filedialog.askopenfilename(filetypes=[('Config', '*.json')])
        if not path:
            return
        with open(path, 'r') as f:
            cfg = json.load(f)
        n_plumes = int(cfg['plumes'])
        cx = float(cfg['centre_x'])
        cy = float(cfg['centre_y'])
        bins, sig, _ = angle_signal_density(self.video, cx, cy, N_bins=360)
        summed = sig.sum(axis=0)
        fft_vals = np.fft.rfft(summed)
        phase = np.angle(fft_vals[n_plumes]) if n_plumes < len(fft_vals) else 0.0
        offset = (-phase / n_plumes) * 180.0 / np.pi
        offset %= 360.0
        inner = 0 # int(cfg.get('calib_radius', 0))
        outer = 380 # int(cfg.get('outer_radius', inner * 2))
        crop = generate_CropRect(inner, outer, n_plumes, cx, cy)
        mask = generate_plume_mask(inner, outer, crop[2], crop[3])
        angles = np.linspace(0, 360, n_plumes, endpoint=False) - offset
        self.plume_videos = []
        for ang in angles:
            seg = rotate_and_crop(self.video, ang, crop, (cx, cy), is_video=True, mask=mask)
            self.plume_videos.append(seg)
        self.plume_masks = [ [np.zeros(seg[0].shape, dtype=np.uint8) for _ in range(seg.shape[0])] for seg in self.plume_videos ]
        self.current_frame = 0
        self.current_plume = 0
        self.update_image()

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
        self.current_frame = min(self.plume_videos[0].shape[0] - 1, self.current_frame + 1)
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
        self.canvas.delete('all')
        if not self.plume_videos:
            return
        frame = self.plume_videos[self.current_plume][self.current_frame]
        img8 = self.apply_gain_gamma(frame)
        self.current_img = img8
        disp = enlarge_image(Image.fromarray(img8), int(self.zoom))
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.config(scrollregion=(0, 0, disp.width, disp.height))
        mask = self.plume_masks[self.current_plume][self.current_frame]
        if np.any(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                pts = cnt.reshape(-1, 2) * self.zoom
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    self.canvas.create_line(x1, y1, x2, y2, fill='yellow', dash=(4, 2))

    # ---------------------------------------------------------------
    #                       Mask editing
    # ---------------------------------------------------------------
    def on_left_press(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if self.tool == 'brush':
            self.last_pos = (x, y)
            mask = self.plume_masks[self.current_plume][self.current_frame]
            cv2.circle(mask, (x, y), self.brush_size, 1, -1)
            self.update_image()
        else:
            self.start_pos = (x, y)

    def on_left_drag(self, event):
        if not self.plume_videos or self.tool != 'brush':
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        mask = self.plume_masks[self.current_plume][self.current_frame]
        cv2.line(mask, self.last_pos, (x, y), 1, self.brush_size * 2)
        self.last_pos = (x, y)
        self.update_image()

    def on_left_release(self, event):
        if not self.plume_videos or self.tool != 'grabcut':
            return
        x0, y0 = self.start_pos
        x1 = int(self.canvas.canvasx(event.x) / self.zoom)
        y1 = int(self.canvas.canvasy(event.y) / self.zoom)
        rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        self.run_grabcut(rect, add=True)

    def on_right_press(self, event):
        if not self.plume_videos:
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        if self.tool == 'brush':
            self.last_pos = (x, y)
            mask = self.plume_masks[self.current_plume][self.current_frame]
            cv2.circle(mask, (x, y), self.brush_size, 0, -1)
            self.update_image()
        else:
            self.start_pos = (x, y)

    def on_right_drag(self, event):
        if not self.plume_videos or self.tool != 'brush':
            return
        x = int(self.canvas.canvasx(event.x) / self.zoom)
        y = int(self.canvas.canvasy(event.y) / self.zoom)
        mask = self.plume_masks[self.current_plume][self.current_frame]
        cv2.line(mask, self.last_pos, (x, y), 0, self.brush_size * 2)
        self.last_pos = (x, y)
        self.update_image()

    def on_right_release(self, event):
        if not self.plume_videos or self.tool != 'grabcut':
            return
        x0, y0 = self.start_pos
        x1 = int(self.canvas.canvasx(event.x) / self.zoom)
        y1 = int(self.canvas.canvasy(event.y) / self.zoom)
        mask = self.plume_masks[self.current_plume][self.current_frame]
        cv2.rectangle(mask, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), 0, -1)
        self.update_image()

    def _on_zoom(self, event):
        direction = 1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4 else -1
        self.zoom = max(1, self.zoom + direction)
        self.update_image()

    def set_tool(self, tool):
        self.tool = tool

    def run_grabcut(self, rect, add=True):
        frame = self.current_img
        mask = self.plume_masks[self.current_plume][self.current_frame]
        gc_mask = np.where(mask, cv2.GC_FGD, cv2.GC_BGD).astype('uint8')
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(frame, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return
        new_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        if add:
            mask[:] = np.maximum(mask, new_mask)
        else:
            mask[:] = mask & (~new_mask)
        self.update_image()

    # ---------------------------------------------------------------
    #                       Saving
    # ---------------------------------------------------------------
    def save_current(self):
        if not self.plume_videos:
            return
        img = self.current_img
        mask = self.plume_masks[self.current_plume][self.current_frame]
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        name = f"{base}_f{self.current_frame:04d}_p{self.current_plume:02d}.npz"
        path = filedialog.asksaveasfilename(defaultextension='.npz', initialfile=name)
        if not path:
            return
        np.savez_compressed(path, image=img, mask=mask)
        messagebox.showinfo('Saved', f'Saved to {path}')

if __name__ == '__main__':
    root = tk.Tk()
    app = ManualSegmenter(root)
    root.mainloop()
