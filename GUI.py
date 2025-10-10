import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageOps
import numpy as np
import os
import json

from mie_postprocessing.zoom_utils import enlarge_image
from mie_postprocessing.cine_utils import CineReader
from mie_postprocessing.circ_calculator import calc_circle
from mie_postprocessing.cone_angle import angle_signal_density, plot_angle_signal_density
from mie_postprocessing.functions_videos import video_histogram_with_contour


import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class FrameSelector(tk.Toplevel):
    """Simple viewer to pick a frame from the loaded video."""

    def __init__(self, parent):
        super().__init__(parent.master)
        self.parent = parent
        self.reader = parent.reader
        self.current_index = parent.current_index
        self.zoom_factor = 1

        self.title("Select Frame")

        self._build_ui()
        self.show_frame(self.current_index)

    def _build_ui(self):
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.prev_btn = ttk.Button(ctrl, text="Prev", command=self.prev_frame)
        self.prev_btn.grid(row=0, column=0, padx=2)
        self.next_btn = ttk.Button(ctrl, text="Next", command=self.next_frame)
        self.next_btn.grid(row=0, column=1, padx=2)

        ttk.Label(ctrl, text="Frame:").grid(row=0, column=2, padx=(10, 0))
        self.frame_var = tk.IntVar(value=self.current_index + 1)
        ttk.Entry(ctrl, textvariable=self.frame_var, width=6).grid(row=0, column=3)
        ttk.Button(ctrl, text="Go", command=self.goto_frame).grid(row=0, column=4, padx=2)
        ttk.Button(ctrl, text="Use Frame", command=self.use_frame).grid(row=0, column=5, padx=(10, 0))

        self.info = ttk.Label(ctrl, text="")
        self.info.grid(row=0, column=6, padx=5)

        cf = ttk.Frame(self)
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

        self.canvas.bind('<MouseWheel>', self._on_zoom)
        self.canvas.bind('<Button-4>', self._on_zoom)
        self.canvas.bind('<Button-5>', self._on_zoom)

    def read_frame(self, idx):
        frame = self.reader.read_frame(idx)
        img8 = np.clip(frame / 16, 0, 255).astype(np.uint8)
        return Image.fromarray(img8)

    def show_frame(self, idx):
        if 0 <= idx < self.reader.frame_count:
            self.current_index = idx
            img = enlarge_image(self.read_frame(idx), self.zoom_factor)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete('IMG')
            self.canvas.create_image(0, 0, anchor='nw', image=self.photo, tags='IMG')
            self.canvas.config(scrollregion=(0, 0, img.width, img.height))
            self.info.config(text=f"Frame {idx + 1}/{self.reader.frame_count}")
            self.frame_var.set(idx + 1)

    def prev_frame(self):
        if self.current_index > 0:
            self.show_frame(self.current_index - 1)

    def next_frame(self):
        if self.current_index < self.reader.frame_count - 1:
            self.show_frame(self.current_index + 1)

    def goto_frame(self):
        idx = self.frame_var.get() - 1
        if 0 <= idx < self.reader.frame_count:
            self.show_frame(idx)

    def _on_zoom(self, event):
        direction = 1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4 else -1
        self.zoom_factor = max(1, self.zoom_factor + direction)
        self.show_frame(self.current_index)

    def use_frame(self):
        self.parent.current_index = self.current_index
        self.parent.update_image()
        self.destroy()
        
class CircleSelector(tk.Toplevel):
    """Window to pick N points and compute a best-fit circle."""

    def __init__(self, parent, image, num_points):
        super().__init__(parent.master)
        self.parent = parent
        self.image = image
        self.num_points = max(3, int(num_points))
        self.zoom_factor = 1
        self.points = []

        self.title("Select Circle")

        cf = ttk.Frame(self)
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
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<MouseWheel>', self._on_zoom)
        self.canvas.bind('<Button-4>', self._on_zoom)
        self.canvas.bind('<Button-5>', self._on_zoom)

        self.show_frame()



    def show_frame(self):
        img = enlarge_image(self.image, int(self.zoom_factor))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.config(scrollregion=(0, 0, self.photo.width(), self.photo.height()))
        self._draw_points()

    def _on_zoom(self, event):
        direction = 1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4 else -1
        self.zoom_factor = max(1, self.zoom_factor + direction)
        self.show_frame()

    def _draw_points(self):
        r = 3
        self.canvas.delete('POINT')
        for x, y in self.points:
            sx = x * self.zoom_factor
            sy = y * self.zoom_factor
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                   outline='red', tags='POINT')

    def _on_click(self, event):
        x = self.canvas.canvasx(event.x) / self.zoom_factor
        y = self.canvas.canvasy(event.y) / self.zoom_factor
        self.points.append((x, y))
        self._draw_points()
        if len(self.points) == self.num_points:
            center, radius, offset = calc_circle(*self.points)
            print(f"Circle radius: {radius}")
            self.parent.coord_x.set(center[0])
            self.parent.coord_y.set(center[1])
            self.parent.calib_radius_var.set(radius)
            try:
                self.parent.inner_radius.set(int(round(radius)))
            except Exception:
                pass
            try:
                self.parent.plume_offset.set(offset % 360.0)
            except Exception:
                pass
            self.parent._draw_scaled()
            self.destroy()


# ------------------------------------------------------------------
#                         MAIN ANNOTATOR UI
# ------------------------------------------------------------------
class VideoAnnotatorUI:
    def __init__(self, master):
        self.master = master
        master.title("Cine Video Annotator")

        # Layout grid positions (col indices)
        self.layout_positions = {
            'load_btn': 0, 'frame_label': 1, 'prev_btn': 2,
            'next_btn': 3, 'select_btn': 4,
            'param_start_col': 1, 'confirm_btn': 13
        }

        # Video / data
        self.reader = CineReader()
        self.total_frames = 0
        self.current_index = 0
        self.zoom_factor = 1
        self.orig_img = np.zeros_like  # placeholder for PIL Image
        self.base_rgba = None          # RGBA cached base frame
        self.display_pad = 100           # extra border for visualization
        # Offsets for panning within the zoomed image
        self.offset_x = 0
        self.offset_y = 0

        self.mask = np.zeros_like      # H×W uint8 mask (0/1)
        # Brush settings
        self.brush_color = (255, 0, 0)
        self.alpha_var = tk.IntVar(value=255)
        self.brush_shape = tk.StringVar(value='circle')
        self.brush_size = tk.IntVar(value=10)
        self.show_mask = tk.BooleanVar(value=True)
        self.coord_x = tk.DoubleVar(value=0.0)
        self.coord_y = tk.DoubleVar(value=0.0)
        self.calib_radius_var = tk.DoubleVar(value=0.0)
        self.calib_radius_var.trace_add('write', lambda *a: self._draw_scaled())
        # Processing params dictionary (gain, gamma, etc.)
        self.vars = {}

        # Build UI
        self._build_controls(master)
        self._build_content(master)
        self._update_calib_button()

    # ------------------------------------------------------------------
    #                           CONTROL BAR
    # ------------------------------------------------------------------

    def _build_controls(self, parent):
        ctrl = ttk.Frame(parent)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lp = self.layout_positions
        bc = lp['param_start_col']

        # Row 0: Load / navigation controls
        self.load_btn = ttk.Button(ctrl, text="Load Video", command=self.load_video)
        self.load_btn.grid(row=0, column=lp['load_btn'], padx=2)
        self.frame_label = ttk.Label(ctrl, text="Frame: 0/0")
        self.frame_label.grid(row=0, column=lp['frame_label'], padx=5)
        self.frame_entry = tk.IntVar(value=1)
        ttk.Entry(ctrl, textvariable=self.frame_entry, width=5).grid(row=0, column=lp['frame_label']+1)
        ttk.Button(ctrl, text="Go", command=self._on_go_frame).grid(row=0, column=lp['frame_label']+2)
        self.prev_btn = ttk.Button(ctrl, text="Prev Frame", command=self.prev_frame, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=lp['prev_btn'], padx=2)
        self.next_btn = ttk.Button(ctrl, text="Next Frame", command=self.next_frame, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=lp['next_btn'], padx=2)
        self.select_btn = ttk.Button(ctrl, text="Select Frame", command=self.open_frame_selector, state=tk.DISABLED)
        self.select_btn.grid(row=0, column=lp['select_btn'], padx=2)
        self.export_btn = ttk.Button(ctrl, text="Export Mask", command=self.export_mask, state=tk.DISABLED)
        self.export_btn.grid(row=0, column=lp['param_start_col']+10, padx=2)

        # Row 1: Plume visualization parameters 
        ttk.Label(ctrl, text="Plumes:").grid(row=1, column=bc, pady=(5,0))
        self.num_plumes = tk.IntVar(value=0)
        ttk.Entry(ctrl, textvariable=self.num_plumes, width=5).grid(row=1, column=bc+1, pady=(5,0))
        ttk.Label(ctrl, text="Offset:").grid(row=1, column=bc+2, pady=(5,0))
        self.plume_offset = tk.DoubleVar(value=0.0)
        ttk.Entry(ctrl, textvariable=self.plume_offset, width=5).grid(row=1, column=bc+3, pady=(5,0))

        # Nozzle coordinates and processing parameters 
        # Update calibration button when number of plumes changes
        self.num_plumes.trace_add('write', lambda *args: self._update_calib_button())
        ttk.Label(ctrl, text="Centre X:").grid(row=1, column=bc+4, pady=(5,0))
        ttk.Entry(ctrl, textvariable=self.coord_x, width=7).grid(row=1, column=bc+5, pady=(5,0))
        ttk.Label(ctrl, text="Centre Y:").grid(row=1, column=bc+6, pady=(5,0))
        ttk.Entry(ctrl, textvariable=self.coord_y, width=7).grid(row=1, column=bc+7, pady=(5,0))
        ttk.Label(ctrl, text="Calib R:").grid(row=1, column=bc+8, pady=(5,0))
        ttk.Entry(ctrl, textvariable=self.calib_radius_var, width=7).grid(row=1, column=bc+9, pady=(5,0))

        self.circle_btn = ttk.Button(ctrl, text="Calibration", command=self.open_circle_selector, state=tk.DISABLED)
        self.circle_btn.grid(row=1, column=bc+10, padx=2)

        self.save_cfg_btn = ttk.Button(ctrl, text="Save Config", command=self.save_config)
        self.save_cfg_btn.grid(row=1, column=bc+11, padx=2)



        # Row 2: Gain/Gamma/Black/White + Apply
        for i,name in enumerate(["Gain","Gamma","Black","White"]):
            ttk.Label(ctrl, text=f"{name}:").grid(
                row=2, column=lp['param_start_col']+i*2, pady=(5,0))
            v = tk.DoubleVar(value=1.0 if name in ("Gain","Gamma") else 0.0)
            v = tk.DoubleVar(value=255 if name in ("White") else v.get())
            ttk.Entry(ctrl, textvariable=v, width=5).grid(
                row=2, column=lp['param_start_col']+i*2+1, pady=(5,0))
            self.vars[name.lower()] = v
            self.confirm_btn = ttk.Button(ctrl, text="Apply", command=lambda: self.update_image(compute_global=False), state=tk.DISABLED)
        self.confirm_btn.grid(row=2, column=lp['param_start_col']+10, padx=5, pady=(5,0))




        # Row 3: Brush controls
        ttk.Label(ctrl, text="Brush:").grid(row=3, column=bc, pady=(5,0))
        ttk.Combobox(ctrl, textvariable=self.brush_shape, values=['circle','square'], width=10)\
            .grid(row=3, column=bc+1, pady=(5,0))
        ttk.Checkbutton(ctrl, text="Show Mask", variable=self.show_mask, command=self._draw_scaled).grid(row=3, column=bc+3, pady=(5,0))
        ttk.Label(ctrl, text="Size:").grid(row=3, column=bc+4, pady=(5,0))
        ttk.Entry(ctrl, textvariable=self.brush_size, width=5).grid(row=3, column=bc+5, pady=(5,0))
        ttk.Button(ctrl, text="Color", command=self.choose_color)\
            .grid(row=3, column=bc+7, padx=(10,0), pady=(5,0))
        ttk.Label(ctrl, text="Alpha:").grid(row=3, column=bc+8, pady=(5,0))
        tk.Scale(ctrl, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.alpha_var, length=100).grid(
            row=3, column=bc+9, pady=(5,0), sticky='w')
        
        # Row 4: Ring mask parameters
        ttk.Label(ctrl, text="Inner R:").grid(row=4, column=bc, pady=(5,0))
        self.inner_radius = tk.IntVar(value=0)
        ttk.Entry(ctrl, textvariable=self.inner_radius, width=5).grid(row=4, column=bc+1, pady=(5,0))
        ttk.Label(ctrl, text="Outer R:").grid(row=4, column=bc+2, pady=(5,0))
        self.outer_radius = tk.IntVar(value=0)
        ttk.Entry(ctrl, textvariable=self.outer_radius, width=5).grid(row=4, column=bc+3, pady=(5,0))
        ttk.Button(ctrl, text="Add Ring", command=self.add_ring_mask).grid(row=4, column=bc+4, padx=(10,0), pady=(5,0))

    def add_ring_mask(self):
        """Add a ring (or circle) mask centered at the calibration coordinate."""
        if self.mask is None:
            return
        cx = self.coord_x.get()
        cy = self.coord_y.get()
        r_in = max(0, self.inner_radius.get())
        r_out = self.outer_radius.get()
        if r_out <= 0:
            return

        yy, xx = np.ogrid[:self.mask.shape[0], :self.mask.shape[1]]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = dist2 <= r_out ** 2
        if r_in > 0:
            ring &= dist2 >= r_in ** 2
        self.mask[ring] = 1
        self._draw_scaled()

    def choose_color(self):
        color = colorchooser.askcolor(color='#%02x%02x%02x' % self.brush_color, title='Select brush color')
        if color[0]:
            self.brush_color = tuple(map(int, color[0]))

    # ------------------------------------------------------------------
    #                       MAIN CONTENT AREA
    # ------------------------------------------------------------------
    def _build_content(self, parent):
        content = ttk.Frame(parent)
        content.pack(fill=tk.BOTH, expand=True)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        # Canvas with scrollbars
        cf = ttk.Frame(content)
        cf.grid(row=0, column=0, sticky='nsew')
        self.canvas = tk.Canvas(cf, bg='black')
        self.hbar = ttk.Scrollbar(cf, orient=tk.HORIZONTAL, command=self._scroll_x)
        self.vbar = ttk.Scrollbar(cf, orient=tk.VERTICAL,   command=self._scroll_y)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.hbar.grid(row=1, column=0, sticky='we')
        self.vbar.grid(row=0, column=1, sticky='ns')
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        # Histogram panel
        hf = ttk.Frame(content)
        hf.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        hf.rowconfigure(0, weight=1)
        hf.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(4, 4))
        self.ax_hist = self.fig.add_subplot(211)
        self.ax_angle = self.fig.add_subplot(212)
        self.ax_angle.set_xlim(-360, 360)
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=hf)
        self.canvas_hist.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        hf.bind('<Configure>', lambda e: self._on_hist_resize(e.width))

        # Canvas bindings
        self.canvas.bind('<MouseWheel>',    self._on_zoom)
        self.canvas.bind('<Button-4>',      self._on_zoom)
        self.canvas.bind('<Button-5>',      self._on_zoom)
        self.canvas.bind('<B1-Motion>',     lambda e: self._on_paint(e, True))
        self.canvas.bind('<ButtonPress-1>', lambda e: self._on_paint(e, True))
        self.canvas.bind('<B3-Motion>',     lambda e: self._on_paint(e, False))
        self.canvas.bind('<ButtonPress-3>', lambda e: self._on_paint(e, False))
        self.canvas.bind('<Configure>',     lambda e: self._draw_scaled())

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[('Cine','*.cine')])
        try:
            self.reader.load(path)
        except Exception as e:
            messagebox.showerror('Error', f'Cannot load video:\n{e}')
            return

        self.total_frames = self.reader.frame_count
        self.mask = np.zeros((self.reader.height, self.reader.width), dtype=np.uint8)
        self.current_index = 0
        for w in (self.prev_btn, self.next_btn, self.confirm_btn, self.select_btn, self.export_btn, self.circle_btn):
            w.config(state=tk.NORMAL)
        self.update_image(compute_global=False)

    def prev_frame(self):
        if self.current_index>0:
            self.current_index-=1
            self.update_image()

    def next_frame(self):
        if self.current_index<self.total_frames-1:
            self.current_index+=1
            self.update_image()

    def update_image(self, compute_global=False):
        frame = self.reader.read_frame(self.current_index).astype(np.float32)
        if self.mask is None or self.mask.shape != frame.shape:
            self.mask = np.zeros(frame.shape, dtype=np.uint8)
        g, gm, bl, wh = [self.vars[k].get() for k in ('gain','gamma','black','white')]
        img = frame/4096 * g
        img = np.clip(img,0,1)
        if gm>0 and gm!=1: img=img**gm
        img8 = (img*255).astype(np.uint8)
        if wh>bl and bl>=0:
            # img8 = np.clip((img8-bl)*(255/(wh-bl)),0,255).astype(np.uint8)
            img8[img8 < bl] = 0
            img8[img8 > wh] = 255
        self.orig_img = Image.fromarray(img8)
        self.base_rgba = self.orig_img.convert('RGBA')
        self.base_rgba_pad = ImageOps.expand(
            self.base_rgba,
            border=(0, 0, self.display_pad, self.display_pad),
            fill=(0, 0, 0, 255))
        self.offset_x = 0
        self.offset_y = 0
        self.ax_hist.clear()
        # display histogram with logarithmic y-axis for better dynamic range
        self.ax_hist.hist(img8.ravel(), bins=256, log=True)
        self.ax_hist.set_title('Processed Histogram')
        self.ax_hist.set_ylabel('Count (log scale)')

        cx = self.coord_x.get()
        cy = self.coord_y.get()
        self.ax_angle.clear()
        if cx != 0 or cy != 0:
            bins, sig, _ = angle_signal_density(img8, cx, cy, N_bins=90)
            bins_ext = np.concatenate((bins - 360, bins))
            sig_ext = np.concatenate((sig, sig))
            # logarithmic scale requires strictly positive values
            # sig_ext = sig_ext + 1e-9
            # self.ax_angle.semilogy(bins_ext, sig_ext, color='green')
            self.ax_angle.plot(bins_ext, sig_ext, color='green', alpha=0.5)
            n_plumes = int(self.num_plumes.get()) if self.num_plumes.get() > 0 else 0
            if n_plumes > 0:
                step = 360.0 / n_plumes
                offset = self.plume_offset.get() % 360.0
                for i in range(n_plumes):
                    ang = offset + i * step
                    for shift in (-360, 0):
                        if ang + shift > -180:
                            display_ang = (ang + shift) % 360 
                            if display_ang > 180:
                                display_ang -= 360
                            self.ax_angle.axvline(display_ang, color='cyan', linestyle='--')
        self.ax_angle.set_xlim(-180, 180)
        # self.ax_angle.set_title('Angular Signal Density (ln scale)')
        self.ax_angle.set_title('Angular Signal Density')
        self.ax_angle.set_xlabel('Angle (deg)')
        # self.ax_angle.set_ylabel('Signal (log scale)')
        self.ax_angle.set_ylabel('Signal')
        self.canvas_hist.draw()
        if compute_global:
            self._compute_global_plots()
        self._draw_scaled(); self.frame_label.config(text=f"Frame: {self.current_index+1}/{self.total_frames}")
    
    def _update_zoomed_base(self):
        """Cache a zoomed version of the base image for faster drawing."""
        if self.base_rgba is not None:
            self.scaled_base = enlarge_image(self.base_rgba_pad, int(self.zoom_factor))
    
    def _on_hist_resize(self, width):
        """Resize matplotlib figure width when the side panel changes."""
        if width <= 0:
            return
        w_in = width / self.fig.dpi
        h_in = self.fig.get_size_inches()[1]
        self.fig.set_size_inches(max(3, w_in), h_in, forward=True)
        self.canvas_hist.draw_idle()

    def _draw_scaled(self):
        """Redraw the canvas showing only the visible zoomed region."""

        if self.base_rgba is None:
            return

        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        scaled_w = (self.orig_img.width + self.display_pad) * self.zoom_factor
        scaled_h = (self.orig_img.height + self.display_pad) * self.zoom_factor

        if cw <= 1 or ch <= 1:
            # Canvas not yet properly sized; draw the whole image
            cw, ch = scaled_w, scaled_h

        x0s = max(0, int(self.canvas.canvasx(0)))
        y0s = max(0, int(self.canvas.canvasy(0)))
        x1s = min(x0s + cw, scaled_w)
        y1s = min(y0s + ch, scaled_h)

        if x1s <= x0s or y1s <= y0s:
            x0s, y0s = 0, 0
            x1s, y1s = scaled_w, scaled_h

        x0 = x0s // self.zoom_factor
        y0 = y0s // self.zoom_factor
        x1 = int(np.ceil(x1s / self.zoom_factor))
        y1 = int(np.ceil(y1s / self.zoom_factor))

        base_tile = self.base_rgba_pad.crop((x0, y0, x1, y1))
        
        mask_pad = np.pad(
            self.mask,
            ((0, self.display_pad), (0, self.display_pad)),
            mode='constant',
            constant_values=0,
        )

        mask_tile = mask_pad[y0:y1, x0:x1]

        base_tile = enlarge_image(base_tile, int(self.zoom_factor))
        composited = base_tile.convert('RGBA')
        if self.show_mask.get():
            mask_img = Image.fromarray((mask_tile * 255).astype(np.uint8))
            mask_img = enlarge_image(mask_img, int(self.zoom_factor)).convert('L')
            overlay = Image.new('RGBA', mask_img.size,
                                (*self.brush_color, self.alpha_var.get()))
            composited.paste(overlay, (0, 0), mask_img)

        self.photo = ImageTk.PhotoImage(composited)
        self.canvas.delete('IMG')
        self.canvas.create_image(x0s, y0s, anchor='nw', image=self.photo, tags='IMG')
        self.canvas.delete('CENTER')
        self.canvas.delete('PLUME')
        self.canvas.delete('CALIBCIRCLE')
        cx = self.coord_x.get()*self.zoom_factor
        cy = self.coord_y.get()*self.zoom_factor
        r = 5
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline='yellow', width=2, tags='CENTER')

        radius = self.calib_radius_var.get()
        if radius > 0:
            rr = radius * self.zoom_factor
            self.canvas.create_oval(cx-rr, cy-rr, cx+rr, cy+rr,
                                   outline='red', dash=(4,), tags='CALIBCIRCLE')
            


        n_plumes = int(self.num_plumes.get()) if self.num_plumes.get() > 0 else 0
        if n_plumes > 0:
            step = 360.0 / n_plumes
            offset = self.plume_offset.get() % 360.0
            length = max(self.mask.shape) * self.zoom_factor
            for i in range(n_plumes):
                ang = np.deg2rad(offset + i * step)
                x_end = cx + length * np.cos(ang)
                y_end = cy - length * np.sin(ang)
                self.canvas.create_line(cx, cy, x_end, y_end, fill='cyan', tags='PLUME')
                mid_ang = np.deg2rad(offset + i * step + step / 2)
                mx = cx + length * np.cos(mid_ang)
                my = cy - length * np.sin(mid_ang)
                self.canvas.create_line(cx, cy, mx, my, fill='white', dash=(5,), tags='PLUME')

        self.canvas.config(scrollregion=(0, 0, scaled_w, scaled_h))

    def _scroll_x(self, *args):
        self.canvas.xview(*args)
        self._draw_scaled()

    def _scroll_y(self, *args):
        self.canvas.yview(*args)
        self._draw_scaled()

    def _on_zoom(self, event):
        """Zoom in or out in integer steps using the mouse wheel."""
        direction = 1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', None) == 4 else -1
        self.zoom_factor = max(1, self.zoom_factor + direction)
        self._update_zoomed_base()
        self._draw_scaled()

    def _on_paint(self,event,paint=True):
        x=int(self.canvas.canvasx(event.x)/self.zoom_factor)
        y=int(self.canvas.canvasy(event.y)/self.zoom_factor)
        x=max(0, min(x, self.mask.shape[1]-1))
        y=max(0, min(y, self.mask.shape[0]-1))
        
        size=self.brush_size.get()
        if self.brush_shape.get()=='circle':
            yy,xx=np.ogrid[-y:self.mask.shape[0]-y, -x:self.mask.shape[1]-x]; mask_area = xx*xx+yy*yy<=size*size
            self.mask[mask_area] = 1 if paint else 0
        else:
            x0,x1=max(0,x-size),min(self.mask.shape[1],x+size)
            y0,y1=max(0,y-size),min(self.mask.shape[0],y+size)
            self.mask[y0:y1,x0:x1] = 1 if paint else 0
        self._draw_scaled()
    
    def _on_go_frame(self):
        """Jump to a specified frame number from entry box."""
        idx = self.frame_entry.get() - 1
        if 0 <= idx < self.total_frames:
            self.current_index = idx
            self.update_image()

    def open_frame_selector(self):
        """Open a dialog to choose a frame visually."""
        if self.total_frames:
            FrameSelector(self)
    
    def open_circle_selector(self):
        """Open a dialog to pick points and compute a calibration circle."""
        if not self.total_frames:
            return
        n = self.num_plumes.get()
        if n <= 0:
            print("Set number of plumes before calibration")
            return
        CircleSelector(self, self.orig_img, n)

    def _update_calib_button(self):
        state = tk.NORMAL if (self.total_frames and self.num_plumes.get() > 0) else tk.DISABLED
        self.circle_btn.config(state=state)

    def _process_frame(self, frame):
        """Apply gain/gamma/black/white adjustments and return float frame."""
        g, gm, bl, wh = [self.vars[k].get() for k in ('gain', 'gamma', 'black', 'white')]
        img = frame / 4096.0 * g
        img = np.clip(img, 0, 1)
        if gm > 0 and gm != 1:
            img = img ** gm
        img8 = (img * 255).astype(np.uint8)
        if wh > bl and bl >= 0:
            img8 = np.clip((img8 - bl) * (255.0 / (wh - bl)), 0, 255).astype(np.uint8)
        return img8.astype(np.float32) / 255.0

    def _compute_global_plots(self):
        """Precompute histogram and angle density for the entire video."""
        if self.total_frames == 0:
            return

        frames = []
        for i in range(self.total_frames):
            f = self.reader.read_frame(i).astype(np.float32)
            frames.append(self._process_frame(f))
        video = np.stack(frames, axis=0)

        video_histogram_with_contour(video, bins=100, exclude_zero=True, log=True)

        cx = self.coord_x.get()
        cy = self.coord_y.get()
        if cx != 0 or cy != 0:
            bins, sig, _ = angle_signal_density(video, cx, cy, N_bins=360)
            n_plumes = int(self.num_plumes.get()) if self.num_plumes.get() > 0 else 0
            plume_angles = None
            if n_plumes > 0:
                step = 360.0 / n_plumes
                offset = self.plume_offset.get() % 360.0
                plume_angles = [offset + i * step for i in range(n_plumes)]
                plot_angle_signal_density(bins, sig, log=True, plume_angles=plume_angles)

    def export_mask(self):
        """Save the current mask as .npy and .jpg"""
        if self.total_frames == 0:
            messagebox.showerror('Error', 'No video loaded')
            return
        file_path = filedialog.asksaveasfilename(defaultextension='.npy',
                                                 filetypes=[('NumPy file','*.npy')])
        if not file_path:
            return
        np.save(file_path, self.mask.astype(np.bool_))
        img = Image.fromarray((self.mask * 255).astype(np.uint8))
        img.save(os.path.splitext(file_path)[0] + '.jpg')
        messagebox.showinfo('Export', 'Mask exported')
    
    def save_config(self):
        """Save plume and calibration parameters to a config JSON file."""
        folder = filedialog.askdirectory(title='Select Folder to Save Config')
        if not folder:
            return
        cfg = {
            'plumes': int(self.num_plumes.get()),
            'offset': float(self.plume_offset.get()),
            'centre_x': float(self.coord_x.get()),
            'centre_y': float(self.coord_y.get()),
            'calib_radius': float(self.calib_radius_var.get())
        }
        path = os.path.join(folder, 'config.json')
        try:
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
            messagebox.showinfo('Config', f'Configuration saved to {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Could not save config:\n{e}')

if __name__=='__main__':
    root=tk.Tk(); app=VideoAnnotatorUI(root); root.mainloop()


