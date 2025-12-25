import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Penetration inference helpers
from OSCC_postprocessing.analysis.inference_penetration import (
    load_run as load_penetration_run,
    frames_to_time as frames_to_time_pen,
    predict_time_range as predict_time_range_pen,
)

global scale
scale = 0.21345

class InferenceApp:
    """GUI for penetration MLP inference using physical inputs.

    Overwrites the old 3-output GUI. This app loads a saved run directory
    (best_model.pt, scalers.json, model_config.json), accepts physical inputs,
    and plots penetration mean ± std over a chosen time/frames range.
    """

    def __init__(self, root):
        self.root = root
        root.title("Penetration MLP Inference")

        # Section header
        row = 0
        tk.Label(root, text="Penetration MLP (mean ± std)", font=("Arial", 11, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 4))
        row += 1

        # Run directory chooser (expects best_model.pt, scalers.json, model_config.json)
        tk.Label(root, text="Run Directory:").grid(row=row, column=0, sticky="e")
        self.pen_run_entry = tk.Entry(root, width=40)
        self.pen_run_entry.grid(row=row, column=1)
        tk.Button(root, text="Browse", command=self.browse_pen_run).grid(row=row, column=2)
        row += 1

        # Physical parameter inputs
        self.pen_entries = {}
        pen_params = [
            ("chamber_pressure", "Chamber Pressure"),
            ("injection_pressure", "Injection Pressure"),
            ("injection_duration", "Injection Duration (µs)"),
            ("control_backpressure", "Control Backpressure"),
        ]
        for key, label in pen_params:
            tk.Label(root, text=label + ":").grid(row=row, column=0, sticky="e")
            e = tk.Entry(root)
            e.grid(row=row, column=1)
            self.pen_entries[key] = e
            row += 1

        # Domain selection: frames vs seconds
        tk.Label(root, text="Domain:").grid(row=row, column=0, sticky="e")
        self.pen_domain_var = tk.StringVar(value="frames")
        tk.Radiobutton(root, text="Frames", variable=self.pen_domain_var, value="frames").grid(row=row, column=1, sticky="w")
        tk.Radiobutton(root, text="Seconds", variable=self.pen_domain_var, value="seconds").grid(row=row, column=1, sticky="e")
        row += 1

        # Start/End inputs (frame index or seconds depending on domain)
        tk.Label(root, text="Start:").grid(row=row, column=0, sticky="e")
        self.pen_start_entry = tk.Entry(root)
        self.pen_start_entry.grid(row=row, column=1, sticky="w")
        row += 1
        tk.Label(root, text="End:").grid(row=row, column=0, sticky="e")
        self.pen_end_entry = tk.Entry(root)
        self.pen_end_entry.grid(row=row, column=1, sticky="w")
        row += 1

        # Output space: corrected vs projected
        tk.Label(root, text="Output Space:").grid(row=row, column=0, sticky="e")
        self.pen_space_var = tk.StringVar(value="corrected")
        tk.OptionMenu(root, self.pen_space_var, "corrected", "projected").grid(row=row, column=1, sticky="w")
        row += 1

        tk.Button(root, text="Plot Penetration", command=self.plot_penetration).grid(row=row, column=1, pady=(6, 0))

        # Single-axis figure for penetration
        self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 4))
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Penetration")
        self.ax.grid(True, alpha=0.4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=20, padx=(10, 0))

    def browse_pen_run(self):
        path = filedialog.askdirectory()
        if path:
            self.pen_run_entry.delete(0, tk.END)
            self.pen_run_entry.insert(0, path)

    def _ensure_pen_run_loaded(self):
        if not hasattr(self, 'pen_run'):
            run_dir = self.pen_run_entry.get()
            if not run_dir:
                raise ValueError("Please select a Run Directory (contains best_model.pt, scalers.json, model_config.json)")
            self.pen_run = load_penetration_run(run_dir)

    def plot_penetration(self):
        try:
            # Load run/config
            self._ensure_pen_run_loaded()
            fr = float(self.pen_run['cfg'].get('frame_rate_hz', 34_000.0))

            # Read parameters
            params = {k: float(self.pen_entries[k].get()) for k in self.pen_entries}

            # Build time array based on domain
            domain = self.pen_domain_var.get()
            if domain == 'frames':
                start = int(self.pen_start_entry.get())
                end = int(self.pen_end_entry.get())
                if end < start:
                    raise ValueError("End must be >= Start")
                time_s = frames_to_time_pen(start, end, fr)
                x_vals = np.arange(start, end + 1)
                x_label = "Frame"
            else:
                start_s = float(self.pen_start_entry.get())
                end_s = float(self.pen_end_entry.get())
                if end_s < start_s:
                    raise ValueError("End time must be >= Start time")
                # Snap to frame grid
                start_idx = int(np.ceil(start_s * fr))
                end_idx = int(np.floor(end_s * fr))
                if end_idx < start_idx:
                    raise ValueError("Time window too narrow for the frame rate")
                time_s = frames_to_time_pen(start_idx, end_idx, fr)
                x_vals = time_s
                x_label = "Time (s)"

            # Predict
            out = predict_time_range_pen(self.pen_run, params, time_s=time_s, output_space=self.pen_space_var.get())
            
            # Plot mean ± std
            self.ax.clear()
            self.ax.plot(x_vals, scale*out['mean'], label='mean', lw=2)
            self.ax.fill_between(x_vals, scale*(out['mean'] - out['std']), scale*(out['mean'] + out['std']), alpha=0.25, label='±1 std')
            title_space = 'Corrected' if self.pen_space_var.get() == 'corrected' else 'Projected'
            self.ax.set_title(f"Penetration ({title_space})")
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel("Penetration")
            self.ax.grid(True, alpha=0.4)
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


