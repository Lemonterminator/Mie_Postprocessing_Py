import tkinter as tk
from tkinter import filedialog, messagebox
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from OSCC_postprocessing.analysis.inference_penetration_hetero import (
    load_run as load_hetero_run,
    predict_time_range,
)

FEATURE_LABELS = {
    "tilt_angle_radian": "Tilt Angle (rad)",
    "plumes": "Plumes (count)",
    "diameter_mm": "Diameter (mm)",
    "chamber_pressure": "Chamber Pressure",
    "injection_duration": "Injection Duration (µs)",
}
TIME_LABELS = {
    "time_ms": "Time (ms)",
}


class HeteroPenetrationGUI:
    """Interactive GUI for heteroscedastic penetration inference."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Penetration MLP (Heteroscedastic)")

        self.run = None
        self.feature_entries: dict[str, tk.Entry] = {}
        self.time_feature: str | None = None
        self.time_frame_label = tk.StringVar(value="time")

        self._build_layout()

    def _build_layout(self) -> None:
        row = 0
        tk.Label(
            self.root,
            text="Heteroscedastic Penetration MLP",
            font=("Arial", 12, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 6))
        row += 1

        tk.Label(self.root, text="Run Directory:").grid(row=row, column=0, sticky="e")
        self.run_dir_entry = tk.Entry(self.root, width=42)
        self.run_dir_entry.grid(row=row, column=1, sticky="we", padx=4)
        tk.Button(self.root, text="Browse", command=self._browse_run_dir).grid(row=row, column=2, padx=4)
        row += 1

        tk.Button(self.root, text="Load Run", command=self._load_run).grid(row=row, column=1, sticky="w")
        row += 1

        self.feature_frame = tk.LabelFrame(self.root, text="Feature Inputs")
        self.feature_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=4, pady=(6, 4))
        row += 1

        time_frame = tk.LabelFrame(self.root, text="Time Sweep")
        time_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=4, pady=(4, 6))
        row += 1

        tk.Label(time_frame, text="Start:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.time_start_entry = tk.Entry(time_frame, width=10)
        self.time_start_entry.grid(row=0, column=1, padx=2, pady=2)

        tk.Label(time_frame, text="End:").grid(row=0, column=2, sticky="e", padx=2, pady=2)
        self.time_end_entry = tk.Entry(time_frame, width=10)
        self.time_end_entry.grid(row=0, column=3, padx=2, pady=2)

        tk.Label(time_frame, text="Step:").grid(row=0, column=4, sticky="e", padx=2, pady=2)
        self.time_step_entry = tk.Entry(time_frame, width=10)
        self.time_step_entry.insert(0, "1.0")
        self.time_step_entry.grid(row=0, column=5, padx=2, pady=2)

        self.time_label = tk.Label(time_frame, textvariable=self.time_frame_label)
        self.time_label.grid(row=0, column=6, padx=2, pady=2)

        tk.Button(self.root, text="Plot Prediction", command=self._plot_prediction).grid(row=row, column=1, pady=4)

        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Penetration")
        self.ax.grid(True, alpha=0.4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=row + 2, padx=(12, 4), pady=6)

    def _browse_run_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.run_dir_entry.delete(0, tk.END)
            self.run_dir_entry.insert(0, path)

    def _clear_feature_entries(self) -> None:
        for widget in self.feature_frame.winfo_children():
            widget.destroy()
        self.feature_entries.clear()

    def _resolve_run_dir(self, path_str: str) -> Path:
        """Resolve a run directory path.

        Accept either a direct run folder (contains best_model.pt), or a parent
        directory that contains one or more child run folders with best_model.pt.
        Chooses the most recently modified child if multiple are present.
        """
        base = Path(path_str)
        model_file = base / "best_model.pt"
        if model_file.exists():
            return base

        # Search child directories for best_model.pt
        candidates = []
        if base.exists() and base.is_dir():
            for p in base.iterdir():
                if p.is_dir():
                    f = p / "best_model.pt"
                    if f.exists():
                        try:
                            mtime = f.stat().st_mtime
                        except Exception:
                            mtime = 0.0
                        candidates.append((mtime, p))
        if not candidates:
            raise FileNotFoundError(
                f"Missing model weights at {model_file}. Select a run folder "
                f"that contains best_model.pt, or select a parent folder that "
                f"contains run subfolders."
            )
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    def _load_run(self) -> None:
        try:
            run_dir = self.run_dir_entry.get().strip()
            if not run_dir:
                raise ValueError("Please select a run directory.")
            resolved = self._resolve_run_dir(run_dir)
            self.run = load_hetero_run(str(resolved))
            feature_columns = self.run["feature_columns"]
            time_feature = self.run["time_feature"]
            self.time_feature = time_feature

            self.time_frame_label.set(TIME_LABELS.get(time_feature, time_feature or "index"))
            self._clear_feature_entries()

            row = 0
            for name in feature_columns:
                if name == time_feature:
                    continue
                label = FEATURE_LABELS.get(name, name.replace("_", " "))
                tk.Label(self.feature_frame, text=f"{label}:").grid(row=row, column=0, sticky="e", padx=4, pady=2)
                entry = tk.Entry(self.feature_frame, width=18)
                entry.grid(row=row, column=1, padx=4, pady=2)
                self.feature_entries[name] = entry
                row += 1

            if time_feature:
                self.time_start_entry.delete(0, tk.END)
                self.time_end_entry.delete(0, tk.END)
                self.time_step_entry.delete(0, tk.END)
                self.time_step_entry.insert(0, "1.0")

            if str(resolved) != run_dir:
                messagebox.showinfo("Loaded", f"Loaded latest run: {resolved.name}")
            else:
                messagebox.showinfo("Loaded", f"Loaded run: {resolved.name}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _collect_params(self) -> dict[str, float]:
        params: dict[str, float] = {}
        for name, entry in self.feature_entries.items():
            text = entry.get().strip()
            if text == "":
                raise ValueError(f"Please provide a value for '{name}'.")
            try:
                params[name] = float(text)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value for '{name}': {text}") from exc
        return params

    def _build_time_values(self) -> np.ndarray | None:
        if self.time_feature is None:
            return None
        start_text = self.time_start_entry.get().strip()
        end_text = self.time_end_entry.get().strip()
        step_text = self.time_step_entry.get().strip() or "1.0"
        if not start_text or not end_text:
            raise ValueError("Please provide start and end for the time sweep.")
        start = float(start_text)
        end = float(end_text)
        step = float(step_text)
        if end < start:
            raise ValueError("End must be greater than or equal to start.")
        if step <= 0:
            raise ValueError("Step must be positive.")
        n_steps = int(math.floor((end - start) / step)) + 1
        return (start + step * np.arange(n_steps, dtype=np.float32)).astype(np.float32)

    def _plot_prediction(self) -> None:
        try:
            if self.run is None:
                raise ValueError("Load a run before plotting.")
            params = self._collect_params()
            time_values = self._build_time_values()

            result = predict_time_range(
                self.run,
                params,
                time_values=time_values,
            )

            time_axis = result["time"]
            if time_axis is None:
                time_axis = np.arange(len(result["mean"]), dtype=np.float32)
                x_label = "Index"
            else:
                x_label = TIME_LABELS.get(self.time_feature, self.time_feature or "Time")

            mean = result["mean"]
            std = result["std"]

            self.ax.clear()
            self.ax.plot(time_axis, mean, color="#1f77b4", label="Mean", linewidth=2)
            self.ax.fill_between(
                time_axis,
                mean - std,
                mean + std,
                color="#1f77b4",
                alpha=0.25,
                label="±1 Std Dev",
            )
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel("Penetration (pixels)")
            self.ax.grid(True, alpha=0.4)
            self.ax.legend()
            self.canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = HeteroPenetrationGUI(root)
    root.mainloop()

