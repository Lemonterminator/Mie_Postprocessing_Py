import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox


def make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'")


class PenetrationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    make_activation(activation),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage1InferenceGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Stage-1 Median Penetration Inference")

        self.model: PenetrationMLP | None = None
        self.scaler_state: dict | None = None
        self.train_config: dict | None = None
        self.feature_columns: list[str] = []
        self.time_feature = "time_norm_0_5ms"

        self.param_entries: dict[str, tk.Entry] = {}
        self._build_layout()

    def _build_layout(self) -> None:
        row = 0
        tk.Label(
            self.root,
            text="Stage-1 Penetration Model (manual inputs + time sweep)",
            font=("Arial", 11, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 6))
        row += 1

        tk.Label(self.root, text="Run Directory:").grid(row=row, column=0, sticky="e")
        self.run_dir_entry = tk.Entry(self.root, width=46)
        self.run_dir_entry.grid(row=row, column=1, sticky="we", padx=4)
        tk.Button(self.root, text="Browse", command=self._browse_run_dir).grid(row=row, column=2, padx=4)
        row += 1

        tk.Button(self.root, text="Load Run", command=self._load_run).grid(row=row, column=1, sticky="w", pady=(2, 6))
        row += 1

        self.param_frame = tk.LabelFrame(self.root, text="Manual Physical Inputs")
        self.param_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=4, pady=(2, 6))
        row += 1

        fields = [
            ("tilt_angle_radian", "Tilt Angle (rad)", "0.349066"),
            ("plumes", "Plumes", "10"),
            ("diameter_mm", "Diameter (mm)", "0.355"),
            ("injection_duration_us", "Injection Duration (us)", "800"),
            ("injection_pressure_bar", "Injection Pressure (bar)", "2000"),
            ("chamber_pressure_bar", "Chamber Pressure (bar)", "5"),
            ("control_backpressure_bar", "Control Backpressure (bar)", "4"),
        ]
        for i, (key, label, default_value) in enumerate(fields):
            tk.Label(self.param_frame, text=f"{label}:").grid(row=i, column=0, sticky="e", padx=4, pady=2)
            entry = tk.Entry(self.param_frame, width=14)
            entry.insert(0, default_value)
            entry.grid(row=i, column=1, sticky="w", padx=4, pady=2)
            self.param_entries[key] = entry

        sweep_frame = tk.LabelFrame(self.root, text="Time Sweep (ms)")
        sweep_frame.grid(row=row, column=0, columnspan=3, sticky="we", padx=4, pady=(2, 6))
        row += 1

        tk.Label(sweep_frame, text="Start:").grid(row=0, column=0, sticky="e", padx=3, pady=3)
        self.time_start_entry = tk.Entry(sweep_frame, width=10)
        self.time_start_entry.insert(0, "0.0")
        self.time_start_entry.grid(row=0, column=1, padx=3, pady=3)

        tk.Label(sweep_frame, text="End:").grid(row=0, column=2, sticky="e", padx=3, pady=3)
        self.time_end_entry = tk.Entry(sweep_frame, width=10)
        self.time_end_entry.insert(0, "5.0")
        self.time_end_entry.grid(row=0, column=3, padx=3, pady=3)

        tk.Label(sweep_frame, text="Step:").grid(row=0, column=4, sticky="e", padx=3, pady=3)
        self.time_step_entry = tk.Entry(sweep_frame, width=10)
        self.time_step_entry.insert(0, "0.01")
        self.time_step_entry.grid(row=0, column=5, padx=3, pady=3)

        tk.Button(self.root, text="Plot Prediction", command=self._plot_prediction).grid(
            row=row, column=1, sticky="w", pady=(2, 4)
        )

        self.fig, self.ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
        self.ax.set_xlabel("Time (ms)")
        self.ax.set_ylabel("Predicted Penetration")
        self.ax.grid(True, alpha=0.35)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=3, rowspan=row + 2, padx=(10, 4), pady=6)

    def _browse_run_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.run_dir_entry.delete(0, tk.END)
            self.run_dir_entry.insert(0, path)

    def _resolve_run_dir(self, path_str: str) -> Path:
        base = Path(path_str)
        if (base / "best_model_stage1.pt").exists():
            return base

        candidates: list[tuple[float, Path]] = []
        if base.exists() and base.is_dir():
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                model_path = child / "best_model_stage1.pt"
                if model_path.exists():
                    try:
                        mtime = model_path.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    candidates.append((mtime, child))
        if not candidates:
            raise FileNotFoundError(
                "Could not find 'best_model_stage1.pt'. Select a stage1_median_penetration_* run folder "
                "or its parent directory."
            )
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    def _load_run(self) -> None:
        try:
            run_dir_raw = self.run_dir_entry.get().strip()
            if not run_dir_raw:
                raise ValueError("Please select a run directory first.")

            run_dir = self._resolve_run_dir(run_dir_raw)
            model_path = run_dir / "best_model_stage1.pt"
            scaler_path = run_dir / "scaler_state.json"
            config_path = run_dir / "train_config_used.json"

            if not scaler_path.exists():
                raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config file: {config_path}")

            with open(scaler_path, "r", encoding="utf-8") as f:
                self.scaler_state = json.load(f)
            with open(config_path, "r", encoding="utf-8") as f:
                self.train_config = json.load(f)

            self.feature_columns = list(self.train_config["feature_columns"])
            self.time_feature = str(self.train_config.get("time_feature", "time_norm_0_5ms"))

            model = PenetrationMLP(
                input_dim=int(self.train_config["input_dim"]),
                hidden_dims=[int(x) for x in self.train_config["hidden_dims"]],
                output_dim=int(self.train_config["output_dim"]),
                activation=str(self.train_config.get("activation", "relu")),
                dropout=float(self.train_config.get("dropout", 0.0)),
            )
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.model = model

            if str(run_dir) != run_dir_raw:
                messagebox.showinfo("Loaded", f"Loaded latest run: {run_dir}")
            else:
                messagebox.showinfo("Loaded", f"Loaded run: {run_dir}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _collect_raw_params(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for key, entry in self.param_entries.items():
            txt = entry.get().strip()
            if txt == "":
                raise ValueError(f"Please provide a value for '{key}'.")
            try:
                values[key] = float(txt)
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value for '{key}': {txt}") from exc
        return values

    def _build_time_ms(self) -> np.ndarray:
        start = float(self.time_start_entry.get().strip())
        end = float(self.time_end_entry.get().strip())
        step = float(self.time_step_entry.get().strip())
        if end < start:
            raise ValueError("End time must be >= start time.")
        if step <= 0:
            raise ValueError("Step must be > 0.")
        n_steps = int(math.floor((end - start) / step)) + 1
        return (start + step * np.arange(n_steps, dtype=np.float32)).astype(np.float32)

    def _zscore(self, value: float, z_col: str) -> float:
        if self.scaler_state is None:
            raise ValueError("Scaler state is not loaded.")
        stats = self.scaler_state["zscore"][z_col]
        return (float(value) - float(stats["mean"])) / (float(stats["std"]) + 1e-12)

    def _build_feature_matrix(self, raw: dict[str, float], time_ms: np.ndarray) -> np.ndarray:
        if self.scaler_state is None:
            raise ValueError("Scaler state is not loaded.")

        time_min = float(self.scaler_state["time"]["min_ms"])
        time_max = float(self.scaler_state["time"]["max_ms"])
        time_norm = (np.clip(time_ms, time_min, time_max) - time_min) / max(time_max - time_min, 1e-12)

        p_inj = float(raw["injection_pressure_bar"])
        p_ch = float(raw["chamber_pressure_bar"])
        delta_p = max(p_inj - p_ch, 1e-6)

        feature_series: dict[str, np.ndarray] = {
            self.time_feature: time_norm.astype(np.float32),
            "tilt_angle_radian_z": np.full_like(time_norm, self._zscore(raw["tilt_angle_radian"], "tilt_angle_radian_z"), dtype=np.float32),
            "plumes_z": np.full_like(time_norm, self._zscore(raw["plumes"], "plumes_z"), dtype=np.float32),
            "diameter_mm_z": np.full_like(time_norm, self._zscore(raw["diameter_mm"], "diameter_mm_z"), dtype=np.float32),
            "injection_duration_us_z": np.full_like(
                time_norm, self._zscore(raw["injection_duration_us"], "injection_duration_us_z"), dtype=np.float32
            ),
            "log_injection_pressure_bar_z": np.full_like(
                time_norm, self._zscore(np.log(p_inj), "log_injection_pressure_bar_z"), dtype=np.float32
            ),
            "log_chamber_pressure_bar_z": np.full_like(
                time_norm, self._zscore(np.log(max(p_ch, 1e-6)), "log_chamber_pressure_bar_z"), dtype=np.float32
            ),
            "log_delta_pressure_bar_z": np.full_like(
                time_norm, self._zscore(np.log(delta_p), "log_delta_pressure_bar_z"), dtype=np.float32
            ),
            "control_backpressure_bar_z": np.full_like(
                time_norm, self._zscore(raw["control_backpressure_bar"], "control_backpressure_bar_z"), dtype=np.float32
            ),
        }

        cols: list[np.ndarray] = []
        for name in self.feature_columns:
            if name not in feature_series:
                raise KeyError(f"Unsupported feature column in config: {name}")
            cols.append(feature_series[name])
        return np.column_stack(cols).astype(np.float32)

    def _predict(self, raw: dict[str, float], time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model is not loaded. Click 'Load Run' first.")
        X = self._build_feature_matrix(raw, time_ms)
        with torch.no_grad():
            out = self.model(torch.from_numpy(X))
            mu = out[:, 0].cpu().numpy()
            log_var = out[:, 1].cpu().numpy()

        log_var = np.clip(log_var, -20.0, 20.0)
        std = np.exp(0.5 * log_var)
        std_floor = float(self.train_config.get("std_clamp_min", 1e-3)) if self.train_config else 1e-3
        std = np.maximum(std, std_floor)
        return mu, std

    def _plot_prediction(self) -> None:
        try:
            if self.model is None or self.scaler_state is None or self.train_config is None:
                raise ValueError("Please load a run first.")
            raw = self._collect_raw_params()
            time_ms = self._build_time_ms()
            mu, std = self._predict(raw, time_ms)

            self.ax.clear()
            self.ax.plot(time_ms, mu, color="#1f77b4", linewidth=2, label="Mean")
            self.ax.fill_between(time_ms, mu - std, mu + std, color="#1f77b4", alpha=0.25, label="±1 Std")
            self.ax.set_title("Stage-1 Penetration Inference")
            self.ax.set_xlabel("Time (ms)")
            self.ax.set_ylabel("Predicted Penetration")
            self.ax.grid(True, alpha=0.35)
            self.ax.legend()
            self.canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = Stage1InferenceGUI(root)
    root.mainloop()
