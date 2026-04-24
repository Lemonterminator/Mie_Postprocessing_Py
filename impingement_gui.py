from __future__ import annotations

import copy
import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
for search_path in (PROJECT_ROOT, PROJECT_ROOT / "MLP", PROJECT_ROOT / "MLP" / "v2_engineered_feature"):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from MLP.impingement_core import (  # noqa: E402
    NotebookDefaults,
    WizardState,
    build_default_state,
    build_summary_figure,
    run_impingement_case,
    serialize_wizard_state,
    validate_wizard_state,
)

# ── ALL DEFAULT PARAMETERS ───────────────────────────────────────────────────
# Edit this top section to change the no-input/default GUI run. Values match
# MLP/impingement_with_piston_prototype.ipynb.

# ─── Model artifacts ─────────────────────────────────────────────────────────
RUN_DIR = PROJECT_ROOT / "MLP" / "runs_mlp" / "distill_cdf_onset_v2_20260410_103413"

# ─── Mesh fineness + canvas ──────────────────────────────────────────────────
DEFAULT_SOLVER = {
    "grid_size": 0.25,
    "right_padding": 5.0,
    "canvas_h_mm": 200.0,
}

# ─── Time grid (model evaluation + animation) ────────────────────────────────
DEFAULT_TIME_GRID = {
    "n_frames": 60,
    "t_start_ms": 0.0,
    "t_end_ms": 5.0,
}

# ─── Spray inputs (neural-network features + cone) ───────────────────────────
DEFAULT_NN_CONDITIONS = {
    "tilt_angle_radian": float(np.deg2rad(20.0)),
    "plumes": 10.0,
    "diameter_mm": 0.34,
    "injection_duration_us": 800.0,
    "injection_pressure_bar": 2000.0,
    "ambient_pressure_bar_phys": 15.0,
    "control_backpressure_bar": 4.0,
    "cone_angle_deg": 20.0,
    "injector_x_mm": 0.0,
    "injector_y_mm": 0.0,
}

# ─── Piston motion (slider-crank approximation) ──────────────────────────────
DEFAULT_PISTON_MOTION = {
    "RPM": 250.0,
    "stroke_half": 55.0,
    "offset_baseline": 75.0,
    "crank_phase_deg": 0.0,
}

# ─── Piston Design A — bowl piston via concentric radii ──────────────────────
DEFAULT_PISTON_A_PARAMS = {
    "r_bore": 120.0,
    "r_outer_bowl": 100.0,
    "r_topland": 5.0,
    "r_ring": 15.0,
    "r_inner_bowl": 80.0,
    "r_lip": 15.0,
    "r_floor": 5.0,
    "r_center_circle": 0.0,
    "h_vertical": 10.0,
    "piston_height": 30.0,
}

# ─── Piston Design Hermite — quintic Hermite crown spline ────────────────────
DEFAULT_HERMITE_PARAMS = {
    "r_bore": 120.0,
    "r_topland": 5.0,
    "piston_height": 30.0,
}
DEFAULT_HERMITE_CTRL_PTS_MM = (
    (0.0, 3.0),
    (55.0, 14.0),
    (100.0, 2.0),
    (115.0, 0.0),
)
DEFAULT_HERMITE_CTRL_VELS = (
    (33.0, 0.0),
    (28.0, 0.0),
    (9.0, -3.0),
    (4.5, -0.5),
)

# ─── Piston Design B — tangent two-arc bowl ──────────────────────────────────
DEFAULT_PISTON_B_PARAMS = {
    "r_bore": 125.0,
    "piston_height": 30.0,
    "flat_depth_mm": 5.0,
    "x_flat_end_mm": 15.0,
    "x_arc1_arc2_join_mm": 35.0,
    "arc1_radius_mm": 63.98,
    "x_arc2_lip_join_mm": 100.0,
    "arc2_radius_mm": 78.38,
    "lip_radius_mm": 6.0,
    "x_lip_flat_join_mm": 103.1,
    "x_topland_start_mm": 115.0,
    "r_topland": 10.0,
}

NOTEBOOK_DEFAULTS = NotebookDefaults(
    run_dir=RUN_DIR,
    solver=DEFAULT_SOLVER,
    time_grid=DEFAULT_TIME_GRID,
    nn_conditions=DEFAULT_NN_CONDITIONS,
    piston_motion=DEFAULT_PISTON_MOTION,
    piston_a_params=DEFAULT_PISTON_A_PARAMS,
    piston_b_params=DEFAULT_PISTON_B_PARAMS,
    hermite_params=DEFAULT_HERMITE_PARAMS,
    hermite_ctrl_pts_mm=DEFAULT_HERMITE_CTRL_PTS_MM,
    hermite_ctrl_vels=DEFAULT_HERMITE_CTRL_VELS,
    selected_design="A",
)


class ImpingementWizardGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Spray Impingement GUI")
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)

        self.default_state = build_default_state(NOTEBOOK_DEFAULTS)
        self.state = copy.deepcopy(self.default_state)
        self.current_step = 0
        self.entries: dict[str, tk.Entry] = {}
        self.hermite_entries: dict[str, tk.Entry] = {}
        self.result_canvas: FigureCanvasTkAgg | None = None
        self.result_figure: plt.Figure | None = None
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.is_running = False

        self.style = ttk.Style()
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        self._build_shell()
        self._show_step(0)
        self._poll_worker_queue()

    def _build_shell(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(16, 14, 16, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        title = ttk.Label(header, text="Spray Impingement on Moving Piston", font=("Segoe UI", 17, "bold"))
        title.grid(row=0, column=0, sticky="w")
        self.step_label = ttk.Label(header, text="", font=("Segoe UI", 10))
        self.step_label.grid(row=0, column=1, sticky="e")

        self.body = ttk.Frame(self.root, padding=(16, 8, 16, 8))
        self.body.grid(row=1, column=0, sticky="nsew")
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)

        footer = ttk.Frame(self.root, padding=(16, 8, 16, 16))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(2, weight=1)
        self.back_button = ttk.Button(footer, text="Back", command=self._back)
        self.back_button.grid(row=0, column=0, padx=(0, 8))
        self.next_button = ttk.Button(footer, text="Next", command=self._next, style="Accent.TButton")
        self.next_button.grid(row=0, column=1, padx=(0, 8))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(footer, textvariable=self.status_var, anchor="w").grid(row=0, column=2, sticky="ew")

    def _clear_body(self) -> None:
        if self.result_canvas is not None:
            self.result_canvas = None
        if self.result_figure is not None:
            plt.close(self.result_figure)
            self.result_figure = None
        for child in self.body.winfo_children():
            child.destroy()
        self.entries.clear()
        self.hermite_entries.clear()

    def _show_step(self, step: int) -> None:
        if self.is_running:
            return
        self.current_step = step
        self._clear_body()
        builders = [
            self._build_start_page,
            self._build_design_page,
            self._build_design_parameters_page,
            self._build_nn_conditions_page,
            self._build_solver_page,
            self._build_piston_motion_page,
            self._build_review_page,
            self._build_results_page,
        ]
        builders[step]()
        total_steps = 6
        if step == 0:
            self.step_label.config(text="Start")
        elif 1 <= step <= 6:
            self.step_label.config(text=f"Step {step} / {total_steps}")
        else:
            self.step_label.config(text="Results")

        self.back_button.config(state=tk.DISABLED if step == 0 else tk.NORMAL)
        if step == 0:
            self.next_button.config(text="Customize", command=lambda: self._show_step(1))
        elif step == 6:
            self.next_button.config(text="Run", command=self._run_from_review)
        elif step == 7:
            self.next_button.config(text="New Run", command=self._reset_to_defaults)
        else:
            self.next_button.config(text="Next", command=self._next)

    def _build_start_page(self) -> None:
        frame = ttk.Frame(self.body, padding=24)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        ttk.Label(frame, text="Run with Defaults or Customize Step by Step", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        intro = (
            "Default values match the notebook's ALL TUNABLE PARAMETERS section.\n"
            "You can run immediately with defaults, or customize the design, operating conditions, solver, and piston motion page by page."
        )
        ttk.Label(frame, text=intro, font=("Segoe UI", 11), justify="left").grid(row=1, column=0, sticky="w", pady=(8, 18))

        run_frame = ttk.LabelFrame(frame, text="Model artifacts", padding=12)
        run_frame.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        run_frame.columnconfigure(1, weight=1)
        ttk.Label(run_frame, text="RUN_DIR").grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.run_dir_var = tk.StringVar(value=str(self.state.run_dir))
        run_entry = ttk.Entry(run_frame, textvariable=self.run_dir_var)
        run_entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(run_frame, text="Browse", command=self._browse_run_dir).grid(row=0, column=2, padx=(8, 0))

        button_row = ttk.Frame(frame)
        button_row.grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Button(button_row, text="Run with Defaults", command=self._run_defaults, style="Accent.TButton").grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_row, text="Customize", command=lambda: self._show_step(1)).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_row, text="Reset Defaults", command=self._reset_to_defaults).grid(row=0, column=2)

        notes = (
            "Output location: outputs/impingement_gui/<timestamp>/\n"
            "If Manim is unavailable, the GUI will still save the numeric summary, preview image, and cache."
        )
        ttk.Label(frame, text=notes, foreground="#475569", justify="left").grid(row=4, column=0, sticky="w", pady=(22, 0))

    def _build_design_page(self) -> None:
        frame = ttk.Frame(self.body, padding=24)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text="Step 1: Select Piston Design", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.design_var = tk.StringVar(value=self.state.selected_design)
        options = [
            ("Design A", "A", "Concentric-radius bowl piston, with explicit radius/height constraints."),
            ("Design B", "B", "Tangent two-arc bowl, with segment breakpoint and tangency validation."),
            ("Hermite", "Hermite", "Quintic Hermite crown spline with four control points and four velocity vectors."),
        ]
        for row, (label, value, description) in enumerate(options, start=1):
            item = ttk.Frame(frame, padding=(0, 12, 0, 0))
            item.grid(row=row, column=0, sticky="w")
            ttk.Radiobutton(item, text=label, value=value, variable=self.design_var).grid(row=0, column=0, sticky="w")
            ttk.Label(item, text=description, foreground="#475569").grid(row=1, column=0, sticky="w", padx=(24, 0))

    def _build_design_parameters_page(self) -> None:
        frame = ttk.Frame(self.body, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        title = f"Step 2: {self.state.selected_design} design parameters"
        ttk.Label(frame, text=title, font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))

        if self.state.selected_design == "A":
            self._build_numeric_form(frame, self.state.piston_a_params, "piston_a", columns=2)
            hint = "Design A constraints: outer bowl + topland + ring = bore; inner bowl + lip + floor = outer bowl; vertical + lip + floor = piston height."
        elif self.state.selected_design == "B":
            self._build_numeric_form(frame, self.state.piston_b_params, "piston_b", columns=2)
            hint = "Design B validates segment order, bore matching, and tangent arc conditions."
        else:
            top = ttk.LabelFrame(frame, text="Hermite scalar parameters", padding=10)
            top.grid(row=1, column=0, sticky="ew", pady=(0, 12))
            self._build_numeric_form(top, self.state.hermite_params, "hermite", columns=3)
            table = ttk.LabelFrame(frame, text="Hermite control points and velocities", padding=10)
            table.grid(row=2, column=0, sticky="w")
            headers = ["ctrl_x", "ctrl_depth", "vel_x", "vel_depth"]
            for col, header in enumerate(headers):
                ttk.Label(table, text=header, font=("Segoe UI", 9, "bold")).grid(row=0, column=col + 1, padx=4, pady=3)
            for row_index in range(4):
                ttk.Label(table, text=f"P{row_index + 1}").grid(row=row_index + 1, column=0, sticky="e", padx=4, pady=3)
                values = [
                    self.state.hermite_ctrl_pts_mm[row_index][0],
                    self.state.hermite_ctrl_pts_mm[row_index][1],
                    self.state.hermite_ctrl_vels[row_index][0],
                    self.state.hermite_ctrl_vels[row_index][1],
                ]
                for col_index, value in enumerate(values):
                    key = f"hermite_table.{row_index}.{col_index}"
                    entry = ttk.Entry(table, width=12)
                    entry.insert(0, _format_float(value))
                    entry.grid(row=row_index + 1, column=col_index + 1, padx=4, pady=3)
                    self.hermite_entries[key] = entry
            hint = "Hermite v1 uses four fixed control points and four velocity vectors; ctrl_accels is fixed internally at 0."
        ttk.Label(frame, text=hint, foreground="#475569", wraplength=920).grid(row=99, column=0, sticky="w", pady=(16, 0))

    def _build_nn_conditions_page(self) -> None:
        frame = ttk.Frame(self.body, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text="Step 3: NN Operating Conditions and Cone", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self._build_numeric_form(frame, self.state.nn_conditions, "nn", columns=2)
        ttk.Label(
            frame,
            text="tilt_angle_radian is in radians. For 20 degrees, keep the default value 0.349066.",
            foreground="#475569",
        ).grid(row=20, column=0, sticky="w", pady=(12, 0))

    def _build_solver_page(self) -> None:
        frame = ttk.Frame(self.body, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text="Step 4: Cylinder + Solver", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        solver_frame = ttk.LabelFrame(frame, text="Mesh / canvas", padding=10)
        solver_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self._build_numeric_form(solver_frame, self.state.solver, "solver", columns=3)
        time_frame = ttk.LabelFrame(frame, text="Time grid", padding=10)
        time_frame.grid(row=2, column=0, sticky="ew")
        self._build_numeric_form(time_frame, self.state.time_grid, "time", columns=3)
        ttk.Label(
            frame,
            text="Smaller grid_size gives finer PDF and animation detail, but simulation and Manim rendering will be slower.",
            foreground="#475569",
        ).grid(row=3, column=0, sticky="w", pady=(12, 0))

    def _build_piston_motion_page(self) -> None:
        frame = ttk.Frame(self.body, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(frame, text="Step 5: Piston motion", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self._build_numeric_form(frame, self.state.piston_motion, "motion", columns=2)
        explanation = (
            "piston_offset(t) = offset_baseline - stroke_half * cos(omega*t + phase).\n"
            "The GUI blocks cases where the crown rises above the cylinder head or the piston bottom exceeds the canvas."
        )
        ttk.Label(frame, text=explanation, foreground="#475569", justify="left").grid(row=20, column=0, sticky="w", pady=(14, 0))

    def _build_review_page(self) -> None:
        frame = ttk.Frame(self.body, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        ttk.Label(frame, text="Step 6: Review + Run", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(frame)
        actions.grid(row=1, column=0, sticky="ew", pady=(10, 8))
        ttk.Button(actions, text="Validate", command=self._validate_current_state).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Reset to Notebook Defaults", command=self._reset_to_defaults).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(actions, text="Run", command=self._run_from_review, style="Accent.TButton").grid(row=0, column=2)

        text = tk.Text(frame, wrap="word", height=26, font=("Consolas", 10))
        text.grid(row=2, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(frame, command=text.yview)
        scrollbar.grid(row=2, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        text.insert("1.0", self._review_text())
        text.configure(state="disabled")

    def _build_results_page(self) -> None:
        frame = ttk.Frame(self.body, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="Results", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        if not hasattr(self, "last_result"):
            ttk.Label(frame, text="No result yet.").grid(row=1, column=0, sticky="w", pady=(8, 0))
            return

        result = self.last_result
        metrics_text = (
            f"Output: {result.output_dir}\n"
            f"Peak P(piston): {result.metrics['peak_piston_impact']:.4e} at {result.metrics['peak_piston_time_ms']:.3f} ms\n"
            f"Integral ∫P dt: {result.metrics['cumulative_piston_impact_ms']:.4e} ms\n"
            f"Peak P(wall): {result.metrics['peak_wall_impact']:.4e}\n"
            f"Video: {result.video_path if result.video_path else 'Skipped / unavailable'}"
        )
        ttk.Label(frame, text=metrics_text, justify="left").grid(row=1, column=0, sticky="w", pady=(8, 10))

        if result.warnings:
            ttk.Label(frame, text="\n".join(result.warnings), foreground="#b45309", wraplength=980).grid(row=2, column=0, sticky="ew", pady=(0, 8))
            plot_row = 3
        else:
            plot_row = 2

        if self.result_canvas is not None:
            self.result_canvas.get_tk_widget().destroy()
            self.result_canvas = None
        if self.result_figure is not None:
            plt.close(self.result_figure)
        self.result_figure = build_summary_figure(result)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, master=frame)
        self.result_canvas.get_tk_widget().grid(row=plot_row, column=0, sticky="nsew")
        frame.rowconfigure(plot_row, weight=1)
        self.result_canvas.draw()

    def _build_numeric_form(self, parent: ttk.Frame, values: dict[str, float], prefix: str, *, columns: int) -> None:
        for index, (key, value) in enumerate(values.items()):
            row = index // columns
            col = (index % columns) * 2
            ttk.Label(parent, text=key).grid(row=row, column=col, sticky="e", padx=(6, 6), pady=5)
            entry = ttk.Entry(parent, width=18)
            entry.insert(0, _format_float(value))
            entry.grid(row=row, column=col + 1, sticky="w", padx=(0, 16), pady=5)
            self.entries[f"{prefix}.{key}"] = entry

    def _browse_run_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=str(PROJECT_ROOT / "MLP" / "runs_mlp"))
        if path:
            self.run_dir_var.set(path)
            self.state.run_dir = Path(path)

    def _next(self) -> None:
        if not self._save_current_step():
            return
        self._show_step(min(self.current_step + 1, 7))

    def _back(self) -> None:
        if self.current_step in {1, 2, 3, 4, 5, 6}:
            self._save_current_step(show_errors=False)
        self._show_step(max(self.current_step - 1, 0))

    def _save_current_step(self, *, show_errors: bool = True) -> bool:
        try:
            if self.current_step == 0 and hasattr(self, "run_dir_var"):
                self.state.run_dir = Path(self.run_dir_var.get().strip())
            elif self.current_step == 1:
                self.state.selected_design = self.design_var.get()
            elif self.current_step == 2:
                if self.state.selected_design == "A":
                    self.state.piston_a_params = self._collect_entries("piston_a")
                elif self.state.selected_design == "B":
                    self.state.piston_b_params = self._collect_entries("piston_b")
                else:
                    self.state.hermite_params = self._collect_entries("hermite")
                    pts: list[tuple[float, float]] = []
                    vels: list[tuple[float, float]] = []
                    for row_index in range(4):
                        vals = [
                            self._parse_entry(self.hermite_entries[f"hermite_table.{row_index}.{col_index}"], f"Hermite row {row_index + 1}")
                            for col_index in range(4)
                        ]
                        pts.append((vals[0], vals[1]))
                        vels.append((vals[2], vals[3]))
                    self.state.hermite_ctrl_pts_mm = pts
                    self.state.hermite_ctrl_vels = vels
            elif self.current_step == 3:
                self.state.nn_conditions = self._collect_entries("nn")
            elif self.current_step == 4:
                self.state.solver = self._collect_entries("solver")
                self.state.time_grid = self._collect_entries("time")
            elif self.current_step == 5:
                self.state.piston_motion = self._collect_entries("motion")
            return True
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Invalid input", str(exc))
            return False

    def _collect_entries(self, prefix: str) -> dict[str, float]:
        values: dict[str, float] = {}
        prefix_dot = f"{prefix}."
        for full_key, entry in self.entries.items():
            if not full_key.startswith(prefix_dot):
                continue
            key = full_key[len(prefix_dot):]
            values[key] = self._parse_entry(entry, key)
        return values

    def _parse_entry(self, entry: tk.Entry, label: str) -> float:
        text = entry.get().strip()
        if not text:
            raise ValueError(f"{label} cannot be empty.")
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value for {label}: {text}") from exc

    def _review_text(self) -> str:
        state_payload = serialize_wizard_state(self.state)
        default_payload = serialize_wizard_state(self.default_state)
        lines = [
            "Selected configuration",
            "=" * 72,
            f"Design: {self.state.selected_design}",
            f"RUN_DIR: {self.state.run_dir}",
            "",
            "Changed values from notebook defaults",
            "=" * 72,
        ]
        differences = list(_diff_payload(state_payload, default_payload))
        if not differences:
            lines.append("No differences. This run uses notebook defaults.")
        else:
            lines.extend(differences)
        lines.extend(["", "Full configuration", "=" * 72])
        lines.extend(_format_payload_lines(state_payload))
        return "\n".join(lines)

    def _validate_current_state(self) -> None:
        try:
            validate_wizard_state(self.state)
        except Exception as exc:
            messagebox.showerror("Validation failed", str(exc))
            return
        messagebox.showinfo("Validation", "Configuration is valid.")

    def _run_defaults(self) -> None:
        selected_run_dir = Path(self.run_dir_var.get().strip()) if hasattr(self, "run_dir_var") else self.state.run_dir
        self.state = copy.deepcopy(self.default_state)
        self.state.run_dir = selected_run_dir
        if hasattr(self, "run_dir_var"):
            self.run_dir_var.set(str(selected_run_dir))
        self._start_run()

    def _run_from_review(self) -> None:
        if not self._save_current_step():
            return
        self._start_run()

    def _start_run(self) -> None:
        try:
            validate_wizard_state(self.state)
        except Exception as exc:
            messagebox.showerror("Validation failed", str(exc))
            return
        self.is_running = True
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.status_var.set("Running...")
        state_for_worker = copy.deepcopy(self.state)

        def worker() -> None:
            try:
                result = run_impingement_case(
                    state_for_worker,
                    python_executable=sys.executable,
                    status_callback=lambda msg: self.worker_queue.put(("status", msg)),
                )
            except Exception:
                self.worker_queue.put(("error", traceback.format_exc()))
            else:
                self.worker_queue.put(("result", result))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "error":
                    self.is_running = False
                    self.back_button.config(state=tk.NORMAL if self.current_step > 0 else tk.DISABLED)
                    self.next_button.config(state=tk.NORMAL)
                    messagebox.showerror("Run failed", str(payload))
                    self.status_var.set("Run failed.")
                elif kind == "result":
                    self.is_running = False
                    self.last_result = payload
                    self.status_var.set("Run complete.")
                    self._show_step(7)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_worker_queue)

    def _reset_to_defaults(self) -> None:
        self.state = copy.deepcopy(self.default_state)
        self.status_var.set("Reset to notebook defaults.")
        self._show_step(0)


def _format_float(value: float) -> str:
    if abs(float(value)) >= 1e5 or (0.0 < abs(float(value)) < 1e-4):
        return f"{float(value):.8g}"
    return f"{float(value):.6g}"


def _format_payload_lines(payload: object, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_format_payload_lines(value, prefix + "  "))
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}[{index}]:")
                lines.extend(_format_payload_lines(value, prefix + "  "))
            else:
                lines.append(f"{prefix}[{index}]: {value}")
    else:
        lines.append(f"{prefix}{payload}")
    return lines


def _diff_payload(current: object, default: object, prefix: str = ""):
    if isinstance(current, dict) and isinstance(default, dict):
        for key in current:
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _diff_payload(current[key], default.get(key), next_prefix)
    elif isinstance(current, list) and isinstance(default, list):
        for index, value in enumerate(current):
            next_default = default[index] if index < len(default) else None
            yield from _diff_payload(value, next_default, f"{prefix}[{index}]")
    else:
        if current != default:
            yield f"{prefix}: {default} -> {current}"


def main() -> int:
    root = tk.Tk()
    ImpingementWizardGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
