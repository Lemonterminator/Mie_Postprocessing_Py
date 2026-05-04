from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLP_DIR = Path(__file__).resolve().parent
V2_DIR = MLP_DIR / "training"
for search_path in (PROJECT_ROOT, MLP_DIR, V2_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from engineered_feature_common import (  # noqa: E402
    TIME_FEATURE,
    build_dataset_registry,
    build_feature_matrix_np,
    infer_feature_family,
    load_run_artifacts,
)
from piston.design_A import _validate_design_a_geometry, piston_design_A  # noqa: E402
from piston.design_B import build_piston_design_b, piston_design_B  # noqa: E402
from piston.design_hermite import build_piston_design_hermite, piston_design_hermite  # noqa: E402

DesignChoice = Literal["A", "B", "Hermite"]
StatusCallback = Callable[[str], None] | None

DESIGN_LABELS: dict[DesignChoice, str] = {
    "A": "Design A",
    "B": "Design B",
    "Hermite": "Design Hermite",
}
FRAME_RATE = 30
PREVIEW_BG_PANEL = "#ffffff"
PREVIEW_FG_TEXT = "#111827"
PREVIEW_SCALE = 25.0


@dataclass(frozen=True)
class NotebookDefaults:
    run_dir: Path
    solver: dict[str, float]
    time_grid: dict[str, float]
    nn_conditions: dict[str, float]
    piston_motion: dict[str, float]
    piston_a_params: dict[str, float]
    piston_b_params: dict[str, float]
    hermite_params: dict[str, float]
    hermite_ctrl_pts_mm: tuple[tuple[float, float], ...]
    hermite_ctrl_vels: tuple[tuple[float, float], ...]
    selected_design: DesignChoice = "A"


@dataclass
class WizardState:
    run_dir: Path
    selected_design: DesignChoice
    solver: dict[str, float]
    time_grid: dict[str, float]
    nn_conditions: dict[str, float]
    piston_motion: dict[str, float]
    piston_a_params: dict[str, float]
    piston_b_params: dict[str, float]
    hermite_params: dict[str, float]
    hermite_ctrl_pts_mm: list[tuple[float, float]]
    hermite_ctrl_vels: list[tuple[float, float]]

    @property
    def design_label(self) -> str:
        return DESIGN_LABELS[self.selected_design]

    @property
    def selected_piston_height(self) -> float:
        return float(self.selected_design_dict()["piston_height"])

    def selected_design_dict(self) -> dict[str, float]:
        if self.selected_design == "A":
            return self.piston_a_params
        if self.selected_design == "B":
            return self.piston_b_params
        return self.hermite_params


@dataclass
class ImpingementRunResult:
    state: WizardState
    output_dir: Path
    cache_path: Path
    summary_json_path: Path
    summary_plot_path: Path
    preview_plot_path: Path
    video_path: Path | None
    warnings: list[str]
    metrics: dict[str, float]
    time_ms: np.ndarray
    collision_prob: np.ndarray
    wall_prob: np.ndarray
    cumulative_collision: np.ndarray
    onset_prob: np.ndarray | None


@dataclass
class _SimulationSnapshot:
    design_label: str
    design_code: DesignChoice
    output_dir: Path
    time_ms: np.ndarray
    collision_prob: np.ndarray
    wall_prob: np.ndarray
    cumulative_collision: np.ndarray
    onset_prob: np.ndarray | None
    pdf_frames: list[np.ndarray]
    piston_frames: list[np.ndarray]
    piston_top_y_frames: np.ndarray
    tip_x_arr: np.ndarray
    tip_y_arr: np.ndarray
    sigma_axis_arr: np.ndarray
    sigma_ortho_arr: np.ndarray
    mu_plot_arr: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    canvas_w_mm: float
    canvas_h_mm: float
    grid_size: float
    bore_radius: float
    cone_angle_deg: float
    half_cone_rad: float
    tilt_rad: float
    injector_xy: tuple[float, float]
    rpm: float
    crank_phase_deg: float
    t_start_ms: float
    t_end_ms: float
    fps: int


def _emit_status(callback: StatusCallback, message: str) -> None:
    if callback is not None:
        callback(message)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid_fn(y, x))


def build_default_state(defaults: NotebookDefaults) -> WizardState:
    return WizardState(
        run_dir=Path(defaults.run_dir),
        selected_design=defaults.selected_design,
        solver={key: float(value) for key, value in defaults.solver.items()},
        time_grid={key: float(value) for key, value in defaults.time_grid.items()},
        nn_conditions={key: float(value) for key, value in defaults.nn_conditions.items()},
        piston_motion={key: float(value) for key, value in defaults.piston_motion.items()},
        piston_a_params={key: float(value) for key, value in defaults.piston_a_params.items()},
        piston_b_params={key: float(value) for key, value in defaults.piston_b_params.items()},
        hermite_params={key: float(value) for key, value in defaults.hermite_params.items()},
        hermite_ctrl_pts_mm=[(float(x), float(y)) for x, y in defaults.hermite_ctrl_pts_mm],
        hermite_ctrl_vels=[(float(x), float(y)) for x, y in defaults.hermite_ctrl_vels],
    )


def serialize_wizard_state(state: WizardState) -> dict[str, Any]:
    return {
        "run_dir": str(Path(state.run_dir)),
        "selected_design": state.selected_design,
        "solver": dict(state.solver),
        "time_grid": {
            **{key: float(value) for key, value in state.time_grid.items() if key != "n_frames"},
            "n_frames": int(round(float(state.time_grid["n_frames"]))),
        },
        "nn_conditions": dict(state.nn_conditions),
        "piston_motion": dict(state.piston_motion),
        "piston_a_params": dict(state.piston_a_params),
        "piston_b_params": dict(state.piston_b_params),
        "hermite_params": dict(state.hermite_params),
        "hermite_ctrl_pts_mm": [[float(x), float(y)] for x, y in state.hermite_ctrl_pts_mm],
        "hermite_ctrl_vels": [[float(x), float(y)] for x, y in state.hermite_ctrl_vels],
    }


def build_summary_figure(result: ImpingementRunResult) -> plt.Figure:
    return _create_summary_figure(
        design_label=result.state.design_label,
        time_ms=result.time_ms,
        collision_prob=result.collision_prob,
        wall_prob=result.wall_prob,
        cumulative_collision=result.cumulative_collision,
        onset_prob=result.onset_prob,
        piston_motion=result.state.piston_motion,
        time_grid=result.state.time_grid,
    )


def validate_wizard_state(state: WizardState) -> None:
    run_dir = Path(state.run_dir).expanduser()
    validate_impingement_run_dir(run_dir)

    grid_size = float(state.solver["grid_size"])
    right_padding = float(state.solver["right_padding"])
    canvas_h_mm = float(state.solver["canvas_h_mm"])
    if grid_size <= 0.0:
        raise ValueError("grid_size must be > 0.")
    if right_padding < 0.0:
        raise ValueError("right_padding must be >= 0.")
    if canvas_h_mm <= 0.0:
        raise ValueError("canvas_h_mm must be > 0.")

    n_frames = int(round(float(state.time_grid["n_frames"])))
    t_start_ms = float(state.time_grid["t_start_ms"])
    t_end_ms = float(state.time_grid["t_end_ms"])
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2.")
    if t_end_ms <= t_start_ms:
        raise ValueError("t_end_ms must be greater than t_start_ms.")

    rpm = float(state.piston_motion["RPM"])
    stroke_half = float(state.piston_motion["stroke_half"])
    offset_baseline = float(state.piston_motion["offset_baseline"])
    if rpm < 0.0:
        raise ValueError("RPM must be >= 0.")
    if stroke_half < 0.0:
        raise ValueError("stroke_half must be >= 0.")
    if offset_baseline < 0.0:
        raise ValueError("offset_baseline must be >= 0.")

    _require_positive(state.nn_conditions, "plumes")
    _require_positive(state.nn_conditions, "diameter_mm")
    _require_positive(state.nn_conditions, "injection_duration_us")
    _require_positive(state.nn_conditions, "injection_pressure_bar")
    _require_positive(state.nn_conditions, "ambient_pressure_bar_phys")
    _require_positive(state.nn_conditions, "control_backpressure_bar")
    cone_angle_deg = float(state.nn_conditions["cone_angle_deg"])
    if cone_angle_deg <= 0.0 or cone_angle_deg >= 180.0:
        raise ValueError("cone_angle_deg must stay between 0 and 180.")

    if state.selected_design not in DESIGN_LABELS:
        raise ValueError(f"Unsupported design selection: {state.selected_design!r}")

    if state.selected_design == "A":
        _validate_design_a_geometry(
            r_bore=float(state.piston_a_params["r_bore"]),
            r_outer_bowl=float(state.piston_a_params["r_outer_bowl"]),
            r_topland=float(state.piston_a_params["r_topland"]),
            r_ring=float(state.piston_a_params["r_ring"]),
            r_inner_bowl=float(state.piston_a_params["r_inner_bowl"]),
            r_lip=float(state.piston_a_params["r_lip"]),
            r_floor=float(state.piston_a_params["r_floor"]),
            h_vertical=float(state.piston_a_params["h_vertical"]),
            piston_height=float(state.piston_a_params["piston_height"]),
        )
    elif state.selected_design == "B":
        build_piston_design_b(
            _build_validation_canvas(state.piston_b_params["r_bore"], state),
            grid_size,
            cylinder_head_offset=offset_baseline,
            **state.piston_b_params,
        )
    else:
        ctrl_pts = np.asarray(state.hermite_ctrl_pts_mm, dtype=float)
        ctrl_vels = np.asarray(state.hermite_ctrl_vels, dtype=float)
        if ctrl_pts.shape != (4, 2):
            raise ValueError("Hermite ctrl_pts_mm must be a 4x2 table.")
        if ctrl_vels.shape != (4, 2):
            raise ValueError("Hermite ctrl_vels must be a 4x2 table.")
        if np.any(np.diff(ctrl_pts[:, 0]) < 0.0):
            raise ValueError("Hermite control-point x coordinates must be nondecreasing.")
        build_piston_design_hermite(
            _build_validation_canvas(state.hermite_params["r_bore"], state),
            grid_size,
            cylinder_head_offset=offset_baseline,
            ctrl_pts_mm=ctrl_pts,
            ctrl_vels=ctrl_vels,
            **state.hermite_params,
        )

    piston_height = state.selected_piston_height
    crown_top_min = offset_baseline - stroke_half
    piston_bottom_max = offset_baseline + stroke_half + piston_height
    if crown_top_min < 0.0:
        raise ValueError(
            "Piston motion lifts the crown above the cylinder head. "
            "Increase offset_baseline or reduce stroke_half."
        )
    if piston_bottom_max > canvas_h_mm:
        raise ValueError(
            "Piston motion exceeds canvas_h_mm. Increase canvas_h_mm or reduce stroke_half/offset_baseline."
        )


def validate_impingement_run_dir(run_dir: Path) -> None:
    resolved = Path(run_dir).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"RUN_DIR does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"RUN_DIR must point to a directory: {resolved}")

    config_path = resolved / "train_config_used.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"RUN_DIR is missing train_config_used.json: {config_path}. "
            "Select a saved engineered_v2 run directory."
        )

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse train_config_used.json under {resolved}: {exc}") from exc

    feature_columns = payload.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(isinstance(item, str) for item in feature_columns):
        raise ValueError(
            f"RUN_DIR {resolved} has an invalid feature_columns payload in train_config_used.json."
        )

    feature_family = infer_feature_family(feature_columns)
    if feature_family != "engineered_v2":
        raise ValueError(
            f"RUN_DIR {resolved} is a {feature_family!r} run. "
            "The impingement GUI requires an engineered_v2 run."
        )

    model_candidates = (
        "best_model_refinement.pt",
        "best_model_stage2.pt",
        "best_model_stage1.pt",
    )
    if not any((resolved / name).exists() for name in model_candidates):
        formatted = ", ".join(model_candidates)
        raise FileNotFoundError(
            f"RUN_DIR {resolved} is missing model weights. Expected one of: {formatted}."
        )


def run_impingement_case(
    state: WizardState,
    *,
    output_root: Path | None = None,
    render_manim: bool = True,
    python_executable: str | None = None,
    status_callback: StatusCallback = None,
) -> ImpingementRunResult:
    validate_wizard_state(state)
    _emit_status(status_callback, "Loading model artifacts...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = load_run_artifacts(Path(state.run_dir).expanduser(), device=device)
    feature_columns = list(artifacts.train_config["feature_columns"])
    feature_family = infer_feature_family(feature_columns)
    if feature_family != "engineered_v2":
        raise RuntimeError(
            f"RUN_DIR {artifacts.run_dir} is not an engineered_v2 run. Got {feature_family!r}."
        )

    registry = build_dataset_registry(PROJECT_ROOT / "test_matrix_json")
    output_dir = _make_output_dir(state, output_root)
    _emit_status(status_callback, f"Writing outputs to {output_dir}...")

    snapshot = _simulate_selected_design(state, artifacts, registry, output_dir, status_callback)

    summary_plot_path = output_dir / "summary.png"
    preview_plot_path = output_dir / "preview.png"
    cache_path = output_dir / "impingement_frames.npz"
    summary_json_path = output_dir / "summary.json"

    _emit_status(status_callback, "Saving summary plot...")
    summary_figure = _create_summary_figure(
        design_label=snapshot.design_label,
        time_ms=snapshot.time_ms,
        collision_prob=snapshot.collision_prob,
        wall_prob=snapshot.wall_prob,
        cumulative_collision=snapshot.cumulative_collision,
        onset_prob=snapshot.onset_prob,
        piston_motion=state.piston_motion,
        time_grid=state.time_grid,
    )
    summary_figure.savefig(summary_plot_path, dpi=180, bbox_inches="tight")
    plt.close(summary_figure)

    _emit_status(status_callback, "Saving preview figure...")
    preview_figure = _create_preview_figure(snapshot)
    preview_figure.savefig(preview_plot_path, dpi=180, bbox_inches="tight")
    plt.close(preview_figure)

    _emit_status(status_callback, "Preparing Manim cache...")
    _write_manim_cache(cache_path, snapshot)

    warnings: list[str] = []
    video_path: Path | None = None
    if render_manim:
        _emit_status(status_callback, "Rendering Manim video...")
        video_path, warnings = render_manim_video(
            cache_path,
            output_dir,
            python_executable=python_executable or sys.executable,
            status_callback=status_callback,
        )

    metrics = {
        "peak_piston_impact": float(np.max(snapshot.collision_prob)),
        "peak_piston_time_ms": float(snapshot.time_ms[int(np.argmax(snapshot.collision_prob))]),
        "cumulative_piston_impact_ms": _trapezoid(snapshot.collision_prob, snapshot.time_ms),
        "peak_wall_impact": float(np.max(snapshot.wall_prob)),
    }

    result = ImpingementRunResult(
        state=state,
        output_dir=output_dir,
        cache_path=cache_path,
        summary_json_path=summary_json_path,
        summary_plot_path=summary_plot_path,
        preview_plot_path=preview_plot_path,
        video_path=video_path,
        warnings=warnings,
        metrics=metrics,
        time_ms=snapshot.time_ms,
        collision_prob=snapshot.collision_prob,
        wall_prob=snapshot.wall_prob,
        cumulative_collision=snapshot.cumulative_collision,
        onset_prob=snapshot.onset_prob,
    )
    _write_summary_json(result)
    _emit_status(status_callback, "Done.")
    return result


def render_manim_video(
    cache_path: Path,
    output_dir: Path,
    *,
    python_executable: str,
    status_callback: StatusCallback = None,
) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    if importlib.util.find_spec("manim") is None:
        warnings.append("Manim is not installed in the current Python environment. Skipped MP4 export.")
        return None, warnings

    scene_path = MLP_DIR / "impingement_manim_scene.py"
    media_dir = output_dir / "manim_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-m",
        "manim",
        "-v",
        "WARNING",
        "-qm",
        str(scene_path),
        "PistonImpingementScene",
        "--media_dir",
        str(media_dir),
        "-o",
        "impingement_scene",
    ]
    environment = os.environ.copy()
    environment["IMPINGEMENT_MANIM_CACHE"] = str(cache_path)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        cwd=PROJECT_ROOT,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            warnings.append(f"Manim render failed: {stderr.splitlines()[-1]}")
        else:
            warnings.append("Manim render failed with a non-zero exit code.")
        return None, warnings

    rendered_videos = sorted(media_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime)
    if not rendered_videos:
        warnings.append("Manim finished without producing an MP4 file.")
        return None, warnings

    source_video = rendered_videos[-1]
    final_video = output_dir / "impingement_scene.mp4"
    if source_video.resolve() != final_video.resolve():
        shutil.copy2(source_video, final_video)
    _emit_status(status_callback, f"Saved video to {final_video}.")
    return final_video, warnings


def _simulate_selected_design(
    state: WizardState,
    artifacts: Any,
    registry: dict[str, Any],
    output_dir: Path,
    status_callback: StatusCallback,
) -> _SimulationSnapshot:
    _emit_status(status_callback, "Building selected piston geometry...")
    selected_params = state.selected_design_dict()
    grid_size = float(state.solver["grid_size"])
    right_padding = float(state.solver["right_padding"])
    canvas_h_mm = float(state.solver["canvas_h_mm"])
    bore_radius = float(selected_params["r_bore"])
    canvas_w_mm, canvas_w_px, canvas_h_px = _canvas_dims_mm_to_px(
        bore_radius=bore_radius,
        right_padding=right_padding,
        grid_size=grid_size,
        canvas_h_mm=canvas_h_mm,
    )
    canvas = np.zeros((canvas_h_px, canvas_w_px), dtype=np.uint8)
    x_axis = np.arange(canvas_w_px, dtype=float) * grid_size
    y_axis = np.arange(canvas_h_px, dtype=float) * grid_size
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")
    wall_mask = np.zeros((canvas_h_px, canvas_w_px), dtype=bool)
    wall_mask[:, _mm_to_grid_index(bore_radius, grid_size):] = True

    make_piston_mask = _make_selected_mask_builder(state, canvas)

    _emit_status(status_callback, "Running the MLP time sweep...")
    time_ms = np.linspace(
        float(state.time_grid["t_start_ms"]),
        float(state.time_grid["t_end_ms"]),
        int(round(float(state.time_grid["n_frames"]))),
        dtype=np.float32,
    )
    prediction = _predict_physical_with_onset(state, artifacts, registry, time_ms)
    mu_arr = prediction["mu_np"]
    std_arr = prediction["std_np"]
    onset_prob_arr = prediction["onset_prob_np"]

    tilt_rad = float(state.nn_conditions["tilt_angle_radian"])
    cone_angle_deg = float(state.nn_conditions["cone_angle_deg"])
    half_cone_rad = float(np.deg2rad(cone_angle_deg / 2.0))
    injector_xy = (
        float(state.nn_conditions["injector_x_mm"]),
        float(state.nn_conditions["injector_y_mm"]),
    )

    rpm = float(state.piston_motion["RPM"])
    stroke_half = float(state.piston_motion["stroke_half"])
    offset_baseline = float(state.piston_motion["offset_baseline"])
    crank_phase_deg = float(state.piston_motion["crank_phase_deg"])
    omega_rad_per_ms = 2.0 * np.pi * rpm / 60.0 / 1000.0
    phase_rad = np.deg2rad(crank_phase_deg)
    piston_height = float(selected_params["piston_height"])

    piston_frames: list[np.ndarray] = []
    pdf_frames: list[np.ndarray] = []
    mu_plot_arr = np.zeros_like(time_ms, dtype=float)
    sigma_axis_arr = np.zeros_like(time_ms, dtype=float)
    sigma_ortho_arr = np.zeros_like(time_ms, dtype=float)
    tip_x_arr = np.zeros_like(time_ms, dtype=float)
    tip_y_arr = np.zeros_like(time_ms, dtype=float)
    piston_top_y_frames = np.full((len(time_ms), canvas_w_px), np.nan, dtype=float)
    collision_prob = np.zeros_like(time_ms, dtype=float)
    wall_prob = np.zeros_like(time_ms, dtype=float)

    _emit_status(status_callback, "Rendering simulation frames...")
    for index, t_ms in enumerate(time_ms):
        offset = offset_baseline - stroke_half * np.cos(omega_rad_per_ms * float(t_ms) + phase_rad)
        piston_mask = np.asarray(make_piston_mask(offset), dtype=bool)

        mu_raw = float(mu_arr[index])
        mu_plot = max(mu_raw, 0.0)
        sigma_axis = max(float(std_arr[index]), 1.5 * grid_size)
        sigma_ortho = max(max(mu_plot, sigma_axis) * np.tan(half_cone_rad) / 3.0, 2.0 * grid_size)
        tip_x = injector_xy[0] + mu_plot * np.cos(tilt_rad)
        tip_y = injector_xy[1] + mu_plot * np.sin(tilt_rad)

        pdf = gaussian_2d_rotated(X, Y, tip_x, tip_y, sigma_axis, sigma_ortho, tilt_rad)
        pdf /= pdf.sum() * grid_size**2 + 1e-30
        pdf = pdf.astype(np.float32, copy=False)

        collision_prob[index] = float((piston_mask * pdf).sum() * grid_size**2)
        wall_prob[index] = float((wall_mask * pdf).sum() * grid_size**2)

        piston_frames.append(piston_mask)
        pdf_frames.append(pdf)
        mu_plot_arr[index] = mu_plot
        sigma_axis_arr[index] = sigma_axis
        sigma_ortho_arr[index] = sigma_ortho
        tip_x_arr[index] = tip_x
        tip_y_arr[index] = tip_y

        active_cols = np.any(piston_mask, axis=0)
        if np.any(active_cols):
            top_rows = np.argmax(piston_mask, axis=0)
            piston_top_y_frames[index, active_cols] = y_axis[top_rows[active_cols]]

    cumulative_collision = np.array(
        [
            _trapezoid(collision_prob[: frame_index + 1], time_ms[: frame_index + 1])
            for frame_index in range(len(time_ms))
        ],
        dtype=float,
    )

    return _SimulationSnapshot(
        design_label=state.design_label,
        design_code=state.selected_design,
        output_dir=output_dir,
        time_ms=time_ms.astype(np.float32),
        collision_prob=collision_prob.astype(np.float32),
        wall_prob=wall_prob.astype(np.float32),
        cumulative_collision=cumulative_collision.astype(np.float32),
        onset_prob=None if onset_prob_arr is None else onset_prob_arr.astype(np.float32),
        pdf_frames=pdf_frames,
        piston_frames=piston_frames,
        piston_top_y_frames=piston_top_y_frames.astype(np.float32),
        tip_x_arr=tip_x_arr.astype(np.float32),
        tip_y_arr=tip_y_arr.astype(np.float32),
        sigma_axis_arr=sigma_axis_arr.astype(np.float32),
        sigma_ortho_arr=sigma_ortho_arr.astype(np.float32),
        mu_plot_arr=mu_plot_arr.astype(np.float32),
        x_axis=x_axis.astype(np.float32),
        y_axis=y_axis.astype(np.float32),
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        canvas_w_mm=float(canvas_w_mm),
        canvas_h_mm=float(canvas_h_mm),
        grid_size=float(grid_size),
        bore_radius=float(bore_radius),
        cone_angle_deg=float(cone_angle_deg),
        half_cone_rad=float(half_cone_rad),
        tilt_rad=float(tilt_rad),
        injector_xy=injector_xy,
        rpm=rpm,
        crank_phase_deg=crank_phase_deg,
        t_start_ms=float(state.time_grid["t_start_ms"]),
        t_end_ms=float(state.time_grid["t_end_ms"]),
        fps=FRAME_RATE,
    )


def _predict_physical_with_onset(
    state: WizardState,
    artifacts: Any,
    registry: dict[str, Any],
    time_ms: np.ndarray,
) -> dict[str, np.ndarray | None]:
    feature_columns = list(artifacts.train_config["feature_columns"])
    time_feature = str(artifacts.train_config.get("time_feature", TIME_FEATURE))
    raw = {
        "tilt_angle_radian": float(state.nn_conditions["tilt_angle_radian"]),
        "plumes": float(state.nn_conditions["plumes"]),
        "diameter_mm": float(state.nn_conditions["diameter_mm"]),
        "injection_duration_us": float(state.nn_conditions["injection_duration_us"]),
        "injection_pressure_bar": float(state.nn_conditions["injection_pressure_bar"]),
        "ambient_pressure_bar_phys": float(state.nn_conditions["ambient_pressure_bar_phys"]),
        "control_backpressure_bar": float(state.nn_conditions["control_backpressure_bar"]),
    }
    features_np, a_scale_np, _canonical = build_feature_matrix_np(
        raw,
        time_ms,
        artifacts.scaler_state,
        feature_columns,
        registry,
        time_feature=time_feature,
    )
    device = next(artifacts.model.parameters()).device
    features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    a_scale = torch.as_tensor(a_scale_np.reshape(-1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        model_output = artifacts.model(features)
        mu_hat = model_output[..., :1]
        log_var_hat = model_output[..., 1:2]
        onset_logit = model_output[..., 2:3] if model_output.shape[-1] >= 3 else None
        log_var_hat = torch.clamp(log_var_hat, min=-20.0, max=20.0)
        std_floor = float(artifacts.train_config.get("std_clamp_min", 0.0))
        mu_phys = (a_scale * mu_hat).cpu().numpy().reshape(-1)
        std_phys = torch.clamp(a_scale * torch.exp(0.5 * log_var_hat), min=std_floor)
        onset_prob = None
        if onset_logit is not None:
            onset_prob = torch.sigmoid(onset_logit).cpu().numpy().reshape(-1)
    return {
        "mu_np": mu_phys.astype(np.float32),
        "std_np": std_phys.cpu().numpy().reshape(-1).astype(np.float32),
        "onset_prob_np": None if onset_prob is None else onset_prob.astype(np.float32),
    }


def _make_selected_mask_builder(state: WizardState, canvas: np.ndarray) -> Callable[[float], np.ndarray]:
    grid_size = float(state.solver["grid_size"])
    if state.selected_design == "A":
        params = dict(state.piston_a_params)

        def _builder(offset: float) -> np.ndarray:
            return piston_design_A(
                canvas,
                grid_size,
                cylinder_head_offset=float(offset),
                **params,
            )

        return _builder

    if state.selected_design == "B":
        params = dict(state.piston_b_params)

        def _builder(offset: float) -> np.ndarray:
            return piston_design_B(
                canvas,
                grid_size,
                cylinder_head_offset=float(offset),
                **params,
            )

        return _builder

    params = dict(state.hermite_params)
    ctrl_pts = np.asarray(state.hermite_ctrl_pts_mm, dtype=float)
    ctrl_vels = np.asarray(state.hermite_ctrl_vels, dtype=float)

    def _builder(offset: float) -> np.ndarray:
        return piston_design_hermite(
            canvas,
            grid_size,
            cylinder_head_offset=float(offset),
            ctrl_pts_mm=ctrl_pts,
            ctrl_vels=ctrl_vels,
            **params,
        )

    return _builder


def _create_summary_figure(
    *,
    design_label: str,
    time_ms: np.ndarray,
    collision_prob: np.ndarray,
    wall_prob: np.ndarray,
    cumulative_collision: np.ndarray,
    onset_prob: np.ndarray | None,
    piston_motion: dict[str, float],
    time_grid: dict[str, float],
) -> plt.Figure:
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
    ax_top.plot(time_ms, collision_prob, color="#2563eb", linewidth=2.0, label="piston impact")
    ax_top.plot(time_ms, wall_prob, color="#ef4444", linewidth=1.5, linestyle="--", label="bore wall impact")
    if onset_prob is not None:
        ax_top.plot(time_ms, onset_prob, color="#64748b", linewidth=1.2, linestyle="-.", label="onset probability")
    ax_top.set_ylabel("Probability")
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=8)
    ax_top.set_title(
        f"{design_label} — RPM {piston_motion['RPM']:.0f}, "
        f"phase {piston_motion['crank_phase_deg']:.0f}°, "
        f"{time_grid['t_start_ms']:.2f}–{time_grid['t_end_ms']:.2f} ms"
    )

    total_integral = _trapezoid(collision_prob, time_ms)
    ax_bottom.plot(time_ms, cumulative_collision, color="#0891b2", linewidth=2.0)
    ax_bottom.set_xlabel("Time [ms]")
    ax_bottom.set_ylabel("∫ P(piston) dt [ms]")
    ax_bottom.grid(alpha=0.3)
    ax_bottom.set_title(f"Cumulative piston impact integral = {total_integral:.3e} ms")

    fig.tight_layout()
    return fig


def _create_preview_figure(snapshot: _SimulationSnapshot) -> plt.Figure:
    peak_index = int(np.argmax(snapshot.collision_prob))
    figure, axis = plt.subplots(
        figsize=(snapshot.canvas_w_mm / PREVIEW_SCALE, snapshot.canvas_h_mm / PREVIEW_SCALE),
        facecolor="white",
    )
    axis.set_facecolor(PREVIEW_BG_PANEL)
    for spine in axis.spines.values():
        spine.set_edgecolor("#111827")
        spine.set_linewidth(1.2)
    axis.tick_params(colors=PREVIEW_FG_TEXT, which="both")
    axis.xaxis.label.set_color(PREVIEW_FG_TEXT)
    axis.yaxis.label.set_color(PREVIEW_FG_TEXT)

    extent = [0.0, snapshot.canvas_w_mm, snapshot.canvas_h_mm, 0.0]
    piston_overlay = np.ma.masked_where(
        ~snapshot.piston_frames[peak_index],
        np.ones_like(snapshot.piston_frames[peak_index], dtype=float),
    )
    pdf_cmap, pdf_norm = _build_pdf_cmap_and_norm(snapshot.pdf_frames)
    piston_cmap = mcolors.ListedColormap(["#000000"])
    piston_cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    line_glow = [pe.Stroke(linewidth=5.0, foreground="#000000", alpha=0.95), pe.Normal()]
    ellipse_glow = [pe.Stroke(linewidth=3.3, foreground="#ffffff", alpha=0.95), pe.Normal()]
    marker_glow = [pe.Stroke(linewidth=3.5, foreground="#ffffff", alpha=0.85), pe.Normal()]

    axis.imshow(
        piston_overlay,
        cmap=piston_cmap,
        origin="upper",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        alpha=0.82,
        zorder=1,
    )
    image_pdf = axis.imshow(
        snapshot.pdf_frames[peak_index],
        cmap=pdf_cmap,
        norm=pdf_norm,
        origin="upper",
        extent=extent,
        aspect="equal",
        alpha=_pdf_alpha_array_from_vmax(snapshot.pdf_frames[peak_index], float(pdf_norm.vmax)),
        zorder=2,
    )

    phi = np.linspace(0.0, 2.0 * np.pi, 181)
    c_tilt = math.cos(snapshot.tilt_rad)
    s_tilt = math.sin(snapshot.tilt_rad)

    def ellipse_xy(n_sigma: float) -> tuple[np.ndarray, np.ndarray]:
        sigma_axis = max(float(snapshot.sigma_axis_arr[peak_index]), 1e-3)
        sigma_ortho = max(float(snapshot.sigma_ortho_arr[peak_index]), 1e-3)
        u = n_sigma * sigma_axis * np.cos(phi)
        v = n_sigma * sigma_ortho * np.sin(phi)
        x = float(snapshot.tip_x_arr[peak_index]) + c_tilt * u - s_tilt * v
        y = float(snapshot.tip_y_arr[peak_index]) + s_tilt * u + c_tilt * v
        return x, y

    ell1_x, ell1_y = ellipse_xy(1.0)
    ell2_x, ell2_y = ellipse_xy(2.0)
    axis.plot(
        snapshot.x_axis,
        snapshot.piston_top_y_frames[peak_index],
        color="#ffffff",
        linewidth=2.8,
        solid_capstyle="round",
        alpha=0.98,
        zorder=5,
        label="piston crown",
        path_effects=line_glow,
    )
    axis.plot(
        snapshot.tip_x_arr[: peak_index + 1],
        snapshot.tip_y_arr[: peak_index + 1],
        color="#1d4ed8",
        linewidth=1.35,
        alpha=0.72,
        zorder=6,
        label="Gaussian center path",
    )
    axis.plot(
        snapshot.tip_x_arr[peak_index],
        snapshot.tip_y_arr[peak_index],
        marker="o",
        color="#fde68a",
        markeredgecolor="#0f172a",
        markersize=7.2,
        zorder=7,
        label="Gaussian center",
        path_effects=marker_glow,
    )
    axis.plot(
        ell1_x,
        ell1_y,
        color="#111827",
        linestyle="--",
        linewidth=1.75,
        alpha=0.98,
        zorder=6,
        label="1σ PDF ellipse",
        path_effects=ellipse_glow,
    )
    axis.plot(
        ell2_x,
        ell2_y,
        color="#0891b2",
        linestyle=":",
        linewidth=1.75,
        alpha=0.98,
        zorder=6,
        label="2σ PDF ellipse",
        path_effects=ellipse_glow,
    )
    axis.axvline(
        snapshot.bore_radius,
        color="#fb7185",
        linestyle="--",
        linewidth=1.25,
        alpha=0.95,
        zorder=6,
        label="bore wall",
    )

    guide_len = float(np.hypot(snapshot.canvas_w_mm, snapshot.canvas_h_mm))
    low_segment, high_segment = cone_guide_endpoints(
        snapshot.injector_xy,
        snapshot.tilt_rad,
        snapshot.half_cone_rad,
        guide_len,
    )
    low_segment = _clip_segment_to_canvas(low_segment, snapshot.canvas_w_mm, snapshot.canvas_h_mm)
    high_segment = _clip_segment_to_canvas(high_segment, snapshot.canvas_w_mm, snapshot.canvas_h_mm)
    axis.plot(
        [low_segment[0], low_segment[2]],
        [low_segment[1], low_segment[3]],
        color="#0891b2",
        linestyle=":",
        linewidth=1.15,
        alpha=0.78,
        zorder=6,
        label=f"cone ±{snapshot.cone_angle_deg / 2.0:.0f}°",
    )
    axis.plot(
        [high_segment[0], high_segment[2]],
        [high_segment[1], high_segment[3]],
        color="#0891b2",
        linestyle=":",
        linewidth=1.15,
        alpha=0.78,
        zorder=6,
    )
    axis.plot(
        *snapshot.injector_xy,
        marker="*",
        color="#fde047",
        markeredgecolor="#0f172a",
        markersize=13,
        zorder=7,
        label="injector",
    )

    axis.set_xlim(0.0, snapshot.canvas_w_mm)
    axis.set_ylim(snapshot.canvas_h_mm, 0.0)
    axis.set_xlabel("radial x [mm]")
    axis.set_ylabel("axial y [mm]")
    axis.grid(False)
    axis.set_title(f"{snapshot.design_label} — peak impact frame", color=PREVIEW_FG_TEXT, pad=10.0)

    colorbar = figure.colorbar(image_pdf, ax=axis, fraction=0.046, pad=0.02)
    colorbar.set_label("spray PDF [1/mm²]", color=PREVIEW_FG_TEXT)
    colorbar.ax.yaxis.set_tick_params(color=PREVIEW_FG_TEXT)
    plt.setp(colorbar.ax.yaxis.get_ticklabels(), color=PREVIEW_FG_TEXT)
    colorbar.outline.set_edgecolor("#111827")

    legend = axis.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.90,
        facecolor=PREVIEW_BG_PANEL,
        edgecolor="#111827",
    )
    for text in legend.get_texts():
        text.set_color(PREVIEW_FG_TEXT)

    axis.text(
        0.98,
        0.98,
        (
            f"t = {snapshot.time_ms[peak_index]:.2f} ms\n"
            f"μplot = {snapshot.mu_plot_arr[peak_index]:.1f} mm\n"
            f"σ∥ = {snapshot.sigma_axis_arr[peak_index]:.2f} mm\n"
            f"σ⊥ = {snapshot.sigma_ortho_arr[peak_index]:.2f} mm\n"
            f"P(impact) = {snapshot.collision_prob[peak_index]:.3e}\n"
            f"P(wall) = {snapshot.wall_prob[peak_index]:.3e}"
        ),
        transform=axis.transAxes,
        fontsize=9,
        color=PREVIEW_FG_TEXT,
        va="top",
        ha="right",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ffffff",
            "edgecolor": "#111827",
            "linewidth": 0.9,
            "alpha": 0.92,
        },
        zorder=8,
    )

    figure.tight_layout()
    return figure


def _write_manim_cache(cache_path: Path, snapshot: _SimulationSnapshot) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_cmap, pdf_norm = _build_pdf_cmap_and_norm(snapshot.pdf_frames)
    pdf_rgba_frames = np.stack(
        [rgba_frame_from_pdf(pdf, pdf_cmap=pdf_cmap, pdf_norm=pdf_norm) for pdf in snapshot.pdf_frames],
        axis=0,
    )
    pdf_colorbar_rgba = _build_pdf_colorbar_rgba(pdf_cmap, pdf_norm)
    piston_top_valid = np.isfinite(snapshot.piston_top_y_frames)
    piston_top_dense = np.where(
        piston_top_valid,
        snapshot.piston_top_y_frames,
        snapshot.canvas_h_mm + 5.0,
    ).astype(np.float32)

    c_tilt = float(np.cos(snapshot.tilt_rad))
    s_tilt = float(np.sin(snapshot.tilt_rad))

    def sigma_shell_rgba(frame_index: int) -> np.ndarray:
        sigma_axis = max(float(snapshot.sigma_axis_arr[frame_index]), 1e-3)
        sigma_ortho = max(float(snapshot.sigma_ortho_arr[frame_index]), 1e-3)
        dx = snapshot.X - float(snapshot.tip_x_arr[frame_index])
        dy = snapshot.Y - float(snapshot.tip_y_arr[frame_index])
        u = c_tilt * dx + s_tilt * dy
        v = -s_tilt * dx + c_tilt * dy
        r_squared = (u / sigma_axis) ** 2 + (v / sigma_ortho) ** 2
        pdf_value = np.exp(-0.5 * r_squared) / (2.0 * np.pi * sigma_axis * sigma_ortho)
        rgba = pdf_cmap(pdf_norm(pdf_value)).copy()
        scaled = np.clip(pdf_value / max(pdf_norm.vmax, 1e-12), 0.0, 1.0)
        alpha = np.where(
            r_squared <= 9.0,
            0.94 * np.clip(np.clip((scaled - 0.015) / 0.985, 0.0, 1.0) ** 0.50, 0.0, 1.0),
            0.0,
        )
        rgba[..., 3] = alpha
        return np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)

    sigma_shell_rgba_frames = np.stack(
        [sigma_shell_rgba(frame_index) for frame_index in range(len(snapshot.time_ms))],
        axis=0,
    )

    np.savez_compressed(
        cache_path,
        pdf_rgba_frames=pdf_rgba_frames,
        pdf_colorbar_rgba=pdf_colorbar_rgba,
        pdf_vmin=np.float32(pdf_norm.vmin),
        pdf_vmax=np.float32(pdf_norm.vmax),
        sigma_shell_rgba_frames=sigma_shell_rgba_frames,
        piston_top_y_frames=piston_top_dense,
        piston_top_valid=piston_top_valid,
        tip_x_arr=snapshot.tip_x_arr.astype(np.float32),
        tip_y_arr=snapshot.tip_y_arr.astype(np.float32),
        sigma_axis_arr=snapshot.sigma_axis_arr.astype(np.float32),
        sigma_ortho_arr=snapshot.sigma_ortho_arr.astype(np.float32),
        mu_plot_arr=snapshot.mu_plot_arr.astype(np.float32),
        collision_prob=snapshot.collision_prob.astype(np.float32),
        wall_prob=snapshot.wall_prob.astype(np.float32),
        toy_time_ms=snapshot.time_ms.astype(np.float32),
        x_axis=snapshot.x_axis.astype(np.float32),
        injector_xy=np.asarray(snapshot.injector_xy, dtype=np.float32),
        canvas_w_mm=np.float32(snapshot.canvas_w_mm),
        canvas_h_mm=np.float32(snapshot.canvas_h_mm),
        cylinder_radius=np.float32(snapshot.bore_radius),
        anim_bore_radius=np.float32(snapshot.bore_radius),
        cone_angle_deg=np.float32(snapshot.cone_angle_deg),
        half_cone_rad=np.float32(snapshot.half_cone_rad),
        tilt=np.float32(snapshot.tilt_rad),
        fps=np.int16(snapshot.fps),
        design_code=np.array(snapshot.design_code),
        design_label=np.array(snapshot.design_label),
    )


def _write_summary_json(result: ImpingementRunResult) -> None:
    payload = {
        "state": serialize_wizard_state(result.state),
        "metrics": result.metrics,
        "warnings": list(result.warnings),
        "outputs": {
            "output_dir": str(result.output_dir),
            "cache_path": str(result.cache_path),
            "summary_plot_path": str(result.summary_plot_path),
            "preview_plot_path": str(result.preview_plot_path),
            "summary_json_path": str(result.summary_json_path),
            "video_path": None if result.video_path is None else str(result.video_path),
        },
    }
    with result.summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _make_output_dir(state: WizardState, output_root: Path | None) -> Path:
    base = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs" / "impingement_gui"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{state.selected_design.lower()}"
    for suffix in range(100):
        output_dir = base / (stem if suffix == 0 else f"{stem}_{suffix:02d}")
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return output_dir
    raise FileExistsError(f"Could not create a unique output directory under {base}.")


def _canvas_dims_mm_to_px(
    *,
    bore_radius: float,
    right_padding: float,
    grid_size: float,
    canvas_h_mm: float,
) -> tuple[float, int, int]:
    canvas_w_mm = float(bore_radius) + float(right_padding)
    canvas_w_px = int(round(canvas_w_mm / float(grid_size)))
    canvas_h_px = int(round(float(canvas_h_mm) / float(grid_size)))
    if canvas_w_px <= 0 or canvas_h_px <= 0:
        raise ValueError("Canvas dimensions must resolve to at least one pixel.")
    return canvas_w_mm, canvas_w_px, canvas_h_px


def _build_validation_canvas(r_bore: float, state: WizardState) -> np.ndarray:
    _, canvas_w_px, canvas_h_px = _canvas_dims_mm_to_px(
        bore_radius=float(r_bore),
        right_padding=float(state.solver["right_padding"]),
        grid_size=float(state.solver["grid_size"]),
        canvas_h_mm=float(state.solver["canvas_h_mm"]),
    )
    return np.zeros((canvas_h_px, canvas_w_px), dtype=np.uint8)


def _mm_to_grid_index(value_mm: float, grid_size: float, mode: str = "round") -> int:
    scaled = float(value_mm) / float(grid_size)
    if mode == "round":
        return int(np.rint(scaled))
    if mode == "floor":
        return int(np.floor(scaled + 1e-9))
    if mode == "ceil":
        return int(np.ceil(scaled - 1e-9))
    raise ValueError(f"Unsupported mode: {mode!r}")


def _build_pdf_cmap_and_norm(pdf_frames: list[np.ndarray]) -> tuple[mcolors.Colormap, mcolors.Normalize]:
    positive_slices = [frame[frame > 0.0].ravel() for frame in pdf_frames if np.any(frame > 0.0)]
    pdf_positive = np.concatenate(positive_slices) if positive_slices else np.array([1.0])
    pdf_vmax = float(np.percentile(pdf_positive, 99.7))
    pdf_vmax = max(pdf_vmax, max(float(frame.max()) for frame in pdf_frames))
    pdf_cmap = mcolors.LinearSegmentedColormap.from_list(
        "spray_pdf",
        ["#26104a", "#2349ff", "#14cfff", "#88f36b", "#ffe45c", "#ff922b", "#d7191c"],
    )
    pdf_norm = mcolors.PowerNorm(gamma=0.42, vmin=0.0, vmax=pdf_vmax)
    return pdf_cmap, pdf_norm


def _build_pdf_colorbar_rgba(
    pdf_cmap: mcolors.Colormap,
    pdf_norm: mcolors.Normalize,
    *,
    height_px: int = 256,
    width_px: int = 18,
) -> np.ndarray:
    values = np.linspace(float(pdf_norm.vmax), float(pdf_norm.vmin), height_px, dtype=np.float32)[:, None]
    rgba = pdf_cmap(pdf_norm(values))
    rgba[..., -1] = 1.0
    rgba = np.repeat(rgba, width_px, axis=1)
    return np.clip(np.rint(255.0 * rgba), 0, 255).astype(np.uint8)


def _pdf_alpha_array_from_vmax(pdf: np.ndarray, pdf_vmax: float) -> np.ndarray:
    scaled = np.clip(pdf / max(pdf_vmax, 1e-12), 0.0, 1.0)
    alpha = np.clip((scaled - 0.002) / 0.998, 0.0, 1.0) ** 0.35
    return np.clip(0.98 * alpha, 0.0, 0.98)


def rgba_frame_from_pdf(
    pdf: np.ndarray,
    *,
    pdf_cmap: mcolors.Colormap,
    pdf_norm: mcolors.Normalize,
) -> np.ndarray:
    rgba = pdf_cmap(pdf_norm(pdf))
    rgba[..., -1] = _pdf_alpha_array_from_vmax(pdf, float(pdf_norm.vmax))
    return np.clip(np.rint(255.0 * rgba), 0, 255).astype(np.uint8)


def gaussian_2d_rotated(
    X: np.ndarray,
    Y: np.ndarray,
    x0: float,
    y0: float,
    sigma_axis: float,
    sigma_ortho: float,
    theta: float,
) -> np.ndarray:
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    dx = X - x0
    dy = Y - y0
    u = c_theta * dx + s_theta * dy
    v = -s_theta * dx + c_theta * dy
    return np.exp(-0.5 * ((u / sigma_axis) ** 2 + (v / sigma_ortho) ** 2)) / (
        2.0 * np.pi * sigma_axis * sigma_ortho
    )


def cone_guide_endpoints(
    injector_xy: tuple[float, float],
    tilt_rad: float,
    half_cone_rad: float,
    length_mm: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    x0, y0 = injector_xy
    a_low = tilt_rad - half_cone_rad
    a_high = tilt_rad + half_cone_rad
    return (
        (x0, y0, x0 + length_mm * np.cos(a_low), y0 + length_mm * np.sin(a_low)),
        (x0, y0, x0 + length_mm * np.cos(a_high), y0 + length_mm * np.sin(a_high)),
    )


def _clip_segment_to_canvas(
    segment: tuple[float, float, float, float],
    canvas_w_mm: float,
    canvas_h_mm: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in segment)
    dx = x1 - x0
    dy = y1 - y0
    t_min = 0.0
    t_max = 1.0
    bounds = (
        (-dx, x0),
        (dx, float(canvas_w_mm) - x0),
        (-dy, y0),
        (dy, float(canvas_h_mm) - y0),
    )
    for direction, distance in bounds:
        if abs(direction) < 1e-12:
            if distance < 0.0:
                clamped_x = min(max(x0, 0.0), float(canvas_w_mm))
                clamped_y = min(max(y0, 0.0), float(canvas_h_mm))
                return (clamped_x, clamped_y, clamped_x, clamped_y)
            continue
        t = distance / direction
        if direction < 0.0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
        if t_min > t_max:
            clamped_x = min(max(x0, 0.0), float(canvas_w_mm))
            clamped_y = min(max(y0, 0.0), float(canvas_h_mm))
            return (clamped_x, clamped_y, clamped_x, clamped_y)
    return (x0 + t_min * dx, y0 + t_min * dy, x0 + t_max * dx, y0 + t_max * dy)


def _require_positive(values: dict[str, float], key: str) -> None:
    if float(values[key]) <= 0.0:
        raise ValueError(f"{key} must be > 0.")
