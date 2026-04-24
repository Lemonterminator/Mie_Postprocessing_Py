from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from manim import *

config.background_color = WHITE

def _resolve_cache_path() -> Path:
    candidates: list[Path] = []
    env_cache = os.environ.get("IMPINGEMENT_MANIM_CACHE")
    if env_cache:
        candidates.append(Path(env_cache))
    candidates.extend(
        [
            Path("outputs/impingement_gui/latest/impingement_frames.npz"),
            Path("MLP/_manim_cache/impingement_with_piston_frames.npz"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Manim cache not found. Set IMPINGEMENT_MANIM_CACHE or generate the cache first."
    )


def _scalar_text(data: np.lib.npyio.NpzFile, key: str, fallback: str) -> str:
    if key not in data.files:
        return fallback
    value = data[key]
    if np.isscalar(value):
        return str(value)
    if getattr(value, "shape", None) == ():
        return str(value.item())
    return fallback


CACHE_PATH = _resolve_cache_path()
DATA = np.load(CACHE_PATH)

pdf_rgba_frames = DATA["pdf_rgba_frames"]
if "pdf_colorbar_rgba" in DATA.files:
    pdf_colorbar_rgba = DATA["pdf_colorbar_rgba"]
else:
    colorbar_ramp = np.linspace(255, 0, 256, dtype=np.uint8)[:, None]
    pdf_colorbar_rgba = np.dstack(
        [
            colorbar_ramp,
            np.zeros_like(colorbar_ramp),
            255 - colorbar_ramp,
            np.full_like(colorbar_ramp, 255),
        ]
    )
pdf_vmin = float(DATA["pdf_vmin"]) if "pdf_vmin" in DATA.files else 0.0
pdf_vmax = float(DATA["pdf_vmax"]) if "pdf_vmax" in DATA.files else 1.0
if "sigma_shell_rgba_frames" in DATA.files:
    sigma_shell_rgba_frames = DATA["sigma_shell_rgba_frames"]
else:
    height, width = pdf_rgba_frames.shape[1:3]
    sigma_shell_rgba_frames = np.zeros((len(pdf_rgba_frames), height, width, 4), dtype=np.uint8)

piston_top_y_frames = DATA["piston_top_y_frames"]
piston_top_valid = DATA["piston_top_valid"]
tip_x_arr = DATA["tip_x_arr"]
tip_y_arr = DATA["tip_y_arr"]
sigma_axis_arr = DATA["sigma_axis_arr"]
sigma_ortho_arr = DATA["sigma_ortho_arr"]
mu_plot_arr = DATA["mu_plot_arr"]
collision_prob = DATA["collision_prob"]
toy_time_ms = DATA["toy_time_ms"]
x_axis = DATA["x_axis"]
injector_xy = DATA["injector_xy"]
canvas_w_mm = float(DATA["canvas_w_mm"])
canvas_h_mm = float(DATA["canvas_h_mm"])
cylinder_radius = float(DATA["cylinder_radius"])
anim_bore_radius = float(DATA.get("anim_bore_radius", cylinder_radius))
cone_angle_deg = float(DATA["cone_angle_deg"])
half_cone_rad = float(DATA["half_cone_rad"])
tilt = float(DATA["tilt"])
base_fps = int(DATA["fps"])
design_label = _scalar_text(DATA, "design_label", "Selected Design")

config.frame_rate = max(60, base_fps * 2)

FG_TEXT = "#1e293b"
PLOT_EDGE = "#64748b"
PANEL_FILL = "#f8fafc"
PANEL_EDGE = "#94a3b8"

n_frames = len(toy_time_ms)
phi = np.linspace(0.0, 2.0 * np.pi, 181, dtype=np.float32)
c_tilt = float(np.cos(tilt))
s_tilt = float(np.sin(tilt))


def split_frame_pos(frame_pos: float) -> tuple[float, int, int, float]:
    frame_pos = float(np.clip(frame_pos, 0.0, n_frames - 1.0))
    lo = int(np.floor(frame_pos))
    hi = min(lo + 1, n_frames - 1)
    alpha = frame_pos - lo
    return frame_pos, lo, hi, alpha


def interp_scalar(arr: np.ndarray, frame_pos: float) -> float:
    _, lo, hi, alpha = split_frame_pos(frame_pos)
    return float((1.0 - alpha) * arr[lo] + alpha * arr[hi])


def blended_rgba(frame_pos: float) -> np.ndarray:
    _, lo, hi, alpha = split_frame_pos(frame_pos)
    if hi == lo or alpha <= 1e-6:
        return pdf_rgba_frames[lo]
    blend = (1.0 - alpha) * pdf_rgba_frames[lo].astype(np.float32) + alpha * pdf_rgba_frames[hi].astype(np.float32)
    return np.clip(np.rint(blend), 0, 255).astype(np.uint8)


def blended_sigma_shell(frame_pos: float) -> np.ndarray:
    _, lo, hi, alpha = split_frame_pos(frame_pos)
    if hi == lo or alpha <= 1e-6:
        return sigma_shell_rgba_frames[lo]
    blend = (
        (1.0 - alpha) * sigma_shell_rgba_frames[lo].astype(np.float32)
        + alpha * sigma_shell_rgba_frames[hi].astype(np.float32)
    )
    return np.clip(np.rint(blend), 0, 255).astype(np.uint8)


def cone_guide_endpoints(
    injector_xy: np.ndarray,
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


def clip_segment_to_canvas(
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


class PistonImpingementScene(Scene):
    def construct(self) -> None:
        axes_height = 7.1
        axes_width = axes_height * canvas_w_mm / canvas_h_mm
        axes = Axes(
            x_range=[0.0, canvas_w_mm, 20.0],
            y_range=[0.0, canvas_h_mm, 40.0],
            x_length=axes_width,
            y_length=axes_height,
            axis_config={
                "color": PLOT_EDGE,
                "include_tip": False,
                "stroke_width": 1.5,
            },
        )
        axes.to_edge(LEFT, buff=0.8)

        plot_panel = RoundedRectangle(
            corner_radius=0.18,
            width=axes.width + 0.55,
            height=axes.height + 0.55,
        ).move_to(axes)
        plot_panel.set_fill(WHITE, opacity=0.0)
        plot_panel.set_stroke(PANEL_EDGE, width=1.0, opacity=0.6)

        def mm_to_scene(x_mm: float, y_mm: float):
            return axes.c2p(float(x_mm), canvas_h_mm - float(y_mm))

        plot_center = mm_to_scene(canvas_w_mm / 2.0, canvas_h_mm / 2.0)

        title = VGroup(
            Text("Spray Impingement", color="#0f172a", font_size=30),
            Text(f"{design_label} / moving piston / Gaussian PDF", color="#3b82f6", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        title.next_to(plot_panel, UP, buff=0.26).align_to(plot_panel, LEFT)

        x_label = Text("radial x [mm]", color=FG_TEXT, font_size=18)
        x_label.next_to(plot_panel, DOWN, buff=0.16)

        y_label = Text("axial y [mm]", color=FG_TEXT, font_size=18)
        y_label.rotate(PI / 2)
        y_label.next_to(plot_panel, LEFT, buff=0.34)

        colorbar = (
            ImageMobject(pdf_colorbar_rgba)
            .stretch_to_fit_width(0.18)
            .stretch_to_fit_height(2.9)
        )
        colorbar.next_to(plot_panel, RIGHT, buff=0.22).align_to(plot_panel, UP).shift(DOWN * 0.55)
        colorbar_border = SurroundingRectangle(colorbar, color=PLOT_EDGE, buff=0.025, stroke_width=1.0)
        colorbar_title = Text("PDF", color=FG_TEXT, font_size=14)
        colorbar_title.next_to(colorbar, UP, buff=0.06)
        colorbar_max = Text(f"{pdf_vmax:.1e}", color=FG_TEXT, font_size=12)
        colorbar_max.next_to(colorbar, RIGHT, buff=0.06).align_to(colorbar, UP)
        colorbar_min = Text(f"{pdf_vmin:.0e}", color=FG_TEXT, font_size=12)
        colorbar_min.next_to(colorbar, RIGHT, buff=0.06).align_to(colorbar, DOWN)

        wall_line = DashedLine(
            mm_to_scene(anim_bore_radius, 0.0),
            mm_to_scene(anim_bore_radius, canvas_h_mm),
            dash_length=0.12,
            color="#fb7185",
            stroke_width=2.0,
        )

        guide_len = 1.25 * float(np.hypot(canvas_w_mm, canvas_h_mm))
        low_segment, high_segment = cone_guide_endpoints(injector_xy, tilt, half_cone_rad, guide_len)
        low_segment = clip_segment_to_canvas(low_segment, canvas_w_mm, canvas_h_mm)
        high_segment = clip_segment_to_canvas(high_segment, canvas_w_mm, canvas_h_mm)
        cone_left = DashedLine(
            mm_to_scene(low_segment[0], low_segment[1]),
            mm_to_scene(low_segment[2], low_segment[3]),
            dash_length=0.10,
            color="#0891b2",
            stroke_width=1.5,
        ).set_opacity(0.80)
        cone_right = DashedLine(
            mm_to_scene(high_segment[0], high_segment[1]),
            mm_to_scene(high_segment[2], high_segment[3]),
            dash_length=0.10,
            color="#0891b2",
            stroke_width=1.5,
        ).set_opacity(0.80)

        injector_glow = Dot(mm_to_scene(*injector_xy), radius=0.16, color="#f59e0b").set_opacity(0.30)
        injector = Dot(mm_to_scene(*injector_xy), radius=0.06, color="#d97706")
        injector_label = Text("injector", color="#d97706", font_size=16)
        injector_label.next_to(injector, UL, buff=0.10)

        frame_tracker = ValueTracker(0.0)

        def current_profile(frame_pos: float) -> tuple[np.ndarray, np.ndarray]:
            _, lo, hi, alpha = split_frame_pos(frame_pos)
            valid = piston_top_valid[lo] | piston_top_valid[hi]
            profile = (1.0 - alpha) * piston_top_y_frames[lo] + alpha * piston_top_y_frames[hi]
            xs = x_axis[valid]
            ys = profile[valid]
            finite = np.isfinite(ys) & (ys <= canvas_h_mm + 1.0)
            return xs[finite], ys[finite]

        def path_points_up_to(frame_pos: float):
            _, lo, hi, alpha = split_frame_pos(frame_pos)
            xs = tip_x_arr[: lo + 1].tolist()
            ys = tip_y_arr[: lo + 1].tolist()
            if hi != lo and alpha > 1e-6:
                xs.append((1.0 - alpha) * float(tip_x_arr[lo]) + alpha * float(tip_x_arr[hi]))
                ys.append((1.0 - alpha) * float(tip_y_arr[lo]) + alpha * float(tip_y_arr[hi]))
            return [mm_to_scene(x_mm, y_mm) for x_mm, y_mm in zip(xs, ys)]

        pdf_image = always_redraw(
            lambda: ImageMobject(blended_rgba(frame_tracker.get_value()))
            .stretch_to_fit_width(axes.x_length)
            .stretch_to_fit_height(axes.y_length)
            .move_to(plot_center)
        )
        piston_fill = always_redraw(
            lambda: self.build_piston_fill(current_profile(frame_tracker.get_value()), mm_to_scene, canvas_h_mm)
        )
        piston_line = always_redraw(
            lambda: self.build_piston_line(current_profile(frame_tracker.get_value()), mm_to_scene)
        )
        tip_path = always_redraw(
            lambda: self.build_tip_path(path_points_up_to(frame_tracker.get_value()))
        )
        sigma_shell_image = always_redraw(
            lambda: ImageMobject(blended_sigma_shell(frame_tracker.get_value()))
            .stretch_to_fit_width(axes.x_length)
            .stretch_to_fit_height(axes.y_length)
            .move_to(plot_center)
        )
        tip_center = always_redraw(
            lambda: self.build_tip_center(
                interp_scalar(tip_x_arr, frame_tracker.get_value()),
                interp_scalar(tip_y_arr, frame_tracker.get_value()),
                mm_to_scene,
            )
        )

        panel_center = RIGHT * 4.85 + UP * 2.0
        panel_box = RoundedRectangle(corner_radius=0.16, width=3.8, height=1.85)
        panel_box.set_fill(PANEL_FILL, opacity=0.92)
        panel_box.set_stroke(PLOT_EDGE, width=1.4, opacity=1.0)
        panel_box.move_to(panel_center)
        row_anchors = [panel_center + UP * (0.64 - row * 0.33) for row in range(5)]
        panel_text_left_x = float(panel_center[0]) - 1.62
        panel_lines = []
        init_rows = [
            ("t   =   0.00 ms", FG_TEXT, 0),
            ("μ   =     0.0 mm", FG_TEXT, 1),
            ("σ∥  =    0.00 mm", FG_TEXT, 2),
            ("σ⊥  =    0.00 mm", FG_TEXT, 3),
            ("P(impact) = 0.000e+00", "#d97706", 4),
        ]
        for text_value, color_value, row in init_rows:
            text_obj = Text(text_value, color=color_value, font_size=20)
            text_obj.align_to(np.array([panel_text_left_x, 0.0, 0.0]), LEFT)
            text_obj.set_y(float(row_anchors[row][1]))
            panel_lines.append(text_obj)

        formatters = [
            lambda fp: f"t   = {interp_scalar(toy_time_ms, fp):6.2f} ms",
            lambda fp: f"μ   = {interp_scalar(mu_plot_arr, fp):6.1f} mm",
            lambda fp: f"σ∥  = {interp_scalar(sigma_axis_arr, fp):6.2f} mm",
            lambda fp: f"σ⊥  = {interp_scalar(sigma_ortho_arr, fp):6.2f} mm",
            lambda fp: f"P(impact) = {interp_scalar(collision_prob, fp):.3e}",
        ]
        panel_colors = [FG_TEXT, FG_TEXT, FG_TEXT, FG_TEXT, "#d97706"]

        def make_line_updater(formatter, color_value, anchor):
            def _updater(mobject):
                mobject.become(Text(formatter(frame_tracker.get_value()), color=color_value, font_size=20))
                mobject.align_to(np.array([panel_text_left_x, 0.0, 0.0]), LEFT)
                mobject.set_y(float(anchor[1]))

            return _updater

        for text_obj, formatter, color_value, anchor in zip(panel_lines, formatters, panel_colors, row_anchors):
            text_obj.add_updater(make_line_updater(formatter, color_value, anchor))

        guide_label = Text(
            f"cone +/- {cone_angle_deg / 2.0:.0f} deg",
            color="#0891b2",
            font_size=16,
        )
        guide_label.to_edge(RIGHT, buff=0.72).shift(DOWN * 2.65)

        legend = VGroup(
            Text("σ-shell: 3σ Gaussian overlay", color="#0284c7", font_size=16),
            Text("PDF: spray distribution", color="#d97706", font_size=16),
            Text("rose: bore wall", color="#fb7185", font_size=16),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        legend.to_edge(RIGHT, buff=0.72).shift(DOWN * 1.25)

        self.play(
            FadeIn(plot_panel),
            Create(axes),
            FadeIn(title),
            FadeIn(x_label),
            FadeIn(y_label),
            FadeIn(colorbar),
            FadeIn(colorbar_border),
            FadeIn(colorbar_title),
            FadeIn(colorbar_max),
            FadeIn(colorbar_min),
            run_time=1.0,
        )
        self.play(
            Create(wall_line),
            Create(cone_left),
            Create(cone_right),
            FadeIn(injector_glow),
            FadeIn(injector),
            FadeIn(injector_label),
            FadeIn(guide_label),
            FadeIn(legend),
            run_time=0.9,
        )
        self.add(
            pdf_image,
            piston_fill,
            sigma_shell_image,
            piston_line,
            tip_path,
            tip_center,
            panel_box,
            *panel_lines,
        )

        run_time = max(5.0, 1.6 * (n_frames - 1) / max(base_fps, 1))
        self.play(frame_tracker.animate.set_value(n_frames - 1), run_time=run_time, rate_func=linear)
        self.wait(0.3)

    @staticmethod
    def build_tip_path(points):
        if len(points) < 2:
            return VGroup()
        path = VMobject()
        path.set_points_as_corners(points)
        path.set_stroke(color="#3b82f6", width=2.0, opacity=0.55)
        return path

    @staticmethod
    def build_tip_center(x_mm: float, y_mm: float, mm_to_scene):
        center = mm_to_scene(x_mm, y_mm)
        glow = Dot(center, radius=0.14, color="#f59e0b").set_opacity(0.30)
        dot = Dot(center, radius=0.065, color="#d97706")
        dot.set_stroke("#ffffff", width=1.5, opacity=1.0)
        return VGroup(glow, dot)

    @staticmethod
    def build_piston_line(profile: tuple[np.ndarray, np.ndarray], mm_to_scene):
        xs, ys = profile
        if len(xs) < 2:
            return VGroup()
        points = [mm_to_scene(x_mm, y_mm) for x_mm, y_mm in zip(xs, ys)]
        glow = VMobject()
        glow.set_points_as_corners(points)
        glow.set_stroke(color="#93c5fd", width=10.0, opacity=0.20)
        line = VMobject()
        line.set_points_as_corners(points)
        line.set_stroke(color="#1e293b", width=4.0, opacity=0.98)
        return VGroup(glow, line)

    @staticmethod
    def build_piston_fill(profile: tuple[np.ndarray, np.ndarray], mm_to_scene, canvas_h_mm: float):
        xs, ys = profile
        if len(xs) < 2:
            return VGroup()
        top_points = [mm_to_scene(x_mm, y_mm) for x_mm, y_mm in zip(xs, ys)]
        polygon = Polygon(
            mm_to_scene(xs[0], canvas_h_mm),
            *top_points,
            mm_to_scene(xs[-1], canvas_h_mm),
        )
        polygon.set_fill("#94a3b8", opacity=0.28)
        polygon.set_stroke(width=0)
        return polygon
