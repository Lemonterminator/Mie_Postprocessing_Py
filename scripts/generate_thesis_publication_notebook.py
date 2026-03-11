from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "examples" / "mie" / "thesis_publication_figures.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def build_notebook():
    cells = [
        md(
            """
            # Thesis Publication Figures

            This notebook prepares publication-style figures for the multihole Mie pipeline and the Stage-1 MLP section.

            Generated figures in this version:

            - `images/fig_mie_preproc_background.png`
            - `images/fig_mie_preproc_sobel.png`
            - `images/fig_angular_occupancy.png`
            - `images/fig_efficient_rotation.png`
            - `images/fig_round1_median_selection.png`
            - `images/fig_round1_loss.png`

            Deferred/manual figures:

            - `fig_gui_main_window` (manual)
            - `fig_gui_point_selection` (manual)
            - `fig_gui_calibration_overlay` (manual)
            - `fig_round2_uncertainty_placeholder` (deferred)
            - `fig_round3_refinement_placeholder` (deferred)
            """
        ),
        md(
            r"""
            ## Source Anchors

            This notebook is anchored to the following dataset:

            - cine: `F:\\LubeOil\\BC20241010_HZ_Nozzle5\\Cine\\T13\\5.cine`
            - config: `F:\\LubeOil\\BC20241010_HZ_Nozzle5\\Cine\\T13\\config.json`

            Pipeline references reused conceptually or directly:

            - `main.py`
            - `mie_multihole_pipeline.py`
            - `OSCC_postprocessing/binary_ops/masking.py`
            - `OSCC_postprocessing/analysis/cone_angle.py`
            - `OSCC_postprocessing/analysis/multihole_utils.py`
            - `OSCC_postprocessing/rotation/rotate_with_alignment_cpu.py`
            - `MLP/median_penetration_MSE.ipynb`
            """
        ),
        md("## 00 Setup"),
        code(
            r"""
            from __future__ import annotations

            import json
            import math
            from pathlib import Path

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from matplotlib.patches import Circle


            def find_project_root(start: Path | None = None) -> Path:
                start = Path.cwd().resolve() if start is None else Path(start).resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                return start


            PROJECT_ROOT = find_project_root()
            IMAGE_DIR = PROJECT_ROOT / "images"
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)

            CINE_PATH = Path(r"F:\LubeOil\BC20241010_HZ_Nozzle5\Cine\T13\5.cine")
            CONFIG_PATH = Path(r"F:\LubeOil\BC20241010_HZ_Nozzle5\Cine\T13\config.json")
            ROUND1_LOSS_CSV = PROJECT_ROOT / "MLP" / "stage1_median_penetration_20260306_130134" / "epoch_loss.csv"

            plt.rcParams.update(
                {
                    "figure.dpi": 140,
                    "savefig.dpi": 300,
                    "font.family": "DejaVu Serif",
                    "axes.titlesize": 12,
                    "axes.labelsize": 10,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "legend.fontsize": 8,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "axes.grid": False,
                    "figure.facecolor": "white",
                    "axes.facecolor": "white",
                }
            )

            COLORS = {
                "ink": "#16213e",
                "accent": "#0f766e",
                "warm": "#c2410c",
                "gold": "#b45309",
                "red": "#b91c1c",
                "blue": "#1d4ed8",
                "gray": "#6b7280",
            }

            print(f"PROJECT_ROOT = {PROJECT_ROOT}")
            print(f"IMAGE_DIR     = {IMAGE_DIR}")
            print(f"CINE_PATH     = {CINE_PATH}")
            print(f"CONFIG_PATH   = {CONFIG_PATH}")
            """
        ),
        md("## 01 Dataset Anchor"),
        code(
            """
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                CONFIG = json.load(f)

            MIE_GEOMETRY = {
                "plumes": int(CONFIG["plumes"]),
                "offset": float(CONFIG["offset"]),
                "centre": (float(CONFIG["centre_x"]), float(CONFIG["centre_y"])),
                "inner_radius": float(CONFIG["inner_radius"]),
                "outer_radius": float(CONFIG["outer_radius"]),
            }

            print(json.dumps(MIE_GEOMETRY, indent=2))
            """
        ),
        md("## 02 Shared Helpers"),
        code(
            """
            OPTIONAL_IMPORTS: dict[str, str] = {}


            def optional_import(name: str):
                try:
                    module = __import__(name, fromlist=["*"])
                    OPTIONAL_IMPORTS[name] = "OK"
                    return module
                except Exception as exc:
                    OPTIONAL_IMPORTS[name] = f"FAILED: {type(exc).__name__}: {exc}"
                    return None


            repo_masking = optional_import("OSCC_postprocessing.binary_ops.masking")
            repo_cone = optional_import("OSCC_postprocessing.analysis.cone_angle")
            repo_multi = optional_import("OSCC_postprocessing.analysis.multihole_utils")
            repo_rotate = optional_import("OSCC_postprocessing.rotation.rotate_with_alignment_cpu")
            repo_hysteresis = optional_import("OSCC_postprocessing.analysis.hysteresis")
            repo_cine = optional_import("OSCC_postprocessing.cine.functions_videos")

            OPTIONAL_IMPORTS
            """
        ),
        code(
            """
            def add_panel_labels(axs, labels=None, x=0.01, y=0.99):
                labels = labels or list("abcdefghijklmnopqrstuvwxyz")
                for ax, label in zip(np.ravel(axs), labels):
                    ax.text(
                        x,
                        y,
                        f"({label})",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=11,
                        fontweight="bold",
                        color=COLORS["ink"],
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
                    )


            def export_figure(fig, filename: str):
                out_path = IMAGE_DIR / filename
                fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
                print(f"Saved {out_path}")
                return out_path


            def to_numpy(arr):
                if hasattr(arr, "get"):
                    arr = arr.get()
                elif hasattr(arr, "__cuda_array_interface__"):
                    try:
                        import cupy as cp
                        arr = cp.asnumpy(arr)
                    except Exception:
                        pass
                return np.asarray(arr)


            def robust_scale_local(arr, q_min=5.0, q_max=99.5, clip=True, eps=1e-8):
                arr = to_numpy(arr).astype(float, copy=False)
                lo = np.nanpercentile(arr, q_min)
                hi = np.nanpercentile(arr, q_max)
                scaled = (arr - lo) / max(hi - lo, eps)
                if clip:
                    scaled = np.clip(scaled, 0.0, 1.0)
                return scaled


            def angular_distance_deg(theta_deg, ref_deg):
                delta = (theta_deg - ref_deg + 180.0) % 360.0 - 180.0
                return delta


            def generate_ring_mask_local(height, width, centre, inner_radius, outer_radius):
                yy, xx = np.indices((height, width))
                dx = xx - centre[0]
                dy = yy - centre[1]
                rr2 = dx * dx + dy * dy
                return (rr2 >= inner_radius ** 2) & (rr2 <= outer_radius ** 2)


            def generate_plume_mask_local(width, height, angle_deg=None, x0=0.0, y0=None):
                y0 = height / 2.0 if y0 is None else y0
                yy, xx = np.indices((height, width), dtype=float)
                if angle_deg is None:
                    return xx >= x0
                half_angle = np.deg2rad(angle_deg / 2.0)
                top = -width * np.tan(half_angle) + height / 2.0
                bottom = width * np.tan(half_angle) + height / 2.0
                slope_top = (top - y0) / max(width - x0, 1e-6)
                slope_bottom = (bottom - y0) / max(width - x0, 1e-6)
                left_region = xx <= x0
                between = (yy >= (y0 + slope_top * (xx - x0))) & (yy <= (y0 + slope_bottom * (xx - x0)))
                return left_region | between


            generate_ring_mask = generate_ring_mask_local
            generate_plume_mask = generate_plume_mask_local


            def angle_signal_density_local(video, x0, y0, n_bins=360, N_bins=None):
                if N_bins is not None:
                    n_bins = N_bins
                arr = to_numpy(video)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                frames, height, width = arr.shape
                yy, xx = np.indices((height, width))
                theta = np.degrees(np.arctan2(yy - y0, xx - x0)) % 360.0
                edges = np.linspace(0.0, 360.0, n_bins + 1)
                inds = np.digitize(theta.ravel(), edges) - 1
                inds = np.clip(inds, 0, n_bins - 1)
                counts = np.bincount(inds, minlength=n_bins)
                signal = np.empty((frames, n_bins), dtype=float)
                flat = arr.reshape(frames, -1)
                for frame_idx in range(frames):
                    signal[frame_idx] = np.bincount(inds, weights=flat[frame_idx], minlength=n_bins)
                density = signal / np.maximum(counts, 1)[None, :]
                centers = edges[:-1] + 0.5 * (edges[1] - edges[0])
                return centers, signal, density


            angle_signal_density_auto = angle_signal_density_local


            def estimate_offset_from_fft_local(signal, number_of_plumes):
                signal = to_numpy(signal).astype(float, copy=False)
                summed_signal = signal.sum(axis=0)
                fft_vals = np.fft.rfft(summed_signal)
                if number_of_plumes >= len(fft_vals):
                    return 0.0
                phase = np.angle(fft_vals[number_of_plumes])
                offset = (-phase / number_of_plumes) * 180.0 / np.pi
                offset %= 360.0
                return min(offset, offset - 360.0, key=abs)


            estimate_offset_from_fft = estimate_offset_from_fft_local


            def fill_short_false_runs_local(mask, max_len=3):
                mask = to_numpy(mask).astype(bool, copy=True)
                n = mask.size
                if n == 0 or mask.all() or (~mask).all():
                    return mask
                doubled = np.concatenate([mask, mask])
                idx = 0
                while idx < 2 * n:
                    if doubled[idx]:
                        idx += 1
                        continue
                    start = idx
                    while idx < 2 * n and not doubled[idx]:
                        idx += 1
                    stop = idx
                    run_len = stop - start
                    left_true = doubled[start - 1] if start > 0 else doubled[-1]
                    right_true = doubled[stop] if stop < 2 * n else doubled[0]
                    if left_true and right_true and run_len <= max_len:
                        doubled[start:stop] = True
                return doubled[:n]


            fill_short_false_runs = fill_short_false_runs_local


            def safe_log_subtracted_foreground(video, frames_before_soi=10, noise_floor_multiplier=2.5):
                video = to_numpy(video).astype(float, copy=False)
                frames_before_soi = max(1, min(frames_before_soi, len(video)))
                eps = 1e-9
                log_video = np.log(video + eps)
                log_background = np.median(log_video[:frames_before_soi], axis=0, keepdims=True)
                background = np.exp(log_background[0])
                dff = np.expm1(log_video - log_background)
                noise_mask = video > (background[None, ...] * noise_floor_multiplier)
                dff *= noise_mask
                baseline = np.median(dff[:frames_before_soi], axis=0, keepdims=True)
                dff -= baseline
                foreground = np.maximum(dff - 0.05, 0.0)
                return background, robust_scale_local(foreground, q_min=5.0, q_max=99.99)


            def sobel_magnitude_video_local(video):
                video = np.asarray(video, dtype=float)
                grad_y = np.gradient(video, axis=1)
                grad_x = np.gradient(video, axis=2)
                return np.sqrt(grad_x ** 2 + grad_y ** 2)


            def occupancy_mask_from_profile(profile, max_gap=3):
                profile = robust_scale_local(profile, q_min=2.0, q_max=99.8)
                positive = profile[profile > 0]
                if positive.size == 0:
                    return np.zeros_like(profile, dtype=bool)
                threshold = np.clip(np.median(positive) + 0.25 * np.std(positive), 0.2, 0.75)
                mask = profile >= threshold
                return fill_short_false_runs(mask, max_len=max_gap)
            """
        ),
        code(
            """
            def synthetic_frame_shape(centre, outer_radius, margin=56):
                max_extent = int(np.ceil(max(centre[0], centre[1], outer_radius) * 2 + margin))
                return max_extent, max_extent


            def make_synthetic_mie_video(config, n_frames=24, seed=7):
                rng = np.random.default_rng(seed)
                centre = config["centre"]
                inner_radius = config["inner_radius"]
                outer_radius = config["outer_radius"]
                plumes = config["plumes"]
                offset = config["offset"]
                height, width = synthetic_frame_shape(centre, outer_radius)
                yy, xx = np.indices((height, width), dtype=float)
                dx = xx - centre[0]
                dy = yy - centre[1]
                rr = np.sqrt(dx * dx + dy * dy)
                theta = np.degrees(np.arctan2(dy, dx)) % 360.0
                radial_background = 0.10 + 0.10 * (rr / max(outer_radius, 1.0))
                illumination = 0.08 * (xx / max(width - 1, 1)) + 0.04 * (yy / max(height - 1, 1))
                ring_texture = 0.03 * np.cos((rr - inner_radius) / max(outer_radius, 1.0) * np.pi * 4)
                base = radial_background + illumination + ring_texture
                angles = np.linspace(0.0, 360.0, plumes, endpoint=False) - offset
                sector_width = 0.48 * (360.0 / plumes)
                frames = []
                for frame_idx in range(n_frames):
                    frame = base.copy()
                    progress = np.clip((frame_idx - 2) / max(n_frames - 6, 1), 0.0, 1.0)
                    for plume_idx, angle in enumerate(angles):
                        plume_phase = 0.25 * plume_idx
                        front = inner_radius + 18.0 + progress * (outer_radius - inner_radius - 28.0) * (0.82 + 0.08 * np.sin(plume_phase))
                        softness = 8.0 + 1.2 * plume_idx / max(plumes - 1, 1)
                        radial_gate = 1.0 / (1.0 + np.exp((rr - front) / softness))
                        radial_gate *= rr >= inner_radius
                        angle_gate = np.exp(-(angular_distance_deg(theta, angle) / sector_width) ** 2)
                        filament = 0.05 * np.sin(0.05 * rr + 0.4 * frame_idx + plume_idx)
                        plume = (0.22 + 0.04 * np.cos(0.35 * frame_idx + plume_idx)) * angle_gate * radial_gate
                        plume *= 1.0 + filament
                        frame += plume
                    frame += 0.01 * rng.standard_normal(frame.shape)
                    frame = np.clip(frame, 0.0, None)
                    frames.append(frame)
                video = np.stack(frames, axis=0)
                video = robust_scale_local(video, q_min=1.0, q_max=99.8)
                return video.astype(np.float32)


            def try_load_cine_video_sample(cine_path: Path, frame_limit=24):
                if repo_cine is None or not hasattr(repo_cine, "load_cine_video"):
                    return None, "cine loader unavailable in current kernel"
                try:
                    video = repo_cine.load_cine_video(cine_path, frame_limit=frame_limit)
                    video = to_numpy(video).astype(np.float32, copy=False)
                    if video.size == 0:
                        return None, "cine loader returned an empty array"
                    vmax = float(np.nanmax(video))
                    if vmax > 0:
                        video = video / vmax
                    return video, f"real cine sample loaded ({video.shape[0]} frames)"
                except Exception as exc:
                    return None, f"cine loading failed: {type(exc).__name__}: {exc}"


            def build_mie_bundle(config, cine_path, frame_limit=24, frames_before_soi=10, noise_floor_multiplier=2.5, bins=720, seed=7):
                video_real, note = try_load_cine_video_sample(cine_path, frame_limit=frame_limit)
                if video_real is None:
                    video = make_synthetic_mie_video(config, n_frames=frame_limit, seed=seed)
                    source = "synthetic"
                    source_note = f"Synthetic fallback in use. Reason: {note}."
                else:
                    video = video_real
                    source = "real"
                    source_note = f"Real cine reference in use. {note}."

                height, width = video.shape[1:3]
                ring_mask = generate_ring_mask(height, width, config["centre"], config["inner_radius"], config["outer_radius"])
                background, foreground = safe_log_subtracted_foreground(video, frames_before_soi=frames_before_soi, noise_floor_multiplier=noise_floor_multiplier)
                foreground = foreground * ring_mask[None, :, :]
                sobel_mag = sobel_magnitude_video_local(foreground)
                highpass = robust_scale_local(sobel_mag, q_min=5.0, q_max=99.95) * ring_mask[None, :, :]
                rep_idx = int(np.argmax(highpass.sum(axis=(1, 2))))

                angle_bins, angle_signal, _ = angle_signal_density_auto(foreground, config["centre"][0], config["centre"][1], N_bins=bins)
                angle_signal = to_numpy(angle_signal).astype(float, copy=False)
                summed_profile = angle_signal.sum(axis=0)
                fft_offset = float(estimate_offset_from_fft(angle_signal, config["plumes"]))
                occupancy_mask = occupancy_mask_from_profile(summed_profile, max_gap=3)
                plume_angles = np.linspace(0.0, 360.0, config["plumes"], endpoint=False) - fft_offset

                frame_for_rotation = highpass[rep_idx]
                out_shape = (int(config["outer_radius"] // 2), int(config["outer_radius"]))
                reference_angle = float(plume_angles[0])
                if repo_rotate is not None and hasattr(repo_rotate, "rotate_video_nozzle_at_0_half_numpy") and hasattr(repo_rotate, "build_nozzle_rotation_maps"):
                    rotated_strip, mapx, mapy = repo_rotate.rotate_video_nozzle_at_0_half_numpy(
                        np.asarray([frame_for_rotation], dtype=np.float32),
                        config["centre"],
                        reference_angle,
                        interpolation="bilinear",
                        out_shape=out_shape,
                        border_mode="constant",
                        cval=0.0,
                        stack=True,
                        plot_maps=False,
                    )
                    rotated_strip = to_numpy(rotated_strip[0]).astype(float, copy=False)
                    mapx, mapy = repo_rotate.build_nozzle_rotation_maps(
                        frame_for_rotation.shape,
                        config["centre"],
                        -reference_angle,
                        out_shape=out_shape,
                        calibration_point=(0.0, frame_for_rotation.shape[0] / 2.0),
                        keep_left_edge=True,
                    )
                else:
                    out_h, out_w = out_shape
                    xs = np.linspace(config["centre"][0], min(width - 1, config["centre"][0] + out_w), out_w)
                    ys = np.linspace(max(0, config["centre"][1] - out_h / 2.0), min(height - 1, config["centre"][1] + out_h / 2.0), out_h)
                    mapx, mapy = np.meshgrid(xs, ys)
                    xi = np.clip(np.rint(mapx).astype(int), 0, width - 1)
                    yi = np.clip(np.rint(mapy).astype(int), 0, height - 1)
                    rotated_strip = frame_for_rotation[yi, xi]

                plume_mask = generate_plume_mask(out_shape[1], out_shape[0], 360.0 / config["plumes"], int(config["inner_radius"]))

                return {
                    "source": source,
                    "source_note": source_note,
                    "video": video,
                    "ring_mask": ring_mask,
                    "background": background,
                    "foreground": foreground,
                    "sobel_mag": sobel_mag,
                    "highpass": highpass,
                    "rep_idx": rep_idx,
                    "angle_bins": np.asarray(angle_bins, dtype=float),
                    "angle_signal": angle_signal,
                    "summed_profile": summed_profile,
                    "fft_offset": fft_offset,
                    "occupancy_mask": occupancy_mask,
                    "plume_angles": plume_angles,
                    "mapx": to_numpy(mapx).astype(float, copy=False),
                    "mapy": to_numpy(mapy).astype(float, copy=False),
                    "rotated_strip": to_numpy(rotated_strip).astype(float, copy=False),
                    "plume_mask": to_numpy(plume_mask).astype(bool, copy=False),
                    "reference_angle": reference_angle,
                }


            def draw_geometry_overlay(ax, config, color=COLORS["warm"]):
                center = config["centre"]
                inner = Circle(center, radius=config["inner_radius"], fill=False, linewidth=1.2, edgecolor=color, alpha=0.9)
                outer = Circle(center, radius=config["outer_radius"], fill=False, linewidth=1.4, edgecolor=color, alpha=0.9)
                ax.add_patch(inner)
                ax.add_patch(outer)
                ax.plot(center[0], center[1], marker="+", color=COLORS["red"], markersize=8, mew=1.4)
            """
        ),
        code(
            """
            MIE_BASE_PARAMS = {
                "frame_limit": 24,
                "frames_before_soi": 10,
                "noise_floor_multiplier": 2.5,
                "bins": 720,
                "seed": 7,
            }

            mie_bundle = build_mie_bundle(MIE_GEOMETRY, CINE_PATH, **MIE_BASE_PARAMS)
            print(mie_bundle["source_note"])
            print(f"Representative frame index: {mie_bundle['rep_idx']}")
            """
        ),
        md("## 03 Mie Figures"),
        md("### fig_mie_preproc_background_placeholder"),
        code(
            """
            FIG_BG_PARAMS = {
                "filename": "fig_mie_preproc_background.png",
                "frame_idx": mie_bundle["rep_idx"],
            }
            FIG_BG_PARAMS
            """
        ),
        code(
            """
            fig_bg_data = {
                "raw": mie_bundle["video"][FIG_BG_PARAMS["frame_idx"]],
                "background": mie_bundle["background"],
                "foreground": mie_bundle["foreground"][FIG_BG_PARAMS["frame_idx"]],
                "source_note": mie_bundle["source_note"],
            }
            """
        ),
        code(
            """
            fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
            im0 = axs[0].imshow(fig_bg_data["raw"], cmap="gray")
            axs[0].set_title("Raw frame")
            draw_geometry_overlay(axs[0], MIE_GEOMETRY)
            axs[0].set_axis_off()
            im1 = axs[1].imshow(fig_bg_data["background"], cmap="gray")
            axs[1].set_title("Estimated background")
            axs[1].set_axis_off()
            im2 = axs[2].imshow(fig_bg_data["foreground"], cmap="magma")
            axs[2].set_title("Log-subtracted foreground")
            axs[2].set_axis_off()
            for ax, im in zip(axs, (im0, im1, im2)):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.suptitle(f"Mie preprocessing background removal ({mie_bundle['source']})", y=1.03)
            add_panel_labels(axs, labels=["a", "b", "c"])
            plt.show()
            """
        ),
        code(
            """
            export_figure(fig, FIG_BG_PARAMS["filename"])
            plt.close(fig)
            """
        ),
        md("### fig_mie_preproc_sobel_placeholder"),
        code(
            """
            FIG_SOBEL_PARAMS = {
                "filename": "fig_mie_preproc_sobel.png",
                "frame_idx": mie_bundle["rep_idx"],
            }
            FIG_SOBEL_PARAMS
            """
        ),
        code(
            """
            fig_sobel_data = {
                "foreground": mie_bundle["foreground"][FIG_SOBEL_PARAMS["frame_idx"]],
                "sobel": mie_bundle["sobel_mag"][FIG_SOBEL_PARAMS["frame_idx"]],
                "highpass": mie_bundle["highpass"][FIG_SOBEL_PARAMS["frame_idx"]],
            }
            """
        ),
        code(
            """
            fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
            im0 = axs[0].imshow(fig_sobel_data["foreground"], cmap="magma")
            axs[0].set_title("Foreground")
            axs[0].set_axis_off()
            im1 = axs[1].imshow(fig_sobel_data["sobel"], cmap="cividis")
            axs[1].set_title("Sobel magnitude")
            axs[1].set_axis_off()
            im2 = axs[2].imshow(fig_sobel_data["highpass"], cmap="inferno")
            axs[2].set_title("Robust-scaled high-pass")
            axs[2].set_axis_off()
            for ax, im in zip(axs, (im0, im1, im2)):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.suptitle("Edge-enhancing preprocessing for multihole Mie segmentation", y=1.03)
            add_panel_labels(axs, labels=["a", "b", "c"])
            plt.show()
            """
        ),
        code(
            """
            export_figure(fig, FIG_SOBEL_PARAMS["filename"])
            plt.close(fig)
            """
        ),
        md("### fig_angular_occupancy_placeholder"),
        code(
            """
            FIG_ANGULAR_PARAMS = {
                "filename": "fig_angular_occupancy.png",
                "max_harmonic": 40,
            }
            FIG_ANGULAR_PARAMS
            """
        ),
        code(
            """
            fft_vals = np.fft.rfft(mie_bundle["summed_profile"])
            fft_amp = np.abs(fft_vals)
            harmonics = np.arange(len(fft_amp))
            fig_angular_data = {
                "profiles": mie_bundle["angle_signal"],
                "bins": mie_bundle["angle_bins"],
                "summed_profile": mie_bundle["summed_profile"],
                "fft_amp": fft_amp,
                "harmonics": harmonics,
                "offset": mie_bundle["fft_offset"],
                "occupancy_mask": mie_bundle["occupancy_mask"],
                "plume_angles": mie_bundle["plume_angles"],
            }
            """
        ),
        code(
            """
            fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
            im = axs[0, 0].imshow(robust_scale_local(fig_angular_data["profiles"], q_min=2.0, q_max=99.8), aspect="auto", origin="lower", cmap="viridis", extent=[0.0, 360.0, 0, fig_angular_data["profiles"].shape[0]])
            axs[0, 0].set_title("Per-frame angular profiles")
            axs[0, 0].set_xlabel("Angle (deg)")
            axs[0, 0].set_ylabel("Frame index")
            plt.colorbar(im, ax=axs[0, 0], fraction=0.046, pad=0.02)
            axs[0, 1].plot(fig_angular_data["bins"], robust_scale_local(fig_angular_data["summed_profile"]), color=COLORS["ink"], lw=2.0)
            for angle in fig_angular_data["plume_angles"]:
                axs[0, 1].axvline(angle % 360.0, color=COLORS["warm"], lw=0.8, alpha=0.35)
            axs[0, 1].set_xlim(0.0, 360.0)
            axs[0, 1].set_title("Summed profile with plume guide angles")
            axs[0, 1].set_xlabel("Angle (deg)")
            axs[0, 1].set_ylabel("Normalized intensity")
            max_harmonic = min(FIG_ANGULAR_PARAMS["max_harmonic"], len(fig_angular_data["fft_amp"]) - 1)
            axs[1, 0].plot(fig_angular_data["harmonics"][1:max_harmonic + 1], fig_angular_data["fft_amp"][1:max_harmonic + 1], color=COLORS["accent"], lw=2.0)
            axs[1, 0].axvline(MIE_GEOMETRY["plumes"], color=COLORS["red"], lw=1.2, ls="--")
            axs[1, 0].annotate(f"offset = {fig_angular_data['offset']:.2f} deg", xy=(MIE_GEOMETRY["plumes"], fig_angular_data["fft_amp"][MIE_GEOMETRY["plumes"]]), xytext=(MIE_GEOMETRY["plumes"] + 2.0, 0.85 * fig_angular_data["fft_amp"][1:max_harmonic + 1].max()), arrowprops={"arrowstyle": "->", "color": COLORS["red"]}, fontsize=9)
            axs[1, 0].set_title("FFT plume harmonic and inferred phase offset")
            axs[1, 0].set_xlabel("Harmonic index")
            axs[1, 0].set_ylabel("Amplitude")
            axs[1, 1].imshow(fig_angular_data["occupancy_mask"][None, :], aspect="auto", cmap="gray_r", extent=[0.0, 360.0, 0.0, 1.0])
            axs[1, 1].set_title("Occupied-angle mask")
            axs[1, 1].set_xlabel("Angle (deg)")
            axs[1, 1].set_yticks([])
            fig.suptitle("Angular occupancy analysis used to align multihole plumes", y=1.01)
            add_panel_labels(axs, labels=["a", "b", "c", "d"])
            plt.show()
            """
        ),
        code(
            """
            export_figure(fig, FIG_ANGULAR_PARAMS["filename"])
            plt.close(fig)
            """
        ),
        md("### fig_efficient_rotation_placeholder"),
        code(
            """
            FIG_ROTATION_PARAMS = {
                "filename": "fig_efficient_rotation.png",
            }
            FIG_ROTATION_PARAMS
            """
        ),
        code(
            """
            fig_rotation_data = {
                "mapx": mie_bundle["mapx"],
                "mapy": mie_bundle["mapy"],
                "rotated_strip": mie_bundle["rotated_strip"],
                "plume_mask": mie_bundle["plume_mask"],
                "reference_angle": mie_bundle["reference_angle"],
            }
            """
        ),
        code(
            """
            fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
            im0 = axs[0, 0].imshow(fig_rotation_data["mapx"], cmap="viridis")
            axs[0, 0].set_title("Inverse affine mapx")
            axs[0, 0].set_axis_off()
            plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.02)
            im1 = axs[0, 1].imshow(fig_rotation_data["mapy"], cmap="magma")
            axs[0, 1].set_title("Inverse affine mapy")
            axs[0, 1].set_axis_off()
            plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.02)
            im2 = axs[1, 0].imshow(fig_rotation_data["rotated_strip"], cmap="inferno", aspect="auto")
            axs[1, 0].set_title(f"Rotated strip ROI at {fig_rotation_data['reference_angle']:.2f} deg")
            axs[1, 0].set_xlabel("Downstream distance (px)")
            axs[1, 0].set_ylabel("Cross-stream coordinate (px)")
            plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.02)
            axs[1, 1].imshow(fig_rotation_data["plume_mask"], cmap="gray_r", aspect="auto")
            axs[1, 1].set_title("Triangular plume mask")
            axs[1, 1].set_xlabel("Downstream distance (px)")
            axs[1, 1].set_ylabel("Cross-stream coordinate (px)")
            fig.suptitle("Efficient rotation and strip extraction for one plume sector", y=1.01)
            add_panel_labels(axs, labels=["a", "b", "c", "d"])
            plt.show()
            """
        ),
        code(
            """
            export_figure(fig, FIG_ROTATION_PARAMS["filename"])
            plt.close(fig)
            """
        ),
        md("## 04 Stage-1 MLP Figures"),
        md("### fig_round1_median_selection_placeholder"),
        code(
            """
            FIG_ROUND1_SELECTION_PARAMS = {
                "filename": "fig_round1_median_selection.png",
                "comparison_time_s": 5e-3,
                "n_conditions": 3,
                "samples_per_condition": 7,
            }
            FIG_ROUND1_SELECTION_PARAMS
            """
        ),
        code(
            """
            def spray_penetration_model_sigmoid(params, t):
                log_k_sqrt, log_k_quarter, log_t0, log_s = params
                k_sqrt = np.exp(log_k_sqrt)
                k_quarter = np.exp(log_k_quarter)
                t0 = np.exp(log_t0)
                s = np.exp(log_s)
                t = np.clip(np.asarray(t, dtype=float), 1e-9, None)
                sqrt_segment = k_sqrt * np.sqrt(t)
                quarter_segment = k_quarter * np.power(t, 0.25)
                blend = 1.0 / (1.0 + np.exp(-(t - t0) / s))
                return (1.0 - blend) * sqrt_segment + blend * quarter_segment


            def select_row_closest_to_median_penetration(rows, compare_time_s):
                compare_vals = np.array([spray_penetration_model_sigmoid(row["params"], compare_time_s) for row in rows], dtype=float)
                median_val = float(np.nanmedian(compare_vals))
                selected_idx = int(np.nanargmin(np.abs(compare_vals - median_val)))
                return selected_idx, median_val, compare_vals


            time_grid = np.linspace(0.0, 8e-3, 240)
            condition_names = ["A: low chamber p", "B: medium chamber p", "C: high chamber p"]
            condition_rows = []
            base_param_sets = [
                np.log([95.0, 62.0, 2.5e-3, 0.65e-3]),
                np.log([118.0, 74.0, 2.1e-3, 0.58e-3]),
                np.log([136.0, 87.0, 1.8e-3, 0.52e-3]),
            ]

            for condition_name, base_params in zip(condition_names, base_param_sets):
                rows = []
                for sample_idx in range(FIG_ROUND1_SELECTION_PARAMS["samples_per_condition"]):
                    perturb = np.array([0.06 * (sample_idx - 3), -0.03 * math.cos(sample_idx), 0.05 * math.sin(sample_idx / 2.0), -0.04 * math.cos(sample_idx / 3.0)])
                    params = base_params + perturb
                    rows.append({"condition": condition_name, "sample_idx": sample_idx, "params": params, "curve": spray_penetration_model_sigmoid(params, time_grid)})
                selected_idx, median_val, compare_vals = select_row_closest_to_median_penetration(rows, FIG_ROUND1_SELECTION_PARAMS["comparison_time_s"])
                for row, compare_val in zip(rows, compare_vals):
                    row["compare_val"] = float(compare_val)
                condition_rows.append({"condition": condition_name, "rows": rows, "selected_idx": selected_idx, "median_val": median_val})
            """
        ),
        code(
            """
            fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.5), constrained_layout=True)
            condition_palette = [COLORS["blue"], COLORS["accent"], COLORS["warm"]]
            compare_time_ms = 1e3 * FIG_ROUND1_SELECTION_PARAMS["comparison_time_s"]
            for cond_data, color in zip(condition_rows, condition_palette):
                for row in cond_data["rows"]:
                    alpha = 0.20
                    lw = 1.1
                    zorder = 1
                    if row["sample_idx"] == cond_data["selected_idx"]:
                        alpha = 1.0
                        lw = 2.4
                        zorder = 4
                    axs[0].plot(1e3 * time_grid, row["curve"], color=color, alpha=alpha, lw=lw, zorder=zorder)
                selected_row = cond_data["rows"][cond_data["selected_idx"]]
                axs[0].scatter(compare_time_ms, selected_row["compare_val"], color=color, edgecolor="white", s=44, zorder=5)
            axs[0].axvline(compare_time_ms, color=COLORS["gray"], ls="--", lw=1.0)
            axs[0].set_title("Toy trajectories grouped by condition")
            axs[0].set_xlabel("Time (ms)")
            axs[0].set_ylabel("Penetration (a.u.)")
            for cond_y, (cond_data, color) in enumerate(zip(condition_rows, condition_palette), start=1):
                vals = [row["compare_val"] for row in cond_data["rows"]]
                axs[1].scatter(vals, np.full(len(vals), cond_y), color=color, alpha=0.30, s=38)
                axs[1].axvline(cond_data["median_val"], ymin=(cond_y - 1) / 3.2, ymax=cond_y / 3.2, color=color, lw=2.2)
                selected_val = cond_data["rows"][cond_data["selected_idx"]]["compare_val"]
                axs[1].scatter(selected_val, cond_y, color=color, edgecolor="black", s=72, zorder=5)
            axs[1].set_yticks([1, 2, 3], labels=condition_names)
            axs[1].set_title("Median-at-5 ms representative selection")
            axs[1].set_xlabel("Penetration at 5 ms (a.u.)")
            axs[1].set_xlim(left=0)
            fig.suptitle("Stage-1 median representative trajectory selection", y=1.04)
            add_panel_labels(axs, labels=["a", "b"])
            plt.show()
            """
        ),
        code(
            """
            export_figure(fig, FIG_ROUND1_SELECTION_PARAMS["filename"])
            plt.close(fig)
            """
        ),
        md("### fig_round1_loss_placeholder"),
        code(
            """
            FIG_ROUND1_LOSS_PARAMS = {
                "filename": "fig_round1_loss.png",
                "csv_path": ROUND1_LOSS_CSV,
            }
            FIG_ROUND1_LOSS_PARAMS
            """
        ),
        code(
            """
            df_loss = pd.read_csv(FIG_ROUND1_LOSS_PARAMS["csv_path"])
            df_epoch = df_loss.copy()
            df_epoch.head()
            """
        ),
        code(
            """
            fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
            for split, linestyle in [("train", "-"), ("val", "--")]:
                sel = df_epoch[df_epoch["split"] == split]
                axs[0].plot(sel["epoch"], sel["loss"], color=COLORS["ink"], lw=2.0, ls=linestyle, label=f"{split} total")
                axs[0].plot(sel["epoch"], sel["mse"], color=COLORS["accent"], lw=1.8, ls=linestyle, label=f"{split} mse")
                axs[1].plot(sel["epoch"], sel["d1_penalty"], color=COLORS["warm"], lw=1.8, ls=linestyle, label=f"{split} d1")
                axs[1].plot(sel["epoch"], sel["d2_penalty"], color=COLORS["red"], lw=1.8, ls=linestyle, label=f"{split} d2")
            axs[0].set_title("Total loss and MSE")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].legend(frameon=False, ncols=2)
            axs[1].set_title("Derivative regularization penalties")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Penalty")
            axs[1].legend(frameon=False, ncols=2)
            for ax in axs:
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
                ax.grid(True, alpha=0.20)
            fig.suptitle("Stage-1 training history from epoch_loss.csv", y=1.03)
            add_panel_labels(axs, labels=["a", "b"])
            plt.show()
            """
        ),
        code(
            """
            export_figure(fig, FIG_ROUND1_LOSS_PARAMS["filename"])
            plt.close(fig)
            """
        ),
        md(
            """
            ## 05 Manual and Deferred Figures

            Manual GUI figures:

            - `images/fig_gui_main_window.png`
            - `images/fig_gui_point_selection.png`
            - `images/fig_gui_calibration_overlay.png`

            Deferred deep-learning figures for later rounds:

            - `images/fig_round2_uncertainty.png`
            - `images/fig_round3_refinement.png`

            Current notebook behavior:

            - uses the `T13/config.json` geometry directly for all Mie overlays
            - prefers one real cine sample when the kernel can load it
            - falls back to a geometry-matched synthetic video when cine loading is unavailable
            """
        ),
    ]

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    }
    return nb


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        nbf.write(build_notebook(), f)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
