"""One-click regeneration of every Ch.04/05 thesis figure from current (sanitized) data.

Motivation
----------
Confidential campaign names (``BC########_HZ_*``) must not appear in any thesis
figure. ``MLP/synthetic_data`` (and its QC-gated variant) have already been
sanitized to ``Nozzle0..8``, but many figures were rendered from *snapshot*
CSV/JSON inputs (eval dirs, Thesis/generated tables) that still carry the old
names. This script:

1. **Sanitize** — strips the ``BC########_HZ_`` prefix in-place from all text
   inputs (CSV/JSON/MD/TEX) that feed the figure generators, so label/join keys
   stay consistent.
2. **Regenerate** — re-runs every figure generator mapped in
   ``Thesis/figure_provenance_ch04_05.md``.
3. **Copy/rename** — places outputs at the exact paths referenced by the
   ``.tex`` sources (some figures are renamed on promotion).
4. **Scan** — reports any file that still contains the confidential pattern,
   plus the manual follow-ups this script cannot automate.

Usage (from the repo root, ALWAYS with the project venv — system python has no torch):

    .venv\\Scripts\\python Thesis\\regenerate_thesis_figures.py              # everything
    .venv\\Scripts\\python Thesis\\regenerate_thesis_figures.py --list       # show steps
    .venv\\Scripts\\python Thesis\\regenerate_thesis_figures.py --only lono_figures hermite
    .venv\\Scripts\\python Thesis\\regenerate_thesis_figures.py --skip censor_library
    .venv\\Scripts\\python Thesis\\regenerate_thesis_figures.py --scan-only  # just report leaks
    .venv\\Scripts\\python Thesis\\regenerate_thesis_figures.py --dry-run

Steps run independently; a failure is logged and the run continues. Per-step
logs land in ``Thesis/regen_logs/<timestamp>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Windows consoles often default to cp1252, which cannot print the box-drawing
# characters used in the progress output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS = PROJECT_ROOT / "Thesis"
IMAGES = THESIS / "images"
GEN_CURRENT = THESIS / "generated" / "current"
SLIDES_FH = THESIS / "slides" / "slides_residual_family_head_production"
RESIDUAL_IMG = IMAGES / "residual_family_head_20260608"
NN_FIT_IMG = IMAGES / "neural_network_fit_results"
NN_FIT_2052_IMG = IMAGES / "neural_network_fit_results_20260521"
EVAL_20260709_IMG = IMAGES / "eval_20260709"
PY = sys.executable

CONFIDENTIAL_RE = re.compile(r"BC\d{8}_HZ_")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}
MAX_SANITIZE_BYTES = 1_500_000_000  # safety valve; points.csv is ~91 MB

# Text inputs that feed the figure generators below. Snapshot eval dirs are
# included so axis labels / join keys (folder, experiment_name) match the
# sanitized synthetic_data naming. Globs are relative to the repo root.
SANITIZE_GLOBS = [
    "Thesis/generated/**/*",
    "Thesis/images/**/*.csv",
    "Thesis/slides/slides_residual_family_head_production/**/*",
    # make_figures (neural_network_fit_results_20260521) eval snapshot
    "MLP/eval/rmse_eval_clean_20260521_123603_*/**/*",
    # seed99 latest_* assets (pred_vs_actual_by_censoring.py)
    "MLP/eval/rmse_eval_clean_20260509_215303_newdp_seed99_points/**/*",
    # eval_stage3_thesis_sync.py default points (student / teacher / baseline)
    "MLP/eval/rmse_eval_clean_20260519_*/**/*",
    # calibration_coverage_audit.py default eval dir
    "MLP/eval/rmse_eval_clean_20260429_130733_winner_full/**/*",
    # LONO per-fold tables + SVGP metrics payloads (dir NAMES keep the BC
    # prefix; make_lono_figures.py strips it when matching)
    "MLP/runs_mlp/*lono*/**/*",
    # baseline comparison report tables
    "MLP/baseline/comparison_reports/**/*",
    # spatial-censoring audit snapshot joined against the 20260519 points
    "MLP/synthetic_data_20260509/spatial_censoring_audit/*.csv",
    # root-level fit reports (filter_survival figure labels nozzles from here)
    "MLP/synthetic_data_clean_lv2/*.csv",
    # lv2 spatial-censoring audit (joined by seed99_assets / stage3_kd_sync)
    "MLP/synthetic_data_clean_lv2/spatial_censoring_audit/*.csv",
    # production stage3 run tables (raw_coverage heatmap input)
    "MLP/runs_mlp/stage3_diag_kd_mse_mu_plus_sigma_w5p0_20260519_153706/*.csv",
    # fallback input root used by raw_coverage_heatmap.find_input_csv
    "MLP/figures/fit_bias_audit_cdf/*.csv",
    # promoted eval_pipeline figure provenance
    "Thesis/images/eval_20260709/*.json",
]


@dataclass
class Step:
    name: str
    desc: str
    cmds: list[list[str]] = field(default_factory=list)
    # (src, dst) copies executed after the commands succeed; absolute paths
    copies: list[tuple[Path, Path]] = field(default_factory=list)
    # paths that must exist before running, else the step is skipped
    requires: list[Path] = field(default_factory=list)
    note: str = ""
    # extra environment variables for the subprocess (e.g. PYTHONPATH)
    extra_env: dict[str, str] = field(default_factory=dict)
    # optional machine-readable provenance written after copies complete
    provenance_path: Path | None = None
    provenance_payload: dict[str, object] = field(default_factory=dict)


def _script(rel: str, *args: str) -> list[str]:
    return [PY, str(PROJECT_ROOT / rel), *args]


def _latest_impingement_npz() -> Path:
    """Newest impingement_frames.npz; the GUI writes timestamped run dirs, not 'latest'."""
    candidates = sorted(
        PROJECT_ROOT.glob("outputs/impingement_gui/*/impingement_frames.npz"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else PROJECT_ROOT / "outputs/impingement_gui/latest/impingement_frames.npz"


FIT_REPORT_CSV = PROJECT_ROOT / "MLP/synthetic_data_clean_lv2/fit_report.csv"
IMPINGEMENT_NPZ = _latest_impingement_npz()
# The QC-gated tree behind the MLP/synthetic_data junction only keeps "clean"
# subdirs; the spatial-censoring audit needs the full cdf/all + series_wide_all
# layout, which survives in lv2.
LV2_ROOT = PROJECT_ROOT / "MLP/synthetic_data_clean_lv2"
LV2_AUDIT_DIR = LV2_ROOT / "spatial_censoring_audit"
LV2_AUDIT_CSV = LV2_AUDIT_DIR / "plume_spatial_censoring_audit.csv"
# Production stage-3 run whose cdf_regime_bins.csv feeds the B.5 coverage heatmap.
PROD_STAGE3_RUN = PROJECT_ROOT / "MLP/runs_mlp/stage3_diag_kd_mse_mu_plus_sigma_w5p0_20260519_153706"
EVAL_PIPELINE_20260709_RUN = (
    PROJECT_ROOT
    / "MLP/eval_pipeline/runs"
    / "evalrun_20260709_223144_thesis_full_fullclean_bootstrap_fast_20260709_223142"
)
EVAL_PIPELINE_20260709_COMPARISON = (
    EVAL_PIPELINE_20260709_RUN
    / "figures/comparison/lv3_qc_gated/cdf_uncensored"
)
EVAL_PIPELINE_20260709_FULL_CLEAN = (
    EVAL_PIPELINE_20260709_RUN
    / "figures/comparison/lv3_qc_gated/full_clean"
)
EVAL_PIPELINE_20260709_SEED99 = (
    EVAL_PIPELINE_20260709_RUN
    / "figures/models/thesis_mlp_seed_99/lv3_qc_gated/cdf_uncensored"
)
EVAL_PIPELINE_20260709_FIGURES = (
    "metric_bars_rmse_mae_p95.png",
    "probabilistic_metric_bars.png",
    "coverage_comparison.png",
    "reliability_overlay.png",
    "crps_sharpness_scatter.png",
    "pit_histograms_grid.png",
    "per_seed_rmse_thesis_mlp_modeA.png",
)
EVAL_PIPELINE_20260709_COPIES = [
    (EVAL_PIPELINE_20260709_COMPARISON / name, EVAL_20260709_IMG / name)
    for name in EVAL_PIPELINE_20260709_FIGURES
] + [
    (EVAL_PIPELINE_20260709_SEED99 / "pred_vs_actual.png",
     EVAL_20260709_IMG / "mlp_seed99_pred_vs_actual.png"),
    (EVAL_PIPELINE_20260709_SEED99 / "residual_histogram.png",
     EVAL_20260709_IMG / "mlp_seed99_residual_histogram.png"),
    (EVAL_PIPELINE_20260709_SEED99 / "trajectory_best.png",
     EVAL_20260709_IMG / "mlp_seed99_trajectory_best.png"),
    (EVAL_PIPELINE_20260709_SEED99 / "trajectory_worst.png",
     EVAL_20260709_IMG / "mlp_seed99_trajectory_worst.png"),
    (EVAL_PIPELINE_20260709_SEED99 / "per_condition_rmse.png",
     EVAL_20260709_IMG / "mlp_seed99_per_condition_rmse.png"),
    (EVAL_PIPELINE_20260709_SEED99 / "coverage_curve.png",
     EVAL_20260709_IMG / "mlp_seed99_coverage_curve.png"),
    (EVAL_PIPELINE_20260709_SEED99 / "sigma_bin_calibration.png",
     EVAL_20260709_IMG / "mlp_seed99_sigma_bin_calibration.png"),
    (EVAL_PIPELINE_20260709_FULL_CLEAN / "reliability_overlay.png",
     EVAL_20260709_IMG / "reliability_overlay_full_clean.png"),
    (EVAL_PIPELINE_20260709_FULL_CLEAN / "crps_sharpness_scatter.png",
     EVAL_20260709_IMG / "crps_sharpness_scatter_full_clean.png"),
]


STEPS: list[Step] = [
    # ───────────────────────────── Chapter 04 ─────────────────────────────
    Step(
        name="fit_diagnostics",
        desc="fig_q1_param_scaling / fig_q1_residual_structure / fig_q1_identifiability (auto-copies into Thesis/images)",
        cmds=[_script("MLP/curve_fit/reports/fit_diagnostics.py")],
    ),
    Step(
        name="filter_survival",
        desc="filter_survival_cdf_by_nozzle.png (auto-copies into Thesis/images)",
        cmds=[_script("MLP/curve_fit/reports/summarize_filter_survival.py",
                      "--fit-report", str(FIT_REPORT_CSV))],
        requires=[FIT_REPORT_CSV],
    ),
    Step(
        name="time_windowed",
        desc="time_windowed_exponents.png",
        cmds=[_script("MLP/curve_fit/reports/time_windowed_exponent_regression.py")],
        copies=[(
            PROJECT_ROOT / "MLP/synthetic_data/fit_diagnostics/time_windowed_exponent/time_windowed_exponents.png",
            IMAGES / "time_windowed_exponents.png",
        )],
    ),
    Step(
        name="spatial_censoring_audit",
        desc="fig_spatial_censoring_fraction_by_nozzle_clean.png (scans raw + lv2 roots; slow-ish)",
        # the QC-gated junction lacks cdf/all + series_wide_all, so audit lv2
        cmds=[_script(
            "MLP/curve_fit/reports/audit_cdf_spatial_censoring.py", "--plots",
            "--synthetic-root", str(LV2_ROOT),
            "--out-dir", str(LV2_AUDIT_DIR),
        )],
        copies=[(
            LV2_AUDIT_DIR / "spatial_censored_fraction_by_nozzle_clean.png",
            IMAGES / "fig_spatial_censoring_fraction_by_nozzle_clean.png",
        )],
        requires=[PROJECT_ROOT / "Mie_scattering_top_view_results", LV2_ROOT],
    ),
    Step(
        name="sparse_instability",
        desc="fig_sparse_support_topology.png / fig_sparse_diameter_interpolants.png",
        cmds=[_script("Thesis/slides/legacy_notebook_sources/slides_sparse_feature_instability/export_sparse_feature_instability_figures.py")],
        # legacy export imports ood_sanity from the MLP package root
        extra_env={"PYTHONPATH": str(PROJECT_ROOT / "MLP") + os.pathsep + str(PROJECT_ROOT)},
        # needs the OLD (pathological) MLP checkpoints, deleted in cleanup; the
        # step self-skips until they are restored. The existing PNGs carry no
        # confidential labels (axes are pressures/diameters), so keeping them is safe.
        requires=[PROJECT_ROOT / "MLP/runs_mlp/distill_cdf_onset_20260331_194213",
                  PROJECT_ROOT / "MLP/runs_mlp/stage2_NLL_penetration_20260317_194155"],
        copies=[
            (THESIS / "slides/legacy_notebook_sources/slides_sparse_feature_instability/figures/support_topology.png",
             IMAGES / "fig_sparse_support_topology.png"),
            (THESIS / "slides/legacy_notebook_sources/slides_sparse_feature_instability/figures/diameter_interpolants.png",
             IMAGES / "fig_sparse_diameter_interpolants.png"),
        ],
    ),
    Step(
        name="censor_library",
        desc="per-condition censoring diagnostics (source pool for fig_censor_fov_saturation / fig_censor_density_drop)",
        cmds=[_script("MLP/curve_fit/workflows/cdf_censoring_points.py", "--plots")],
        note=("MANUAL FOLLOW-UP: the two thesis examples are hand-picked. Browse the freshly written "
              "<out-dir>/plots/ (path printed in the step log), pick one FOV-saturation case and one "
              "density-drop case, and copy them to Thesis/images/fig_censor_fov_saturation.png and "
              "Thesis/images/fig_censor_density_drop.png. The old PNGs embed confidential condition_ids "
              "in the footer, so this re-pick IS required."),
    ),
    Step(
        name="raw_coverage",
        desc="generated/current/raw_coverage_heatmap.png (B.5 heatmap, group labels were leaking)",
        # cdf_regime_bins.csv from the production stage-3 run is the
        # plume-coverage table the heatmap expects (the old by_bin output is a
        # processed artifact with broken condition_group_x/_y columns)
        cmds=[_script(
            "pipelines/audit/raw_coverage_heatmap.py",
            "--fit-run-dir", str(PROD_STAGE3_RUN),
            "--out-dir", str(GEN_CURRENT / "b5_raw_coverage"),
        )],
        copies=[(
            GEN_CURRENT / "b5_raw_coverage/raw_coverage_heatmap_with_thresholds.png",
            GEN_CURRENT / "raw_coverage_heatmap.png",
        )],
        requires=[PROD_STAGE3_RUN / "cdf_regime_bins.csv"],
    ),
    Step(
        name="alpha_sensitivity",
        desc="generated/current/alpha_sensitivity_figure.png",
        cmds=[_script(
            "pipelines/report/alpha_sensitivity_sweep.py",
            "--frames-npz", str(IMPINGEMENT_NPZ),
            "--out-dir", str(GEN_CURRENT / "e2_alpha_sensitivity"),
        )],
        copies=[(
            GEN_CURRENT / "e2_alpha_sensitivity/alpha_sensitivity_curves.png",
            GEN_CURRENT / "alpha_sensitivity_figure.png",
        )],
        requires=[IMPINGEMENT_NPZ],
    ),
    Step(
        name="hermite",
        desc="hermite_crown_profile.png (writes directly into Thesis/images)",
        cmds=[_script("piston/generate_hermite_figure.py")],
    ),
    Step(
        name="nozzle_geometry",
        desc="nozzle_geometry_overview.png (Ch.03 nozzle design-space figure from Table 3.1 data)",
        cmds=[_script("Thesis/generated/make_nozzle_geometry_figure.py")],
    ),
    Step(
        name="mlp_diagrams",
        desc="mlp_training_curriculum.png + mlp_architecture_overview.png (Ch.04 method diagrams)",
        cmds=[_script("Thesis/generated/make_mlp_curriculum_figures.py")],
    ),
    # ───────────────────────────── Chapter 05 ─────────────────────────────
    Step(
        name="eval_pipeline_20260709",
        desc="promote eval_pipeline lv3 comparison and seed-99 diagnostic figures into Thesis/images/eval_20260709",
        copies=EVAL_PIPELINE_20260709_COPIES,
        requires=[
            EVAL_PIPELINE_20260709_RUN / "metrics_wide.csv",
            EVAL_PIPELINE_20260709_RUN / "figure_manifest.json",
            *[src for src, _ in EVAL_PIPELINE_20260709_COPIES],
        ],
        provenance_path=EVAL_20260709_IMG / "provenance.json",
        provenance_payload={
            "description": "Promoted eval_pipeline comparison and seed-99 diagnostic figures for thesis evaluation.",
            "source_run": str(EVAL_PIPELINE_20260709_RUN.relative_to(PROJECT_ROOT)),
            "source_figure_dir": str((EVAL_PIPELINE_20260709_RUN / "figures").relative_to(PROJECT_ROOT)),
            "source_metrics": str((EVAL_PIPELINE_20260709_RUN / "metrics_wide.csv").relative_to(PROJECT_ROOT)),
            "dataset": "lv3_qc_gated",
            "eval_set": "cdf_uncensored",
            "render_command": (
                ".venv\\Scripts\\python.exe MLP\\eval_pipeline\\make_figures.py "
                "--run MLP\\eval_pipeline\\runs\\evalrun_20260709_223144_thesis_full_fullclean_bootstrap_fast_20260709_223142"
            ),
            "promote_command": (
                ".venv\\Scripts\\python.exe Thesis\\regenerate_thesis_figures.py "
                "--only eval_pipeline_20260709 --no-sanitize"
            ),
        },
        note=("This step does not rerun model inference; it promotes already rendered Layer-2 "
              "comparison and per-model figures and writes Thesis/images/eval_20260709/provenance.json."),
    ),
    Step(
        name="baseline_comparison",
        desc="baseline_comparison_20260521/* (full_clean_metric_bars, production_mlp_per_seed_rmse, ...)",
        cmds=[_script("Thesis/generated/baseline_comparison_20260521/make_figures.py")],
    ),
    Step(
        name="nn_fit_results",
        desc="neural_network_fit_results: pred_vs_actual_best / per_folder_rmse_best / traj_best / traj_worst",
        cmds=[_script("Thesis/generated/neural_network_fit_results_20260521/make_figures.py")],
        copies=[
            (NN_FIT_2052_IMG / "pred_vs_actual_seed42.png", NN_FIT_IMG / "pred_vs_actual_best.png"),
            (NN_FIT_2052_IMG / "per_folder_rmse_seed42.png", NN_FIT_IMG / "per_folder_rmse_best.png"),
            (NN_FIT_2052_IMG / "best_traj_seed42.png", NN_FIT_IMG / "traj_best_nozzle7_T19.png"),
            (NN_FIT_2052_IMG / "worst_traj_seed42.png", NN_FIT_IMG / "traj_worst_nozzle3_T3.png"),
        ],
        note=("The tex filenames hard-code the best/worst trajectory identity (nozzle7_T19 / nozzle3_T3). "
              "If the underlying eval changed, check the new best/worst in the step log and update the tex "
              "captions + filenames accordingly."),
    ),
    Step(
        name="seed99_assets",
        desc="latest_pred_vs_actual_seed99 / latest_residual_histogram_seed99 (+ other latest_* assets)",
        # default --audit points at the deleted synthetic_data_20260509 snapshot
        cmds=[_script("MLP/curve_fit/reports/pred_vs_actual_by_censoring.py",
                      "--audit", str(LV2_AUDIT_CSV))],
        requires=[PROJECT_ROOT / "MLP/eval/rmse_eval_clean_20260509_215303_newdp_seed99_points/points.csv",
                  LV2_AUDIT_CSV],
        note="The old latest_worst_traj_seed99.png title contained a BC name; this rerun fixes it.",
    ),
    Step(
        name="calibration_coverage",
        desc="stage3_calibration_coverage.png (auto-copies into Thesis/images)",
        cmds=[_script("MLP/eval/calibration_coverage_audit.py")],
    ),
    Step(
        name="calibration_diag",
        desc="calibration_20260521/reliability_overlay.png + crps_sharpness_scatter.png",
        cmds=[_script("MLP/eval/calibration_diagnostics.py", "--skip-rerun-new-points")],
        note="Drop --skip-rerun-new-points (edit this step) to fully recompute model predictions on GPU.",
    ),
    Step(
        name="stage3_kd_sync",
        desc="stage3_kd_mse_mu_plus_sigma_{overlay_baseline,residual_vs_truth,sigma_calibration}.png + metrics CSVs",
        # default --audit points at the deleted synthetic_data_20260509 snapshot
        cmds=[_script("MLP/curve_fit/reports/eval_stage3_thesis_sync.py",
                      "--audit", str(LV2_AUDIT_CSV))],
        requires=[LV2_AUDIT_CSV],
    ),
    Step(
        name="lono_figures",
        desc="stage2_anchor_ablation / ablation_comparison_best / svgp_lono_comparison / lono_{rmse,coverage}_by_fold",
        cmds=[_script("Thesis/generated/make_lono_figures.py")],
    ),
    Step(
        name="residual_fh_figs",
        desc="residual_family_head_20260608: delta_l2_sweep + residual_svgp_context_rmse_comparison",
        cmds=[
            _script("MLP/MLP_training/ablations/plot_residual_family_head_production.py", "--figure", "delta-l2-sweep"),
            _script("MLP/MLP_training/ablations/plot_residual_family_head_production.py", "--figure", "svgp-context-rmse"),
        ],
        copies=[
            (SLIDES_FH / "figs/delta_l2_sweep.png", RESIDUAL_IMG / "delta_l2_sweep.png"),
            (SLIDES_FH / "figs/residual_svgp_context_rmse_comparison.png",
             RESIDUAL_IMG / "residual_svgp_context_rmse_comparison.png"),
        ],
    ),
    Step(
        name="n0_fewshot",
        desc="residual_family_head_20260608/n0_fewshot_adaptation_curve.png",
        cmds=[_script("Thesis/generated/make_n0_fewshot_figure.py")],
        copies=[(SLIDES_FH / "figs/n0_fewshot_adaptation_curve.png",
                 RESIDUAL_IMG / "n0_fewshot_adaptation_curve.png")],
    ),
    Step(
        name="injector_campaign",
        desc="residual_family_head_20260608: injector_comparison_dp050 + injector_scatter_all_nozzles",
        cmds=[_script("MLP/MLP_training/ablations/compare_campaign_penetration.py", "--dp-exp", "0.50")],
        copies=[
            (PROJECT_ROOT / "MLP/figures/injector_comparison/injector_comparison_dp050.png",
             RESIDUAL_IMG / "injector_comparison_dp050.png"),
            (PROJECT_ROOT / "MLP/figures/injector_comparison/scatter_all_nozzles.png",
             RESIDUAL_IMG / "injector_scatter_all_nozzles.png"),
        ],
    ),
    Step(
        name="arch_diagram",
        desc="residual_family_head_20260608/mlp_family_conditioning_architecture.png (copy of Graphviz render)",
        copies=[(SLIDES_FH / "figs/mlp_family_conditioning_architecture.png",
                 RESIDUAL_IMG / "mlp_family_conditioning_architecture.png")],
        note="Source is figs/mlp_family_conditioning_architecture.dot; re-render with `dot -Tpng` if the diagram changes.",
    ),
]

# Figures this script cannot regenerate — keep in sync with figure_provenance_ch04_05.md §4.
MANUAL_ITEMS = [
    "fig_sparse_support_topology.png / fig_sparse_diameter_interpolants.png — the legacy MLP checkpoints "
    "they demonstrate (runs_mlp/distill_cdf_onset_20260331_194213, stage2_NLL_penetration_20260317_194155) "
    "were deleted; the figures cannot be regenerated but contain no confidential labels — keep the "
    "existing PNGs as historical artifacts.",
    "stage2_loss_curves.png / distillation_loss_curves.png — training-time artifacts (efc/checkpoint.py); "
    "only retraining regenerates them. They plot losses vs epoch, no campaign labels expected.",
    "stage2_toy_inference_2000bar_5bar.png — run manually: "
    ".venv\\Scripts\\python MLP/MLP_training/ablations/toy_inference_from_run.py <stage2_run_dir> "
    "--injection-pressure-bar 2000 --chamber-state-raw 5 --no-show "
    "--save-path Thesis/images/stage2_toy_inference_2000bar_5bar.png",
    "raw_uncertain_vs_teacher_gaussian_2000bar_5bar.png — NO generator in repo (provenance §4); "
    "inspect for confidential labels and rebuild by hand if needed.",
    "representative_clean_fit.png — NO generator in repo (provenance §4); the figure/caption may name a "
    "condition — verify it uses sanitized naming.",
    "neural_network_fit_results/stage3_kd_mse_mu_plus_sigma_modeA_per_seed.png — NO generator in repo; "
    "data lives in Thesis/generated/current/stage3_mse_mu_plus_sigma_modeA_per_seed.csv (sanitized by this run).",
    "residual_family_head_20260608/lono_modified_only_rmse_by_fold.png — NO generator in repo; data in "
    "Thesis/slides/slides_residual_family_head_production/lono_modified_only/ (sanitized by this run).",
    "fig_censor_fov_saturation.png / fig_censor_density_drop.png — re-pick from the censor_library step "
    "output (see that step's note).",
    "failure_mode_nozzle3/* and impingement_* — CV-pipeline side, out of scope here.",
]


# ─────────────────────────────── sanitizer ───────────────────────────────

def iter_sanitize_files() -> list[Path]:
    seen: set[Path] = set()
    for pattern in SANITIZE_GLOBS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                seen.add(path)
    return sorted(seen)


def sanitize_file(path: Path, dry_run: bool) -> int:
    try:
        if path.stat().st_size > MAX_SANITIZE_BYTES:
            print(f"  [sanitize] SKIP (too large): {path}")
            return 0
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except OSError:
            return 0
    except OSError:
        return 0
    new_text, n = CONFIDENTIAL_RE.subn("", text)
    if n and not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return n


def run_sanitize(dry_run: bool) -> None:
    print("── Phase 1: sanitize text inputs (BC########_HZ_ → removed) ──")
    total_files, total_hits = 0, 0
    for path in iter_sanitize_files():
        n = sanitize_file(path, dry_run)
        if n:
            total_files += 1
            total_hits += n
            rel = path.relative_to(PROJECT_ROOT)
            print(f"  [sanitize]{' (dry)' if dry_run else ''} {n:>5} hits  {rel}")
    print(f"  → {total_hits} replacement(s) in {total_files} file(s)\n")


def run_scan() -> list[Path]:
    """Report files still containing the confidential pattern."""
    leftovers: list[Path] = []
    for path in iter_sanitize_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if CONFIDENTIAL_RE.search(text):
            leftovers.append(path)
    return leftovers


# ─────────────────────────────── execution ───────────────────────────────

def run_step(step: Step, log_dir: Path, dry_run: bool) -> str:
    missing = [p for p in step.requires if not p.exists()]
    if missing:
        print(f"  SKIP (missing input): {missing[0]}")
        return "SKIP"
    if dry_run:
        for cmd in step.cmds:
            print(f"  (dry) {' '.join(cmd[1:2] + cmd[2:])}")
        for src, dst in step.copies:
            print(f"  (dry) copy {src.relative_to(PROJECT_ROOT)} -> {dst.relative_to(PROJECT_ROOT)}")
        if step.provenance_path:
            print(f"  (dry) write provenance -> {step.provenance_path.relative_to(PROJECT_ROOT)}")
        return "DRY"

    env = dict(os.environ, MPLBACKEND="Agg", **step.extra_env)
    log_path = log_dir / f"{step.name}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        for cmd in step.cmds:
            log.write(f"$ {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(
                cmd, cwd=PROJECT_ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            if proc.returncode != 0:
                print(f"  FAIL (exit {proc.returncode}) — see {log_path.relative_to(PROJECT_ROOT)}")
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                for line in tail:
                    print(f"    | {line}")
                return "FAIL"

    status = "OK"
    copied_files: list[dict[str, str]] = []
    for src, dst in step.copies:
        if not src.exists():
            print(f"  WARN: expected output missing: {src.relative_to(PROJECT_ROOT)}")
            status = "WARN"
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append({
            "src": str(src.relative_to(PROJECT_ROOT)),
            "dst": str(dst.relative_to(PROJECT_ROOT)),
        })
        print(f"  copied -> {dst.relative_to(PROJECT_ROOT)}")
    if step.provenance_path:
        payload = dict(step.provenance_payload)
        payload["copy_status"] = status
        payload["copied_files"] = copied_files
        step.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        step.provenance_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  provenance -> {step.provenance_path.relative_to(PROJECT_ROOT)}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List steps and exit.")
    parser.add_argument("--only", nargs="+", default=None, help="Run only these step names.")
    parser.add_argument("--skip", nargs="+", default=[], help="Skip these step names.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run, change nothing.")
    parser.add_argument("--no-sanitize", action="store_true", help="Skip the input-sanitizing phase.")
    parser.add_argument("--scan-only", action="store_true", help="Only scan for confidential strings and exit.")
    args = parser.parse_args()

    if args.list:
        for step in STEPS:
            print(f"{step.name:24s} {step.desc}")
        return 0

    if args.scan_only:
        leftovers = run_scan()
        if leftovers:
            print(f"{len(leftovers)} file(s) still contain BC########_HZ_:")
            for p in leftovers:
                print(f"  {p.relative_to(PROJECT_ROOT)}")
        else:
            print("No confidential strings found in monitored text inputs.")
        return 1 if leftovers else 0

    known = {s.name for s in STEPS}
    for name in (args.only or []) + list(args.skip):
        if name not in known:
            parser.error(f"unknown step: {name} (use --list)")

    selected = [s for s in STEPS
                if (args.only is None or s.name in args.only) and s.name not in args.skip]

    if not args.no_sanitize:
        run_sanitize(args.dry_run)

    log_dir = THESIS / "regen_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"── Phase 2: regenerate ({len(selected)} step(s)) ──")
    results: dict[str, str] = {}
    for step in selected:
        print(f"\n[{step.name}] {step.desc}")
        t0 = time.time()
        results[step.name] = run_step(step, log_dir, args.dry_run)
        print(f"  status={results[step.name]}  ({time.time() - t0:.1f}s)")
        if step.note:
            print(f"  NOTE: {step.note}")

    print("\n── Phase 3: post-run scan ──")
    if args.dry_run:
        print("  (dry-run: skipped)")
    else:
        leftovers = run_scan()
        if leftovers:
            print(f"  WARNING: {len(leftovers)} file(s) still contain BC########_HZ_:")
            for p in leftovers:
                print(f"    {p.relative_to(PROJECT_ROOT)}")
        else:
            print("  Clean: no confidential strings left in monitored text inputs.")

    print("\n── Summary ──")
    for name, status in results.items():
        print(f"  {status:5s} {name}")

    print("\n── Manual follow-ups (cannot be automated) ──")
    for item in MANUAL_ITEMS:
        print(f"  • {item}")
    print("\nAfter reviewing the new PNGs, rebuild the thesis (Thesis/build/build_thesis.cmd).")

    return 1 if any(v == "FAIL" for v in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
