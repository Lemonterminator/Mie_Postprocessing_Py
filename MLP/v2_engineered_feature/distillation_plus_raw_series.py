"""
Distillation + raw series refinement for v2 engineered-feature architecture.

Loads a trained Stage-2 NLL model (v2), raw CDF series, builds regime labels,
then trains a student model via knowledge distillation with raw-CDF supervision.

Converted from MLP/v1_direct_feature_training/distillation_plus_raw_series.ipynb,
updated to use the v2 engineered-feature common module (A-scaled penetration).
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from engineered_feature_common import (
    PenetrationMLP,
    build_dataset_registry,
    build_feature_matrix_np,
    build_model,
    infer_feature_family,
    load_run_artifacts,
    split_mu_logvar,
    RunArtifacts,
    TIME_FEATURE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_ROOT = PROJECT_ROOT / "MLP"
SYNTHETIC_ROOT = MLP_ROOT / "synthetic_data"

SOURCES = ["cdf", "bw_x", "bw_polar"]

COMMON_META_COLS = [
    "plumes",
    "diameter_mm",
    "umbrella_angle_deg",
    "fps",
    "chamber_pressure_bar",
    "injection_duration_us",
    "injection_pressure_bar",
    "control_backpressure_bar",
]
MERGE_KEYS = ["experiment_name", "file_path", "file_name", "file_stem", "plume_idx"]

# ── regime labeling constants ──
BIN_MS = 0.1
REGIME_TIME_MAX_MS = 5.0
N_BINS = int(REGIME_TIME_MAX_MS / BIN_MS)

REGIME_GROUP_COLS = [
    "experiment_name",
    "plumes",
    "diameter_mm",
    "umbrella_angle_deg",
    "fps",
    "chamber_pressure_bar",
    "injection_duration_us",
    "injection_pressure_bar",
    "control_backpressure_bar",
]
CDF_SAMPLE_ID_COLS = ["experiment_name", "file_path", "plume_idx"]

UNCERTAIN_RATIO = 0.7
TEACHER_RATIO = 0.2
TEACHER_MIN_COUNT = 4
CONSECUTIVE_BINS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Wide / long helpers  (carried from the notebook as-is)
# ═══════════════════════════════════════════════════════════════════════════════

def prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    return cols


def available_frame_ids(df: pd.DataFrame, prefix: str) -> list[int]:
    return [int(col.rsplit("_", 1)[1]) for col in prefixed_columns(df, prefix)]


def extract_prefixed_matrix(df: pd.DataFrame, prefix: str, frame_ids: list[int] | None = None) -> np.ndarray:
    if frame_ids is None:
        cols = prefixed_columns(df, prefix)
        if not cols:
            return np.empty((len(df), 0), dtype=np.float32)
        return df.loc[:, cols].to_numpy(dtype=np.float32)

    cols = []
    for frame_id in frame_ids:
        col = f"{prefix}{frame_id:03d}"
        if col in df.columns:
            cols.append(df[col].to_numpy(dtype=np.float32))
        else:
            cols.append(np.full(len(df), np.nan, dtype=np.float32))
    if not cols:
        return np.empty((len(df), 0), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32)


def build_wide_from_long(series_df: pd.DataFrame) -> pd.DataFrame:
    if series_df.empty:
        return pd.DataFrame(columns=MERGE_KEYS + COMMON_META_COLS + ["delay_frames_raw", "delay_frames_used", "delay_source", "seq_len"])

    df = series_df.copy()
    key_cols = ["file_path", "file_name", "file_stem", "plume_idx"]

    if "frame_pos" not in df.columns:
        df = df.sort_values(key_cols + ["time_ms"])
        df["frame_pos"] = df.groupby(key_cols, dropna=False).cumcount()

    base = (
        df.loc[:, key_cols + ["frame_pos", "delay_frames_raw", "delay_frames_used", "delay_source"]]
        .sort_values(key_cols + ["frame_pos"])
        .groupby(key_cols, dropna=False)
        .agg(
            delay_frames_raw=("delay_frames_raw", "first"),
            delay_frames_used=("delay_frames_used", "first"),
            delay_source=("delay_source", "first"),
            seq_len=("frame_pos", "count"),
        )
        .reset_index()
    )

    time_wide = (
        df.pivot_table(index=key_cols, columns="frame_pos", values="time_ms", aggfunc="first")
        .sort_index(axis=1)
        .rename(columns=lambda c: f"time_ms_{int(c):03d}")
        .reset_index()
    )
    pen_wide = (
        df.pivot_table(index=key_cols, columns="frame_pos", values="penetration_mm", aggfunc="first")
        .sort_index(axis=1)
        .rename(columns=lambda c: f"penetration_mm_{int(c):03d}")
        .reset_index()
    )

    extra_cols = [
        col for col in df.columns
        if col not in key_cols + ["frame_pos", "time_s", "time_ms", "penetration_mm"]
    ]
    extra = df.loc[:, key_cols + extra_cols].drop_duplicates(key_cols)

    return base.merge(extra, on=key_cols, how="left").merge(time_wide, on=key_cols, how="left").merge(pen_wide, on=key_cols, how="left")


def split_dir_names(split: str) -> tuple[str, str]:
    if split == "clean":
        return "series_wide_clean", "series_clean"
    if split == "all":
        return "series_wide_all", "series_all"
    raise ValueError(f"Unsupported split: {split}")


def load_source_table(source: str, split: str = "clean") -> pd.DataFrame:
    wide_dir_name, long_dir_name = split_dir_names(split)
    frames: list[pd.DataFrame] = []

    for experiment_dir in sorted(SYNTHETIC_ROOT.iterdir()):
        if not experiment_dir.is_dir():
            continue

        source_dir = experiment_dir / source
        if not source_dir.exists():
            continue

        wide_files = sorted((source_dir / wide_dir_name).glob("*.csv")) if (source_dir / wide_dir_name).exists() else []
        if wide_files:
            for path in wide_files:
                df = pd.read_csv(path)
                df.insert(0, "experiment_name", experiment_dir.name)
                frames.append(df)
            continue

        long_files = sorted((source_dir / long_dir_name).glob("*.csv")) if (source_dir / long_dir_name).exists() else []
        for path in long_files:
            df_long = pd.read_csv(path)
            df_wide = build_wide_from_long(df_long)
            df_wide.insert(0, "experiment_name", experiment_dir.name)
            frames.append(df_wide)

    if not frames:
        raise FileNotFoundError(
            f"No raw series files found for source='{source}', split='{split}'. "
            "Run fit_raw_data.py first to populate synthetic_data outputs."
        )

    return pd.concat(frames, ignore_index=True, sort=False)


def rename_source_specific_columns(df: pd.DataFrame, source: str, keep_common_meta: bool) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        if col in MERGE_KEYS:
            continue
        if keep_common_meta and col in COMMON_META_COLS:
            continue
        rename_map[col] = f"{source}__{col}"
    return df.rename(columns=rename_map)


def merge_source_tables(source_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for idx, source in enumerate(SOURCES):
        table = source_tables[source]
        table = rename_source_specific_columns(table, source, keep_common_meta=(idx == 0))
        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on=MERGE_KEYS, how="inner")
    if merged is None:
        raise ValueError("No source tables to merge.")
    return merged


class RawSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, mask: np.ndarray, time_ms: np.ndarray, meta: pd.DataFrame):
        self.series = torch.as_tensor(series, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.time_ms = torch.as_tensor(time_ms, dtype=torch.float32)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int):
        return {
            "series": self.series[idx],
            "mask": self.mask[idx],
            "time_ms": self.time_ms[idx],
            "meta": self.meta.iloc[idx].to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CDF regime labeling
# ═══════════════════════════════════════════════════════════════════════════════

def wide_source_to_long(source_wide_df: pd.DataFrame) -> pd.DataFrame:
    frame_ids = sorted({
        frame_id
        for frame_id in available_frame_ids(source_wide_df, "time_ms_")
        if frame_id in set(available_frame_ids(source_wide_df, "penetration_mm_"))
    })
    if not frame_ids:
        return pd.DataFrame(columns=REGIME_GROUP_COLS + CDF_SAMPLE_ID_COLS[1:] + ["file_name", "file_stem", "time_bin", "time_ms", "penetration_mm", "frame_pos"])

    time_mat = extract_prefixed_matrix(source_wide_df, "time_ms_", frame_ids)
    pen_mat = extract_prefixed_matrix(source_wide_df, "penetration_mm_", frame_ids)
    base_cols = [
        "experiment_name", "file_path", "file_name", "file_stem", "plume_idx",
        *COMMON_META_COLS,
        "delay_frames_raw", "delay_frames_used", "delay_source", "seq_len",
    ]
    base_cols = [c for c in base_cols if c in source_wide_df.columns]
    repeated = source_wide_df.loc[:, base_cols].loc[source_wide_df.index.repeat(len(frame_ids))].reset_index(drop=True)
    repeated["frame_pos"] = np.tile(np.asarray(frame_ids, dtype=np.int32), len(source_wide_df))
    repeated["time_ms"] = time_mat.reshape(-1)
    repeated["penetration_mm"] = pen_mat.reshape(-1)
    repeated = repeated.loc[np.isfinite(repeated["time_ms"]) & np.isfinite(repeated["penetration_mm"])].copy()
    repeated["time_bin"] = np.floor(repeated["time_ms"] / BIN_MS).astype(int).clip(0, N_BINS - 1)
    return repeated.reset_index(drop=True)


def _first_consecutive(mask: np.ndarray, run_len: int) -> int | None:
    if len(mask) < run_len:
        return None
    for i in range(len(mask) - run_len + 1):
        if np.all(mask[i : i + run_len]):
            return i
    return None


def build_time_bin_regimes(cdf_long_df: pd.DataFrame) -> pd.DataFrame:
    df = cdf_long_df.copy()
    dedup = df.drop_duplicates(subset=REGIME_GROUP_COLS + CDF_SAMPLE_ID_COLS[1:] + ["time_bin"])

    rows = []
    for group_key, g in dedup.groupby(REGIME_GROUP_COLS, dropna=False):
        counts = (
            g.groupby("time_bin", dropna=False)
            .size()
            .reindex(range(N_BINS), fill_value=0)
            .to_numpy(dtype=float)
        )
        counts_smooth = (
            pd.Series(counts)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        b_peak = int(np.argmax(counts_smooth))
        n_ref = float(max(counts_smooth[b_peak], 1.0))
        coverage_ratio = counts_smooth / n_ref

        uncertain_mask = coverage_ratio[b_peak:] < UNCERTAIN_RATIO
        rel_uncertain = _first_consecutive(uncertain_mask, CONSECUTIVE_BINS)
        b_uncertain_start = b_peak + rel_uncertain if rel_uncertain is not None else N_BINS

        teacher_mask = (coverage_ratio[b_uncertain_start:] < TEACHER_RATIO) | (counts[b_uncertain_start:] < TEACHER_MIN_COUNT)
        rel_teacher = _first_consecutive(teacher_mask, CONSECUTIVE_BINS)
        b_teacher_start = b_uncertain_start + rel_teacher if rel_teacher is not None else N_BINS

        for b in range(N_BINS):
            if b < b_uncertain_start:
                regime = "raw_reliable"
            elif b < b_teacher_start:
                regime = "raw_uncertain"
            else:
                regime = "teacher_only"

            row = {col: val for col, val in zip(REGIME_GROUP_COLS, group_key)}
            row.update({
                "time_bin": b,
                "time_bin_start_ms": b * BIN_MS,
                "time_bin_end_ms": (b + 1) * BIN_MS,
                "n_raw": int(counts[b]),
                "n_raw_smooth": float(counts_smooth[b]),
                "coverage_ratio": float(coverage_ratio[b]),
                "b_peak": int(b_peak),
                "b_uncertain_start": int(b_uncertain_start),
                "b_teacher_start": int(b_teacher_start),
                "regime": regime,
            })
            rows.append(row)

    return pd.DataFrame(rows)


def attach_regimes_to_cdf_long(cdf_long_df: pd.DataFrame, regime_bins_df: pd.DataFrame) -> pd.DataFrame:
    out = cdf_long_df.merge(
        regime_bins_df,
        on=REGIME_GROUP_COLS + ["time_bin"],
        how="left",
    )
    out["recommended_supervision"] = np.where(out["regime"] == "raw_reliable", "raw", "teacher")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# v2 teacher helpers  (A-scaled)
# ═══════════════════════════════════════════════════════════════════════════════

def umbrella_to_tilt_radian(umbrella_angle_deg: float) -> float:
    return float(np.deg2rad((180.0 - float(umbrella_angle_deg)) / 2.0))


def build_teacher_raw_dict(meta_row: pd.Series) -> dict[str, Any]:
    """Build the raw-input dict expected by ``build_feature_matrix_np``."""
    return {
        "tilt_angle_radian": umbrella_to_tilt_radian(meta_row["umbrella_angle_deg"]),
        "plumes": float(meta_row["plumes"]),
        "diameter_mm": float(meta_row["diameter_mm"]),
        "injection_duration_us": float(meta_row["injection_duration_us"]),
        "injection_pressure_bar": float(meta_row["injection_pressure_bar"]),
        "chamber_pressure_bar": float(meta_row["chamber_pressure_bar"]),
        "control_backpressure_bar": float(meta_row["control_backpressure_bar"]),
    }


def predict_teacher_gaussian(
    artifacts: RunArtifacts,
    meta_row: pd.Series,
    time_ms: np.ndarray,
    registry: dict,
) -> pd.DataFrame:
    """Run teacher forward pass and return physical-space mu / std."""
    feature_columns = list(artifacts.train_config["feature_columns"])
    raw = build_teacher_raw_dict(meta_row)
    features_np, a_scale_np, _ = build_feature_matrix_np(
        raw,
        np.asarray(time_ms, dtype=np.float32),
        artifacts.scaler_state,
        feature_columns,
        registry,
        time_feature=str(artifacts.train_config.get("time_feature", TIME_FEATURE)),
    )
    device = next(artifacts.model.parameters()).device
    features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
    a_scale = torch.as_tensor(a_scale_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = artifacts.model(features)
        mu_hat, log_var_hat = split_mu_logvar(out)

    family = infer_feature_family(feature_columns)
    if family == "engineered_v2":
        mu_phys = (a_scale * mu_hat).detach().cpu().numpy().reshape(-1)
        std_phys = (a_scale * torch.exp(0.5 * log_var_hat)).detach().cpu().numpy().reshape(-1)
    else:
        mu_phys = mu_hat.detach().cpu().numpy().reshape(-1)
        std_phys = torch.exp(0.5 * log_var_hat).detach().cpu().numpy().reshape(-1)

    std_floor = float(artifacts.train_config.get("std_clamp_min", 0.0))
    std_phys = np.maximum(std_phys, std_floor)
    return pd.DataFrame({
        "time_ms": np.asarray(time_ms, dtype=float),
        "mu_mm": mu_phys.astype(float),
        "std_mm": std_phys.astype(float),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement dataset  (v2: features via build_feature_matrix_np)
# ═══════════════════════════════════════════════════════════════════════════════

def split_indices(n_items: int, *, seed: int, val_frac: float, test_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items, dtype=int)
    rng.shuffle(indices)

    n_test = int(round(n_items * test_frac))
    n_val = int(round(n_items * val_frac))
    n_test = min(max(n_test, 1), max(n_items - 2, 1))
    n_val = min(max(n_val, 1), max(n_items - n_test - 1, 1))

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


class CDFRefinementSequenceDataset(Dataset):
    """Per-sample dataset for CDF distillation refinement (v2 engineered features)."""

    def __init__(
        self,
        df: pd.DataFrame,
        regime_bins_df: pd.DataFrame,
        *,
        artifacts: RunArtifacts,
        registry: dict,
        config: dict,
        global_time_bin_weights: np.ndarray,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.artifacts = artifacts
        self.registry = registry
        self.config = dict(config)
        self.global_time_bin_weights = global_time_bin_weights

        self.feature_columns = list(artifacts.train_config["feature_columns"])
        self.time_feature = str(artifacts.train_config.get("time_feature", TIME_FEATURE))

        self.time_min_ms = float(config["time_min_ms"])
        self.time_max_ms = float(config["time_max_ms"])
        self.n_points = int(config["n_points"])
        self.time_grid_ms = np.linspace(self.time_min_ms, self.time_max_ms, self.n_points, dtype=np.float32)
        self.time_bins = np.floor(self.time_grid_ms / BIN_MS).astype(int).clip(0, N_BINS - 1)

        self.time_cols = prefixed_columns(self.df, "time_ms_")
        self.pen_cols = prefixed_columns(self.df, "penetration_mm_")
        if len(self.time_cols) != len(self.pen_cols):
            raise ValueError("CDF wide table time/penetration columns do not match.")

        self._raw_weight_lut, self._kd_weight_lut = self._build_regime_weight_lookup(regime_bins_df)
        self.onset_bins = self._build_onset_bins()

    def _build_regime_weight_lookup(self, regime_bins_df: pd.DataFrame) -> tuple[dict, dict]:
        raw_lut: dict[tuple, np.ndarray] = {}
        kd_lut: dict[tuple, np.ndarray] = {}
        for group_key, g in regime_bins_df.groupby(REGIME_GROUP_COLS, dropna=False):
            arr = np.full(N_BINS, "teacher_only", dtype=object)
            g_sorted = g.sort_values("time_bin")
            arr[g_sorted["time_bin"].to_numpy(dtype=int)] = g_sorted["regime"].to_numpy(dtype=object)
            raw_lut[group_key] = np.asarray([self.config["raw_weights"][reg] for reg in arr], dtype=np.float32)
            kd_lut[group_key] = np.asarray([self.config["kd_weights"][reg] for reg in arr], dtype=np.float32)
        return raw_lut, kd_lut

    def _row_group_key(self, row: pd.Series) -> tuple:
        return tuple(row[col] for col in REGIME_GROUP_COLS)

    def _build_onset_bins(self) -> np.ndarray:
        onset_bins = np.zeros(len(self.df), dtype=np.int64)
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            time_vals = row[self.time_cols].to_numpy(dtype=float)
            pen_vals = row[self.pen_cols].to_numpy(dtype=float)
            valid = np.isfinite(time_vals) & np.isfinite(pen_vals)
            if np.any(valid):
                first_t = float(np.nanmin(time_vals[valid]))
                onset_bins[idx] = int(np.floor(first_t / BIN_MS))
        return onset_bins.clip(0, N_BINS - 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        raw = build_teacher_raw_dict(row)

        features_np, a_scale_np, _ = build_feature_matrix_np(
            raw,
            self.time_grid_ms,
            self.artifacts.scaler_state,
            self.feature_columns,
            self.registry,
            time_feature=self.time_feature,
        )

        raw_target = np.full(self.n_points, np.nan, dtype=np.float32)
        raw_valid = np.zeros(self.n_points, dtype=bool)

        observed_time = row[self.time_cols].to_numpy(dtype=float)
        observed_pen = row[self.pen_cols].to_numpy(dtype=float)
        finite = np.isfinite(observed_time) & np.isfinite(observed_pen)
        if np.any(finite):
            obs_time = observed_time[finite]
            obs_pen = observed_pen[finite]
            obs_idx = np.clip(
                np.round((obs_time - self.time_min_ms) / max(self.time_max_ms - self.time_min_ms, 1e-12) * (self.n_points - 1)).astype(int),
                0,
                self.n_points - 1,
            )
            for grid_idx in np.unique(obs_idx):
                vals = obs_pen[obs_idx == grid_idx]
                raw_target[grid_idx] = np.float32(np.mean(vals))
                raw_valid[grid_idx] = True

        group_key = self._row_group_key(row)
        raw_regime_weights = self._raw_weight_lut[group_key][self.time_bins].copy()
        kd_regime_weights = self._kd_weight_lut[group_key][self.time_bins].copy()
        time_bin_weights = self.global_time_bin_weights[self.time_bins].astype(np.float32)
        raw_weights = raw_regime_weights * raw_valid.astype(np.float32) * time_bin_weights
        kd_regime_weights = kd_regime_weights * time_bin_weights

        onset_target = np.clip(self.time_grid_ms / max(float(self.config["onset_ramp_ms"]), 1e-6), 0.0, 1.0).astype(np.float32)
        onset_loss_mask = ((self.time_grid_ms >= 0.0) & (self.time_grid_ms <= float(self.config["onset_loss_window_ms"]))).astype(np.float32)
        anchor_weight = np.clip(1.0 - self.time_grid_ms / max(float(self.config["anchor_window_ms"]), 1e-6), 0.0, 1.0).astype(np.float32)

        return {
            "features": torch.from_numpy(features_np.astype(np.float32)),
            "a_scale": torch.from_numpy(a_scale_np.astype(np.float32)),
            "time_ms": torch.from_numpy(self.time_grid_ms.copy()),
            "raw_target": torch.from_numpy(np.nan_to_num(raw_target, nan=0.0).astype(np.float32)),
            "raw_valid": torch.from_numpy(raw_valid),
            "raw_weight": torch.from_numpy(raw_weights.astype(np.float32)),
            "kd_weight": torch.from_numpy(kd_regime_weights.astype(np.float32)),
            "onset_target": torch.from_numpy(onset_target),
            "onset_loss_mask": torch.from_numpy(onset_loss_mask),
            "anchor_weight": torch.from_numpy(anchor_weight),
            "sample_idx": torch.tensor(int(idx), dtype=torch.long),
        }


def build_sequence_sampler_weights(dataset: CDFRefinementSequenceDataset) -> np.ndarray:
    counts = pd.Series(dataset.onset_bins).value_counts().to_dict()
    weights = np.asarray([1.0 / counts[int(bin_id)] for bin_id in dataset.onset_bins], dtype=np.float64)
    return weights / weights.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Student model (output_dim=3: mu_hat, logvar_hat, onset_logit)
# ═══════════════════════════════════════════════════════════════════════════════

def build_student_model_from_teacher(
    teacher_model: torch.nn.Module,
    teacher_config: dict,
    device: torch.device,
) -> PenetrationMLP:
    student = PenetrationMLP(
        input_dim=int(teacher_config["input_dim"]),
        hidden_dims=[int(x) for x in teacher_config["hidden_dims"]],
        output_dim=3,
        activation=str(teacher_config.get("activation", "gelu")),
        dropout=float(teacher_config.get("dropout", 0.0)),
    ).to(device)

    teacher_state = teacher_model.state_dict()
    student_state = student.state_dict()
    for key, value in teacher_state.items():
        if key in student_state and student_state[key].shape == value.shape:
            student_state[key] = value.clone()

    final_weight_key = [k for k in student_state if k.endswith("weight")][-1]
    final_bias_key = [k for k in student_state if k.endswith("bias")][-1]
    student_state[final_weight_key][:2] = teacher_state[final_weight_key]
    student_state[final_bias_key][:2] = teacher_state[final_bias_key]
    student_state[final_weight_key][2].zero_()
    student_state[final_bias_key][2].zero_()
    student.load_state_dict(student_state)
    return student


def split_student_output(model_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, log_var, onset_logit = torch.split(model_output, [1, 1, 1], dim=-1)
    return mu, log_var, onset_logit


# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions  (v2: operates in A-scaled space)
# ═══════════════════════════════════════════════════════════════════════════════

def weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = weights.sum()
    if float(denom.detach().cpu()) <= eps:
        return values.new_tensor(0.0)
    return (values * weights).sum() / denom


def derivative_shape_penalty(
    mu_physical: torch.Tensor,
    time_ms: torch.Tensor,
    *,
    d2_start_ms: float,
    d2_transition_ms: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    d1 = mu_physical[:, 1:] - mu_physical[:, :-1]
    d1_penalty = torch.relu(-d1).pow(2).mean()

    if mu_physical.shape[1] <= 2:
        return d1_penalty, mu_physical.new_tensor(0.0)

    d2 = mu_physical[:, 2:] - 2.0 * mu_physical[:, 1:-1] + mu_physical[:, :-2]
    t_center_ms = time_ms[:, 1:-1]
    gate = torch.sigmoid((t_center_ms - float(d2_start_ms)) / max(float(d2_transition_ms), 1e-6))
    d2_penalty = (torch.relu(d2).pow(2) * gate).mean()
    return d1_penalty, d2_penalty


def compute_teacher_outputs(
    teacher_model: torch.nn.Module,
    features: torch.Tensor,
    a_scale: torch.Tensor,
    *,
    log_var_bounds: tuple[float, float],
    nll_eps: float,
    std_clamp_min: float,
    is_engineered_v2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teacher forward pass returning physical-space mu, logvar, var."""
    bsz, n_points, feat_dim = features.shape
    with torch.no_grad():
        teacher_flat = teacher_model(features.reshape(bsz * n_points, feat_dim))
        mu_hat, log_var_hat = split_mu_logvar(teacher_flat)

    mu_hat = mu_hat.reshape(bsz, n_points, 1)
    log_var_hat = log_var_hat.reshape(bsz, n_points, 1)
    log_var_hat = torch.clamp(log_var_hat, min=log_var_bounds[0], max=log_var_bounds[1])

    # a_scale: [bsz, n_points, 1]
    a = a_scale.reshape(bsz, n_points, 1) if a_scale.ndim == 3 else a_scale.unsqueeze(-1) if a_scale.ndim == 2 else a_scale.reshape(bsz, 1, 1)

    if is_engineered_v2:
        teacher_mu = a * mu_hat
        teacher_std = torch.clamp(a * torch.exp(0.5 * log_var_hat), min=std_clamp_min)
    else:
        teacher_mu = mu_hat
        teacher_std = torch.clamp(torch.exp(0.5 * log_var_hat), min=std_clamp_min)

    teacher_var = teacher_std.pow(2) + nll_eps
    return teacher_mu, log_var_hat, teacher_var


def refinement_loss(
    student_output: torch.Tensor,
    a_scale: torch.Tensor,
    teacher_mu_phys: torch.Tensor,
    teacher_var_phys: torch.Tensor,
    batch: dict,
    *,
    config: dict,
    is_engineered_v2: bool,
) -> tuple[torch.Tensor, dict]:
    mu_hat, log_var_hat, onset_logit = split_student_output(student_output)
    log_var_hat = torch.clamp(log_var_hat, min=config["log_var_bounds"][0], max=config["log_var_bounds"][1])

    bsz, n_points, _ = mu_hat.shape
    a = a_scale.reshape(bsz, n_points, 1) if a_scale.ndim == 3 else a_scale.unsqueeze(-1) if a_scale.ndim == 2 else a_scale.reshape(bsz, 1, 1)

    if is_engineered_v2:
        mu_phys = a * mu_hat
        var_phys = (a.pow(2)) * torch.exp(log_var_hat) + float(config["nll_eps"])
    else:
        mu_phys = mu_hat
        var_phys = torch.exp(log_var_hat) + float(config["nll_eps"])
    log_var_phys = torch.log(var_phys)

    raw_target = batch["raw_target"].unsqueeze(-1)  # physical mm
    raw_weight = batch["raw_weight"]
    kd_weight = batch["kd_weight"]
    onset_target = batch["onset_target"]
    onset_loss_mask = batch["onset_loss_mask"]
    anchor_weight = batch["anchor_weight"]
    time_ms = batch["time_ms"]

    # Raw NLL in physical space
    raw_nll_point = 0.5 * (log_var_phys + (mu_phys - raw_target).pow(2) / var_phys)
    raw_loss = weighted_mean(raw_nll_point.squeeze(-1), raw_weight)

    # KD KL in physical space
    teacher_std_phys = torch.sqrt(teacher_var_phys)
    kl_point = 0.5 * (torch.log(var_phys / teacher_var_phys) + (teacher_var_phys + (teacher_mu_phys - mu_phys).pow(2)) / var_phys - 1.0)
    conf_weight = torch.clamp(float(config["sigma_conf_ref_mm"]) / teacher_std_phys, min=0.25, max=1.0).squeeze(-1)
    kd_loss = weighted_mean(kl_point.squeeze(-1), kd_weight * conf_weight)

    # Onset BCE
    onset_bce = F.binary_cross_entropy_with_logits(onset_logit.squeeze(-1), onset_target, reduction="none")
    onset_loss = weighted_mean(onset_bce, onset_loss_mask)

    # Anchor and shape penalties in physical space
    anchor_loss = weighted_mean(mu_phys.squeeze(-1).pow(2), anchor_weight)

    d1_penalty, d2_penalty = derivative_shape_penalty(
        mu_phys.squeeze(-1),
        time_ms,
        d2_start_ms=config["d2_start_ms"],
        d2_transition_ms=config["d2_transition_ms"],
    )

    loss = (
        raw_loss
        + kd_loss
        + float(config["lambda_onset"]) * onset_loss
        + float(config["lambda_anchor"]) * anchor_loss
        + float(config["d1_positive_weight"]) * d1_penalty
        + float(config["d2_concave_weight"]) * d2_penalty
    )

    metrics = {
        "loss": float(loss.detach().cpu()),
        "raw_nll": float(raw_loss.detach().cpu()),
        "kd_kl": float(kd_loss.detach().cpu()),
        "onset_bce": float(onset_loss.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()),
        "d1_penalty": float(d1_penalty.detach().cpu()),
        "d2_penalty": float(d2_penalty.detach().cpu()),
        "teacher_conf_min": float(conf_weight.min().detach().cpu()),
        "teacher_conf_max": float(conf_weight.max().detach().cpu()),
    }
    return loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_refinement_epoch(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: dict,
    is_engineered_v2: bool,
) -> dict:
    is_train = optimizer is not None
    student_model.train(mode=is_train)

    totals = {
        "loss": 0.0,
        "raw_nll": 0.0,
        "kd_kl": 0.0,
        "onset_bce": 0.0,
        "anchor": 0.0,
        "d1_penalty": 0.0,
        "d2_penalty": 0.0,
    }
    total_sequences = 0

    for batch in dataloader:
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        features = batch_device["features"]
        a_scale = batch_device["a_scale"]
        batch_size, n_points, feat_dim = features.shape

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        student_out = student_model(features.reshape(batch_size * n_points, feat_dim)).reshape(batch_size, n_points, 3)
        teacher_mu, teacher_log_var, teacher_var = compute_teacher_outputs(
            teacher_model,
            features,
            a_scale,
            log_var_bounds=config["log_var_bounds"],
            nll_eps=config["nll_eps"],
            std_clamp_min=float(config.get("std_clamp_min", 0.0)),
            is_engineered_v2=is_engineered_v2,
        )
        loss, metrics = refinement_loss(
            student_out, a_scale, teacher_mu, teacher_var, batch_device,
            config=config, is_engineered_v2=is_engineered_v2,
        )

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()

        total_sequences += batch_size
        for key in totals:
            totals[key] += metrics[key] * batch_size

    denom = max(total_sequences, 1)
    return {key: value / denom for key, value in totals.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distillation + raw CDF refinement (v2 engineered feature).")
    p.add_argument("run_dir", help="Stage-2 NLL run directory to use as teacher.")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--series-split", choices=("clean", "all"), default="clean")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no-train", action="store_true", help="Only run diagnostics, skip training.")
    p.add_argument("--save-figures", action="store_true")
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()

    # ── resolve run artifacts ──
    device_str = None if args.device == "auto" else args.device
    artifacts = load_run_artifacts(args.run_dir, device=device_str)
    device = next(artifacts.model.parameters()).device
    train_config = artifacts.train_config
    scaler_state = artifacts.scaler_state
    teacher_model = artifacts.model
    teacher_model.eval()

    feature_columns = list(train_config["feature_columns"])
    is_engineered_v2 = infer_feature_family(feature_columns) == "engineered_v2"
    registry = build_dataset_registry()

    print(f"Run dir: {artifacts.run_dir}")
    print(f"Model path: {artifacts.model_path}")
    print(f"Device: {device}")
    print(f"Feature columns: {feature_columns}")
    print(f"Feature family: {'engineered_v2' if is_engineered_v2 else 'legacy_raw'}")
    print(teacher_model)

    # ── load raw series ──
    source_tables = {source: load_source_table(source, split=args.series_split) for source in SOURCES}
    for source, table in source_tables.items():
        print(source, table.shape)

    raw_series_df = merge_source_tables(source_tables)
    print("merged shape:", raw_series_df.shape)

    # ── tensorization ──
    all_frame_ids = sorted({
        frame_id
        for source in SOURCES
        for frame_id in available_frame_ids(raw_series_df, f"{source}__penetration_mm_")
    })

    series_stack = np.stack(
        [extract_prefixed_matrix(raw_series_df, f"{source}__penetration_mm_", all_frame_ids) for source in SOURCES],
        axis=-1,
    )
    time_stack = np.stack(
        [extract_prefixed_matrix(raw_series_df, f"{source}__time_ms_", all_frame_ids) for source in SOURCES],
        axis=-1,
    )

    mask_stack = np.isfinite(series_stack)
    series_tensor = np.nan_to_num(series_stack, nan=0.0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        time_ms_tensor = np.nanmean(np.where(np.isfinite(time_stack), time_stack, np.nan), axis=-1).astype(np.float32)

    meta_cols = MERGE_KEYS + COMMON_META_COLS + [
        f"{src}__{col}"
        for src in SOURCES
        for col in ["delay_frames_raw", "delay_frames_used", "delay_source", "seq_len"]
    ]
    meta_df = raw_series_df.loc[:, [c for c in meta_cols if c in raw_series_df.columns]].copy()

    dataset = RawSeriesDataset(series_tensor, mask_stack, time_ms_tensor, meta_df)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print("series_tensor shape:", series_tensor.shape)
    print("mask_stack shape:", mask_stack.shape)
    print("time_ms_tensor shape:", time_ms_tensor.shape)
    print("dataset size:", len(dataset))

    # ── CDF regime labeling ──
    cdf_wide_df = source_tables["cdf"].copy()
    cdf_long_df = wide_source_to_long(cdf_wide_df)
    cdf_regime_bins_df = build_time_bin_regimes(cdf_long_df)
    cdf_labeled_df = attach_regimes_to_cdf_long(cdf_long_df, cdf_regime_bins_df)

    print("cdf_wide_df shape:", cdf_wide_df.shape)
    print("cdf_long_df shape:", cdf_long_df.shape)
    print("cdf_regime_bins_df shape:", cdf_regime_bins_df.shape)
    print("cdf_labeled_df shape:", cdf_labeled_df.shape)
    print()
    print("Regime bin counts:")
    print(cdf_regime_bins_df["regime"].value_counts().to_string())
    print()
    print("Recommended supervision counts:")
    print(cdf_labeled_df["recommended_supervision"].value_counts().to_string())

    # ── delay report ──
    delay_report_df = cdf_wide_df.loc[:, [
        c for c in ["experiment_name", "file_path", "file_name", "plume_idx", "fps", "delay_frames_raw", "delay_frames_used", "delay_source"]
        if c in cdf_wide_df.columns
    ]].copy()
    delay_report_df["delay_frames_raw"] = pd.to_numeric(delay_report_df["delay_frames_raw"], errors="coerce")
    delay_report_df["delay_frames_used"] = pd.to_numeric(delay_report_df["delay_frames_used"], errors="coerce")
    delay_report_df["fps"] = pd.to_numeric(delay_report_df["fps"], errors="coerce")
    delay_report_df["delay_ms"] = delay_report_df["delay_frames_used"] / delay_report_df["fps"] * 1e3
    print()
    print("Delay overall summary (ms):")
    print(delay_report_df["delay_ms"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame().T.to_string(index=False))

    # ── refinement config ──
    refine_config = {
        "seed": int(train_config.get("seed", 42)),
        "time_min_ms": float(scaler_state["time"]["min_ms"]),
        "time_max_ms": float(scaler_state["time"]["max_ms"]),
        "n_points": int(train_config.get("n_points", 1024)),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": float(train_config.get("weight_decay", 6e-4)),
        "val_frac": float(train_config.get("val_ratio", 0.15)),
        "test_frac": float(train_config.get("test_ratio", 0.15)),
        "num_workers": 0,
        "pin_memory": False,
        "log_var_bounds": tuple(train_config.get("log_var_bounds", (-10.0, 6.0))),
        "nll_eps": float(train_config.get("nll_eps", 1e-12)),
        "std_clamp_min": float(train_config.get("std_clamp_min", 1e-3)),
        "lambda_onset": 0.1,
        "lambda_anchor": 1e-2,
        "d1_positive_weight": 5e-5,
        "d2_concave_weight": 5e-4,
        "d2_start_ms": 0.5,
        "d2_transition_ms": 0.05,
        "onset_ramp_ms": 0.12,
        "onset_loss_window_ms": 0.2,
        "anchor_window_ms": 0.15,
        "sigma_conf_ref_mm": 10.0,
        "raw_weights": {"raw_reliable": 1.0, "raw_uncertain": 0.0, "teacher_only": 0.0},
        "kd_weights": {"raw_reliable": 0.25, "raw_uncertain": 1.0, "teacher_only": 0.75},
        "runs_root": str(MLP_ROOT / "runs_mlp"),
    }
    set_global_seed(refine_config["seed"])

    raw_time_bin_counts = cdf_labeled_df.groupby("time_bin", dropna=False).size().reindex(range(N_BINS), fill_value=0).astype(float)
    raw_time_bin_reference = float(raw_time_bin_counts[raw_time_bin_counts > 0].mean()) if np.any(raw_time_bin_counts > 0) else 1.0
    global_time_bin_weights = (raw_time_bin_reference / raw_time_bin_counts.replace(0.0, np.nan)).fillna(1.0).clip(0.5, 2.0).to_numpy(dtype=np.float32)

    # ── build refinement datasets ──
    train_idx, val_idx, test_idx = split_indices(
        len(cdf_wide_df),
        seed=refine_config["seed"],
        val_frac=refine_config["val_frac"],
        test_frac=refine_config["test_frac"],
    )

    ds_kwargs = dict(
        regime_bins_df=cdf_regime_bins_df,
        artifacts=artifacts,
        registry=registry,
        config=refine_config,
        global_time_bin_weights=global_time_bin_weights,
    )
    dataset_refine_train = CDFRefinementSequenceDataset(cdf_wide_df.iloc[train_idx].reset_index(drop=True), **ds_kwargs)
    dataset_refine_val = CDFRefinementSequenceDataset(cdf_wide_df.iloc[val_idx].reset_index(drop=True), **ds_kwargs)
    dataset_refine_test = CDFRefinementSequenceDataset(cdf_wide_df.iloc[test_idx].reset_index(drop=True), **ds_kwargs)

    train_sampler_weights = build_sequence_sampler_weights(dataset_refine_train)
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(train_sampler_weights, dtype=torch.double),
        num_samples=len(dataset_refine_train),
        replacement=True,
    )

    train_loader_refine = DataLoader(
        dataset_refine_train,
        batch_size=refine_config["batch_size"],
        sampler=train_sampler,
        num_workers=refine_config["num_workers"],
        pin_memory=refine_config["pin_memory"],
    )
    val_loader_refine = DataLoader(
        dataset_refine_val,
        batch_size=refine_config["batch_size"],
        shuffle=False,
        num_workers=refine_config["num_workers"],
        pin_memory=refine_config["pin_memory"],
    )
    test_loader_refine = DataLoader(
        dataset_refine_test,
        batch_size=refine_config["batch_size"],
        shuffle=False,
        num_workers=refine_config["num_workers"],
        pin_memory=refine_config["pin_memory"],
    )

    print()
    print("refinement split sizes:", {
        "train": len(dataset_refine_train),
        "val": len(dataset_refine_val),
        "test": len(dataset_refine_test),
    })

    # ── static check ──
    student_model = build_student_model_from_teacher(teacher_model, train_config, device)
    student_model.eval()

    refine_first_batch = next(iter(train_loader_refine))
    refine_features = refine_first_batch["features"].to(device)
    refine_a_scale = refine_first_batch["a_scale"].to(device)

    with torch.no_grad():
        refine_student_out = student_model(refine_features.reshape(-1, refine_features.shape[-1])).reshape(refine_features.shape[0], refine_features.shape[1], 3)
        refine_teacher_mu, _, refine_teacher_var = compute_teacher_outputs(
            teacher_model,
            refine_features,
            refine_a_scale,
            log_var_bounds=refine_config["log_var_bounds"],
            nll_eps=refine_config["nll_eps"],
            std_clamp_min=float(refine_config.get("std_clamp_min", 0.0)),
            is_engineered_v2=is_engineered_v2,
        )

    refine_loss_val, refine_metrics = refinement_loss(
        refine_student_out,
        refine_a_scale,
        refine_teacher_mu,
        refine_teacher_var,
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in refine_first_batch.items()},
        config=refine_config,
        is_engineered_v2=is_engineered_v2,
    )

    assert refine_student_out.shape[-1] == 3, "Student output must be [B, T, 3]."
    assert np.isfinite(refine_metrics["loss"]), "Refinement loss must be finite."
    print("Static refinement checks passed.")
    print("student output shape:", tuple(refine_student_out.shape))
    print("refinement metrics:", refine_metrics)

    # ── tiny overfit check ──
    tiny_n = min(8, len(dataset_refine_train))
    df_tiny = cdf_wide_df.iloc[train_idx[:tiny_n]].reset_index(drop=True)
    dataset_tiny = CDFRefinementSequenceDataset(df_tiny, **ds_kwargs)
    loader_tiny = DataLoader(dataset_tiny, batch_size=len(dataset_tiny), shuffle=False)

    tiny_model = build_student_model_from_teacher(teacher_model, train_config, device)
    tiny_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=refine_config["learning_rate"], weight_decay=refine_config["weight_decay"])
    tiny_batch = next(iter(loader_tiny))
    tiny_batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in tiny_batch.items()}

    overfit_history = []
    for step in range(30):
        tiny_model.train()
        tiny_optimizer.zero_grad(set_to_none=True)
        features = tiny_batch_device["features"]
        a_sc = tiny_batch_device["a_scale"]
        student_out = tiny_model(features.reshape(-1, features.shape[-1])).reshape(features.shape[0], features.shape[1], 3)
        t_mu, _, t_var = compute_teacher_outputs(
            teacher_model, features, a_sc,
            log_var_bounds=refine_config["log_var_bounds"],
            nll_eps=refine_config["nll_eps"],
            std_clamp_min=float(refine_config.get("std_clamp_min", 0.0)),
            is_engineered_v2=is_engineered_v2,
        )
        loss, metrics = refinement_loss(
            student_out, a_sc, t_mu, t_var, tiny_batch_device,
            config=refine_config, is_engineered_v2=is_engineered_v2,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tiny_model.parameters(), 1.0)
        tiny_optimizer.step()
        overfit_history.append(metrics)

    print("tiny overfit final metrics:", overfit_history[-1])

    if args.no_train:
        print("--no-train specified; skipping full refinement.")
        return

    # ── full refinement training ──
    student_model_full = build_student_model_from_teacher(teacher_model, train_config, device)
    optimizer_full = torch.optim.AdamW(
        student_model_full.parameters(),
        lr=refine_config["learning_rate"],
        weight_decay=refine_config["weight_decay"],
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_refine = Path(refine_config["runs_root"]) / f"distill_cdf_onset_v2_{run_stamp}"
    run_dir_refine.mkdir(parents=True, exist_ok=False)
    (run_dir_refine / "refine_config.json").write_text(json.dumps(refine_config, indent=2, default=str), encoding="utf-8")
    (run_dir_refine / "teacher_run_dir.txt").write_text(str(artifacts.run_dir), encoding="utf-8")

    # Save train_config_used.json and scaler_state.json so load_run_artifacts works
    student_train_config = dict(train_config)
    student_train_config["output_dim"] = 3
    student_train_config["stage"] = "refinement"
    student_train_config["teacher_run_dir"] = str(artifacts.run_dir)
    (run_dir_refine / "train_config_used.json").write_text(json.dumps(student_train_config, indent=2, default=str), encoding="utf-8")
    (run_dir_refine / "scaler_state.json").write_text(json.dumps(scaler_state, indent=2, default=str), encoding="utf-8")

    best_val = float("inf")
    patience_left = args.patience
    history = []

    for epoch in range(1, int(refine_config["epochs"]) + 1):
        train_metrics = run_refinement_epoch(
            student_model_full, teacher_model, train_loader_refine,
            optimizer=optimizer_full, device=device, config=refine_config,
            is_engineered_v2=is_engineered_v2,
        )
        val_metrics = run_refinement_epoch(
            student_model_full, teacher_model, val_loader_refine,
            optimizer=None, device=device, config=refine_config,
            is_engineered_v2=is_engineered_v2,
        )
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(f"epoch={epoch:03d} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} val_kd={val_metrics['kd_kl']:.6f}")

        if val_metrics["loss"] < best_val - 1e-5:
            best_val = val_metrics["loss"]
            patience_left = args.patience
            torch.save(student_model_full.state_dict(), run_dir_refine / "best_model_refinement.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    # ── test evaluation ──
    best_state = torch.load(run_dir_refine / "best_model_refinement.pt", map_location=device)
    student_model_full.load_state_dict(best_state)
    test_metrics = run_refinement_epoch(
        student_model_full, teacher_model, test_loader_refine,
        optimizer=None, device=device, config=refine_config,
        is_engineered_v2=is_engineered_v2,
    )
    print("test metrics:", test_metrics)

    # ── save outputs ──
    df_history = pd.DataFrame(history)
    df_history.to_csv(run_dir_refine / "epoch_loss.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(run_dir_refine / "test_summary.csv", index=False)

    # loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_history["epoch"], df_history["train_loss"], label="train")
    ax.plot(df_history["epoch"], df_history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Refinement loss curves")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir_refine / "loss_curves.png", dpi=150)
    plt.close(fig)

    print("Saved refinement run to:", run_dir_refine)


if __name__ == "__main__":
    main()
