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
import sys
import time
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
else:
    from .engineered_feature_common import (
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AVAILABLE_SOURCES = ["cdf", "bw_x", "bw_polar"]
DEFAULT_SOURCES = ["cdf"]

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

# ── regime labeling defaults ──
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
TIME_BIN_WEIGHT_MIN = 0.5
TIME_BIN_WEIGHT_MAX = 2.0
TEACHER_CONF_WEIGHT_MIN = 0.25
TEACHER_CONF_WEIGHT_MAX = 1.0


def value_or_default(value: Any, default: float | int) -> float | int:
    return default if value is None else value


def compute_n_regime_bins(regime_bin_ms: float, regime_time_max_ms: float) -> int:
    if regime_bin_ms <= 0:
        raise ValueError(f"regime_bin_ms must be > 0, got {regime_bin_ms}.")
    if regime_time_max_ms <= 0:
        raise ValueError(f"regime_time_max_ms must be > 0, got {regime_time_max_ms}.")
    return max(1, int(math.ceil(regime_time_max_ms / regime_bin_ms)))


def build_regime_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    regime_bin_ms = float(value_or_default(args.regime_bin_ms, BIN_MS))
    regime_time_max_ms = float(value_or_default(args.regime_time_max_ms, REGIME_TIME_MAX_MS))
    config = {
        "regime_bin_ms": regime_bin_ms,
        "regime_time_max_ms": regime_time_max_ms,
        "n_regime_bins": compute_n_regime_bins(regime_bin_ms, regime_time_max_ms),
        "uncertain_ratio": float(value_or_default(args.uncertain_ratio, UNCERTAIN_RATIO)),
        "teacher_ratio": float(value_or_default(args.teacher_ratio, TEACHER_RATIO)),
        "teacher_min_count": int(value_or_default(args.teacher_min_count, TEACHER_MIN_COUNT)),
        "consecutive_bins": int(value_or_default(args.consecutive_bins, CONSECUTIVE_BINS)),
        "time_bin_weight_min": float(value_or_default(args.time_bin_weight_min, TIME_BIN_WEIGHT_MIN)),
        "time_bin_weight_max": float(value_or_default(args.time_bin_weight_max, TIME_BIN_WEIGHT_MAX)),
    }
    validate_regime_config(config)
    return config


def validate_regime_config(config: dict[str, Any]) -> None:
    if not (0.0 <= float(config["uncertain_ratio"]) <= 1.0):
        raise ValueError(f"uncertain_ratio must be in [0, 1], got {config['uncertain_ratio']}.")
    if not (0.0 <= float(config["teacher_ratio"]) <= 1.0):
        raise ValueError(f"teacher_ratio must be in [0, 1], got {config['teacher_ratio']}.")
    if int(config["teacher_min_count"]) < 1:
        raise ValueError(f"teacher_min_count must be >= 1, got {config['teacher_min_count']}.")
    if int(config["consecutive_bins"]) < 1:
        raise ValueError(f"consecutive_bins must be >= 1, got {config['consecutive_bins']}.")
    if float(config["time_bin_weight_min"]) <= 0:
        raise ValueError(f"time_bin_weight_min must be > 0, got {config['time_bin_weight_min']}.")
    if float(config["time_bin_weight_max"]) < float(config["time_bin_weight_min"]):
        raise ValueError(
            "time_bin_weight_max must be >= time_bin_weight_min, "
            f"got {config['time_bin_weight_max']} < {config['time_bin_weight_min']}."
        )


def time_ms_to_regime_bins(time_ms: np.ndarray, *, regime_bin_ms: float, n_regime_bins: int) -> np.ndarray:
    return np.floor(np.asarray(time_ms) / regime_bin_ms).astype(int).clip(0, n_regime_bins - 1)


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


def normalize_sources(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_SOURCES)

    sources: list[str] = []
    for value in values:
        for item in str(value).split(","):
            token = item.strip()
            if not token:
                continue
            if token == "all":
                sources.extend(AVAILABLE_SOURCES)
                continue
            if token not in AVAILABLE_SOURCES:
                raise ValueError(
                    f"Unsupported source '{token}'. Expected one of: {', '.join(AVAILABLE_SOURCES)}."
                )
            sources.append(token)

    deduped = list(dict.fromkeys(sources))
    if "cdf" not in deduped:
        raise ValueError("Stage3 refinement requires 'cdf' in --sources for regime labeling.")
    return deduped


def merge_source_tables(source_tables: dict[str, pd.DataFrame], sources: list[str]) -> pd.DataFrame:
    merged = None
    for idx, source in enumerate(sources):
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

def wide_source_to_long(source_wide_df: pd.DataFrame, *, regime_config: dict[str, Any] | None = None) -> pd.DataFrame:
    if regime_config is None:
        regime_config = {
            "regime_bin_ms": BIN_MS,
            "n_regime_bins": N_BINS,
        }
    regime_bin_ms = float(regime_config["regime_bin_ms"])
    n_regime_bins = int(regime_config["n_regime_bins"])
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
    repeated["time_bin"] = time_ms_to_regime_bins(
        repeated["time_ms"],
        regime_bin_ms=regime_bin_ms,
        n_regime_bins=n_regime_bins,
    )
    return repeated.reset_index(drop=True)


def _first_consecutive(mask: np.ndarray, run_len: int) -> int | None:
    if len(mask) < run_len:
        return None
    for i in range(len(mask) - run_len + 1):
        if np.all(mask[i : i + run_len]):
            return i
    return None


def build_time_bin_regimes(cdf_long_df: pd.DataFrame, *, regime_config: dict[str, Any] | None = None) -> pd.DataFrame:
    if regime_config is None:
        regime_config = {
            "regime_bin_ms": BIN_MS,
            "n_regime_bins": N_BINS,
            "uncertain_ratio": UNCERTAIN_RATIO,
            "teacher_ratio": TEACHER_RATIO,
            "teacher_min_count": TEACHER_MIN_COUNT,
            "consecutive_bins": CONSECUTIVE_BINS,
        }
    regime_bin_ms = float(regime_config["regime_bin_ms"])
    n_regime_bins = int(regime_config["n_regime_bins"])
    uncertain_ratio = float(regime_config["uncertain_ratio"])
    teacher_ratio = float(regime_config["teacher_ratio"])
    teacher_min_count = int(regime_config["teacher_min_count"])
    consecutive_bins = int(regime_config["consecutive_bins"])

    df = cdf_long_df.copy()
    dedup = df.drop_duplicates(subset=REGIME_GROUP_COLS + CDF_SAMPLE_ID_COLS[1:] + ["time_bin"])

    rows = []
    for group_key, g in dedup.groupby(REGIME_GROUP_COLS, dropna=False):
        counts = (
            g.groupby("time_bin", dropna=False)
            .size()
            .reindex(range(n_regime_bins), fill_value=0)
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

        uncertain_mask = coverage_ratio[b_peak:] < uncertain_ratio
        rel_uncertain = _first_consecutive(uncertain_mask, consecutive_bins)
        b_uncertain_start = b_peak + rel_uncertain if rel_uncertain is not None else n_regime_bins

        teacher_mask = (coverage_ratio[b_uncertain_start:] < teacher_ratio) | (counts[b_uncertain_start:] < teacher_min_count)
        rel_teacher = _first_consecutive(teacher_mask, consecutive_bins)
        b_teacher_start = b_uncertain_start + rel_teacher if rel_teacher is not None else n_regime_bins

        for b in range(n_regime_bins):
            if b < b_uncertain_start:
                regime = "raw_reliable"
            elif b < b_teacher_start:
                regime = "raw_uncertain"
            else:
                regime = "teacher_only"

            row = {col: val for col, val in zip(REGIME_GROUP_COLS, group_key)}
            row.update({
                "time_bin": b,
                "time_bin_start_ms": b * regime_bin_ms,
                "time_bin_end_ms": (b + 1) * regime_bin_ms,
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
        self.regime_bin_ms = float(config.get("regime_bin_ms", BIN_MS))
        self.n_regime_bins = int(config.get("n_regime_bins", N_BINS))
        self.time_bins = time_ms_to_regime_bins(
            self.time_grid_ms,
            regime_bin_ms=self.regime_bin_ms,
            n_regime_bins=self.n_regime_bins,
        )

        self.time_cols = prefixed_columns(self.df, "time_ms_")
        self.pen_cols = prefixed_columns(self.df, "penetration_mm_")
        if len(self.time_cols) != len(self.pen_cols):
            raise ValueError("CDF wide table time/penetration columns do not match.")

        self._raw_weight_lut, self._kd_weight_lut = self._build_regime_weight_lookup(regime_bins_df)
        self.onset_bins = self._build_onset_bins()
        self.precompute = bool(config.get("precompute_dataset", True))
        self._cache: dict[str, torch.Tensor] | None = None
        if self.precompute and len(self.df) > 0:
            self._build_cache()

    def _build_regime_weight_lookup(self, regime_bins_df: pd.DataFrame) -> tuple[dict, dict]:
        raw_lut: dict[tuple, np.ndarray] = {}
        kd_lut: dict[tuple, np.ndarray] = {}
        for group_key, g in regime_bins_df.groupby(REGIME_GROUP_COLS, dropna=False):
            arr = np.full(self.n_regime_bins, "teacher_only", dtype=object)
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
                onset_bins[idx] = int(np.floor(first_t / self.regime_bin_ms))
        return onset_bins.clip(0, self.n_regime_bins - 1)

    def __len__(self) -> int:
        return len(self.df)

    def _build_sample(self, idx: int) -> dict[str, torch.Tensor]:
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

    def _build_cache(self) -> None:
        start = time.perf_counter()
        samples = [self._build_sample(idx) for idx in range(len(self.df))]
        keys = list(samples[0].keys())
        self._cache = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
        total_bytes = sum(t.nelement() * t.element_size() for t in self._cache.values())
        elapsed_s = time.perf_counter() - start
        print(
            f"Cached CDFRefinementSequenceDataset: {len(self.df)} samples, "
            f"{total_bytes / (1024 ** 2):.1f} MiB, {elapsed_s:.2f}s"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = int(idx)
        if self._cache is not None:
            return {key: value[idx] for key, value in self._cache.items()}
        return self._build_sample(idx)


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
    conf_weight = torch.clamp(
        float(config["sigma_conf_ref_mm"]) / teacher_std_phys,
        min=float(config.get("teacher_conf_weight_min", TEACHER_CONF_WEIGHT_MIN)),
        max=float(config.get("teacher_conf_weight_max", TEACHER_CONF_WEIGHT_MAX)),
    ).squeeze(-1)
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
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help=(
            "Raw penetration sources to load. Defaults to 'cdf'. Use 'all' or "
            "'cdf bw_x bw_polar' after generating all sources."
        ),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader worker count. Default is 0; keep 0 when precomputing on Windows.")
    pin_group = p.add_mutually_exclusive_group()
    pin_group.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None)
    pin_group.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    cache_group = p.add_mutually_exclusive_group()
    cache_group.add_argument("--precompute-dataset", dest="precompute_dataset", action="store_true", default=None,
                             help="Cache per-sample tensors once before training.")
    cache_group.add_argument("--no-precompute-dataset", dest="precompute_dataset", action="store_false",
                             help="Build sample tensors inside DataLoader __getitem__ each epoch.")
    p.add_argument("--no-train", action="store_true", help="Only run diagnostics, skip training.")
    p.add_argument("--save-figures", action="store_true")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed inherited from Stage-2 train_config (controls split, init, dropout).")
    p.add_argument("--run-name-prefix", default="distill_cdf_onset_v2",
                   help="Prefix for the refinement run directory under MLP/runs_mlp.")
    p.add_argument("--ablation-name", default=None,
                   help="Optional label stored in refine_config.json.")
    p.add_argument("--lono-holdout", type=str, default=None,
                   help="If set, treat all rows with experiment_name=<value> as the test "
                        "split (leave-one-nozzle-out OOD protocol). Train+val is carved "
                        "from the remaining experiments only.")

    loss_group = p.add_argument_group("Stage3 loss tuning")
    loss_group.add_argument("--lambda-anchor", type=float, default=None,
                            help="Override early-time anchor loss weight. Default: 1e-2.")
    loss_group.add_argument("--anchor-window-ms", type=float, default=None,
                            help="Override early-time anchor window length. Default: 0.15.")
    loss_group.add_argument("--lambda-onset", type=float, default=None,
                            help="Override onset BCE loss weight. Default: 0.1.")
    loss_group.add_argument("--onset-ramp-ms", type=float, default=None,
                            help="Override onset target ramp length. Default: 0.12.")
    loss_group.add_argument("--onset-loss-window-ms", type=float, default=None,
                            help="Override onset BCE time window. Default: 0.2.")
    loss_group.add_argument("--d1-positive-weight", type=float, default=None,
                            help="Override positive-slope penalty weight. Default: 5e-5.")
    loss_group.add_argument("--d2-concave-weight", type=float, default=None,
                            help="Override concavity penalty weight. Default: 5e-4.")
    loss_group.add_argument("--d2-start-ms", type=float, default=None,
                            help="Override d2 penalty sigmoid start time. Default: 0.5.")
    loss_group.add_argument("--d2-transition-ms", type=float, default=None,
                            help="Override d2 penalty sigmoid transition width. Default: 0.05.")
    loss_group.add_argument("--log-var-min", type=float, default=None,
                            help="Override minimum log-variance clamp. Default inherits teacher config or -10.0.")
    loss_group.add_argument("--log-var-max", type=float, default=None,
                            help="Override maximum log-variance clamp. Default inherits teacher config or 6.0.")
    loss_group.add_argument("--nll-eps", type=float, default=None,
                            help="Override variance epsilon in NLL/KL terms. Default inherits teacher config or 1e-12.")
    loss_group.add_argument("--std-clamp-min", type=float, default=None,
                            help="Override teacher std clamp minimum. Default inherits teacher config or 1e-3.")
    loss_group.add_argument("--sigma-conf-ref-mm", type=float, default=None,
                            help="Override teacher confidence reference sigma in mm. Default: 10.0.")
    loss_group.add_argument("--teacher-conf-weight-min", type=float, default=None,
                            help="Override lower clamp for KD confidence weights. Default: 0.25.")
    loss_group.add_argument("--teacher-conf-weight-max", type=float, default=None,
                            help="Override upper clamp for KD confidence weights. Default: 1.0.")

    loss_group.add_argument("--raw-reliable-raw-weight", type=float, default=None)
    loss_group.add_argument("--raw-reliable-kd-weight", type=float, default=None)
    loss_group.add_argument("--raw-uncertain-raw-weight", type=float, default=None)
    loss_group.add_argument("--raw-uncertain-kd-weight", type=float, default=None)
    loss_group.add_argument("--teacher-only-raw-weight", type=float, default=None)
    loss_group.add_argument("--teacher-only-kd-weight", type=float, default=None)

    loss_group.add_argument("--regime-bin-ms", type=float, default=None,
                            help="Override regime time-bin width. Default: 0.1.")
    loss_group.add_argument("--regime-time-max-ms", type=float, default=None,
                            help="Override regime labeling horizon. Default: 5.0.")
    loss_group.add_argument("--uncertain-ratio", type=float, default=None,
                            help="Override raw-uncertain coverage ratio threshold. Default: 0.7.")
    loss_group.add_argument("--teacher-ratio", type=float, default=None,
                            help="Override teacher-only coverage ratio threshold. Default: 0.2.")
    loss_group.add_argument("--teacher-min-count", type=int, default=None,
                            help="Override teacher-only minimum raw count threshold. Default: 4.")
    loss_group.add_argument("--consecutive-bins", type=int, default=None,
                            help="Override consecutive bins required for a regime transition. Default: 2.")
    loss_group.add_argument("--time-bin-weight-min", type=float, default=None,
                            help="Override lower clamp for global time-bin reweighting. Default: 0.5.")
    loss_group.add_argument("--time-bin-weight-max", type=float, default=None,
                            help="Override upper clamp for global time-bin reweighting. Default: 2.0.")

    p.add_argument("--skip-post-train-eval", action="store_true",
                   help="Skip automatic RMSE inference evaluation after training.")
    p.add_argument("--eval-split", choices=("clean", "all"), default="clean",
                   help="series_wide split for automatic post-training RMSE evaluation.")
    p.add_argument("--eval-t-min-ms", type=float, default=0.0)
    p.add_argument("--eval-t-max-ms", type=float, default=5.0)
    p.add_argument("--eval-rel-err-floor-mm", type=float, default=5.0)
    p.add_argument("--eval-output-root", type=Path, default=MLP_ROOT / "eval")
    p.add_argument("--eval-tag", type=str, default=None,
                   help="Optional extra tag appended to the automatic RMSE evaluation folder.")
    p.add_argument("--eval-batch-points", type=int, default=65536)
    p.add_argument("--eval-fast", action="store_true",
                   help="Run post-training RMSE evaluation in metrics-only fast mode.")
    eval_points_group = p.add_mutually_exclusive_group()
    eval_points_group.add_argument("--eval-save-points", dest="eval_save_points", action="store_true", default=None)
    eval_points_group.add_argument("--no-eval-save-points", dest="eval_save_points", action="store_false")
    eval_plots_group = p.add_mutually_exclusive_group()
    eval_plots_group.add_argument("--eval-save-plots", dest="eval_save_plots", action="store_true", default=None)
    eval_plots_group.add_argument("--no-eval-save-plots", dest="eval_save_plots", action="store_false")
    p.add_argument("--eval-max-traj-plots", type=int, default=None)
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_run_name(text: str) -> str:
    """Keep run directory names portable across Windows and WSL."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text).strip())
    return cleaned.strip("_") or "distill_cdf_onset_v2"


def apply_refine_config_overrides(refine_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI ablation overrides while recording exactly what changed."""
    overrides: dict[str, Any] = {}

    scalar_map = {
        "lambda_anchor": args.lambda_anchor,
        "anchor_window_ms": args.anchor_window_ms,
        "lambda_onset": args.lambda_onset,
        "onset_ramp_ms": args.onset_ramp_ms,
        "onset_loss_window_ms": args.onset_loss_window_ms,
        "d1_positive_weight": args.d1_positive_weight,
        "d2_concave_weight": args.d2_concave_weight,
        "d2_start_ms": args.d2_start_ms,
        "d2_transition_ms": args.d2_transition_ms,
        "nll_eps": args.nll_eps,
        "std_clamp_min": args.std_clamp_min,
        "sigma_conf_ref_mm": args.sigma_conf_ref_mm,
        "teacher_conf_weight_min": args.teacher_conf_weight_min,
        "teacher_conf_weight_max": args.teacher_conf_weight_max,
        "regime_bin_ms": args.regime_bin_ms,
        "regime_time_max_ms": args.regime_time_max_ms,
        "uncertain_ratio": args.uncertain_ratio,
        "teacher_ratio": args.teacher_ratio,
        "time_bin_weight_min": args.time_bin_weight_min,
        "time_bin_weight_max": args.time_bin_weight_max,
    }
    for key, value in scalar_map.items():
        if value is not None:
            refine_config[key] = float(value)
            overrides[key] = float(value)

    int_map = {
        "teacher_min_count": args.teacher_min_count,
        "consecutive_bins": args.consecutive_bins,
    }
    for key, value in int_map.items():
        if value is not None:
            refine_config[key] = int(value)
            overrides[key] = int(value)

    log_var_bounds = list(refine_config["log_var_bounds"])
    log_var_changed = False
    if args.log_var_min is not None:
        log_var_bounds[0] = float(args.log_var_min)
        log_var_changed = True
    if args.log_var_max is not None:
        log_var_bounds[1] = float(args.log_var_max)
        log_var_changed = True
    if log_var_changed:
        refine_config["log_var_bounds"] = tuple(log_var_bounds)
        overrides["log_var_bounds"] = [float(log_var_bounds[0]), float(log_var_bounds[1])]

    if args.num_workers is not None:
        refine_config["num_workers"] = int(args.num_workers)
        overrides["num_workers"] = int(args.num_workers)
    if args.pin_memory is not None:
        refine_config["pin_memory"] = bool(args.pin_memory)
        overrides["pin_memory"] = bool(args.pin_memory)
    if args.precompute_dataset is not None:
        refine_config["precompute_dataset"] = bool(args.precompute_dataset)
        overrides["precompute_dataset"] = bool(args.precompute_dataset)

    if any(value is not None for value in (args.regime_bin_ms, args.regime_time_max_ms)):
        refine_config["n_regime_bins"] = compute_n_regime_bins(
            float(refine_config["regime_bin_ms"]),
            float(refine_config["regime_time_max_ms"]),
        )
        overrides["n_regime_bins"] = int(refine_config["n_regime_bins"])

    weight_map = {
        ("raw_weights", "raw_reliable"): args.raw_reliable_raw_weight,
        ("kd_weights", "raw_reliable"): args.raw_reliable_kd_weight,
        ("raw_weights", "raw_uncertain"): args.raw_uncertain_raw_weight,
        ("kd_weights", "raw_uncertain"): args.raw_uncertain_kd_weight,
        ("raw_weights", "teacher_only"): args.teacher_only_raw_weight,
        ("kd_weights", "teacher_only"): args.teacher_only_kd_weight,
    }
    for (weight_group, regime), value in weight_map.items():
        if value is not None:
            refine_config[weight_group][regime] = float(value)
            overrides.setdefault(weight_group, {})[regime] = float(value)

    refine_config["run_name_prefix"] = sanitize_run_name(args.run_name_prefix)
    if args.ablation_name:
        refine_config["ablation_name"] = str(args.ablation_name)
    if overrides:
        refine_config["cli_overrides"] = overrides
        print()
        print("Applied refinement ablation overrides:")
        print(json.dumps(overrides, indent=2, default=str))

    validate_refine_config(refine_config)
    return refine_config


def validate_refine_config(refine_config: dict[str, Any]) -> None:
    log_var_min, log_var_max = [float(x) for x in refine_config["log_var_bounds"]]
    if log_var_max < log_var_min:
        raise ValueError(f"log_var_max must be >= log_var_min, got {log_var_max} < {log_var_min}.")
    if float(refine_config["nll_eps"]) <= 0:
        raise ValueError(f"nll_eps must be > 0, got {refine_config['nll_eps']}.")
    if float(refine_config["std_clamp_min"]) < 0:
        raise ValueError(f"std_clamp_min must be >= 0, got {refine_config['std_clamp_min']}.")
    if float(refine_config["teacher_conf_weight_min"]) < 0:
        raise ValueError(
            f"teacher_conf_weight_min must be >= 0, got {refine_config['teacher_conf_weight_min']}."
        )
    if float(refine_config["teacher_conf_weight_max"]) < float(refine_config["teacher_conf_weight_min"]):
        raise ValueError(
            "teacher_conf_weight_max must be >= teacher_conf_weight_min, "
            f"got {refine_config['teacher_conf_weight_max']} < {refine_config['teacher_conf_weight_min']}."
        )
    if int(refine_config.get("num_workers", 0)) < 0:
        raise ValueError(f"num_workers must be >= 0, got {refine_config['num_workers']}.")
    validate_regime_config(refine_config)


def run_post_training_rmse_eval(
    *,
    run_dir_refine: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Run the same RMSE analysis as MLP/eval after training."""
    from MLP.eval.inference_rmse_on_series import run_rmse_evaluation

    eval_tag = args.eval_tag or run_dir_refine.name
    print()
    print("Running automatic post-training RMSE evaluation...")
    out_dir, summary = run_rmse_evaluation(
        refinement_run=run_dir_refine,
        split=args.eval_split,
        device=device,
        t_min_ms=float(args.eval_t_min_ms),
        t_max_ms=float(args.eval_t_max_ms),
        rel_err_floor_mm=float(args.eval_rel_err_floor_mm),
        output_root=args.eval_output_root,
        tag=eval_tag,
        batch_points=int(args.eval_batch_points),
        fast=bool(args.eval_fast),
        save_points=args.eval_save_points,
        save_plots=args.eval_save_plots,
        max_traj_plots=args.eval_max_traj_plots,
    )
    pointer = {
        "eval_output_dir": str(out_dir),
        "metrics_summary": str(out_dir / "metrics_summary.json"),
        "split": args.eval_split,
        "overall": summary.get("overall", {}),
    }
    (run_dir_refine / "post_train_rmse_eval.json").write_text(
        json.dumps(pointer, indent=2, default=str),
        encoding="utf-8",
    )
    print("Post-training RMSE evaluation saved to:", out_dir)


def main() -> None:
    args = parse_args()
    sources = normalize_sources(args.sources)
    regime_config = build_regime_config_from_args(args)

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
    print(f"Raw sources: {sources}")
    print(teacher_model)

    # ── load raw series ──
    source_tables = {source: load_source_table(source, split=args.series_split) for source in sources}
    for source, table in source_tables.items():
        print(source, table.shape)

    raw_series_df = merge_source_tables(source_tables, sources)
    print("merged shape:", raw_series_df.shape)

    # ── tensorization ──
    all_frame_ids = sorted({
        frame_id
        for source in sources
        for frame_id in available_frame_ids(raw_series_df, f"{source}__penetration_mm_")
    })

    series_stack = np.stack(
        [extract_prefixed_matrix(raw_series_df, f"{source}__penetration_mm_", all_frame_ids) for source in sources],
        axis=-1,
    )
    time_stack = np.stack(
        [extract_prefixed_matrix(raw_series_df, f"{source}__time_ms_", all_frame_ids) for source in sources],
        axis=-1,
    )

    mask_stack = np.isfinite(series_stack)
    series_tensor = np.nan_to_num(series_stack, nan=0.0).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        time_ms_tensor = np.nanmean(np.where(np.isfinite(time_stack), time_stack, np.nan), axis=-1).astype(np.float32)

    meta_cols = MERGE_KEYS + COMMON_META_COLS + [
        f"{src}__{col}"
        for src in sources
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
    cdf_long_df = wide_source_to_long(cdf_wide_df, regime_config=regime_config)
    cdf_regime_bins_df = build_time_bin_regimes(cdf_long_df, regime_config=regime_config)
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
        "pin_memory": bool(device.type == "cuda"),
        "precompute_dataset": True,
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
        "teacher_conf_weight_min": TEACHER_CONF_WEIGHT_MIN,
        "teacher_conf_weight_max": TEACHER_CONF_WEIGHT_MAX,
        "raw_weights": {"raw_reliable": 1.0, "raw_uncertain": 0.0, "teacher_only": 0.0},
        "kd_weights": {"raw_reliable": 0.25, "raw_uncertain": 1.0, "teacher_only": 0.75},
        "runs_root": str(MLP_ROOT / "runs_mlp"),
        **regime_config,
    }
    apply_refine_config_overrides(refine_config, args)
    if getattr(args, "seed", None) is not None:
        refine_config["seed"] = int(args.seed)
    set_global_seed(refine_config["seed"])

    raw_time_bin_counts = cdf_labeled_df.groupby("time_bin", dropna=False).size().reindex(
        range(int(refine_config["n_regime_bins"])),
        fill_value=0,
    ).astype(float)
    raw_time_bin_reference = float(raw_time_bin_counts[raw_time_bin_counts > 0].mean()) if np.any(raw_time_bin_counts > 0) else 1.0
    global_time_bin_weights = (
        raw_time_bin_reference / raw_time_bin_counts.replace(0.0, np.nan)
    ).fillna(1.0).clip(
        float(refine_config["time_bin_weight_min"]),
        float(refine_config["time_bin_weight_max"]),
    ).to_numpy(dtype=np.float32)

    # ── build refinement datasets ──
    if getattr(args, "lono_holdout", None) is not None:
        # Leave-one-nozzle-out: held-out experiment_name -> entire test split.
        # Train+val carved from remaining experiments via split_indices.
        if "experiment_name" not in cdf_wide_df.columns:
            raise KeyError("cdf_wide_df missing experiment_name column required for LONO split.")
        is_holdout = cdf_wide_df["experiment_name"].astype(str) == str(args.lono_holdout)
        n_holdout = int(is_holdout.sum())
        if n_holdout == 0:
            raise ValueError(
                f"--lono-holdout={args.lono_holdout!r} matched 0 rows in cdf_wide_df."
            )
        remaining_idx = np.where(~is_holdout.to_numpy())[0]
        holdout_idx = np.where(is_holdout.to_numpy())[0]
        # Use split_indices to carve train+val from remaining; ignore its test slice.
        rng = np.random.default_rng(int(refine_config["seed"]))
        shuffled = remaining_idx.copy()
        rng.shuffle(shuffled)
        n_remaining = len(shuffled)
        n_val = int(round(n_remaining * float(refine_config["val_frac"])))
        n_val = min(max(n_val, 1), max(n_remaining - 1, 1))
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        test_idx = holdout_idx
        print(
            f"LONO split: holdout='{args.lono_holdout}', "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )
    else:
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
    run_dir_refine = Path(refine_config["runs_root"]) / f"{refine_config['run_name_prefix']}_{run_stamp}"
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
    if args.skip_post_train_eval:
        print("Automatic post-training RMSE evaluation skipped.")
    else:
        run_post_training_rmse_eval(run_dir_refine=run_dir_refine, device=device, args=args)


if __name__ == "__main__":
    main()
