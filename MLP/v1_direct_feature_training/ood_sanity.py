from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_CONTINUOUS_COLS = (
    "tilt_angle_radian",
    "diameter_mm",
    "injection_duration_us",
    "injection_pressure_bar",
    "chamber_pressure_bar",
    "control_backpressure_bar",
)
DEFAULT_DISCRETE_COLS = ("plumes",)
DEFAULT_COMBO_COLS = (
    "injection_pressure_bar",
    "chamber_pressure_bar",
    "control_backpressure_bar",
)


@dataclass(frozen=True)
class SupportConfig:
    continuous_cols: tuple[str, ...] = DEFAULT_CONTINUOUS_COLS
    discrete_cols: tuple[str, ...] = DEFAULT_DISCRETE_COLS
    combo_cols: tuple[str, ...] = DEFAULT_COMBO_COLS
    z_warn: float = 2.0
    z_strong: float = 3.0
    minmax_tol: float = 1e-9
    nearest_combo_topk: int = 3


def _default_audit_csv(project_root: Path) -> Path:
    return project_root / "MLP" / "figures" / "fit_bias_audit_cdf" / "cdf_plume_audit.csv"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(round(float(value)))
    except Exception:
        return None


def _build_continuous_support(df: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, dict[str, float]]:
    support: dict[str, dict[str, float]] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            support[col] = {
                "count": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "p01": np.nan,
                "p99": np.nan,
            }
            continue
        support[col] = {
            "count": int(valid.size),
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=0)),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "p01": float(valid.quantile(0.01)),
            "p99": float(valid.quantile(0.99)),
        }
    return support


def _build_discrete_support(df: pd.DataFrame, cols: tuple[str, ...]) -> dict[str, list[Any]]:
    support: dict[str, list[Any]] = {}
    for col in cols:
        values = pd.Series(df[col]).dropna().unique().tolist()
        numeric_values = []
        fallback_values = []
        for value in values:
            ivalue = _safe_int(value)
            if ivalue is None:
                fallback_values.append(value)
            else:
                numeric_values.append(ivalue)
        support[col] = sorted(set(numeric_values)) if numeric_values else sorted(set(fallback_values))
    return support


def _build_combo_support(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    combo_df = df.loc[:, list(cols)].copy()
    for col in cols:
        combo_df[col] = pd.to_numeric(combo_df[col], errors="coerce")
    combo_df = combo_df.dropna().drop_duplicates().reset_index(drop=True)
    return combo_df


def _normalize_support_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "tilt_angle_radian" not in out.columns and "umbrella_angle_deg" in out.columns:
        umbrella = pd.to_numeric(out["umbrella_angle_deg"], errors="coerce")
        out["tilt_angle_radian"] = np.deg2rad((180.0 - umbrella) / 2.0)
    return out


def build_empirical_support(
    df: pd.DataFrame,
    *,
    config: SupportConfig | None = None,
    split_filter: str | None = "clean",
) -> dict[str, Any]:
    cfg = config or SupportConfig()
    work_df = _normalize_support_df(df)
    if split_filter is not None:
        if "sample_split" in work_df.columns:
            work_df = work_df.loc[work_df["sample_split"] == split_filter].copy()
        elif split_filter == "clean" and "flag_bad_fit" in work_df.columns:
            work_df = work_df.loc[~work_df["flag_bad_fit"].fillna(False)].copy()
    if work_df.empty:
        raise ValueError("No rows available after applying the requested split filter.")

    missing_cols = [
        col
        for col in (*cfg.continuous_cols, *cfg.discrete_cols, *cfg.combo_cols)
        if col not in work_df.columns
    ]
    if missing_cols:
        raise KeyError(f"Support dataframe is missing required columns: {missing_cols}")

    support = {
        "n_rows": int(len(work_df)),
        "split_filter": split_filter,
        "continuous_cols": cfg.continuous_cols,
        "discrete_cols": cfg.discrete_cols,
        "combo_cols": cfg.combo_cols,
        "continuous": _build_continuous_support(work_df, cfg.continuous_cols),
        "discrete": _build_discrete_support(work_df, cfg.discrete_cols),
        "combo_df": _build_combo_support(work_df, cfg.combo_cols),
        "config": {
            "z_warn": cfg.z_warn,
            "z_strong": cfg.z_strong,
            "minmax_tol": cfg.minmax_tol,
            "nearest_combo_topk": cfg.nearest_combo_topk,
        },
    }
    return support


def load_cdf_empirical_support(
    project_root: Path,
    *,
    audit_csv_path: Path | None = None,
    config: SupportConfig | None = None,
    split_filter: str | None = "clean",
) -> dict[str, Any]:
    csv_path = (audit_csv_path or _default_audit_csv(project_root)).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find CDF audit table at {csv_path}. "
            "Run MLP/analyze_cdf_fit_bias.py first or pass audit_csv_path explicitly."
        )
    audit_df = pd.read_csv(csv_path, low_memory=False)
    support = build_empirical_support(audit_df, config=config, split_filter=split_filter)
    support["audit_csv_path"] = str(csv_path)
    return support


def _nearest_combo_rows(
    sample: Mapping[str, Any],
    combo_df: pd.DataFrame,
    cols: tuple[str, ...],
    *,
    topk: int,
) -> list[dict[str, Any]]:
    if combo_df.empty:
        return []

    sample_vec = []
    scale = []
    valid_cols: list[str] = []
    for col in cols:
        sample_value = _safe_float(sample.get(col, np.nan))
        col_values = pd.to_numeric(combo_df[col], errors="coerce")
        col_std = float(col_values.std(ddof=0)) if col_values.notna().any() else np.nan
        if not np.isfinite(sample_value) or not np.isfinite(col_std) or col_std <= 1e-12:
            continue
        sample_vec.append(sample_value)
        scale.append(col_std)
        valid_cols.append(col)

    if not valid_cols:
        return []

    mat = combo_df.loc[:, valid_cols].to_numpy(dtype=float)
    d = np.sqrt(np.sum(np.square((mat - np.asarray(sample_vec)) / np.asarray(scale)), axis=1))
    order = np.argsort(d)[:topk]
    rows: list[dict[str, Any]] = []
    for idx in order:
        item = {col: _safe_float(combo_df.iloc[idx][col]) for col in valid_cols}
        item["distance"] = float(d[idx])
        rows.append(item)
    return rows


def check_input_sanity(
    sample: Mapping[str, Any],
    support: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = support["config"]
    continuous_checks: list[dict[str, Any]] = []
    discrete_checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    severity = "ok"

    for col in support["continuous_cols"]:
        stats = support["continuous"][col]
        value = _safe_float(sample.get(col, np.nan))
        std = _safe_float(stats["std"])
        z_score = np.nan
        if np.isfinite(value) and np.isfinite(std) and std > 1e-12:
            z_score = (value - float(stats["mean"])) / std
        outside_minmax = False
        if np.isfinite(value) and np.isfinite(stats["min"]) and np.isfinite(stats["max"]):
            outside_minmax = bool(
                value < float(stats["min"]) - float(cfg["minmax_tol"])
                or value > float(stats["max"]) + float(cfg["minmax_tol"])
            )
        level = "ok"
        if outside_minmax or (np.isfinite(z_score) and abs(z_score) >= float(cfg["z_strong"])):
            level = "strong"
            severity = "strong"
            warnings.append(
                f"{col}={value:.4g} is outside empirical support "
                f"(z={z_score:.2f}, range=[{stats['min']:.4g}, {stats['max']:.4g}])."
            )
        elif np.isfinite(z_score) and abs(z_score) >= float(cfg["z_warn"]) and severity == "ok":
            level = "warn"
            severity = "warn"
            warnings.append(f"{col}={value:.4g} is far from the training center (z={z_score:.2f}).")
        continuous_checks.append(
            {
                "feature": col,
                "value": value,
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "z_score": z_score,
                "outside_minmax": outside_minmax,
                "level": level,
            }
        )

    for col in support["discrete_cols"]:
        value = sample.get(col, np.nan)
        int_value = _safe_int(value)
        supported_values = support["discrete"][col]
        seen = int_value in supported_values if int_value is not None else value in supported_values
        level = "ok" if seen else "strong"
        if not seen:
            severity = "strong"
            warnings.append(f"{col}={value} was not observed in the training support.")
        discrete_checks.append(
            {
                "feature": col,
                "value": value,
                "supported_values": supported_values,
                "seen": seen,
                "level": level,
            }
        )

    combo_cols = tuple(support["combo_cols"])
    combo_df = support["combo_df"]
    combo_query = {col: _safe_float(sample.get(col, np.nan)) for col in combo_cols}
    combo_exact_match = False
    if not combo_df.empty and all(np.isfinite(combo_query[col]) for col in combo_cols):
        mask = np.ones(len(combo_df), dtype=bool)
        for col in combo_cols:
            mask &= np.isclose(
                pd.to_numeric(combo_df[col], errors="coerce").to_numpy(dtype=float),
                combo_query[col],
                atol=1e-9,
                rtol=0.0,
            )
        combo_exact_match = bool(mask.any())
    nearest_seen = _nearest_combo_rows(
        sample,
        combo_df,
        combo_cols,
        topk=int(cfg["nearest_combo_topk"]),
    )
    if not combo_exact_match:
        severity = "strong"
        combo_desc = ", ".join(f"{col}={combo_query[col]:.4g}" for col in combo_cols if np.isfinite(combo_query[col]))
        warnings.append(f"Condition combination not observed in training support: {combo_desc}.")

    is_ood = severity != "ok" or not combo_exact_match
    return {
        "is_ood": bool(is_ood),
        "severity": severity,
        "warnings": warnings,
        "continuous_checks": continuous_checks,
        "discrete_checks": discrete_checks,
        "combo_cols": combo_cols,
        "combo_query": combo_query,
        "combo_exact_match": combo_exact_match,
        "nearest_seen_conditions": nearest_seen,
        "audit_csv_path": support.get("audit_csv_path"),
        "n_support_rows": support.get("n_rows"),
    }


def format_sanity_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"Sanity check severity: {report['severity']}",
        f"OOD warning: {report['is_ood']}",
        f"Exact combo match: {report['combo_exact_match']}",
    ]
    if report.get("audit_csv_path"):
        lines.append(f"Support source: {report['audit_csv_path']}")
    if report.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in report["warnings"])
    nearest = report.get("nearest_seen_conditions", [])
    if nearest:
        lines.append("Nearest seen conditions:")
        for item in nearest:
            desc = ", ".join(f"{key}={value:.4g}" for key, value in item.items() if key != "distance")
            lines.append(f"- {desc} (distance={item['distance']:.3f})")
    return "\n".join(lines)


def concise_warning_text(report: Mapping[str, Any], *, max_lines: int = 4) -> str:
    if not report.get("is_ood"):
        return "Input sanity check: within empirical training support."
    lines = ["Warning: empirical training-support check indicates OOD or weak support."]
    for warning in report.get("warnings", [])[: max(0, max_lines - 1)]:
        lines.append(warning)
    return "\n".join(lines[:max_lines])
