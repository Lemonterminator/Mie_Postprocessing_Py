from __future__ import annotations

"""Artifact saving, run-directory management, and loss-curve plotting.

create_run_dir          — create a timestamped run directory under runs_root and
                           write a run_manifest.json with label, variant, timestamp.
save_training_outputs   — save model checkpoint, train_config_used.json, scaler_state.json,
                           row_table.csv, iter_history.csv, epoch_history.csv, and
                           a precheck_report.json to the run directory.
plot_loss_curves        — generate per-epoch loss / physical-MAE / shape-penalty plots
                           as PNG in the run directory.
sanitize_config_for_json — convert Path, tuple, and other non-JSON-native values to
                            JSON-serialisable equivalents before json.dump.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def sanitize_config_for_json(config: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        elif isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [str(item) if isinstance(item, Path) else item for item in value]
        elif isinstance(value, dict):
            out[key] = sanitize_config_for_json(value)
    return out


def create_run_dir(
    runs_root: Path | str,
    prefix: str,
    variant: str,
    *,
    parent_run_id: str | None = None,
    parent_run_ids: Mapping[str, str] | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / f"{prefix}_{variant}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    parents: dict[str, str] = {}
    if os.environ.get("MLP_PARENT_FIT_RUN_ID"):
        parents["fit"] = str(os.environ["MLP_PARENT_FIT_RUN_ID"])
    if os.environ.get("MLP_PARENT_AUDIT_RUN_ID"):
        parents["audit"] = str(os.environ["MLP_PARENT_AUDIT_RUN_ID"])
    if parent_run_id:
        parents["parent"] = str(parent_run_id)
    if parent_run_ids:
        parents.update({str(k): str(v) for k, v in parent_run_ids.items() if v is not None})
    if parents:
        metadata = {
            "phase": "train",
            "run_id": run_dir.name,
            "created_at": datetime.now().isoformat(),
            "prefix": prefix,
            "variant": variant,
            "parent_run_ids": parents,
        }
        (run_dir / "_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return run_dir


def save_training_outputs(
    run_dir: Path,
    *,
    model: nn.Module,
    checkpoint_name: str,
    train_config: Mapping[str, Any],
    scaler_state: Mapping[str, Any],
    row_table: pd.DataFrame,
    iter_history: pd.DataFrame,
    epoch_history: pd.DataFrame,
    precheck_report: Mapping[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / checkpoint_name)
    iter_history.to_csv(run_dir / "iteration_loss.csv", index=False)
    epoch_history.to_csv(run_dir / "epoch_loss.csv", index=False)
    row_table.to_csv(run_dir / "row_table.csv", index=False)
    with (run_dir / "scaler_state.json").open("w", encoding="utf-8") as f:
        json.dump(scaler_state, f, indent=2)
    with (run_dir / "train_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(sanitize_config_for_json(train_config), f, indent=2)
    if precheck_report is not None:
        with (run_dir / "pretrain_collapse_report.json").open("w", encoding="utf-8") as f:
            json.dump(dict(precheck_report), f, indent=2)


def plot_loss_curves(epoch_history: pd.DataFrame, run_dir: Path, *, objective_name: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for split in ("train", "val", "test"):
        group = epoch_history.loc[epoch_history["split"] == split]
        if group.empty:
            continue
        axes[0].plot(group["epoch"], group["loss"], marker="o", label=split)
        if "physical_mae" in group.columns:
            axes[1].plot(group["epoch"], group["physical_mae"], marker="o", label=split)
        if "d1_penalty" in group.columns:
            axes[2].plot(group["epoch"], group["d1_penalty"], marker="o", label=f"{split} d1")
        if "d2_penalty" in group.columns:
            axes[2].plot(group["epoch"], group["d2_penalty"], marker="x", label=f"{split} d2")
    axes[0].set_title("Epoch loss")
    axes[1].set_title("Epoch physical MAE")
    axes[2].set_title("Epoch shape penalties")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = run_dir / f"{objective_name}_loss_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
