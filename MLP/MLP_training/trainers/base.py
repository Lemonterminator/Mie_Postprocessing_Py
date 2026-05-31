from __future__ import annotations

"""Template-method base class for all stage trainers.

TrainerBase.run() owns the complete orchestration: parse args → merge config →
set random seed → build stage tables → assign splits → build row table → build
model → train → save outputs.  Subclasses supply stage-specific logic through
~10 abstract hook methods; optional hooks have sensible no-op defaults.

Hook execution order inside run():
    1.  parse_args()
    2.  extra_overrides(args)            → stage-specific CLI → config mappings;
                                           may stash computed state on self.
    3.  merge_config + post_merge_adjustments(config, args)
    4.  build_all_stage_tables(...)      → produces StageTables
    5.  row_selection_df(stage_tables)   → .representative (Stage-1) or .filtered (Stage-2)
    6.  assign_splits_*()
    7.  build_row_table(split_df, config) → (row_table, scaler_state, feature_columns)
    8.  make_dataloaders(...)
    9.  on_dataloaders_ready(...)
    10. build_model(config, device)
    11. train_with_early_stopping(...)
    12. save_training_outputs(...)
    13. on_finish(run_dir, ...)
"""

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engineered_feature_common import (
    StageTables,
    assign_splits_by_group,
    assign_splits_leave_one_out,
    build_all_stage_tables,
    build_dataset_registry,
    create_run_dir,
    make_dataloaders,
    merge_config,
    plot_loss_curves,
    save_training_outputs,
    train_with_early_stopping,
)


def _set_if_not_none(d: dict[str, Any], key: str, value: Any, cast: type) -> None:
    if value is not None:
        d[key] = cast(value)


class TrainerBase(ABC):
    """Template-method base for stage trainers.

    run() is the sealed orchestration loop; subclasses implement abstract hooks
    and may override optional hooks for stage-specific behaviour (e.g. Stage-2
    GPU pre-loading, anchor lambda defaults).

    Instance state set in extra_overrides() (e.g. self._stage1_artifacts for
    Stage-2) is available to all subsequent hooks because run() calls
    extra_overrides() before any hook that reads it.
    """

    # ------------------------------------------------------------------ abstract

    @abstractmethod
    def parse_args(self) -> argparse.Namespace: ...

    @abstractmethod
    def default_config(self) -> dict[str, Any]: ...

    @abstractmethod
    def extra_overrides(self, args: argparse.Namespace) -> dict[str, Any]:
        """Stage-specific CLI args → config overrides.

        Called before merge_config; may stash computed values on self.
        """
        ...

    @abstractmethod
    def run_label(self, config: dict[str, Any]) -> str:
        """Prefix for the run directory name (e.g. 'stage1_engineered_mse')."""
        ...

    @abstractmethod
    def get_a_scale_dp_exp(
        self, config: dict[str, Any], args: argparse.Namespace
    ) -> float:
        """Delta-pressure exponent passed to build_all_stage_tables."""
        ...

    @abstractmethod
    def row_selection_df(self, stage_tables: StageTables) -> pd.DataFrame:
        """Which sub-table to use for split assignment (.representative or .filtered)."""
        ...

    @abstractmethod
    def build_row_table(
        self, split_df: pd.DataFrame, config: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
        """Build (row_table, scaler_state, feature_columns) from the split dataframe."""
        ...

    @abstractmethod
    def build_model(
        self, config: dict[str, Any], device: torch.device
    ) -> nn.Module:
        """Construct (and optionally warm-start) the model, already moved to device."""
        ...

    @abstractmethod
    def objective_name(self) -> str: ...

    @abstractmethod
    def objective_kwargs(self, config: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def checkpoint_name(self) -> str: ...

    # ------------------------------------------------------------------ optional hooks

    def post_merge_adjustments(
        self, config: dict[str, Any], args: argparse.Namespace
    ) -> dict[str, Any]:
        """Post-merge config fixups (e.g. stage-2 anchor lambda defaults)."""
        return config

    def should_enforce_precheck(self, config: dict[str, Any]) -> bool:
        """Return False to skip the precheck gate (stage-1 skips when target_scale_mode=none)."""
        return True

    def dataloader_extra_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extra keyword args forwarded to make_dataloaders (e.g. precompute_dataset)."""
        return {}

    def on_dataloaders_ready(
        self,
        datasets: dict,
        dataloaders: dict,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        """Hook called after dataloaders are built (e.g. GPU preload in stage-2)."""

    def on_finish(
        self,
        run_dir: Path,
        loss_curve_path: Path,
        config: dict[str, Any],
        model: nn.Module | None = None,
    ) -> None:
        """Hook called after all outputs are saved (e.g. extra print lines).

        model is the trained model (post-early-stopping). Subclasses can use
        it to dump auxiliary artifacts like learned per-nozzle lambda_d2
        (Tier 3B) without re-loading from disk.
        """

    def row_count_label(self) -> str:
        return "Stage"

    # ------------------------------------------------------------------ template

    def run(self) -> None:
        args = self.parse_args()

        overrides: dict[str, Any] = {
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
            "runs_root": str(Path(args.runs_root).expanduser().resolve()),
            "max_curves": args.max_curves,
        }
        _set_if_not_none(overrides, "epochs", args.epochs, int)
        _set_if_not_none(overrides, "batch_size", args.batch_size, int)
        _set_if_not_none(overrides, "n_points", args.n_points, int)
        _set_if_not_none(overrides, "device", args.device, str)
        _set_if_not_none(overrides, "learning_rate", args.learning_rate, float)
        _set_if_not_none(overrides, "weight_decay", args.weight_decay, float)
        _set_if_not_none(overrides, "seed", args.seed, int)
        hidden_dims_arg = getattr(args, "hidden_dims", None)
        if hidden_dims_arg is not None:
            overrides["hidden_dims"] = [
                int(x) for x in str(hidden_dims_arg).split(",") if x.strip()
            ]
        if getattr(args, "allow_failed_precheck", False):
            overrides["allow_failed_precheck"] = True
        if getattr(args, "no_shuffle", False):
            overrides["shuffle_train"] = False

        overrides.update(self.extra_overrides(args))
        config = merge_config(self.default_config(), overrides)
        config = self.post_merge_adjustments(config, args)

        seed = int(config["seed"])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = torch.device(config["device"])

        registry = build_dataset_registry(
            Path(args.test_matrix_root).expanduser().resolve()
            if args.test_matrix_root
            else None
        )
        run_dir = create_run_dir(
            config["runs_root"], self.run_label(config), config["variant"]
        )

        stage_tables = build_all_stage_tables(
            config["data_dir"],
            registry,
            comparison_time_s=float(config["comparison_time_s"]),
            max_curves=config.get("max_curves"),
            output_dir=run_dir,
            a_scale_delta_pressure_exp=self.get_a_scale_dp_exp(config, args),
        )

        if self.should_enforce_precheck(config) and not stage_tables.representative_precheck[
            "passed"
        ] and not bool(config.get("allow_failed_precheck", False)):
            raise RuntimeError(
                "Pretrain collapse check failed. Re-run with --allow-failed-precheck to override."
            )

        split_df = self.row_selection_df(stage_tables)
        lono_holdout = getattr(args, "lono_holdout", None)
        if lono_holdout is not None:
            split_df = assign_splits_leave_one_out(
                split_df,
                holdout_value=lono_holdout,
                holdout_column="experiment_name",
                val_ratio=float(config["val_ratio"]),
                seed=int(config["seed"]),
            )
        else:
            split_df = assign_splits_by_group(
                split_df,
                val_ratio=float(config["val_ratio"]),
                test_ratio=float(config["test_ratio"]),
                seed=int(config["seed"]),
            )

        row_table, scaler_state, feature_columns = self.build_row_table(split_df, config)
        config["feature_columns"] = feature_columns
        config["time_feature"] = feature_columns[0]
        config["input_dim"] = len(feature_columns)
        config["output_dim"] = 2

        datasets, dataloaders = make_dataloaders(
            row_table,
            feature_columns=feature_columns,
            batch_size=int(config["batch_size"]),
            n_points=int(config["n_points"]),
            time_min_ms=float(config["time_min_ms"]),
            time_max_ms=float(config["time_max_ms"]),
            shuffle_train=bool(config["shuffle_train"]),
            num_workers=int(config["num_workers"]),
            **self.dataloader_extra_kwargs(config),
        )
        self.on_dataloaders_ready(datasets, dataloaders, device, config)

        first_batch = next(iter(dataloaders["train"]))
        print(f"{self.row_count_label()} row count:", len(row_table))
        print("Split sizes:", {split: len(ds) for split, ds in datasets.items()})
        print("Feature batch shape:", tuple(first_batch["features"].shape))
        print("Target batch shape:", tuple(first_batch["target_scaled"].shape))
        print("Precheck passed:", stage_tables.representative_precheck["passed"])

        model = self.build_model(config, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

        model, iter_history, epoch_history = train_with_early_stopping(
            model=model,
            dataloaders=dataloaders,
            device=device,
            objective_name=self.objective_name(),
            objective_kwargs=self.objective_kwargs(config),
            epochs=int(config["epochs"]),
            optimizer=optimizer,
            patience=int(config["early_stopping_patience"]),
            min_delta=float(config["early_stopping_min_delta"]),
            log_every=int(config["log_interval"]),
        )

        save_training_outputs(
            run_dir,
            model=model,
            checkpoint_name=self.checkpoint_name(),
            train_config=config,
            scaler_state=scaler_state,
            row_table=row_table,
            iter_history=iter_history,
            epoch_history=epoch_history,
            precheck_report=stage_tables.representative_precheck,
        )
        loss_curve_path = plot_loss_curves(
            epoch_history, run_dir, objective_name=self.objective_name()
        )

        summary = (
            epoch_history.loc[epoch_history["split"] == "test"]
            .tail(1)
            .to_dict(orient="records")
        )
        pd.DataFrame(summary).to_csv(run_dir / "test_summary.csv", index=False)
        print("Saved run_dir:", run_dir)
        print("Saved loss curves:", loss_curve_path)
        self.on_finish(run_dir, loss_curve_path, config, model=model)
