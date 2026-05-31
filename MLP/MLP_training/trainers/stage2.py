from __future__ import annotations

"""Stage-2 NLL trainer: warm-start fine-tuning for calibrated uncertainty.

Loads a Stage-1 checkpoint, inherits its architecture, feature columns, and
scaler state without refitting, then fine-tunes with Gaussian NLL loss in
A-scaled space.  Optional anchor penalties (stage2_ablation) prevent the
predicted mean and std from inflating in the data-sparse injection onset region.

Training data: filtered rows — all rows passing quality cuts — to expose the
full variance of injection conditions to the NLL optimiser (vs. the balanced
representative subset used in Stage-1).

Entry point: python train_stage2_nll.py  (or python -m trainers.stage2)
Required: --stage1-run-dir pointing to a ``stage1_engineered_mse_*`` directory.
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engineered_feature_common import (
    DEFAULT_STAGE2_CONFIG,
    StageTables,
    apply_saved_scaler_state,
    build_model,
    load_run_artifacts,
    scaler_a_scale_dp_exp,
)
from .base import TrainerBase


class Stage2Trainer(TrainerBase):
    """Stage-2 NLL trainer.  Inherits architecture and scaler state from Stage-1.

    The Stage-1 scaler is reused without refit so the z-scored feature space is
    identical between stages.  self._stage1_artifacts is set in extra_overrides()
    and consumed by get_a_scale_dp_exp(), build_row_table(), build_model(), and
    on_finish().
    """

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train Stage-2 NLL on A-scaled penetration.")
        parser.add_argument("stage1_run_dir", nargs="?", type=str,
            help="Stage-1 engineered run directory for warm start and scaler reuse.")
        parser.add_argument("--stage1-run-dir", "--stage1_run_dir", dest="stage1_run_dir_flag",
            type=str, default=None,
            help="Stage-1 engineered run directory for warm start and scaler reuse.")
        parser.add_argument("--data-dir", type=str, default=DEFAULT_STAGE2_CONFIG["data_dir"])
        parser.add_argument("--runs-root", type=str, default=DEFAULT_STAGE2_CONFIG["runs_root"])
        parser.add_argument("--test-matrix-root", type=str, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--n-points", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument("--prefetch-factor", type=int, default=None)
        parser.add_argument("--no-precompute", action="store_true")
        parser.add_argument("--persistent-workers", action="store_true")
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--weight-decay", type=float, default=None)
        parser.add_argument("--hidden-dims", type=str, default=None,
            help="Comma-separated hidden layer widths. Must match the Stage-1 run "
                 "being warm-started from; mismatched values raise SystemExit.")
        parser.add_argument("--stage2-ablation", choices=("no_anchor", "mu_anchor", "mu_sigma_anchor"),
            default=None, help="Stage-2 loss ablation. Defaults to mu_anchor unless overridden by config.")
        parser.add_argument("--lambda-mu-anchor", type=float, default=None)
        parser.add_argument("--lambda-sigma-anchor", type=float, default=None)
        parser.add_argument("--anchor-window-ms", type=float, default=None)
        parser.add_argument("--sigma-anchor-floor-mm", type=float, default=None)
        parser.add_argument("--max-curves", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None, help="Override config seed...")
        parser.add_argument("--allow-failed-precheck", action="store_true")
        parser.add_argument("--no-shuffle", action="store_true")
        parser.add_argument("--lono-holdout", type=str, default=None,
            help="If set, hold out experiment_name=<value> as test; use leave-one-nozzle-out split.")
        # Tier-2C onset auxiliary head. Stage 2 inherits the architectural flag
        # from Stage 1 automatically; --lambda-aux here just sets the Stage-2
        # aux loss weight. --onset-aux-head still has to be passed because
        # the LONO pipeline doesn't keep state across stages.
        parser.add_argument("--onset-aux-head", action="store_true",
            help="Use the aux-head architecture. Must match the Stage-1 run being warm-started from.")
        parser.add_argument("--lambda-aux", type=float, default=None,
            help="Weight on the onset aux MSE term added to the Stage-2 loss.")
        parser.add_argument("--onset-t-ms-max", type=float, default=None,
            help="Onset window cutoff in ms (default: 0.3).")
        # Tier-3A: Stage 2 inherits architecture_mode from the Stage-1 run;
        # accepting the flag here just lets the LONO pipeline pass it through
        # without parser errors (mismatch raises a clear error).
        parser.add_argument("--architecture-mode", choices=("single", "family_head", "residual_family_head",
                                                            "residual_film_last_block", "residual_film_all_blocks"),
            default=None,
            help="Inherited from Stage 1; pass-through CLI only. Mismatch raises SystemExit.")
        parser.add_argument("--n-families", type=int, default=None,
            help="Inherited from Stage 1; pass-through CLI only.")
        return parser.parse_args()

    def default_config(self) -> dict[str, Any]:
        return DEFAULT_STAGE2_CONFIG

    def extra_overrides(self, args) -> dict[str, Any]:
        stage1_run_dir = args.stage1_run_dir_flag or args.stage1_run_dir
        if not stage1_run_dir:
            raise SystemExit("stage1_run_dir is required. Pass it positionally or with --stage1-run-dir.")
        self._stage1_artifacts = load_run_artifacts(
            Path(stage1_run_dir).expanduser().resolve(), device=args.device or None
        )
        stage1_config = dict(self._stage1_artifacts.train_config)
        if "stage1_engineered_mse_" not in self._stage1_artifacts.run_dir.name:
            raise ValueError(
                f"Expected a Stage-1 engineered run directory, got: {self._stage1_artifacts.run_dir}"
            )
        variant = str(stage1_config["variant"])

        inherited_hidden = list(stage1_config["hidden_dims"])
        cli_hidden_arg = getattr(args, "hidden_dims", None)
        if cli_hidden_arg is not None:
            cli_hidden = [int(x) for x in str(cli_hidden_arg).split(",") if x.strip()]
            if cli_hidden != inherited_hidden:
                raise SystemExit(
                    f"--hidden-dims {cli_hidden} disagrees with Stage-1 run "
                    f"({inherited_hidden}); Stage 2 warm-start requires matching shapes."
                )

        # Tier-2C onset aux head must match the Stage-1 architecture exactly,
        # because Stage 2 warm-starts from Stage-1's state_dict.
        inherited_aux_head = bool(stage1_config.get("onset_aux_head", False))
        cli_aux_flag = bool(getattr(args, "onset_aux_head", False))
        if cli_aux_flag and not inherited_aux_head:
            raise SystemExit(
                "--onset-aux-head set on Stage 2 but the Stage-1 run was trained "
                "without it; warm-start would fail. Re-run Stage 1 with --onset-aux-head."
            )
        if inherited_aux_head and not cli_aux_flag:
            # Stage 1 had the aux head; Stage 2 must keep it for warm-start.
            print("[stage2] Inheriting onset_aux_head=True from Stage 1.")

        overrides = {
            "variant": variant,
            "hidden_dims": inherited_hidden,
            "dropout": float(stage1_config["dropout"]),
            "activation": str(stage1_config["activation"]),
            "feature_columns": list(stage1_config["feature_columns"]),
            "time_feature": str(stage1_config.get("time_feature", "time_norm_0_5ms")),
            "input_dim": int(stage1_config["input_dim"]),
            "output_dim": int(stage1_config["output_dim"]),
            "std_clamp_min": float(stage1_config.get("std_clamp_min", DEFAULT_STAGE2_CONFIG["std_clamp_min"])),
            "stage1_run_dir": str(self._stage1_artifacts.run_dir),
        }

        # device: prefer CLI arg, fallback to device inferred from loaded model
        if args.device is None:
            overrides["device"] = str(next(self._stage1_artifacts.model.parameters()).device)

        # optional CLI → config
        if args.num_workers is not None:
            overrides["num_workers"] = int(args.num_workers)
        if args.prefetch_factor is not None:
            overrides["prefetch_factor"] = int(args.prefetch_factor)
        if args.no_precompute:
            overrides["precompute_dataset"] = False
        if args.persistent_workers:
            overrides["persistent_workers"] = True
        if args.stage2_ablation is not None:
            overrides["stage2_ablation"] = str(args.stage2_ablation)
        if args.lambda_mu_anchor is not None:
            overrides["lambda_mu_anchor"] = float(args.lambda_mu_anchor)
        if args.lambda_sigma_anchor is not None:
            overrides["lambda_sigma_anchor"] = float(args.lambda_sigma_anchor)
        if args.anchor_window_ms is not None:
            overrides["anchor_window_ms"] = float(args.anchor_window_ms)
        if args.sigma_anchor_floor_mm is not None:
            overrides["sigma_anchor_floor_mm"] = float(args.sigma_anchor_floor_mm)

        # Tier-2C aux-head wiring
        if inherited_aux_head:
            overrides["onset_aux_head"] = True
            overrides["onset_aux_hidden"] = int(stage1_config.get("onset_aux_hidden", 64))
            overrides["onset_aux_weight"] = (
                float(args.lambda_aux) if args.lambda_aux is not None else 0.1
            )
        elif args.lambda_aux is not None:
            overrides["onset_aux_weight"] = float(args.lambda_aux)
        if args.onset_t_ms_max is not None:
            overrides["onset_t_ms_max"] = float(args.onset_t_ms_max)

        # Tier-3A family-aware architecture: inherit from Stage 1.
        inherited_arch = str(stage1_config.get("architecture_mode", "single"))
        cli_arch = getattr(args, "architecture_mode", None)
        if cli_arch is not None and str(cli_arch) != inherited_arch:
            raise SystemExit(
                f"--architecture-mode {cli_arch} disagrees with Stage-1 run "
                f"(architecture_mode={inherited_arch}); warm-start would fail."
            )
        overrides["architecture_mode"] = inherited_arch
        overrides["n_families"] = int(stage1_config.get("n_families", 2))
        if "family_head_dims" in stage1_config:
            overrides["family_head_dims"] = list(stage1_config["family_head_dims"])
        overrides["fallback_family_id"] = int(stage1_config.get("fallback_family_id", 1))

        # Tier-3B: Stage 2 doesn't learn lambda_d2 itself but needs the dim
        # for the d2 penalty path to be consistent. Inherit if Stage 1 had it.
        if bool(stage1_config.get("learnable_d2", False)):
            overrides["learnable_d2"] = True
            overrides["n_families_for_d2"] = int(stage1_config.get("n_families_for_d2", 6))
            overrides["learnable_d2_floor"] = float(stage1_config.get("learnable_d2_floor", 1e-5))

        return overrides

    def post_merge_adjustments(self, config, args) -> dict:
        if config["stage2_ablation"] == "mu_anchor" and args.lambda_mu_anchor is None:
            config["lambda_mu_anchor"] = 1e-2
            config["lambda_sigma_anchor"] = 0.0 if args.lambda_sigma_anchor is None else float(config["lambda_sigma_anchor"])
        elif config["stage2_ablation"] == "mu_sigma_anchor":
            if args.lambda_mu_anchor is None:
                config["lambda_mu_anchor"] = 1e-2
            if args.lambda_sigma_anchor is None:
                config["lambda_sigma_anchor"] = 1e-3
        elif config["stage2_ablation"] == "no_anchor":
            if args.lambda_mu_anchor is None:
                config["lambda_mu_anchor"] = 0.0
            if args.lambda_sigma_anchor is None:
                config["lambda_sigma_anchor"] = 0.0
        return config

    def run_label(self, config) -> str:
        return f"stage2_engineered_nll_{config['stage2_ablation']}"

    def get_a_scale_dp_exp(self, config, args) -> float:
        return scaler_a_scale_dp_exp(self._stage1_artifacts.scaler_state)

    def row_selection_df(self, stage_tables) -> pd.DataFrame:
        return stage_tables.filtered

    def build_row_table(self, split_df, config) -> tuple[pd.DataFrame, dict, list[str]]:
        scaler_state = dict(self._stage1_artifacts.scaler_state)
        row_table = apply_saved_scaler_state(split_df, scaler_state).reset_index(drop=True)
        feature_columns = list(config["feature_columns"])
        return row_table, scaler_state, feature_columns

    def build_model(self, config, device) -> nn.Module:
        model = build_model(config).to(device)
        model.load_state_dict(self._stage1_artifacts.model.state_dict())
        return model

    def objective_name(self) -> str:
        return "stage2"

    def objective_kwargs(self, config) -> dict[str, Any]:
        return {
            "n_points": int(config["n_points"]),
            "time_max_ms": float(config["time_max_ms"]),
            "log_var_bounds": tuple(config["log_var_bounds"]),
            "nll_eps": float(config["nll_eps"]),
            "d1_positive_weight": float(config["d1_positive_weight"]),
            "d2_concave_weight": float(config["d2_concave_weight"]),
            "d2_start_ms": float(config["d2_start_ms"]),
            "d2_transition_ms": float(config["d2_transition_ms"]),
            "lambda_mu_anchor": float(config["lambda_mu_anchor"]),
            "lambda_sigma_anchor": float(config["lambda_sigma_anchor"]),
            "anchor_window_ms": float(config["anchor_window_ms"]),
            "sigma_anchor_floor_mm": float(config["sigma_anchor_floor_mm"]),
            "grad_clip_norm": float(config["grad_clip_norm"]) if config.get("grad_clip_norm") is not None else None,
            "onset_aux_weight": float(config.get("onset_aux_weight", 0.0)),
            "onset_t_ms_max": float(config.get("onset_t_ms_max", 0.3)),
        }

    def checkpoint_name(self) -> str:
        return "best_model_stage2.pt"

    def row_count_label(self) -> str:
        return "Stage-2 filtered"

    def dataloader_extra_kwargs(self, config) -> dict:
        return {
            "precompute_dataset": bool(config.get("precompute_dataset", False)),
            "persistent_workers": bool(config.get("persistent_workers", False)),
            "prefetch_factor": int(config["prefetch_factor"]) if config.get("prefetch_factor") is not None else None,
        }

    def on_dataloaders_ready(self, datasets, dataloaders, device, config) -> None:
        # GPU preloading for precomputed datasets
        if bool(config.get("precompute_dataset", False)) and device.type != "cpu":
            print(f"Preloading datasets to {device} to maximize GPU utilization...")
            for dataset in datasets.values():
                if getattr(dataset, "_cached_features", None) is not None:
                    dataset._cached_features = dataset._cached_features.to(device)
                    dataset._cached_target_scaled = dataset._cached_target_scaled.to(device)
                    dataset._cached_target_physical = dataset._cached_target_physical.to(device)
                    dataset._cached_a_scale = dataset._cached_a_scale.to(device)
        # print loader config diagnostic
        print("Loader config:", {
            "num_workers": int(config["num_workers"]),
            "precompute_dataset": bool(config.get("precompute_dataset", False)),
            "persistent_workers": bool(config.get("persistent_workers", False)),
            "prefetch_factor": config.get("prefetch_factor"),
        })

    def on_finish(self, run_dir, loss_curve_path, config, model=None) -> None:
        print("Warm-started from:", self._stage1_artifacts.run_dir)


if __name__ == "__main__":
    Stage2Trainer().run()
