from __future__ import annotations

"""Stage-1 MSE trainer: heteroscedastic MLP trained on A-scaled penetration.

Trains PenetrationMLP from scratch to predict (mu_hat, log_var_hat) in the
dimensionless A-scaled space using MSE loss + log-variance regulariser + physics
shape penalties.  The resulting checkpoint is used as the warm-start for Stage-2
NLL fine-tuning.

Training data: representative rows — one row per unique injection condition —
sampled from the cleaned CSV files.  This gives a balanced view of the operating
condition space regardless of how many repeated measurements exist per condition.

Entry point: python train_stage1_mse.py  (or python -m trainers.stage1)
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engineered_feature_common import (
    DEFAULT_STAGE1_CONFIG,
    FEATURE_COLUMNS_BY_VARIANT,
    StageTables,
    build_model,
    build_variant_feature_table,
    variant_a_scale_dp_exp,
    variant_target_scale_mode,
)
from .base import TrainerBase


class Stage1Trainer(TrainerBase):
    """Stage-1 trainer: MSE warm-start for Stage-2 NLL fine-tuning.

    Uses the representative sub-table (one row per unique condition) so the
    model sees a balanced operating-condition sample rather than the raw
    imbalanced CSV distribution.

    The precheck gate (should_enforce_precheck) is active only when
    target_scale_mode != "none" — i.e., when A-scaling is applied — because
    the pre-train collapse check is meaningful only in the scaled target space.
    """

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train Stage-1 MSE on A-scaled penetration.")
        parser.add_argument("--variant", choices=tuple(FEATURE_COLUMNS_BY_VARIANT.keys()), default="a_only")
        parser.add_argument("--data-dir", type=str, default=DEFAULT_STAGE1_CONFIG["data_dir"])
        parser.add_argument("--runs-root", type=str, default=DEFAULT_STAGE1_CONFIG["runs_root"])
        parser.add_argument("--test-matrix-root", type=str, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument("--n-points", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--learning-rate", type=float, default=None)
        parser.add_argument("--weight-decay", type=float, default=None)
        parser.add_argument("--d2-concave-weight", type=float, default=None,
            help="Override d2_concave_weight. Set to 0 for the Tier-3B no_d2_penalty variant; "
                 "ignored when --learnable-d2 is on (the learnable lambdas replace the scalar weight).")
        parser.add_argument("--hidden-dims", type=str, default=None,
            help="Comma-separated hidden layer widths, e.g. '256,256,64'. "
                 "Overrides DEFAULT_STAGE1_CONFIG['hidden_dims']. Stage 2 must use the same dims.")
        parser.add_argument("--max-curves", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None, help="Override config seed...")
        parser.add_argument("--allow-failed-precheck", action="store_true")
        parser.add_argument("--no-shuffle", action="store_true")
        parser.add_argument("--lono-holdout", type=str, default=None,
                            help="If set, hold out experiment_name=<value> as test; use leave-one-nozzle-out split.")
        # Tier-2C onset auxiliary head
        parser.add_argument("--onset-aux-head", action="store_true",
                            help="Add an auxiliary regression head trained on the onset window (t < onset_t_ms_max).")
        parser.add_argument("--lambda-aux", type=float, default=None,
                            help="Weight on the onset aux MSE term added to the Stage-1 loss.")
        parser.add_argument("--onset-aux-hidden", type=int, default=None,
                            help="Hidden width of the aux head (default: 64).")
        parser.add_argument("--onset-t-ms-max", type=float, default=None,
                            help="Onset window cutoff in ms (default: 0.3).")
        # Tier-3A family-aware architecture
        parser.add_argument("--architecture-mode", choices=("single", "family_head", "residual_family_head",
                                                            "residual_film_last_block", "residual_film_all_blocks"),
                            default=None,
                            help="single: PenetrationMLP. family_head: shared trunk + per-family mu heads (Tier 3A). "
                                 "residual_family_head: shared mu plus per-family residual deltas. "
                                 "residual_film_*: residual head plus identity FiLM adapters.")
        parser.add_argument("--n-families", type=int, default=None,
                            help="Number of families for the routing head (default: 2; factory-fresh vs modified).")
        # Tier-3B learnable per-nozzle lambda_d2
        parser.add_argument("--learnable-d2", action="store_true",
                            help="Replace the scalar d2_concave_weight with a learnable per-nozzle lambda (Tier 3B).")
        parser.add_argument("--n-families-for-d2", type=int, default=None,
                            help="How many learnable d2 weights to instantiate (default: 6, one per nozzle).")
        return parser.parse_args()

    def default_config(self) -> dict[str, Any]:
        return DEFAULT_STAGE1_CONFIG

    def extra_overrides(self, args) -> dict[str, Any]:
        variant = args.variant
        a_scale_dp_exp = variant_a_scale_dp_exp(variant)
        target_scale_mode = variant_target_scale_mode(variant)
        self._a_scale_dp_exp = a_scale_dp_exp
        self._target_scale_mode = target_scale_mode
        overrides: dict[str, Any] = {
            "variant": variant,
            "row_selection_mode": "representative",
            "target_scale_mode": target_scale_mode,
            "a_scale_delta_pressure_exp": float(a_scale_dp_exp),
        }
        if getattr(args, "onset_aux_head", False):
            overrides["onset_aux_head"] = True
            # Default lambda_aux=0.1 once the head is enabled (plan section 5.3
            # uses 0.1 for aux_with_anchor / aux_no_anchor).
            overrides["onset_aux_weight"] = (
                float(args.lambda_aux) if args.lambda_aux is not None else 0.1
            )
        elif args.lambda_aux is not None:
            overrides["onset_aux_weight"] = float(args.lambda_aux)
        if args.onset_aux_hidden is not None:
            overrides["onset_aux_hidden"] = int(args.onset_aux_hidden)
        if args.onset_t_ms_max is not None:
            overrides["onset_t_ms_max"] = float(args.onset_t_ms_max)
        if getattr(args, "architecture_mode", None) is not None:
            overrides["architecture_mode"] = str(args.architecture_mode)
        if getattr(args, "n_families", None) is not None:
            overrides["n_families"] = int(args.n_families)
        if getattr(args, "d2_concave_weight", None) is not None:
            overrides["d2_concave_weight"] = float(args.d2_concave_weight)
        if getattr(args, "learnable_d2", False):
            overrides["learnable_d2"] = True
        if getattr(args, "n_families_for_d2", None) is not None:
            overrides["n_families_for_d2"] = int(args.n_families_for_d2)
        return overrides

    def run_label(self, config) -> str:
        return "stage1_engineered_mse"

    def get_a_scale_dp_exp(self, config, args) -> float:
        return self._a_scale_dp_exp

    def row_selection_df(self, stage_tables) -> pd.DataFrame:
        return stage_tables.representative

    def build_row_table(self, split_df, config) -> tuple[pd.DataFrame, dict, list[str]]:
        add_family_id = str(config.get("architecture_mode", "single")) in {
            "family_head",
            "residual_family_head",
            "residual_film_last_block",
            "residual_film_all_blocks",
        }
        add_nozzle_id = bool(config.get("learnable_d2", False))
        row_table, scaler_state, feature_columns = build_variant_feature_table(
            split_df,
            variant=config["variant"],
            time_min_ms=float(config["time_min_ms"]),
            time_max_ms=float(config["time_max_ms"]),
            add_family_id=add_family_id,
            add_nozzle_id=add_nozzle_id,
        )
        return row_table, scaler_state, feature_columns

    def build_model(self, config, device) -> nn.Module:
        return build_model(config).to(device)

    def objective_name(self) -> str:
        return "stage1"

    def objective_kwargs(self, config) -> dict[str, Any]:
        feature_columns = list(config.get("feature_columns") or [])
        nozzle_idx = (
            feature_columns.index("nozzle_id") if "nozzle_id" in feature_columns else None
        )
        return {
            "n_points": int(config["n_points"]),
            "time_max_ms": float(config["time_max_ms"]),
            "var_reg_weight": float(config["var_reg_weight"]),
            "log_var_prior": float(config["log_var_prior"]),
            "log_var_bounds": tuple(config["log_var_bounds"]),
            "d1_positive_weight": float(config["d1_positive_weight"]),
            "d2_concave_weight": float(config["d2_concave_weight"]),
            "d2_start_ms": float(config["d2_start_ms"]),
            "d2_transition_ms": float(config["d2_transition_ms"]),
            "grad_clip_norm": float(config["grad_clip_norm"]) if config.get("grad_clip_norm") is not None else None,
            "onset_aux_weight": float(config.get("onset_aux_weight", 0.0)),
            "onset_t_ms_max": float(config.get("onset_t_ms_max", 0.3)),
            "nozzle_id_channel_idx": nozzle_idx,
        }

    def checkpoint_name(self) -> str:
        return "best_model_stage1.pt"

    def row_count_label(self) -> str:
        return "Stage-1 representative"

    def should_enforce_precheck(self, config) -> bool:
        return config.get("target_scale_mode", "none") != "none"

    def on_finish(self, run_dir, loss_curve_path, config, model=None) -> None:
        # Tier-3B: dump the final per-nozzle lambda_d2 values for the verdict
        # script to consume. Skipped silently when learnable_d2 is off.
        if model is None:
            return
        log_lambda = getattr(model, "log_lambda_d2", None)
        if log_lambda is None:
            return
        floor = float(getattr(model, "learnable_d2_floor", 1e-5))
        with torch.no_grad():
            lambdas = (F.softplus(log_lambda) + floor).detach().cpu().tolist()
        rows = [
            {"family_id": int(i), "lambda_d2": float(v), "floor": floor}
            for i, v in enumerate(lambdas)
        ]
        out_path = Path(run_dir) / "learned_lambda_d2_values.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Wrote learned lambda_d2 values: {out_path}")


if __name__ == "__main__":
    Stage1Trainer().run()
