from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from engineered_feature_common import (
    DEFAULT_STAGE2_CONFIG,
    apply_saved_scaler_state,
    assign_splits_by_group,
    assign_splits_leave_one_out,
    build_all_stage_tables,
    build_dataset_registry,
    build_model,
    create_run_dir,
    load_run_artifacts,
    make_dataloaders,
    merge_config,
    plot_loss_curves,
    save_training_outputs,
    scaler_a_scale_dp_exp,
    train_with_early_stopping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 NLL on A-scaled penetration.")
    parser.add_argument(
        "stage1_run_dir",
        nargs="?",
        type=str,
        help="Stage-1 engineered run directory for warm start and scaler reuse.",
    )
    parser.add_argument(
        "--stage1-run-dir",
        "--stage1_run_dir",
        dest="stage1_run_dir_flag",
        type=str,
        default=None,
        help="Stage-1 engineered run directory for warm start and scaler reuse.",
    )
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
    parser.add_argument(
        "--stage2-ablation",
        choices=("no_anchor", "mu_anchor", "mu_sigma_anchor"),
        default=None,
        help="Stage-2 loss ablation. Defaults to no_anchor unless overridden by config.",
    )
    parser.add_argument("--lambda-mu-anchor", type=float, default=None)
    parser.add_argument("--lambda-sigma-anchor", type=float, default=None)
    parser.add_argument("--anchor-window-ms", type=float, default=None)
    parser.add_argument("--sigma-anchor-floor-mm", type=float, default=None)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (controls split assignment and torch RNG).")
    parser.add_argument("--allow-failed-precheck", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--lono-holdout", type=str, default=None,
                        help="If set, hold out experiment_name=<value> as test; "
                             "use leave-one-nozzle-out split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_run_dir = args.stage1_run_dir_flag or args.stage1_run_dir
    if not stage1_run_dir:
        raise SystemExit("stage1_run_dir is required. Pass it positionally or with --stage1-run-dir.")
    stage1_artifacts = load_run_artifacts(Path(stage1_run_dir).expanduser().resolve(), device=args.device or None)
    stage1_config = dict(stage1_artifacts.train_config)
    if "stage1_engineered_mse_" not in stage1_artifacts.run_dir.name:
        raise ValueError(
            f"Expected a Stage-1 engineered run directory, got: {stage1_artifacts.run_dir}"
        )
    variant = str(stage1_config["variant"])

    overrides = {
        "variant": variant,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "runs_root": str(Path(args.runs_root).expanduser().resolve()),
        "hidden_dims": list(stage1_config["hidden_dims"]),
        "dropout": float(stage1_config["dropout"]),
        "activation": str(stage1_config["activation"]),
        "feature_columns": list(stage1_config["feature_columns"]),
        "time_feature": str(stage1_config.get("time_feature", "time_norm_0_5ms")),
        "input_dim": int(stage1_config["input_dim"]),
        "output_dim": int(stage1_config["output_dim"]),
        "std_clamp_min": float(stage1_config.get("std_clamp_min", DEFAULT_STAGE2_CONFIG["std_clamp_min"])),
        "max_curves": args.max_curves,
        "stage1_run_dir": str(stage1_artifacts.run_dir),
    }
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.n_points is not None:
        overrides["n_points"] = int(args.n_points)
    if args.device is not None:
        overrides["device"] = str(args.device)
    else:
        overrides["device"] = str(next(stage1_artifacts.model.parameters()).device)
    if args.num_workers is not None:
        overrides["num_workers"] = int(args.num_workers)
    if args.prefetch_factor is not None:
        overrides["prefetch_factor"] = int(args.prefetch_factor)
    if args.no_precompute:
        overrides["precompute_dataset"] = False
    if args.persistent_workers:
        overrides["persistent_workers"] = True
    if args.learning_rate is not None:
        overrides["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        overrides["weight_decay"] = float(args.weight_decay)
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
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.allow_failed_precheck:
        overrides["allow_failed_precheck"] = True
    if args.no_shuffle:
        overrides["shuffle_train"] = False

    config = merge_config(DEFAULT_STAGE2_CONFIG, overrides)
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
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config["device"])
    registry = build_dataset_registry(Path(args.test_matrix_root).expanduser().resolve() if args.test_matrix_root else None)
    run_dir = create_run_dir(config["runs_root"], f"stage2_engineered_nll_{config['stage2_ablation']}", config["variant"])

    stage_tables = build_all_stage_tables(
        config["data_dir"],
        registry,
        comparison_time_s=float(config["comparison_time_s"]),
        max_curves=config.get("max_curves"),
        output_dir=run_dir,
        a_scale_delta_pressure_exp=scaler_a_scale_dp_exp(stage1_artifacts.scaler_state),
    )
    if not stage_tables.representative_precheck["passed"] and not bool(config.get("allow_failed_precheck", False)):
        raise RuntimeError(
            "Representative pretrain collapse check failed. Re-run with --allow-failed-precheck to override."
        )

    if args.lono_holdout is not None:
        filtered_df = assign_splits_leave_one_out(
            stage_tables.filtered,
            holdout_value=args.lono_holdout,
            holdout_column="experiment_name",
            val_ratio=float(config["val_ratio"]),
            seed=int(config["seed"]),
        )
    else:
        filtered_df = assign_splits_by_group(
            stage_tables.filtered,
            val_ratio=float(config["val_ratio"]),
            test_ratio=float(config["test_ratio"]),
            seed=int(config["seed"]),
        )
    scaler_state = dict(stage1_artifacts.scaler_state)
    row_table = apply_saved_scaler_state(filtered_df, scaler_state).reset_index(drop=True)

    datasets, dataloaders = make_dataloaders(
        row_table,
        feature_columns=list(config["feature_columns"]),
        batch_size=int(config["batch_size"]),
        n_points=int(config["n_points"]),
        time_min_ms=float(config["time_min_ms"]),
        time_max_ms=float(config["time_max_ms"]),
        shuffle_train=bool(config["shuffle_train"]),
        num_workers=int(config["num_workers"]),
        precompute_dataset=bool(config.get("precompute_dataset", False)),
        persistent_workers=bool(config.get("persistent_workers", False)),
        prefetch_factor=int(config["prefetch_factor"]) if config.get("prefetch_factor") is not None else None,
    )

    if bool(config.get("precompute_dataset", False)) and device.type != "cpu":
        print(f"Preloading datasets to {device} to maximize GPU utilization...")
        for dataset in datasets.values():
            if getattr(dataset, "_cached_features", None) is not None:
                dataset._cached_features = dataset._cached_features.to(device)
                dataset._cached_target_scaled = dataset._cached_target_scaled.to(device)
                dataset._cached_target_physical = dataset._cached_target_physical.to(device)
                dataset._cached_a_scale = dataset._cached_a_scale.to(device)

    first_batch = next(iter(dataloaders["train"]))
    print("Stage-2 filtered row count:", len(row_table))
    print("Split sizes:", {split: len(dataset) for split, dataset in datasets.items()})
    print("Feature batch shape:", tuple(first_batch["features"].shape))
    print("Target batch shape:", tuple(first_batch["target_scaled"].shape))
    print("Precheck passed:", stage_tables.representative_precheck["passed"])
    print(
        "Loader config:",
        {
            "num_workers": int(config["num_workers"]),
            "precompute_dataset": bool(config.get("precompute_dataset", False)),
            "persistent_workers": bool(config.get("persistent_workers", False)),
            "prefetch_factor": config.get("prefetch_factor"),
        },
    )

    model = build_model(config).to(device)
    model.load_state_dict(stage1_artifacts.model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    objective_kwargs = {
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
    }
    model, iter_history, epoch_history = train_with_early_stopping(
        model=model,
        dataloaders=dataloaders,
        device=device,
        objective_name="stage2",
        objective_kwargs=objective_kwargs,
        epochs=int(config["epochs"]),
        optimizer=optimizer,
        patience=int(config["early_stopping_patience"]),
        min_delta=float(config["early_stopping_min_delta"]),
        log_every=int(config["log_interval"]),
    )

    save_training_outputs(
        run_dir,
        model=model,
        checkpoint_name="best_model_stage2.pt",
        train_config=config,
        scaler_state=scaler_state,
        row_table=row_table,
        iter_history=iter_history,
        epoch_history=epoch_history,
        precheck_report=stage_tables.representative_precheck,
    )
    loss_curve_path = plot_loss_curves(epoch_history, run_dir, objective_name="stage2")

    summary = epoch_history.loc[epoch_history["split"] == "test"].tail(1).to_dict(orient="records")
    pd.DataFrame(summary).to_csv(run_dir / "test_summary.csv", index=False)
    print("Warm-started from:", stage1_artifacts.run_dir)
    print("Saved run_dir:", run_dir)
    print("Saved loss curves:", loss_curve_path)


if __name__ == "__main__":
    main()
