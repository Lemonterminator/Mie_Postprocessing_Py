from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from engineered_feature_common import (
    DEFAULT_STAGE1_CONFIG,
    build_all_stage_tables,
    build_dataset_registry,
    build_model,
    build_variant_feature_table,
    create_run_dir,
    make_dataloaders,
    merge_config,
    plot_loss_curves,
    save_training_outputs,
    assign_splits_by_group,
    train_with_early_stopping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 MSE on A-scaled penetration.")
    parser.add_argument("--variant", choices=("a_only", "a_plus_log_a"), default="a_only")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_STAGE1_CONFIG["data_dir"])
    parser.add_argument("--runs-root", type=str, default=DEFAULT_STAGE1_CONFIG["runs_root"])
    parser.add_argument("--test-matrix-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-curves", type=int, default=None)
    parser.add_argument("--allow-failed-precheck", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        "variant": args.variant,
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "runs_root": str(Path(args.runs_root).expanduser().resolve()),
        "max_curves": args.max_curves,
    }
    if args.epochs is not None:
        overrides["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.n_points is not None:
        overrides["n_points"] = int(args.n_points)
    if args.device is not None:
        overrides["device"] = str(args.device)
    if args.learning_rate is not None:
        overrides["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        overrides["weight_decay"] = float(args.weight_decay)
    if args.allow_failed_precheck:
        overrides["allow_failed_precheck"] = True
    if args.no_shuffle:
        overrides["shuffle_train"] = False

    config = merge_config(DEFAULT_STAGE1_CONFIG, overrides)
    device = torch.device(config["device"])
    registry = build_dataset_registry(Path(args.test_matrix_root).expanduser().resolve() if args.test_matrix_root else None)
    run_dir = create_run_dir(config["runs_root"], "stage1_engineered_mse", config["variant"])

    stage_tables = build_all_stage_tables(
        config["data_dir"],
        registry,
        comparison_time_s=float(config["comparison_time_s"]),
        max_curves=config.get("max_curves"),
        output_dir=run_dir,
    )
    if not stage_tables.representative_precheck["passed"] and not bool(config.get("allow_failed_precheck", False)):
        raise RuntimeError(
            "Pretrain collapse check failed. Re-run with --allow-failed-precheck to override."
        )

    representative_df = assign_splits_by_group(
        stage_tables.representative,
        val_ratio=float(config["val_ratio"]),
        test_ratio=float(config["test_ratio"]),
        seed=int(config["seed"]),
    )
    row_table, scaler_state, feature_columns = build_variant_feature_table(
        representative_df,
        variant=config["variant"],
        time_min_ms=float(config["time_min_ms"]),
        time_max_ms=float(config["time_max_ms"]),
    )
    config["feature_columns"] = feature_columns
    config["time_feature"] = feature_columns[0]
    config["input_dim"] = len(feature_columns)
    config["output_dim"] = 2
    config["row_selection_mode"] = "representative"

    datasets, dataloaders = make_dataloaders(
        row_table,
        feature_columns=feature_columns,
        batch_size=int(config["batch_size"]),
        n_points=int(config["n_points"]),
        time_min_ms=float(config["time_min_ms"]),
        time_max_ms=float(config["time_max_ms"]),
        shuffle_train=bool(config["shuffle_train"]),
        num_workers=int(config["num_workers"]),
    )
    first_batch = next(iter(dataloaders["train"]))
    print("Stage-1 representative row count:", len(row_table))
    print("Split sizes:", {split: len(dataset) for split, dataset in datasets.items()})
    print("Feature batch shape:", tuple(first_batch["features"].shape))
    print("Target batch shape:", tuple(first_batch["target_scaled"].shape))
    print("Precheck passed:", stage_tables.representative_precheck["passed"])

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    objective_kwargs = {
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
    }
    model, iter_history, epoch_history = train_with_early_stopping(
        model=model,
        dataloaders=dataloaders,
        device=device,
        objective_name="stage1",
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
        checkpoint_name="best_model_stage1.pt",
        train_config=config,
        scaler_state=scaler_state,
        row_table=row_table,
        iter_history=iter_history,
        epoch_history=epoch_history,
        precheck_report=stage_tables.representative_precheck,
    )
    loss_curve_path = plot_loss_curves(epoch_history, run_dir, objective_name="stage1")

    summary = epoch_history.loc[epoch_history["split"] == "test"].tail(1).to_dict(orient="records")
    pd.DataFrame(summary).to_csv(run_dir / "test_summary.csv", index=False)
    print("Saved run_dir:", run_dir)
    print("Saved loss curves:", loss_curve_path)


if __name__ == "__main__":
    main()
