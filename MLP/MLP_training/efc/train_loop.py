from __future__ import annotations

"""Dataset, dataloader, and training loop for the spray-penetration MLP.

PointwisePenetrationDataset  — samples n_points times uniformly in [time_min_ms,
                                time_max_ms] for each spray curve; optionally
                                pre-computes and caches feature / target tensors
                                in RAM (or GPU) for fast repeated access.
collate_pointwise            — default collate: stacks batch tensors along dim 0.
make_dataloaders             — builds train / val / test DataLoaders from a row table.
run_epoch                    — one forward + (optionally) backward pass over a loader;
                                dispatches to stage1_objective or stage2_objective.
train_with_early_stopping    — full training loop with patience-based stopping and
                                best-checkpoint tracking; returns the best model plus
                                per-iteration and per-epoch history DataFrames.
"""

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .models import MIN_TIME_SHIFT_S, reconstruct_penetration_series, sigmoid
from .objectives import (
    stage1_objective,
    stage2_objective,
)
from .feature_engineering import infer_feature_family


class PointwisePenetrationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        n_points: int,
        time_min_ms: float,
        time_max_ms: float,
        precompute: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.feature_columns = list(feature_columns)
        self.n_points = int(n_points)
        self.time_min_ms = float(time_min_ms)
        self.time_max_ms = float(time_max_ms)
        self.precompute = bool(precompute)
        self.time_span_ms = max(self.time_max_ms - self.time_min_ms, 1e-12)
        self.time_grid_ms = np.linspace(self.time_min_ms, self.time_max_ms, self.n_points, dtype=np.float32)
        self.time_grid_s = self.time_grid_ms * 1e-3
        self.time_norm = ((self.time_grid_ms - self.time_min_ms) / self.time_span_ms).astype(np.float32)
        self.target_scale_mode = "none" if infer_feature_family(self.feature_columns) == "legacy_raw" else "a_scale"
        self._cached_features: torch.Tensor | None = None
        self._cached_target_scaled: torch.Tensor | None = None
        self._cached_target_physical: torch.Tensor | None = None
        self._cached_a_scale: torch.Tensor | None = None

        required_cols = list(self.feature_columns[1:]) + ["A_scale", "log_k_sqrt", "log_k_quarter", "log_t0", "log_s"]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise KeyError(f"Dataset missing required columns: {missing}")
        if self.precompute and len(self.df) > 0:
            self._build_cache()

    def __len__(self) -> int:
        return len(self.df)

    def _build_cache(self) -> None:
        n_samples = len(self.df)
        feature_dim = len(self.feature_columns)
        time_axis = np.broadcast_to(self.time_norm.reshape(1, self.n_points, 1), (n_samples, self.n_points, 1))
        if len(self.feature_columns) > 1:
            static_values = self.df[self.feature_columns[1:]].to_numpy(dtype=np.float32)
            static_block = np.broadcast_to(
                static_values[:, None, :],
                (n_samples, self.n_points, static_values.shape[1]),
            )
            features_np = np.concatenate([time_axis, static_block], axis=2)
        else:
            features_np = time_axis

        time_s = self.time_grid_s.astype(np.float64)[None, :]
        log_k_sqrt = self.df["log_k_sqrt"].to_numpy(dtype=np.float64)[:, None]
        log_k_quarter = self.df["log_k_quarter"].to_numpy(dtype=np.float64)[:, None]
        log_t0 = self.df["log_t0"].to_numpy(dtype=np.float64)[:, None]
        log_s = self.df["log_s"].to_numpy(dtype=np.float64)[:, None]

        k_sqrt = np.exp(log_k_sqrt)
        k_quarter = np.exp(log_k_quarter)
        t0 = np.exp(log_t0) + MIN_TIME_SHIFT_S
        sharpness = np.exp(log_s)
        sqrt_segment = k_sqrt * np.sqrt(time_s)
        quarter_segment = k_quarter * np.power(time_s, 0.25)
        blend = sigmoid((time_s - t0) / np.maximum(sharpness, 1e-12))
        penetration = ((1.0 - blend) * sqrt_segment + blend * quarter_segment).astype(np.float32)

        if self.target_scale_mode == "none":
            a_scale = np.ones((n_samples, 1), dtype=np.float32)
        else:
            a_scale = self.df["A_scale"].to_numpy(dtype=np.float32)[:, None]
        a_scale_block = np.broadcast_to(a_scale[:, None, :], (n_samples, self.n_points, 1))
        target_physical = penetration[..., None]
        target_scaled = (penetration / a_scale).astype(np.float32)[..., None]

        self._cached_features = torch.from_numpy(np.ascontiguousarray(features_np.reshape(n_samples, self.n_points, feature_dim)))
        self._cached_target_scaled = torch.from_numpy(np.ascontiguousarray(target_scaled))
        self._cached_target_physical = torch.from_numpy(np.ascontiguousarray(target_physical))
        self._cached_a_scale = torch.from_numpy(np.ascontiguousarray(a_scale_block))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._cached_features is not None:
            return {
                "features": self._cached_features[int(idx)],
                "target_scaled": self._cached_target_scaled[int(idx)],
                "target_physical": self._cached_target_physical[int(idx)],
                "a_scale": self._cached_a_scale[int(idx)],
                "sample_idx": torch.full((self.n_points,), int(idx), dtype=torch.long),
            }

        row = self.df.iloc[int(idx)]
        penetration = reconstruct_penetration_series(
            float(row["log_k_sqrt"]),
            float(row["log_k_quarter"]),
            float(row["log_t0"]),
            float(row["log_s"]),
            self.time_grid_s,
        ).astype(np.float32)
        a_scale = 1.0 if self.target_scale_mode == "none" else float(row["A_scale"])
        target_scaled = (penetration / a_scale).reshape(-1, 1).astype(np.float32)
        target_physical = penetration.reshape(-1, 1).astype(np.float32)
        a_repeat = np.full((self.n_points, 1), a_scale, dtype=np.float32)

        static_vec = row[self.feature_columns[1:]].to_numpy(dtype=np.float32)
        static_mat = np.repeat(static_vec[None, :], self.n_points, axis=0)
        features = np.column_stack([self.time_norm, static_mat]).astype(np.float32)
        return {
            "features": torch.from_numpy(features),
            "target_scaled": torch.from_numpy(target_scaled),
            "target_physical": torch.from_numpy(target_physical),
            "a_scale": torch.from_numpy(a_repeat),
            "sample_idx": torch.full((self.n_points,), int(idx), dtype=torch.long),
        }


def collate_pointwise(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([item[key] for item in batch], dim=0)
        for key in ("features", "target_scaled", "target_physical", "a_scale", "sample_idx")
    }


def make_dataloaders(
    df_in: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    batch_size: int,
    n_points: int,
    time_min_ms: float,
    time_max_ms: float,
    shuffle_train: bool,
    num_workers: int,
    precompute_dataset: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[dict[str, PointwisePenetrationDataset], dict[str, DataLoader]]:
    datasets = {
        split: PointwisePenetrationDataset(
            df_in.loc[df_in["sample_split"] == split].reset_index(drop=True),
            feature_columns=feature_columns,
            n_points=n_points,
            time_min_ms=time_min_ms,
            time_max_ms=time_max_ms,
            precompute=precompute_dataset,
        )
        for split in ("train", "val", "test")
    }
    common_loader_kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available() and not precompute_dataset,
        "collate_fn": collate_pointwise,
    }
    if int(num_workers) > 0:
        common_loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            common_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(batch_size),
            shuffle=bool(shuffle_train),
            **common_loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=int(batch_size),
            shuffle=False,
            **common_loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=int(batch_size),
            shuffle=False,
            **common_loader_kwargs,
        ),
    }
    return datasets, dataloaders


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    objective_name: str,
    objective_kwargs: Mapping[str, Any],
    global_iter_start: int,
    epoch_idx: int,
    log_every: int,
) -> tuple[dict[str, float], list[dict[str, float]], int]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    epoch_totals: dict[str, float] = {}
    iter_logs: list[dict[str, float]] = []
    total_points = 0
    global_iter = int(global_iter_start)

    grad_clip_norm = objective_kwargs.get("grad_clip_norm")
    objective_core = {k: v for k, v in objective_kwargs.items() if k != "grad_clip_norm"}

    for step_idx, batch in enumerate(dataloader, start=1):
        batch_device = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        output = model(batch_device["features"])
        if objective_name == "stage1":
            loss, metrics = stage1_objective(output, batch_device, model=model, **objective_core)
        elif objective_name == "stage2":
            loss, metrics = stage2_objective(output, batch_device, model=model, **objective_core)
        else:
            raise KeyError(f"Unsupported objective_name '{objective_name}'.")

        if is_train:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

        n = int(batch_device["features"].shape[0])
        total_points += n
        for key, value in metrics.items():
            epoch_totals[key] = epoch_totals.get(key, 0.0) + float(value) * n

        log_row = {
            "epoch": float(epoch_idx),
            "step": float(step_idx),
            "global_iter": float(global_iter),
            "split": "train" if is_train else "val",
        }
        log_row.update({key: float(value) for key, value in metrics.items()})
        iter_logs.append(log_row)

        if is_train and step_idx % max(1, int(log_every)) == 0:
            summary = " ".join(
                f"{key}={value:.6f}"
                for key, value in metrics.items()
                if key in ("loss", "mse_scaled", "nll_scaled", "physical_mae")
            )
            print(f"[Epoch {epoch_idx:03d}] step={step_idx:04d} iter={global_iter:06d} {summary}")
        global_iter += 1

    denom = max(total_points, 1)
    epoch_metrics = {"epoch": float(epoch_idx), "points": float(total_points)}
    epoch_metrics.update({key: value / denom for key, value in epoch_totals.items()})
    return epoch_metrics, iter_logs, global_iter


def train_with_early_stopping(
    *,
    model: nn.Module,
    dataloaders: Mapping[str, DataLoader],
    device: torch.device,
    objective_name: str,
    objective_kwargs: Mapping[str, Any],
    epochs: int,
    optimizer: torch.optim.Optimizer,
    patience: int,
    min_delta: float,
    log_every: int,
) -> tuple[nn.Module, pd.DataFrame, pd.DataFrame]:
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    iter_history: list[dict[str, float]] = []
    epoch_history: list[dict[str, float]] = []
    global_iter = 0
    no_improve = 0
    start_t = time.time()

    for epoch in range(1, int(epochs) + 1):
        train_metrics, train_iter, global_iter = run_epoch(
            model,
            dataloaders["train"],
            optimizer=optimizer,
            device=device,
            objective_name=objective_name,
            objective_kwargs=objective_kwargs,
            global_iter_start=global_iter,
            epoch_idx=epoch,
            log_every=log_every,
        )
        train_metrics["split"] = "train"

        with torch.no_grad():
            val_metrics, val_iter, global_iter = run_epoch(
                model,
                dataloaders["val"],
                optimizer=None,
                device=device,
                objective_name=objective_name,
                objective_kwargs=objective_kwargs,
                global_iter_start=global_iter,
                epoch_idx=epoch,
                log_every=log_every,
            )
        val_metrics["split"] = "val"
        iter_history.extend(train_iter)
        iter_history.extend(val_iter)
        epoch_history.append(train_metrics)
        epoch_history.append(val_metrics)

        improved = (best_val_loss - float(val_metrics["loss"])) > float(min_delta)
        if improved:
            best_val_loss = float(val_metrics["loss"])
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}/{int(epochs)} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"no_improve={no_improve}/{int(patience)}"
            )
        if no_improve >= int(patience):
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics, _, _ = run_epoch(
            model,
            dataloaders["test"],
            optimizer=None,
            device=device,
            objective_name=objective_name,
            objective_kwargs=objective_kwargs,
            global_iter_start=global_iter,
            epoch_idx=epoch_history[-1]["epoch"] if epoch_history else 0,
            log_every=log_every,
        )
    test_metrics["split"] = "test"
    epoch_history.append(test_metrics)
    elapsed = time.time() - start_t
    print(f"Training completed in {elapsed:.1f}s. Best val loss={best_val_loss:.6f}")
    return model, pd.DataFrame(iter_history), pd.DataFrame(epoch_history)
