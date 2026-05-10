from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.MLP_training.engineered_feature_common import (  # noqa: E402
    FEATURE_COLUMNS_BY_VARIANT,
    TIME_FEATURE,
    PointwisePenetrationDataset,
    build_dataset_registry,
    build_feature_matrix_np,
)
from MLP.MLP_training.train_stage3_distillation_plus_raw_series import (  # noqa: E402
    build_teacher_raw_dict,
    extract_prefixed_matrix,
    load_source_table,
)


DEFAULT_MLP_BOOTSTRAP = PROJECT_ROOT / "MLP" / "runs_mlp" / "full_pipeline_C_20260509_110100" / "bootstrap_summary.json"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "MLP" / "runs_mlp"
DEFAULT_REPORT_SOURCE = (
    PROJECT_ROOT
    / "MLP"
    / "baseline"
    / "comparison_reports"
    / "stage3_dp_exp_0p5_vs_HA_NS_20260509"
    / "headline_comparison.csv"
)
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "MLP" / "baseline" / "comparison_reports"

METRIC_KEYS = [
    "rmse_mm",
    "mae_mm",
    "bias_mm",
    "p95_abs_err_mm",
    "coverage_1sigma",
    "coverage_2sigma",
]


def bootstrap_ci(values: Sequence[float], *, n_resample: int = 5000, alpha: float = 0.05) -> dict[str, float]:
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and (v != v))],
        dtype=float,
    )
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0, "ci_lo": float(arr[0]), "ci_hi": float(arr[0]), "n": 1}
    rng = np.random.default_rng(20260508)
    means = rng.choice(arr, size=(n_resample, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_lo": float(np.quantile(means, alpha / 2)),
        "ci_hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(arr.size),
    }


def finite_metrics(truth: np.ndarray, pred: np.ndarray, std: np.ndarray) -> dict[str, float | int]:
    resid = pred - truth
    abs_err = np.abs(resid)
    std_safe = np.maximum(std, 1e-12)
    return {
        "n_points": int(truth.size),
        "rmse_mm": float(np.sqrt(np.mean(resid**2))) if truth.size else float("nan"),
        "mae_mm": float(np.mean(abs_err)) if truth.size else float("nan"),
        "bias_mm": float(np.mean(resid)) if truth.size else float("nan"),
        "median_abs_err_mm": float(np.median(abs_err)) if truth.size else float("nan"),
        "p90_abs_err_mm": float(np.quantile(abs_err, 0.90)) if truth.size else float("nan"),
        "p95_abs_err_mm": float(np.quantile(abs_err, 0.95)) if truth.size else float("nan"),
        "coverage_1sigma": float(np.mean(abs_err <= std_safe)) if truth.size else float("nan"),
        "coverage_2sigma": float(np.mean(abs_err <= 2.0 * std_safe)) if truth.size else float("nan"),
        "mean_pred_std_mm": float(np.mean(std)) if truth.size else float("nan"),
    }


def gaussian_nll_scaled(y: np.ndarray, mu: np.ndarray, std: np.ndarray) -> float:
    var = np.maximum(std, 1e-12) ** 2
    return float(np.mean(0.5 * (np.log(var) + (y - mu) ** 2 / var))) if y.size else float("nan")


def resolve_path(path_like: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    raw = str(path_like)
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()

    # The MLP bootstrap manifest may contain Windows absolute paths. Convert
    # those when the runner is launched from WSL/Linux.
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if match and sys.platform != "win32":
        drive, rest = match.groups()
        candidate = Path("/mnt") / drive.lower() / rest.replace("\\", "/")
        if candidate.exists():
            return candidate.resolve()

    if not path.is_absolute():
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
        return candidate
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_a_scale(df: pd.DataFrame, *, rel_tol: float = 5e-5) -> dict[str, float | bool]:
    required = ["delta_pressure_bar_phys", "ambient_density_kg_m3", "diameter_mm", "A_scale"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return {"checked": False, "max_relative_error": float("nan")}
    expected = (
        np.power(pd.to_numeric(df["delta_pressure_bar_phys"], errors="coerce").to_numpy(dtype=float), 0.5)
        * np.power(pd.to_numeric(df["ambient_density_kg_m3"], errors="coerce").to_numpy(dtype=float), -0.25)
        * np.sqrt(pd.to_numeric(df["diameter_mm"], errors="coerce").to_numpy(dtype=float))
    )
    actual = pd.to_numeric(df["A_scale"], errors="coerce").to_numpy(dtype=float)
    rel = np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-12)
    max_rel = float(np.nanmax(rel))
    return {"checked": True, "max_relative_error": max_rel, "passed": bool(max_rel <= rel_tol)}


def maybe_limit_curves(df: pd.DataFrame, *, max_curves_per_split: int | None, seed: int) -> pd.DataFrame:
    if max_curves_per_split is None:
        return df
    parts: list[pd.DataFrame] = []
    for split, group in df.groupby("sample_split", sort=False):
        n = min(int(max_curves_per_split), len(group))
        parts.append(group.sample(n=n, random_state=int(seed)) if n < len(group) else group)
    return pd.concat(parts, ignore_index=True)


class PointTensorDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        target_scaled: torch.Tensor,
        target_physical: torch.Tensor,
        a_scale: torch.Tensor,
    ) -> None:
        self.features = features.detach().cpu().contiguous().float()
        self.target_scaled = target_scaled.detach().cpu().reshape(-1).contiguous().float()
        self.target_physical = target_physical.detach().cpu().reshape(-1).contiguous().float()
        self.a_scale = a_scale.detach().cpu().reshape(-1).contiguous().float()
        if not (
            len(self.features)
            == len(self.target_scaled)
            == len(self.target_physical)
            == len(self.a_scale)
        ):
            raise ValueError("Flattened point tensors have inconsistent lengths.")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "target_scaled": self.target_scaled[idx],
            "target_physical": self.target_physical[idx],
            "a_scale": self.a_scale[idx],
        }

    def with_target_scaled(self, target_scaled: torch.Tensor) -> "PointTensorDataset":
        return PointTensorDataset(self.features, target_scaled, self.target_physical, self.a_scale)


def flatten_curve_dataset(curve_ds: PointwisePenetrationDataset) -> PointTensorDataset:
    if getattr(curve_ds, "_cached_features", None) is None:
        curve_ds._build_cache()
    features = curve_ds._cached_features.reshape(-1, len(curve_ds.feature_columns))
    return PointTensorDataset(
        features,
        curve_ds._cached_target_scaled.reshape(-1),
        curve_ds._cached_target_physical.reshape(-1),
        curve_ds._cached_a_scale.reshape(-1),
    )


def make_point_dataset(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    n_points: int,
    time_min_ms: float,
    time_max_ms: float,
) -> PointTensorDataset:
    curve_ds = PointwisePenetrationDataset(
        df.reset_index(drop=True),
        feature_columns=feature_columns,
        n_points=n_points,
        time_min_ms=time_min_ms,
        time_max_ms=time_max_ms,
        precompute=True,
    )
    return flatten_curve_dataset(curve_ds)


def make_loader(
    dataset: Dataset,
    *,
    batch_points: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_points),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers > 0),
    )


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, *, num_dims: int):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=int(num_dims))
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


@dataclass
class TrainResult:
    model: SVGPModel
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    history: pd.DataFrame
    best_val_loss: float
    best_epoch: int


@dataclass
class GPArtifacts:
    mean_model: SVGPModel
    mean_likelihood: gpytorch.likelihoods.GaussianLikelihood
    var_model: SVGPModel | None
    var_likelihood: gpytorch.likelihoods.GaussianLikelihood | None
    config: dict[str, Any]
    scaler_state: dict[str, Any]
    device: torch.device


def init_inducing_points(
    dataset: PointTensorDataset,
    *,
    num_inducing: int,
    kmeans_samples: int,
    seed: int,
) -> torch.Tensor:
    n_available = len(dataset)
    n_clusters = min(int(num_inducing), n_available)
    if n_clusters <= 0:
        raise ValueError("Cannot initialize inducing points from an empty dataset.")
    rng = np.random.default_rng(int(seed))
    if n_available <= int(kmeans_samples):
        features_np = dataset.features.numpy()
    else:
        idx = rng.choice(n_available, size=int(kmeans_samples), replace=False)
        features_np = dataset.features[idx].numpy()

    if len(features_np) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} k-means samples, got {len(features_np)}.")
    print(f"Running KMeans for {n_clusters} inducing points from {len(features_np)} sampled points...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init="auto")
    kmeans.fit(features_np)
    return torch.as_tensor(kmeans.cluster_centers_, dtype=torch.float32)


def train_svgp(
    *,
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_data: int,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    min_delta: float,
    log_interval: int,
    label: str,
) -> TrainResult:
    model.to(device)
    likelihood.to(device)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=float(lr),
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=int(num_data))
    history: list[dict[str, float | int | str]] = []
    best_state: dict[str, Any] | None = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, int(epochs) + 1):
        train_loss = run_svgp_epoch(
            model=model,
            likelihood=likelihood,
            loader=train_loader,
            mll=mll,
            device=device,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_loss = run_svgp_epoch(
                model=model,
                likelihood=likelihood,
                loader=val_loader,
                mll=mll,
                device=device,
                optimizer=None,
            )
        history.append({"model": label, "epoch": epoch, "split": "train", "loss": train_loss})
        history.append({"model": label, "epoch": epoch, "split": "val", "loss": val_loss})

        if val_loss < best_val - float(min_delta):
            best_val = float(val_loss)
            best_epoch = int(epoch)
            no_improve = 0
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "likelihood": copy.deepcopy(likelihood.state_dict()),
            }
        else:
            no_improve += 1

        if epoch == 1 or epoch % int(log_interval) == 0 or no_improve >= int(patience):
            print(
                f"[{label}] epoch {epoch:03d}/{int(epochs)} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"best={best_val:.6f}@{best_epoch} no_improve={no_improve}/{int(patience)}"
            )
        if no_improve >= int(patience):
            print(f"[{label}] early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    return TrainResult(
        model=model,
        likelihood=likelihood,
        history=pd.DataFrame(history),
        best_val_loss=float(best_val),
        best_epoch=int(best_epoch),
    )


def run_svgp_epoch(
    *,
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    loader: DataLoader,
    mll: gpytorch.mlls.VariationalELBO,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(mode=is_train)
    likelihood.train(mode=is_train)
    total_loss = 0.0
    total_points = 0
    for batch in loader:
        x = batch["features"].to(device, non_blocking=True)
        y = batch["target_scaled"].to(device, non_blocking=True).reshape(-1)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = -mll(output, y)
        if is_train:
            loss.backward()
            optimizer.step()
        batch_n = int(y.numel())
        total_loss += float(loss.detach().cpu()) * batch_n
        total_points += batch_n
    return total_loss / max(total_points, 1)


def predict_scaled(
    model: SVGPModel,
    dataset_or_features: PointTensorDataset | np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    batch_points: int,
) -> np.ndarray:
    model.eval()
    if isinstance(dataset_or_features, PointTensorDataset):
        features = dataset_or_features.features
    elif isinstance(dataset_or_features, np.ndarray):
        features = torch.as_tensor(dataset_or_features, dtype=torch.float32)
    else:
        features = dataset_or_features.detach().cpu().float()

    chunks: list[np.ndarray] = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(features), int(batch_points)):
            stop = min(start + int(batch_points), len(features))
            x = features[start:stop].to(device, non_blocking=True)
            chunks.append(model(x).mean.detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)


def predict_physical(
    artifacts: GPArtifacts,
    features: np.ndarray | torch.Tensor,
    a_scale: np.ndarray | torch.Tensor,
    *,
    batch_points: int,
    include_mean_posterior_var: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifacts.mean_model.eval()
    artifacts.mean_likelihood.eval()
    if artifacts.var_model is not None:
        artifacts.var_model.eval()
    features_t = torch.as_tensor(features, dtype=torch.float32).detach().cpu()
    a_scale_t = torch.as_tensor(a_scale, dtype=torch.float32).reshape(-1).detach().cpu()
    mean_chunks: list[np.ndarray] = []
    std_chunks: list[np.ndarray] = []
    mean_scaled_chunks: list[np.ndarray] = []
    std_scaled_chunks: list[np.ndarray] = []

    logvar_min = float(artifacts.config.get("logvar_clamp_min", -20.0))
    logvar_max = float(artifacts.config.get("logvar_clamp_max", 10.0))
    std_floor_scaled = float(artifacts.config.get("std_floor_scaled", 1e-6))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, len(features_t), int(batch_points)):
            stop = min(start + int(batch_points), len(features_t))
            x = features_t[start:stop].to(artifacts.device, non_blocking=True)
            scale = a_scale_t[start:stop].to(artifacts.device, non_blocking=True)
            mean_dist = artifacts.mean_model(x)
            mean_scaled = mean_dist.mean.reshape(-1)

            if artifacts.var_model is not None:
                log_var_scaled = artifacts.var_model(x).mean.reshape(-1)
                log_var_scaled = log_var_scaled + float(artifacts.config.get("logvar_bias_correction", 0.0))
                var_scaled = torch.exp(torch.clamp(log_var_scaled, min=logvar_min, max=logvar_max))
                if include_mean_posterior_var:
                    var_scaled = var_scaled + torch.clamp(mean_dist.variance.reshape(-1), min=0.0)
            else:
                var_scaled = torch.clamp(mean_dist.variance.reshape(-1), min=float(artifacts.config.get("std_floor_scaled", 1e-6)) ** 2)
                noise = getattr(artifacts.mean_likelihood, "noise", None)
                if noise is not None:
                    var_scaled = var_scaled + torch.clamp(noise.reshape(-1)[0], min=0.0)

            std_scaled = torch.sqrt(torch.clamp(var_scaled, min=std_floor_scaled**2))
            mean_physical = mean_scaled * scale
            std_physical = std_scaled * scale

            mean_scaled_chunks.append(mean_scaled.detach().cpu().numpy())
            std_scaled_chunks.append(std_scaled.detach().cpu().numpy())
            mean_chunks.append(mean_physical.detach().cpu().numpy())
            std_chunks.append(std_physical.detach().cpu().numpy())

    return (
        np.concatenate(mean_chunks) if mean_chunks else np.asarray([], dtype=np.float32),
        np.concatenate(std_chunks) if std_chunks else np.asarray([], dtype=np.float32),
        np.concatenate(mean_scaled_chunks) if mean_scaled_chunks else np.asarray([], dtype=np.float32),
        np.concatenate(std_scaled_chunks) if std_scaled_chunks else np.asarray([], dtype=np.float32),
    )


def residual_log_targets(
    *,
    model: SVGPModel,
    dataset: PointTensorDataset,
    device: torch.device,
    batch_points: int,
    residual_eps: float,
) -> torch.Tensor:
    pred = predict_scaled(model, dataset, device=device, batch_points=batch_points)
    residual_sq = (dataset.target_scaled.numpy() - pred) ** 2
    return torch.as_tensor(np.log(residual_sq + float(residual_eps)), dtype=torch.float32)


def estimate_logvar_bias_correction(
    *,
    var_model: SVGPModel,
    var_dataset: PointTensorDataset,
    device: torch.device,
    batch_points: int,
) -> float:
    pred_logvar = predict_scaled(var_model, var_dataset, device=device, batch_points=batch_points)
    target_logvar = var_dataset.target_scaled.numpy()
    ratio = np.exp(np.clip(target_logvar - pred_logvar, -30.0, 30.0))
    finite = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if finite.size == 0:
        return 0.0
    return float(np.log(np.mean(finite)))


def evaluate_dataset(
    artifacts: GPArtifacts,
    dataset: PointTensorDataset,
    *,
    batch_points: int,
    include_mean_posterior_var: bool,
) -> dict[str, float | int]:
    pred, std, pred_scaled, std_scaled = predict_physical(
        artifacts,
        dataset.features,
        dataset.a_scale,
        batch_points=batch_points,
        include_mean_posterior_var=include_mean_posterior_var,
    )
    truth = dataset.target_physical.numpy()
    metrics = finite_metrics(truth, pred, std)
    metrics["nll_scaled"] = gaussian_nll_scaled(dataset.target_scaled.numpy(), pred_scaled, std_scaled)
    return metrics


def benchmark_prediction_latency(
    artifacts: GPArtifacts,
    dataset: PointTensorDataset,
    *,
    n_points: int,
    repeats: int,
    batch_points: int,
    include_mean_posterior_var: bool,
) -> dict[str, float | int]:
    n = min(int(n_points), len(dataset))
    if n <= 0:
        return {"n_points": 0, "median_ms_per_point": float("nan"), "median_total_ms": float("nan")}
    features = dataset.features[:n]
    a_scale = dataset.a_scale[:n]
    times_ms: list[float] = []
    for _ in range(max(int(repeats), 1)):
        cuda_synchronize(artifacts.device)
        t0 = time.perf_counter()
        predict_physical(
            artifacts,
            features,
            a_scale,
            batch_points=batch_points,
            include_mean_posterior_var=include_mean_posterior_var,
        )
        cuda_synchronize(artifacts.device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "n_points": int(n),
        "repeats": int(max(int(repeats), 1)),
        "median_total_ms": float(np.median(times_ms)),
        "median_ms_per_point": float(np.median(times_ms) / n),
    }


def save_checkpoint(
    path: Path,
    *,
    artifacts: GPArtifacts,
    seed: int,
    stage1_run: Path,
    train_history: pd.DataFrame,
) -> None:
    payload = {
        "method": artifacts.config["method"],
        "seed": int(seed),
        "stage1_run": str(stage1_run),
        "config": artifacts.config,
        "scaler_state": artifacts.scaler_state,
        "mean_model_state": artifacts.mean_model.state_dict(),
        "mean_likelihood_state": artifacts.mean_likelihood.state_dict(),
        "mean_inducing_points": artifacts.mean_model.variational_strategy.inducing_points.detach().cpu(),
        "var_model_state": artifacts.var_model.state_dict() if artifacts.var_model is not None else None,
        "var_likelihood_state": artifacts.var_likelihood.state_dict() if artifacts.var_likelihood is not None else None,
        "var_inducing_points": (
            artifacts.var_model.variational_strategy.inducing_points.detach().cpu()
            if artifacts.var_model is not None
            else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    train_history.to_csv(path.parent / "train_log.csv", index=False)


def load_gp_artifacts(checkpoint_path: Path, *, device: torch.device) -> GPArtifacts:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = dict(payload["config"])
    feature_columns = list(config["feature_columns"])
    mean_inducing = payload["mean_inducing_points"].float()
    mean_model = SVGPModel(mean_inducing, num_dims=len(feature_columns))
    mean_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean_model.load_state_dict(payload["mean_model_state"])
    mean_likelihood.load_state_dict(payload["mean_likelihood_state"])

    var_model = None
    var_likelihood = None
    if payload.get("var_model_state") is not None and payload.get("var_inducing_points") is not None:
        var_model = SVGPModel(payload["var_inducing_points"].float(), num_dims=len(feature_columns))
        var_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        var_model.load_state_dict(payload["var_model_state"])
        var_likelihood.load_state_dict(payload["var_likelihood_state"])

    mean_model.to(device).eval()
    mean_likelihood.to(device).eval()
    if var_model is not None:
        var_model.to(device).eval()
    if var_likelihood is not None:
        var_likelihood.to(device).eval()

    return GPArtifacts(
        mean_model=mean_model,
        mean_likelihood=mean_likelihood,
        var_model=var_model,
        var_likelihood=var_likelihood,
        config=config,
        scaler_state=dict(payload["scaler_state"]),
        device=device,
    )


def run_seed(
    *,
    seed_info: Mapping[str, Any],
    run_dir: Path,
    base_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed = int(seed_info["seed"])
    set_seed(seed)
    stage1_run = resolve_path(seed_info["stage1_run"])
    row_table_path = stage1_run / "row_table.csv"
    if not row_table_path.exists():
        raise FileNotFoundError(f"Missing row_table.csv for seed {seed}: {row_table_path}")
    seed_dir = run_dir / "per_seed" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(row_table_path)
    df = maybe_limit_curves(df, max_curves_per_split=base_config.get("max_curves_per_split"), seed=seed)
    a_scale_check = validate_a_scale(df)
    if a_scale_check.get("checked") and not a_scale_check.get("passed") and not base_config.get("allow_a_scale_mismatch", False):
        raise ValueError(
            f"A_scale mismatch in {row_table_path}; max relative error "
            f"{a_scale_check['max_relative_error']:.3e}. Pass --allow-a-scale-mismatch to override."
        )

    stage1_config_path = stage1_run / "train_config_used.json"
    scaler_state_path = stage1_run / "scaler_state.json"
    stage1_config = read_json(stage1_config_path) if stage1_config_path.exists() else {}
    scaler_state = read_json(scaler_state_path) if scaler_state_path.exists() else {}
    feature_columns = list(base_config["feature_columns"])
    n_points = int(base_config["n_points"])
    time_min_ms = float(base_config["time_min_ms"])
    time_max_ms = float(base_config["time_max_ms"])

    split_counts = df["sample_split"].value_counts().to_dict()
    print(f"\n--- GP seed {seed} ---")
    print("Stage-1 run:", stage1_run)
    print("Curve split counts:", split_counts)
    print("A_scale check:", a_scale_check)

    datasets = {
        split: make_point_dataset(
            df.loc[df["sample_split"] == split],
            feature_columns=feature_columns,
            n_points=n_points,
            time_min_ms=time_min_ms,
            time_max_ms=time_max_ms,
        )
        for split in ("train", "val", "test")
    }
    print("Point split counts:", {split: len(ds) for split, ds in datasets.items()})

    loaders = {
        "train": make_loader(
            datasets["train"],
            batch_points=int(base_config["batch_points"]),
            shuffle=True,
            seed=seed,
            num_workers=int(base_config["num_workers"]),
            device=device,
        ),
        "val": make_loader(
            datasets["val"],
            batch_points=int(base_config["eval_batch_points"]),
            shuffle=False,
            seed=seed,
            num_workers=int(base_config["num_workers"]),
            device=device,
        ),
        "test": make_loader(
            datasets["test"],
            batch_points=int(base_config["eval_batch_points"]),
            shuffle=False,
            seed=seed,
            num_workers=int(base_config["num_workers"]),
            device=device,
        ),
    }

    inducing = init_inducing_points(
        datasets["train"],
        num_inducing=int(base_config["num_inducing"]),
        kmeans_samples=int(base_config["kmeans_samples"]),
        seed=seed,
    ).to(device)

    mean_model = SVGPModel(inducing.clone(), num_dims=len(feature_columns))
    mean_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    print("Training mean SVGP...")
    mean_result = train_svgp(
        model=mean_model,
        likelihood=mean_likelihood,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        num_data=len(datasets["train"]),
        device=device,
        epochs=int(base_config["epochs"]),
        lr=float(base_config["learning_rate"]),
        patience=int(base_config["early_stopping_patience"]),
        min_delta=float(base_config["early_stopping_min_delta"]),
        log_interval=int(base_config["log_interval"]),
        label="mean",
    )

    var_result: TrainResult | None = None
    logvar_bias_correction = 0.0
    if not bool(base_config.get("mean_only", False)):
        print("Computing log-residual targets for variance SVGP...")
        train_log_resid = residual_log_targets(
            model=mean_result.model,
            dataset=datasets["train"],
            device=device,
            batch_points=int(base_config["eval_batch_points"]),
            residual_eps=float(base_config["residual_eps"]),
        )
        val_log_resid = residual_log_targets(
            model=mean_result.model,
            dataset=datasets["val"],
            device=device,
            batch_points=int(base_config["eval_batch_points"]),
            residual_eps=float(base_config["residual_eps"]),
        )
        var_train_ds = datasets["train"].with_target_scaled(train_log_resid)
        var_val_ds = datasets["val"].with_target_scaled(val_log_resid)
        var_train_loader = make_loader(
            var_train_ds,
            batch_points=int(base_config["batch_points"]),
            shuffle=True,
            seed=seed + 1009,
            num_workers=int(base_config["num_workers"]),
            device=device,
        )
        var_val_loader = make_loader(
            var_val_ds,
            batch_points=int(base_config["eval_batch_points"]),
            shuffle=False,
            seed=seed + 1009,
            num_workers=int(base_config["num_workers"]),
            device=device,
        )

        var_model = SVGPModel(inducing.clone(), num_dims=len(feature_columns))
        var_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        print("Training log-variance SVGP...")
        var_result = train_svgp(
            model=var_model,
            likelihood=var_likelihood,
            train_loader=var_train_loader,
            val_loader=var_val_loader,
            num_data=len(var_train_ds),
            device=device,
            epochs=int(base_config["var_epochs"]),
            lr=float(base_config["var_learning_rate"]),
            patience=int(base_config["early_stopping_patience"]),
            min_delta=float(base_config["early_stopping_min_delta"]),
            log_interval=int(base_config["log_interval"]),
            label="logvar",
        )
        if bool(base_config.get("calibrate_logvar_bias", True)):
            logvar_bias_correction = estimate_logvar_bias_correction(
                var_model=var_result.model,
                var_dataset=var_val_ds,
                device=device,
                batch_points=int(base_config["eval_batch_points"]),
            )
            print(f"Validation log-variance bias correction: {logvar_bias_correction:.6f}")

    seed_config = {
        **dict(base_config),
        "seed": seed,
        "stage1_run": str(stage1_run),
        "stage1_feature_columns": stage1_config.get("feature_columns"),
        "split_curve_counts": split_counts,
        "a_scale_check": a_scale_check,
        "logvar_bias_correction": float(logvar_bias_correction),
        "method": "svgp_heteroscedastic_two_stage" if var_result is not None else "svgp_homoscedastic",
    }
    artifacts = GPArtifacts(
        mean_model=mean_result.model,
        mean_likelihood=mean_result.likelihood,
        var_model=var_result.model if var_result is not None else None,
        var_likelihood=var_result.likelihood if var_result is not None else None,
        config=seed_config,
        scaler_state=scaler_state,
        device=device,
    )

    print("Evaluating GP on held-out in-pipeline test split...")
    test_metrics = evaluate_dataset(
        artifacts,
        datasets["test"],
        batch_points=int(base_config["eval_batch_points"]),
        include_mean_posterior_var=bool(base_config["include_mean_posterior_var"]),
    )
    val_metrics = evaluate_dataset(
        artifacts,
        datasets["val"],
        batch_points=int(base_config["eval_batch_points"]),
        include_mean_posterior_var=bool(base_config["include_mean_posterior_var"]),
    )
    latency = benchmark_prediction_latency(
        artifacts,
        datasets["test"],
        n_points=int(base_config["latency_points"]),
        repeats=int(base_config["latency_repeats"]),
        batch_points=int(base_config["eval_batch_points"]),
        include_mean_posterior_var=bool(base_config["include_mean_posterior_var"]),
    )

    histories = [mean_result.history]
    if var_result is not None:
        histories.append(var_result.history)
    train_history = pd.concat(histories, ignore_index=True)
    checkpoint_path = seed_dir / "model.pt"
    save_checkpoint(
        checkpoint_path,
        artifacts=artifacts,
        seed=seed,
        stage1_run=stage1_run,
        train_history=train_history,
    )
    write_json(seed_dir / "test_metrics.json", test_metrics)
    write_json(seed_dir / "val_metrics.json", val_metrics)
    write_json(seed_dir / "latency.json", latency)
    write_json(seed_dir / "seed_config.json", seed_config)
    write_json(seed_dir / "scaler_state.json", scaler_state)

    result = {
        "seed": seed,
        "stage1_run": str(stage1_run),
        "checkpoint": str(checkpoint_path),
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "latency": latency,
        "best_epochs": {
            "mean": mean_result.best_epoch,
            "logvar": var_result.best_epoch if var_result is not None else None,
        },
        "best_val_losses": {
            "mean": mean_result.best_val_loss,
            "logvar": var_result.best_val_loss if var_result is not None else None,
        },
    }
    print("Test metrics:", json.dumps(to_jsonable(test_metrics), indent=2))
    print("Latency:", json.dumps(to_jsonable(latency), indent=2))
    return result


def aggregate_results(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bootstrap = {"metrics": {}}
    for key in METRIC_KEYS:
        bootstrap["metrics"][key] = bootstrap_ci([float(r["test_metrics"][key]) for r in seed_results])
    latency_values = [
        float(r.get("latency", {}).get("median_ms_per_point", float("nan")))
        for r in seed_results
    ]
    bootstrap["latency_ms_per_point"] = bootstrap_ci(latency_values)
    return bootstrap


def run_external_rmse_evaluation(
    *,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    output_root: Path,
    tag: str,
    batch_points: int,
    fast: bool,
    save_points: bool,
) -> tuple[Path, dict[str, Any]]:
    artifacts = load_gp_artifacts(checkpoint_path, device=device)
    registry = build_dataset_registry()
    cdf_wide_df = load_source_table("cdf", split=split)

    time_cols = [c for c in cdf_wide_df.columns if c.startswith("time_ms_")]
    time_cols.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    frame_ids = [int(c.rsplit("_", 1)[1]) for c in time_cols]

    feature_blocks: list[np.ndarray] = []
    a_scale_blocks: list[np.ndarray] = []
    truth_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []
    traj_meta_rows: list[dict[str, Any]] = []

    feature_columns = list(artifacts.config["feature_columns"])
    time_feature = str(artifacts.config.get("time_feature", TIME_FEATURE))
    t_min_ms = float(artifacts.config.get("time_min_ms", 0.0))
    t_max_ms = float(artifacts.config.get("time_max_ms", 5.0))

    for row_idx, row in cdf_wide_df.iterrows():
        time_vals = extract_prefixed_matrix(cdf_wide_df.iloc[[row_idx]], "time_ms_", frame_ids).reshape(-1)
        pen_vals = extract_prefixed_matrix(cdf_wide_df.iloc[[row_idx]], "penetration_mm_", frame_ids).reshape(-1)
        valid = np.isfinite(time_vals) & np.isfinite(pen_vals) & (time_vals >= t_min_ms) & (time_vals <= t_max_ms)
        if not np.any(valid):
            continue

        time_valid = time_vals[valid].astype(np.float32)
        truth_valid = pen_vals[valid].astype(np.float32)
        raw = build_teacher_raw_dict(row)
        features_np, a_scale_np, _ = build_feature_matrix_np(
            raw,
            time_valid,
            artifacts.scaler_state,
            feature_columns,
            registry,
            time_feature=time_feature,
        )

        folder = str(row.get("experiment_name", "unknown"))
        test_name = str(Path(str(row.get("file_path", row.get("file_name", row_idx)))).parent.name)
        traj_key = f"{folder}|{test_name}|{row.get('file_name', row_idx)}|plume={int(row.get('plume_idx', -1))}"
        feature_blocks.append(features_np)
        a_scale_blocks.append(a_scale_np.reshape(-1))
        truth_blocks.append(truth_valid)
        meta_blocks.append(
            pd.DataFrame(
                {
                    "folder": folder,
                    "test_name": test_name,
                    "traj_key": traj_key,
                    "row_idx": int(row_idx),
                    "plume_idx": int(row.get("plume_idx", -1)),
                    "injection_duration_us": float(row.get("injection_duration_us", np.nan)),
                    "time_ms": time_valid,
                }
            )
        )
        traj_meta_rows.append(
            {
                "traj_key": traj_key,
                "folder": folder,
                "test_name": test_name,
                "row_idx": int(row_idx),
                "plume_idx": int(row.get("plume_idx", -1)),
                "injection_duration_us": float(row.get("injection_duration_us", np.nan)),
                "max_observed_time_ms": float(np.max(time_valid)),
            }
        )

        if fast and len(traj_meta_rows) >= 512:
            break

    if not feature_blocks:
        raise RuntimeError("No finite CDF points found for external GP RMSE evaluation.")

    features = np.vstack(feature_blocks).astype(np.float32)
    a_scale = np.concatenate(a_scale_blocks).astype(np.float32)
    truth = np.concatenate(truth_blocks).astype(np.float32)
    meta_df = pd.concat(meta_blocks, ignore_index=True)
    pred, std, _pred_scaled, _std_scaled = predict_physical(
        artifacts,
        features,
        a_scale,
        batch_points=int(batch_points),
        include_mean_posterior_var=bool(artifacts.config.get("include_mean_posterior_var", False)),
    )

    points_df = meta_df.copy()
    points_df["pen_true_mm"] = truth
    points_df["pen_pred_mm"] = pred
    points_df["pen_std_mm"] = std
    points_df["resid_mm"] = pred - truth

    traj_meta = pd.DataFrame(traj_meta_rows)
    per_traj_rows: list[dict[str, Any]] = []
    for traj_key, group in points_df.groupby("traj_key", dropna=False):
        resid = group["resid_mm"].to_numpy(dtype=float)
        abs_resid = np.abs(resid)
        std_safe = np.maximum(group["pen_std_mm"].to_numpy(dtype=float), 1e-12)
        per_traj_rows.append(
            {
                "traj_key": traj_key,
                "n_points": int(len(group)),
                "rmse_mm": float(np.sqrt(np.mean(resid**2))),
                "mae_mm": float(np.mean(abs_resid)),
                "mean_pen_mm": float(np.mean(group["pen_true_mm"])),
                "max_pen_mm": float(np.max(group["pen_true_mm"])),
                "coverage_1sigma": float(np.mean(abs_resid <= std_safe)),
                "coverage_2sigma": float(np.mean(abs_resid <= 2.0 * std_safe)),
            }
        )
    per_traj = traj_meta.merge(pd.DataFrame(per_traj_rows), on="traj_key", how="inner")
    per_traj["is_censored"] = per_traj["max_observed_time_ms"] < (t_max_ms - 1e-6)
    per_traj = per_traj.drop(columns=["max_observed_time_ms"])
    per_folder = (
        per_traj.groupby("folder", dropna=False)
        .agg(
            n_traj=("traj_key", "count"),
            rmse_mean_mm=("rmse_mm", "mean"),
            rmse_median_mm=("rmse_mm", "median"),
            mae_mean_mm=("mae_mm", "mean"),
            coverage_1sigma=("coverage_1sigma", "mean"),
            coverage_2sigma=("coverage_2sigma", "mean"),
        )
        .reset_index()
    )

    metrics = finite_metrics(truth, pred, std)
    rel_abs = np.abs(points_df["resid_mm"].to_numpy(dtype=float)) / np.maximum(
        np.abs(points_df["pen_true_mm"].to_numpy(dtype=float)),
        5.0,
    )
    truth_range = float(np.max(truth) - np.min(truth))
    overall = {
        **metrics,
        "n_trajectories": int(len(per_traj)),
        "mean_rel_err": float(np.mean(rel_abs)),
        "median_rel_err": float(np.median(rel_abs)),
        "truth_min_mm": float(np.min(truth)),
        "truth_max_mm": float(np.max(truth)),
        "truth_range_mm": truth_range,
        "nrmse_range": float(metrics["rmse_mm"] / truth_range) if truth_range > 0 else float("nan"),
    }

    out_dir = output_root / f"gp_rmse_eval_{split}_{pd.Timestamp.now():%Y%m%d_%H%M%S}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=False)
    if save_points:
        points_df.to_csv(out_dir / "points.csv", index=False)
    per_traj.to_csv(out_dir / "per_trajectory.csv", index=False)
    per_folder.to_csv(out_dir / "per_folder.csv", index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "eval_mode": {"fast": bool(fast), "save_points": bool(save_points), "batch_points": int(batch_points)},
        "feature_columns": feature_columns,
        "overall": overall,
    }
    write_json(out_dir / "metrics_summary.json", summary)
    return out_dir, summary


def write_external_comparison_report(
    *,
    source_headline: Path,
    output_dir: Path,
    external_results: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    if source_headline.exists():
        frames.append(pd.read_csv(source_headline))

    rows: list[dict[str, Any]] = []
    for item in external_results:
        overall = dict(item["summary"]["overall"])
        seed = int(item["seed"])
        rows.append(
            {
                "model": f"Sparse heteroscedastic GP (seed {seed})",
                **{k: overall.get(k) for k in [
                    "n_points",
                    "n_trajectories",
                    "rmse_mm",
                    "mae_mm",
                    "bias_mm",
                    "median_abs_err_mm",
                    "p90_abs_err_mm",
                    "p95_abs_err_mm",
                    "coverage_1sigma",
                    "coverage_2sigma",
                    "mean_pred_std_mm",
                    "mean_rel_err",
                    "median_rel_err",
                    "nrmse_range",
                ]},
                "seed": seed,
                "std": np.nan,
            }
        )
    if rows:
        seed_df = pd.DataFrame(rows)
        mean_row: dict[str, Any] = {"model": "Sparse heteroscedastic GP mean", "seed": np.nan}
        for col in seed_df.columns:
            if col in {"model", "seed"}:
                continue
            if pd.api.types.is_numeric_dtype(seed_df[col]):
                mean_row[col] = float(seed_df[col].mean())
        mean_row["std"] = float(seed_df["rmse_mm"].std(ddof=1)) if len(seed_df) > 1 else 0.0
        frames.append(pd.concat([seed_df, pd.DataFrame([mean_row])], ignore_index=True))

    if not frames:
        return
    headline = pd.concat(frames, ignore_index=True, sort=False)
    csv_path = output_dir / "headline_comparison.csv"
    md_path = output_dir / "headline_comparison.md"
    headline.to_csv(csv_path, index=False)
    try:
        md_text = headline.to_markdown(index=False)
    except ImportError:
        md_text = "```\n" + headline.to_string(index=False) + "\n```\n"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Wrote comparison report: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sparse heteroscedastic GP baseline against the Stage-2 NLL MLP.")
    parser.add_argument("--mlp-bootstrap", type=Path, default=DEFAULT_MLP_BOOTSTRAP)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Subset/override seeds from the MLP bootstrap manifest.")
    parser.add_argument("--variant", choices=tuple(FEATURE_COLUMNS_BY_VARIANT.keys()), default="a_only")
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--batch-points", type=int, default=1024)
    parser.add_argument("--eval-batch-points", type=int, default=65536)
    parser.add_argument("--num-inducing", type=int, default=256)
    parser.add_argument("--kmeans-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--var-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--var-learning-rate", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--time-min-ms", type=float, default=0.0)
    parser.add_argument("--time-max-ms", type=float, default=5.0)
    parser.add_argument("--residual-eps", type=float, default=1e-6)
    parser.add_argument("--std-floor-scaled", type=float, default=1e-6)
    parser.add_argument("--logvar-clamp-min", type=float, default=-20.0)
    parser.add_argument("--logvar-clamp-max", type=float, default=10.0)
    parser.add_argument("--mean-only", action="store_true", help="Train only the homoscedastic mean SVGP sanity baseline.")
    parser.add_argument("--include-mean-posterior-var", action="store_true")
    parser.add_argument("--no-logvar-bias-calibration", action="store_true")
    parser.add_argument("--max-curves-per-split", type=int, default=None, help="Debug knob; not for paper metrics.")
    parser.add_argument("--allow-a-scale-mismatch", action="store_true")
    parser.add_argument("--latency-points", type=int, default=10000)
    parser.add_argument("--latency-repeats", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--external-eval", action="store_true", help="Run the external clean CDF diagnostic after training.")
    parser.add_argument("--external-split", choices=("clean", "all"), default="clean")
    parser.add_argument("--external-fast", action="store_true")
    parser.add_argument("--external-save-points", action="store_true")
    parser.add_argument("--report-source-headline", type=Path, default=DEFAULT_REPORT_SOURCE)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    mlp_bootstrap = resolve_path(args.mlp_bootstrap)
    bootstrap_data = read_json(mlp_bootstrap)
    per_seed_manifest = list(bootstrap_data["per_seed"])
    if args.seeds:
        requested = {int(seed) for seed in args.seeds}
        per_seed_manifest = [rec for rec in per_seed_manifest if int(rec["seed"]) in requested]
        missing = requested - {int(rec["seed"]) for rec in per_seed_manifest}
        if missing:
            raise ValueError(f"Requested seeds not found in bootstrap manifest: {sorted(missing)}")

    feature_columns = FEATURE_COLUMNS_BY_VARIANT[args.variant]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = resolve_path(args.runs_root)
    out_dir = runs_root / f"gp_baseline_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = {
        "method": "svgp_heteroscedastic_two_stage",
        "mlp_bootstrap": str(mlp_bootstrap),
        "variant": args.variant,
        "feature_columns": feature_columns,
        "time_feature": TIME_FEATURE,
        "n_points": int(args.n_points),
        "batch_points": int(args.batch_points),
        "eval_batch_points": int(args.eval_batch_points),
        "num_inducing": int(args.num_inducing),
        "kmeans_samples": int(args.kmeans_samples),
        "epochs": int(args.epochs),
        "var_epochs": int(args.var_epochs if args.var_epochs is not None else args.epochs),
        "learning_rate": float(args.learning_rate),
        "var_learning_rate": float(args.var_learning_rate if args.var_learning_rate is not None else args.learning_rate),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "log_interval": int(args.log_interval),
        "num_workers": int(args.num_workers),
        "time_min_ms": float(args.time_min_ms),
        "time_max_ms": float(args.time_max_ms),
        "residual_eps": float(args.residual_eps),
        "std_floor_scaled": float(args.std_floor_scaled),
        "logvar_clamp_min": float(args.logvar_clamp_min),
        "logvar_clamp_max": float(args.logvar_clamp_max),
        "mean_only": bool(args.mean_only),
        "include_mean_posterior_var": bool(args.include_mean_posterior_var),
        "calibrate_logvar_bias": not bool(args.no_logvar_bias_calibration),
        "max_curves_per_split": args.max_curves_per_split,
        "allow_a_scale_mismatch": bool(args.allow_a_scale_mismatch),
        "latency_points": int(args.latency_points),
        "latency_repeats": int(args.latency_repeats),
        "device": str(device),
    }
    write_json(out_dir / "gp_config_resolved.json", base_config)
    print("Output dir:", out_dir)
    print("Device:", device)
    print("Seeds:", [int(rec["seed"]) for rec in per_seed_manifest])
    print("Feature columns:", feature_columns)
    if args.dry_run:
        for rec in per_seed_manifest:
            stage1_run = resolve_path(rec["stage1_run"])
            print(f"[dry-run] seed {rec['seed']} row_table={stage1_run / 'row_table.csv'}")
        return

    seed_results: list[dict[str, Any]] = []
    external_results: list[dict[str, Any]] = []
    for seed_info in per_seed_manifest:
        result = run_seed(seed_info=seed_info, run_dir=out_dir, base_config=base_config, device=device)
        seed_results.append(result)
        pd.DataFrame(
            [
                {"seed": item["seed"], **{k: item["test_metrics"].get(k) for k in item["test_metrics"]}}
                for item in seed_results
            ]
        ).to_csv(out_dir / "gp_baseline_per_seed.csv", index=False)

        if args.external_eval:
            seed = int(result["seed"])
            ext_dir, ext_summary = run_external_rmse_evaluation(
                checkpoint_path=Path(result["checkpoint"]),
                split=args.external_split,
                device=device,
                output_root=out_dir / "external_eval",
                tag=f"seed_{seed}",
                batch_points=int(args.eval_batch_points),
                fast=bool(args.external_fast),
                save_points=bool(args.external_save_points),
            )
            external_results.append({"seed": seed, "out_dir": str(ext_dir), "summary": ext_summary})
            write_json(out_dir / "external_eval_summary.json", {"per_seed": external_results})

    bootstrap = aggregate_results(seed_results)
    summary = {
        "method": base_config["method"],
        "seeds": [int(r["seed"]) for r in seed_results],
        "per_seed": seed_results,
        "bootstrap": bootstrap,
        "config": base_config,
    }
    write_json(out_dir / "bootstrap_summary.json", summary)

    if external_results:
        report_dir = resolve_path(args.report_root) / f"gp_vs_mlp_{datetime.now():%Y%m%d}"
        write_external_comparison_report(
            source_headline=resolve_path(args.report_source_headline),
            output_dir=report_dir,
            external_results=external_results,
        )

    print("\n--- Final GP Bootstrap Metrics ---")
    for key, stats in bootstrap["metrics"].items():
        print(
            f"{key}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
            f"(95% CI {stats['ci_lo']:.4f} to {stats['ci_hi']:.4f}, n={stats['n']})"
        )
    latency = bootstrap.get("latency_ms_per_point", {})
    if latency:
        print(
            "latency_ms_per_point: "
            f"{latency['mean']:.6f} +/- {latency['std']:.6f} "
            f"(95% CI {latency['ci_lo']:.6f} to {latency['ci_hi']:.6f})"
        )
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
