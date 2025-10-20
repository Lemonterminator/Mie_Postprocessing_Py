# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import torch
from torch import nn


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("Feature values must be scalar or 1-D arrays.")
    return arr


def _standardize(arr: np.ndarray, scaler_state: Mapping[str, Any]) -> np.ndarray:
    mu = np.asarray(scaler_state["mu"], dtype=np.float32)
    sigma = np.asarray(scaler_state["sigma"], dtype=np.float32)
    return (arr - mu) / sigma


def _destandardize(arr: np.ndarray, scaler_state: Mapping[str, Any]) -> np.ndarray:
    mu = np.asarray(scaler_state["mu"], dtype=np.float32)
    sigma = np.asarray(scaler_state["sigma"], dtype=np.float32)
    return arr * sigma + mu


def make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'")


class PenetrationMLP(nn.Module):
    """Mirror of the training-time MLP architecture."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        *,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    make_activation(activation),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_run(run_dir: str | Path) -> Dict[str, Any]:
    """Load heteroscedastic penetration run artifacts."""
    run_path = Path(run_dir)
    model_path = run_path / "best_model.pt"
    scalers_path = run_path / "scalers.json"
    cfg_path = run_path / "model_config.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights at {model_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Missing scalers file at {scalers_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing model config at {cfg_path}")

    cfg = _load_json(cfg_path)
    scalers = _load_json(scalers_path)

    model_cfg = cfg["model"]
    model = PenetrationMLP(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dims=list(model_cfg["hidden_dims"]),
        output_dim=int(model_cfg["output_dim"]),
        activation=model_cfg["activation"],
        dropout=float(model_cfg["dropout"]),
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    data_cfg = cfg["data"]
    std_clamp_min = float(data_cfg.get("std_clamp_min", 0.0))

    return {
        "model": model,
        "device": torch.device("cpu"),
        "scalers": scalers,
        "config": cfg,
        "feature_columns": list(data_cfg["feature_columns"]),
        "time_feature": data_cfg.get("time_feature"),
        "target_column": data_cfg["target_column"],
        "std_clamp_min": std_clamp_min,
        "run_dir": run_path,
    }


def build_feature_matrix(
    params: Mapping[str, Any],
    *,
    feature_columns: Iterable[str],
    time_feature: str | None,
    time_values: np.ndarray | None,
) -> np.ndarray:
    """Construct feature matrix in the order expected by the model."""
    cols = list(feature_columns)
    if time_feature is not None:
        if time_values is None:
            raise ValueError("time_values must be provided when time_feature is defined.")
        time_values = np.asarray(time_values, dtype=np.float32)
        if time_values.ndim != 1:
            raise ValueError("time_values must be 1-D.")
        n_samples = time_values.shape[0]
    else:
        if time_values is not None:
            raise ValueError("time_values provided but model does not use a time feature.")
        n_samples = None

    matrix_columns: list[np.ndarray] = []
    for name in cols:
        if name == time_feature:
            column = time_values
        else:
            if name not in params:
                raise KeyError(f"Missing feature '{name}' in params.")
            value = _as_float_array(params[name])
            if n_samples is None:
                n_samples = value.shape[0]
            if value.shape[0] == 1 and n_samples is not None and n_samples > 1:
                column = np.full(n_samples, value.item(), dtype=np.float32)
            else:
                if n_samples is None:
                    n_samples = value.shape[0]
                if value.shape[0] != n_samples:
                    raise ValueError(
                        f"Feature '{name}' has length {value.shape[0]} but expected {n_samples}."
                    )
                column = value.astype(np.float32, copy=False)
        matrix_columns.append(column.astype(np.float32, copy=False))

    if n_samples is None:
        n_samples = 1
    X = np.stack(matrix_columns, axis=1)
    return X.astype(np.float32, copy=False)


def predict_time_range(
    run: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    start: float | None = None,
    end: float | None = None,
    step: float = 1.0,
    time_values: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Predict penetration mean ± std over a time sweep.

    Provide either explicit `time_values` or `start`/`end` (inclusive) with `step`.
    Units must match the training data (e.g., milliseconds).
    """
    feature_columns = run["feature_columns"]
    time_feature = run["time_feature"]
    model: nn.Module = run["model"]
    scalers = run["scalers"]
    cfg = run["config"]
    std_clamp_min = float(run.get("std_clamp_min", 0.0))

    if time_feature is not None:
        if time_values is None:
            if start is None or end is None:
                raise ValueError("Provide start/end for time sweep or explicit time_values.")
            if step <= 0:
                raise ValueError("step must be > 0.")
            n_steps = int(np.floor((end - start) / step)) + 1
            time_values = (start + step * np.arange(n_steps, dtype=np.float32)).astype(np.float32)
        else:
            time_values = np.asarray(time_values, dtype=np.float32)
        if time_values.ndim != 1:
            raise ValueError("time_values must be 1-D.")
    else:
        time_values = None

    X = build_feature_matrix(
        params,
        feature_columns=feature_columns,
        time_feature=time_feature,
        time_values=time_values,
    )
    X_std = _standardize(X, scalers["x"]).astype(np.float32, copy=False)

    with torch.no_grad():
        tensor_in = torch.from_numpy(X_std)
        outputs = model(tensor_in)
    mu_std, log_var_std = outputs.chunk(2, dim=-1)

    mu_std_np = mu_std.numpy()
    log_var_std_np = log_var_std.numpy()

    mu_raw = _destandardize(mu_std_np, scalers["y"])
    y_sigma = np.asarray(scalers["y"]["sigma"], dtype=np.float32)
    std_std = np.exp(0.5 * log_var_std_np)
    std_raw = std_std * y_sigma
    std_raw = np.maximum(std_raw, std_clamp_min * y_sigma)

    return {
        "time": time_values,
        "mean": mu_raw.squeeze(-1),
        "std": std_raw.squeeze(-1),
        "mu_std": mu_std_np.squeeze(-1),
        "log_var_std": log_var_std_np.squeeze(-1),
        "feature_matrix": X,
        "config": cfg,
    }
