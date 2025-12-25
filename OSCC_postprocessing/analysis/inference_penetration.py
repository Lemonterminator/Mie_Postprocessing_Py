"""
Inference utilities for the penetration MLP trained in train_mlp_penetration.py.

This module loads a trained model and its scalers/configuration, builds
feature matrices from real, physical inputs, and returns frame-wise predictions
for penetration mean and std in physical units.

Key conventions
- Features order: [chamber_pressure, injection_pressure, injection_duration_us,
                   control_backpressure, time_s]
- Targets order:  [mean, std] — already geometry-corrected during training.
- Time base:      seconds; to convert frames -> seconds, use frame_rate_hz.

Typical usage
-------------
from OSCC_postprocessing.analysis.inference_penetration import (
    load_run, frames_to_time, predict_time_range
)
run = load_run(r"runs_mlp/penetration_frame")
time_s = frames_to_time(start_frame=0, end_frame=49, frame_rate_hz=run['cfg']['frame_rate_hz'])
params = dict(chamber_pressure=15, injection_pressure=2200, injection_duration=520, control_backpressure=1)
pred = predict_time_range(run, params, time_s=time_s, output_space='corrected')
mean, std = pred['mean'], pred['std']
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP mirroring the training architecture.

    The structure is defined by (in_dim, hidden sizes, out_dim) and optional
    activation/normalization/dropout. Matches train_mlp_penetration.MLP.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden, activation: str,
                 dropout: float = 0.0, normalization: str = "none",
                 weight_norm: bool = False, output_activation: str = "none"):
        super().__init__()
        acts = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "prelu": nn.PReLU(),
            "tanh": nn.Tanh(),
        }
        act = acts.get((activation or "relu").lower(), nn.ReLU())
        dims = [in_dim] + list(hidden) + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            d_in, d_out = dims[i], dims[i + 1]
            lin = nn.Linear(d_in, d_out)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            layers.append(lin)
            if normalization == "batch":
                layers.append(nn.BatchNorm1d(d_out))
            elif normalization == "layer":
                layers.append(nn.LayerNorm(d_out))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        if output_activation == "relu":
            self.out_act = nn.ReLU()
        elif output_activation == "softplus":
            self.out_act = nn.Softplus()
        else:
            self.out_act = nn.Identity()

    def forward(self, x):
        return self.out_act(self.net(x))


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_run(run_dir: os.PathLike) -> Dict[str, Any]:
    """Load model, scalers and config from a training run directory.

    Expects files:
      - best_model.pt
      - scalers.json
      - model_config.json (written by the training script)
    Returns a dict with: 'model', 'scalers', 'cfg', 'device'.
    """
    run_dir = Path(run_dir)
    model_path = run_dir / "best_model.pt"
    scalers_path = run_dir / "scalers.json"
    cfg_path = run_dir / "model_config.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Missing scalers file: {scalers_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing model config file: {cfg_path}")

    cfg = _load_json(cfg_path)
    scalers = _load_json(scalers_path)

    model = MLP(
        in_dim=int(cfg["in_dim"]),
        out_dim=int(cfg["out_dim"]),
        hidden=cfg["hidden_sizes"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        normalization=cfg["normalization"],
        weight_norm=cfg["weight_norm"],
        output_activation=cfg["output_activation"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.float().eval()
    return {"model": model, "scalers": scalers, "cfg": cfg, "device": torch.device("cpu")}


def frames_to_time(start_frame: int, end_frame: int, frame_rate_hz: float) -> np.ndarray:
    """Convert an inclusive frame index range to seconds at the given frame rate."""
    frames = np.arange(int(start_frame), int(end_frame) + 1)
    return frames.astype(float) / float(frame_rate_hz)


def _standardize(X: np.ndarray, scalers: Dict[str, Any]) -> np.ndarray:
    """Apply feature standardization using the saved scaler state, if present.

    Returns float32 to match model weights and avoid dtype mismatch.
    """
    x_state = scalers.get("x")
    Xf = X.astype(np.float32, copy=False)
    if x_state is None:
        return Xf
    mu = np.array(x_state["mu"], dtype=np.float32)
    sigma = np.array(x_state["sigma"], dtype=np.float32)
    return (Xf - mu) / sigma


def _inverse_targets(Y_std: np.ndarray, scalers: Dict[str, Any], std_clamp_min: float) -> np.ndarray:
    """Inverse-transform targets from standardized space and clamp std ≥ min."""
    y_state = scalers.get("y")
    Y = Y_std.copy()
    if y_state is not None:
        mu = np.array(y_state["mu"])
        sigma = np.array(y_state["sigma"])
        Y = Y * sigma + mu
    # Clamp std (column 1)
    Y[:, 1] = np.maximum(float(std_clamp_min), Y[:, 1])
    return Y


def build_feature_matrix(params: Dict[str, float], time_s: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Construct feature matrix (N, 5) in the expected order from params and time array.

    Required params keys: 'chamber_pressure', 'injection_pressure',
    'injection_duration', 'control_backpressure'. Units must match training
    (duration in microseconds, time in seconds).
    """
    cp = float(params["chamber_pressure"])  # e.g., bar
    ip = float(params["injection_pressure"])  # e.g., bar
    inj = float(params["injection_duration"])  # microseconds
    cb = float(params["control_backpressure"])  # e.g., bar
    # Map into the expected order by feature_names
    feats = []
    for name in feature_names:
        if name == "time_s":
            feats.append(time_s.astype(float))
        elif name == "chamber_pressure":
            feats.append(np.full_like(time_s, cp))
        elif name == "injection_pressure":
            feats.append(np.full_like(time_s, ip))
        elif name == "injection_duration":
            feats.append(np.full_like(time_s, inj))
        elif name == "control_backpressure":
            feats.append(np.full_like(time_s, cb))
        else:
            raise KeyError(f"Unknown feature name: {name}")
    X = np.stack(feats, axis=1)
    return X.astype(np.float32)


def predict_time_range(run: Dict[str, Any], params: Dict[str, float], *,
                       start_frame: Optional[int] = None, end_frame: Optional[int] = None,
                       time_s: Optional[np.ndarray] = None,
                       output_space: str = "corrected") -> Dict[str, np.ndarray]:
    """Predict penetration mean/std over a time range.

    Provide either [start_frame, end_frame] or an explicit time array `time_s`.

    - output_space: 'corrected' (default) returns geometry-corrected outputs (as
      used during training). 'projected' returns horizontal-projection values by
      dividing by the correction factor.
    Returns dict with keys: 'time_s', 'mean', 'std'.
    """
    model, scalers, cfg = run["model"], run["scalers"], run["cfg"]
    if time_s is None:
        assert start_frame is not None and end_frame is not None, "Provide either time_s or frame range"
        time_s = frames_to_time(start_frame, end_frame, cfg.get("frame_rate_hz", 34_000.0))
    X = build_feature_matrix(params, time_s, cfg["feature_names"])  # (N,5)
    X_std = _standardize(X, scalers).astype(np.float32, copy=False)
    with torch.no_grad():
        Y_std = model(torch.from_numpy(X_std)).cpu().numpy()
    Y = _inverse_targets(Y_std, scalers, cfg.get("std_clamp_min", 0.0))  # (N,2)
    # Optionally convert back to projected space
    if output_space == "projected":
        cf = float(cfg.get("correction_factor", 1.0))
        Y = Y / cf
    elif output_space != "corrected":
        raise ValueError("output_space must be 'corrected' or 'projected'")
    return {"time_s": time_s, "mean": Y[:, 0], "std": Y[:, 1]}

