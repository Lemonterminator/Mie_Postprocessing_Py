"""
Train an MLP to predict frame-wise penetration mean/std from operating
conditions + time.

Data source: outputs created by mie_postprocessing.prepare_penetration_data
  Cine/penetration_results/T*/condition_*_frame_stats.npz

Each frame across all conditions is a supervised sample with:
  Features (X):
    [chamber_pressure, injection_pressure, injection_duration_us,
     control_backpressure, time_s]
  Targets (Y):
    [mean, std]   # both already geometry-corrected in preparation step

Improvements over the initial version:
  - Per-output metrics (mean vs std) reported separately.
  - Std positivity enforced by clamping std >= 0 after inverse transform.
  - Grouped split by condition (T-group × condition index) to evaluate
    performance on unseen conditions, not just unseen frames.
  - Residual analysis by time: plots mean absolute error vs. frame time.

Normalization, training loop and tricks are aligned with the old
train_mlp_spray.py for consistency (standardization, AMP, schedulers,
early stopping, etc.).
"""

from __future__ import annotations

import os
import re
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =====================
# Experimental mapping
# =====================

from test_matrix.Nozzle2 import T_GROUP_TO_COND




# =====================
# Config
# =====================
CONFIG = {
    # Data roots
    "cine_root": r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241017_HZ_Nozzle2",
    "results_root": r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241017_HZ_Nozzle2\penetration_results",
    "repetitions_per_condition": 5,

    # Training / data handling
    "seed": 42,
    "use_cuda_if_available": True,
    "train_val_test_split": [0.7, 0.15, 0.15],
    # When True, split by condition (T-group × condition index) rather than frames.
    # This better reflects generalization to unseen operating conditions.
    "grouped_split": True,
    "batch_size": 256,
    "num_workers": 0,

    # Model
    "hidden_sizes": [64, 64, 32],
    "activation": "tanh",        # relu | gelu | prelu | tanh
    "dropout": 0.3,
    "normalization": "layer",    # none | batch | layer
    "weight_norm": False,
    # Output head activation: we keep overall output activation as 'none' and
    # enforce std positivity post inverse-transform (more robust with standardized targets).
    "output_activation": "none", # none | relu | softplus (unused for std; we clamp instead)

    # Optimization
    "epochs": 500,
    "optimizer": "adamw",
    "lr": 5e-3,
    "weight_decay": 1e-4,
    "l1_lambda": 0.0,
    "gradient_clip_norm": 1.0,

    # Regularization tricks

    "mixup_alpha": 0.1,           # 0 disables

    # Scheduler
    "scheduler": "plateau",      # none | plateau | cosine
    "plateau_patience": 10,
    "plateau_factor": 0.5,
    "cosine_T_max": 100,

    # Early stopping
    "early_stop_patience": 80,
    "min_delta": 1e-5,

    # Automatic Mixed Precision (AMP)
    "use_amp": True,

    # Output 
    "out_dir": None,

    # Target/feature standardization toggles (align to the original script behavior)
    "standardize_features": True,
    "standardize_targets": True,

    # Std enforcement: minimum value after inverse-transform (in raw units)
    "std_clamp_min": 0.0,

    # Residual plots
    "plot_residuals": True,
    "residual_bins": 100,

    # Names and physical metadata used for downstream inference
    "feature_names": [
        "chamber_pressure",
        "injection_pressure",
        "injection_duration",   # microseconds
        "control_backpressure",
        "time_s"                 # seconds
    ],
    "target_names": ["mean", "std"],
    "frame_rate_hz": 34_000.0,
    "correction_factor": float(1.0 / np.cos(np.deg2rad(20.0))),
}

    # Outputs
CONFIG["out_dir"]= Path("runs_mlp/" + str(Path(CONFIG["cine_root"]).name))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device(
    "cuda" if (torch.cuda.is_available() and CONFIG["use_cuda_if_available"]) else "cpu"
)


# =====================
# Utils / preprocessing
# =====================
def numeric_then_alpha_key(p: Path):
    m = re.search(r"\d+", p.stem)
    if m:
        return (0, int(m.group(0)))
    return (1, p.name.lower())


def infer_t_group_from_path(path: Path) -> Optional[int]:
    m = re.search(r"[\\/]T(\d+)[\\/]", str(path))
    if m:
        return int(m.group(1))
    return None


def parse_condition_idx(name: str) -> Optional[int]:
    m = re.search(r"condition_(\d+)", name)
    return int(m.group(1)) if m else None


def build_injection_duration_map(cine_root: Path, t_group: int, reps_per_cond: int) -> Dict[int, float]:
    """
    Build a mapping: condition_index (1-based) -> injection_duration_us

    Uses the same grouping logic as in penetration_files_to_data.py: sort the files
    in Cine/T*/penetration numerically and group by reps_per_cond; the first file
    of each group carries the cine number.
    """
    pen_dir = cine_root / f"T{t_group}" / "penetration"
    if not pen_dir.exists():
        return {}
    files = sorted([p for p in pen_dir.iterdir() if p.is_file()], key=numeric_then_alpha_key)
    if not files:
        return {}
    cond_map: Dict[int, float] = {}
    total_conditions = len(files) // reps_per_cond
    for cond_idx in range(1, total_conditions + 1):
        first_file = files[(cond_idx - 1) * reps_per_cond]
        m = re.search(r"(\d+)", first_file.stem)
        if not m:
            continue
        cine_number = int(m.group(1))
        inj_dur = cine_to_injection_duration_us(cine_number)
        cond_map[cond_idx] = float(inj_dur)
    return cond_map


def load_frame_stats_to_arrays(
    cine_root: Path,
    results_root: Path,
    reps_per_cond: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan results_root for T*/condition_*_frame_stats.npz and build X, Y arrays.

    X features: [chamber_pressure, injection_pressure, injection_duration_us, control_backpressure, time_s]
    Y targets: [mean, std]
    Also returns:
      - groups: 1D array of group labels for grouped split (encoded as int ids)
      - time_s_all: 1D array of time_s per sample (for residual analysis)
      - split_keys: 1D array of condition string keys (e.g., 'T1_cond01') for traceability
    """
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    groups_str_list: List[str] = []  # condition key per row
    times_list: List[np.ndarray] = []
    t_dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("T")],
                    key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 9999)
    for t_dir in t_dirs:
        t_group = int(t_dir.name[1:])
        cond = T_GROUP_TO_COND.get(t_group)
        if cond is None:
            continue
        
        # Gather condition frame files
        files = sorted(t_dir.glob("condition_*_frame_stats.npz"), key=lambda p: parse_condition_idx(p.name) or 0)
        for f in files:
            cidx = parse_condition_idx(f.name)
            if cidx is None:
                continue
            try:
                z = np.load(str(f), allow_pickle=True)
                time_s = z["time_s"].astype(float)
                mean = z["mean"].astype(float)
                std = z["std"].astype(float)
            except Exception:
                continue
            
            # load injection duration if provided
            try: 
                inj_dur = float(cond["injection_duration"])
            except Exception:
                inj_map = build_injection_duration_map(cine_root, t_group, reps_per_cond)
                inj_dur = inj_map.get(cidx, np.nan)
            # Build rows per frame, dropping NaNs in targets
            valid = ~(np.isnan(mean) | np.isnan(std) | np.isnan(time_s))
            if not np.any(valid):
                continue
            cp = float(cond["chamber_pressure"])
            try:
                ip = float(cond["injection_pressure"])
            except Exception:
                ip = 2000.0
            # Control back pressure is 4 by default
            try:
                cb = float(cond["control_backpressure"])
            except Exception:
                cb = 4.0
            inj = float(inj_dur)
            feats = np.stack([
                np.full_like(time_s, cp),
                np.full_like(time_s, ip),
                np.full_like(time_s, inj),
                np.full_like(time_s, cb),
                time_s,
            ], axis=1)
            f_valid = feats[valid]
            y_valid = np.stack([mean, std], axis=1)[valid]
            X_list.append(f_valid)
            Y_list.append(y_valid)
            # Book-keeping arrays for grouped split and residual plots
            key = f"T{t_group}_cond{cidx:02d}"
            groups_str_list.extend([key] * f_valid.shape[0])
            times_list.append(time_s[valid])
    if not X_list:
        raise RuntimeError(f"No usable frame stats found under {results_root}")
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    # Drop rows with NaNs in any feature (e.g., missing inj duration)
    mask = ~np.isnan(X).any(axis=1)
    X, Y = X[mask], Y[mask]
    # Rebuild group labels and times for the mask
    times = np.concatenate(times_list, axis=0)[mask]
    groups_str = np.array(groups_str_list, dtype=object)[mask]
    # Map string groups to integer ids for convenience
    unique_keys = {k: i for i, k in enumerate(sorted(set(groups_str.tolist())))}
    groups = np.array([unique_keys[k] for k in groups_str.tolist()], dtype=int)
    return X, Y, groups, times, groups_str


class Standardizer:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, x: np.ndarray):
        self.mu = np.nanmean(x, axis=0, keepdims=True)
        self.sigma = np.nanstd(x, axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x: np.ndarray):
        return x * self.sigma + self.mu

    def state_dict(self):
        return {"mu": self.mu.tolist(), "sigma": self.sigma.tolist()}

    def load_state_dict(self, d):
        self.mu = np.array(d["mu"]) if not isinstance(d["mu"], np.ndarray) else d["mu"]
        self.sigma = np.array(d["sigma"]) if not isinstance(d["sigma"], np.ndarray) else d["sigma"]


def make_split(n: int, ratios: List[float], seed: int) -> Tuple[List[int], List[int], List[int]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train + n_val].tolist()
    test_idx = idx[n_train + n_val:].tolist()
    return train_idx, val_idx, test_idx


def make_grouped_split(groups: np.ndarray, ratios: List[float], seed: int) -> Tuple[List[int], List[int], List[int]]:
    """Split indices by disjoint group labels (e.g., condition ids).

    - groups: int array of shape (N,), group id for each sample.
    - ratios: [train, val, test]
    - Returns: lists of sample indices for each split.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    g_tr = set(uniq[:n_train].tolist())
    g_v = set(uniq[n_train:n_train + n_val].tolist())
    g_te = set(uniq[n_train + n_val:].tolist())
    idx = np.arange(groups.size)
    tr_idx = idx[np.isin(groups, list(g_tr))].tolist()
    v_idx = idx[np.isin(groups, list(g_v))].tolist()
    te_idx = idx[np.isin(groups, list(g_te))].tolist()
    return tr_idx, v_idx, te_idx


class FrameDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, input_noise_std: float = 0.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.input_noise_std = float(input_noise_std)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.Y[idx]
        if self.input_noise_std > 0:
            x = x + torch.randn_like(x) * self.input_noise_std
        return x, y


def activation_factory(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "prelu":
        return nn.PReLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


def norm_factory(name: str, dim: int):
    name = (name or "none").lower()
    if name == "batch":
        return nn.BatchNorm1d(dim)
    if name == "layer":
        return nn.LayerNorm(dim)
    return None


def maybe_weight_norm(layer: nn.Module, enabled: bool):
    return nn.utils.weight_norm(layer) if enabled else layer


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], activation: str,
                 dropout: float = 0.0, normalization: str = "none", weight_norm: bool = False,
                 output_activation: str = "none"):
        super().__init__()
        dims = [in_dim] + hidden + [out_dim]
        act = activation_factory(activation)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            d_in, d_out = dims[i], dims[i + 1]
            lin = maybe_weight_norm(nn.Linear(d_in, d_out), weight_norm)
            layers.append(lin)
            norm = norm_factory(normalization, d_out)
            if norm is not None:
                layers.append(norm)
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.output_activation = output_activation

    def forward(self, x):
        x = self.net(x)
        if self.output_activation == "relu":
            x = torch.relu(x)
        elif self.output_activation == "softplus":
            x = torch.nn.functional.softplus(x)
        return x


def mixup_regression(x, y, alpha: float = 0.0):
    if not alpha or alpha <= 0.0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[perm]
    y = lam * y + (1 - lam) * y[perm]
    return x, y


def train_model(X: np.ndarray, Y: np.ndarray, cfg: dict):
    """Train the model and return loaders plus training history and scalers.

    Note: Splitting logic is handled by caller to support grouped splits.
    """
    # Expect caller to pass explicit split indices in cfg if grouped split is used
    if all(k in cfg for k in ("train_idx", "val_idx", "test_idx")):
        train_idx, val_idx, test_idx = cfg["train_idx"], cfg["val_idx"], cfg["test_idx"]
    else:
        n = X.shape[0]
        train_idx, val_idx, test_idx = make_split(n, cfg["train_val_test_split"], cfg["seed"])
    # Standardization
    x_scaler = Standardizer().fit(X[train_idx]) if cfg.get("standardize_features", True) else None
    y_scaler = Standardizer().fit(Y[train_idx]) if cfg.get("standardize_targets", True) else None
    Xtr = x_scaler.transform(X[train_idx]) if x_scaler else X[train_idx]
    Ytr = y_scaler.transform(Y[train_idx]) if y_scaler else Y[train_idx]
    Xv = x_scaler.transform(X[val_idx]) if x_scaler else X[val_idx]
    Yv = y_scaler.transform(Y[val_idx]) if y_scaler else Y[val_idx]
    Xte = x_scaler.transform(X[test_idx]) if x_scaler else X[test_idx]
    Yte = y_scaler.transform(Y[test_idx]) if y_scaler else Y[test_idx]

    # Dataloaders
    dl_tr = DataLoader(FrameDataset(Xtr, Ytr, input_noise_std=cfg["input_noise_std"]),
                       batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    dl_v = DataLoader(FrameDataset(Xv, Yv, input_noise_std=0.0),
                      batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    dl_te = DataLoader(FrameDataset(Xte, Yte, input_noise_std=0.0),
                       batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # Model
    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    model = MLP(in_dim, out_dim, cfg["hidden_sizes"], cfg["activation"],
                dropout=cfg["dropout"], normalization=cfg["normalization"],
                weight_norm=cfg["weight_norm"], output_activation=cfg["output_activation"]).to(device)

    # Optimizer
    if cfg["optimizer"].lower() == "adamw":
        opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Scheduler
    scheduler = None
    if cfg["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=cfg["plateau_factor"], patience=cfg["plateau_patience"])
    elif cfg["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=cfg["cosine_T_max"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"] and (device.type == "cuda"))
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf"); best_epoch = -1

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            xb, yb = mixup_regression(xb, yb, cfg["mixup_alpha"])
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["use_amp"] and (device.type == "cuda")):
                preds = model(xb)
                loss = criterion(preds, yb)
                if cfg["l1_lambda"] and cfg["l1_lambda"] > 0.0:
                    l1 = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + cfg["l1_lambda"] * l1
            scaler.scale(loss).backward()
            if cfg["gradient_clip_norm"] is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            scaler.step(opt); scaler.update()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(dl_tr.dataset)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_v:
                xb = xb.to(device); yb = yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
        val_loss /= max(1, len(dl_v.dataset))

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss); current_lr = opt.param_groups[0]["lr"]
            else:
                scheduler.step(); current_lr = opt.param_groups[0]["lr"]
        else:
            current_lr = opt.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        improved = (best_val - val_loss) > cfg["min_delta"]
        if improved:
            best_val = val_loss; best_epoch = epoch
            os.makedirs(cfg["out_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg["out_dir"], "best_model.pt"))
        if (epoch - best_epoch) >= cfg["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch}. Best val @ {best_epoch}: {best_val:.6f}")
            break
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

    # Load best
    model.load_state_dict(torch.load(os.path.join(cfg["out_dir"], "best_model.pt"), map_location=device))
    scalers = {"x": x_scaler.state_dict() if x_scaler else None,
               "y": y_scaler.state_dict() if y_scaler else None}
    return model, dl_tr, dl_v, dl_te, history, scalers


def _inverse_transform_y(arr: np.ndarray, y_scaler_state, cfg: dict) -> np.ndarray:
    """Inverse-transform Y using scaler state if present, then clamp std >= min.

    - arr: (N,2) predictions or targets in standardized space.
    - returns: (N,2) in raw units with std clamped.
    """
    out = arr.copy()
    if y_scaler_state is not None:
        mu = np.array(y_scaler_state["mu"])
        sigma = np.array(y_scaler_state["sigma"])
        out = out * sigma + mu
    # Clamp std (second column) to be >= specified minimum
    out[:, 1] = np.maximum(cfg.get("std_clamp_min", 0.0), out[:, 1])
    return out


def _metrics_pairwise(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute aggregate metrics over both outputs jointly."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _metrics_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each output separately and return a dict keyed by name."""
    names = ["mean", "std"]
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[name] = {
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": float(r2_score(yt, yp)),
        }
    return out


def evaluate(model: nn.Module, dataloader: DataLoader, y_scaler_state, cfg: dict):
    """Run inference on a dataloader and compute overall and per-output metrics in raw units."""
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
    preds_std = np.concatenate(preds_list, axis=0)
    targets_std = np.concatenate(targets_list, axis=0)
    preds = _inverse_transform_y(preds_std, y_scaler_state, cfg)
    targets = _inverse_transform_y(targets_std, y_scaler_state, cfg)
    metrics_overall = _metrics_pairwise(targets, preds)
    metrics_per_output = _metrics_per_output(targets, preds)
    return preds, targets, {"overall": metrics_overall, "per_output": metrics_per_output}


def plot_error_vs_time(out_dir: str, times: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, split_name: str, cfg: dict):
    """Plot mean absolute error vs frame time for each output and save figures.

    - times: (N,) seconds
    - y_true, y_pred: (N,2) arrays in raw units
    """
    import matplotlib.pyplot as plt
    # Compute absolute errors
    abs_err = np.abs(y_pred - y_true)  # (N,2)
    # Bin by time
    nbins = int(cfg.get("residual_bins", 100))
    tmin, tmax = float(times.min()), float(times.max())
    if tmax <= tmin:
        return
    edges = np.linspace(tmin, tmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Aggregate mean abs error per bin
    mae_mean = np.zeros(nbins)
    mae_std = np.zeros(nbins)
    for i in range(nbins):
        m = (times >= edges[i]) & (times < edges[i + 1])
        if not np.any(m):
            mae_mean[i] = np.nan
            mae_std[i] = np.nan
        else:
            mae_mean[i] = np.nanmean(abs_err[m, 0])
            mae_std[i] = np.nanmean(abs_err[m, 1])
    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(centers, mae_mean, label="MAE mean", lw=1.8)
    plt.plot(centers, mae_std, label="MAE std", lw=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Absolute Error (raw units)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"residuals_vs_time_{split_name}.png"), dpi=150)
    plt.close()


def main(cfg: dict):
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    cine_root = Path(cfg["cine_root"]) ; results_root = Path(cfg["results_root"])
    # Load features/targets along with grouping keys and per-sample time for residual plots
    X, Y, groups, times, group_keys = load_frame_stats_to_arrays(
        cine_root, results_root, cfg["repetitions_per_condition"]
    )
    print(f"Loaded dataset: X={X.shape}, Y={Y.shape}")
    # Build split indices: grouped by condition if requested
    if cfg.get("grouped_split", True):
        train_idx, val_idx, test_idx = make_grouped_split(groups, cfg["train_val_test_split"], cfg["seed"])
    else:
        n = X.shape[0]
        train_idx, val_idx, test_idx = make_split(n, cfg["train_val_test_split"], cfg["seed"])
    # Train
    cfg = cfg.copy()
    cfg.update({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx})
    model, dl_tr, dl_v, dl_te, history, scalers = train_model(X, Y, cfg)
    # Evaluate (predictions/targets in raw units with std clamped)
    preds_tr, y_tr, m_tr = evaluate(model, dl_tr, y_scaler_state=scalers["y"], cfg=cfg)
    preds_v,  y_v,  m_v  = evaluate(model, dl_v,  y_scaler_state=scalers["y"], cfg=cfg)
    preds_te, y_te, m_te = evaluate(model, dl_te, y_scaler_state=scalers["y"], cfg=cfg)
    # Gather corresponding times for each split (dataloader maintains order)
    times_tr = times[train_idx]
    times_v  = times[val_idx]
    times_te = times[test_idx]
    # Save
    with open(os.path.join(cfg["out_dir"], "scalers.json"), "w") as f:
        json.dump(scalers, f)
    np.savez(os.path.join(cfg["out_dir"], "predictions.npz"),
             train_preds=preds_tr, train_targets=y_tr,
             val_preds=preds_v, val_targets=y_v,
             test_preds=preds_te, test_targets=y_te,
             times_train=times_tr, times_val=times_v, times_test=times_te,
             group_keys=group_keys)
    metrics = {"train": m_tr, "val": m_v, "test": m_te}
    with open(os.path.join(cfg["out_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    # Persist model configuration for easy inference later
    model_cfg = {
        "in_dim": int(X.shape[1]),
        "out_dim": int(Y.shape[1]),
        "hidden_sizes": cfg["hidden_sizes"],
        "activation": cfg["activation"],
        "dropout": cfg["dropout"],
        "normalization": cfg["normalization"],
        "weight_norm": cfg["weight_norm"],
        "output_activation": cfg["output_activation"],
        "feature_names": cfg.get("feature_names"),
        "target_names": cfg.get("target_names"),
        "standardize_features": cfg.get("standardize_features", True),
        "standardize_targets": cfg.get("standardize_targets", True),
        "std_clamp_min": cfg.get("std_clamp_min", 0.0),
        "frame_rate_hz": cfg.get("frame_rate_hz", 34_000.0),
        "correction_factor": cfg.get("correction_factor", float(1.0 / np.cos(np.deg2rad(20.0)))),
    }
    with open(os.path.join(cfg["out_dir"], "model_config.json"), "w") as f:
        json.dump(model_cfg, f, indent=2)
    # Residual analysis by time
    if cfg.get("plot_residuals", True):
        plot_error_vs_time(cfg["out_dir"], times_tr, y_tr, preds_tr, split_name="train", cfg=cfg)
        plot_error_vs_time(cfg["out_dir"], times_v,  y_v,  preds_v,  split_name="val",   cfg=cfg)
        plot_error_vs_time(cfg["out_dir"], times_te, y_te, preds_te, split_name="test",  cfg=cfg)


if __name__ == "__main__":
    main(CONFIG)
