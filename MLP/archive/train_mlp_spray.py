# Auto-generated training script for spray MLP

# Imports
import os, re, json, math, random, time, pathlib, copy, warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================
# Configuration
# =====================
CONFIG = {
    # Data
    "root_dir": r"C:/Users/Jiang/Documents/Mie_Py/Mie_Postprocessing_Py/Cine",
    "csv_glob": "**/*_features.csv",
    "saturation_px": 365.0,
    "drop_nan_penetration": True,
    "use_cuda_if_available": True,
    "train_val_test_split": [0.7, 0.15, 0.15],
    "seed": 42,
    "REGENERATE_SPLIT": True,
    "split_save_path": "splits_indices.json",

    # Features/Targets
    "feature_names": ["chamber_pressure", "injection_pressure", "injection_duration", "control_backpressure"],
    "target_names": ["cone_angle", "penetration_index", "area"],
    "standardize_features": True,
    "standardize_targets": True,

    # Dataloader
    "batch_size": 256,
    "num_workers": 0,

    # Model
    # "hidden_sizes": [128, 128, 64],
    "hidden_sizes": [32, 32, 16],
    "activation": "tanh",         # relu | gelu | prelu | tanh
    "dropout": 0.5,
    "normalization": "layer",     # none | batch | layer
    "weight_norm": False,
    "output_activation": "none",  # none | relu | softplus

    # Optimization
    "epochs": 200,
    "optimizer": "adamw",
    # "lr": 3e-3,
    "lr" : 1e-2,
    "weight_decay": 1e-4,
    "l1_lambda": 0.0,
    "gradient_clip_norm": 1.0,

    # Regularization tricks
    "input_noise_std": 0.01,
    "mixup_alpha": 0.0,           # 0 disables

    # Scheduler
    "scheduler": "plateau",       # none | plateau | cosine
    "plateau_patience": 10,
    "plateau_factor": 0.5,
    "cosine_T_max": 100,

    # Early stopping
    "early_stop_patience": 20,
    "min_delta": 1e-5,

    # AMP
    "use_amp": True,

    # Outputs
    "out_dir": "runs_mlp",
}
os.makedirs(CONFIG["out_dir"], exist_ok=True)
set_seed(CONFIG["seed"])
device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG["use_cuda_if_available"]) else "cpu")
print("Using device:", device)

# =====================
# Experimental mapping
# =====================
T_GROUP_TO_COND = {
    1:  {"chamber_pressure": 5,  "injection_pressure": 2200, "control_backpressure": 1},
    2:  {"chamber_pressure": 15, "injection_pressure": 2200, "control_backpressure": 1},
    3:  {"chamber_pressure": 25, "injection_pressure": 2200, "control_backpressure": 1},
    4:  {"chamber_pressure": 35, "injection_pressure": 2200, "control_backpressure": 1},
    5:  {"chamber_pressure": 5,  "injection_pressure": 1400, "control_backpressure": 1},
    6:  {"chamber_pressure": 15, "injection_pressure": 1400, "control_backpressure": 1},
    7:  {"chamber_pressure": 35, "injection_pressure": 1400, "control_backpressure": 1},
    8:  {"chamber_pressure": 5,  "injection_pressure": 2200, "control_backpressure": 4},
    9:  {"chamber_pressure": 15, "injection_pressure": 2200, "control_backpressure": 4},
    10: {"chamber_pressure": 35, "injection_pressure": 2200, "control_backpressure": 4},
    11: {"chamber_pressure": 5,  "injection_pressure": 1600, "control_backpressure": 1},
    12: {"chamber_pressure": 35, "injection_pressure": 1600, "control_backpressure": 1},
}

def cine_to_injection_duration_us(cine_number: int) -> float:
    # 1..5 -> 340; +20 each block up to 91..95 -> 700
    # 96..100 -> 750; then +50 per block up to 141..145 -> 1200
    cine_number = max(1, min(145, int(cine_number)))
    block = (cine_number - 1) // 5
    if block <= 18:
        return 340 + 20 * block
    else:
        return 750 + 50 * (block - 19)

# =====================
# Data ingestion
# =====================
def infer_t_group_from_path(path: str) -> Optional[int]:
    m = re.search(r"[\\/]T(\d+)[\\/]", path)
    if m:
        return int(m.group(1))
    m = re.search(r"[\\/]T(\d+)_", path)
    if m:
        return int(m.group(1))
    return None

def infer_cine_from_row_or_filename(row: pd.Series, csv_path: str) -> Optional[int]:
    if "video" in row and pd.notna(row["video"]):
        try:
            return int(row["video"])
        except Exception:
            pass
    fname = os.path.basename(csv_path)
    m = re.search(r"(\d+)", fname)
    if m:
        return int(m.group(1))
    return None

def scan_csvs(root_dir: str, pattern: str) -> List[str]:
    files = [str(p) for p in pathlib.Path(root_dir).glob(pattern) if p.is_file()]
    return files

def build_dataframe(root_dir: str, pattern: str) -> pd.DataFrame:
    csv_files = scan_csvs(root_dir, pattern)
    rows = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            warnings.warn(f"Failed to read {csv_path}: {e}")
            continue
        t_group = infer_t_group_from_path(csv_path)
        if t_group is None:
            warnings.warn(f"Could not infer T group from path: {csv_path}. Skipping.")
            continue
        needed = ["video", "segment", "frame", "cone_angle", "penetration_index", "area"]
        miss = [c for c in needed if c not in df.columns]
        if miss:
            warnings.warn(f"{csv_path} missing {miss}. Skipping file.")
            continue
        df["_csv_path"] = csv_path
        df["_t_group"] = t_group
        if "video" in df.columns and pd.api.types.is_numeric_dtype(df["video"]):
            df["_cine"] = df["video"].astype(int)
        else:
            df["_cine"] = df.apply(lambda r: infer_cine_from_row_or_filename(r, csv_path), axis=1)
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["video","segment","frame","cone_angle","penetration_index","area","_csv_path","_t_group","_cine"])
    big = pd.concat(rows, ignore_index=True)
    return big

def add_conditions_and_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if df.empty:
        return df
    cond_cols = []
    for _, row in df.iterrows():
        tg = int(row["_t_group"])
        cine = int(row["_cine"]) if not pd.isna(row["_cine"]) else None
        if tg not in T_GROUP_TO_COND or cine is None:
            cond_cols.append((np.nan, np.nan, np.nan, np.nan))
            continue
        cond = T_GROUP_TO_COND[tg]
        inj_dur = cine_to_injection_duration_us(cine)
        cond_cols.append((cond["chamber_pressure"], cond["injection_pressure"], inj_dur, cond["control_backpressure"]))
    cond_arr = np.array(cond_cols, dtype=float)
    df["chamber_pressure"] = cond_arr[:,0]
    df["injection_pressure"] = cond_arr[:,1]
    df["injection_duration"] = cond_arr[:,2]
    df["control_backpressure"] = cond_arr[:,3]

    if cfg["drop_nan_penetration"]:
        df = df[~df["penetration_index"].isna()]
    if cfg["saturation_px"] is not None:
        df = df[df["penetration_index"] < float(cfg["saturation_px"])]
    df = df.dropna(subset=cfg["feature_names"] + cfg["target_names"])
    df = df.reset_index(drop=True)
    return df

# =====================
# Split & scalers
# =====================
def save_split(path: str, train_idx, val_idx, test_idx):
    with open(path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)

def load_split(path: str):
    with open(path, "r") as f:
        obj = json.load(f)
    return obj["train"], obj["val"], obj["test"]

def make_split(n: int, ratios: List[float], seed: int) -> Tuple[List[int], List[int], List[int]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:n_train+n_val].tolist()
    test_idx = idx[n_train+n_val:].tolist()
    return train_idx, val_idx, test_idx

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
        self.mu = np.array(d["mu"])
        self.sigma = np.array(d["sigma"])

def build_tensors_and_split(df: pd.DataFrame, cfg: dict):
    if df.empty:
        raise RuntimeError("No data available after filtering.")
    X = df[cfg["feature_names"]].to_numpy(dtype=np.float32)
    Y = df[cfg["target_names"]].to_numpy(dtype=np.float32)

    split_path = os.path.join(cfg["out_dir"], cfg["split_save_path"])
    if cfg["REGENERATE_SPLIT"] or (not os.path.exists(split_path)):
        train_idx, val_idx, test_idx = make_split(len(df), cfg["train_val_test_split"], cfg["seed"])
        save_split(split_path, train_idx, val_idx, test_idx)
    else:
        train_idx, val_idx, test_idx = load_split(split_path)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    x_scaler = Standardizer()
    y_scaler = Standardizer()
    if cfg["standardize_features"]:
        x_scaler.fit(X_train)
        X_train = x_scaler.transform(X_train)
        X_val = x_scaler.transform(X_val)
        X_test = x_scaler.transform(X_test)
    if cfg["standardize_targets"]:
        y_scaler.fit(Y_train)
        Y_train = y_scaler.transform(Y_train)
        Y_val = y_scaler.transform(Y_val)
        Y_test = y_scaler.transform(Y_test)

    scalers = {"x": x_scaler.state_dict() if cfg["standardize_features"] else None,
               "y": y_scaler.state_dict() if cfg["standardize_targets"] else None}
    return (X_train, Y_train, X_val, Y_val, X_test, Y_test,
            train_idx, val_idx, test_idx, scalers)

# =====================
# Dataset & DataLoader
# =====================
class SprayDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, input_noise_std: float = 0.0, train: bool = False):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.input_noise_std = float(input_noise_std)
        self.train = bool(train)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.Y[idx].copy()
        if self.train and self.input_noise_std > 0.0:
            x = x + np.random.normal(0.0, self.input_noise_std, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

def make_dataloaders(Xtr, Ytr, Xv, Yv, Xte, Yte, cfg: dict):
    ds_tr = SprayDataset(Xtr, Ytr, input_noise_std=cfg["input_noise_std"], train=True)
    ds_v  = SprayDataset(Xv, Yv, input_noise_std=0.0, train=False)
    ds_te = SprayDataset(Xte, Yte, input_noise_std=0.0, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=False)
    dl_v  = DataLoader(ds_v,  batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], drop_last=False)
    return dl_tr, dl_v, dl_te

# =====================
# Model
# =====================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int], activation: str = "relu",
                 dropout: float = 0.0, normalization: str = "none", weight_norm: bool = False,
                 output_activation: str = "none"):
        super().__init__()
        acts = {"relu": nn.ReLU(), "gelu": nn.GELU(), "prelu": nn.PReLU(), "tanh": nn.Tanh()}
        act = acts.get(activation, nn.ReLU())
        layers = []
        prev = in_dim
        for h in hidden:
            lin = nn.Linear(prev, h)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            layers.append(lin)
            if normalization == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif normalization == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        if output_activation == "relu":
            self.out_act = nn.ReLU()
        elif output_activation == "softplus":
            self.out_act = nn.Softplus()
        else:
            self.out_act = nn.Identity()
    def forward(self, x):
        return self.out_act(self.net(x))

def mixup_regression(x, y, alpha: float):
    if alpha is None or alpha <= 0.0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y

def l1_regularization(model: nn.Module):
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    return l1

def train_model(Xtr, Ytr, Xv, Yv, Xte, Yte, cfg: dict):
    dl_tr, dl_v, dl_te = make_dataloaders(Xtr, Ytr, Xv, Yv, Xte, Yte, cfg)
    model = MLP(in_dim=Xtr.shape[1], out_dim=Ytr.shape[1], hidden=cfg["hidden_sizes"],
                activation=cfg["activation"], dropout=cfg["dropout"],
                normalization=cfg["normalization"], weight_norm=cfg["weight_norm"],
                output_activation=cfg["output_activation"]).to(device)

    if cfg["optimizer"].lower() == "adamw":
        opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(opt, mode="min", factor=cfg["plateau_factor"],
                                      patience=cfg["plateau_patience"])
    elif cfg["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=cfg["cosine_T_max"])
    else:
        scheduler = None


    scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"] and (device.type == "cuda"))
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf"); best_epoch = -1

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch}")
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            xb, yb = mixup_regression(xb, yb, cfg["mixup_alpha"])
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["use_amp"] and (device.type == "cuda")):
                preds = model(xb)
                loss = criterion(preds, yb)
                if cfg["l1_lambda"] and cfg["l1_lambda"] > 0.0:
                    loss = loss + cfg["l1_lambda"] * l1_regularization(model)
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
        val_loss /= len(dl_v.dataset)

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
            torch.save(model.state_dict(), os.path.join(cfg["out_dir"], "best_model.pt"))
        if (epoch - best_epoch) >= cfg["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch}. Best val @ {best_epoch}: {best_val:.6f}")
            break
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: train {train_loss:.6f} | val {val_loss:.6f} | lr {current_lr:.2e}")

    model.load_state_dict(torch.load(os.path.join(cfg["out_dir"], "best_model.pt"), map_location=device))
    return model, dl_tr, dl_v, dl_te, history

def evaluate_dataloader(model, dataloader, y_scaler_state=None):
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    if y_scaler_state is not None:
        mu = np.array(y_scaler_state["mu"]); sigma = np.array(y_scaler_state["sigma"])
        preds = preds * sigma + mu; targets = targets * sigma + mu
    metrics = {"MAE": float(mean_absolute_error(targets, preds)),
               "RMSE": float(np.sqrt(mean_squared_error(targets, preds))),
               "R2": float(r2_score(targets, preds))}
    return preds, targets, metrics

def plot_history(history: dict, out_dir: str):
    plt.figure(); plt.plot(history["train_loss"], label="train_loss"); plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=150); plt.show()
    plt.figure(); plt.plot(history["lr"], label="lr")
    plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_curve.png"), dpi=150); plt.show()

# =====================
# Main
# =====================
def main(cfg: dict):
    set_seed(cfg["seed"]); os.makedirs(cfg["out_dir"], exist_ok=True)
    df_raw = build_dataframe(cfg["root_dir"], cfg["csv_glob"])
    if df_raw.empty:
        raise SystemExit(f"No data found under {cfg['root_dir']} with pattern {cfg['csv_glob']}")
    df = add_conditions_and_filter(df_raw, cfg)
    (Xtr, Ytr, Xv, Yv, Xte, Yte, train_idx, val_idx, test_idx, scalers) = build_tensors_and_split(df, cfg)
    with open(os.path.join(cfg["out_dir"], "scalers.json"), "w") as f: json.dump(scalers, f)
    model, dl_tr, dl_v, dl_te, history = train_model(Xtr, Ytr, Xv, Yv, Xte, Yte, cfg)
    ysc = scalers["y"] if cfg["standardize_targets"] else None
    preds_tr, t_tr, m_tr = evaluate_dataloader(model, dl_tr, y_scaler_state=ysc)
    preds_v,  t_v,  m_v  = evaluate_dataloader(model, dl_v,  y_scaler_state=ysc)
    preds_te, t_te, m_te = evaluate_dataloader(model, dl_te, y_scaler_state=ysc)
    np.savez(os.path.join(cfg["out_dir"], "predictions.npz"),
             train_preds=preds_tr, train_targets=t_tr,
             val_preds=preds_v, val_targets=t_v,
             test_preds=preds_te, test_targets=t_te)
    metrics = {"train": m_tr, "val": m_v, "test": m_te}
    with open(os.path.join(cfg["out_dir"], "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    plot_history(history, cfg["out_dir"])
    print("Metrics:"); print(json.dumps(metrics, indent=2))

# To run:
main(CONFIG)
