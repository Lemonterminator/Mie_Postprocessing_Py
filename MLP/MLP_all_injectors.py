# %%
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from datetime import datetime
import json
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from standarizer import Standardizer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_penetration"
RUNS_ROOT = PROJECT_ROOT / "runs_mlp"

# %%
def load_penetration_dataframe(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].describe().T[["mean", "std", "min", "max"]]


raw_df = load_penetration_dataframe(DATA_DIR).dropna().reset_index(drop=True)
print(f"Loaded {len(raw_df)} rows from {raw_df['source_file'].nunique()} files in {DATA_DIR}.")
print(summarize_numeric(raw_df).round(3))

# %%
TARGET_COLUMN = "penetration_pixels"
IGNORED_COLUMNS = {"source_file", "is_right_censored"}
EXPECTED_FEATURE_COLUMNS = [
    "time_ms",
    "tilt_angle_radian",
    "plumes",
    "diameter_mm",
    "chamber_pressure",
    "injection_duration",
]
missing_features = [c for c in EXPECTED_FEATURE_COLUMNS if c not in raw_df.columns]
if missing_features:
    raise ValueError(f"Missing expected feature columns: {missing_features}")
FEATURE_COLUMNS = EXPECTED_FEATURE_COLUMNS
TIME_FEATURE = "time_ms"

print("Feature columns used for training:", FEATURE_COLUMNS)

CONFIG = {
    "seed": 42,
    "data_dir": str(DATA_DIR),
    "target_column": TARGET_COLUMN,
    "feature_columns": FEATURE_COLUMNS,
    "time_feature": TIME_FEATURE,
    "splits": {"val": 0.15, "test": 0.15},
    "batch_size": 512,
    "hidden_dims": [256, 128, 64],
    "dropout": 0.05,
    "activation": "relu",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 0,
    "pin_memory": torch.cuda.is_available(),
    "shuffle_train": True,
    "epochs": 50,<
    "grad_clip_norm": 1.0,
    "log_interval": 50,
    "log_var_bounds": (-10.0, 6.0),
    "std_clamp_min": 1e-3,
    "runs_root": str(RUNS_ROOT.resolve()),
}
CONFIG["input_dim"] = len(CONFIG["feature_columns"])
CONFIG["output_dim"] = 2
CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'")


class PenetrationMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    make_activation(activation),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = PenetrationMLP(
    input_dim=CONFIG["input_dim"],
    hidden_dims=CONFIG["hidden_dims"],
    output_dim=CONFIG["output_dim"],
    activation=CONFIG["activation"],
    dropout=CONFIG["dropout"],
).to(CONFIG["device"])

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"Trainable parameters: {trainable_params:,}")

# %%
class PenetrationDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        censor_flags: np.ndarray,
        censor_thresholds: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.censor_flags = torch.as_tensor(censor_flags, dtype=torch.float32)
        self.censor_thresholds = torch.as_tensor(censor_thresholds, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.targets[idx],
            self.censor_flags[idx],
            self.censor_thresholds[idx],
        )


def train_val_test_split(
    n_samples: int,
    val_ratio: float,
    test_ratio: float,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Validation and test ratios must be in (0, 1).")
    if val_ratio + test_ratio >= 1:
        raise ValueError("Sum of validation and test ratios must be < 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(np.floor(test_ratio * n_samples))
    n_val = int(np.floor(val_ratio * n_samples))
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


split_cfg = CONFIG["splits"]
train_idx, val_idx, test_idx = train_val_test_split(
    len(raw_df),
    split_cfg["val"],
    split_cfg["test"],
    seed=CONFIG["seed"],
)

feature_array = raw_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
target_array = raw_df[[TARGET_COLUMN]].to_numpy(dtype=np.float32, copy=True)
censor_array = raw_df[["is_right_censored"]].to_numpy(dtype=np.float32, copy=True)

x_scaler = Standardizer().fit(feature_array[train_idx])
y_scaler = Standardizer().fit(target_array[train_idx])

feature_scaled = x_scaler.transform(feature_array).astype(np.float32, copy=False)
target_scaled = y_scaler.transform(target_array).astype(np.float32, copy=False)

X_train = feature_scaled[train_idx]
X_val = feature_scaled[val_idx]
X_test = feature_scaled[test_idx]

y_train = target_scaled[train_idx]
y_val = target_scaled[val_idx]
y_test = target_scaled[test_idx]

censor_train = censor_array[train_idx]
censor_val = censor_array[val_idx]
censor_test = censor_array[test_idx]

threshold_train = target_scaled[train_idx]
threshold_val = target_scaled[val_idx]
threshold_test = target_scaled[test_idx]

train_dataset = PenetrationDataset(X_train, y_train, censor_train, threshold_train)
val_dataset = PenetrationDataset(X_val, y_val, censor_val, threshold_val)
test_dataset = PenetrationDataset(X_test, y_test, censor_test, threshold_test)

dataloader_kwargs = {
    "batch_size": CONFIG["batch_size"],
    "num_workers": CONFIG["num_workers"],
    "pin_memory": CONFIG["pin_memory"],
}

train_loader = DataLoader(train_dataset, shuffle=CONFIG["shuffle_train"], **dataloader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)

batch_features, batch_targets, batch_censored, batch_thresholds = next(iter(train_loader))
print(f"Train batch features shape: {batch_features.shape}")
print(f"Train batch targets shape: {batch_targets.shape}")
print(f"Train batch censor flags shape: {batch_censored.shape}")
print(f"Train batch censor thresholds shape: {batch_thresholds.shape}")

scalers_state = {"x": x_scaler.state_dict(), "y": y_scaler.state_dict()}
print("Saved scaler keys:", list(scalers_state.keys()))

# %%
LOG_2PI = math.log(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = log_var.exp().clamp_min(1e-8)
    return 0.5 * ((target - mu) ** 2 / var + log_var + LOG_2PI)


def right_censored_tail_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    threshold: torch.Tensor,
    *,
    std_min: float,
) -> torch.Tensor:
    std = torch.exp(0.5 * log_var).clamp_min(std_min)
    z = (threshold - mu) / std
    tail = 0.5 * torch.erfc(z / SQRT_2)
    return -torch.log(torch.clamp(tail, min=1e-12))


def heteroscedastic_censored_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
    censor_flags: torch.Tensor,
    *,
    std_min: float,
) -> torch.Tensor:
    censor_mask = censor_flags > 0.5
    base = gaussian_nll(mu, log_var, target)
    censored = right_censored_tail_nll(mu, log_var, threshold, std_min=std_min)
    return torch.where(censor_mask, censored, base)


def split_mu_logvar(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu, log_var = output.chunk(2, dim=-1)
    return mu, log_var


# %%
def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    y_scaler: Standardizer,
    config: dict,
) -> dict[str, float]:
    device = config["device"]
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0
    unc_loss_sum = 0.0
    cens_loss_sum = 0.0
    unc_count = 0
    cens_count = 0
    mae_sum = 0.0
    mse_sum = 0.0
    unc_metric_count = 0

    log_var_min, log_var_max = config["log_var_bounds"]
    std_min = config["std_clamp_min"]
    grad_clip = config.get("grad_clip_norm")

    for batch_idx, batch in enumerate(dataloader, start=1):
        features, targets, censor_flags, thresholds = batch
        features = features.to(device)
        targets = targets.to(device)
        censor_flags = censor_flags.to(device)
        thresholds = thresholds.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(features)
        mu, log_var_raw = split_mu_logvar(outputs)
        log_var = torch.clamp(log_var_raw, min=log_var_min, max=log_var_max)

        sample_losses = heteroscedastic_censored_loss(
            mu,
            log_var,
            targets,
            thresholds,
            censor_flags,
            std_min=std_min,
        )
        loss = sample_losses.mean()

        if is_train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = features.size(0)
        total_samples += batch_size
        total_loss += loss.detach().item() * batch_size

        with torch.no_grad():
            mu_det = mu.detach()
            log_var_det = log_var.detach()
            targets_det = targets.detach()
            thresholds_det = thresholds.detach()
            censor_mask = (censor_flags > 0.5).detach()

            base_losses = gaussian_nll(mu_det, log_var_det, targets_det).squeeze(-1)
            cens_losses = right_censored_tail_nll(
                mu_det,
                log_var_det,
                thresholds_det,
                std_min=std_min,
            ).squeeze(-1)

            mask_cens = censor_mask.squeeze(-1).to(dtype=torch.bool)
            mask_unc = ~mask_cens

            unc_loss_sum += base_losses[mask_unc].sum().item()
            cens_loss_sum += cens_losses[mask_cens].sum().item()
            unc_count += int(mask_unc.sum().item())
            cens_count += int(mask_cens.sum().item())

            mask_unc_np = mask_unc.cpu().numpy().astype(bool)
            if mask_unc_np.any():
                mu_np = mu_det.cpu().numpy()
                targets_np = targets_det.cpu().numpy()
                mu_raw = y_scaler.inverse_transform(mu_np)
                targets_raw = y_scaler.inverse_transform(targets_np)
                diff = mu_raw[mask_unc_np] - targets_raw[mask_unc_np]
                mae_sum += np.abs(diff).sum()
                mse_sum += np.square(diff).sum()
                unc_metric_count += int(mask_unc_np.sum())

    avg_loss = total_loss / max(total_samples, 1)
    avg_unc_loss = unc_loss_sum / max(unc_count, 1) if unc_count else float("nan")
    avg_cens_loss = cens_loss_sum / max(cens_count, 1) if cens_count else float("nan")
    mae = mae_sum / max(unc_metric_count, 1) if unc_metric_count else float("nan")
    rmse = math.sqrt(mse_sum / max(unc_metric_count, 1)) if unc_metric_count else float("nan")

    return {
        "loss": avg_loss,
        "loss_uncensored": avg_unc_loss,
        "loss_censored": avg_cens_loss,
        "mae_raw": mae,
        "rmse_raw": rmse,
        "samples": total_samples,
        "n_uncensored": unc_count,
        "n_censored": cens_count,
    }


# %%
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)

history: list[dict[str, float | int]] = []
best_checkpoint = {
    "epoch": 0,
    "val_loss": float("inf"),
    "model_state": deepcopy(model.state_dict()),
}

for epoch in range(1, CONFIG["epochs"] + 1):
    train_metrics = run_epoch(
        model,
        train_loader,
        optimizer=optimizer,
        y_scaler=y_scaler,
        config=CONFIG,
    )
    val_metrics = run_epoch(
        model,
        val_loader,
        optimizer=None,
        y_scaler=y_scaler,
        config=CONFIG,
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae_raw"],
            "val_rmse": val_metrics["rmse_raw"],
        }
    )

    print(
        f"Epoch {epoch:03d} | "
        f"train loss {train_metrics['loss']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} | "
        f"val MAE {val_metrics['mae_raw']:.4f} | "
        f"val RMSE {val_metrics['rmse_raw']:.4f}"
    )

    if val_metrics["loss"] < best_checkpoint["val_loss"]:
        best_checkpoint["epoch"] = epoch
        best_checkpoint["val_loss"] = val_metrics["loss"]
        best_checkpoint["model_state"] = deepcopy(model.state_dict())

model.load_state_dict(best_checkpoint["model_state"])
print(
    f"Loaded best model from epoch {best_checkpoint['epoch']} "
    f"with validation loss {best_checkpoint['val_loss']:.4f}."
)

test_metrics = run_epoch(
    model,
    test_loader,
    optimizer=None,
    y_scaler=y_scaler,
    config=CONFIG,
)

print(
    "Test metrics | "
    f"loss {test_metrics['loss']:.4f} | "
    f"MAE {test_metrics['mae_raw']:.4f} | "
    f"RMSE {test_metrics['rmse_raw']:.4f} | "
    f"uncensored samples {test_metrics['n_uncensored']} | "
    f"censored samples {test_metrics['n_censored']}"
)

history_df = pd.DataFrame(history)
print(history_df.tail())

# %%
runs_root_path = Path(CONFIG["runs_root"])
runs_root_path.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"penetration_hetero_{timestamp}"
run_dir = runs_root_path / run_name
run_dir.mkdir(parents=True, exist_ok=False)

torch.save(best_checkpoint["model_state"], run_dir / "best_model.pt")
with open(run_dir / "scalers.json", "w", encoding="utf-8") as f:
    json.dump(scalers_state, f, indent=2)

def _to_serializable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    return value

train_config_export = _to_serializable({k: v for k, v in CONFIG.items()})
test_metrics_export = _to_serializable(test_metrics)
best_history_entry = min(history, key=lambda d: d["val_loss"]) if history else None
metadata = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "run_name": run_name,
    "best_epoch": int(best_checkpoint["epoch"]),
    "best_val_loss": float(best_checkpoint["val_loss"]),
    "train_rows": int(len(train_dataset)),
    "val_rows": int(len(val_dataset)),
    "test_rows": int(len(test_dataset)),
    "history_best": _to_serializable(best_history_entry) if best_history_entry else None,
    "test_metrics": test_metrics_export,
}
with open(run_dir / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "model": {
                "input_dim": CONFIG["input_dim"],
                "output_dim": CONFIG["output_dim"],
                "hidden_dims": CONFIG["hidden_dims"],
                "activation": CONFIG["activation"],
                "dropout": CONFIG["dropout"],
            },
            "data": {
                "feature_columns": CONFIG["feature_columns"],
                "time_feature": CONFIG["time_feature"],
                "target_column": CONFIG["target_column"],
                "std_clamp_min": CONFIG["std_clamp_min"],
            },
            "training": train_config_export,
            "metadata": metadata,
        },
        f,
        indent=2,
    )
history_df.to_csv(run_dir / "history.csv", index=False)
with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
    f.write(f"Run directory: {run_dir}\n")
    f.write(f"Best epoch: {best_checkpoint['epoch']}\n")
    f.write(f"Validation loss: {best_checkpoint['val_loss']:.4f}\n")
    f.write(f"Test metrics: {test_metrics_export}\n")

print(f"Saved run artifacts to {run_dir}")
