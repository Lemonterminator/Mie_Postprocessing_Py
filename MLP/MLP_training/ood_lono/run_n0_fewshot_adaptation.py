"""Nozzle0 few-shot adaptation curve for the residual family head.

Starting from a residual-family-head student whose trunk was trained with
Nozzle0 held out (true OOD), this driver answers the deployment question:
"once a new injector arrives, how many of its conditions must we collect before
the per-family residual delta head recovers in-family accuracy?"

Protocol
--------
- Freeze everything except ``delta_heads[family_id]`` for the new family
  (default family 0 = Nozzle0). The shared trunk, shared mean, the other
  family's delta head and the log-var head stay frozen, so this is exactly the
  cheap per-family adaptation the residual design enables.
- For each adaptation budget ``k`` (number of *conditions* from the new nozzle
  exposed to training), draw ``repeats`` disjoint-from-test condition subsets,
  fine-tune only delta_k, and evaluate uncensored RMSE on a fixed held-out test
  split of the same nozzle. ``k=0`` performs no training -> genuine zero-shot
  (delta_k stays at its zero init -> shared predictor).
- Report RMSE-vs-k mean / std / per-repeat, plus the in-family reference band.

Feature building, prediction and metric conventions are reused verbatim from
run_lono_pipeline so numbers are directly comparable to the LONO tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))
    sys.path.insert(0, str(_here.parent))

from engineered_feature_common import (
    build_dataset_registry,
    build_feature_matrix_np,
    infer_feature_family,
    load_run_artifacts,
    split_mu_logvar,
)

# Reuse the exact metric + path constants the LONO pipeline uses.
from run_lono_pipeline import (  # type: ignore
    UNCENSORED_POINTS_CSV,
    metrics_from_points,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLP_ROOT = PROJECT_ROOT / "MLP"
RUNS_ROOT = MLP_ROOT / "runs_mlp"


def _build_condition_blocks(
    df: pd.DataFrame,
    *,
    scaler_state: dict[str, Any],
    feature_columns: list[str],
    registry,
    time_feature: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-condition (features, a_scale, truth) blocks once, cached by condition_id."""
    blocks: dict[str, dict[str, np.ndarray]] = {}
    for cond_id, grp in df.groupby("condition_id", sort=False):
        row0 = grp.iloc[0]
        raw = {
            "umbrella_angle_deg": float(row0["umbrella_angle_deg"]),
            "plumes": float(row0["plumes"]),
            "diameter_mm": float(row0["diameter_mm"]),
            "injection_duration_us": float(row0["injection_duration_us"]),
            "injection_pressure_bar": float(row0["injection_pressure_bar"]),
            "control_backpressure_bar": float(row0["control_backpressure_bar"]),
            "chamber_pressure_bar": float(row0["chamber_pressure_bar"]),
            "dataset_key": str(row0["experiment_name"]),
        }
        time_ms = grp["time_ms"].to_numpy(dtype=np.float32)
        try:
            features_np, a_scale_np, _ = build_feature_matrix_np(
                raw, time_ms, scaler_state, feature_columns, registry,
                time_feature=time_feature,
            )
        except Exception as exc:  # pragma: no cover - matches pipeline's lenient skip
            print(f"[warn] feature build failed for condition {cond_id}: {exc}")
            continue
        blocks[str(cond_id)] = {
            "features": features_np,
            "a_scale": a_scale_np.reshape(-1),
            "truth": grp["penetration_mm"].to_numpy(dtype=np.float32),
        }
    return blocks


def _stack(blocks: dict[str, dict[str, np.ndarray]], cond_ids: list[str]) -> dict[str, np.ndarray]:
    feats = [blocks[c]["features"] for c in cond_ids if c in blocks]
    scal = [blocks[c]["a_scale"] for c in cond_ids if c in blocks]
    truth = [blocks[c]["truth"] for c in cond_ids if c in blocks]
    return {
        "features": np.concatenate(feats) if feats else np.empty((0, 0), np.float32),
        "a_scale": np.concatenate(scal) if scal else np.empty((0,), np.float32),
        "truth": np.concatenate(truth) if truth else np.empty((0,), np.float32),
    }


def _freeze_all_but_family_delta(model: torch.nn.Module, family_id: int) -> int:
    """Freeze every parameter except delta_heads[family_id]; return #trainable params."""
    if not hasattr(model, "delta_heads"):
        raise AttributeError("Model has no delta_heads; expected a residual family head student.")
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable = 0
    for p in model.delta_heads[int(family_id)].parameters():  # type: ignore[index]
        p.requires_grad_(True)
        n_trainable += p.numel()
    return n_trainable


def _predict_metrics(
    model: torch.nn.Module,
    train_config: dict[str, Any],
    stacked: dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int = 262144,
) -> dict[str, float]:
    family = infer_feature_family(train_config["feature_columns"])
    std_floor = float(train_config.get("std_clamp_min", 0.0))
    feats, scale, truth = stacked["features"], stacked["a_scale"], stacked["truth"]
    mu_chunks, std_chunks = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            ft = torch.as_tensor(feats[start:start + batch_size], dtype=torch.float32, device=device)
            sc = torch.as_tensor(scale[start:start + batch_size, None], dtype=torch.float32, device=device)
            mu_hat, log_var_hat = split_mu_logvar(model(ft))
            log_var_hat = torch.clamp(log_var_hat, min=-20.0, max=20.0)
            if family == "engineered_v2":
                mu = sc * mu_hat
                std = sc * torch.exp(0.5 * log_var_hat)
            else:
                mu = mu_hat
                std = torch.exp(0.5 * log_var_hat)
            std = torch.clamp(std, min=std_floor)
            mu_chunks.append(mu.cpu().numpy().reshape(-1))
            std_chunks.append(std.cpu().numpy().reshape(-1))
    pts = pd.DataFrame({
        "pen_pred_mm": np.concatenate(mu_chunks),
        "pen_true_mm": truth.astype(float),
        "pen_std_mm": np.concatenate(std_chunks),
    })
    return metrics_from_points(pts)


def _adapt_delta(
    model: torch.nn.Module,
    train_config: dict[str, Any],
    stacked: dict[str, np.ndarray],
    *,
    family_id: int,
    device: torch.device,
    epochs: int,
    lr: float,
    delta_l2: float,
    loss_mode: str = "nll",
) -> None:
    """Fine-tune only delta_heads[family_id] on the adaptation points.

    loss_mode='nll' weights the mean error by the (frozen) predicted precision —
    matching production residual training, but for an OOD nozzle the frozen
    log-var head emits a hugely inflated sigma, throttling the mean gradient.
    loss_mode='mse' adapts the mean directly in A-scaled space, immune to the
    broken frozen sigma, isolating what delta alone can achieve on the mean.
    """
    family = infer_feature_family(train_config["feature_columns"])
    feats = torch.as_tensor(stacked["features"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(stacked["a_scale"][:, None], dtype=torch.float32, device=device)
    truth = torch.as_tensor(stacked["truth"][:, None], dtype=torch.float32, device=device)
    # Targets live in A-scaled space when family == engineered_v2.
    target_scaled = truth / scale if family == "engineered_v2" else truth

    params = [p for p in model.delta_heads[int(family_id)].parameters()]  # type: ignore[index]
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        mu_hat, log_var_hat = split_mu_logvar(model(feats))
        log_var_hat = torch.clamp(log_var_hat, min=-20.0, max=20.0)
        if str(loss_mode).lower() == "mse":
            data_term = ((mu_hat - target_scaled) ** 2).mean()
        else:
            inv_var = torch.exp(-log_var_hat)
            data_term = 0.5 * (inv_var * (mu_hat - target_scaled) ** 2 + log_var_hat).mean()
        # delta shrinkage (same knob as production residual training)
        delta = model.forward_parts(feats)[1] if hasattr(model, "forward_parts") else None
        reg = float(delta_l2) * (delta.pow(2).mean()) if (delta is not None and delta_l2 > 0) else 0.0
        loss = data_term + reg
        loss.backward()
        opt.step()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-run", type=Path, required=True,
                   help="Residual-family-head student run dir (stage3 output) whose trunk was trained "
                        "with the target nozzle held out.")
    p.add_argument("--nozzle", type=str, default="BC20220627_HZ_Nozzle0",
                   help="experiment_name of the new injector to adapt to.")
    p.add_argument("--family-id", type=int, default=0,
                   help="Family index whose delta head is adapted (Nozzle0 -> 0).")
    p.add_argument("--k-list", type=str, default="0,1,2,5,10,20,50,all",
                   help="Comma-separated adaptation budgets (conditions). 'all' = every train condition.")
    p.add_argument("--repeats", type=int, default=5, help="Random condition draws per k (k>0).")
    p.add_argument("--test-frac", type=float, default=0.4, help="Fraction of nozzle conditions held out for test.")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--delta-l2", type=float, default=1e-4)
    p.add_argument("--loss-mode", choices=("nll", "mse"), default="nll",
                   help="nll matches production training (mean error weighted by frozen precision); "
                        "mse adapts the mean directly, immune to the OOD-inflated frozen sigma.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--uncensored-csv", type=Path, default=UNCENSORED_POINTS_CSV)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    rng = np.random.default_rng(int(args.seed))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.output_dir or (RUNS_ROOT / f"n0_fewshot_{ts}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Load nozzle rows + build per-condition feature blocks once.
    df = pd.read_csv(args.uncensored_csv, low_memory=False)
    df = df[df["experiment_name"].astype(str) == str(args.nozzle)].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"No uncensored points for nozzle={args.nozzle!r}")

    registry = build_dataset_registry()
    base_artifacts = load_run_artifacts(args.student_run, device=device)
    feature_columns = list(base_artifacts.train_config["feature_columns"])
    time_feature = str(base_artifacts.train_config.get("time_feature", "time_norm_0_5ms"))
    arch_mode = str(base_artifacts.train_config.get("architecture_mode", "single"))
    print(f"Student arch={arch_mode}, features={len(feature_columns)}, device={device}")

    blocks = _build_condition_blocks(
        df, scaler_state=base_artifacts.scaler_state, feature_columns=feature_columns,
        registry=registry, time_feature=time_feature,
    )
    all_conds = sorted(blocks.keys())
    rng.shuffle(all_conds)
    n_test = max(1, int(round(len(all_conds) * float(args.test_frac))))
    test_conds = all_conds[:n_test]
    train_pool = all_conds[n_test:]
    test_stack = _stack(blocks, test_conds)
    print(f"{args.nozzle}: {len(all_conds)} conditions -> {len(train_pool)} train pool / {len(test_conds)} test "
          f"({len(test_stack['truth'])} test points)")

    k_list: list[int | str] = []
    for tok in str(args.k_list).split(","):
        tok = tok.strip()
        if tok == "all":
            k_list.append("all")
        elif tok:
            k_list.append(int(tok))

    rows: list[dict[str, Any]] = []
    for k in k_list:
        k_eff = len(train_pool) if k == "all" else int(k)
        k_eff = min(k_eff, len(train_pool))
        reps = 1 if (k == "all" or k_eff == 0) else int(args.repeats)
        for rep in range(reps):
            # Fresh student each repeat.
            artifacts = load_run_artifacts(args.student_run, device=device)
            model = artifacts.model.to(device)
            n_train_params = _freeze_all_but_family_delta(model, args.family_id)
            if k_eff > 0:
                pick = list(rng.choice(train_pool, size=k_eff, replace=False))
                adapt_stack = _stack(blocks, pick)
                _adapt_delta(
                    model, artifacts.train_config, adapt_stack,
                    family_id=args.family_id, device=device,
                    epochs=args.epochs, lr=args.lr, delta_l2=args.delta_l2,
                    loss_mode=args.loss_mode,
                )
            m = _predict_metrics(model, artifacts.train_config, test_stack, device=device)
            rows.append({
                "k": k_eff, "k_label": str(k), "repeat": rep,
                "n_train_params": n_train_params,
                "rmse_mm": m["rmse_mm"], "mae_mm": m["mae_mm"], "bias_mm": m["bias_mm"],
                "coverage_1sigma": m["coverage_1sigma"], "n_test_points": int(len(test_stack["truth"])),
            })
            print(f"  k={str(k):>3s} rep={rep}: rmse={m['rmse_mm']:.3f} bias={m['bias_mm']:+.3f} "
                  f"cov1s={m['coverage_1sigma']:.3f}")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "fewshot_per_run.csv", index=False)

    # Aggregate per k.
    agg = (per_run.groupby("k", as_index=False)
           .agg(rmse_mean=("rmse_mm", "mean"), rmse_std=("rmse_mm", "std"),
                bias_mean=("bias_mm", "mean"), cov1s_mean=("coverage_1sigma", "mean"),
                n_repeats=("rmse_mm", "size"))
           .sort_values("k"))
    agg["rmse_std"] = agg["rmse_std"].fillna(0.0)
    agg.to_csv(out_dir / "fewshot_curve.csv", index=False)

    (out_dir / "config.json").write_text(json.dumps({
        "student_run": str(args.student_run), "nozzle": args.nozzle, "family_id": args.family_id,
        "k_list": str(args.k_list), "repeats": args.repeats, "test_frac": args.test_frac,
        "epochs": args.epochs, "lr": args.lr, "delta_l2": args.delta_l2,
        "loss_mode": args.loss_mode, "seed": args.seed,
        "n_conditions": len(all_conds), "n_test_conditions": len(test_conds),
        "n_train_pool": len(train_pool), "n_test_points": int(len(test_stack["truth"])),
    }, indent=2), encoding="utf-8")

    print("\n=== Few-shot adaptation curve (uncensored RMSE) ===")
    for _, r in agg.iterrows():
        print(f"  k={int(r['k']):4d}: {r['rmse_mean']:6.3f} +/- {r['rmse_std']:4.3f} mm  "
              f"(bias {r['bias_mean']:+.2f}, cov {r['cov1s_mean']:.2f}, n_rep={int(r['n_repeats'])})")
    print(f"\nWrote: {out_dir/'fewshot_curve.csv'}")


if __name__ == "__main__":
    main()
