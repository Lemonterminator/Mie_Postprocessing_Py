"""Create an exactly equivalent two-family bridge from a pooled Stage-3 run.

The bridge retains the pooled P0 trunk, duplicates its mean projection into
two direct family heads, and preserves the shared log-variance projection.
Its predictions are therefore identical to P0 for either binary family id
before any family-specific refinement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from engineered_feature_common import (  # noqa: E402
    build_model,
    load_run_artifacts,
    single_state_dict_to_family_head,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path, help="Completed pooled P0 refinement run.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New, empty directory for the family-head bridge artifact.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run = args.source_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source = load_run_artifacts(source_run, device=None if args.device == "auto" else args.device)
    source_config = dict(source.train_config)
    source_architecture = str(source_config.get("architecture_mode", "single")).lower()
    source_features = list(source_config.get("feature_columns") or [])
    if source_architecture != "single" or int(source_config.get("output_dim", 0)) != 2:
        raise ValueError("Source must be a two-output pooled single-MLP refinement run.")
    if "family_id" in source_features:
        raise ValueError("Source already contains family_id; refusing to append a duplicate routing channel.")

    target_config = dict(source_config)
    target_config.update({
        "stage": "stage3_family_head_bridge",
        "architecture_mode": "family_head",
        "feature_columns": source_features + ["family_id"],
        "input_dim": len(source_features) + 1,
        "output_dim": 2,
        "n_families": 2,
        "family_head_dims": [],
        "fallback_family_id": 1,
        "teacher_run_dir": str(source.run_dir),
        "conversion_metadata": {
            "source_run_dir": str(source.run_dir),
            "source_checkpoint": str(source.model_path),
            "source_checkpoint_sha256": sha256(source.model_path),
            "mapping_policy": "copy pooled trunk; duplicate mu row into both direct family heads; copy logvar row",
            "function_match": "exact for family_id=0 and family_id=1 in eval mode",
        },
    })
    target_model = build_model(target_config).to(next(source.model.parameters()).device)
    target_model.load_state_dict(
        single_state_dict_to_family_head(source.model.state_dict(), target_model.state_dict()),
        strict=True,
    )
    source.model.eval()
    target_model.eval()

    device = next(source.model.parameters()).device
    features = torch.randn(257, len(source_features), device=device)
    with torch.no_grad():
        source_output = source.model(features)
        max_abs_diff = max(
            torch.max(torch.abs(
                source_output - target_model(torch.cat([
                    features,
                    torch.full((len(features), 1), family_id, device=device),
                ], dim=-1)),
            )).item()
            for family_id in (0.0, 1.0)
        )
    # The algebraic mapping is exact; GPU kernels can differ by a few ULPs
    # because the pooled final Linear is split into three routed Linear calls.
    if max_abs_diff > 1e-5:
        raise AssertionError(f"Bridge is not function-preserving: max_abs_diff={max_abs_diff:.3e}")

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best_model_refinement.pt"
    torch.save(target_model.state_dict(), checkpoint)
    (output_dir / "train_config_used.json").write_text(json.dumps(target_config, indent=2), encoding="utf-8")
    (output_dir / "scaler_state.json").write_text(json.dumps(source.scaler_state, indent=2), encoding="utf-8")
    (output_dir / "teacher_run_dir.txt").write_text(str(source.run_dir), encoding="utf-8")
    manifest = {
        "source_run_dir": str(source.run_dir),
        "source_checkpoint": str(source.model_path),
        "source_checkpoint_sha256": sha256(source.model_path),
        "bridge_checkpoint_sha256": sha256(checkpoint),
        "max_abs_diff": float(max_abs_diff),
        "family_head_dims": [],
        "feature_columns": target_config["feature_columns"],
    }
    (output_dir / "bridge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
