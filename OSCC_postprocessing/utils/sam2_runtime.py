"""Runtime path helpers for SAM2 assets.

This module does not load SAM models. It only resolves checkpoint/config paths
based on the agreed environment-variable + manifest contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # .../OSCC_postprocessing/utils/sam2_runtime.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _default_sam2_home() -> Path:
    return _repo_root() / ".models" / "sam2"


def _fail_with_setup_hint(reason: str) -> RuntimeError:
    return RuntimeError(
        f"{reason}\n"
        "SAM2 assets not found. Run: python scripts/download_sam2.py --model small\n"
        "You can also set SAM2_HOME or explicit SAM2_CKPT/SAM2_CONFIG env vars."
    )


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _fail_with_setup_hint(f"Failed to read manifest: {manifest_path} ({exc})") from exc
    if not isinstance(data, dict) or "models" not in data:
        raise _fail_with_setup_hint(f"Invalid manifest format: {manifest_path}")
    return data


def resolve_sam2_paths() -> dict[str, str]:
    """Resolve SAM2 checkpoint/config using env vars first, then local manifest.

    Resolution order:
    1) `SAM2_CKPT` + `SAM2_CONFIG` if both are set.
    2) `<SAM2_HOME>/manifest.json` (SAM2_HOME defaults to repo `/.models/sam2`).
    3) Raise clear setup error with downloader command.
    """

    env_ckpt = os.getenv("SAM2_CKPT")
    env_cfg = os.getenv("SAM2_CONFIG")
    if env_ckpt or env_cfg:
        if not (env_ckpt and env_cfg):
            raise _fail_with_setup_hint(
                "Both SAM2_CKPT and SAM2_CONFIG must be set together when overriding."
            )
        ckpt_path = Path(env_ckpt).expanduser().resolve()
        cfg_path = Path(env_cfg).expanduser().resolve()
        if not ckpt_path.exists():
            raise _fail_with_setup_hint(f"SAM2_CKPT does not exist: {ckpt_path}")
        if not cfg_path.exists():
            raise _fail_with_setup_hint(f"SAM2_CONFIG does not exist: {cfg_path}")
        return {
            "home": str(ckpt_path.parent.parent),
            "model_key": "env_override",
            "model_id": "env_override",
            "checkpoint": str(ckpt_path),
            "config": str(cfg_path),
            "source": "env",
        }

    home = Path(os.getenv("SAM2_HOME", str(_default_sam2_home()))).expanduser().resolve()
    manifest_path = home / "manifest.json"
    if not manifest_path.exists():
        raise _fail_with_setup_hint(f"Manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    model_key = str(manifest.get("active_model_key", "small"))
    models = manifest.get("models", {})
    if model_key not in models:
        if "small" in models:
            model_key = "small"
        else:
            raise _fail_with_setup_hint(
                f"Manifest active model '{model_key}' is missing and no 'small' fallback exists."
            )

    model_spec = models[model_key]
    ckpt_rel = model_spec.get("checkpoint_path")
    cfg_rel = model_spec.get("config_path")
    if not ckpt_rel or not cfg_rel:
        raise _fail_with_setup_hint(
            f"Manifest model entry '{model_key}' is missing checkpoint_path/config_path."
        )

    ckpt_path = (home / ckpt_rel).resolve()
    cfg_path = (home / cfg_rel).resolve()
    if not ckpt_path.exists():
        raise _fail_with_setup_hint(f"Checkpoint path from manifest does not exist: {ckpt_path}")
    if not cfg_path.exists():
        raise _fail_with_setup_hint(f"Config path from manifest does not exist: {cfg_path}")

    return {
        "home": str(home),
        "model_key": model_key,
        "model_id": str(model_spec.get("model_id", "")),
        "checkpoint": str(ckpt_path),
        "config": str(cfg_path),
        "source": "manifest",
    }

