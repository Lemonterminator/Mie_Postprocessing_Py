#!/usr/bin/env python
"""Download SAM2 checkpoints/configs into a local, git-ignored folder.

Example:
    python scripts/download_sam2.py --model small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "tiny": {
        "model_id": "sam2.1_hiera_tiny",
        "checkpoint_urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        ],
        "checkpoint_filename": "sam2.1_hiera_tiny.pt",
        "checkpoint_sha256": None,
        "config_urls": [
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
            # Fallback for older SAM2 repo layouts.
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_t.yaml",
        ],
        "config_filename": "sam2.1_hiera_t.yaml",
        "config_sha256": None,
    },
    "small": {
        "model_id": "sam2.1_hiera_small",
        "checkpoint_urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        ],
        "checkpoint_filename": "sam2.1_hiera_small.pt",
        "checkpoint_sha256": None,
        "config_urls": [
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_s.yaml",
            # Fallback for older SAM2 repo layouts.
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_s.yaml",
        ],
        "config_filename": "sam2.1_hiera_s.yaml",
        "config_sha256": None,
    },
    "base_plus": {
        "model_id": "sam2.1_hiera_base_plus",
        "checkpoint_urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        ],
        "checkpoint_filename": "sam2.1_hiera_base_plus.pt",
        "checkpoint_sha256": None,
        "config_urls": [
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml",
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_b%2B.yaml",
            # Fallback for older SAM2 repo layouts.
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_b+.yaml",
        ],
        "config_filename": "sam2.1_hiera_b+.yaml",
        "config_sha256": None,
    },
    "large": {
        "model_id": "sam2.1_hiera_large",
        "checkpoint_urls": [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        ],
        "checkpoint_filename": "sam2.1_hiera_large.pt",
        "checkpoint_sha256": None,
        "config_urls": [
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            # Fallback for older SAM2 repo layouts.
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_l.yaml",
        ],
        "config_filename": "sam2.1_hiera_l.yaml",
        "config_sha256": None,
    },
}

DEFAULT_MODEL = "small"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_config_readable(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    # Minimal sanity checks for YAML-like model configs.
    return len(text) >= 64 and ":" in text and ("model" in text or "encoder" in text)


def is_checkpoint_readable(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(256)
    except Exception:
        return False
    if len(head) < 16:
        return False
    head_lower = head.lower()
    # Reject common HTTP error/html bodies.
    if b"<html" in head_lower or head_lower.startswith(b"<!doctype"):
        return False
    return True


def is_file_valid(path: Path, expected_sha256: str | None, kind: str) -> bool:
    if not path.exists() or not path.is_file():
        return False

    if expected_sha256:
        try:
            return file_sha256(path) == expected_sha256.lower()
        except Exception:
            return False

    size_ok = path.stat().st_size >= (64 if kind == "config" else 1024 * 1024)
    if not size_ok:
        return False

    if kind == "config":
        return is_config_readable(path)
    if kind == "checkpoint":
        return is_checkpoint_readable(path)
    return False


def download_to_path(url: str, dst: Path, timeout_s: int = 120) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url, timeout=timeout_s) as resp, NamedTemporaryFile(
            "wb", delete=False, dir=str(dst.parent)
        ) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                block = resp.read(1024 * 1024)
                if not block:
                    break
                tmp.write(block)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected download error for {url}: {exc}") from exc

    os.replace(tmp_path, dst)


def download_with_fallback(urls: list[str], dst: Path) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            download_to_path(url, dst)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError(
        "All download sources failed for "
        f"{dst.name}:\n- " + "\n- ".join(errors)
    )


def write_manifest(dest: Path, selected_model_key: str) -> Path:
    models_payload: dict[str, Any] = {}
    for key, spec in MODEL_SPECS.items():
        models_payload[key] = {
            "model_id": spec["model_id"],
            "checkpoint_path": str(Path("checkpoints") / spec["checkpoint_filename"]),
            "config_path": str(Path("configs") / spec["config_filename"]),
            "checkpoint_urls": spec["checkpoint_urls"],
            "config_urls": spec["config_urls"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "config_sha256": spec["config_sha256"],
        }

    manifest = {
        "schema_version": 1,
        "updated_at_utc": utc_now_iso(),
        "active_model_key": selected_model_key,
        "active_model_id": MODEL_SPECS[selected_model_key]["model_id"],
        "models": models_payload,
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def ensure_model_files(dest: Path, model_key: str, force: bool) -> None:
    spec = MODEL_SPECS[model_key]
    ckpt_path = dest / "checkpoints" / spec["checkpoint_filename"]
    cfg_path = dest / "configs" / spec["config_filename"]

    targets = [
        ("checkpoint", ckpt_path, spec["checkpoint_urls"], spec["checkpoint_sha256"]),
        ("config", cfg_path, spec["config_urls"], spec["config_sha256"]),
    ]

    for kind, path, urls, sha256 in targets:
        valid = is_file_valid(path, sha256, kind)
        if valid and not force:
            print(f"[skip] {kind}: {path}")
            continue
        if path.exists():
            print(f"[redo] {kind}: {path}")
        else:
            print(f"[get]  {kind}: {path}")
        download_with_fallback(list(urls), path)
        if not is_file_valid(path, sha256, kind):
            raise RuntimeError(f"Downloaded {kind} is invalid: {path}")
        print(f"[ok]   {kind}: {path}")


def list_models() -> None:
    print("Available SAM2 models:")
    for key in ("tiny", "small", "base_plus", "large"):
        spec = MODEL_SPECS[key]
        print(
            f"- {key}: {spec['model_id']} | "
            f"ckpt={spec['checkpoint_filename']} | cfg={spec['config_filename']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SAM2 checkpoints/configs into a local model folder."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS.keys()),
        default=DEFAULT_MODEL,
        help="Model size preset to download.",
    )
    parser.add_argument(
        "--dest",
        default=str(Path(".models") / "sam2"),
        help="Destination folder. Default: .models/sam2",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even when local files look valid.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available model presets and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        list_models()
        return 0

    dest = Path(args.dest).expanduser().resolve()
    (dest / "checkpoints").mkdir(parents=True, exist_ok=True)
    (dest / "configs").mkdir(parents=True, exist_ok=True)

    try:
        ensure_model_files(dest=dest, model_key=args.model, force=args.force)
        manifest_path = write_manifest(dest, selected_model_key=args.model)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        print(
            "Tip: check internet access and URL reachability, then retry. "
            "Use --force to re-download corrupted files."
        )
        return 1

    print(f"[ok]   manifest: {manifest_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
