#!/usr/bin/env python
"""Download the gated Hugging Face SAM3 checkpoint into a local folder.

Example:
    python scripts/download_sam3.py --token <hf_token>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "facebook/sam3"
DEFAULT_DEST = Path(".models") / "sam3_hf"
TOKEN_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def resolve_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token.strip()
    for env_name in TOKEN_ENV_NAMES:
        value = os.environ.get(env_name)
        if value:
            return value.strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Hugging Face SAM3 model into a local folder."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face repo id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"Local output folder. Default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face access token with permission to read the gated SAM3 repo. "
            "If omitted, the script checks HF_TOKEN / HUGGINGFACE_HUB_TOKEN / "
            "HUGGING_FACE_HUB_TOKEN."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = resolve_token(args.token)
    if not token:
        print("Error: no Hugging Face token was provided.")
        print("Set one of these env vars or pass --token:")
        for env_name in TOKEN_ENV_NAMES:
            print(f"  - {env_name}")
        print(
            "The SAM3 repository is gated, so anonymous downloads do not work. "
            "Request access to https://huggingface.co/facebook/sam3 first."
        )
        return 1

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[info] model_id={args.model_id}")
    print(f"[info] dest={dest}")

    try:
        snapshot_download(
            repo_id=args.model_id,
            repo_type="model",
            token=token,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"],
        )
    except Exception as exc:
        print(f"Error: failed to download SAM3 assets: {exc}")
        print(
            "Make sure the token can access the gated repo and that network access "
            "to huggingface.co is available."
        )
        return 1

    print("[ok] SAM3 assets are available locally.")
    print(f"[ok] Load path: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
