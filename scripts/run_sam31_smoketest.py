#!/usr/bin/env python
"""Best-effort SAM 3.1 smoke test using the official Meta `sam3` package."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path


def _fail(message: str, code: int = 1) -> int:
    print(f"Error: {message}")
    return code


def _find_default_video_dir(sam3_module_path: Path) -> Path | None:
    sam3_root = sam3_module_path.parent.parent
    candidates = [
        sam3_root / "assets" / "videos" / "0001",
        sam3_root / "assets" / "videos" / "bedroom.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "Results" / "sam31_smoketest.json"

    print(f"[info] platform={platform.platform()}")
    print(f"[info] python={sys.version.split()[0]}")

    try:
        import torch
        import sam3
        from sam3.model_builder import build_sam3_predictor
    except ModuleNotFoundError as exc:
        return _fail(
            "missing runtime dependency while importing sam3. "
            f"Missing module: {exc.name}"
        )
    except Exception as exc:
        return _fail(f"failed to import sam3 runtime: {exc}")

    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[info] sam3_version={getattr(sam3, '__version__', 'unknown')}")
    print(f"[info] sam3_module={Path(sam3.__file__).resolve()}")

    if not torch.cuda.is_available():
        print("[warn] CUDA is not available. Official SAM 3.1 expects a CUDA GPU.")

    video_path = _find_default_video_dir(Path(sam3.__file__).resolve())
    if video_path is None:
        return _fail("could not locate an example video asset for the smoke test.")

    print(f"[info] smoke_test_video={video_path}")

    token_present = any(
        os.environ.get(env_name)
        for env_name in (
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        )
    )
    if not token_present:
        print("[warn] no HF token detected in environment; checkpoint download may fail.")

    try:
        predictor = build_sam3_predictor(
            version="sam3.1",
            use_fa3=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
    except Exception as exc:
        return _fail(f"failed to build SAM 3.1 predictor: {exc}")

    response = predictor.handle_request(
        request={
            "type": "start_session",
            "resource_path": str(video_path),
        }
    )
    session_id = response["session_id"]
    print(f"[ok] session_id={session_id}")

    response = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": "person",
        }
    )
    outputs = response.get("outputs")
    output_summary = {
        "outputs_type": type(outputs).__name__,
        "output_keys": sorted(outputs.keys()) if isinstance(outputs, dict) else None,
    }
    print(f"[ok] add_prompt_summary={output_summary}")

    streamed_frames = 0
    for streamed_frames, _ in enumerate(
        predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ),
        start=1,
    ):
        if streamed_frames >= 3:
            break

    predictor.handle_request(
        request={
            "type": "close_session",
            "session_id": session_id,
        }
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "sam3_version": getattr(sam3, "__version__", "unknown"),
        "video_path": str(video_path),
        "session_id": session_id,
        "add_prompt_summary": output_summary,
        "streamed_frames_observed": streamed_frames,
    }
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] wrote smoke test summary: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
