#!/usr/bin/env python
"""Run the official Meta SAM3 video predictor against a local video resource."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_checkpoint = repo_root / ".models" / "sam3_hf" / "sam3.pt"
    default_output = repo_root / "Results" / "sam3_local_video_smoketest.json"

    parser = argparse.ArgumentParser(
        description=(
            "Validate the official facebookresearch/sam3 video predictor against "
            "a local JPEG frame folder or MP4 file."
        )
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to a local JPEG frame directory or MP4 file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint),
        help=f"Path to a local SAM3 checkpoint. Default: {default_checkpoint}",
    )
    parser.add_argument(
        "--version",
        default="sam3",
        choices=("sam3", "sam3.1"),
        help="Official predictor version to build. Default: sam3",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "fluid spray near the center horizontal axis, darker than background, "
            "uniform texture, evolving over time"
        ),
        help='Text prompt to seed the first frame.',
    )

    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index for the initial prompt. Default: 0",
    )
    parser.add_argument(
        "--max-stream-frames",
        type=int,
        default=3,
        help="How many propagated frames to observe before stopping. Default: 3",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help=f"Where to write the JSON summary. Default: {default_output}",
    )
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable FlashAttention 3. Off by default for compatibility.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile in the official builder.",
    )
    return parser.parse_args()


def fail(message: str, code: int = 1) -> int:
    print(f"Error: {message}")
    return code


def summarize_outputs(outputs: object) -> dict[str, object]:
    if isinstance(outputs, dict):
        summary: dict[str, object] = {
            "type": "dict",
            "keys": sorted(outputs.keys()),
        }
        for key in ("pred_scores", "obj_ids", "frame_idx"):
            if key in outputs:
                value = outputs[key]
                if hasattr(value, "shape"):
                    summary[f"{key}_shape"] = tuple(value.shape)
                else:
                    summary[key] = value
        return summary
    return {"type": type(outputs).__name__}


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    print(f"[info] platform={platform.platform()}")
    print(f"[info] python={sys.version.split()[0]}")
    print(f"[info] video_path={video_path}")
    print(f"[info] checkpoint_path={checkpoint_path}")

    if not video_path.exists():
        return fail(f"video path does not exist: {video_path}")
    if not checkpoint_path.exists():
        return fail(f"checkpoint does not exist: {checkpoint_path}")

    try:
        import torch
        import sam3
        from sam3.model_builder import build_sam3_predictor
    except ModuleNotFoundError as exc:
        return fail(f"missing runtime dependency: {exc.name}")
    except Exception as exc:
        return fail(f"failed to import SAM3 runtime: {exc}")

    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"[info] sam3_version={getattr(sam3, '__version__', 'unknown')}")

    predictor = build_sam3_predictor(
        version=args.version,
        checkpoint_path=str(checkpoint_path),
        use_fa3=args.use_fa3,
        compile=args.compile,
        warm_up=False,
        async_loading_frames=False,
    )

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
            "frame_index": args.frame_index,
            "text": args.prompt,
        }
    )
    add_prompt_summary = summarize_outputs(response.get("outputs"))
    print(f"[ok] add_prompt_summary={add_prompt_summary}")

    streamed_frames = 0
    first_stream_summary: dict[str, object] | None = None
    for streamed_frames, streamed_output in enumerate(
        predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ),
        start=1,
    ):
        if first_stream_summary is None:
            first_stream_summary = summarize_outputs(streamed_output)
            print(f"[ok] first_stream_summary={first_stream_summary}")
        if streamed_frames >= args.max_stream_frames:
            break

    predictor.handle_request(
        request={
            "type": "close_session",
            "session_id": session_id,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "sam3_version": getattr(sam3, "__version__", "unknown"),
        "video_path": str(video_path),
        "checkpoint_path": str(checkpoint_path),
        "version": args.version,
        "prompt": args.prompt,
        "frame_index": args.frame_index,
        "session_id": session_id,
        "add_prompt_summary": add_prompt_summary,
        "first_stream_summary": first_stream_summary,
        "streamed_frames_observed": streamed_frames,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] wrote smoke test summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
