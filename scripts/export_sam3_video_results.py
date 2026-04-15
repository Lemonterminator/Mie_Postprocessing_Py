#!/usr/bin/env python
"""Run the official SAM3 video predictor and save propagated outputs to disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM3_OFFICIAL_ROOT = REPO_ROOT / "third_party" / "sam3_official"
if SAM3_OFFICIAL_ROOT.exists() and str(SAM3_OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_OFFICIAL_ROOT))

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    default_checkpoint = REPO_ROOT / ".models" / "sam3_hf" / "sam3.pt"
    default_output_root = REPO_ROOT / "Results" / "sam3_video_export"

    parser = argparse.ArgumentParser(
        description=(
            "Run the official facebookresearch/sam3 video predictor and save "
            "masks, metadata, overlays, and optional MP4 output."
        )
    )
    parser.add_argument("--video-path", required=True, help="Local JPEG/PNG frame directory or video file.")
    parser.add_argument(
        "--checkpoint",
        default=str(default_checkpoint),
        help=f"Local SAM3 checkpoint path. Default: {default_checkpoint}",
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
        help="Text prompt used on the first prompted frame.",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Initial prompt frame index.")
    parser.add_argument(
        "--output-root",
        default=str(default_output_root),
        help=f"Output directory. Default: {default_output_root}",
    )
    parser.add_argument(
        "--save-format",
        default="all",
        choices=("all", "npz", "png", "overlay", "mp4"),
        help="Which artifacts to emit. Default: all",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for overlay frames when saving as .jpg. Default: 95",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS for output MP4. Default: 30")
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Overlay alpha for mask visualization. Default: 0.45",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Deprecated alias for --score-threshold. Default: 0.5",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Object probability threshold used when exporting propagated masks. Defaults to --mask-threshold.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on propagated frames to save.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N streamed frames. Use 1 for every frame. Default: 10",
    )
    parser.add_argument(
        "--points-json",
        default=None,
        help="Optional JSON list of normalized point prompts [[x, y], ...] for tracker refinement.",
    )
    parser.add_argument(
        "--point-labels-json",
        default=None,
        help="Optional JSON list of point labels [1, 0, ...] matching --points-json.",
    )
    parser.add_argument(
        "--boxes-json",
        default=None,
        help="Optional JSON list of normalized xywh box prompts [[x, y, w, h], ...] for the initial prompt.",
    )
    parser.add_argument(
        "--box-labels-json",
        default=None,
        help="Optional JSON list of box labels [1, 0, ...] matching --boxes-json.",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=None,
        help="Object id to refine with --points-json. Defaults to the first object from the initial prompt.",
    )
    parser.add_argument("--use-fa3", action="store_true", help="Enable FlashAttention 3.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"Error: {message}")
    return 1


def resolve_frame_paths(video_path: Path) -> list[Path]:
    return sorted([p for p in video_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_frame_rgb(frame_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_video_frames_rgb(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def load_export_frames(video_path: Path) -> tuple[list[str], list[np.ndarray] | list[Path]]:
    if video_path.is_dir():
        frame_paths = resolve_frame_paths(video_path)
        if not frame_paths:
            raise RuntimeError(f"no image frames found in: {video_path}")
        return [p.stem for p in frame_paths], frame_paths

    if video_path.suffix.lower() in VIDEO_EXTS:
        frames = load_video_frames_rgb(video_path)
        return [f"{idx:05d}" for idx in range(len(frames))], frames

    raise RuntimeError(f"unsupported video resource: {video_path}")


def get_frame_rgb(frame_refs: list[np.ndarray] | list[Path], frame_index: int) -> np.ndarray:
    if frame_index < 0 or frame_index >= len(frame_refs):
        raise IndexError(f"frame index {frame_index} is outside available frames: {len(frame_refs)}")
    frame_ref = frame_refs[frame_index]
    if isinstance(frame_ref, Path):
        return load_frame_rgb(frame_ref)
    return frame_ref


def ensure_dirs(output_root: Path, save_npz: bool, save_png: bool, save_overlay: bool) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "root": output_root,
        "npz": output_root / "mask_npz",
        "png": output_root / "mask_png",
        "overlay": output_root / "overlay_frames",
    }
    if save_npz:
        dirs["npz"].mkdir(parents=True, exist_ok=True)
    if save_png:
        dirs["png"].mkdir(parents=True, exist_ok=True)
    if save_overlay:
        dirs["overlay"].mkdir(parents=True, exist_ok=True)
    return dirs


def colorize_masks(mask_stack: np.ndarray, alpha: float) -> np.ndarray:
    if mask_stack.size == 0:
        raise ValueError("mask_stack must not be empty")
    colors = np.array(
        [
            [255, 64, 64],
            [64, 255, 128],
            [64, 160, 255],
            [255, 192, 64],
            [192, 96, 255],
            [64, 255, 255],
        ],
        dtype=np.float32,
    )
    height, width = mask_stack.shape[1:]
    overlay = np.zeros((height, width, 3), dtype=np.float32)
    for idx, mask in enumerate(mask_stack):
        overlay[mask] = colors[idx % len(colors)]
    return (overlay, np.clip(alpha, 0.0, 1.0))


def blend_overlay(frame_rgb: np.ndarray, mask_stack: np.ndarray, alpha: float) -> np.ndarray:
    if mask_stack.shape[0] == 0:
        return frame_rgb
    overlay_rgb, overlay_alpha = colorize_masks(mask_stack, alpha)
    frame_f = frame_rgb.astype(np.float32)
    mask_any = mask_stack.any(axis=0)
    blended = frame_f.copy()
    blended[mask_any] = (
        (1.0 - overlay_alpha) * frame_f[mask_any] + overlay_alpha * overlay_rgb[mask_any]
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def write_mask_png(mask_stack: np.ndarray, png_path: Path) -> None:
    if mask_stack.shape[0] == 0:
        canvas = np.zeros(mask_stack.shape[1:], dtype=np.uint8)
    else:
        canvas = np.any(mask_stack, axis=0).astype(np.uint8) * 255
    ok = cv2.imwrite(str(png_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write mask PNG: {png_path}")


def write_overlay_image(image_rgb: np.ndarray, overlay_path: Path, jpeg_quality: int) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if overlay_path.suffix.lower() in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(overlay_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        ok = cv2.imwrite(str(overlay_path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write overlay image: {overlay_path}")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_optional_json(value, name: str):
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON for {name}: {exc}") from exc


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not video_path.exists():
        return fail(f"video path does not exist: {video_path}")
    if not checkpoint_path.exists():
        return fail(f"checkpoint does not exist: {checkpoint_path}")

    try:
        import torch
        from sam3.model_builder import build_sam3_predictor
    except ModuleNotFoundError as exc:
        if exc.name == "triton" and sys.platform.startswith("win"):
            return fail(
                "missing runtime dependency: triton. The official SAM3 video predictor "
                "is expected to run under WSL/Linux in this repository; use "
                "scripts/run_in_sam3_wsl_env.sh or the notebook's WSL command path."
            )
        return fail(f"missing runtime dependency: {exc.name}")
    except Exception as exc:
        return fail(f"failed to import SAM3 runtime: {exc}")

    try:
        frame_names, frame_refs = load_export_frames(video_path)
    except Exception as exc:
        return fail(str(exc))

    try:
        points = parse_optional_json(args.points_json, "--points-json")
        point_labels = parse_optional_json(args.point_labels_json, "--point-labels-json")
        boxes = parse_optional_json(args.boxes_json, "--boxes-json")
        box_labels = parse_optional_json(args.box_labels_json, "--box-labels-json")
    except ValueError as exc:
        return fail(str(exc))

    if (points is None) != (point_labels is None):
        return fail("--points-json and --point-labels-json must be provided together")
    if (boxes is None) != (box_labels is None):
        return fail("--boxes-json and --box-labels-json must be provided together")
    prompt_boxes = boxes
    prompt_box_labels = box_labels
    score_threshold = float(args.mask_threshold if args.score_threshold is None else args.score_threshold)

    save_npz = args.save_format in {"all", "npz"}
    save_png = args.save_format in {"all", "png"}
    save_overlay = args.save_format in {"all", "overlay", "mp4"}
    save_mp4 = args.save_format in {"all", "mp4"}
    dirs = ensure_dirs(output_root, save_npz=save_npz, save_png=save_png, save_overlay=save_overlay)

    predictor = build_sam3_predictor(
        version=args.version,
        checkpoint_path=str(checkpoint_path),
        use_fa3=args.use_fa3,
        compile=args.compile,
        warm_up=False,
        async_loading_frames=False,
    )

    response = predictor.handle_request({"type": "start_session", "resource_path": str(video_path)})
    session_id = response["session_id"]
    print(f"[ok] session_id={session_id}", flush=True)

    add_prompt_request = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": args.frame_index,
        "text": args.prompt,
    }
    if boxes is not None:
        add_prompt_request["bounding_boxes"] = boxes
        add_prompt_request["bounding_box_labels"] = box_labels

    response = predictor.handle_request(add_prompt_request)
    print(f"[ok] add_prompt_keys={sorted(response['outputs'].keys())}", flush=True)
    add_prompt_outputs = response["outputs"]

    refine_obj_id = args.obj_id
    if points is not None:
        print("[info] warming up SAM3 video cache before point refinement", flush=True)
        warmup_frames = 0
        for warmup_frames, _ in enumerate(
            predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": args.frame_index,
                }
            ),
            start=1,
        ):
            if args.max_frames is not None and warmup_frames >= args.max_frames:
                break
        print(f"[ok] warmup_frames={warmup_frames}", flush=True)

        if refine_obj_id is None:
            out_obj_ids = np.asarray(add_prompt_outputs.get("out_obj_ids", []))
            if out_obj_ids.size == 0:
                refine_obj_id = 1
                print(
                    "[warn] initial prompt returned no objects; using obj_id=1 for point refinement",
                    flush=True,
                )
            else:
                refine_obj_id = int(out_obj_ids.reshape(-1)[0])
        refine_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": args.frame_index,
                "points": points,
                "point_labels": point_labels,
                "obj_id": refine_obj_id,
            }
        )
        print(
            f"[ok] point_refine_obj_id={refine_obj_id} "
            f"point_count={len(points)} refine_keys={sorted(refine_response['outputs'].keys())}",
            flush=True,
        )

    records: list[dict[str, object]] = []
    overlay_writer = None
    overlay_mp4_path = output_root / "overlay.mp4"

    try:
        for frame_counter, streamed_output in enumerate(
            predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "start_frame_index": args.frame_index,
                }
            ),
            start=1,
        ):
            frame_index = int(streamed_output["frame_index"])
            outputs = streamed_output["outputs"]
            obj_ids = np.asarray(outputs["out_obj_ids"])
            probs = np.asarray(outputs["out_probs"], dtype=np.float32)
            frame_boxes = np.asarray(outputs["out_boxes_xywh"], dtype=np.float32)
            masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
            frame_stats_raw = outputs.get("frame_stats")
            frame_stats = to_jsonable(frame_stats_raw if frame_stats_raw is not None else {})
            frame_name = frame_names[frame_index]
            raw_obj_ids = obj_ids.copy()
            raw_probs = probs.copy()
            raw_boxes = frame_boxes.copy()
            raw_masks = masks.copy()
            raw_num_objects = int(masks.shape[0])
            raw_max_prob = float(np.max(raw_probs)) if raw_probs.size else None

            if probs.ndim >= 1 and masks.shape[0] == probs.shape[0]:
                keep = probs >= score_threshold
                obj_ids = obj_ids[keep]
                probs = probs[keep]
                frame_boxes = frame_boxes[keep]
                masks = masks[keep]

            record = {
                "frame_index": frame_index,
                "frame_name": frame_name,
                "raw_num_objects": raw_num_objects,
                "raw_obj_ids": raw_obj_ids.tolist(),
                "raw_probabilities": raw_probs.tolist(),
                "raw_boxes_xywh": raw_boxes.tolist(),
                "raw_max_probability": raw_max_prob,
                "num_objects": int(masks.shape[0]),
                "obj_ids": obj_ids.tolist(),
                "probabilities": probs.tolist(),
                "boxes_xywh": frame_boxes.tolist(),
                "frame_stats": frame_stats,
            }
            records.append(record)

            if save_npz:
                np.savez_compressed(
                    dirs["npz"] / f"{frame_name}.npz",
                    frame_index=np.int32(frame_index),
                    obj_ids=obj_ids,
                    probabilities=probs,
                    boxes_xywh=frame_boxes,
                    raw_obj_ids=raw_obj_ids,
                    raw_probabilities=raw_probs,
                    raw_boxes_xywh=raw_boxes,
                    raw_masks=raw_masks.astype(np.uint8),
                    masks=masks.astype(np.uint8),
                )

            if save_png:
                write_mask_png(masks, dirs["png"] / f"{frame_name}.png")

            if save_overlay:
                frame_rgb = get_frame_rgb(frame_refs, frame_index)
                overlay_rgb = blend_overlay(frame_rgb, masks, alpha=args.overlay_alpha)
                overlay_path = dirs["overlay"] / f"{frame_name}.jpg"
                write_overlay_image(overlay_rgb, overlay_path, jpeg_quality=args.jpeg_quality)

                if save_mp4:
                    if overlay_writer is None:
                        height, width = overlay_rgb.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        overlay_writer = cv2.VideoWriter(str(overlay_mp4_path), fourcc, float(args.fps), (width, height))
                        if not overlay_writer.isOpened():
                            raise RuntimeError(f"Failed to open MP4 writer: {overlay_mp4_path}")
                    overlay_writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

            if frame_counter == 1:
                print(f"[ok] first_frame_record={record}", flush=True)
            progress_every = max(1, int(args.progress_every))
            target_frames = len(frame_names)
            if args.max_frames is not None:
                target_frames = min(target_frames, int(args.max_frames))
            if frame_counter == 1 or frame_counter % progress_every == 0 or frame_counter >= target_frames:
                print(
                    f"[progress] saved {frame_counter}/{target_frames} "
                    f"frame_index={frame_index} objects={int(masks.shape[0])} "
                    f"raw_objects={raw_num_objects} raw_max_prob={raw_max_prob}",
                    flush=True,
                )
            if args.max_frames is not None and frame_counter >= args.max_frames:
                break
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        if overlay_writer is not None:
            overlay_writer.release()

    summary = {
        "video_path": str(video_path),
        "checkpoint_path": str(checkpoint_path),
        "version": args.version,
        "prompt": args.prompt,
        "frame_index": args.frame_index,
        "points": points,
        "point_labels": point_labels,
        "boxes_xywh": prompt_boxes,
        "box_labels": prompt_box_labels,
        "refine_obj_id": refine_obj_id,
        "session_id": session_id,
        "save_format": args.save_format,
        "mask_threshold": args.mask_threshold,
        "score_threshold": score_threshold,
        "frames_saved": len(records),
        "output_root": str(output_root),
        "artifacts": {
            "mask_npz_dir": str(dirs["npz"]) if save_npz else None,
            "mask_png_dir": str(dirs["png"]) if save_png else None,
            "overlay_dir": str(dirs["overlay"]) if save_overlay else None,
            "overlay_mp4": str(overlay_mp4_path) if save_mp4 else None,
        },
        "records": records,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"[ok] wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
